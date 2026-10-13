// projfields.cpp — 用「逐字段累加」的方式测各段真实成本
//
// 上一版用 Clock::now() 对每段计时，失败：Clock::now() 自身 ~40ns，
// 把真实的几 ns 全部淹没（每段都显示 ~45ns，明显是噪声）。
//
// 本版改用**消融法（ablation）**：用一个 volatile 累加器把每一段的
// 输入「吃掉」，然后整体计时。通过 -DABLATE_xxx 编出多个变体，
// 比较「完整」与「去掉某段」的整轮耗时差，得到该段的真实成本。
//
// 被消融的段用 volatile sink 强制保留（否则会被优化掉）。

#include "camera.h"
#include "ply_loader.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace gsplat;
using Clock = std::chrono::high_resolution_clock;
static inline double ms_since(const Clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static volatile float g_sink = 0.f;

int main(int argc, char** argv) {
    const char* ply = (argc > 1) ? argv[1] : "data/pilot.ply";
    std::vector<Splat> S;
    std::vector<ShCoeffs> sh;
    if (!load_ply_splats(ply, S, sh, 1, false)) return 1;

    Camera cam = Camera::look_at({0, 0, 60}, {0, 0, 0}, {0, 1, 0}, 512, 288, 45.f);
    const int N_ITER = 10;
    const size_t n = S.size();
    const Mat3 W3 = cam.world_to_cam.upper3x3();
    const Mat4& W = cam.world_to_cam;

    auto t0 = Clock::now();
    float sink = 0.f;
    for (int it = 0; it < N_ITER; ++it) {
        for (size_t i = 0; i < n; ++i) {
            const Splat& sp = S[i];

            // A. 世界→视空间（始终保留：后续都依赖它）
            const Vec4 vh = W.transform(Vec4{sp.x, sp.y, sp.z, 1.f});
            const float vx = vh.x, vy = vh.y, vz = vh.z;
            if (vz < cam.znear || vz > cam.zfar) continue;
            const float inv_z  = 1.f / vz;
            const float inv_z2 = inv_z * inv_z;

            // C. quat → R
#ifdef ABLATE_QUAT
            Mat3 R = Mat3::identity();
            sink += sp.rot0 * 1e-30f;   // 吃掉输入，防优化掉
#else
            const Mat3 R = quat_to_mat3(sp.rot0, sp.rot1, sp.rot2, sp.rot3);
#endif

            // D. C = W3 · R · S
            const float s0 = sp.scale0, s1 = sp.scale1, s2 = sp.scale2;
            float C[3][3];
#ifdef ABLATE_WRS
            for (int r=0;r<3;++r){C[r][0]=R.m[r][0];C[r][1]=R.m[r][1];C[r][2]=R.m[r][2];}
            sink += (s0+s1+s2)*1e-30f;
#else
            float rs[3][3];
            rs[0][0]=R.m[0][0]*s0; rs[1][0]=R.m[1][0]*s0; rs[2][0]=R.m[2][0]*s0;
            rs[0][1]=R.m[0][1]*s1; rs[1][1]=R.m[1][1]*s1; rs[2][1]=R.m[2][1]*s1;
            rs[0][2]=R.m[0][2]*s2; rs[1][2]=R.m[1][2]*s2; rs[2][2]=R.m[2][2]*s2;
            for (int r=0;r<3;++r){
                const float w0=W3.m[r][0], w1=W3.m[r][1], w2=W3.m[r][2];
                C[r][0]=w0*rs[0][0]+w1*rs[1][0]+w2*rs[2][0];
                C[r][1]=w0*rs[0][1]+w1*rs[1][1]+w2*rs[2][1];
                C[r][2]=w0*rs[0][2]+w1*rs[1][2]+w2*rs[2][2];
            }
#endif

            // E. A = J · C
            const float j00 = cam.fx*inv_z,  j02 = -cam.fx*vx*inv_z2;
            const float j11 = cam.fy*inv_z,  j12 = -cam.fy*vy*inv_z2;
            float A0[3], A1[3];
#ifdef ABLATE_JC
            A0[0]=C[0][0]; A0[1]=C[0][1]; A0[2]=C[0][2];
            A1[0]=C[1][0]; A1[1]=C[1][1]; A1[2]=C[1][2];
            sink += (j00+j02+j11+j12)*1e-30f;
#else
            A0[0]=j00*C[0][0]+j02*C[2][0]; A0[1]=j00*C[0][1]+j02*C[2][1]; A0[2]=j00*C[0][2]+j02*C[2][2];
            A1[0]=j11*C[1][0]+j12*C[2][0]; A1[1]=j11*C[1][1]+j12*C[2][1]; A1[2]=j11*C[1][2]+j12*C[2][2];
#endif

            // F. Σ2D = A·Aᵀ
            float a = A0[0]*A0[0]+A0[1]*A0[1]+A0[2]*A0[2];
            float b = A0[0]*A1[0]+A0[1]*A1[1]+A0[2]*A1[2];
            float c = A1[0]*A1[0]+A1[1]*A1[1]+A1[2]*A1[2];
            a += 0.3f; c += 0.3f;

            // G. 各向异性钳制
#ifdef ABLATE_ANISO
            sink += b * 1e-30f;
#else
            {
                constexpr float kMaxAniso = 15.0f;
                const float mid0 = 0.5f*(a+c);
                const float rad0 = std::sqrt(std::fmax(0.f, 0.25f*(a-c)*(a-c)+b*b));
                const float l_max = mid0+rad0, l_min = mid0-rad0;
                if (l_max > 0.f) {
                    const float fl = l_max/kMaxAniso;
                    if (l_min < fl) {
                        const float mid2 = 0.5f*(l_max+fl);
                        const float rad2 = 0.5f*(l_max-fl);
                        float ct = 1.f, st = 0.f;
                        if (rad0 > 1e-12f) { ct = 0.5f*(a-c)/rad0; st = b/rad0; }
                        a = mid2 + rad2*ct; c = mid2 - rad2*ct; b = rad2*st;
                    }
                }
            }
#endif

            // H. 屏幕中心 + 半径
            const float px = cam.fx*vx*inv_z + cam.cx;
            const float py = cam.fy*vy*inv_z + cam.cy;
            const float det = a*c - b*b;
            if (det <= 1e-12f) continue;
            const float mid = 0.5f*(a+c);
            const float rad = std::sqrt(std::fmax(0.f, 0.25f*(a-c)*(a-c)+b*b));
            const float sr = 3.f*std::sqrt(std::fmax(mid+rad, 0.f));
            sink += px + py + sr;
        }
    }
    double ms = ms_since(t0);
    g_sink = sink;

    printf("%-14s  %8.2f ms /%d轮   %6.2f ns/点\n",
#ifdef ABLATE_QUAT
           "无 quat→R",
#elif defined(ABLATE_WRS)
           "无 C=W3RS",
#elif defined(ABLATE_JC)
           "无 A=J*C",
#elif defined(ABLATE_ANISO)
           "无 各向异性",
#else
           "完整(基线)",
#endif
           ms, N_ITER, ms*1e6/(double)(N_ITER*n));
    return 0;
}
