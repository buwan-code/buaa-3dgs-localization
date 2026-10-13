// projprofile.cpp — 拆解 project_gaussian 的耗时构成
//
// 目的：NEON 化之前先搞清楚「时间到底花在哪」。
//   把 project_gaussian 拆成若干可独立计时的子步骤，用同一批
//   随机高斯跑 N 次，报告各段耗时。
//
//   拆解维度（按代码顺序）：
//     A. 世界→视空间变换（加载 + 4x4 transform）
//     B. 视锥剔除（分支）
//     C. quat_to_mat3       （四元数→R，含 sqrt 归一化）
//     D. C = W3·R·S         （27 mul）
//     E. A = J·C            （12 mul）
//     F. Σ2D = A·Aᵀ         （9 mul）
//     G. 低通 + 各向异性钳制 （含 2 个 sqrt）
//     H. 屏幕中心 + 半径 + 剔除（含 2 个 sqrt）
//
// 用法：./build/projprofile [ply]

#include "camera.h"
#include "ply_loader.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace gsplat;
using Clock = std::chrono::high_resolution_clock;

static inline double ms_since(const Clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int main(int argc, char** argv) {
    const char* ply = (argc > 1) ? argv[1] : "data/pilot.ply";
    std::vector<Splat> S;
    std::vector<ShCoeffs> sh;
    if (!load_ply_splats(ply, S, sh, 1, true)) {
        fprintf(stderr, "load failed\n"); return 1;
    }
    printf("载入 %zu 高斯\n", S.size());

    Camera cam = Camera::look_at({0, 0, 60}, {0, 0, 0}, {0, 1, 0}, 512, 288, 45.f);

    const int N_ITER = 20;
    const size_t n = S.size();

    double t_world = 0, t_quat = 0, t_wrs = 0, t_jc = 0, t_aat = 0,
           t_clamp = 0, t_center = 0, t_total = 0;

    for (int it = 0; it < N_ITER; ++it) {
        for (size_t i = 0; i < n; ++i) {
            const Splat& sp = S[i];
            const Vec3 p{sp.x, sp.y, sp.z};
            const Vec3 s3{sp.scale0, sp.scale1, sp.scale2};

            Clock::time_point t0 = Clock::now();
            // A. 世界→视空间
            const Vec4 vh = cam.world_to_cam.transform(Vec4{p.x, p.y, p.z, 1.f});
            const float vx = vh.x, vy = vh.y, vz = vh.z;
            t_world += ms_since(t0);
            if (vz < cam.znear || vz > cam.zfar) continue;
            const float inv_z = 1.f / vz;
            const float inv_z2 = inv_z * inv_z;

            // C. quat → R
            t0 = Clock::now();
            const Mat3 R = quat_to_mat3(sp.rot0, sp.rot1, sp.rot2, sp.rot3);
            t_quat += ms_since(t0);

            // D. C = W3·R·S
            t0 = Clock::now();
            const Mat3 W3 = cam.world_to_cam.upper3x3();
            const float s0 = s3.x, s1 = s3.y, s2 = s3.z;
            float rs[3][3];
            rs[0][0]=R.m[0][0]*s0; rs[1][0]=R.m[1][0]*s0; rs[2][0]=R.m[2][0]*s0;
            rs[0][1]=R.m[0][1]*s1; rs[1][1]=R.m[1][1]*s1; rs[2][1]=R.m[2][1]*s1;
            rs[0][2]=R.m[0][2]*s2; rs[1][2]=R.m[1][2]*s2; rs[2][2]=R.m[2][2]*s2;
            float C[3][3];
            for (int r = 0; r < 3; ++r) {
                const float w0 = W3.m[r][0], w1 = W3.m[r][1], w2 = W3.m[r][2];
                C[r][0] = w0*rs[0][0] + w1*rs[1][0] + w2*rs[2][0];
                C[r][1] = w0*rs[0][1] + w1*rs[1][1] + w2*rs[2][1];
                C[r][2] = w0*rs[0][2] + w1*rs[1][2] + w2*rs[2][2];
            }
            t_wrs += ms_since(t0);

            // E. A = J·C
            t0 = Clock::now();
            const float j00 = cam.fx*inv_z, j02 = -cam.fx*vx*inv_z2;
            const float j11 = cam.fy*inv_z, j12 = -cam.fy*vy*inv_z2;
            float A0[3], A1[3];
            A0[0]=j00*C[0][0]+j02*C[2][0]; A0[1]=j00*C[0][1]+j02*C[2][1]; A0[2]=j00*C[0][2]+j02*C[2][2];
            A1[0]=j11*C[1][0]+j12*C[2][0]; A1[1]=j11*C[1][1]+j12*C[2][1]; A1[2]=j11*C[1][2]+j12*C[2][2];
            t_jc += ms_since(t0);

            // F. Σ2D = A·Aᵀ
            t0 = Clock::now();
            float a = A0[0]*A0[0]+A0[1]*A0[1]+A0[2]*A0[2];
            float b = A0[0]*A1[0]+A0[1]*A1[1]+A0[2]*A1[2];
            float c = A1[0]*A1[0]+A1[1]*A1[1]+A1[2]*A1[2];
            t_aat += ms_since(t0);

            // G. 低通 + 各向异性钳制
            t0 = Clock::now();
            a += 0.3f; c += 0.3f;
            constexpr float kMaxAniso = 15.0f;
            {
                const float mid0 = 0.5f*(a+c);
                const float rad0 = std::sqrt(std::fmax(0.f, 0.25f*(a-c)*(a-c)+b*b));
                const float l_max = mid0+rad0, l_min = mid0-rad0;
                if (l_max > 0.f) {
                    const float floor_ = l_max/kMaxAniso;
                    if (l_min < floor_) {
                        const float mid2 = 0.5f*(l_max+floor_);
                        const float rad2 = 0.5f*(l_max-floor_);
                        float cos2t = 1.f, sin2t = 0.f;
                        if (rad0 > 1e-12f) { cos2t = 0.5f*(a-c)/rad0; sin2t = b/rad0; }
                        a = mid2 + rad2*cos2t; c = mid2 - rad2*cos2t; b = rad2*sin2t;
                    }
                }
            }
            t_clamp += ms_since(t0);

            // H. 屏幕中心 + 半径
            t0 = Clock::now();
            const float px = cam.fx*vx*inv_z + cam.cx;
            const float py = cam.fy*vy*inv_z + cam.cy;
            const float det = a*c - b*b;
            if (det <= 1e-12f) continue;
            const float mid = 0.5f*(a+c);
            const float rad = std::sqrt(std::fmax(0.f, 0.25f*(a-c)*(a-c)+b*b));
            const float lam_major = mid + rad;
            const float sr = 3.f*std::sqrt(std::fmax(lam_major, 0.f));
            volatile float sink = px+py+sr; (void)sink;
            t_center += ms_since(t0);
        }
    }

    // 总计：整体跑一遍 project_gaussian
    Clock::time_point tt = Clock::now();
    volatile float acc = 0;
    for (int it = 0; it < N_ITER; ++it)
        for (size_t i = 0; i < n; ++i) {
            const Splat& sp = S[i];
            Splat2D r = project_gaussian(Vec3{sp.x,sp.y,sp.z},
                                         Vec3{sp.scale0,sp.scale1,sp.scale2},
                                         sp.rot0,sp.rot1,sp.rot2,sp.rot3, cam);
            acc += r.cx;
        }
    t_total = ms_since(tt);

    const double per = 1.0/(double)(N_ITER*n);
    printf("\n=== 单点耗时拆解（%d 次 × %zu 点）===\n", N_ITER, n);
    printf("  A 世界→视空间    %7.3f ms  (%6.2f ns/点, %5.1f%%)\n", t_world,  t_world*1e6*per, 100*t_world/t_total);
    printf("  C quat→R         %7.3f ms  (%6.2f ns/点, %5.1f%%)\n", t_quat,   t_quat*1e6*per,  100*t_quat/t_total);
    printf("  D C=W3·R·S       %7.3f ms  (%6.2f ns/点, %5.1f%%)\n", t_wrs,    t_wrs*1e6*per,   100*t_wrs/t_total);
    printf("  E A=J·C          %7.3f ms  (%6.2f ns/点, %5.1f%%)\n", t_jc,     t_jc*1e6*per,    100*t_jc/t_total);
    printf("  F Σ=A·Aᵀ         %7.3f ms  (%6.2f ns/点, %5.1f%%)\n", t_aat,    t_aat*1e6*per,   100*t_aat/t_total);
    printf("  G 低通+钳制      %7.3f ms  (%6.2f ns/点, %5.1f%%)\n", t_clamp,  t_clamp*1e6*per, 100*t_clamp/t_total);
    printf("  H 中心+半径      %7.3f ms  (%6.2f ns/点, %5.1f%%)\n", t_center, t_center*1e6*per,100*t_center/t_total);
    printf("  ─────────────────────────────────────────────\n");
    printf("  子项合计         %7.3f ms\n", t_world+t_quat+t_wrs+t_jc+t_aat+t_clamp+t_center);
    printf("  ★ 整体 project   %7.3f ms  (%6.2f ns/点)\n", t_total, t_total*1e6*per);
    printf("    整体 / 3M 点    %7.1f ms\n", t_total/(double)N_ITER/1e6*3.0);
    return 0;
}
