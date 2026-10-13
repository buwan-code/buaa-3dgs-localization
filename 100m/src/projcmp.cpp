// projcmp.cpp — 对比 project_gaussian（标量）与 project_gaussian_fast（折叠 M=J·W3）
//
// 1) 正确性：同一批高斯，两条路径的 (cx,cy,a,b,c,sigma_radius,valid,inv_det)
//    逐项比对，报告最大绝对/相对偏差、valid 位是否一致。
// 2) 速度：各跑 N 轮，报告 ns/点。

#include "camera.h"
#include "ply_loader.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace gsplat;
using Clock = std::chrono::high_resolution_clock;
static inline double ms_since(const Clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// 从点云估 AABB，摆一个「看得见东西」的相机（与 gsrender 一致）
static Camera auto_camera(const std::vector<Splat>& S, int w, int h) {
    float mnx=1e30f,mny=1e30f,mnz=1e30f,mxx=-1e30f,mxy=-1e30f,mxz=-1e30f;
    for (const auto& s : S) {
        mnx=std::fmin(mnx,s.x); mxx=std::fmax(mxx,s.x);
        mny=std::fmin(mny,s.y); mxy=std::fmax(mxy,s.y);
        mnz=std::fmin(mnz,s.z); mxz=std::fmax(mxz,s.z);
    }
    const Vec3 c{(mnx+mxx)*0.5f,(mny+mxy)*0.5f,(mnz+mxz)*0.5f};
    const float dx=mxx-mnx, dy=mxy-mny, dz=mxz-mnz;
    const float rad = 0.5f*std::sqrt(dx*dx+dy*dy+dz*dz);
    const float dist = rad*2.2f;
    printf("AABB 中心 (%.1f, %.1f, %.1f)  半径 %.1f  相机距离 %.1f\n",
           c.x,c.y,c.z,rad,dist);
    Camera cam = Camera::look_at({c.x, c.y - dist*0.9f, c.z + dist},
                                 c, {0.f, -1.f, 0.f}, w, h, 45.f);
    cam.znear = 0.05f;
    cam.zfar  = dist + rad*6.f;
    return cam;
}

#include "projfast.h"
using gsplat::project_gaussian_fast;

static volatile float g_sink = 0.f;

int main(int argc, char** argv) {
    const char* ply = (argc > 1) ? argv[1] : "data/pilot.ply";
    std::vector<Splat> S; std::vector<ShCoeffs> sh;
    if (!load_ply_splats(ply, S, sh, 1, false)) return 1;
    printf("载入 %zu 高斯\n", S.size());

    Camera cam = auto_camera(S, 512, 288);
    const ProjConsts PK = ProjConsts::from_camera(cam);

    // ── 正确性 ──────────────────────────────────────────────────
    double dcx=0, dcy=0, da=0, db=0, dcc=0, dsr=0, did=0;
    long nvalid_a=0, nvalid_b=0, nvalid_diff=0;
    long ncmp=0;
    for (size_t i = 0; i < S.size(); ++i) {
        const Splat& sp = S[i];
        const Vec3 P{sp.x, sp.y, sp.z}, SC{sp.scale0, sp.scale1, sp.scale2};
        Splat2D A = project_gaussian(P, SC, sp.rot0,sp.rot1,sp.rot2,sp.rot3, cam);
        Splat2D B = project_gaussian_fast(P, SC, sp.rot0,sp.rot1,sp.rot2,sp.rot3, cam, PK);
        if (A.valid) nvalid_a++;
        if (B.valid) nvalid_b++;
        if (A.valid != B.valid) nvalid_diff++;
        if (!A.valid && !B.valid) continue;
        ncmp++;
        dcx = std::fmax(dcx, std::fabs(A.cx-B.cx));
        dcy = std::fmax(dcy, std::fabs(A.cy-B.cy));
        da  = std::fmax(da,  std::fabs(A.a -B.a ));
        db  = std::fmax(db,  std::fabs(A.b -B.b ));
        dcc = std::fmax(dcc, std::fabs(A.c -B.c ));
        dsr = std::fmax(dsr, std::fabs(A.sigma_radius-B.sigma_radius));
        if (A.valid && B.valid && A.inv_det != 0.f)
            did = std::fmax(did, std::fabs(A.inv_det-B.inv_det)/std::fabs(A.inv_det));
    }
    printf("\n=== 正确性 ===\n");
    printf("  valid: 标量 %ld / 折叠 %ld / 不一致 %ld\n", nvalid_a, nvalid_b, nvalid_diff);
    printf("  参与比对 %ld 点\n", ncmp);
    printf("  max|Δcx|=%.3e  max|Δcy|=%.3e\n", dcx, dcy);
    printf("  max|Δa|=%.3e   max|Δb|=%.3e  max|Δc|=%.3e\n", da, db, dcc);
    printf("  max|Δsigma_radius|=%.3e\n", dsr);
    printf("  max 相对 |Δinv_det|=%.3e\n", did);

    // ── 速度 ────────────────────────────────────────────────────
    const int N = 8;
    const size_t n = S.size();
    float sink = 0;

    auto t0 = Clock::now();
    for (int it=0; it<N; ++it)
        for (size_t i=0;i<n;++i) {
            const Splat& sp=S[i];
            Splat2D r = project_gaussian(Vec3{sp.x,sp.y,sp.z},
                Vec3{sp.scale0,sp.scale1,sp.scale2}, sp.rot0,sp.rot1,sp.rot2,sp.rot3, cam);
            sink += r.cx;
        }
    double ms_a = ms_since(t0);
    g_sink = sink;

    t0 = Clock::now();
    for (int it=0; it<N; ++it)
        for (size_t i=0;i<n;++i) {
            const Splat& sp=S[i];
            Splat2D r = project_gaussian_fast(Vec3{sp.x,sp.y,sp.z},
                Vec3{sp.scale0,sp.scale1,sp.scale2}, sp.rot0,sp.rot1,sp.rot2,sp.rot3, cam, PK);
            sink += r.cx;
        }
    double ms_b = ms_since(t0);
    g_sink = sink;

    const double per = 1.0/(double)(N*n);
    printf("\n=== 速度（单线程，%d 轮）===\n", N);
    printf("  标量   %8.1f ms   %6.2f ns/点\n", ms_a, ms_a*1e6*per);
    printf("  折叠   %8.1f ms   %6.2f ns/点\n", ms_b, ms_b*1e6*per);
    printf("  加速   %.3fx  (−%.1f%%)\n", ms_a/ms_b, 100*(1-ms_b/ms_a));
    printf("  折算 2.77M 点：标量 %.0f ms → 折叠 %.0f ms\n",
           ms_a/N/1e6*2.76756, ms_b/N/1e6*2.76756);
    return 0;
}
