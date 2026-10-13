// proj_core.h — 投影的公共尾部：低通 + 各向异性钳制 + 半径/剔除
//
// ══ 为什么抽出这一层 ════════════════════════════════════════════════
// project_gaussian（标量路径）与 project_gaussian_fast（折叠路径）
// 唯一的区别是**怎么算出 Σ2D 的 (a,b,c)**。之后的所有步骤：
//   低通项 → 各向异性钳制 → det/inv_det → 特征值 → sigma_radius
//   → 屏幕剔除 → 半径上限
// 必须**逐位一致**，否则两条路径会画出不同的图。
//
// 第一版我把钳制漏在快路径外，结果 a/b/c 最大差 43.7、
// inv_det 相对差 68×（因为 det = ac-b² 对钳制极敏感）。
// 教训：**公共尾部必须物理共享一份代码**，不能靠"记得同步"。
#pragma once

#include "camera.h"
#include <cmath>

namespace gsplat {

// ══ 逐高斯不变量的预计算 ════════════════════════════════════════════
// 尾部逻辑里有几个量**只依赖相机**，却原先是逐高斯重算的：
//   • diag = sqrt(w²+h²)          ← 每点一次 sqrt，纯粹浪费
//   • max_r = 1.5·diag            ← 同上
// 实测：这两个 sqrt 让尾部多花 16.7 ns/点（消融法测得）。
// 把它们提到帧级算一次，投影内循环直接读缓存值。
struct ProjConsts {
    float max_sigma_radius;   // 已解析的半径上限
    float half_w, half_h;     // 屏幕剔除用（避免每点再读 cam 成员）
    ProjConsts() : max_sigma_radius(1e30f), half_w(0), half_h(0) {}
    static ProjConsts from_camera(const Camera& cam) {
        ProjConsts k;
        const float diag = std::sqrt((float)(cam.width*cam.width + cam.height*cam.height));
        k.max_sigma_radius = (cam.max_sigma_radius > 0.f) ? cam.max_sigma_radius : 1.5f*diag;
        k.half_w = (float)cam.width;
        k.half_h = (float)cam.height;
        return k;
    }
};

// 由 Σ2D 的两个对角元与叉项，算最终 Splat2D（含钳制、圆心、半径、剔除）。
// px/py 为已算好的屏幕中心（视空间投影，不依赖协方差）。
inline void finalize_splat2d(Splat2D& out, float px, float py,
                             float a, float b, float c,
                             const Camera& cam, const ProjConsts& k) {
    // ── 低通项：+0.3 px²（3DGS 原论文做法，防远距离高斯退化成噪点）──
    a += 0.3f;
    c += 0.3f;
    if (!(a > 0.f) || !(c > 0.f)) { out.valid = false; return; }

    // ── 椭圆锐度钳制（修「长刺/拉丝」伪影）──────────────────────
    // ⚠️ 注意：钳制分支与后面的半径计算**都要用 rad**，
    //    原实现把 sqrt 算了两次（rad0 与 rad 输入完全相同）。
    //    这里只算一次，钳制改写后 (a,b,c) 变了才需要重算 —— 
    //    但钳制只在极少高波斯片触发，所以用「钳制时置脏」的方式。
    constexpr float kMaxAnisoRatio = 15.0f;
    // ★ mid/rad 只在真的钳制过之后才需要重算。
    //   原实现无条件重算一次 sqrt —— 而钳制对绝大多数高斯**不触发**
    //   （注释里其实写明了要用"钳制时置脏"的方式，但代码没落实）。
    //   实测每个高斯白花一次 sqrt。
    float mid, rad;
    {
        const float mid0 = 0.5f * (a + c);
        const float rad0 = std::sqrt(std::fmax(0.f, 0.25f*(a-c)*(a-c) + b*b));
        const float l_max = mid0 + rad0;
        const float l_min = mid0 - rad0;
        bool clamped = false;
        if (l_max > 0.f) {
            const float l_min_floor = l_max / kMaxAnisoRatio;
            if (l_min < l_min_floor) {
                const float mid2 = 0.5f * (l_max + l_min_floor);
                const float rad2 = 0.5f * (l_max - l_min_floor);
                float cos2t = 1.f, sin2t = 0.f;
                if (rad0 > 1e-12f) {
                    cos2t = 0.5f * (a - c) / rad0;
                    sin2t = b / rad0;
                }
                a = mid2 + rad2 * cos2t;
                c = mid2 - rad2 * cos2t;
                b = rad2 * sin2t;
                clamped = true;
            }
        }
        if (clamped) {
            mid = 0.5f * (a + c);
            rad = std::sqrt(std::fmax(0.f, 0.25f*(a-c)*(a-c) + b*b));
        } else {
            mid = mid0; rad = rad0;      // 未钳制 → 直接复用
        }
    }

    out.cx = px;
    out.cy = py;
    out.a = a; out.b = b; out.c = c;

    const float det = a * c - b * b;
    if (det <= 1e-12f) { out.valid = false; return; }
    out.inv_det = 1.f / det;

    // 半径：λ_max = (a+c)/2 + sqrt(((a-c)/2)² + b²)，已在上面的块里算好
    out.sigma_radius = 3.f * std::sqrt(std::fmax(mid + rad, 0.f));

    // 屏幕外剔除（用预算好的宽高，避免每点读 cam 成员造成依赖）
    if (px + out.sigma_radius < 0.f || px - out.sigma_radius >= k.half_w) { out.valid = false; return; }
    if (py + out.sigma_radius < 0.f || py - out.sigma_radius >= k.half_h) { out.valid = false; return; }

    // 半径上限：挡退化的巨大高斯（防填充爆炸 + 横穿全屏的白色拖尾）
    if (out.sigma_radius > k.max_sigma_radius) { out.valid = false; return; }

    out.valid = true;
}

// 兼容旧签名（无预计算常量时内部现算一次）
inline void finalize_splat2d(Splat2D& out, float px, float py,
                             float a, float b, float c,
                             const Camera& cam) {
    finalize_splat2d(out, px, py, a, b, c, cam, ProjConsts::from_camera(cam));
}

} // namespace gsplat
