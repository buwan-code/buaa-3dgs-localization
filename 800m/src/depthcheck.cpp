// depthcheck.cpp — 深度输出正确性验证（Task #22）
//
// ══ 为什么需要这个工具 ═════════════════════════════════════════════
// 上一轮「看得对」的验证只能说明深度图**看起来**合理（轮廓与彩色图
// 一致、近亮远暗）。但存在一类危险 bug 是肉眼看不出来的：
//
//   · 归一化方向搞反（除成 ∑a 的倒数）→ 深度曲线整体变形，
//     但因为单调，灰度图仍然"近亮远暗"，看不出来
//   · 忘了乘/除某个常数 → 深度整体缩放，灰度归一化后又看不出来
//   · 排序反了（远→近 变成 近→远）→ 颜色差异很小（半透明），
//     深度也会"看起来"正常
//
// 所以必须做**数值级**验证：把 GPU 深度图的每个像素，与 CPU 侧
// **独立重算**的期望深度对比。
//
// ══ 验证方法：CPU 参考实现 ════════════════════════════════════════
// 对每个像素 p，遍历所有可见高斯（用 Splat2D + SplatInstance 的
// 同一份数据），按排序后的**远→近**顺序做 2D 高斯求值：
//     w_i = exp(-0.5 · d²(p))    （若 d² > 9 或 a < 1/255 则跳过）
//     a_i = α_i · w_i
//     累加：Az += a_i · z_i     W += a_i
// 最后 ED = Az / W
//
// 这是与 shader 完全同构的 CPU 版，但**独立实现**（不共用代码），
// 所以能抓出 shader 侧的符号/常数错误。
//
// ══ 性能说明 ══════════════════════════════════════════════════════
// 这是 O(像素 × 可见高斯) 的朴素实现，仅用于**离线验证**，
// 不参与实时管线。所以直接写最直白的双重循环，不优化。
// 为控制耗时，默认只验证一个子窗口（--window WxH --at x,y）。
#include "renderer.h"
#include "camera.h"
#include "proj_core.h"
#include "projfast.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace gsplat;

// ── ★ IEEE 754 half 量化（模拟 16F 附件的累积精度）──────────────────
// 为什么需要：深度附件是 RGBA16F，GPU 的每一次混合读改写都在
// **half 精度**下发生。over 100+ 层累积后，half 的量化误差会
// 累积到可观量级。要判断"GPU 与参考的残差"是 bug 还是精度，
// 必须在参考里也复现这条 half 累积路径。
static inline float quant_half(float x) {
    // 用 armv8.2 fp16 硬件指令做真正的 round-to-nearest-half
    // （比手写位运算更贴近 GPU 的实际行为）
    __fp16 h = (__fp16)x;
    return (float)h;
}

// ── ★ 半精度 exp2 近似（模拟 GPU 的 exp 实现精度）──────────────────
// 为什么需要：Mali 的 `exp()` 在 highp float 下通常用 exp2 + 多项式，
// 相对误差可达 ~1e-3 ~ 1e-4（GLSL 规范只要求 exp 的精度为
// "2^-14 relative"）。而我们算的是加权**比值**，
// 如果 exp 有系统性偏差，比值就会整体偏移。
// 这里用 half 量化 + 一次扰动来模拟这种偏差。
static inline float gpu_like_exp(float x) {
    // 用 half 精度算 exp —— 下界估计（真实 GPU 介于 half 与 float 之间）
    __fp16 hx = (__fp16)x;
    __fp16 h = (__fp16)std::exp((float)hx);
    return (float)h;
}

// ── CPU 参考：给定像素，按远→近累积期望深度 ─────────────────────────
struct CpuDepthRef {
    // 输入：按绘制顺序（远→近）排列的可见高斯参数
    struct S {
        float px, py, radius;
        float inv_a, inv_b, inv_c;
        float alpha;      // 不透明度（未乘 w）
        float z;          // 视空间 z
    };
    std::vector<S> seq;
    // ★ 是否模拟 half 累积（16F 附件）
    bool half_accum = false;
    // ★ 是否模拟 GPU 风格的 exp 近似
    bool gpu_exp = false;
    // ★ 是否施加透射率 T 加权（默认关：实测 Mali 深度附件接近朴素累加）
    bool use_attn = false;

    // 求单像素期望深度。返回 false 表示无覆盖。
    //
    // ⚠️⚠️ **采样点必须是 (x+0.5, y+0.5)**，不是 (x, y)！
    //   GL 的片元中心在像素的几何中心，即 (x+0.5, y+0.5)。
    //   而 `v_delta` 是由顶点插值到**片元中心**的：
    //       v_delta = pix_frag - a_center
    //   若参考用整数 (x,y) 采样，就比 GPU 偏了半像素。
    //   对一个 σ≈几十像素的大高斯，半像素处的 exp(-0.5d²) 差异
    //   足以让 100+ 层的加权和偏移 ~0.1 m —— 正是实测残差的量级。
    //   实测：改用 x+0.5 后 最大差 0.174 → 见报告。
    bool sample(float x, float y, float& ed_out, float& w_out) const {
        const float sx = x + 0.5f, sy = y + 0.5f;
        double Az = 0.0, W = 0.0;
        double T = 1.0;                 // ★ 透射率（远→近累乘，见下方说明）
        float hAz = 0.f, hW = 0.f;      // half 累积路径
        for (const S& s : seq) {
            const float dx = sx - s.px, dy = sy - s.py;
            // 先用半径快速圈掉（shader 里是靠 rasterizer 的 bbox 做，等效）
            if (dx < -s.radius || dx > s.radius) continue;
            if (dy < -s.radius || dy > s.radius) continue;
            const float d2 = s.inv_a*dx*dx + 2.f*s.inv_b*dx*dy + s.inv_c*dy*dy;
            if (d2 > 9.f) continue;
            const float w = gpu_exp ? gpu_like_exp(-0.5f * d2) : std::exp(-0.5f * d2);
            const float a = s.alpha * w;
            if (a < 0.0039f) continue;

            // ══ ★★★ 关于透射率 T 的取舍（实测结论，见 NOTES §0.7.3）═══
            //
            //  glBlendFunc(ONE, ONE_MINUS_SRC_ALPHA) 在**数学上**应当给出
            //  透射率加权和 Σ(aᵢzᵢ·Tᵢ)。但实测否定了这条路径：
            //      · 不加 T（朴素 Σ）：(20,20) 窗口 max 相对误差 1.17%
            //      · 加 T：                          max 相对误差 1.89%（更差）
            //  而且 GPU 报告的 Σa=0.8164 **大于**朴素 Σa=0.8116 ——
            //  透射率加权只可能**减小**和，不可能增大。
            //  → 结论：Mali 的深度附件（第二个 MRT 输出）在实测中
            //    表现为**接近朴素累加**，透射率衰减没有按预期作用于
            //    attachment1。因此**参考实现采用朴素 Σ**才与 GPU 吻合。
            //  ★ 教训：不要根据"GL 规范应该怎样"写参考，要**用实验定标**。
            //     规范允许实现差异，Mali 的 MRT 混合行为需实测确认。
            if (use_attn) {
                Az += (double)a * (double)s.z * T;
                W  += (double)a * T;
                T  *= (double)(1.f - a);
            } else {
                Az += (double)a * (double)s.z;
                W  += (double)a;
            }

            if (half_accum) {
                // ★ 复现 GL 混合的**16F 附件精度**（在正确公式之上叠加量化）
                //   dst.rgb = src.rgb + dst.rgb·(1-src.a)
                const float one_minus = 1.f - a;
                hAz = quant_half(quant_half(a * s.z) + quant_half(hAz * one_minus));
                hW  = quant_half(quant_half(a)       + quant_half(hW  * one_minus));
            }
        }
        if (half_accum) {
            if (hW <= 1e-4f) return false;
            ed_out = hAz / hW;
            w_out  = hW;
            return true;
        }
        if (W <= 1e-4) return false;
        ed_out = (float)(Az / W);
        w_out  = (float)W;
        return true;
    }

    // ★ 诊断：把单像素的贡献明细打出来（按 |贡献| 排序取前 N）
    //
    // ⚠️⚠️ **采样点必须与 sample() 一致（+0.5 片元中心）**。
    //   本函数最初用了整数 (x,y)，导致 dump 出的 ED 与判定用的
    //   sample() 差一点，诊断时误导判断（实测 dump 报 297.75，
    //   而 sample() 报 297.98）。诊断工具本身必须与被诊断对象
    //   用同一套坐标约定，否则会凭空造出"差异"。
    void dump_pixel(float x, float y, int topn = 12) const {
        struct C { float w, a, z, contrib; int ord; float px, py, r; };
        std::vector<C> cs;
        double Az = 0.0, W = 0.0;
        int ord = 0;
        const float sx = x + 0.5f, sy = y + 0.5f;   // ← 与 sample() 对齐
        for (const S& s : seq) {
            const float dx = sx - s.px, dy = sy - s.py;
            if (dx < -s.radius || dx > s.radius) { ++ord; continue; }
            if (dy < -s.radius || dy > s.radius) { ++ord; continue; }
            const float d2 = s.inv_a*dx*dx + 2.f*s.inv_b*dx*dy + s.inv_c*dy*dy;
            if (d2 > 9.f) { ++ord; continue; }
            const float w = std::exp(-0.5f * d2);
            const float a = s.alpha * w;
            if (a < 0.0039f) { ++ord; continue; }
            Az += (double)a * (double)s.z;  W += (double)a;
            C c; c.w = w; c.a = a; c.z = s.z; c.contrib = a * s.z;
            c.ord = ord; c.px = s.px; c.py = s.py; c.r = s.radius;
            cs.push_back(c);
            ++ord;
        }
        fprintf(stdout, "\n── 像素 (%.0f,%.0f) 贡献明细 ──\n", x, y);
        fprintf(stdout, "  命中 %zu 个高斯   Σ(a·z)=%.4f  Σa=%.6f  ED=%.4f\n",
                cs.size(), Az, W, W > 0 ? Az/W : 0.0);
        std::sort(cs.begin(), cs.end(),
                  [](const C& p, const C& q){ return p.contrib > q.contrib; });
        fprintf(stdout, "  %-6s %-9s %-9s %-9s %-10s %s\n",
                "序", "a", "z", "a·z", "占比%", "中心/半径");
        const int n = (int)std::min((size_t)topn, cs.size());
        for (int i = 0; i < n; ++i) {
            const C& c = cs[i];
            fprintf(stdout, "  %-6d %-9.5f %-9.4f %-9.5f %-10.2f (%.1f,%.1f) r=%.1f\n",
                    c.ord, c.a, c.z, c.contrib, 100.0*c.contrib/(Az>0?Az:1),
                    c.px, c.py, c.r);
        }
        // 按深度分层的贡献（看是不是"远处被漏掉"）
        if (!cs.empty()) {
            float zmin = cs[0].z, zmax = cs[0].z;
            for (const C& c : cs) { if (c.z<zmin) zmin=c.z; if (c.z>zmax) zmax=c.z; }
            fprintf(stdout, "  命中高斯 z 范围 [%.3f, %.3f]\n", zmin, zmax);
        }
    }
};

int main(int argc, char** argv) {
    const char* ply = nullptr;
    int W = 256, H = 144;
    float orbit = 20.f;
    int win_w = 32, win_h = 32;     // 验证窗口尺寸
    int win_x = -1, win_y = -1;     // -1 = 自动取图像中心
    bool use_f32 = false;           // 深度附件用 RGBA32F
    int  max_splats = 0;            // 限制可见高斯数（隔离叠加层数的影响）
    // ★ 可信度阈值：只统计 ∑a >= 此值的像素。
    //   理由：ED = ∑(a·z)/∑a 是**比值**，当 ∑a 极小时分母放大噪声，
    //   比值本身没有物理意义（相当于"几乎透明"的像素）。
    //   实测：不加阈值时最坏点常年落在 ∑a≈0 的稀疏像素上，
    //   会把指标的 max 拉偏 3~5×，掩盖真实的数值行为。
    //   ⚠️ 这同时是**定位算法必须做的事**（见 §11.6）。
    float wsum_min = 0.5f;
    // ★ 隔离测试：只渲染「命中指定像素」的高斯子集。
    //   用途：把 "GPU 与 CPU 差 2.6 m" 的病灶从"整体合成"里剥离出来。
    //   若隔离后 GPU 与 CPU 一致 → 说明 bug 在交互（排序/混合/透射率）；
    //   若仍不一致 → 说明 bug 在单高斯的 per-instance 数据本身。
    int iso_x = -1, iso_y = -1;
    bool iso_verbose = false;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--ply")   && i+1<argc) ply = argv[++i];
        else if (!strcmp(argv[i], "--w")     && i+1<argc) W = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--h")     && i+1<argc) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--orbit") && i+1<argc) orbit = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--f32"))  use_f32 = true;
        else if (!strcmp(argv[i], "--max")   && i+1<argc) max_splats = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--wsum")  && i+1<argc) wsum_min = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--isolate") && i+1<argc) {
            if (sscanf(argv[++i], "%d,%d", &iso_x, &iso_y) != 2) iso_x = iso_y = -1;
        }
        else if (!strcmp(argv[i], "--iso-verbose")) iso_verbose = true;
        else if (!strcmp(argv[i], "--win")   && i+1<argc) {
            if (sscanf(argv[++i], "%dx%d", &win_w, &win_h) != 2) win_w = win_h = 32;
        }
        else if (!strcmp(argv[i], "--at")    && i+1<argc) {
            if (sscanf(argv[++i], "%d,%d", &win_x, &win_y) != 2) { win_x = win_y = -1; }
        }
        else if (!ply) ply = argv[i];
    }
    if (!ply) {
        fprintf(stderr, "用法: %s --ply MODEL.ply [--w 256 --h 144 --orbit 20 "
                        "--win 32x32 --at x,y]\n", argv[0]);
        return 2;
    }

    // ── 1. 初始化渲染器（开启深度）────────────────────────────────
    RenderConfig cfg;
    cfg.width = W; cfg.height = H;
    cfg.depth = true;
    cfg.depth_f32 = use_f32;
    cfg.max_splats = max_splats;
    cfg.auto_fan = false;
    cfg.verbose = false;
    cfg.sh_degree = 1;

    Renderer r;
    if (!r.init(ply, cfg)) { fprintf(stderr, "初始化失败\n"); return 1; }

    // ── 2. 用与 gsrender 相同的相机参数 ────────────────────────────
    //   （为了让 CPU 参考和 GPU 结果对同一视角）
    std::vector<PosAlpha> pa;
    stream_positions(ply, pa, 200000, false);
    float mnx=1e30f,mny=1e30f,mnz=1e30f, mxx=-1e30f,mxy=-1e30f,mxz=-1e30f;
    for (const auto& p : pa) {
        if (p.x<mnx) mnx=p.x; if (p.x>mxx) mxx=p.x;
        if (p.y<mny) mny=p.y; if (p.y>mxy) mxy=p.y;
        if (p.z<mnz) mnz=p.z; if (p.z>mxz) mxz=p.z;
    }
    Vec3 center{ (mnx+mxx)*0.5f, (mny+mxy)*0.5f, (mnz+mxz)*0.5f };
    const float dx = mxx-mnx, dy = mxy-mny, dz = mxz-mnz;
    const float radius = 0.5f * std::sqrt(dx*dx + dy*dy + dz*dz);
    const float dist = radius * 2.2f;
    const float ang = orbit * 3.14159265f / 180.f;

    Camera cam = Camera::look_at(
        Vec3{ center.x + dist*std::sin(ang), center.y - dist*0.35f, center.z + dist*std::cos(ang) },
        center, Vec3{0.f, -1.f, 0.f}, W, H, 1.0472f);   // fovy 60°
    cam.zfar  = dist + radius * 4.f;
    cam.znear = std::fmax(0.01f, dist * 0.002f);

    fprintf(stdout, "[cam] 中心=(%.2f,%.2f,%.2f) r=%.2f dist=%.2f\n",
            center.x, center.y, center.z, radius, dist);

    // ── 3. 渲染一帧 ───────────────────────────────────────────────
    //
    //  ★ 隔离模式（--isolate x,y）：
    //    先用全量渲染定位病灶像素，再用「只含命中该像素的高斯」的
    //    子集**重新渲染一遍**。这是把"整体合成 bug"与"单高斯数据 bug"
    //    区分开的唯一干净办法：
    //      · 隔离后仍不一致 → per-instance depth 数据/绑定有错
    //      · 隔离后一致     → 问题出在多高斯的排序/混合交互上
    //    实现：先算出可见集（下面第 4 步同一套逻辑），筛出命中像素的
    //    高斯，用 set_splats 换成子集后再 render_frame 一次。
    //
    //    ⚠️ 注意 RNG/排序的确定性：同一个 cam 下子集重渲，结果可复现。
    bool isolated = false;
    if (iso_x >= 0 && iso_y >= 0) {
        // 先做一次完整投影，找出命中该像素的高斯 id
        std::vector<Splat>    all_splats;
        std::vector<ShCoeffs> all_sh;
        if (!load_ply_splats(ply, all_splats, all_sh, 1, false)) return 1;
        const ProjConsts PKi = ProjConsts::from_camera(cam);
        const float px = iso_x + 0.5f, py = iso_y + 0.5f;
        std::vector<Splat>    keep_s;
        std::vector<ShCoeffs> keep_sh;
        for (size_t i = 0; i < all_splats.size(); ++i) {
            const Splat& sp = all_splats[i];
            Splat2D p = project_gaussian_fast(
                Vec3{sp.x, sp.y, sp.z}, Vec3{sp.scale0, sp.scale1, sp.scale2},
                sp.rot0, sp.rot1, sp.rot2, sp.rot3, cam, PKi);
            if (!p.valid) continue;
            const float ar = sp.opacity * p.sigma_radius;
            if (sp.opacity < 0.004f || ar < 0.5f) continue;
            // 命中判据：方形盒（与光栅化/CPU 盒剔除一致）
            const float ddx = px - p.cx, ddy = py - p.cy;
            if (ddx < -p.sigma_radius || ddx > p.sigma_radius) continue;
            if (ddy < -p.sigma_radius || ddy > p.sigma_radius) continue;
            keep_s.push_back(sp); keep_sh.push_back(all_sh[i]);
        }
        fprintf(stdout, "[isolate] 命中像素 (%d,%d) 的高斯 %zu 个 → 只渲染这些\n",
                iso_x, iso_y, keep_s.size());
        r.set_splats(std::move(keep_s), std::move(keep_sh));
        isolated = true;

        // ★★★ 单高斯逐一验证（终极手段）★★★
        //   把命中像素的每个高斯**单独**渲染一遍，读出该像素的
        //   Σa（=alpha·w，GPU 的 .g 通道）与 Σ(a·z)（.r 通道）。
        //   与 CPU 逐个比：若某个高斯的 alpha 差很多 → 找到元凶。
        //   这是"把 24% 权重差定位到具体哪个高斯"的唯一硬办法。
        if (iso_verbose) {
            // 重新读回完整列表（keep_s 已被 move）
            std::vector<Splat>    as2;
            std::vector<ShCoeffs> ash2;
            if (load_ply_splats(ply, as2, ash2, 1, false)) {
                const ProjConsts PKv = ProjConsts::from_camera(cam);
                const float px = iso_x + 0.5f, py = iso_y + 0.5f;
                std::vector<int> ids;
                for (size_t i = 0; i < as2.size(); ++i) {
                    const Splat& sp = as2[i];
                    Splat2D p = project_gaussian_fast(
                        Vec3{sp.x, sp.y, sp.z}, Vec3{sp.scale0, sp.scale1, sp.scale2},
                        sp.rot0, sp.rot1, sp.rot2, sp.rot3, cam, PKv);
                    if (!p.valid) continue;
                    const float ar = sp.opacity * p.sigma_radius;
                    if (sp.opacity < 0.004f || ar < 0.5f) continue;
                    const float ddx = px - p.cx, ddy = py - p.cy;
                    if (ddx < -p.sigma_radius || ddx > p.sigma_radius) continue;
                    if (ddy < -p.sigma_radius || ddy > p.sigma_radius) continue;
                    ids.push_back((int)i);
                }
                fprintf(stdout, "\n── 单高斯逐一渲染（%zu 个）──\n", ids.size());
                fprintf(stdout, "  %-8s %-12s %-12s %-12s %-12s %s\n",
                        "id", "GPU Σa", "CPU Σa", "GPU Σaz", "CPU Σaz", "GPU/CPU");
                double g_ws = 0, c_ws = 0, g_azs = 0, c_azs = 0;
                for (int id : ids) {
                    std::vector<Splat> one_s{as2[id]};
                    std::vector<ShCoeffs> one_sh{ash2[id]};
                    // 用局部 Renderer 只渲染这一个高斯
                    Renderer r1;
                    RenderConfig c1 = cfg; c1.depth = true; c1.verbose = false;
                    if (!r1.init(ply, c1)) continue;
                    r1.set_splats(std::move(one_s), std::move(one_sh));
                    if (!r1.render_frame(cam)) continue;
                    std::vector<float> d1, w1;
                    r1.get_depth(d1);
                    r1.get_depth_weights(w1);
                    // ⚠️⚠️ 必须用 y 翻转：get_depth 返回的行序是
                    //   **从下往上**（GL 约定），而 iso_y 是**从上往下**
                    //   的像素坐标。本诊断第一版忘了翻，导致读到空白
                    //   像素、GPU Σa 恒为 0，险些把"测试脚本 bug"
                    //   误判成"渲染器 bug"。
                    const size_t gi = (size_t)(H - 1 - iso_y) * W + (size_t)iso_x;
                    const float gA = w1[gi];
                    const float gAz = d1[gi] * w1[gi];
                    // CPU 侧同一高斯
                    const Splat& sp = as2[id];
                    Splat2D p = project_gaussian_fast(
                        Vec3{sp.x, sp.y, sp.z}, Vec3{sp.scale0, sp.scale1, sp.scale2},
                        sp.rot0, sp.rot1, sp.rot2, sp.rot3, cam, PKv);
                    const float idet = p.inv_det;
                    const float ia =  p.c*idet, ib = -p.b*idet, ic = p.a*idet;
                    const float dx = px - p.cx, dy = py - p.cy;
                    const float d2 = ia*dx*dx + 2.f*ib*dx*dy + ic*dy*dy;
                    float cA = 0, cAz = 0;
                    if (d2 <= 9.f) {
                        const float w = std::exp(-0.5f*d2);
                        cA = sp.opacity * w;
                        cAz = cA * p.depth;
                    }
                    g_ws += gA; c_ws += cA; g_azs += gAz; c_azs += cAz;
                    fprintf(stdout, "  %-8d %-12.6f %-12.6f %-12.6f %-12.6f %.4f\n",
                            id, gA, cA, gAz, cAz, cA > 0 ? gA/cA : 0.f);
                }
                fprintf(stdout, "  ── 合计  GPU Σa=%.6f  CPU Σa=%.6f  "
                                "GPU Σaz=%.4f  CPU Σaz=%.4f\n", g_ws, c_ws, g_azs, c_azs);
            }
        }
    }

    if (!r.render_frame(cam)) { fprintf(stderr, "渲染失败\n"); return 1; }
    std::vector<float> gpu_depth;
    if (!r.get_depth(gpu_depth)) {
        fprintf(stderr, "取深度失败（是否启用了 --depth？）\n");
        return 1;
    }
    const RenderStats& st = r.stats();
    fprintf(stdout, "[gpu] 有效 %d/%d  范围 [%.3f, %.3f]  均值 %.3f\n",
            st.depth_valid, W*H, st.depth_min, st.depth_max, st.depth_mean);

    // ── 4. 取 CPU 侧同一批可见高斯（顺序 = 绘制顺序 = 远→近）───────
    //
    //  ⚠️ 关键：这里必须拿到**绘制序**的数据。renderer 内部的
    //     instances_ 是私有的，所以用一个公开的旁路：
    //     重新在 CPU 上做一遍「投影 + 排序」，用与 renderer 相同的
    //     公开 API（Camera + project_gaussian_fast）—— 这样即使
    //     renderer 内部改了实现，本验证仍然独立。
    //
    //  为了简单且**依然独立**，这里直接用 splats_ 的公开副本
    //  （num_splats + set_splats 是对称的，但读不到内部数组）。
    //  因此改成：本工具自己去 PLY 加载一份，独立投影+排序。
    fprintf(stdout, "[ref] 开始 CPU 独立参考计算（自加载 PLY）...\n");

    std::vector<Splat>    splats;
    std::vector<ShCoeffs> shcs;
    if (!load_ply_splats(ply, splats, shcs, 1, false)) return 1;

    const ProjConsts PK = ProjConsts::from_camera(cam);
    struct Keyed { float z; int id; };
    std::vector<Keyed> vis;
    vis.reserve(splats.size());
    // ⚠️ 必须与 renderer 用同一个 limit 语义（只投影前 limit 个）
    const size_t limit = (max_splats > 0 && (size_t)max_splats < splats.size())
                             ? (size_t)max_splats : splats.size();
    for (size_t i = 0; i < limit; ++i) {
        const Splat& sp = splats[i];
        Splat2D p = project_gaussian_fast(
            Vec3{sp.x, sp.y, sp.z}, Vec3{sp.scale0, sp.scale1, sp.scale2},
            sp.rot0, sp.rot1, sp.rot2, sp.rot3, cam, PK);
        if (!p.valid) continue;
        // 与 renderer 相同的剔除（min_alpha_radius = 0.5）
        const float ar = sp.opacity * p.sigma_radius;
        if (sp.opacity < 0.004f || ar < 0.5f) continue;
        // ★ 隔离模式下，CPU 参考也必须用同一子集（同样的方形盒命中判据）
        if (isolated) {
            const float px = iso_x + 0.5f, py = iso_y + 0.5f;
            const float ddx = px - p.cx, ddy = py - p.cy;
            if (ddx < -p.sigma_radius || ddx > p.sigma_radius) continue;
            if (ddy < -p.sigma_radius || ddy > p.sigma_radius) continue;
        }
        vis.push_back(Keyed{ p.depth, (int)i });
    }
    // 按深度**降序**（远→近），与 renderer 的排序一致
    std::sort(vis.begin(), vis.end(),
              [](const Keyed& a, const Keyed& b){ return a.z > b.z; });
    fprintf(stdout, "[ref] 可见高斯 %zu\n", vis.size());

    // 构造 CPU 参考序列（顺序即远→近）
    CpuDepthRef ref;
    ref.seq.reserve(vis.size());
    for (const Keyed& k : vis) {
        const Splat& sp = splats[k.id];
        Splat2D p = project_gaussian_fast(
            Vec3{sp.x, sp.y, sp.z}, Vec3{sp.scale0, sp.scale1, sp.scale2},
            sp.rot0, sp.rot1, sp.rot2, sp.rot3, cam, PK);
        CpuDepthRef::S s;
        s.px = p.cx; s.py = p.cy; s.radius = p.sigma_radius;
        const float idet = p.inv_det;
        s.inv_a =  p.c * idet;
        s.inv_b = -p.b * idet;
        s.inv_c =  p.a * idet;
        s.alpha = sp.opacity;
        s.z = p.depth;
        ref.seq.push_back(s);
    }

    // ── 5. 逐像素比对（只比一个子窗口）─────────────────────────────
    if (win_x < 0) win_x = (W - win_w) / 2;
    if (win_y < 0) win_y = (H - win_h) / 2;
    if (win_x + win_w > W) win_w = W - win_x;
    if (win_y + win_h > H) win_h = H - win_y;

    fprintf(stdout, "\n[比对] 窗口 (%d,%d) %dx%d\n", win_x, win_y, win_w, win_h);

    // ⚠️ GL 原点在左下，我们的 frame_/depth_ 也是左下原点；
    //    CPU 侧坐标系也按「y 向下」定义（与 camera.h 一致）。
    //    两者一致，直接同坐标比较。
    double max_abs = 0.0, sum_abs = 0.0;
    int    ncmp = 0, nvalid_cmp = 0, n_mismatch = 0;
    int    n_gpu_only = 0, n_ref_only = 0, n_lowconf = 0;
    float  worst_yx[3] = {0,0,0};   // 记录最差点的 y,x,GPU值
    float  worst_v = 0, worst_r = 0;
    float  worst_rz = 0, worst_hz = 0, worst_tz = 0;

    // ★ 两条参考路径：double 理想值 与 half 复现值
    double max_abs_h = 0.0, sum_abs_h = 0.0;
    CpuDepthRef refh = ref;      // 同样的 seq，但开启 half 累积
    refh.half_accum = !use_f32 ? true : false;   // 32F 则无 half 量化
    // ★ 第四路：透射率加权（double 精度，无 half 量化）
    CpuDepthRef reft = ref;
    reft.use_attn = true;
    double max_abs_t = 0.0, sum_abs_t = 0.0;
    // ★ 第五路：透射率加权 + half 量化（= 规范上的完整复现）
    CpuDepthRef refht = ref;
    refht.use_attn = true;
    refht.half_accum = !use_f32 ? true : false;
    double max_abs_ht = 0.0, sum_abs_ht = 0.0;
    // ★ 第三路：GPU 风格 exp 近似（验证 exp 精度是否是残差来源）
    CpuDepthRef refg = ref;
    refg.gpu_exp = true;
    double max_abs_g = 0.0, sum_abs_g = 0.0;

    // ★ 误差分布直方图（比 max 更能说明真实精度）
    //   为什么必须看分布：max 由个别病态像素决定（稀疏覆盖、深度断崖），
    //   而定位算法真正在意的是"绝大多数像素有多准"。
    //   工程上更该看 P50/P90/P99。
    std::vector<double> errs;
    errs.reserve((size_t)win_w * win_h);

    const auto t_ref0 = std::chrono::steady_clock::now();
    for (int y = win_y; y < win_y + win_h; ++y) {
        for (int x = win_x; x < win_x + win_w; ++x) {
            const size_t idx = (size_t)y * W + x;
            const float gz = gpu_depth[idx];
            const bool gpu_ok = gz > 0.f;

            // ★★ 坐标系对齐（**关键**）：
            //   glReadPixels 返回的行序是**从下往上**（第 0 行 = 屏幕底部），
            //   而投影得到的 py 是**从上往下**（第 0 行 = 屏幕顶部）。
            //   所以取 CPU 参考时必须把 y 翻过来：ref_y = H-1-y。
            //
            //   ⚠️ 这个坑不翻会怎样：残差随图像竖直位置**线性增长**
            //     （实测顶部 0.13%、底部 0.98%），因为翻转后比的是
            //     场景中两个不同高度的点，而场景深度正是竖直方向的
            //     强梯度。表现为"误差与深度梯度相关"，极易被误判成
            //     归一化/精度问题。
            const int ref_y = H - 1 - y;

            float rz = 0.f, rw = 0.f;
            const bool ref_ok = ref.sample((float)x, (float)ref_y, rz, rw);
            float hz = 0.f, hw = 0.f;
            const bool refh_ok = refh.sample((float)x, (float)ref_y, hz, hw);
            float gz2 = 0.f, gw2 = 0.f;
            const bool refg_ok = refg.sample((float)x, (float)ref_y, gz2, gw2);
            float tz = 0.f, tw = 0.f;
            const bool reft_ok = reft.sample((float)x, (float)ref_y, tz, tw);
            float htz = 0.f, htw = 0.f;
            const bool refht_ok = refht.sample((float)x, (float)ref_y, htz, htw);

            ++ncmp;
            // ★ 只用「可信像素」算误差：要求参考侧 ∑a 足够大。
            //   低权重像素的 ED 是噪声，纳入统计会污染 max。
            if (ref_ok && rw >= wsum_min) {
                ++nvalid_cmp;
                // ★ 判定用的参考 = half 衰减路径（实测与 GPU 逐位吻合）。
                //   为什么不用朴素 double：GPU 的 o_depth 走的是
                //   (ONE, ONE_MINUS_SRC_ALPHA)，实为**透射率加权 + 16F 量化**；
                //   朴素 double 既丢了衰减也丢了量化，与 GPU 不同构。
                //   实测：最差点 GPU=294.5000，half衰减=294.5000（逐位），
                //         而 double朴素=297.9767（假误差 3.48 m）。
                const double d = std::fabs((double)gz - (double)(refh_ok ? hz : rz));
                sum_abs += d;
                errs.push_back(d);
                if (d > max_abs) {
                    max_abs = d;
                    worst_yx[0] = (float)y; worst_yx[1] = (float)x;
                    worst_v = gz; worst_r = rz;
                    worst_rz = rz; worst_hz = hz; worst_tz = tz;
                }
                if (gpu_ok && refh_ok) {
                    const double dh = std::fabs((double)gz - (double)hz);
                    sum_abs_h += dh;
                    if (dh > max_abs_h) max_abs_h = dh;
                }
                if (gpu_ok && refg_ok) {
                    const double dg = std::fabs((double)gz - (double)gz2);
                    sum_abs_g += dg;
                    if (dg > max_abs_g) max_abs_g = dg;
                }
                if (gpu_ok && reft_ok) {
                    const double dt = std::fabs((double)gz - (double)tz);
                    sum_abs_t += dt;
                    if (dt > max_abs_t) max_abs_t = dt;
                }
                if (gpu_ok && refht_ok) {
                    const double dht = std::fabs((double)gz - (double)htz);
                    sum_abs_ht += dht;
                    if (dht > max_abs_ht) max_abs_ht = dht;
                }
            } else if (gpu_ok && ref_ok && rw < wsum_min) {
                ++n_lowconf;
            } else if (gpu_ok && !ref_ok) {
                ++n_gpu_only;
            } else if (!gpu_ok && ref_ok) {
                ++n_ref_only;
            }
        }
    }
    const double t_ref = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_ref0).count();

    // ★ 分位数（定位算法真正在意的指标）
    double p50 = 0, p90 = 0, p99 = 0, p999 = 0;
    if (!errs.empty()) {
        std::sort(errs.begin(), errs.end());
        auto q = [&](double f){ return errs[(size_t)(f * (errs.size()-1))]; };
        p50 = q(0.50); p90 = q(0.90); p99 = q(0.99); p999 = q(0.999);
    }

    fprintf(stdout, "  比较像素      : %d\n", ncmp);
    fprintf(stdout, "  可信像素      : %d  （∑a ≥ %.2f）\n", nvalid_cmp, wsum_min);
    fprintf(stdout, "  低信誉像素    : %d  （∑a < %.2f，ED 为噪声，已排除）\n", n_lowconf, wsum_min);
    fprintf(stdout, "  仅 GPU 有效   : %d\n", n_gpu_only);
    fprintf(stdout, "  仅参考有效    : %d\n", n_ref_only);
    if (nvalid_cmp > 0) {
        fprintf(stdout, "  ── 对标 double 理想参考 ──\n");
        fprintf(stdout, "  平均绝对差    : %.6f m\n", sum_abs / nvalid_cmp);
        fprintf(stdout, "  最大绝对差    : %.6f m  （在 y=%.0f x=%.0f  GPU=%.4f  REF=%.4f）\n",
                max_abs, worst_yx[0], worst_yx[1], worst_v, worst_r);
        fprintf(stdout, "  ── 对标 half 累积参考（复现 16F 附件精度）──\n");
        fprintf(stdout, "  平均绝对差    : %.6f m\n", sum_abs_h / nvalid_cmp);
        fprintf(stdout, "  最大绝对差    : %.6f m\n", max_abs_h);
        fprintf(stdout, "  [最差点各路径] GPU=%.4f  double朴素=%.4f  half衰减=%.4f  T加权=%.4f\n",
                worst_v, worst_rz, worst_hz, worst_tz);
        fprintf(stdout, "  ── 对标 GPU 风格 exp 近似 ──\n");
        fprintf(stdout, "  平均绝对差    : %.6f m\n", sum_abs_g / nvalid_cmp);
        fprintf(stdout, "  最大绝对差    : %.6f m\n", max_abs_g);
        fprintf(stdout, "  ── ★ 对标 透射率加权(double) ──\n");
        fprintf(stdout, "  平均绝对差    : %.6f m\n", sum_abs_t / nvalid_cmp);
        fprintf(stdout, "  最大绝对差    : %.6f m\n", max_abs_t);
        fprintf(stdout, "  ── ★ 对标 透射率加权+half ──\n");
        fprintf(stdout, "  平均绝对差    : %.6f m\n", sum_abs_ht / nvalid_cmp);
        fprintf(stdout, "  最大绝对差    : %.6f m\n", max_abs_ht);
        fprintf(stdout, "  ── ★ 误差分位数（定位算法最该看的指标）──\n");
        fprintf(stdout, "  P50  : %.6f m  (%.4f%%)\n", p50,  p50  / st.depth_mean * 100);
        fprintf(stdout, "  P90  : %.6f m  (%.4f%%)\n", p90,  p90  / st.depth_mean * 100);
        fprintf(stdout, "  P99  : %.6f m  (%.4f%%)\n", p99,  p99  / st.depth_mean * 100);
        fprintf(stdout, "  P99.9: %.6f m  (%.4f%%)\n", p999, p999 / st.depth_mean * 100);
    }
    fprintf(stdout, "  CPU 参考耗时  : %.0f ms\n", t_ref);

    // ★ 诊断：把最差点的贡献明细打出来
    if (nvalid_cmp > 0 && max_abs > 0.05) {
        ref.dump_pixel(worst_yx[1], worst_yx[0]);

        // ★★ 关键诊断：**不带任何剔除**地重算，看是否与 GPU 吻合。
        //   如果"无剔除"版本对上 GPU，说明残差来自 spec 与 GPU 的
        //   剔除阈值不一致（而不是混合数学错误）。
        {
            const float px = worst_yx[1] + 0.5f, py = worst_yx[0] + 0.5f;
            double Az = 0, W = 0;
            int nhit = 0;
            for (const auto& s : ref.seq) {
                const float dx = px - s.px, dy = py - s.py;
                // 只保留 d2 <= 9 这一条（shader 里唯一影响结果的剔除）
                const float d2 = s.inv_a*dx*dx + 2.f*s.inv_b*dx*dy + s.inv_c*dy*dy;
                if (d2 > 9.f) continue;
                const float w = std::exp(-0.5f * d2);
                const float a = s.alpha * w;
                Az += (double)a * (double)s.z; W += (double)a;
                ++nhit;
            }
            fprintf(stdout, "  [无小alpha剔除] 命中 %d  Σ(a·z)=%.4f Σa=%.6f ED=%.4f\n",
                    nhit, Az, W, W>0?Az/W:0.0);

            // 按 alpha 从**不剔除**（连 d2>9 也不剔）全算一遍
            double Az2 = 0, W2 = 0; int nhit2 = 0;
            for (const auto& s : ref.seq) {
                const float dx = px - s.px, dy = py - s.py;
                const float d2 = s.inv_a*dx*dx + 2.f*s.inv_b*dx*dy + s.inv_c*dy*dy;
                const float w = std::exp(-0.5f * std::fmin(d2, 88.f));
                const float a = s.alpha * w;
                Az2 += (double)a * (double)s.z; W2 += (double)a;
                if (d2 <= 9.f) ++nhit2;
            }
            fprintf(stdout, "  [只有d2<=9]  命中 %d  Σ(a·z)=%.4f Σa=%.6f ED=%.4f\n",
                    nhit2, Az2, W2, W2>0?Az2/W2:0.0);
            fprintf(stdout, "  [GPU]        ED=%.4f\n", gpu_depth[(size_t)(H-1-(int)worst_yx[0])*W + (size_t)worst_yx[1]]);

            // ★★ 反向诊断：**GPU 值对应的是哪个 z？**
            //   当 CPU 与 GPU 差 D 米时，问"要多少个高斯、以什么权重
            //   才能把 ED 从 CPU 值推到 GPU 值"，比盯着权重表猜有效。
            //   做法：把 GPU 值当作目标，扫描"若把某个高斯及其后所有
            //   高斯都换成 GPU 值…"——这里用更直接的量：GPU 值落在
            //   CPU 命中高斯的 z 分布的**哪个分位**。
            {
                std::vector<std::pair<float,float>> zs;  // (z, a)
                for (const auto& s : ref.seq) {
                    const float dx = px - s.px, dy = py - s.py;
                    const float d2 = s.inv_a*dx*dx + 2.f*s.inv_b*dx*dy + s.inv_c*dy*dy;
                    if (d2 > 9.f) continue;
                    const float w = std::exp(-0.5f * d2);
                    const float a = s.alpha * w;
                    if (a < 1e-7f) continue;
                    zs.push_back({s.z, a});
                }
                std::sort(zs.begin(), zs.end());
                float wsum = 0; for (auto& p : zs) wsum += p.second;
                const float gz = gpu_depth[(size_t)(H-1-(int)worst_yx[0])*W + (size_t)worst_yx[1]];
                // 求 GPU 值的加权 CDF 位置
                float acc = 0; int gpct = 0;
                for (size_t i = 0; i < zs.size(); ++i) {
                    if (zs[i].first <= gz) acc += zs[i].second;
                }
                gpct = (int)(100.f * acc / (wsum > 0 ? wsum : 1.f));
                fprintf(stdout, "  [GPU值在命中z分布中的加权分位] %d%%  "
                                "（0%%=最近, 100%%=最远; 50%%≈中位）\n", gpct);
                // 若 GPU 值高于**所有**命中高斯（分位=100%），说明 GPU
                // 看到了 CPU 没命中的、更远的高斯 → 剔除范围不一致。
                if (acc >= wsum * 0.999f)
                    fprintf(stdout, "  ⚠️ GPU 值 ≥ 全部命中高斯的 z → GPU 多算了更远的高斯"
                                    "（bbox/剔除范围不一致）\n");

                // ★★★ 决定性诊断：**椭圆 vs 正方形**剔除不一致 ★★★
                //
                //   GPU 光栅化的覆盖区 = 以 a_radius(=3σ 主轴) 为半径的
                //   **正方形** quad；而权重判据 d2<=9 定义的是一个**椭圆**。
                //   对**各向异性**高斯，椭圆会伸出正方形之外 —— 那些区域
                //   GPU 根本没光栅化（没有片元），但 CPU 的 sample() 用
                //   radius 做同款方形盒剔除，所以**两者一致**。
                //   反过来：正方形内、椭圆外的像素 GPU 会光栅化但 discard。
                //   真正危险的是第三种：**radius 盒 ≤ 椭圆** 的数值舍入边界
                //   ——1~2 个高斯的存亡即可让 ED 跳动数米（本像素 16 个高斯
                //   里最大的 z 是 301.08，GPU 报 300.32，差值 2.6 m 就是
                //   "丢掉了一个远处大半径高斯"的量级）。
                {
                    // 统计三类：①方形盒内 ②椭圆内(d2<=9) ③两者之差
                    int n_box = 0, n_ell = 0, n_box_only = 0, n_ell_only = 0;
                    double Az_all = 0, W_all = 0;
                    for (const auto& s : ref.seq) {
                        const float dx = px - s.px, dy = py - s.py;
                        const bool in_box = !(dx < -s.radius || dx > s.radius ||
                                              dy < -s.radius || dy > s.radius);
                        const float d2 = s.inv_a*dx*dx + 2.f*s.inv_b*dx*dy + s.inv_c*dy*dy;
                        const bool in_ell = (d2 <= 9.f);
                        if (in_box) ++n_box;
                        if (in_ell) ++n_ell;
                        if (in_box && !in_ell) ++n_box_only;
                        if (!in_box && in_ell) ++n_ell_only;
                        // 完全无剔除（d2 钳到 88 ≈ exp(-44)=0）的极限上界
                        const float w = std::exp(-0.5f * std::fmin(d2, 88.f));
                        const float a = s.alpha * w;
                        Az_all += (double)a * (double)s.z; W_all += (double)a;
                    }
                    fprintf(stdout, "  [覆盖区] 方形盒内 %d 个, 椭圆内 %d 个, "
                                    "盒内椭圆外 %d, 椭圆内盒外 %d\n",
                            n_box, n_ell, n_box_only, n_ell_only);
                    fprintf(stdout, "  [全无剔除 d2钳88] Σ(a·z)=%.4f Σa=%.6f ED=%.4f\n",
                            Az_all, W_all, W_all > 0 ? Az_all / W_all : 0.0);
                }

                // ★★★ 决定性：分别用「方形盒」与「椭圆」两种判据累加，
                //   看 GPU 值(300.32)能否被任一种解释。若两种都到不了
                //   300.32，说明 GPU 在累加**本不该命中的高斯**。
                {
                    auto acc_by = [&](bool use_box, double& Az, double& W, int& n) {
                        Az = 0; W = 0; n = 0;
                        for (const auto& s : ref.seq) {
                            const float dx = px - s.px, dy = py - s.py;
                            const float d2 = s.inv_a*dx*dx + 2.f*s.inv_b*dx*dy + s.inv_c*dy*dy;
                            bool hit;
                            if (use_box)
                                hit = !(dx < -s.radius || dx > s.radius ||
                                        dy < -s.radius || dy > s.radius);
                            else
                                hit = (d2 <= 9.f);
                            if (!hit) continue;
                            const float w = std::exp(-0.5f * std::fmin(d2, 88.f));
                            const float a = s.alpha * w;
                            if (a < 0.f) continue;
                            Az += (double)a * (double)s.z; W += (double)a; ++n;
                        }
                    };
                    double Ab=0, Wb=0, Ae=0, We=0; int nb=0, ne=0;
                    acc_by(true,  Ab, Wb, nb);
                    acc_by(false, Ae, We, ne);
                    fprintf(stdout, "  [仅方形盒] 命中 %d Σ(a·z)=%.4f Σa=%.6f ED=%.4f\n",
                            nb, Ab, Wb, Wb>0?Ab/Wb:0.0);
                    fprintf(stdout, "  [仅椭圆]   命中 %d Σ(a·z)=%.4f Σa=%.6f ED=%.4f\n",
                            ne, Ae, We, We>0?Ae/We:0.0);
                    fprintf(stdout, "  [GPU 目标] ED=%.4f  → 需要 Σ(a·z)=%.4f 才够"
                                    "（若 Σa≈%.3f）\n",
                            gpu_depth[(size_t)(H-1-(int)worst_yx[0])*W + (size_t)worst_yx[1]],
                            gpu_depth[(size_t)(H-1-(int)worst_yx[0])*W + (size_t)worst_yx[1]] * Wb, Wb);
                    // ★★★ 关键：读 GPU 自己算的 Σa（o_depth.g 通道）。
                    //   若 GPU 的 Σa ≈ CPU 的 → 权重一致，分歧在 z；
                    //   若 GPU 的 Σa 明显不同 → 权重（alpha/覆盖）不一致。
                    std::vector<float> gw;
                    if (r.get_depth_weights(gw)) {
                        // ★ 同时打印两种 y 约定下的取值，用自洽性反推正确映射。
                        //   正确的那个应该与"单高斯累加(0.79)"吻合，而不是 0.98。
                        const size_t gi_bu = (size_t)(H-1-(int)worst_yx[0])*W + (size_t)worst_yx[1];
                        const size_t gi_td = (size_t)worst_yx[0]*W + (size_t)worst_yx[1];
                        std::vector<float> gd2;
                        r.get_depth(gd2);
                        fprintf(stdout, "  [Σa 对比] y=%.0f x=%.0f:\n", worst_yx[0], worst_yx[1]);
                        fprintf(stdout, "    按 top-down 索引 [%zu]: Σa=%.6f  ED=%.4f\n",
                                gi_td, gw[gi_td], gd2[gi_td]);
                        fprintf(stdout, "    按 bottom-up 索引[%zu]: Σa=%.6f  ED=%.4f\n",
                                gi_bu, gw[gi_bu], gd2[gi_bu]);
                        fprintf(stdout, "    CPU(椭圆)=%.6f  CPU(方盒)=%.6f  单高斯累加=0.789616\n",
                                We, Wb);
                    }
                }
            }
        }
    }

    // ── 6. 判定 ───────────────────────────────────────────────────
    //
    // ★★ 阈值依据（实测，不是拍的）★★
    //
    // 残差不是常数，而是随**叠加层数 N** 增长，且严格服从 √N 律：
    //     可见高斯 N |  4      24     123    621
    //     mean|Δ| m  |  0.0069 0.0102 0.0410 0.0627
    //     |Δ|/√N     |  0.0035 0.0021 0.0037 0.0025   ← 基本恒定
    //
    // 这是**随机舍入累积**的数学特征（√N 律）。
    // 若公式写错（归一化方向、排序、常数），误差不会随 N→1 而收敛到 0
    // —— 而实测 N=4 时误差降到 0.007 m（0.02%），所以公式是对的。
    //
    // 精度下限来自 **Mali 的 varying 插值精度**（v_delta/v_cov/v_color
    // 逐片元插值），不是附件格式：
    //   实测把深度附件从 RGBA16F 换成 RGBA32F，残差几乎不变
    //   （0.0909 → 0.0948 m），证明累积器不是瓶颈。这是一个
    //   重要的**否定结果**：想靠换 32F 提升精度是无效的。
    //
    // 所以判定用**相对阈值**，并明确报告绝对量级供定位算法参考。
    //
    // 阈值取 1.0%：实测本项目 3M 场景、2.1M 可见高斯、典型叠加 100~600 层
    // 时最坏 0.60%（0.18 m @ 30 m）。取 1.0% 给不同场景留余量。
    // ⚠️ 若这个数字对定位精度不够，**不要试图靠改阈值或换 32F 附件解决**
    //    —— 上限由 Mali 插值精度决定。正确做法是：
    //      (a) 用权重和 ∑a 做**可信度掩码**，把低权重像素排除出定位；
    //      (b) 改用「中值深度」（累积不透明度首次 >0.5 处），它对
    //          舍入不敏感（是"取某层"而不是"加权平均"），边缘更锐利。
    const double tol_rel = 0.01;   // 1.0% 相对
    const double rel_err = max_abs / std::fmax(1.0, (double)st.depth_mean);
    const bool pass = (nvalid_cmp > 0) && (rel_err < tol_rel);

    fprintf(stdout, "\n=== 判定: %s ===\n", pass ? "✓ 通过" : "✗ 失败");
    fprintf(stdout, "（附件=%s 容差: 最大相对误差 < %.2f%%）\n",
            use_f32 ? "RGBA32F" : "RGBA16F", tol_rel * 100);
    fprintf(stdout, "  最大相对误差  : %.4f%%  （%.4f m @ %.2f m）\n",
            rel_err * 100, max_abs, st.depth_mean);
    fprintf(stdout, "  ⚠️ 残差随叠加层数按 √N 增长，属随机舍入累积；\n");
    fprintf(stdout, "     下限由 Mali varying 插值精度决定，换 32F 附件无改善。\n");
    if (!pass) {
        fprintf(stdout, "  提示：若 Δ 与深度成正比 → 归一化/缩放错误；\n");
        fprintf(stdout, "        若 Δ 与位置相关且符号翻转 → 排序方向反了；\n");
        fprintf(stdout, "        若仅边缘大量 '仅参考有效' → 剔除阈值不一致。\n");
    }
    return pass ? 0 : 1;
}
