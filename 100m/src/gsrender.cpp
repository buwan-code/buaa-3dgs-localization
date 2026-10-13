// gsrender.cpp — 3DGS 板端渲染器主程序（M2 可运行版本）
//
// 用法：
//   ./build/gsrender <ply> [选项]
// 选项：
//   --w N --h N          分辨率（默认 640×480）
//   --frames N           渲染多少帧（默认 1）
//   --orbit DEG          相机绕 Y 轴旋转（生成多视角参考视图）
//   --dist D             相机到原点的距离（默认自动）
//   --fovy DEG           垂直视场角（默认 45）
//   --out FILE.ppm       输出图像（多帧时自动加序号）
//   --max N              限制最多画多少个高斯（0=全部）
//   --no-sort            关掉排序（测性能上限）
//   --sh0                只用 DC（关闭 1 阶 SH）
//   --f16                用 RGBA16F 混合（会走 Blit 到 RGBA8 回读）
//   --no-fan             不控风扇
//   --stats              每帧打印详细耗时
//   --auto-cam           用 PLY 的包围盒自动摆相机
#include "renderer.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>

using namespace gsplat;

struct Args {
    const char* ply = nullptr;
    int   w = 640, h = 480;
    int   frames = 1;
    float orbit_deg = 0.f;
    float dist = 0.f;
    float fovy = 45.f;
    const char* out = nullptr;
    int   max_splats = 0;
    int   threads = 0;          // 0 = 自动（核数/2）
    bool  sort = true;
    int   sh_degree = 1;
    bool  f16 = false;
    bool  auto_fan = true;
    bool  stats = false;
    bool  auto_cam = true;
    bool  verbose = false;
    bool  depth = false;        // ★ 输出深度（MRT）
    bool  depth_dump = false;   // ★ 写 PGM 深度图
    bool  depth_f32 = false;    // ★ 深度附件用 RGBA32F（全精度）
    float min_alpha_radius = 0.5f;  // 重要性剔除阈值（alpha × 屏幕半径）
    bool  block_cull = true;        // ★ 空间分块视锥剔除
    int   block_size = 2048;        // ★ 每块高斯数
    bool  cheap_cull = true;        // ★ 完整投影前的廉价粗筛
    bool  async = false;            // ★ PBO 异步回读（结果滞后一帧）
    int   lod_keep = 0;             // ★ LOD：只保留重要度最高的 N 个高斯（0=全部）
};

// ── 从 PLY 估包围盒（流式，省内存）─────────────────────────────────
static void estimate_bounds(const char* path, Vec3& center, float& radius) {
    std::vector<PosAlpha> pa;
    // 采样 20 万个点足够估包围盒
    if (!stream_positions(path, pa, 200000, false) || pa.empty()) {
        center = Vec3{0,0,0}; radius = 10.f; return;
    }
    float mnx=1e30f,mny=1e30f,mnz=1e30f, mxx=-1e30f,mxy=-1e30f,mxz=-1e30f;
    for (const auto& p : pa) {
        if (p.x < mnx) mnx = p.x;  if (p.x > mxx) mxx = p.x;
        if (p.y < mny) mny = p.y;  if (p.y > mxy) mxy = p.y;
        if (p.z < mnz) mnz = p.z;  if (p.z > mxz) mxz = p.z;
    }
    center = Vec3{ (mnx+mxx)*0.5f, (mny+mxy)*0.5f, (mnz+mxz)*0.5f };
    const float dx = mxx-mnx, dy = mxy-mny, dz = mxz-mnz;
    radius = 0.5f * std::sqrt(dx*dx + dy*dy + dz*dz);
    fprintf(stdout, "[cam] 包围盒中心=(%.2f, %.2f, %.2f)  半径=%.2f\n",
            center.x, center.y, center.z, radius);
}

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--w")      && i+1<argc) a.w = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--h")      && i+1<argc) a.h = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--frames") && i+1<argc) a.frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--orbit")  && i+1<argc) a.orbit_deg = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--dist")   && i+1<argc) a.dist = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--fovy")   && i+1<argc) a.fovy = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--out")    && i+1<argc) a.out = argv[++i];
        else if (!strcmp(argv[i], "--max")    && i+1<argc) a.max_splats = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads")&& i+1<argc) a.threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-sort"))  a.sort = false;
        else if (!strcmp(argv[i], "--sh0"))      a.sh_degree = 0;
        else if (!strcmp(argv[i], "--f16"))      a.f16 = true;
        else if (!strcmp(argv[i], "--no-fan"))   a.auto_fan = false;
        else if (!strcmp(argv[i], "--stats"))    a.stats = true;
        else if (!strcmp(argv[i], "--no-auto-cam")) a.auto_cam = false;
        else if (!strcmp(argv[i], "--depth"))    a.depth = true;
        else if (!strcmp(argv[i], "--depth-f32")) { a.depth = true; a.depth_f32 = true; }
        else if (!strcmp(argv[i], "--depth-dump")) { a.depth = true; a.depth_dump = true; }
        else if (!strcmp(argv[i], "--min-alpha-radius") && i+1<argc)
            a.min_alpha_radius = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--no-block-cull")) a.block_cull = false;
        else if (!strcmp(argv[i], "--no-cheap-cull")) a.cheap_cull = false;
        else if (!strcmp(argv[i], "--async"))         a.async = true;
        else if (!strcmp(argv[i], "--block-size") && i+1<argc) a.block_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lod") && i+1<argc) a.lod_keep = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v"))         a.verbose = true;
        else if (!a.ply) a.ply = argv[i];
    }
    if (!a.ply) {
        fprintf(stderr, "用法: %s <ply> [--w N --h N --frames N --orbit DEG "
                        "--out F.ppm ...]\n", argv[0]);
        return 2;
    }

    fprintf(stdout, "=====================================================\n");
    fprintf(stdout, " 3DGS 板端渲染器  %d×%d\n", a.w, a.h);
    fprintf(stdout, "=====================================================\n");

    RenderConfig cfg;
    cfg.width = a.w; cfg.height = a.h;
    cfg.max_splats = a.max_splats;
    cfg.project_threads = a.threads;
    cfg.min_alpha_radius = a.min_alpha_radius;
    cfg.block_cull  = a.block_cull;
    cfg.block_size  = a.block_size;
    cfg.cheap_cull  = a.cheap_cull;
    cfg.async_readback = a.async;
    cfg.lod_keep = a.lod_keep;
    cfg.sh_degree = a.sh_degree;
    cfg.do_sort = a.sort;
    cfg.f16_color = a.f16;
    cfg.auto_fan = a.auto_fan;
    cfg.depth = a.depth;
    cfg.depth_f32 = a.depth_f32;
    cfg.verbose = a.verbose;

    Renderer r;
    const auto t_init0 = std::chrono::steady_clock::now();
    if (!r.init(a.ply, cfg)) {
        fprintf(stderr, "初始化失败\n");
        return 1;
    }
    const double t_init = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_init0).count();
    fprintf(stdout, "[init] 总耗时 %.0f ms（含 PLY 加载 + GL 初始化）\n", t_init);

    // 相机：默认从包围盒外斜上方看
    Vec3 center{0,0,0}; float radius = 10.f;
    estimate_bounds(a.ply, center, radius);
    const float dist = (a.dist > 0.f) ? a.dist : radius * 2.2f;

    const double frame_ms_accum[] = {0};
    (void)frame_ms_accum;

    double total_draw = 0, total_proj = 0, total_sort = 0;
    double total_rb = 0, total_up = 0, total_asm = 0;

    for (int f = 0; f < a.frames; ++f) {
        // 轨道角度：多帧时在 [0, orbit) 均匀分布
        const float ang = (a.frames > 1)
            ? (a.orbit_deg * 3.14159265f / 180.f) * ((float)f / (a.frames - 1))
            : (a.orbit_deg * 3.14159265f / 180.f);

        // 相机绕 Y 轴转，始终看向中心
        const float ex = center.x + dist * std::sin(ang);
        const float ey = center.y - dist * 0.35f;      // 略高
        const float ez = center.z + dist * std::cos(ang);

        Camera cam = Camera::look_at(
            Vec3{ex, ey, ez}, center, Vec3{0.f, -1.f, 0.f},
            a.w, a.h, a.fovy);
        // zfar 按场景尺度设，避免远处被裁
        cam.zfar  = dist + radius * 4.f;
        cam.znear = std::fmax(0.01f, dist * 0.002f);

        if (!r.render_frame(cam)) {
            fprintf(stderr, "渲染第 %d 帧失败\n", f);
            return 1;
        }

        const RenderStats& st = r.stats();
        total_draw += st.ms_draw; total_proj += st.ms_project;
        total_sort += st.ms_sort; total_rb += st.ms_readback;
        total_up   += st.ms_upload; total_asm += st.ms_assemble;

        if (a.stats || f == 0) {
            fprintf(stdout,
                "[帧 %3d] 可见 %7d/%7d  投影 %6.1f  排序 %5.1f  "
                "组装 %5.1f  上传 %5.1f  绘制 %6.1f  回读 %5.2f  合计 %7.1f ms\n",
                f, st.visible_splats, st.total_splats,
                st.ms_project, st.ms_sort, st.ms_assemble,
                st.ms_upload, st.ms_draw, st.ms_readback, st.ms_total);
            fprintf(stdout, "         SoC %.1f°C  风扇 %d/255\n",
                    st.soc_temp, st.fan_duty);
            if (a.stats) {
                // ★ 回读分解：这段决定了「PBO 异步回读」值不值得做。
                //   等待大 → 是 GPU 同步开销，异步化（延迟一帧）可消除；
                //   实读大 → 是真实数据传输，只能靠减少数据量。
                fprintf(stdout,
                    "         ↳ 回读分解: 等待 %6.2f ms | 纯 glReadPixels %6.2f ms\n",
                    st.ms_wait, st.ms_readonly);
            }
            if (st.depth_valid > 0) {
                fprintf(stdout,
                    "         ★深度 有效 %d/%d (%.1f%%)  范围 [%.2f, %.2f]  均值 %.2f m  "
                    "回读 %.2f ms\n",
                    st.depth_valid, a.w * a.h,
                    100.0 * st.depth_valid / (double)(a.w * a.h),
                    st.depth_min, st.depth_max, st.depth_mean,
                    st.ms_depth_readback);
            } else if (a.depth) {
                fprintf(stdout, "         ★深度 无有效像素（可能场景/相机不匹配）\n");
            }
        }

        if (a.out) {
            char buf[1024];
            if (a.frames > 1) snprintf(buf, sizeof(buf), "%s.%03d.ppm", a.out, f);
            else              snprintf(buf, sizeof(buf), "%s", a.out);
            if (r.save_ppm(buf)) fprintf(stdout, "         → 已保存 %s\n", buf);
            else                 fprintf(stderr, "         ⚠️ 保存 %s 失败\n", buf);
        }
        if (a.depth_dump) {
            char dbuf[1024];
            if (a.frames > 1) snprintf(dbuf, sizeof(dbuf), "%s.%03d_depth.pgm", a.out ? a.out : "frame", f);
            else              snprintf(dbuf, sizeof(dbuf), "%s_depth.pgm", a.out ? a.out : "frame");
            if (r.save_pgm(dbuf)) fprintf(stdout, "         → 已保存深度 %s\n", dbuf);
            else                  fprintf(stderr, "         ⚠️ 保存深度 %s 失败\n", dbuf);
        }
    }

    if (a.frames > 0) {
        const double n = a.frames;
        fprintf(stdout, "\n【平均】投影 %.1f  排序 %.1f  组装 %.1f  上传 %.1f  绘制 %.1f  回读 %.2f\n",
                total_proj/n, total_sort/n, total_asm/n, total_up/n,
                total_draw/n, total_rb/n);
        const double tot = (total_proj+total_sort+total_asm+total_up+total_draw+total_rb)/n;
        fprintf(stdout, "【合计】%.1f ms/帧 → %.2f FPS\n", tot, 1000.0/tot);
    }

    fprintf(stdout, "\n=====================================================\n");
    return 0;
}
