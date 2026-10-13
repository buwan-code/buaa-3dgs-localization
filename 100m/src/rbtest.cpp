// rbtest.cpp — 单独测量回读各段耗时，判定 212 ms 到底花在哪
//
// 背景：同一份 renderer.cpp，`回读`段从早先的 36 ms 变成现在的 212 ms。
// 该段包含三件事：
//   (1) glClientWaitSync(fence, FLUSH)   ← 等 GPU 整帧完成
//   (2) glReadPixels                     ← 真正把像素搬回 CPU
//   (3) 两者之间的驱动开销
// 必须分段测，否则不知道该优化谁。
//
// ⚠️ 计时口径警告：Clock::now() 本身约 40 ns，对 ms 级区间完全够用；
//    但**不要**用它测 ns 级的单点操作（那个坑踩过三次）。
#include "renderer.h"
#include "camera.h"
#include <cstdio>
#include <cstring>
#include <chrono>

using namespace gsplat;
using Clock = std::chrono::steady_clock;
static double ms_since(const Clock::time_point& t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "用法: %s <ply> [--w N --h N --frames N]\n", argv[0]); return 2; }
    int w = 512, h = 288, frames = 5;
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--w") && i+1 < argc) w = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--h") && i+1 < argc) h = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--frames") && i+1 < argc) frames = atoi(argv[++i]);
    }

    RenderConfig cfg;
    cfg.width = w; cfg.height = h;
    cfg.project_threads = 8;
    cfg.sh_degree = 1;

    Renderer r;
    if (!r.init(argv[1], cfg)) { fprintf(stderr, "init 失败\n"); return 1; }

    Camera cam = Camera::look_at(Vec3{30.7f, -1.2f, 14.0f},
                                 Vec3{0.7f, -1.2f, 2.0f},
                                 Vec3{0.f, 1.f, 0.f}, w, h, 45.f);

    printf("=== 逐帧各段耗时（%d×%d，%d 帧）===\n", w, h, frames);
    printf("%-6s %10s %10s %10s %10s %10s\n",
           "帧", "投影", "排序", "组装", "上传", "回读", "等栅栏", "纯读取");
    for (int f = 0; f < frames; ++f) {
        if (!r.render_frame(cam)) { fprintf(stderr, "render 失败 @%d\n", f); return 1; }
        const auto& s = r.stats();
        printf("%-6d %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f\n",
               f, s.ms_project, s.ms_sort, s.ms_assemble, s.ms_upload, s.ms_readback, s.ms_wait, s.ms_readonly);
    }
    return 0;
}
