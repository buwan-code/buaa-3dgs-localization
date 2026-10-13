// raster_bench.cpp — M0 光栅化吞吐上限基线（NOTES.md §5 建议的第一步）
//
// 目的：在做排序/剔除等复杂优化之前，先测出「最简情况」下的 GPU 吞吐上限。
// 做法：用 glDrawArraysInstanced 把 N 个点画成纯色小方块，
//       不排序、不 alpha 混合、不读属性纹理。
//
// 这样能回答三个问题：
//   1. N 实例的顶点处理在 Mali-G610 上到底能不能跑（vsync 前的裸吞吐）
//   2. GPU 提频后的实际增益（对比提频/不提频，见 sh/lock_freq.sh）
//   3. 瓶颈偏顶点侧还是片元侧（改 quad 大小看耗时变化）
//
// ⚠️ 本项目已证明纸面推算不可靠（§2.0 排序实测差估算 10 倍）——
//    所以这个基线是必须先做的，不做就没有任何性能标尺。
//
// ── 修订记录 ────────────────────────────────────────────────────────
// [BUG-001] 早期版本用 VBO 里的 a_pos 当实例位置，但 glDrawArraysInstanced 的
//   (first=0, count=4) 是「每实例顶点范围」，gl_VertexID 恒为 0..3，
//   a_pos 恒取 VBO 前 4 个顶点（黄金角螺旋起点，全在原点附近）→
//   3M 个 quad 全部堆在原点，非空像素仅 64/307200（假「渲染成功」）。
//   修正：位置改由 gl_InstanceID 在顶点着色器内程序化生成，
//         完全不需要 VBO 存位置；仍然保留一个最小 VBO 只为满足
//         某些驱动「必须有已启用属性」的要求。
// ────────────────────────────────────────────────────────────────────
//
// 用法：./raster_bench [--w 640] [--h 480] [--n 3000000] [--quad 4] [--iters 60]
//                      [--rgba8] [--grid]    (--grid: 按屏幕网格均匀铺，便于肉眼确认)
#include "offscreen.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>            // sqrt / cos / sin
#include <chrono>
#include <algorithm>        // std::sort
#include <vector>

using namespace gsplat;
using Clock = std::chrono::steady_clock;

// ── 顶点着色器：位置由 gl_InstanceID 程序化生成，不读 VBO ──────────
// gl_InstanceID 是 GLES 3.0 起的内建变量（整数，从 0 开始）。
// 用黄金角螺旋（sunflower）把 N 个实例铺在一个圆盘上：
//   a = i * 2.399963   (黄金角 ≈ 137.508°)
//   r = R * sqrt(i / N)
// 这个分布的覆盖面均匀，不会出现中心堆叠。
//
// ⚠️ 不要把实例位置写进 VBO 再按 divisor 取——那会把「N*3 float 的
//    带宽读取」算进耗时，而本测试要测的是纯光栅化/顶点吞吐。
//    程序化生成让顶点侧几乎零带宽，是最干净的标尺。
static const char* VS = R"(#version 320 es
uniform vec2  u_res;                       // 图像分辨率
uniform float u_quad;                      // quad 半边长(像素)
uniform float u_extent;                    // 场景半径
uniform int   u_n;                         // 实例总数
uniform int   u_grid;                      // 1 = 网格模式(诊断用)
uniform int   u_cheap;                     // 1 = 廉价VS(剔除 sqrt/cos/sin)
out float v_shade;
const float GOLDEN = 2.39996322972865332;   // 黄金角(弧度)

void main() {
    float fid = float(gl_InstanceID);

    // ① 实例中心（世界坐标，圆盘分布）
    vec2 center;
    if (u_cheap == 1) {
        // 廉价路径：用整数解同余做伪随机散布，完全不碰三角函数/sqrt。
        // 用途：把「顶点着色器算术代价」与「光栅化/带宽代价」分离开。
        int i = gl_InstanceID;
        int cx = (i * 1103515245 + 12345) % 2048;   // LCG
        int cy = (i * 214013 + 2531011) % 2048;
        center = (vec2(float(cx), float(cy)) / 2048.0 * 2.0 - 1.0) * u_extent;
    } else {
        float ang = fid * GOLDEN;
        float rad = u_extent * sqrt((fid + 0.5) / float(u_n));
        center = vec2(rad * cos(ang), rad * sin(ang));
    }

    // 诊断网格模式：把实例均匀铺满屏幕，肉眼一眼能看出「有没有画出来」
    if (u_grid == 1) {
        int cols = int(sqrt(float(u_n) * u_res.x / u_res.y));
        if (cols < 1) cols = 1;
        int cxi = gl_InstanceID % cols;
        int cyi = gl_InstanceID / cols;
        center = vec2(
            (float(cxi) + 0.5) / float(cols) * 2.0 * u_extent - u_extent,
            (float(cyi) + 0.5) / float(cols) * 2.0 * u_extent - u_extent
        );
    }

    // ② quad 的四个角：gl_VertexID 0..3 组成 TRIANGLE_STRIP
    //    顺序: (0,0) (1,0) (0,1) (1,1)
    vec2 corner = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));

    // ③ 世界 → 像素 → clip 空间
    vec2 ndc = center / u_extent;                       // 落在 [-1,1]
    vec2 pix = (ndc * 0.5 + 0.5) * u_res + (corner * 2.0 - 1.0) * u_quad;
    vec2 clip = pix / u_res * 2.0 - 1.0;

    gl_Position = vec4(clip, 0.0, 1.0);
    v_shade = 0.5 + 0.5 * float(gl_VertexID);
}
)";

static const char* FS = R"(#version 320 es
precision mediump float;
in float v_shade;
out vec4 o_color;
void main() {
    o_color = vec4(v_shade * 0.6, v_shade * 0.8, 1.0 - v_shade, 1.0);
}
)";

int main(int argc, char** argv) {
    int W = 640, H = 480, N = 3000000, iters = 60;
    float quad = 4.0f;
    bool f16 = true;
    int  grid = 0;
    int  cheap = 0;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--w")    && i+1<argc) W = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--h")   && i+1<argc) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n")   && i+1<argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")&& i+1<argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quad") && i+1<argc) quad = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--rgba8")) f16 = false;
        else if (!strcmp(argv[i], "--grid"))  grid = 1;
        else if (!strcmp(argv[i], "--cheap")) cheap = 1;
    }

    fprintf(stdout, "=====================================================\n");
    fprintf(stdout, " 光栅化吞吐基线  %d×%d, %d 实例, quad=%.0fpx, %s%s%s\n",
            W, H, N, quad, f16 ? "RGBA16F" : "RGBA8",
            grid ? ", GRID模式" : "", cheap ? ", CHEAP-VS" : "");
    fprintf(stdout, "=====================================================\n");

    OffscreenTarget t;
    if (!create_offscreen(t, W, H, true, f16)) {
        fprintf(stderr, "离屏上下文创建失败（检查 DISPLAY=:0）\n");
        return 1;
    }

    // 只为满足「必须有已启用属性」的驱动要求，给一个 4 顶点占位 VBO。
    // 位置数据不再从这里取（见 BUG-001）。
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    { const float dummy[12] = {0}; glBufferData(GL_ARRAY_BUFFER, sizeof(dummy), dummy, GL_STATIC_DRAW); }
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glBindVertexArray(0);

    GLuint prog;
    {
        static const char* names[] = { "a_pos" };
        static const GLuint locs[]  = { 0 };
        prog = build_program(VS, FS, true, names, locs, 1);
    }
    if (!prog) { fprintf(stderr, "着色器编译/链接失败\n"); return 1; }
    glUseProgram(prog);

    {
        GLint vao_bound = 0, a0_enabled = 0, a0_size = 0, loc = -1;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao_bound);
        glGetVertexAttribiv(0, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &a0_enabled);
        glGetVertexAttribiv(0, GL_VERTEX_ATTRIB_ARRAY_SIZE, &a0_size);
        loc = glGetAttribLocation(prog, "a_pos");
        fprintf(stdout, "[diag] VAO=%d attr0(enabled=%d size=%d) a_pos_loc=%d\n",
                vao_bound, a0_enabled, a0_size, loc);
    }

    GLint u_res    = glGetUniformLocation(prog, "u_res");
    GLint u_quad   = glGetUniformLocation(prog, "u_quad");
    GLint u_extent = glGetUniformLocation(prog, "u_extent");
    GLint u_n      = glGetUniformLocation(prog, "u_n");
    GLint u_grid   = glGetUniformLocation(prog, "u_grid");
    GLint u_cheap  = glGetUniformLocation(prog, "u_cheap");

    glUseProgram(prog);
    glUniform2f(u_res, (float)W, (float)H);
    glUniform1f(u_quad, quad);
    glUniform1f(u_extent, 50.0f);
    glUniform1i(u_n, N);
    glUniform1i(u_grid, grid);
    glUniform1i(u_cheap, cheap);

    // 预热（首次调用含管线编译与纹理懒加载开销）
    for (int i = 0; i < 5; ++i) {
        bind_draw_target(t);
        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glBindVertexArray(vao);
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, N);
    }
    glFinish();

    // 计时
    double best = 1e30, sum = 0, worst = 0;
    std::vector<double> samples;
    samples.reserve(iters);

    for (int i = 0; i < iters; ++i) {
        auto t0 = Clock::now();
        bind_draw_target(t);
        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glBindVertexArray(vao);
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, N);
        glFinish();                          // 必须 finish，否则测的是提交耗时
        double ms = std::chrono::duration<double, std::milli>(Clock::now()-t0).count();
        samples.push_back(ms);
        sum += ms;
        if (ms < best) best = ms;
        if (ms > worst) worst = ms;
    }
    double avg = sum / iters;
    std::sort(samples.begin(), samples.end());
    double p50 = samples[iters/2], p95 = samples[(int)(iters*0.95)];

    fprintf(stdout, "\n【结果】\n");
    fprintf(stdout, "  平均帧时间: %.2f ms  → %.1f FPS\n", avg, 1000.0/avg);
    fprintf(stdout, "  P50       : %.2f ms\n", p50);
    fprintf(stdout, "  P95       : %.2f ms\n", p95);
    fprintf(stdout, "  最好/最坏 : %.2f / %.2f ms\n", best, worst);
    fprintf(stdout, "\n【吞吐】\n");
    fprintf(stdout, "  实例吞吐  : %.0f 万实例/秒\n", N / (avg/1000.0) / 10000.0);
    fprintf(stdout, "  顶点吞吐  : %.0f 万顶点/秒\n", N*4 / (avg/1000.0) / 10000.0);

    // 回读耗时（离屏架构必需，§2.5.3）
    // ⚠️ 必须逐次计时：把多次放在一个循环里测平均会被判定可优化，得到 0.00 ms 假结果。
    // ⚠️ BUG-002: F16 附件不能用 GL_UNSIGNED_BYTE 回读。这里分两条路：
    //      - RGBA8 附件 → read_pixels_u8，得到真实 u8 回读带宽
    //      - F16  附件 → read_pixels（读 float + CPU 转 u8），耗时含转换开销，
    //                    只作为「功能可用性」验证，不作为带宽指标
    std::vector<uint8_t> px((size_t)W * H * 4);
    read_pixels(t, px.data());          // 预热（同时验证回读路径无误）
    glFinish();
    std::vector<double> rb;
    rb.reserve(50);
    for (int i = 0; i < 50; ++i) {
        auto r0 = Clock::now();
        if (t.f16_color) read_pixels(t, px.data());
        else             read_pixels_u8(t, px.data());
        rb.push_back(std::chrono::duration<double,std::milli>(Clock::now()-r0).count());
    }
    std::sort(rb.begin(), rb.end());
    double rms = rb[25];
    const double mbps = (double)W*H*4/1048576.0/(rms/1000.0);
    fprintf(stdout, "\n【回读】%s %d×%d →RGBA8 (50 次中位数)\n",
            t.f16_color ? "F16(读float+CPU转换)" : "RGBA8(纯u8)", W, H);
    fprintf(stdout, "  单次: %.3f ms  → %.0f MB/s%s\n", rms, mbps,
            t.f16_color ? "  (含转换开销，非纯带宽)" : "");
    fprintf(stdout, "  min/max: %.3f / %.3f ms\n", rb.front(), rb.back());

    // 非空像素占比，确认真的画出了东西
    size_t nonzero = 0;
    for (size_t i = 0; i < px.size(); i += 4)
        if (px[i] || px[i+1] || px[i+2]) ++nonzero;
    double cover = 100.0 * nonzero / ((size_t)W*H);
    // 理论覆盖率：N 个 quad，每个 (2*quad)^2 像素（会互相重叠，只作量级参照）
    double expect = 100.0 * (double)N * (2.0*quad)*(2.0*quad) / ((double)W*H);
    fprintf(stdout, "\n【验证】非空像素 %zu / %zu (%.1f%%)\n", nonzero, (size_t)W*H, cover);
    fprintf(stdout, "  理论覆盖(含重叠): %.1f%%  → %s\n", expect,
            nonzero > (size_t)W*H/100 ? "✅ 渲染成功" : "⚠️ 画面几乎全黑，检查 shader");
    (void)p95;

    fprintf(stdout, "\n=====================================================\n");
    t.destroy();
    return 0;
}
