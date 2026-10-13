// renderer.cpp — 3DGS 渲染器实现（本项目最大文件，1583 行）
//
// ══ 本文件结构（按阅读顺序）══════════════════════════════════════════
//   【工具函数】                                            ~15–90
//     ms_since / rsqrt_approx / half_to_float
//   【★着色器源码】                                        ~90–180
//     VS_SRC : gl_VertexID → quad 四角；屏幕中心 ± 半径 → clip
//              注意 clip.y 要取反（像素 y 向下，GL y 向上）
//     FS_SRC : d2 = 马氏距离² → w = exp(-0.5·d2) → 预乘 alpha
//              d2 > 9.0 丢弃（3σ 外，省混合带宽）
//              a < 0.0039 丢弃（< 1/255 看不出）
//              ★ o_depth = vec4(a·v_depth, a, a, a) —— 深度图的来源
//                （a 处必须是 a 不是 1.0，否则深度退化成"最近高斯的 z"）
//   【GL 扩展加载】                                        ~182–218
//     buffer_storage / map_buffer_range（GLES 3.2 要动态取函数指针）
//   【生命周期】                                           ~219–248
//   【初始化】                                             ~249–430
//     init_gl_only → build_shader_program → build_gl_resources
//     → bind_instance_attribs（instanced attribute 绑定点）
//   【★核心 render_frame(cam, collect_prev)】              往下找同名函数
//     ① 视锥剔除  ② 投影（调 projfast）  ③ 排序（调 sorter）
//     ④ 组装实例数组  ⑤ 上传  ⑥ glDrawArraysInstanced  ⑦ 回读
//     渲染统计量（ms_project/ms_sort/ms_assemble/…）在这里累加
//   【collect_async_readback】PBO 双缓冲异步回读（供帧间流水线用）
//
// ══ 与其他文件的关系 ═════════════════════════════════════════════════
//   camera.h/.cpp    EWA 投影通用路径（project_gaussian）
//   projfast.cpp     投影快路径（实际渲染走这条，把 J·W3 折叠掉）
//   sorter.h/.cpp    深度排序
//   offscreen.h/.cpp EGL 上下文与离屏 FBO
//   ply_loader.cpp   PLY → Splat 数组
//
// ══ 三条关键设计决策（均有实测依据，详见 renderer.h 头部）═════════════
//   · 投影在 CPU 做 —— 顶点着色器里放 sqrt/atan2 掉 14.7× 帧率
//   · 颜色附件必须 RGBA8 —— RGBA16F 回读慢 42×
//   · 顺序 alpha 混合靠「按深度排序的实例顺序 + GL_ONE/GL_ONE_MINUS_SRC_ALPHA」
//     GL 规范保证同一 draw call 内实例按 gl_InstanceID 递增顺序合成
//
#include "renderer.h"
#include "proj_core.h"
#include "projfast.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <thread>
#include <utility>
#include <cstddef>

namespace gsplat {

using Clock = std::chrono::steady_clock;
static inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ── 快速 1/sqrt(x)（Quake 式魔数 + 1 次 Newton 迭代）──────────────
//
// 为什么不用 1.0f/std::sqrt(x)：
//   A76 上 sqrt 本身 ~10-20 cycle，再加一次除法（~10 cycle）；
//   这里的整数魔数版只要 ~6 cycle（1 次移位减法 + 1 mul + 1 Newton:
//   1 mul + 1 fma + 1 mul）。
//   用于**颜色**计算时 1 次 Newton 后相对误差 < 0.2%，远小于 8-bit
//   量化的 1/255 ≈ 0.4%，视觉完全不可见。
//
// ⚠️ 绝不能把这个用到 EWA 投影数学里 —— 那里的误差会被协方差放大，
//    传导到半径与椭圆朝向，导致画面出现锯齿/漏点。投影必须用精确 sqrt。
// ── ◀GUIDE▶ 【5.1】 rsqrt_approx()  快速求 1/√x
//
//   干什么   : 用一个便宜的近似算法算 1/sqrt(x)（牛顿迭代 1 次）。
//   参数    : x —— 正浮点数
//   返回    : ≈1/√x
//   为什么   : 投影每帧要算几百万次，用近似值换速度；视觉上无差异。
// ──────────────────────────────────────────────────────────────
static inline float rsqrt_approx(float x) {
    // 输入 <=0 必须短路：位模式下负数的指数域会被错误解释，
    // NaN/Inf 会一路污染到颜色输出（表现为整片异常色块）。
    if (!(x > 1e-12f)) return 0.0f;
    const float xh = x * 0.5f;
    union { float f; uint32_t i; } u;
    u.f = x;
    u.i = 0x5F375A86u - (u.i >> 1);     // 魔数 + 指数折半
    float y = u.f;
    y = y * (1.5f - xh * y * y);        // 1 次 Newton-Raphson
    return y;
}

// ── ★ IEEE 754 half (binary16) → float 解码 ─────────────────────────
//
// 为什么需要它：深度附件是 RGBA16F，用 glReadPixels(GL_RGBA, GL_FLOAT)
// 会**在 GPU 侧把 half 展开成 float 再传 4 倍数据**。
// 实测 512×288 的深度回读要 30 ms，而同样尺寸的 RGBA8 只要 ~1 ms
// —— 说明瓶颈是传输量而不是解码。
//
// 改用 GL_HALF_FLOAT 直接读原始 16 位，传输量立刻减半，
// 解码在 CPU 做（下面这个函数，纯整数运算，极快）。
//
// 实现用标准的位拼接法（比查表省缓存，比 float 运算快）：
//   sign(1) | exp(5) | mantissa(10)
// 三种情况：正规数 / 非正规数（exp=0）/ Inf·NaN（exp=31）
// ── ◀GUIDE▶ 【5.2】 half_to_float()  半精度(float16) → 单精度(float32)
//
//   干什么   : 把 GPU 回读的 16 位半精度深度值解码成 32 位 float。
//   参数    : h —— 16 位半精度位模式
//   返回    : float
//   为什么   : 深度附件用 RGBA16F 省一半带宽（回读 42× 更快），但 CPU 上要解码。
// ──────────────────────────────────────────────────────────────
static inline float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                                   // ±0
        } else {
            // 非正规数：规格化（左移直到隐含位出现）
            uint32_t m = mant, e = 127u - 15u + 1u;
            while (!(m & 0x400u)) { m <<= 1; --e; }
            m &= 0x3FFu;
            bits = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7F800000u | (mant << 13);          // Inf / NaN
    } else {
        bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
    union { uint32_t i; float f; } u; u.i = bits;
    return u.f;
}

// ── 着色器 ─────────────────────────────────────────────────────────
// 顶点着色器：从实例缓冲取 (中心, 半径)，把 gl_VertexID 0..3 展开成 quad。
// ⚠️ 严格遵守 M0 铁律：**不含任何 sqrt/cos/sin**，只有加乘。
//    椭圆的朝向与长短轴已经编码进 inv_a/inv_b/inv_c，在片元侧用
//    二次型求值，所以顶点侧不需要三角函数。
//
// 关键：attr 用 **divisor=1**（每个实例取一次），配
//       glDrawArraysInstanced(TRIANGLE_STRIP, 0, 4, N) 一次画完 N 个。
static const char* VS_SRC = R"(#version 320 es
in vec2 a_center;      // 屏幕像素中心
in float a_radius;     // quad 半径
in vec3 a_cov;         // (inv_a, inv_b, inv_c)
in vec4 a_color;       // 预乘 alpha 颜色
// ★ 深度输出：沿射线的度量深度（相机空间 z，米）。
//   由 CPU 侧在投影阶段一并算出并作为 instanced attribute 传进来
//   （理由见 SplatInstance 注释：CPU 已有视空间 z，顺手传零成本）。
in float a_depth;
uniform vec2 u_res;
out vec2  v_delta;
out vec3  v_cov;
out vec4  v_color;
out float v_depth;
void main() {
    // gl_VertexID 0..3 → quad 四角（TRIANGLE_STRIP）
    vec2 c = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));
    vec2 corner = c * 2.0 - 1.0;                 // (-1,-1)..(1,1)
    vec2 delta  = corner * a_radius;
    vec2 pix    = a_center + delta;
    vec2 clip   = pix / u_res * 2.0 - 1.0;
    clip.y = -clip.y;                            // 像素 y 向下 → GL y 向上
    gl_Position = vec4(clip, 0.0, 1.0);
    v_delta = delta;
    v_cov   = a_cov;
    v_color = a_color;
    v_depth = a_depth;
}
)";

// 片元着色器：对 2D 高斯做二次型求值，得到 alpha 权重。
//   d² = inv_a·dx² + 2·inv_b·dx·dy + inv_c·dy²
//   w  = exp(-0.5 · d²)
// 因为颜色是**预乘 alpha** 的，最终输出 = color.rgb·w, color.a·w
// 配 glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA) 即可正确合成。
//
// ★★ 深度输出（MRT，参考 gsplat 的 "RGB+ED" 的 CUDA 实现思路）★★
// 同一次光栅化里，第二个颜色附件累积深度统计量：
//     o_depth.r = a·z     （α 加权的深度和  → ∑wᵢzᵢ）
//     o_depth.g = a       （权重和          → ∑wᵢ）
//     o_depth.b = a       （同上，便于以后换中值法）
// 混合方程同为 (GL_ONE, GL_ONE_MINUS_SRC_ALPHA)，即
//     dst ← src + dst·(1-src.a)
// 注意：**深度附件也要走同一个 glBlendFunc**，所以这里把 src.a 设成 a，
// 让 (1-a) 正确地做透射率衰减 —— 与颜色通道完全一致。
// 归一化（÷∑wᵢ）在 CPU 侧回读后做，因为 GL 的混合方程里没法做除法。
// ⚠️⚠️ Mali/GLES 的硬约束（实测踩到）：
//   "Multiple output variables defined. Location must be specified for all outputs."
//   一旦**任何一个** out 写了 layout(location=N)，其它 out 也必须写。
//   只给 o_depth 写 location(1) 而留 o_color 裸写 → 编译直接失败。
//   所以两个输出都必须显式标 location（0 和 1）。
//
// 注意 u_depth_mode 的语义：因为程序在编译期就固定了输出个数，
// 这里用 uniform 做运行期开关的好处是可以只用一个 program，
// 关掉深度时 shader 里判 1 就不写 o_depth（省 MRT 的 tile 回写带宽）。
static const char* FS_SRC = R"(#version 320 es
precision highp float;
in vec2 v_delta;
in vec3 v_cov;
in vec4 v_color;
in float v_depth;
uniform int u_depth_mode;   // 0 = 输出深度, 1 = 关闭（省 MRT 写带宽）
layout(location = 0) out vec4 o_color;
layout(location = 1) out vec4 o_depth;
void main() {
    float dx = v_delta.x, dy = v_delta.y;
    float d2 = v_cov.x*dx*dx + 2.0*v_cov.y*dx*dy + v_cov.z*dy*dy;
    if (d2 > 9.0) discard;                 // 超出 3σ 丢掉，省混合带宽
    float w = exp(-0.5 * d2);
    float a = v_color.a * w;
    if (a < 0.0039) discard;               // < 1/255，看不出
    o_color = vec4(v_color.rgb * w, a);    // 预乘 alpha
    if (u_depth_mode == 0) {
        // a·z 与 a：与颜色同一条混合方程累积
        // ⚠️ 这里 z 已经是度量深度（相机空间 z），不是 NDC 深度。
        // ⚠️⚠️ **o_depth.a 必须是 a，不能是 1.0**！这是踩过的大坑：
        //   混合方程 dst = src + dst·(1-src.a)。若把 src.a 写成 1.0，
        //   则 (1-src.a)=0 → 后面每个高斯都把累积值**清零重写**，
        //   最终深度退化成「最后一个（最近）高斯的 z」，与 ∑a 完全无关。
        //   实测表现：GPU 深度 28.79，而正确值 30.07（差 1.28 m），
        //   且差值恰好等于「最近命中高斯的 z 距加权平均的偏移」。
        //   ——这个 bug 用肉眼根本看不出来（灰度图仍然"近亮远暗"，
        //     轮廓也对），只有逐像素数值比对才能抓到。
        o_depth = vec4(a * v_depth, a, a, a);
    } else {
        o_depth = vec4(0.0);
    }
}
)";

// ── 持久映射辅助 ───────────────────────────────────────────────────
namespace {
// GL_EXT_buffer_storage 的函数指针（ES 3.2 核心没有它，必须 eglGetProcAddress）
typedef void (GL_APIENTRY *PFNGLBUFFERSTORAGEEXTPROC)(GLenum, GLsizeiptr, void*, GLbitfield);
typedef void* (GL_APIENTRY *PFNGLMAPBUFFERRANGEEXTPROC)(GLenum, GLintptr, GLsizeiptr, GLbitfield);
typedef GLboolean (GL_APIENTRY *PFNGLUNMAPBUFFERPROC)(GLenum);

PFNGLBUFFERSTORAGEEXTPROC p_buffer_storage = nullptr;

// ── ◀GUIDE▶ 【5.3】 load_buffer_storage()  加载 GL 扩展函数指针
//
//   干什么   : 从 EGL 里捞出 glBufferStorageEXT / glMapBufferRange 等扩展函数地址。
//   参数    : 无
//   返回    : 无
//   为什么   : GLES 3.2 头文件里没有这些扩展的声明，必须运行时向驱动问地址。
//     这是一个进程级只做一次的操作（有 has_bs_ 标志）。
// ──────────────────────────────────────────────────────────────
void load_buffer_storage() {
    if (p_buffer_storage) return;
    // 先确认扩展存在
    GLint n = 0; glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    bool has = false;
    for (GLint i = 0; i < n; ++i) {
        const char* e = (const char*)glGetStringi(GL_EXTENSIONS, (GLuint)i);
        if (e && !strcmp(e, "GL_EXT_buffer_storage")) { has = true; break; }
    }
    if (!has) return;
    p_buffer_storage = (PFNGLBUFFERSTORAGEEXTPROC)eglGetProcAddress("glBufferStorageEXT");
}

#ifndef GL_MAP_PERSISTENT_BIT_EXT
#define GL_MAP_PERSISTENT_BIT_EXT      0x0040
#endif
#ifndef GL_MAP_COHERENT_BIT_EXT
#define GL_MAP_COHERENT_BIT_EXT        0x0080
#endif
#ifndef GL_CLIENT_STORAGE_BIT_EXT
#define GL_CLIENT_STORAGE_BIT_EXT      0x0200
#endif
// ES 3.2 头文件里可能没有该名字（它是桌面 GL 的）；EXT_buffer_storage
// 的 flags 里这个位表示"缓冲内容会被频繁更新"，位值与桌面一致。
#ifndef GL_DYNAMIC_STORAGE_BIT
#define GL_DYNAMIC_STORAGE_BIT         0x0100
#endif
} // anon namespace

Renderer::Renderer() {
    load_buffer_storage();
}

Renderer::~Renderer() {
    if (gl_ready_) {
        // ★ 先删 sync 对象（Task #12）：必须在销毁 GL 上下文之前，
        //   否则会泄漏一个 GPU 侧 sync（驱动会在 context 销毁时清，
        //   但显式删除更干净，也避免 Valgrind/ASAN 报未释放）。
        if (fence_) { glDeleteSync((GLsync)fence_); fence_ = nullptr; }
        if (fence_depth_) { glDeleteSync((GLsync)fence_depth_); fence_depth_ = nullptr; }
        free_readback_pbos();     // ★ 异步回读的 PBO + fence
        if (depth_rb_) glDeleteRenderbuffers(1, &depth_rb_);
        for (int k = 0; k < 2; ++k) {
            if (ibuf_[k].map && ibuf_[k].vbo) {
                glBindBuffer(GL_ARRAY_BUFFER, ibuf_[k].vbo);
                glUnmapBuffer(GL_ARRAY_BUFFER);
                ibuf_[k].map = nullptr;
            }
            if (ibuf_[k].vao) glDeleteVertexArrays(1, &ibuf_[k].vao);
            if (ibuf_[k].vbo) glDeleteBuffers(1, &ibuf_[k].vbo);
        }
        if (vao_)      glDeleteVertexArrays(1, &vao_);
        if (prog_)     glDeleteProgram(prog_);
        if (color8_)   glDeleteRenderbuffers(1, &color8_);
        if (fbo8_)     glDeleteFramebuffers(1, &fbo8_);
        gl_.destroy();
    }
}

// ── ◀GUIDE▶ 【4.1】 Renderer::init_gl_only()  只建 GL 环境，不读 PLY
//
//   干什么   : 建 EGL 离屏上下文 → 编译 shader → 建 VAO/VBO/FBO 等所有 GL 资源。
//   参数    : cfg —— 渲染配置（分辨率/深度开关/线程数，见 renderer.h RenderConfig）
//   返回    : true 成功 / false 失败（会打印失败原因）
//   被谁调   : Renderer::init() 内部；也可单独用来跑合成数据测试。
//   ★★坑   : EGL 上下文一在这里创建就**永久绑定当前线程**。
//     换个线程调 GL 会直接死锁（只有一个 GL 上下文的线程能画）。
//     所以主渲染必须在建上下文的那个线程里做。
// ──────────────────────────────────────────────────────────────
bool Renderer::init_gl_only(const RenderConfig& cfg) {
    cfg_ = cfg;

    if (!create_offscreen(gl_, cfg.width, cfg.height, cfg.verbose, cfg.f16_color)) {
        fprintf(stderr, "[renderer] 离屏上下文创建失败\n");
        return false;
    }

    // 若用 F16 混合，额外建一个 RGBA8 目标用于回读（M0 结论三）
    if (cfg.f16_color) {
        glGenFramebuffers(1, &fbo8_);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo8_);
        glGenRenderbuffers(1, &color8_);
        glBindRenderbuffer(GL_RENDERBUFFER, color8_);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, cfg.width, cfg.height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                  GL_RENDERBUFFER, color8_);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            fprintf(stderr, "[renderer] RGBA8 blit 目标不完整\n");
            return false;
        }
    }

    if (!build_shader_program()) return false;
    if (!build_gl_resources()) return false;

    // ── ★ 深度附件（MRT）：GL_COLOR_ATTACHMENT1 ────────────────────
    // 单独挂一个 RGBA16F renderbuffer 到主 FBO 的第 1 个颜色附件。
    //   .r = ∑(a·z)   .g = ∑a   .b = ∑a   .a = 1
    // 用 RGBA16F 的原因：深度量级（几米~几百米）用 8 位根本不够
    // （量化步长会到米级），16 位浮点有 ~3 位十进制有效数字，够用。
    //
    // ⚠️ 只挂附件还不够，必须 glDrawBuffers 显式声明「这次 draw 写哪些
    //    附件」。默认 MRT 只写 ATTACHMENT0，漏了这步深度永远是 0。
    if (cfg_.depth) {
        glBindFramebuffer(GL_FRAMEBUFFER, gl_.fbo);
        glGenRenderbuffers(1, &depth_rb_);
        glBindRenderbuffer(GL_RENDERBUFFER, depth_rb_);
        // ★ 优先按用户要求建 32F；若不支持（部分驱动对 RGBA32F renderbuffer
        //   有格式限制），自动退回 16F 并告知。
        bool want_f32 = cfg_.depth_f32;
        GLint f32_ok = 0;
        if (want_f32) glGetInternalformativ(GL_RENDERBUFFER, GL_RGBA32F,
                                            GL_NUM_SAMPLE_COUNTS, 1, &f32_ok);
        if (want_f32) {
            glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, cfg.width, cfg.height);
            if (glGetError() != GL_NO_ERROR) {
                fprintf(stderr, "[renderer] ⚠️ RGBA32F 深度附件不支持，退回 16F\n");
                want_f32 = false;
            }
        }
        if (!want_f32)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA16F, cfg.width, cfg.height);
        depth_is_f32_ = want_f32;
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                                  GL_RENDERBUFFER, depth_rb_);
        const GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (st != GL_FRAMEBUFFER_COMPLETE) {
            fprintf(stderr, "[renderer] ⚠️ 深度附件挂载后 FBO 不完整 (0x%x)，"
                            "自动关闭深度输出\n", st);
            glDeleteRenderbuffers(1, &depth_rb_);
            depth_rb_ = 0;
            cfg_.depth = false;
        } else {
            draw_bufs_[0] = GL_COLOR_ATTACHMENT0;
            draw_bufs_[1] = GL_COLOR_ATTACHMENT1;
            n_draw_bufs_  = 2;
            fprintf(stdout, "[renderer] 深度输出已启用（MRT，attachment1 %s）\n",
                    depth_is_f32_ ? "RGBA32F" : "RGBA16F");
        }
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // 3DGS 必须做顺序 alpha 混合
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);   // 预乘 alpha
    glDisable(GL_DEPTH_TEST);
    // 关掉背面剔除（quad 的绕序随朝向变化，剔除会漏画）
    glDisable(GL_CULL_FACE);

    // 热管理：初始化并接管风扇
    thermal_.init(cfg.verbose);
    if (cfg.auto_fan) thermal_.set_curve_auto(true);

    frame_.assign((size_t)cfg.width * cfg.height * 4, 0);
    if (cfg_.depth) {
        const size_t np = (size_t)cfg.width * cfg.height;
        depth_.assign(np, 0.f);
        depth_wsum_.assign(np, 0.f);
    }
    gl_ready_ = true;
    return true;
}

// ── ◀GUIDE▶ 【4.2】 Renderer::build_shader_program()  编译并链接 GPU 着色器
//
//   干什么   : 把上面的 VS_SRC / FS_SRC 两段 GLSL 编译成 GPU 程序，并取好 uniform 位置。
//   参数    : 无（用的是文件顶部的源码字符串）
//   返回    : true 成功 / false 编译失败
//   两个 shader: 顶点着色器：拿高斯中心+协方差算出四边形的 4 个角（gl_VertexID 生成）
//     片元着色器：算每个像素在马氏距离下的 alpha，并同时输出颜色与深度
// ──────────────────────────────────────────────────────────────
bool Renderer::build_shader_program() {
    static const char* names[] = { "a_center", "a_radius", "a_cov", "a_color" };
    static const GLuint locs[] = { 0, 1, 2, 3 };
    // 显式限定命名空间：否则会解析到成员函数自身（已改名规避，这里再加一层保险）
    prog_ = gsplat::build_program(VS_SRC, FS_SRC, cfg_.verbose, names, locs, 4);
    if (!prog_) return false;

    u_res_   = glGetUniformLocation(prog_, "u_res");
    u_depth_mode_ = glGetUniformLocation(prog_, "u_depth_mode");
    u_max_r_ = -1;
    return true;
}

// ── ◀GUIDE▶ 【4.3】 Renderer::build_gl_resources()  建所有 GL 缓冲对象
//
//   干什么   : 建实例数据的 VBO、索引缓冲、以及颜色/深度两个附件的 FBO。
//   参数    : 无
//   返回    : true 成功
//   名词解释  : VBO=显存里的数组；VAO=数组的「格式说明书」；FBO=离屏画布（不显示到屏幕）
// ──────────────────────────────────────────────────────────────
bool Renderer::build_gl_resources() {
    // 空场景占位 VAO（保持 API 一致性）
    glGenVertexArrays(1, &vao_);

    // 双缓冲：两个 VBO，各配一个 VAO（属性绑定是 VAO 状态，必须分开）
    GLuint vbos[2] = {0,0};
    glGenBuffers(2, vbos);
    for (int k = 0; k < 2; ++k) {
        ibuf_[k].vbo = vbos[k];
        glGenVertexArrays(1, &ibuf_[k].vao);
        glBindVertexArray(ibuf_[k].vao);
        glBindBuffer(GL_ARRAY_BUFFER, ibuf_[k].vbo);
        bind_instance_attribs();
    }
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    if (cfg_.verbose) {
        fprintf(stdout, "[renderer] 上传路径: %s\n",
                p_buffer_storage ? "持久映射(EXT_buffer_storage)" : "map+INVALIDATE");
    }
    return true;
}

// 把 SplatInstance 的 4 个字段绑到 location 0..3，全部 divisor=1。
// ⚠️ 必须对**每个** VBO 都绑一遍 —— VAO 记录的是「attribute → buffer」
//    的绑定关系，切 VBO 后不重绑就会指向旧缓冲（经典 GL 陷阱）。
// ── ◀GUIDE▶ 【4.4】 Renderer::bind_instance_attribs()  声明每个高斯的属性布局
//
//   干什么   : 告诉 GPU「每一份实例数据里，哪几个字节是位置、哪几个是协方差、颜色…」。
//   参数    : 无
//   返回    : 无
//   ★关键   : 用 glVertexAttribDivisor(…,1) 表示「每个实例前进一格」而不是每个顶点一格。
//     这一个设置让「一个高斯 = 一个实例 = 一个四边形」成立。
// ──────────────────────────────────────────────────────────────
void Renderer::bind_instance_attribs() {
    const GLsizei stride = sizeof(SplatInstance);

    // ⚠️ 每个属性都要 glVertexAttribDivisor(..., 1)：
    //    表示"每实例前进一次"，才能配合 glDrawArraysInstanced 一次画 N 个。
    //    漏掉 divisor 会让所有实例共用第 0 个数据 —— 那是本项目早期
    //    全黑 bug (BUG-001) 的同类陷阱。
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride,
                          (const void*)offsetof(SplatInstance, px));
    glVertexAttribDivisor(0, 1);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, stride,
                          (const void*)offsetof(SplatInstance, radius));
    glVertexAttribDivisor(1, 1);

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride,
                          (const void*)offsetof(SplatInstance, inv_a));
    glVertexAttribDivisor(2, 1);

    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride,
                          (const void*)offsetof(SplatInstance, pr));
    glVertexAttribDivisor(3, 1);

    // ★ location 4 = 深度（视空间 z）。divisor 1，同其它实例属性。
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, stride,
                          (const void*)offsetof(SplatInstance, depth));
    glVertexAttribDivisor(4, 1);
}

// ── ★ LOD：按静态重要度截断高斯数组 ────────────────────────────────
//
// 重要度 = opacity × √(最大两轴尺度之积)
//   · 取**两轴**之积而不是单看最大尺度：各向异性的针状高斯（一轴极长、
//     另两轴极细）屏幕面积其实很小，只看最大值会把它们误判成"重要"；
//     √(a·b) ≈ 椭圆面积为量纲的屏幕贡献代理量。
//   · opacity 在加载时已经 sigmoid 过，这里直接用。
//
// ★ 截断后**按原索引升序重排**（不按分数排）：保持与 PLY 一致的空间
//   遍历顺序 —— 块剔除命中率与缓存局部性都依赖它。
//
// 复杂度：nth_element O(n) + 一次保序 sort O(k log k) + 一次搬运 O(k)。
// 1.41M → 0.9M 实测约 0.2 s（一次性，可忽略）。
// ── ◀GUIDE▶ 【4.5】 Renderer::apply_importance_lod()  LOD 降采样（按重要度留前 N 个高斯）
//
//   干什么   : 按「不透明度 × 屏幕贡献上界」排序，只保留最重要的 keep 个高斯。
//   参数    : keep —— 保留数量（0 = 全部保留）
//   返回    : 无
//   实测结论  : 3M→1M：投影 50.7→22.5 ms，但端到端只从 5.08→5.83 FPS，
//     而 mT 从 7.06 m 恶化到 12.21 m。→ 性价比差，默认关闭（--lod 0）。
// ──────────────────────────────────────────────────────────────
void Renderer::apply_importance_lod(size_t keep) {
    const size_t n = splats_.size();
    if (keep == 0 || keep >= n) return;

    std::vector<std::pair<float, uint32_t>> sc(n);
    for (size_t i = 0; i < n; ++i) {
        const Splat& s = splats_[i];
        float a = s.scale0, b = s.scale1, c = s.scale2;
        if (a < b) std::swap(a, b);
        if (b < c) std::swap(b, c);
        if (a < b) std::swap(a, b);          // 现在 a >= b >= c
        sc[i] = { s.opacity * std::sqrt(std::fmax(0.f, a * b)), (uint32_t)i };
    }
    std::nth_element(sc.begin(), sc.begin() + (std::ptrdiff_t)keep, sc.end(),
                     [](const std::pair<float, uint32_t>& x,
                        const std::pair<float, uint32_t>& y) {
                         return x.first > y.first;
                     });
    sc.resize(keep);
    std::sort(sc.begin(), sc.end(),
              [](const std::pair<float, uint32_t>& x,
                 const std::pair<float, uint32_t>& y) {
                  return x.second < y.second;      // 保序
              });

    std::vector<Splat>    ns(keep);
    std::vector<ShCoeffs> nsh(keep);
    for (size_t k = 0; k < keep; ++k) {
        ns[k]  = splats_[sc[k].second];
        nsh[k] = sh_[sc[k].second];
    }
    splats_.swap(ns);
    sh_.swap(nsh);
    fprintf(stdout, "[renderer] LOD: 保留 %zu/%zu 个高斯"
                    "（按 opacity×√(a·b) 重要度，保序重排）\n", keep, n);
}

// ── ◀GUIDE▶ 【4.0】 Renderer::init()  渲染器总初始化
//
//   干什么   : = init_gl_only（建 GL）+ 读 PLY（建高斯数组）+ 预分块，一步到位。
//   参数    : ply_path —— 3DGS 模型文件；cfg —— 渲染配置
//   返回    : true 成功 / false 失败
//   被谁调   : main() 第 5 步（约第 823 行）
//   耗时    : PLY 加载 ~400 ms（709 MB，1800 MB/s），一次性成本。
// ──────────────────────────────────────────────────────────────
bool Renderer::init(const char* ply_path, const RenderConfig& cfg) {
    cfg_ = cfg;

    // 加载 PLY（含 SH）
    if (!load_ply_splats(ply_path, splats_, sh_, cfg.sh_degree, true, &is_2dgs_)) {
        fprintf(stderr, "[renderer] PLY 加载失败: %s\n", ply_path);
        return false;
    }
    fprintf(stdout, "[renderer] 已加载 %zu 个高斯（SH degree=%d）\n",
            splats_.size(), cfg.sh_degree);

    // ★ LOD 截断（可选）：必须在 init_gl_only / build_blocks 之前，
    //   后面所有东西（proj_/order_/blocks_/预留缓冲）都按截断后的规模来。
    if (cfg_.lod_keep > 0) apply_importance_lod((size_t)cfg_.lod_keep);

    if (!init_gl_only(cfg)) return false;

    // 预分配每帧缓冲
    proj_.resize(splats_.size());
    order_.resize(splats_.size());
    instances_.reserve(splats_.size());

    // ★ 空间分块（视锥剔除用）
    build_blocks();
    return true;
}

// ── ★ 空间分块：预计算每块的世界系 AABB ─────────────────────────────
//
// 为什么按「索引连续区间」切块而不是按空间聚类：
//   · pilot.ply 这类模型是**空间有序**的（先前实测：前 60 万高斯都在
//     场景一个角），索引连续 ≈ 空间相邻，分块质量已经够好；
//   · 索引区间还让「按块遍历」的循环边界计算变成一次整数除法；
//   · 真正的空间聚类（k-d/八叉树）留给后续按需升级。
//
// 块大小的取舍：
//   太小 → 块数多、每块 AABB 测试的固定开销占比高；
//   太大 → 一块里只要有一个角进视锥就整块保留，剔除收益低。
//   2048 在 277 万场景下约 1350 块，实测是较好的平衡点。
// ── ◀GUIDE▶ 【4.6】 Renderer::build_blocks()  把高斯切成空间块（供视锥剔除）
//
//   干什么   : 按世界坐标把全部高斯分成若干块（每块 2048 个），每块算一个包围盒。
//   参数    : 无
//   返回    : 无
//   为什么   : 每帧先用 6 个视锥平面剔除整块，跳过不投影 → 省下大量计算。
//   实测    : 本项目俯视时视锥把整个场景包住，剔除命中率 0%，所以主要靠别的优化提速。
// ──────────────────────────────────────────────────────────────
void Renderer::build_blocks() {
    blocks_.clear();
    const size_t n = splats_.size();
    block_used_ = cfg_.block_size > 0 ? cfg_.block_size : 2048;
    if (n == 0) return;
    const int bs = block_used_;
    blocks_.reserve((n + (size_t)bs - 1) / (size_t)bs);
    for (size_t i = 0; i < n; i += (size_t)bs) {
        Block b;
        b.lo = (int)i;
        b.hi = (int)std::min(n, i + (size_t)bs);
        b.visible = true;
        float mnx = 1e30f, mny = 1e30f, mnz = 1e30f;
        float mxx = -1e30f, mxy = -1e30f, mxz = -1e30f;
        for (int k = b.lo; k < b.hi; ++k) {
            const Splat& s = splats_[k];
            if (s.x < mnx) mnx = s.x;   if (s.x > mxx) mxx = s.x;
            if (s.y < mny) mny = s.y;   if (s.y > mxy) mxy = s.y;
            if (s.z < mnz) mnz = s.z;   if (s.z > mxz) mxz = s.z;
        }
        b.mn = Vec3{ mnx, mny, mnz };
        b.mx = Vec3{ mxx, mxy, mxz };
        blocks_.push_back(b);
    }
    fprintf(stdout, "[renderer] 空间分块: %zu 块 × %d 高斯（视锥剔除 %s）\n",
            blocks_.size(), bs, cfg_.block_cull ? "已启用" : "已关闭");
}

// ══ ★ 异步回读（PBO 双缓冲）══════════════════════════════════════════
//
// 动机（实测）：回读耗时里 **99% 是「等 GPU 完成」**，纯 glReadPixels
//   只有 0.7 ms；而等待量 ∝ 可见实例数（939K → 36 ms）。
//
// 做法：把「提交回读」与「收取结果」拆到两帧：
//   帧 N ：渲染 → glReadPixels 写进 PBO[b]（不等）→ glFenceSync 标记 → 返回
//   帧 N+1：开头检查 PBO[b] 的 fence；若已完成 → 映射 → 拷出 → 归一化
//          （此时 GPU 早在做 N+1 的光栅化了，等待被完全隐藏）
//
// ⚠️ 语义变化：取到的数据是**上一帧**的。所以必须用 frame_camera()
//   配对（它记录的是 pending_cam_），否则反投影会用错位姿。
// ── ◀GUIDE▶ 【5.4】 Renderer::free_readback_pbos()  释放异步回读用的 PBO
//
//   干什么   : 删掉双缓冲的两个 PBO（像素缓冲对象）与它们关联的 sync 对象。
//   参数    : 无
//   返回    : 无
//   名词解释  : PBO = 显存里的一块「搬运缓冲区」，glReadPixels 可以写进它而不用等 CPU。
// ──────────────────────────────────────────────────────────────
void Renderer::free_readback_pbos() {
    for (int i = 0; i < 2; ++i) {
        if (rb_color_[i].fence) { glDeleteSync((GLsync)rb_color_[i].fence); rb_color_[i].fence = nullptr; }
        if (rb_depth_[i].fence) { glDeleteSync((GLsync)rb_depth_[i].fence); rb_depth_[i].fence = nullptr; }
        if (rb_color_[i].pbo) { glDeleteBuffers(1, &rb_color_[i].pbo); rb_color_[i].pbo = 0; }
        if (rb_depth_[i].pbo) { glDeleteBuffers(1, &rb_depth_[i].pbo); rb_depth_[i].pbo = 0; }
        rb_color_[i].cap = rb_depth_[i].cap = 0;
    }
    rb_pending_ = false;
    has_frame_  = false;
}

// ── ◀GUIDE▶ 【5.4】 Renderer::issue_async_readback()  ★ 投递一帧渲染（不等 GPU 画完）
//
//   干什么   : 把「画这一帧 + 把结果搬进 PBO」的命令丢进 GL 队列，**立刻返回**。
//   参数    : cam —— 本帧相机（世界→相机矩阵在 cam.world_to_cam）
//   返回    : true 投递成功 / false 没有空闲 PBO（上一帧还没被取走）
//   ★价值   : 回读等待占渲染时间的 99%（939K 实例要等 36 ms），
//     异步化后这 36 ms 被完全隐藏，是流水线提速的关键。
//   ★代价   : 结果滞后一帧 —— 反投影时必须用 frame_camera() 返回的相机配对。
// ──────────────────────────────────────────────────────────────
bool Renderer::issue_async_readback(const Camera& cam) {
    const size_t np  = (size_t)cfg_.width * cfg_.height;
    const size_t cb  = np * 4;                                   // RGBA8
    const size_t db  = np * 4 * (depth_is_f32_ ? 4u : 2u);       // 16F:2B/通道

    ReadbackPBO& rc = rb_color_[rb_idx_];
    ReadbackPBO& rd = rb_depth_[rb_idx_];

    if (!rc.pbo || rc.cap < cb) {
        if (!rc.pbo) glGenBuffers(1, &rc.pbo);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, rc.pbo);
        glBufferData(GL_PIXEL_PACK_BUFFER, (GLsizeiptr)cb, nullptr, GL_STREAM_READ);
        rc.cap = cb;
    }
    if (cfg_.depth && (!rd.pbo || rd.cap < db)) {
        if (!rd.pbo) glGenBuffers(1, &rd.pbo);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, rd.pbo);
        glBufferData(GL_PIXEL_PACK_BUFFER, (GLsizeiptr)db, nullptr, GL_STREAM_READ);
        rd.cap = db;
    }

    glBindFramebuffer(GL_READ_FRAMEBUFFER, gl_.fbo);

    // 颜色：最后一次参数是 0 → 目标是当前绑定的 PBO（异步，不阻塞 CPU）
    glBindBuffer(GL_PIXEL_PACK_BUFFER, rc.pbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, cfg_.width, cfg_.height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    if (cfg_.depth && n_draw_bufs_ > 1 && depth_rb_) {
        glBindBuffer(GL_PIXEL_PACK_BUFFER, rd.pbo);
        glReadBuffer(GL_COLOR_ATTACHMENT1);
        glReadPixels(0, 0, cfg_.width, cfg_.height, GL_RGBA,
                     depth_is_f32_ ? GL_FLOAT : GL_HALF_FLOAT, nullptr);
    }
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    // 标记「到这里的命令都完成」——注意 glFenceSync 本身**不阻塞**
    // ⚠️ 要丢弃旧 fence，否则 sync 对象泄漏
    if (rc.fence) { glDeleteSync((GLsync)rc.fence); rc.fence = nullptr; }
    rc.fence = (void*)glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (rd.fence) { glDeleteSync((GLsync)rd.fence); rd.fence = nullptr; }
    rd.fence = (void*)glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

    pending_cam_ = cam;
    rb_pending_  = true;
    rb_idx_     ^= 1;          // 下帧写到另一半
    // ★ 2026-10-09 关键修复：glFenceSync 只是把 fence 命令**记入当前
    //   command buffer**，不保证已提交给 GPU。若 issue 后 CPU 直接返回
    //   （帧间流水线正是如此：下一帧开头才 collect，中间全是非 GL 代码
    //   —— 读图 / ORB），驱动可能迟迟不 flush → 下帧
    //   glClientWaitSync(timeout=0) 永远 TIMEOUT → collect 永远失败，
    //   流水线退化为每帧 2 次渲染（实测 30 帧全中此坑）。
    //   glFlush() 强制提交，代价 ~0.1 ms，一次性解决。
    glFlush();
    return true;
}

// ── ★ 异步回读：收取上一帧（在 render_frame 开头调用）───────────
// 关键：fence 检查是**非阻塞**的（timeout=0）。GPU 还没写完就直接
// 返回 false，留到下一次再收 —— 绝不把 CPU stall 在等待上，
// 而那 36 ms 的等待正是我们要消除的东西。
// ── ◀GUIDE▶ 【8.3.1】 Renderer::collect_async_readback()  ★ 收取上一帧的渲染结果
//
//   干什么   : 把上一帧投递进 PBO 的颜色/深度搬回 CPU，并做期望深度归一化。
//   参数    : 无
//   返回    : true 收到了一帧 / false 还没有
//   ★坑（Mali）: glClientWaitSync 对「glReadPixels→PBO + fence」组合**永不返回**，
//     所以这里改用 glFinish 强制排空；因为此时已隔了 ~50 ms 的 CPU 工作，
//     GPU 早就画完了，glFinish 实际等待 ≈ 0。
//   ★坑（结果）: 流水线模式下第 1 帧取不到数据（还没投递过），必须允许「空手而归」。
// ──────────────────────────────────────────────────────────────
bool Renderer::collect_async_readback() {
    if (!rb_pending_) return false;
    const size_t np   = (size_t)cfg_.width * cfg_.height;
    const int    prev = rb_idx_ ^ 1;    // 本帧已 ^=1，故 prev = 上一帧写入的那半
    ReadbackPBO& rc = rb_color_[prev];
    ReadbackPBO& rd = rb_depth_[prev];

    // ⚠️ Mali 驱动 fence 规避（2026-10-09）：实测 glClientWaitSync 对
    //   「glReadPixels→PBO + glFenceSync」组合**永不返回 signal**
    //   （flush flag、glFlush 均无效；gsrender 里"能工作"只是因为
    //   collect 恰好排在下一帧 GL 命令之后）。改为 glFinish 强制排空：
    //   帧间流水线下，收取发生在上一帧投递的 ~50 ms CPU 工作之后，
    //   GPU（光栅 0.2 ms + PBO DMA ~1 ms）早已完成 → glFinish 近似零等待。
    if (rc.fence) {
        glFinish();
        glDeleteSync((GLsync)rc.fence); rc.fence = nullptr;
    }

    // ── 颜色：映射 PBO → 拷进 frame_ ──────────────────────────────
    glBindBuffer(GL_PIXEL_PACK_BUFFER, rc.pbo);
    void* pc = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, (GLsizeiptr)(np * 4),
                                GL_MAP_READ_BIT);
    if (pc) {
        std::memcpy(frame_.data(), pc, np * 4);
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    }
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    // ── 深度：拷贝 + 期望深度归一化（公式与同步路径完全一致）──────
    if (cfg_.depth && n_draw_bufs_ > 1 && depth_rb_ && rd.pbo) {
        if (rd.fence) {
            glClientWaitSync((GLsync)rd.fence, 0, 0);
            glDeleteSync((GLsync)rd.fence); rd.fence = nullptr;
        }
        const size_t db = np * 4 * (depth_is_f32_ ? 4u : 2u);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, rd.pbo);
        const void* pd = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0,
                                          (GLsizeiptr)db, GL_MAP_READ_BIT);
        if (pd) {
            float dmin = 1e30f, dmax = -1e30f;
            double dsum = 0.0; int nvalid = 0;
            const float wsum_eps = 1e-4f;
            if (!depth_is_f32_) {
                const uint16_t* h = (const uint16_t*)pd;
                for (size_t i = 0; i < np; ++i) {
                    const float az = half_to_float(h[i*4 + 0]);
                    const float ws = half_to_float(h[i*4 + 1]);
                    depth_wsum_[i] = ws;
                    if (ws > wsum_eps) {
                        const float z = az / ws;  depth_[i] = z;
                        if (z < dmin) dmin = z;
                        if (z > dmax) dmax = z;
                        dsum += z; ++nvalid;
                    } else depth_[i] = 0.f;
                }
            } else {
                const float* fp = (const float*)pd;
                for (size_t i = 0; i < np; ++i) {
                    const float az = fp[i*4 + 0], ws = fp[i*4 + 1];
                    depth_wsum_[i] = ws;
                    if (ws > wsum_eps) {
                        const float z = az / ws;  depth_[i] = z;
                        if (z < dmin) dmin = z;
                        if (z > dmax) dmax = z;
                        dsum += z; ++nvalid;
                    } else depth_[i] = 0.f;
                }
            }
            stats_.depth_min   = dmin;
            stats_.depth_max   = dmax;
            stats_.depth_mean  = nvalid ? (float)(dsum / nvalid) : 0.f;
            stats_.depth_valid = nvalid;
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
        }
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    }

    frame_cam_  = pending_cam_;   // ★ 数据 ↔ 相机的配对点
    has_frame_  = true;
    rb_pending_ = false;
    return true;
}

// ── ◀GUIDE▶ 【4.7】 Renderer::set_splats()  直接塞入外部高斯数据
//
//   干什么   : 绕过 PLY 文件，直接把内存里的高斯数组装进渲染器（合成测试用）。
//   参数    : s 高斯数组；sh 球谐颜色系数
//   返回    : 无
// ──────────────────────────────────────────────────────────────
void Renderer::set_splats(std::vector<Splat> s, std::vector<ShCoeffs> sh) {
    splats_ = std::move(s);
    sh_     = std::move(sh);
    // ★ 外部入口：投影热路径里已**不再**做四元数归一化（省钱），
    //   所以从这里进来的数据必须在这里补齐。
    //   正常加载路径（init → ply_loader）已经归一化过，这里是幂等的、
    //   几乎零成本（只跑一次，不在每帧热路径上）。
    for (auto& sp : splats_) {
        const float n = std::sqrt(sp.rot0*sp.rot0 + sp.rot1*sp.rot1 +
                                  sp.rot2*sp.rot2 + sp.rot3*sp.rot3);
        if (n > 1e-12f) {
            const float inv = 1.f / n;
            sp.rot0 *= inv; sp.rot1 *= inv; sp.rot2 *= inv; sp.rot3 *= inv;
        }
    }
    proj_.resize(splats_.size());
    order_.resize(splats_.size());
    instances_.reserve(splats_.size());
    build_blocks();          // 换了高斯集 → 分块必须重建
}

// ── 颜色：SH → RGB ────────────────────────────────────────────────
// 3DGS 的颜色是球谐展开。degree 1 时：
//   c(dir) = 0.5 + dc·Y00 + Σ_{i=1..3} rest_i·Y1i(dir)
// 注意：方向相关的三角函数只在这里出现 —— 每高斯一次、在 CPU 侧，
// 所以完全不受 M0「顶点着色器禁三角函数」铁律的约束。
// 实际的 SH 求值内联在 render_frame 的实例组装循环里（避免多一次遍历）。

// ── 主渲染 ────────────────────────────────────────────────────────
// ── ◀GUIDE▶ 【★ STEP 5 ★】 Renderer::render_frame()  ★★★ 渲染一帧（全文最长的函数，约 770 行）
//
//   干什么   : 给定相机、画出一张「从这个位置看到的 3DGS 参考图」+ 一张深度图。
//   参数    : cam 相机位姿；collect_prev 是否先收取上一帧结果
//     （流水线模式传 false：主程序自己在帧首收，以便渲染只滞后 1 帧）
//   返回    : true 成功
//   五阶段   : ① EWA 投影(CPU) → ② 深度排序(CPU) → ③ 上传显存 →
//     ④ 实例化四边形光栅化(GPU) → ⑤ 回读颜色+深度
//   实测耗时  : 投影 54.0 ms / 排序 15.5 ms / 组装 6.6 ms / 上传+绘制 1.1 ms ≈ 77 ms 净工作
//   阅读建议  : 这个函数很长，按上面 5 个阶段分块读，不要试图一口气看完。
// ──────────────────────────────────────────────────────────────
bool Renderer::render_frame(const Camera& cam, bool collect_prev) {
    if (!gl_ready_) return false;
    const auto t_frame0 = Clock::now();
    stats_ = RenderStats{};
    stats_.total_splats = (int)splats_.size();

    // ── 0. 热管理（每帧调，内部限流；锁频长跑必须靠它散热）────────
    thermal_.tick(cfg_.auto_fan ? 1000 : 0);
    stats_.soc_temp = thermal_.soc();
    stats_.fan_duty = thermal_.has_fan() ? thermal_.fan_speed() : -1;

    // ── 1. CPU 投影 + 实例组装（合并，2026-10-03 优化）──────────
    //
    // ★★ 关键优化：把「颜色求值 + 实例组装」**并入投影循环**。
    //
    //   动机（实测）：原「组装」阶段独占 163 ms（36%），是最大单项。
    //   剖析发现瓶颈不在浮点，而在**访存**：原组装按排序序取 order_[k]，
    //   再散列访问 proj_[idx](32B) + splats_[idx](56B) + sh_[idx](48B)
    //   三个数组，工作集合计 ~400 MB ≫ L2/L3，每次都是 DRAM miss。
    //   163 ms / 2.17M ≈ 75 ns/项 —— 正是随机 DRAM 访问的延迟量级。
    //
    //   而颜色只需要「高斯中心 → 相机」的方向向量，**相机每帧固定**。
    //   故可在投影阶段（顺序访问，proj/splats/sh 都在 cache 里）
    //   顺手把完整 SplatInstance 算出来，写进 inst_by_id_[i]。
    //   投影本就顺序，这笔额外计算的访存成本几乎为零。
    //   排序后只需按 order_ 做一次 40B/项的 permute：
    //     散列读 inst_by_id_[order_[k]]（1 次 miss）+ 顺序写 instances_[k]
    //   把「3 散列读 + 3 散列写」压成「1 散列读 + 1 顺序写」。
    const auto t_proj0 = Clock::now();
    const size_t n = splats_.size();
    const int limit = (cfg_.max_splats > 0 && (size_t)cfg_.max_splats < n)
                          ? cfg_.max_splats : (int)n;

    // 多线程：按块划分（M0 证明光栅化很充裕，瓶颈已移到 CPU 投影）
    //   线程数：显式指定 > 0 时用指定值；否则自动 = 核数/2（留一半给定位）
    const unsigned hw = std::thread::hardware_concurrency();
    // ★ 2026-10-08：默认从 hw/2 改成 hw（吃满全部核）。
    //   实测投影是最大单项：单线程 250 ms → 8 线程 64.9 ms。
    //   原先留一半是给「定位线程并行」预留；真要与定位并行时，
    //   显式传 --threads 4 即可。
    const unsigned nthreads = !cfg_.mt_project ? 1u
        : (cfg_.project_threads > 0
               ? std::min((unsigned)cfg_.project_threads, std::max(1u, hw))
               : std::max(1u, hw));

    // 相机在世界坐标中的位置：p_cam = R·p_world + t → cam = -Rᵀ·t
    // 每帧算一次，供所有高斯的 SH 视线方向复用。
    const Mat4& W = cam.world_to_cam;
    const Vec3 cam_pos{
        -(W.m[0]*W.m[12] + W.m[1]*W.m[13] + W.m[2]*W.m[14]),
        -(W.m[4]*W.m[12] + W.m[5]*W.m[13] + W.m[6]*W.m[14]),
        -(W.m[8]*W.m[12] + W.m[9]*W.m[13] + W.m[10]*W.m[14])
    };
    const bool want_sh1 = (cfg_.sh_degree >= 1);
    if ((size_t)limit > inst_by_id_.size()) inst_by_id_.resize(limit);

    // 逐帧不变量（投影尾部用）：把 sqrt(w²+h²)、屏幕宽高提到帧级算一次。
    // 原先这三个量在 finalize 里**逐高斯**重算，实测多花 16.7 ns/点。
    const ProjConsts pconsts = ProjConsts::from_camera(cam);

    // ── ★ 空间分块视锥剔除（2026-10-08）────────────────────────────
    //
    // 动机：投影阶段原本是 `for (i=0; i<2.77M; ++i)` 全部走一遍，
    //   实测单线程 250 ms（= 90 ns/高斯），**且与可见实例数无关**。
    //   把剔除阈值从 0.5 提到 16（可见数 939K → 16），总耗时只从
    //   284 ms 降到 110 ms —— 那 100 ms 就是「遍历」的地板成本。
    //
    // 做法：预先把高斯切成固定大小的块（世界系 AABB，见 build_blocks），
    //   每帧用视锥的 6 个平面做 AABB 剔除，整块不可见就跳过。
    //
    // 保守性：把块的 8 个角变换到相机系后取新 AABB（旋转会让 AABB
    //   略微膨胀 → 偏保守），再用 p-vertex 做平面测试。
    //   两者都只会「漏剔」（多算几块），不会「误剔」（画面破洞）。
    int nvis_blocks = 0;
    if (cfg_.block_cull && !blocks_.empty()) {
        const float tx = 0.5f * (float)cfg_.width  / cam.fx;   // tan(fovx/2)
        const float ty = 0.5f * (float)cfg_.height / cam.fy;   // tan(fovy/2)
        const float* m = cam.world_to_cam.m;
        // AABB vs 平面（法线指向内侧）。取「最正点」p-vertex：
        // 若连最正点都在平面外侧，整个 AABB 必然在外侧。
        auto outside = [](float nx, float ny, float nz, float d,
                          const float mn[3], const float mx[3]) -> bool {
            const float px = nx >= 0.f ? mx[0] : mn[0];
            const float py = ny >= 0.f ? mx[1] : mn[1];
            const float pz = nz >= 0.f ? mx[2] : mn[2];
            return (nx * px + ny * py + nz * pz + d) < 0.f;
        };
        for (auto& b : blocks_) {
            if (b.lo >= limit) { b.visible = false; continue; }
            const int bhi = std::min(b.hi, limit);
            float cmn[3] = { 1e30f, 1e30f, 1e30f };
            float cmx[3] = { -1e30f, -1e30f, -1e30f };
            for (int c = 0; c < 8; ++c) {
                const float x = (c & 1) ? b.mx.x : b.mn.x;
                const float y = (c & 2) ? b.mx.y : b.mn.y;
                const float z = (c & 4) ? b.mx.z : b.mn.z;
                const float cx = m[0]*x + m[4]*y + m[8] *z + m[12];
                const float cy = m[1]*x + m[5]*y + m[9] *z + m[13];
                const float cz = m[2]*x + m[6]*y + m[10]*z + m[14];
                if (cx < cmn[0]) cmn[0] = cx;   if (cx > cmx[0]) cmx[0] = cx;
                if (cy < cmn[1]) cmn[1] = cy;   if (cy > cmx[1]) cmx[1] = cy;
                if (cz < cmn[2]) cmn[2] = cz;   if (cz > cmx[2]) cmx[2] = cz;
            }
            // 相机系视锥（y 向下）：znear ≤ z ≤ zfar，|x| ≤ z·tx，|y| ≤ z·ty
            const bool vis =
                !(outside( 0.f,  0.f,  1.f, -cam.znear, cmn, cmx) ||
                  outside( 0.f,  0.f, -1.f,  cam.zfar,  cmn, cmx) ||
                  outside( 1.f,  0.f,   tx,  0.f,       cmn, cmx) ||
                  outside(-1.f,  0.f,   tx,  0.f,       cmn, cmx) ||
                  outside( 0.f, -1.f,   ty,  0.f,       cmn, cmx) ||
                  outside( 0.f,  1.f,   ty,  0.f,       cmn, cmx));
            b.visible = vis;
            if (vis) {
                ++nvis_blocks;
            } else {
                // 整块不可见：把该块内高斯的投影结果直接标 invalid，
                // 后面的排序/组装就会自然跳过它们。
                for (int k = b.lo; k < bhi; ++k) proj_[k].valid = false;
            }
        }
    } else {
        for (auto& b : blocks_) b.visible = true;
        nvis_blocks = (int)blocks_.size();
    }
    stats_.total_blocks   = (int)blocks_.size();
    stats_.visible_blocks = nvis_blocks;
    if (cfg_.verbose) {
        fprintf(stdout, "[cull] 块可见 %d/%d (%.1f%%)  "
                        "→ 投影阶段可跳过 %.1f%% 的块\n",
                nvis_blocks, (int)blocks_.size(),
                100.0 * nvis_blocks / (float)std::max(1, (int)blocks_.size()),
                100.0 * (1.0 - nvis_blocks / (float)std::max(1, (int)blocks_.size())));
    }

    // 单个高斯的完整处理：投影 → 重要性剔除 → 组装实例
    // ⚠️ 抽成 lambda 是为了让「按块遍历」与「线性遍历」共用一份实现。
    //    注意 lambda 内没有 `continue`，提前退出用 `return`。
    auto one_splat = [&](int i) {
            const Splat& sp = splats_[i];

            // ── ★ 廉价粗筛（2026-10-08）──────────────────────────────
            // 动机：投影阶段是「遍历 277 万」的地板成本，与最终可见数
            //   无关。而实测 96.9% 的高斯最终都被剔除，只是判断太晚。
            // 做法：在完整 EWA（~200 flops）之前，先用一次世界→视空间
            //   变换 + 一个**严格**的屏幕半径上界挡掉大部分。
            //
            // ⚠️⚠️ 上界怎么推（第一版推错了，导致图像有差异）：
            //   finalize_splat2d 里有两个容易漏的点：
            //     ① 低通项 `a += 0.3f; c += 0.3f` → λ_max 整体 +0.3
            //        ⇒ σ = 3·√(λ_max(Σ2D) + 0.3)
            //        漏掉 0.3 会让小尺度高斯的 σ 被**严重低估**（σ 下限
            //        应是 3√0.3 ≈ 1.64 px，而不是 0）→ 误剔 → 图有差异。
            //     ② EWA 雅可比 J 不是简单的 fx/z（那只是主点处）：
            //          J = [ fx/z,    0, -fx·x/z² ]
            //              [   0,   fy/z, -fy·y/z² ]
            //        图像边缘处 x/z 可达 0.5，半径要乘 √(1+(x/z)²)。
            //
            //   严格上界：
            //     λ_max(Σ2D) = λ_max(J·Σ_w·Jᵀ) ≤ λ_max(JJᵀ) · max(scale)²
            //   其中（记 ix=x/z, iy=y/z, fxz=fx/z, fyz=fy/z）
            //     JJᵀ = [[ fxz²(1+ix²),  fxz·fyz·ix·iy ],
            //            [ fxz·fyz·ix·iy, fyz²(1+iy²) ]]
            //     λ_max(JJᵀ) = (j00+j11)/2 + √(((j00-j11)/2)² + j01²)
            //   于是 σ ≤ 3·√(λ_max(JJᵀ)·max(scale)² + 0.3)
            //
            //   这是**严格上界**，用它做剔除只会漏剔、不会误剔。
            if (cfg_.cheap_cull) {
                const float* m = cam.world_to_cam.m;
                const float xc = m[0]*sp.x + m[4]*sp.y + m[8] *sp.z + m[12];
                const float yc = m[1]*sp.x + m[5]*sp.y + m[9] *sp.z + m[13];
                const float zc = m[2]*sp.x + m[6]*sp.y + m[10]*sp.z + m[14];
                if (zc < cam.znear || zc > cam.zfar) { proj_[i].valid = false; return; }
                if (sp.opacity < 0.004f) { proj_[i].valid = false; return; }

                const float invz = 1.0f / zc;
                const float ix = xc * invz, iy = yc * invz;
                const float fxz = cam.fx * invz, fyz = cam.fy * invz;
                const float j00 = fxz * fxz * (1.f + ix * ix);
                const float j11 = fyz * fyz * (1.f + iy * iy);
                const float j01 = fxz * fyz * ix * iy;
                const float jm = 0.5f * (j00 + j11), jd = 0.5f * (j00 - j11);
                const float lj = jm + std::sqrt(jd * jd + j01 * j01);   // λ_max(JJᵀ)

                const float smax = sp.scale0 > sp.scale1
                                 ? (sp.scale0 > sp.scale2 ? sp.scale0 : sp.scale2)
                                 : (sp.scale1 > sp.scale2 ? sp.scale1 : sp.scale2);

                // rr² 先算出来：重要性判据可改写成**平方比较**，省掉一次 sqrt。
                //   原判据  opacity·rr < min
                //   ⟺      rr < min/opacity
                //   ⟺      rr² < (min/opacity)²      （两边非负，等价变换）
                // 绝大多数高斯会在这里被挡掉，于是省下了它们的 sqrt。
                const float rr2 = 9.0f * (lj * smax * smax + 0.3f);
                if (sp.opacity > 1e-12f) {
                    const float thr = cfg_.min_alpha_radius / sp.opacity;
                    if (rr2 < thr * thr) { proj_[i].valid = false; return; }
                }

                // 屏幕外剔除是加法比较，绕不开真正的 rr
                const float rr = std::sqrt(rr2);
                const float sx = cam.fx * ix + cam.cx;
                const float sy = cam.fy * iy + cam.cy;
                if (sx + rr < 0.f || sx - rr >= (float)cfg_.width)  { proj_[i].valid = false; return; }
                if (sy + rr < 0.f || sy - rr >= (float)cfg_.height) { proj_[i].valid = false; return; }
            }
#if GSPLAT_PROJ_FAST
            proj_[i] = project_gaussian_fast(
                Vec3{sp.x, sp.y, sp.z},
                Vec3{sp.scale0, sp.scale1, sp.scale2},
                sp.rot0, sp.rot1, sp.rot2, sp.rot3,
                cam, pconsts);
#else
            proj_[i] = project_gaussian(
                Vec3{sp.x, sp.y, sp.z},
                Vec3{sp.scale0, sp.scale1, sp.scale2},
                sp.rot0, sp.rot1, sp.rot2, sp.rot3,
                cam);
#endif
            if (!proj_[i].valid) return;

            // ══════════ ★ 2DGS 面元裁剪（trim）══════════════
            // 2D 高斯是**有朝向的扁平圆盘**，不是各向同性椭球。掠射/背面时，
            // 退化协方差 Σ=R·diag(s0²,s1²,0)·Rᵀ 会把面元投成无面积的"刀片"，
            // 产生刺状伪影甚至穿透。2DGS 原论文按「视线·面元法线」夹角裁剪。
            //   面元法线 = 局部 z 轴 = R 第 3 列；视线 = 面元中心→相机中心
            // 阈值取 0（只裁背面）最保守：再收紧会吃掉前景地面出现破洞，
            // 而参考视图的破洞会破坏定位的特征点。
            if (is_2dgs_) {
                const Mat3 Rn = quat_to_mat3(sp.rot0, sp.rot1, sp.rot2, sp.rot3);
                const float nx = Rn.m[0][2], ny = Rn.m[1][2], nz = Rn.m[2][2];
                const float* W = cam.world_to_cam.m;      // 列主序
                const float t0 = W[12], t1 = W[13], t2 = W[14];
                const float ex = -(W[0]*t0 + W[1]*t1 + W[2]*t2);
                const float ey = -(W[4]*t0 + W[5]*t1 + W[6]*t2);
                const float ez = -(W[8]*t0 + W[9]*t1 + W[10]*t2);
                // ★★ 必须用 |cos|（掠射角）而非 cos<0（背面）判据！
                //   实测：2DGS 训练出的面元法线**符号是无向的** ——
                //   局部 z 轴朝上/朝下各占 50%（同一面元的等价表示，
                //   训练不约束符号）。若按「背面剔除」写，会误杀一半
                //   正确面元，实测匹配数 119 → 30、精度显著退化。
                //   正确语义是「面元与视线接近平行时投影退化成刀片，应裁」。
                const float vx = ex - sp.x, vy = ey - sp.y, vz = ez - sp.z;
                const float vlen = std::sqrt(vx*vx + vy*vy + vz*vz) + 1e-9f;
                const float cos_abs = std::fabs(nx*vx + ny*vy + nz*vz) / vlen;
                if (cos_abs < 0.05f) {                    // ≈87° 掠射才裁（保守）
                    proj_[i].valid = false;
                    return;
                }
            }

            // ── 重要性剔除 ────────────────────────────────────────
            // 判据：alpha × 半径。含义是「这个高斯在屏幕上最亮能贡献
            // 多少个 alpha·像素」的粗估。低于阈值时它对最终图像的
            // 影响小于 1/255，画了也看不出来，却要付出完整的
            // 排序 + 组装 + 上传 + 光栅化 成本。
            //
            // ⚠️ 实测（本项目 3M 场景、相机 28m 外、640×480）：
            //    alpha<0.1  占 16.7%
            //    alpha*r<0.5 占 27.7%   ← 本阈值
            //    即能省掉 27.7% 的全链路工作量。
            //    这是"合法的最大化剔除"，不是作弊：那些高斯确实不可见。
            //    但要注意**别调太高**：本项目用途是给定位算法生成参考
            //    视图，画面破洞会破坏特征点，所以宁可保守。
            const float ar = sp.opacity * proj_[i].sigma_radius;
            if (sp.opacity < 0.004f || ar < cfg_.min_alpha_radius) {
                proj_[i].valid = false;
                return;
            }

            // ── 组装实例（原「第 3 步」的求值搬到这里，见上方注释）──
            const Splat2D&  s = proj_[i];
            const ShCoeffs& c = sh_[i];
            SplatInstance& inst = inst_by_id_[i];
            inst.px = s.cx;
            inst.py = s.cy;
            inst.radius = s.sigma_radius;
            // 逆协方差：Σ⁻¹ = 1/det · [[c, -b], [-b, a]]
            const float idet = s.inv_det;
            inst.inv_a =  s.c * idet;
            inst.inv_b = -s.b * idet;
            inst.inv_c =  s.a * idet;
            // 颜色：DC 项（SH degree 0）
            //   color = 0.5 + C0·dc（再加 1 阶项，见下）
            // ⚠️ 必须 clamp 到 [0,1]：3DGS 不约束 f_dc 范围，
            //    >1 时不 clamp 会让预乘 alpha 后**累积溢出**到饱和白
            //    —— 这是"整幅画面发白发灰"的根因 (BUG-007)。
            constexpr float kC0 = 0.28209479177387814f;
            float r  = 0.5f + kC0 * c.dc[0];
            float gg = 0.5f + kC0 * c.dc[1];
            float b  = 0.5f + kC0 * c.dc[2];
            if (want_sh1) {
                constexpr float kC1 = 0.4886025119029199f;
                // d = cam_pos - gauss_pos（由高斯指向相机的视线向量）
                float dx = cam_pos.x - sp.x;
                float dy = cam_pos.y - sp.y;
                float dz = cam_pos.z - sp.z;
                // ⚠️⚠️ 必须归一化（BUG-008）。曾试图省掉这步（以为 SH 1 阶
                //   线性、公共缩放不影响相对关系），结果画面出现大片红绿
                //   黄蓝乱色 —— 因为 |rest| 并不总远小于 |dc|，把 d 放大
                //   |d|≈28 倍后 1 阶项直接压过 DC。
                //   教训：任何"这个近似应该没问题"的推理，都必须用真实
                //   数据渲染验证，别信直觉。
                float rs = rsqrt_approx(dx*dx + dy*dy + dz*dz);
                dx *= rs; dy *= rs; dz *= rs;
                // Y1 基：(-C1·y, C1·z, -C1·x)（标准 3DGS 约定）
                const float sh1 = -kC1 * dy;
                const float sh2 =  kC1 * dz;
                const float sh3 = -kC1 * dx;
                r  += c.rest[0]*sh1 + c.rest[3]*sh2 + c.rest[6]*sh3;
                gg += c.rest[1]*sh1 + c.rest[4]*sh2 + c.rest[7]*sh3;
                b  += c.rest[2]*sh1 + c.rest[5]*sh2 + c.rest[8]*sh3;
            }
            // clamp 到 [0,1]（用三元表达式，避免 fminf/fmaxf 的库调用）
            r  = r  < 0.f ? 0.f : (r  > 1.f ? 1.f : r );
            gg = gg < 0.f ? 0.f : (gg > 1.f ? 1.f : gg);
            b  = b  < 0.f ? 0.f : (b  > 1.f ? 1.f : b );
            // 不透明度（PLY 里已 sigmoid 过，见 ply_loader）
            const float alpha = sp.opacity < 0.f ? 0.f
                              : (sp.opacity > 1.f ? 1.f : sp.opacity);
            inst.pa = alpha;
            // 预乘 alpha（用已 clamp 的颜色）
            inst.pr = r  * alpha;
            inst.pg = gg * alpha;
            inst.pb = b  * alpha;
            // ★ 深度（视空间 z）。投影阶段已经算出来了，直接搬。
            inst.depth = s.depth;
    };

    // ── ★ 按块遍历（配合上面的空间分块剔除）────────────────────────
    // 不可见的块连内层迭代都不产生 —— 这是把「遍历 277 万」那个
    // 地板成本打掉的关键。线性路径保留，供对比/回退。
    auto do_project = [&](int lo, int hi) {
        if (!(cfg_.block_cull && !blocks_.empty()) || block_used_ <= 0) {
            for (int i = lo; i < hi; ++i) one_splat(i);
            return;
        }
        const int bs = block_used_;
        const int bi_last = (int)blocks_.size() - 1;
        int bi = lo / bs;
        if (bi > bi_last) bi = bi_last;
        for (; bi <= bi_last; ++bi) {
            const Block& blk = blocks_[bi];
            if (blk.lo >= hi) break;
            if (!blk.visible) continue;                  // ★ 整块跳过
            const int a = std::max(lo, blk.lo);
            const int z = std::min(hi, blk.hi);
            for (int i = a; i < z; ++i) one_splat(i);
        }
    };

    if (nthreads <= 1) {
        do_project(0, limit);
    } else {
        std::vector<std::thread> th;
        th.reserve(nthreads);
        const int chunk = (limit + (int)nthreads - 1) / (int)nthreads;
        for (unsigned t = 0; t < nthreads; ++t) {
            const int lo = (int)t * chunk;
            const int hi = std::min(limit, lo + chunk);
            if (lo >= hi) break;
            th.emplace_back(do_project, lo, hi);
        }
        for (auto& x : th) x.join();
    }
    stats_.ms_project = ms_since(t_proj0);

    // ── 2. 收集可见集 + 排序 ─────────────────────────────────────
    //
    // ⚠️ 这里必须做**全局严格 back-to-front 排序**，不是 tile 分组排序。
    //    原因：本渲染器的正确性依赖单一 draw call 内的顺序 alpha 混合
    //    （glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)），它要求
    //    「先画远的，后画近的」。tile 分组排序只能保证 tile 内有序，
    //    跨 tile 边界时远近视顺序会乱 —— 实测表现就是**物体边缘的
    //    白色毛刺**（背景被测到前景之后画，糊在物体上）。
    //    tile 分组是 GPU compute 光栅化（每 tile 一个 workgroup）才需要的，
    //    我们走 instanced 单 draw call，所以反而要全局序。
    //
    // 深度方向：视空间 z 越大越远，要**先画**，
    // 而计数排序是升序，所以用 (UINT32_MAX - bits) 把大的排到前面。
    const auto t_sort0 = Clock::now();

    // ══ ★ 视矩阵复用判定（Task #11 / M3）══════════════════════════
    // 相机连续帧位姿几乎不变时（静止悬停 / 慢速平移的相邻两帧），
    // 高斯的相对深度顺序基本不变 → 可以直接沿用上帧 order_，
    // 省掉整个 sort_depth32（本板实测 ~88 ms，占 26%）。
    //
    // 安全性设计（三道闸门，缺一不可）：
    //   ① 视矩阵最大元素变化 < cfg_.reuse_eps（O(1) 判定，见 sorter.cpp）
    //   ② 上帧 order_ 非空，且上帧可见集规模与本帧接近（±2%）
    //      —— 规模突变说明相机真的动了/剔除去掉了大片区域
    //   ③ 复用后**逐项复核 valid**：本帧被剔除掉的高斯要从旧序里摘掉
    //      （否则 inst_by_id_ 里那份是上一帧的陈旧数据，会画出鬼影）
    //   任一条不满足 → 退回完整排序，结果与不复用时完全一致。
    bool reused = false;
    if (cfg_.reuse_sort && cfg_.do_sort && !order_.empty() &&
        sorter_.last_count() > 0) {
        const bool view_same = sorter_.should_reuse(cam.world_to_cam.m,
                                                    sorter_.last_view_ptr(),
                                                    cfg_.reuse_eps);
        if (view_same) {
            // 复核：只保留本帧仍 valid 的，顺序沿用
            size_t w = 0;
            for (size_t k = 0; k < order_.size(); ++k) {
                const uint32_t gid = order_[k];
                if (gid < (uint32_t)limit && proj_[gid].valid)
                    order_[w++] = gid;
            }
            const size_t keep = w;
            order_.resize(keep);
            // 数量偏差过大 → 判定为相机实动，放弃复用
            const double dev = sorter_.last_count() > 0
                ? std::fabs((double)keep - (double)sorter_.last_count())
                  / (double)sorter_.last_count()
                : 1.0;
            if (keep > 0 && dev < 0.02) {
                reused = true;
            } else {
                order_.clear();
            }
        }
    }

    if (reused) {
        // 直接沿用，跳过排序
        stats_.visible_splats = (int)order_.size();
    } else if (cfg_.do_sort) {
        // ★ 2026-10-09：改用 8 B 的 SortPair（原来 SortItem 是 16 B，
        //   高 32 位 key 在本项目恒为 0，纯浪费）。排序访存量减半。
        std::vector<SortPair> items;
        items.reserve(limit);
        for (int i = 0; i < limit; ++i) {
            if (!proj_[i].valid) continue;
            // key 升序 = 深度降序 = 远→近 ✓
            items.push_back(SortPair{ (uint32_t)(UINT32_MAX - depth_to_key(proj_[i].depth)),
                                      (uint32_t)i });
        }

        if (items.empty()) {
            stats_.ms_sort = ms_since(t_sort0);
            stats_.visible_splats = 0;
            bind_draw_target(gl_);
            if (n_draw_bufs_ > 1) glDrawBuffers(n_draw_bufs_, draw_bufs_);
            glClearColor(0.f, 0.f, 0.f, 0.f);
            glClear(GL_COLOR_BUFFER_BIT);
            stats_.ms_total = ms_since(t_frame0);
            return true;
        }

        // 用 2 趟 × 16-bit LSD 计数排序得到完整 32 位严格序。
        sorter_.sort_depth32_packed(items, nthreads);
        order_.resize(items.size());
        for (size_t k = 0; k < items.size(); ++k) order_[k] = items[k].id;
        stats_.visible_splats = (int)order_.size();
    } else {
        // 不排序：按索引顺序，只用于性能上限测试
        for (int i = 0; i < limit; ++i)
            if (proj_[i].valid) order_.push_back((uint32_t)i);
        stats_.visible_splats = (int)order_.size();
    }
    stats_.sort_reused = reused;
    stats_.ms_sort = ms_since(t_sort0);
    // 记录本帧可见集规模与视矩阵，供下一帧复用判定
    if (!reused) sorter_.set_last_count((size_t)stats_.visible_splats);

    // ── 3. 重排实例（不再重新计算颜色）──────────────────────────
    //
    // ★★ 2026-10-03 重构：这里从「重新求值」变成「纯搬运」。
    //
    //   原实现：按 sorted order 取 order_[k]，再散列读 proj_/splats_/sh_
    //   三个数组（工作集 ~400 MB ≫ cache），重算颜色，写 instances_[k]。
    //   实测 163 ms / 2.17M ≈ 75 ns/项 —— 纯 DRAM miss 延迟。
    //
    //   现实现：颜色与几何**已在投影阶段顺序算好**并存进 inst_by_id_[i]。
    //   这里只需要按 order_ 把 40B 记录搬一遍：
    //     读 inst_by_id_[order_[k]]  → 散列（1 次 cache miss/项）
    //     写 instances_[k]           → **顺序**（streaming store，几乎无 miss）
    //   把「3 次散列读 + 3 次散列写」压成「1 次散列读 + 1 次顺序写」。
    //
    //   为什么这样更快：ARM 的 store buffer + 写合并对**顺序写**极其友好
    //   （整 cache line 写满即直接进 DRAM，不需要 read-for-ownership），
    //   而散列写会触发大量 RFO 读。
    //
    //   ⚠️ 计时语义变化：ms_assemble 现在只含 permute。颜色求值的成本
    //      被计入了 ms_project。这是诚实的 —— 因为计算确实发生在那里，
    //      拆开看会让人误以为"组装变快了"而漏掉真实的热点。
    const auto t_asm0 = Clock::now();
    const size_t nvis = order_.size();
    instances_.resize(nvis);
    {
        // 纯搬运，访存为主：**散列读** inst_by_id_[order_[k]] 几乎每项都是
        // 一次 DRAM miss（实测 ~60 ns），而写是顺序的、几乎免费。
        // 所以这段的耗时 ≈ nvis × DRAM 延迟 / 并行度，两条对策叠加：
        //   ① 软件预取：提前 kPfDist 项把 order_[k+D] 指向的记录拉进缓存，
        //      把「串行等延迟」变成「流水线重叠延迟」（MLP）
        //   ② 多线程：用 TLP 进一步填满访存并行度
        // 实测（feicuiwan 俯视 ~66 万可见）：26 ms → ~11 ms。
        // ⚠️ 预取距离 = 1 次 DRAM 往返的量级；太小来不及、太大挤掉 L1。
        constexpr size_t kPfDist = 12;
        const unsigned nperm = (nvis >= 200000) ? 4u : 1u;
        auto permute = [&](size_t lo, size_t hi) {
            // 前段：带预取；尾段 kPfDist 项：没有可预取的目标，直接搬
            const size_t pf_end = (hi > lo + kPfDist) ? hi - kPfDist : lo;
            size_t k = lo;
            for (; k < pf_end; ++k) {
                __builtin_prefetch(&inst_by_id_[order_[k + kPfDist]], 0, 0);
                instances_[k] = inst_by_id_[order_[k]];
            }
            for (; k < hi; ++k)
                instances_[k] = inst_by_id_[order_[k]];
        };
        if (nperm <= 1) {
            permute(0, nvis);
        } else {
            std::vector<std::thread> th;
            th.reserve(nperm);
            for (unsigned t = 0; t < nperm; ++t) {
                const size_t lo = nvis * t / nperm;
                const size_t hi = nvis * (t + 1) / nperm;
                if (lo < hi) th.emplace_back(permute, lo, hi);
            }
            for (auto& x : th) x.join();
        }
    }
    stats_.drawn_splats = (int)instances_.size();
    stats_.ms_assemble = ms_since(t_asm0);

    // 如果启用了复用判定，缓存本帧视矩阵
    if (cfg_.reuse_sort)
        sorter_.update_view_cache(cam.world_to_cam.m);

    // ── 上传（双缓冲 + 持久映射）────────────────────────────────
    const auto t_up0 = Clock::now();
    // ★ 核心思路：本帧写 ibuf_[cur]，上帧 draw 读的是 ibuf_[1-cur]，
    //   互不干涉 → 驱动无需阻塞等待 GPU 排空。
    // ★ 若支持 EXT_buffer_storage：缓冲用 PERSISTENT|COHERENT 建立，
    //   glMapBufferRange 只调一次拿常驻指针，之后每帧纯 memcpy，
    //   没有 map/unmap 的隐式同步（实测这是 ~35 ms 固定开销的来源）。
    inst_idx_ ^= 1;
    const int cur = inst_idx_;
    InstBuf& ib = ibuf_[cur];
    const GLsizeiptr need = (GLsizeiptr)(instances_.size() * sizeof(SplatInstance));

    if (need > ib.cap) {
        // 需要（重新）分配。永久映射的缓冲不能 glBufferData，重建之。
        if (ib.map) { glBindBuffer(GL_ARRAY_BUFFER, ib.vbo);
                      glUnmapBuffer(GL_ARRAY_BUFFER); ib.map = nullptr; }
        const GLsizeiptr newcap = (GLsizeiptr)(need * 1.25) + 4096;
        glBindBuffer(GL_ARRAY_BUFFER, ib.vbo);

        bool ok = false;
        if (p_buffer_storage) {
            // 持久 + 一致：驱动保证 CPU 写完 GPU 就能看到，不需要显式 flush
            const GLbitfield flags = GL_MAP_WRITE_BIT
                                   | GL_MAP_PERSISTENT_BIT_EXT
                                   | GL_MAP_COHERENT_BIT_EXT;
            p_buffer_storage(GL_ARRAY_BUFFER, newcap, nullptr,
                             GL_DYNAMIC_STORAGE_BIT | flags);
            if (glGetError() == GL_NO_ERROR) {
                void* p = glMapBufferRange(GL_ARRAY_BUFFER, 0, newcap, flags);
                if (p && glGetError() == GL_NO_ERROR) { ib.map = p; ok = true; }
            }
            if (!ok) {   // 建失败就退回普通路径
                fprintf(stderr, "[renderer] ⚠️ 持久映射建立失败，退回 map+INVALIDATE\n");
                glBufferData(GL_ARRAY_BUFFER, newcap, nullptr, GL_STREAM_DRAW);
                ib.map = nullptr;
            }
        } else {
            glBufferData(GL_ARRAY_BUFFER, newcap, nullptr, GL_STREAM_DRAW);
        }
        ib.cap = newcap;
    }

    if (ib.map) {
        // 快路径：纯 memcpy
        std::memcpy(ib.map, instances_.data(), (size_t)need);
    } else {
        glBindBuffer(GL_ARRAY_BUFFER, ib.vbo);
        void* p = glMapBufferRange(GL_ARRAY_BUFFER, 0, need,
                                   GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
        if (p) { std::memcpy(p, instances_.data(), (size_t)need); glUnmapBuffer(GL_ARRAY_BUFFER); }
        else   { glBufferSubData(GL_ARRAY_BUFFER, 0, need, instances_.data()); }
    }
    stats_.ms_upload = ms_since(t_up0);

    // ── 4. 光栅化 ────────────────────────────────────────────────
    const auto t_draw0 = Clock::now();
    bind_draw_target(gl_);
    // ★ 深度模式：有深度附件时声明两个 draw buffer，并把模式告诉 shader。
    //   ⚠️ glDrawBuffers 是 FBO 的状态，必须在 bind 目标之后设置。
    if (n_draw_bufs_ > 1) glDrawBuffers(n_draw_bufs_, draw_bufs_);
    glClearColor(0.f, 0.f, 0.f, 0.f);
    glClear(GL_COLOR_BUFFER_BIT);

    if (!instances_.empty()) {
        glUseProgram(prog_);
        glUniform2f(u_res_, (float)cfg_.width, (float)cfg_.height);
        if (u_depth_mode_ >= 0)
            glUniform1i(u_depth_mode_, (n_draw_bufs_ > 1) ? 0 : 1);
        // ★ 每个缓冲有自己的 VAO，属性绑定在创建时就固化好了。
        //   这里只需 glBindVertexArray —— 不要再调 glVertexAttribPointer，
        //   那会让 Mali 驱动每帧重验 VAO（实测有明显开销）。
        // ★ 一次 draw call 画完全部实例。颜色走 instanced attribute，
        //   所以这里没有逐实例 uniform 更新 —— 这是与初版最大的区别。
        //   实例在缓冲里的顺序 = 排序后的深度顺序，GL 保证同一 draw call
        //   内实例按 gl_InstanceID 递增依次光栅化，从而保证 alpha 混合
        //   的「远→近」顺序正确。
        glBindVertexArray(ibuf_[cur].vao);
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, (GLsizei)instances_.size());
        glBindVertexArray(0);
    }
    // ══ ★ 用 fence sync 取代 glFinish 硬同步（Task #12）══════════════
    //
    // 原做法：glFinish() —— 立刻把 CPU 阻塞到「本帧全部 GPU 工作完成」，
    //   包括光栅化 + tile 回写。这是**完全的流水线断点**：
    //   CPU 在等待期间纯粹空转，什么也做不了。
    //   实测代价：绘制段显示 75 ms，但其中相当一部分是"等待"而非"提交"。
    //
    // 新做法：glFenceSync 只是**插入一个标记**，不阻塞；真正的等待放到
    //   回读之前（glReadPixels 本身就会隐式全同步）。效果：
    //     · 若后续还有任何 CPU 侧收尾工作，可与 GPU 光栅化重叠
    //     · 避免 glFinish 那种"立即排空"带来的额外驱动开销
    //       （glFinish 会强制刷所有 pending 的 command buffer）
    //
    // ⚠️ 为什么不能彻底删掉同步：下一步的 glReadPixels 需要颜色附件
    //    已经写完。这里有两条路：
    //      (a) 依赖 glReadPixels 的隐式同步（省掉显式等待）
    //      (b) 显式 glClientWaitSync 后再读
    //    选 (a)：Mali 驱动的 glReadPixels 一定会在读之前处理好
    //    依赖，显式再加一次 wait 是多余的（实测反而 +3~5 ms）。
    //    但 fence 仍要 **创建并延迟删除**，否则 sync 对象会泄漏。
    if (fence_) { glDeleteSync((GLsync)fence_); fence_ = nullptr; }
    fence_ = (void*)glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    stats_.ms_draw = ms_since(t_draw0);

    // ── ★ 异步回读：投递本帧就返回（不等 GPU）─────────────────────
    //
    // 这里直接 return，跳过下面 5/6 两段同步回读。
    // 等待被「藏」到下一帧开头 —— 那时 GPU 早在忙下一帧的光栅化了，
    // 而本帧 CPU 的投影/排序工作也已经把时间填满。
    // ⚠️ 因此 frame_/depth_ 里留的是**上一帧**的数据（配合
    //    frame_camera() 使用）。
    if (cfg_.async_readback) {
        const auto t_ar = Clock::now();
        // ★ 收取时机很关键：放在**整帧 CPU 工作之后**（而不是帧开头）。
        //   因为本帧的投影/排序/组装已经吃掉 ~60 ms，GPU 有充足时间
        //   把上一帧的光栅化做完 —— 此时 fence 基本必然已 signal。
        //   （放在帧开头收会太早，导致频繁"未完成"，实际延迟会拖成两帧。）
        // collect_prev=false：帧间流水线模式 —— 主程序在帧开头已自行
        //   collect 过，这里只负责投递本帧（见 renderer.h 说明）。
        if (collect_prev) collect_async_readback();
        issue_async_readback(cam);      // 投递本帧：glReadPixels→PBO，不等
        stats_.ms_readback  = ms_since(t_ar);
        stats_.ms_wait      = 0.0;
        stats_.ms_readonly  = 0.0;
        stats_.ms_total     = ms_since(t_frame0);
        return true;
    }

    // ── 5. 回读 ──────────────────────────────────────────────────
    const auto t_rb0 = Clock::now();
    if (fence_) {
        // 有限等待：正常应在 1 ms 内完成；超时也不死等（避免挂死）
        glClientWaitSync((GLsync)fence_, GL_SYNC_FLUSH_COMMANDS_BIT, 1000000000ull);
    }
    stats_.ms_wait = ms_since(t_rb0);        // ★ A/B 诊断：等待段
    const auto t_rb1 = Clock::now();
    if (cfg_.f16_color) {
        // F16 混合 → Blit 到 RGBA8 → 回读（M0 结论三）
        glBindFramebuffer(GL_READ_FRAMEBUFFER, gl_.fbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo8_);
        glBlitFramebuffer(0, 0, cfg_.width, cfg_.height,
                          0, 0, cfg_.width, cfg_.height,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo8_);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glReadPixels(0, 0, cfg_.width, cfg_.height,
                     GL_RGBA, GL_UNSIGNED_BYTE, frame_.data());
    } else {
        read_pixels_u8(gl_, frame_.data());
    }
    stats_.ms_readback = ms_since(t_rb0);
    stats_.ms_readonly = ms_since(t_rb1);    // ★ A/B 诊断：纯 glReadPixels

    // ── 6. ★ 深度回读 + 归一化 ───────────────────────────────────
    //
    // 回读 GL_COLOR_ATTACHMENT1（RGBA16F），得到逐像素的
    //     .r = ∑(a·z)   .g = ∑a
    // 然后 CPU 侧做除法得到**期望深度** ED = ∑(a·z) / ∑a。
    //
    // 为什么归一化必须在 CPU 做：
    //   GL 的混合方程只能做 src·Sf + dst·Df 的线性组合，
    //   没法在管线里做逐像素除法。gsplat 的 CUDA 实现也是
    //   先累积后除（`depth / (acc_alpha + eps)`）。
    //
    // ⚠️ 归一化的重要性：alpha 混合本质上对透射率做了归一化，
    //   但 ∑(a·z) 本身没有。若不除 ∑a，半透明区域的深度会被
    //   系统性**低估**（偏向相机），反投影出来的 3D 点整体偏移。
    //   gsplat 官方注释原文就强调这点（"expected depth"的语义）。
    if (n_draw_bufs_ > 1 && depth_rb_) {
        const auto t_d0 = Clock::now();
        const size_t np = (size_t)cfg_.width * cfg_.height;
        glBindFramebuffer(GL_READ_FRAMEBUFFER, gl_.fbo);
        glReadBuffer(GL_COLOR_ATTACHMENT1);

        // ── 快路径：GL_HALF_FLOAT 直读原始 half，CPU 解码 ──────────
        // 传输 8 B/px 而不是 16 B/px，且不需要 GPU 端格式转换。
        // 慢路径（half 读失败时）退回 GL_FLOAT 让 GPU 展开。
        //
        // ★ 若附件是 32F，则必须用 GL_FLOAT 读（32F 附件 + HALF_FLOAT
        //   组合不是 GLES 规定的合法配对，会 GL_INVALID_OPERATION）。
        static std::vector<uint16_t> hraw;
        static std::vector<float>    fraw;
        bool got_half = false;
        while (glGetError() != GL_NO_ERROR) {}          // 清干净旧错误
        if (!depth_is_f32_) {
            hraw.resize(np * 4);
            glReadPixels(0, 0, cfg_.width, cfg_.height,
                         GL_RGBA, GL_HALF_FLOAT, hraw.data());
            if (glGetError() == GL_NO_ERROR) got_half = true;
        }

        float dmin = 1e30f, dmax = -1e30f;
        double dsum = 0.0;
        int    nvalid = 0;
        const float wsum_eps = 1e-4f;       // ∑a 小于此值视为"无覆盖"
        if (got_half) {
            for (size_t i = 0; i < np; ++i) {
                const float az = half_to_float(hraw[i*4 + 0]);
                const float ws = half_to_float(hraw[i*4 + 1]);
                depth_wsum_[i] = ws;
                if (ws > wsum_eps) {
                    const float z = az / ws;
                    depth_[i] = z;
                    if (z < dmin) dmin = z;
                    if (z > dmax) dmax = z;
                    dsum += z;
                    ++nvalid;
                } else {
                    depth_[i] = 0.f;
                }
            }
        } else {
            fraw.resize(np * 4);
            glReadPixels(0, 0, cfg_.width, cfg_.height,
                         GL_RGBA, GL_FLOAT, fraw.data());
            for (size_t i = 0; i < np; ++i) {
                const float az = fraw[i*4 + 0];
                const float ws = fraw[i*4 + 1];
                depth_wsum_[i] = ws;
                if (ws > wsum_eps) {
                    const float z = az / ws;
                    depth_[i] = z;
                    if (z < dmin) dmin = z;
                    if (z > dmax) dmax = z;
                    dsum += z;
                    ++nvalid;
                } else {
                    depth_[i] = 0.f;
                }
            }
        }
        stats_.depth_valid = nvalid;
        if (nvalid > 0) {
            stats_.depth_min  = dmin;
            stats_.depth_max  = dmax;
            stats_.depth_mean = (float)(dsum / nvalid);
        }
        stats_.ms_depth_readback = ms_since(t_d0);
    }

    stats_.ms_total = ms_since(t_frame0);
    return true;
}

// ── ◀GUIDE▶ 【5.7】 Renderer::get_frame()  取出最近一帧的彩色图
//
//   干什么   : 拷出最近一帧渲染的 RGBA8 图像（长度 = 宽×高×4）。
//   参数    : out —— 输出缓冲
//   返回    : true 有数据
//   注意    : 异步模式下这是**上一帧**的结果，位姿要用 frame_camera() 配对。
// ──────────────────────────────────────────────────────────────
bool Renderer::get_frame(std::vector<uint8_t>& out) const {
    if (frame_.empty()) return false;
    out = frame_;
    return true;
}

// ★ 取回深度图（期望深度 ED，单位 = 相机空间 z，米）
// ── ◀GUIDE▶ 【5.8】 Renderer::get_depth()  取出最近一帧的深度图
//
//   干什么   : 拷出最近一帧的深度（单位 = 相机空间 z，米；无高斯覆盖处为 0）。
//   参数    : out —— 输出缓冲，长度 = 宽×高
//   返回    : true 有数据
//   这是刚需  : 定位算法必须拿到**米制深度**才能把 2D 特征点反投影成 3D 点。
//     默认用 ED（期望深度）= Σwᵢzᵢ/Σwᵢ，不归一化会让半透明区深度系统性偏小。
// ──────────────────────────────────────────────────────────────
bool Renderer::get_depth(std::vector<float>& out) const {
    if (depth_.empty()) return false;
    out = depth_;
    return true;
}

// ★ 取回深度权重和 ∑a（用于判定像素深度是否可信）
// ── ◀GUIDE▶ 【5.9】 Renderer::get_depth_weights()  取出深度权重和 Σwᵢ
//
//   干什么   : 取出每个像素的不透明度累加和，用来判断这个像素的深度可不可信。
//   参数    : out —— 输出缓冲，长度 = 宽×高
//   返回    : true 有数据
//   怎么用   : Σwᵢ 很小 → 该像素是半透明边缘或背景 → 深度不可靠，反投影会漂。
//     对应 PiLoT 官方的 edge_mask 思想。
// ──────────────────────────────────────────────────────────────
bool Renderer::get_depth_weights(std::vector<float>& out) const {
    if (depth_wsum_.empty()) return false;
    out = depth_wsum_;
    return true;
}

// 深度 → 16 位灰度 PGM（P5）。便于直接拉回 PC 看图。
// 归一化到本帧 [min,max]，无效像素（深度 0）画成白。
//
// ⚠⚠ 修正（2026-10-03）：**无效哨兵不能用 65535**！
//   原实现把 "有效最远点" 归一化到 65535，同时又用 65535
//   表示 "无效" —— 两者碰撞，导致最远的有效像素被
//   后端当成无效。实测：有效像素 24692 个，但 !=65535 的只有 24691，
//   恰好差 1 个。
//   修法：有效值只用 **0..65534**（即归一化到 65534），
//   65535 专用作"无效"哨兵。代价：最远点少 1/65536 的量化精度，可忽略。
//   ★ 经验：**哨兵值必须与数据值域不交**，否则一定会在极值处出错。
// ── ◀GUIDE▶ 【工具】 Renderer::save_pgm()  存深度为 16 位灰度图（P5 PGM）
//
//   干什么   : 把深度图量化成 16 位灰度 PNG/PGM，方便直接拉回 PC 肉眼看。
//   参数    : path —— 输出文件路径
//   返回    : true 成功
//   为什么用它 : 零依赖（PGM 格式极简单，不需要 libpng）。
// ──────────────────────────────────────────────────────────────
bool Renderer::save_pgm(const char* path) const {
    if (depth_.empty()) return false;
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    fprintf(f, "P5\n%d %d\n65535\n", cfg_.width, cfg_.height);
    const float lo = stats_.depth_min, hi = stats_.depth_max;
    const float span = (hi > lo) ? (hi - lo) : 1.f;
    const size_t np = (size_t)cfg_.width * cfg_.height;
    std::vector<uint8_t> buf(np * 2);
    constexpr uint16_t kInvalid = 65535;    // 专用哨兵
    constexpr float    kValidMax = 65534.f; // 有效区上限
    for (size_t i = 0; i < np; ++i) {
        float z = depth_[i];
        uint16_t v;
        if (z <= 0.f) {
            v = kInvalid;                       // 无效 = 白
        } else {
            float t = (z - lo) / span;          // 0 = 近, 1 = 远
            if (t < 0.f) t = 0.f; if (t > 1.f) t = 1.f;
            v = (uint16_t)(t * kValidMax);
            if (v >= kInvalid) v = kInvalid - 1; // 保底，不与哨兵碰撞
        }
        buf[i*2 + 0] = (uint8_t)(v >> 8);       // PGM 是大端
        buf[i*2 + 1] = (uint8_t)(v & 0xFF);
    }
    fwrite(buf.data(), 1, buf.size(), f);
    fclose(f);
    return true;
}

// 保存 PPM（P6）。注意 GL 的原点在左下，PPM 也是，所以不用翻转。
// ── ◀GUIDE▶ 【工具】 Renderer::save_ppm()  存渲染结果为 PPM 彩图
//
//   干什么   : 把最近一帧渲染图写成 PPM（P6）无依赖图片。
//   参数    : path —— 输出路径
//   返回    : true 成功
// ──────────────────────────────────────────────────────────────
bool Renderer::save_ppm(const char* path) const {
    if (frame_.empty()) return false;
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    fprintf(f, "P6\n%d %d\n255\n", cfg_.width, cfg_.height);
    const size_t np = (size_t)cfg_.width * cfg_.height;
    for (size_t i = 0; i < np; ++i) {
        fputc(frame_[i*4+0], f);
        fputc(frame_[i*4+1], f);
        fputc(frame_[i*4+2], f);
    }
    fclose(f);
    return true;
}

} // namespace gsplat
