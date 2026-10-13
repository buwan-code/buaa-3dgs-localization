// offscreen.cpp — EGL 离屏渲染上下文（224 行）
//
// ══ 本文件结构 ═══════════════════════════════════════════════════════
//   gl_version_string       ~9    查询 GL 版本
//   OffscreenTarget::destroy ~14
//   create_offscreen        ~23   ★EGL 初始化：display/config/context/surface
//   compile                 ~129  GLSL 编译（带错误日志）
//   build_program           ~148  链接 program
//   read_pixels / _u8       ~189 / ~212  glReadPixels 回读
//   bind_draw_target        ~219  绑定 FBO
//
// ══ ⚠️⚠️ 必须知道的两件事 ═════════════════════════════════════════════
//   1. **运行前必须 export DISPLAY=:0**
//      RK3588 的 Mali 驱动要用 X11 作 EGL 的 native display，
//      没有 DISPLAY 时 eglInitialize 直接失败、程序退出（退出码 1）。
//   2. **EGL 上下文绑定在创建它的线程上**
//      eglMakeCurrent 是 per-thread 的。渲染器在主线程初始化时绑定了
//      上下文 —— 从别的线程直接调 GL 会**死锁**
//      （实测：进程只剩 5 个线程、全部休眠、CPU 39%、零输出）。
//      → 「把渲染搬到线程」这条路走不通。
//        要并行只能拆 render_frame 的**纯 CPU 段**（投影/排序/组装），
//        GL 段（上传+绘制）留在主线程。
//
#include "offscreen.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace gsplat {

// ── ◀GUIDE▶ 【4.9】 gl_version_string()  查当前 OpenGL 版本字符串
//
//   干什么   : 向驱动问「你支持哪个 GL 版本」（本项目是 OpenGL ES 3.2）。
//   参数    : 无
//   返回    : 字符串，如 "OpenGL ES 3.2 ..."
// ──────────────────────────────────────────────────────────────
const char* gl_version_string() {
    const char* v = (const char*)glGetString(GL_VERSION);
    return v ? v : "(null)";
}

// ── ◀GUIDE▶ 【4.10】 OffscreenTarget::destroy()  拆掉离屏渲染环境
//
//   干什么   : 按「先子后父」顺序删 FBO、renderbuffer、EGL surface/context/display。
//   参数    : 无
//   返回    : 无
//   为什么顺序重要: EGL 资源有依赖关系，乱序销毁在某些驱动上会崩。
// ──────────────────────────────────────────────────────────────
void OffscreenTarget::destroy() {
    if (fbo)       { glDeleteFramebuffers(1, &fbo); fbo = 0; }
    if (color_rb)  { glDeleteRenderbuffers(1, &color_rb); color_rb = 0; }
    if (depth_rb)  { glDeleteRenderbuffers(1, &depth_rb); depth_rb = 0; }
    if (surf != EGL_NO_SURFACE) { eglDestroySurface(dpy, surf); surf = EGL_NO_SURFACE; }
    if (ctx  != EGL_NO_CONTEXT) { eglDestroyContext(dpy, ctx);   ctx  = EGL_NO_CONTEXT; }
    if (dpy  != EGL_NO_DISPLAY) { eglTerminate(dpy);              dpy  = EGL_NO_DISPLAY; }
}

// ── ◀GUIDE▶ 【4.11】 create_offscreen()  ★ 建离屏 EGL 上下文（板端渲染的入口）
//
//   干什么   : 在没有窗口的情况下建一个「凭空画布」：EGL display → context → pbuffer surface
//     → FBO（颜色 + 深度两个附件）。
//   参数    : t 输出目标；width/height 分辨率；verbose 打印细节；use_f16_color 颜色用半精度
//   返回    : true 成功
//   ★★最关键的坑: EGL 上下文**绑定创建它的线程**。跨线程调 GL 会死锁（实测：
//     进程只有 5 个线程全在睡、CPU 39%、零输出）。
//     所以本项目把「NPU 提特征」放到别的线程，GL 全部留在主线程。
//   为什么要离屏: 板子没有显示器也能跑；而且离屏 FBO 可以直接读回像素给算法用。
// ──────────────────────────────────────────────────────────────
bool create_offscreen(OffscreenTarget& t, int width, int height, bool verbose,
                      bool use_f16_color) {
    t.width = width; t.height = height;
    t.f16_color = use_f16_color;

    // 1) EGL 初始化 —— 走 X11/GBM 需要 DISPLAY
    t.dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (t.dpy == EGL_NO_DISPLAY) {
        fprintf(stderr, "[egl] eglGetDisplay failed\n");
        return false;
    }
    EGLint major = 0, minor = 0;
    if (!eglInitialize(t.dpy, &major, &minor)) {
        fprintf(stderr, "[egl] eglInitialize failed (DISPLAY=%s)\n",
                getenv("DISPLAY") ? getenv("DISPLAY") : "(unset)");
        return false;
    }
    if (verbose)
        fprintf(stdout, "[egl] %d.%d  vendor=%s\n", major, minor,
                eglQueryString(t.dpy, EGL_VENDOR));

    // 2) 选择 config：必须有 EGL_SURFACE_TYPE pbuffer
    const EGLint cfg_attr[] = {
        EGL_SURFACE_TYPE,    EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_NONE
    };
    EGLConfig cfg;
    EGLint n_cfg = 0;
    if (!eglChooseConfig(t.dpy, cfg_attr, &cfg, 1, &n_cfg) || n_cfg == 0) {
        fprintf(stderr, "[egl] no suitable config\n");
        return false;
    }

    // 3) 创建 pbuffer surface
    const EGLint pb_attr[] = { EGL_WIDTH, width, EGL_HEIGHT, height, EGL_NONE };
    t.surf = eglCreatePbufferSurface(t.dpy, cfg, pb_attr);
    if (t.surf == EGL_NO_SURFACE) {
        fprintf(stderr, "[egl] eglCreatePbufferSurface failed (0x%x)\n", eglGetError());
        return false;
    }

    // 4) 创建 ES 3.2 上下文
    const EGLint ctx_attr[] = {
        EGL_CONTEXT_CLIENT_VERSION, 3,
        EGL_CONTEXT_MINOR_VERSION, 2,
        EGL_NONE
    };
    t.ctx = eglCreateContext(t.dpy, cfg, EGL_NO_CONTEXT, ctx_attr);
    if (t.ctx == EGL_NO_CONTEXT) {
        // 部分实现不支持单独指定 minor，降级重试
        const EGLint ctx2[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
        t.ctx = eglCreateContext(t.dpy, cfg, EGL_NO_CONTEXT, ctx2);
    }
    if (t.ctx == EGL_NO_CONTEXT) {
        fprintf(stderr, "[egl] eglCreateContext failed (0x%x)\n", eglGetError());
        return false;
    }

    if (!eglMakeCurrent(t.dpy, t.surf, t.surf, t.ctx)) {
        fprintf(stderr, "[egl] eglMakeCurrent failed (0x%x)\n", eglGetError());
        return false;
    }
    if (verbose)
        fprintf(stdout, "[gl] %s | GLSL %s | %d×%d | color=%s\n",
                gl_version_string(),
                (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION),
                width, height, use_f16_color ? "RGBA16F" : "RGBA8");

    // 5) FBO + RGBA16F + DEPTH24
    glGenFramebuffers(1, &t.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);

    glGenRenderbuffers(1, &t.color_rb);
    glBindRenderbuffer(GL_RENDERBUFFER, t.color_rb);
    glRenderbufferStorage(GL_RENDERBUFFER,
                          use_f16_color ? GL_RGBA16F : GL_RGBA8, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              GL_RENDERBUFFER, t.color_rb);

    glGenRenderbuffers(1, &t.depth_rb);
    glBindRenderbuffer(GL_RENDERBUFFER, t.depth_rb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER, t.depth_rb);

    GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (st != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "[gl] FBO incomplete: 0x%x\n", st);
        return false;
    }

    glClearColor(0.f, 0.f, 0.f, 1.f);
    // ⚠️ 默认关掉 blend / depth test —— 这是「裸光栅化吞吐」模式。
    //    真实 3DGS 管线需要 alpha 混合（GL_ONE, GL_ONE_MINUS_SRC_ALPHA）
    //    和深度测试，由调用方按需开启：
    //      glEnable(GL_BLEND); glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    //      glEnable(GL_DEPTH_TEST);
    //    分开是因为混合速率与光栅化速率差异极大，混在一起测不出瓶颈在哪。
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    return true;
}

// ── ◀GUIDE▶ 【4.12】 compile()  编译一段 GLSL 着色器
//
//   干什么   : 把 GLSL 源码编译成 GPU 可执行代码，失败时打印编译日志。
//   参数    : stage GL_VERTEX_SHADER / GL_FRAGMENT_SHADER；src 源码；verbose 打印日志
//   返回    : shader 句柄（0 = 失败）
// ──────────────────────────────────────────────────────────────
static GLuint compile(GLenum stage, const char* src, bool verbose) {
    GLuint s = glCreateShader(stage);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        fprintf(stderr, "[shader] %s compile FAILED:\n%s\n",
                stage == GL_VERTEX_SHADER ? "VS" : "FS", log);
        glDeleteShader(s);
        return 0;
    }
    if (verbose && stage == GL_VERTEX_SHADER)
        fprintf(stdout, "[shader] VS compiled ok\n");
    return s;
}

// ── ◀GUIDE▶ 【4.13】 build_program()  把顶点+片元着色器链接成一个程序
//
//   干什么   : 编译两个着色器 → attach → link → 得到可用的 GL 程序对象。
//   参数    : vs_src 顶点着色器源码；fs_src 片元着色器源码；verbose 打印；out_prog 输出
//   返回    : 程序句柄（0 = 失败）
// ──────────────────────────────────────────────────────────────
GLuint build_program(const char* vs_src, const char* fs_src, bool verbose,
                     const char* const* bind_names, const GLuint* bind_locs,
                     int n_binds) {
    GLuint vs = compile(GL_VERTEX_SHADER, vs_src, verbose);
    if (!vs) return 0;
    GLuint fs = compile(GL_FRAGMENT_SHADER, fs_src, verbose);
    if (!fs) { glDeleteShader(vs); return 0; }

    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    // ⚠️ 必须在 link 前绑定 —— Mali 驱动对 layout(location=) 支持不稳
    if (bind_names && bind_locs && n_binds > 0) {
        for (int i = 0; i < n_binds; ++i)
            glBindAttribLocation(p, bind_locs[i], bind_names[i]);
    }
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        fprintf(stderr, "[shader] link FAILED:\n%s\n", log);
        glDeleteProgram(p);
        p = 0;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return p;
}

// ⚠️ BUG-002: 早期版本对 RGBA16F 附件也用 GL_RGBA/GL_UNSIGNED_BYTE 回读。
//    GLES 3.2 规定浮点颜色附件的合法格式组合是
//      (GL_RGBA, GL_FLOAT)  或  (GL_RGBA, GL_HALF_FLOAT)
//    用 GL_UNSIGNED_BYTE 会返回 GL_INVALID_OPERATION 且不写任何像素 →
//    调用方拿到全 0 缓冲，被误判成「画面全黑」。
//    这是与 BUG-001 叠加的第二个「全黑」来源。
//
// 现在：先把浮点内容回读成 float，再在 CPU 侧做 clamp→u8 转换，
//       保证无论 F16 还是 RGBA8 都能拿到正确图像。
//       仍保留一个纯 u8 快路径（RGBA8 附件）以获得真实带宽数字。
// ── ◀GUIDE▶ 【4.14】 read_pixels()  读回 16 位浮点颜色（诊断用）
//
//   干什么   : 把 FBO 的颜色附件读回 CPU（按 half 存），用于高精度诊断。
//   参数    : t 离屏目标；dst 输出字节缓冲
//   返回    : true 成功
//   实测教训  : RGBA16F 回读比 RGBA8 慢 **42 倍** —— 所以正式路径用 RGBA8 回读。
// ──────────────────────────────────────────────────────────────
bool read_pixels(const OffscreenTarget& t, uint8_t* dst) {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, t.fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);

    if (t.f16_color) {
        // 浮点附件：必须用 GL_FLOAT 回读
        static std::vector<float> tmp;
        const size_t n = (size_t)t.width * t.height * 4;
        tmp.resize(n);
        glReadPixels(0, 0, t.width, t.height, GL_RGBA, GL_FLOAT, tmp.data());
        if (glGetError() != GL_NO_ERROR) return false;
        for (size_t i = 0; i < n; ++i) {
            float v = tmp[i];
            if (v < 0.f) v = 0.f; else if (v > 1.f) v = 1.f;
            dst[i] = (uint8_t)(v * 255.0f + 0.5f);
        }
        return true;
    }
    glReadPixels(0, 0, t.width, t.height, GL_RGBA, GL_UNSIGNED_BYTE, dst);
    return glGetError() == GL_NO_ERROR;
}

// RGBA8 快路径：只有附件确实是 RGBA8 时才用，得到的 MB/s 才是真实带宽。
// ── ◀GUIDE▶ 【4.15】 read_pixels_u8()  读回 8 位颜色（正式路径用这个）
//
//   干什么   : 把颜色附件读成 RGBA8（每像素 4 字节），给特征提取用。
//   参数    : t 离屏目标；dst 输出（长度 = W*H*4）
//   返回    : true 成功
//   注意    : 读出来是 GL 行序（左下角为原点），使用前要上下翻转。
// ──────────────────────────────────────────────────────────────
bool read_pixels_u8(const OffscreenTarget& t, uint8_t* dst) {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, t.fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, t.width, t.height, GL_RGBA, GL_UNSIGNED_BYTE, dst);
    return glGetError() == GL_NO_ERROR;
}

// ── ◀GUIDE▶ 【4.16】 bind_draw_target()  把离屏 FBO 设为当前绘制目标
//
//   干什么   : 告诉 GL「接下来画的东西都画到这块离屏画布上」，并设好视口大小。
//   参数    : t 离屏目标
//   返回    : 无
//   类比    : 相当于把「画纸」从屏幕换成一块内存里的画布。
// ──────────────────────────────────────────────────────────────
void bind_draw_target(const OffscreenTarget& t) {
    glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
    glViewport(0, 0, t.width, t.height);
}

} // namespace gsplat
