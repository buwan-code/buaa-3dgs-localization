// offscreen.h — EGL 离屏渲染上下文（M2）
//
// 依据 NOTES.md §2.5.3：本项目用途是「给定位算法生成参考视图」，
// 不需要窗口系统，只用 Pbuffer surface。
//
// ⚠️ 关键坑（已在调研中踩过）：SSH 会话下 DISPLAY 为空会导致
//    eglInitialize failed（GBM/X11/surfaceless/device 全失败）。
//    必须 export DISPLAY=:0，否则会误判成驱动故障。
#pragma once

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl32.h>
#include <cstdint>
#include <string>
#include <vector>

namespace gsplat {

struct OffscreenTarget {
    EGLDisplay dpy = EGL_NO_DISPLAY;
    EGLContext ctx = EGL_NO_CONTEXT;
    EGLSurface surf = EGL_NO_SURFACE;

    int   width = 0, height = 0;
    GLuint fbo = 0;
    GLuint color_rb = 0;   // RGBA16F 或 RGBA8 颜色渲染缓冲
    GLuint depth_rb = 0;   // DEPTH24 深度渲染缓冲
    bool  f16_color = true; // 记录实际使用的颜色格式（read_pixels 依赖它）

    void destroy();
};

// 创建离屏上下文。width/height 建议用 L1 档 640×480（§2.5.4）
// use_f16_color=true用 RGBA16F（半精度混合，gsplat 方案）
// use_f16_color=false 用 RGBA8（回读兼容性更好，诊断用）
bool create_offscreen(OffscreenTarget& t, int width, int height, bool verbose = true,
                      bool use_f16_color = true);

// 编译/链接着色器，失败时把 info log 打到 stderr
// bind_attribs: 可选，{location, name} 对列表；传nullptr 表示不绑定。
//⚠️ Mali 驱动对 layout(location=) 支持不稳，建议显式传入绑定表。
GLuint build_program(const char* vs_src, const char* fs_src, bool verbose = true,
                     const char* const* bind_names = nullptr,
                     const GLuint* bind_locs = nullptr, int n_binds = 0);

// 读回像素到 RGBA8 的 dst（size = w*h*4）。
// 自动适配 F16/RGBA8 附件：F16 附件会先读成 float 再在 CPU 侧转 u8。
// ⚠️ 因此这个函数的耗时**不等于**纯回读带宽（F16 路径含转换开销）。
//    要测真实回读带宽请用 read_pixels_u8（附件必须本来就是 RGBA8）。
bool read_pixels(const OffscreenTarget& t, uint8_t* dst);

// 纯 RGBA8 回读快路径：附件必须为 RGBA8，否则会 GL_INVALID_OPERATION。
bool read_pixels_u8(const OffscreenTarget& t, uint8_t* dst);

// 绑定 FBO 为目标并设置 viewport —— 每次 draw 前都应调用
void bind_draw_target(const OffscreenTarget& t);

// GLES 版本字符串
const char* gl_version_string();

} // namespace gsplat
