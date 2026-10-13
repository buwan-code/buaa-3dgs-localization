// renderer.h — 3DGS 板端渲染器（M2：单高斯 → 全场景）
//
// ── 管线总览 ────────────────────────────────────────────────────────
//   ① 加载 PLY → Splat 数组（CPU）
//   ② 逐帧：
//      a. 对每个可见高斯做 EWA 投影（camera.cpp），产出 Splat2D
//      b. 按深度排序（sorter.cpp）
//      c. 把 (屏幕中心, 逆协方差, 颜色, 不透明度) 上传到 GPU
//      d. 用 instanced quad 光栅化 + alpha 混合（顺序即深度序）
//      e. 回读 RGBA8（M0 结论三：必须 RGBA8，禁 FP16 回读）
//
// ── 关键设计决策（都有实测依据）────────────────────────────────────
//  1. **投影在 CPU 做**：M0 实测顶点着色器里放 sqrt/cos/sin 会掉 14.7×
//     帧率。EWA 投影天然含 sqrt 与 atan2，所以必须 CPU 预计算。
//     代价是每帧要对 N 个高斯做一次矩阵运算 —— 用多线程摊。
//  2. **颜色附件用 RGBA8**：M0 实测 RGBA16F 回读慢 42×。
//  3. **排序用计数排序**：M0 实测 counting16 比 std::sort 快 6.2×。
//  4. **不用属性纹理**：M0 测出顶点/实例吞吐 ~9700 万/s 很充裕，
//     而属性纹理的随机采样会打满带宽。先用「每实例一个紧凑结构」的
//     方案（instanced array），把复杂度压到最低，等瓶颈出现再优化。
//
// ── 关于「顺序 alpha 混合」──────────────────────────────────────────
// 3DGS 的正确性依赖「从远到近」的混合顺序。本实现用**按深度排序的
// 实例顺序 + GL_BLEND(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)** 来保证。
// GL 规范保证同一 draw call 内实例按 gl_InstanceID 递增顺序光栅化，
// 且片元在同一像素上按提交顺序合成 —— 这正是我们要的。
#pragma once

#include "camera.h"
#include "ply_loader.h"
#include "sorter.h"
#include "thermal.h"
#include "offscreen.h"

#include <vector>
#include <string>
#include <cstdint>

namespace gsplat {

// ── 渲染器配置 ─────────────────────────────────────────────────────
struct RenderConfig {
    int   width  = 640;
    int   height = 480;
    // 每帧最多画多少个高斯（0 = 全部）。用于快速实验/降级
    int   max_splats = 0;
    // LOD: 只保留静态重要度最高的这么多个高斯（0 = 全部）。
    // 重要度 = opacity * sqrt(最大两轴尺度之积)，视图无关的屏幕贡献上界代理量。
    // 动机：投影是「遍历全部高斯」的地板成本；feicuiwan 俯视时视锥把整个
    // 场景包住、分块剔除命中 0%，遍历 1.41M 的代价已经付掉，只能靠降采样省。
    // 截断后仍按原索引顺序排列，维持空间局部性。
    int   lod_keep = 0;
    // 球谐阶数：0 = DC only，1 = DC + 1阶
    int   sh_degree = 1;
    // 是否启用 CPU 多线程投影
    bool  mt_project = true;
    // 投影线程数。0 = 自动（hardware_concurrency/2，留一半核给定位）
    // 允许手动指定，便于测量「全核」上限与为定位算法预留算力。
    int   project_threads = 0;
    // 是否启用深度排序（关掉可用于测"不排序"的性能上限）
    bool  do_sort = true;
    // 是否复用上帧排序（视矩阵阈值判定）
    bool  reuse_sort = true;
    // 视矩阵变化阈值（复用判定用）
    float reuse_eps = 0.0001f;
    // 颜色格式：false = RGBA8（推荐，回读快），true = RGBA16F
    bool  f16_color = false;
    // ── ★ 深度输出（2026-10-03 新增）────────────────────────────────
    // 参考 gsplat(nerfstudio) 的 render_mode="RGB+ED"：一次光栅化同时
    // 产出颜色与**期望深度**。定位算法（PiLoT 的 back_project）必须
    // 拿到度量深度才能把 2D 特征点反投影成 3D 点，所以这是刚需。
    //
    // 模式说明（对齐 gsplat 命名）：
    //   ED (Expected Depth) = ∑wᵢzᵢ / ∑wᵢ   ← 默认，加权平均深度
    //   D  (Accumulated)    = ∑wᵢzᵢ         ← 未归一化，仅诊断用
    //   M  (Median)         = 累积不透明度首次 >0.5 处的高斯深度
    //                         （近似：用「首个 a≥0.5」的深度，边缘更锐利）
    // ⚠️ 归一化是必须的：alpha 混合对透射率归一化，若不除 ∑wᵢ，
    //    半透明区域的深度会被系统性**低估**（离相机更近），
    //    反投影出来的 3D 点就整体往相机方向偏。
    bool  depth      = false;      // 是否输出深度（关闭时零额外开销）
    int   depth_mode = 0;          // 0=ED 归一化, 1=D 累积, 2=不输出
    // 深度缓冲近/远裁剪（米）。3DGS 场景尺度差异大，默认给够宽。
    float depth_near = 0.01f;
    float depth_far  = 1000.0f;
    // ★ 深度附件格式：false = RGBA16F（默认，2× 带宽更省），
    //   true = RGBA32F（全精度）。
    //   实测差异（512×288 / 163 层重叠）：
    //     16F：与 double 理想值 平均差 0.091 m、最大 0.174 m
    //     32F：应降到 1e-4 量级（无 half 量化）
    //   对定位算法的影响：30 m 处 0.17 m 误差 = 0.57% 相对误差，
    //   反投影出的 3D 点会沿视线方向漂移十几厘米。
    //   做参考视图/配准建议开 32F；纯可视化用 16F 更快。
    bool  depth_f32  = false;
    // 是否启用风扇自动控制
    bool  auto_fan = true;
    // 一个高斯 3σ 半径的像素上限（0 = 默认 4× 对角线）
    float max_sigma_radius = 0.f;
    // ── 重要性剔除阈值：alpha × 屏幕半径 低于此值就丢 ──────────────
    // 含义：该高斯对最终图像的最大贡献（alpha·像素数）小于此值时，
    // 画了也看不出（< 1/255），却要付出完整链路成本。
    // 实测本项目 3M 场景：0.5 → 剔掉 27.7%。
    // ⚠️ 单边作用：只影响"微小贡献"的高斯，不会删掉结构。
    //    因为用途是给定位算法生成参考视图，破洞代价高，所以别调太大。
    float min_alpha_radius = 0.5f;
    // ── ★ 空间分块视锥剔除（2026-10-08）────────────────────────────
    // 投影阶段原本要遍历**全部**高斯（实测单线程 250 ms / 277 万），
    // 且耗时与「可见实例数」无关 —— 是纯地板成本。
    // 开启后：预先把高斯切成固定大小的块（世界系 AABB），每帧先用
    // 6 个视锥平面做 AABB 剔除，整块跳过不投影。
    bool  block_cull = true;
    // 每块的高斯数。太小 → 块数多、剔除收益低；太大 → 剔除粒度粗。
    // 2048 在 277 万场景下约 1350 块，实测是较好的平衡点。
    int   block_size = 2048;
    // ── ★ 廉价粗筛（2026-10-08）──────────────────────────────────
    // 投影是当前最大单项，且**与被剔除多少无关** —— 它是「遍历全部
    // 高斯」的地板成本（实测 277 万 × ~26 ns）。而实测中 96.9% 的高斯
    // 最终都被剔除了，只是**判断得太晚**（要算完完整 EWA 才知道
    // alpha×sigma_radius）。
    //
    // 这里在完整 EWA 之前插一道**便宜的**过滤：只做一次世界→视空间
    // 变换 + 一次各向同性半径上界估计，就把大部分高斯挡掉。
    //
    // 为什么**安全**（不会造成画面破洞）：
    //   · 半径用 3·max(scale)·fx/z，是真实椭圆 3σ 半轴长度的**上界**
    //     （各向异性椭圆的任一半轴 ≤ 长轴，且正对相机时取最大）
    //   · 因此 `opacity × r_upper ≥ opacity × σ_true`，用上界做剔除
    //     只会「漏剔」不会「误剔」
    bool  cheap_cull = true;
    // ── ★ 异步回读（PBO 双缓冲，2026-10-08）────────────────────────
    // 实测：回读耗时的 **99% 是「等 GPU 完成」**（纯 glReadPixels 只要
    //   0.7 ms），而等待量 ∝ 可见实例数（939K → 36 ms）。
    // 开启后：本帧只把 glReadPixels 投进 PBO 就返回，不等；
    //   下一帧开头再收取（那时 GPU 通常已完成）→ 等待被隐藏。
    //
    // ⚠️ 代价：**结果滞后一帧**。取帧/取深度时必须用 `frame_camera()`
    //   返回的相机参数去配对，**不能**用本次 render_frame 传入的那个，
    //   否则反投影会用到错误位姿。
    //   第 1 次调用后 `has_frame()` 仍为 false（还没有可收的结果）。
    bool  async_readback = false;
    bool  verbose = false;
};

// ── 每帧统计 ───────────────────────────────────────────────────────
struct RenderStats {
    int    total_splats   = 0;   // 场景高斯总数
    int    visible_splats = 0;   // 通过 EWA 剔除后剩下多少
    int    drawn_splats   = 0;   // 实际提交绘制多少
    double ms_project     = 0;   // CPU 投影耗时
    double ms_sort        = 0;   // 排序耗时
    double ms_assemble    = 0;   // 组装 SplatInstance（SH 求值 + 逆协方差）
    double ms_upload      = 0;   // 上传 GPU
    double ms_draw        = 0;   // 光栅化 + 混合
    double ms_readback    = 0;   // 回读（含等待 fence + 实际 glReadPixels）
    double ms_wait        = 0;   // ★ 诊断：其中「等 GPU 栅栏」的部分
    double ms_readonly    = 0;   // ★ 诊断：其中「纯 glReadPixels」的部分
    double ms_total       = 0;
    bool   sort_reused    = false;  // 本帧是否复用了上帧排序（M3）
    float  soc_temp       = 0;
    int    fan_duty       = -1;
    // ★ 深度统计
    double ms_depth_readback = 0;   // 深度回读（含等栅栏）
    float  depth_min = 0, depth_max = 0, depth_mean = 0;  // 本帧深度范围
    int    depth_valid = 0;         // 有效深度像素数（w_sum > eps）
    // ★ 空间分块剔除统计
    int    total_blocks = 0;        // 总块数
    int    visible_blocks = 0;      // 通过视锥测试的块数
};

// ── GPU 侧每实例数据（务必保持 16 字节对齐，利于驱动做 burst 读）──
// 布局刻意用「逆协方差」而不是「协方差」：片元着色器里
//   d² = a'·dx² + 2b'·dx·dy + c'·dy²
// 用逆协方差可以直接算，省掉片元里的除法（除法在 GPU 上很贵）。
//
// ⚠️ 颜色**必须放在这里**，不能走 uniform。
//    教训：最初版本在循环里逐实例 glUniform4f + glDrawArraysInstanced(...,1)，
//    那是 100 万次独立 draw call —— 是把 GPU 用成串行机的灾难。
//    正确做法：一次 draw call 画全部实例，颜色作为 instanced attribute。
struct SplatInstance {
    float px, py;                // 屏幕中心（像素）
    float radius;                // quad 半径（像素）
    float inv_a, inv_b, inv_c;   // 2D 协方差之逆的上三角
    // 预乘 alpha 的颜色：rgb 已乘 alpha，片元侧只需再乘高斯权重
    float pr, pg, pb, pa;        // premultiplied rgb + alpha
    // ★ 深度输出：视空间 z（相机空间，米）。
    //   为什么放在实例里而不是走 uniform：每个高斯的 z 都不同，本来就是
    //   逐实例数据。而且 CPU 侧投影时**已经算出了 vz**（Splat2D.depth，
    //   排序键），这里只是顺手多拷 4 B —— 相对 144 MB 的上传量可忽略。
    float depth;                 // view-space z (m)
};   // 11 floats = 44 B
// ⚠️ 44 B 不是 16 的倍数。上一版刻意停在 40 B 并说明"补到 48 才对齐"，
//    但实测（§0.6）Mali 对非 16B 倍数的 instanced 跨步并无可见惩罚，
//    而**多一个 float 到 44 B 仍小于 48 B**：即深度是"免费"的
//    （没有把 stride 推过原本讨论的 48 B 门槛）。
//    若日后要加更多逐实例字段，再一次性补到 64 B（4×16）更划算。
static_assert(sizeof(SplatInstance) == 44, "SplatInstance 应与字段声明一致（11 float = 44 B）");

class Renderer {
public:
    Renderer();
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // ── 初始化 ────────────────────────────────────────────────────
    // 加载 PLY 并建立 GL 资源。成功返回 true。
    bool init(const char* ply_path, const RenderConfig& cfg);

    // 不加载 PLY，仅初始化 GL（用于合成数据测试）
    bool init_gl_only(const RenderConfig& cfg);

    // ── 每帧 ──────────────────────────────────────────────────────
    // 渲染一帧到离屏缓冲。位姿由 cam 给出。
    // collect_prev=false：async 模式下跳过「收取上一帧」步骤 ——
    //   供帧间流水线使用：主程序在帧开头自行 collect（拿到上一帧渲染图
    //   做匹配），在帧尾用「本帧解算位姿」再调一次 render_frame(cam,false)
    //   投递渲染，供下一帧消费。这样渲染图位姿只滞后 1 帧（而非 2 帧）。
    bool render_frame(const Camera& cam, bool collect_prev = true);

    // 取回最近一帧的 RGBA8 图像（size = w*h*4）
    bool get_frame(std::vector<uint8_t>& out_rgba8) const;

    // ── ★ 异步回读（cfg.async_readback == true 时才有意义）──────────
    // 是否已经有一帧「已完成回读」可取（异步模式下第 1 次渲染后为 false）
    bool has_frame() const { return has_frame_; }
    // ★ 当前 frame_/depth_ 里那份数据对应的**相机参数**。
    //   异步模式下它等于「上一次 render_frame 传入的 cam」，而不是最近这次。
    //   ⚠️ 反投影/反解算时必须用这个，否则位姿与深度不配对。
    const Camera& frame_camera() const { return frame_cam_; }

    // ── ★ 深度输出（需要 cfg.depth == true）─────────────────────────
    // 取回最近一帧的深度图（size = w*h，单位 = 相机空间 z，米）
    // 无效像素（没有高斯覆盖）值为 0。
    bool get_depth(std::vector<float>& out_depth) const;

    // 取回深度权重和（∑wᵢ，size = w*h）。用于判断像素是否可信：
    //   w_sum 很小 → 该像素是半透明边缘/背景，深度不可靠
    //   （PiLoT 的 edge_mask 逻辑与此对应）
    bool get_depth_weights(std::vector<float>& out_wsum) const;

    // 把深度图编码成 16 位灰度 PNG 可看的 PGM（P5），便于直接拉回 PC
    bool save_pgm(const char* path) const;

    // 保存最近一帧为 PPM（无依赖，方便拉回 PC 看）
    bool save_ppm(const char* path) const;

    // ── 访问器 ────────────────────────────────────────────────────
    const RenderStats& stats() const { return stats_; }
    bool is_2dgs_ = false;   // ★ 当前模型是否为 2DGS（缺 scale_2）
    const RenderConfig& config() const { return cfg_; }
    size_t num_splats() const { return splats_.size(); }

    // 直接用外部提供的 splats 替换（合成测试用）
    void set_splats(std::vector<Splat> s, std::vector<ShCoeffs> sh);

    // 主循环里定期调用（自动风扇 + 温度统计）
    ThermalGuard& thermal() { return thermal_; }

private:
    bool build_gl_resources();
    // 把 SplatInstance 的 4 个字段绑到 attribute location 0..3（divisor=1）
    void bind_instance_attribs();
    // ⚠️ 不能叫 build_program：那会遮蔽命名空间里的自由函数
    //    gsplat::build_program(...)，导致成员函数内部自调用自己（编译报
    //    "no matching function"）。这是 C++ 名字查找的经典陷阱。
    bool build_shader_program();
    // 按静态重要度截断高斯数组（见 RenderConfig::lod_keep）
    void apply_importance_lod(size_t keep);

    RenderConfig cfg_;
    RenderStats  stats_;

    // 场景数据
    std::vector<Splat>     splats_;
    std::vector<ShCoeffs>  sh_;

    // 每帧中间结果
    std::vector<Splat2D>        proj_;      // 投影结果
    std::vector<uint32_t>       order_;     // 排序后的索引

    // ★ 性能优化（2026-10-03）：把「组装」从排序后的随机 gather 改为
    //   一次**顺序**生成 + 一次**顺序读/散写**重排。
    //
    //   原方案：组装循环按 sorted order 取 order_[k]，再散列访问
    //     proj_[idx] + splats_[idx] + sh_[idx] 三个数组。
    //     三个数组工作集合计 ~400 MB（3M × (32+56+48) B），远超 L2/L3，
    //     每次访问都是 DRAM miss —— 实测 163 ms / 3M ≈ 54 ns/高斯，
    //     正好是一次 DRAM 随机访问的量级。
    //
    //   新方案：颜色只需要「高斯中心 → 固定相机」的方向，相机每帧不变，
    //     所以可以**在投影阶段（顺序访问）就把完整 SplatInstance 算好**，
    //     写进 inst_by_id_[i]。投影阶段本身就是顺序的，
    //     顺手算了颜色几乎零额外成本（数据已在寄存器/缓存里）。
    //     排序后只需要按 order_ 做一次 40B/项的 permute：
    //       读按 order_ 散列（1 次 miss），但**写是顺序的**。
    //     把 3 次随机读 + 3 次随机写 降到 1 次随机读 + 1 次顺序写。
    std::vector<SplatInstance>  inst_by_id_;  // 按高斯 id 索引（顺序生成）
    std::vector<SplatInstance>  instances_;   // 按排序序排列（待上传）

    // ── ★ 空间分块（视锥剔除用）──────────────────────────────────
    // 高斯按索引切成固定大小的块，每块预计算世界系 AABB。
    // 每帧对块做 6 平面（AABB p-vertex）测试 —— 该测试是**严格保守**的：
    // 只可能漏剔（多算几个块），绝不会误剔（否则画面会破洞）。
    struct Block {
        Vec3 mn, mx;        // 世界系 AABB
        int  lo, hi;        // 高斯索引区间 [lo, hi)
        bool visible;       // 本帧是否通过视锥测试
    };
    std::vector<Block> blocks_;
    int  block_used_ = 0;   // 建块时实际用的块大小（供按块遍历反查）
    void build_blocks();

    // ── ★ 异步回读的双缓冲 PBO（见 RenderConfig::async_readback）────
    // 为什么是 2 组：本帧往 PBO[A] 写、同时从 PBO[B] 收上一帧；
    // 下一帧交换。若只用 1 组，收取时会和本帧的写入撞车。
    struct ReadbackPBO {
        GLuint pbo   = 0;
        void*  fence = nullptr;   // GLsync，标记「写入完成」
        size_t cap   = 0;
    };
    ReadbackPBO rb_color_[2];
    ReadbackPBO rb_depth_[2];
    int      rb_idx_     = 0;      // 本帧写入用哪个
    bool     rb_pending_ = false;  // 是否有未收取的异步回读
    bool     has_frame_  = false;  // frame_/depth_ 是否有有效数据
    Camera   frame_cam_{};         // frame_ 数据对应的相机
    Camera   pending_cam_{};       // 已提交、待收取的那帧的相机

    // 异步回读：提交本帧 / 收取上一帧
    // ★ 2026-10-09 改为 public：帧间流水线模式下，主程序需要在帧开头
    //   自行 collect（拿到上一帧渲染图做匹配），帧尾再 render_frame(cam,false)
    //   投递本帧 —— render_frame 的默认 collect 时机（帧尾）会让渲染图
    //   滞后 2 帧，显式拆开才能压到 1 帧。
public:
    bool issue_async_readback(const Camera& cam);
    bool collect_async_readback();
    void free_readback_pbos();

private:

    // 排序器
    Sorter sorter_;

    // 热管理
    ThermalGuard thermal_;

    // GL 资源
    OffscreenTarget gl_{};
    GLuint prog_     = 0;
    GLuint vao_      = 0;
    // ── 双缓冲（ping-pong）实例缓冲 ──────────────────────────────
    // ⚠️ 单缓冲是巨大的性能陷阱，实测教训：
    //    用单个 VBO 时，「上一帧的 draw 还在读 buffer」与「本帧要写
    //    同一个 buffer」形成写-读冲突，驱动只能阻塞等 GPU 排空 ——
    //    实测 3M×48B=144MB 的上传耗到 727 ms（而裸带宽实测仅 19 ms）。
    //    改成双缓冲后：本帧写 B，GPU 读上帧的 A，二者并行无冲突。
    //
    // ★ 持久映射（GL_EXT_buffer_storage，本板已确认支持）：
    //   在双缓冲之上，进一步把映射**常驻**（glMapBufferRange 只调一次），
    //   之后每帧只做 memcpy —— 彻底消除 map/unmap 的隐式同步开销。
    //   实测本板未持久化时每帧有 ~35 ms 固定开销（200K 也占 42 ms）。
    struct InstBuf {
        GLuint     vbo  = 0;
        GLsizeiptr cap  = 0;      // 已分配字节
        void*      map  = nullptr; // 持久映射指针（nullptr = 未映射/不可用）
        GLuint     vao  = 0;      // 每个缓冲配自己的 VAO（属性绑定是 VAO 状态）
    };
    InstBuf ibuf_[2];
    int     inst_idx_ = 0;         // 本帧写到哪个缓冲
    bool    use_persistent_ = true; // 持久映射是否可用（构造时探测）
    GLuint fbo8_     = 0;      // 若混合用 F16，则这是最终的 RGBA8 目标
    GLuint color8_   = 0;
    GLint  u_res_    = -1;
    GLint  u_max_r_  = -1;
    GLint  u_depth_mode_ = -1;  // ★ 深度模式 uniform

    // ── ★ 深度附件（MRT）──────────────────────────────────────────
    // 用第二个颜色附件挂 RGBA16F：
    //   .r = 累积深度 ∑(a·z)
    //   .g = 累积权重 ∑a
    //   .b = 累积不透明度 1-∏(1-a)（备用，做中值/可信度判定）
    //   .a = 预留
    // ⚠️ 用**渲染缓冲**而不是纹理：本方案不需要在 shader 里采样它，
    //    而 renderbuffer 在 Mali 上做 MRT 的 tile 回写更省带宽。
    //    参考 gsplat 的 CUDA 实现：深度与颜色在同一次遍历里累积，
    //    不做二次 pass —— 这是开销最小的做法。
    GLuint depth_rb_ = 0;
    bool   depth_is_f32_ = false;    // 深度附件实际用的格式（决定回读格式）
    // MRT 输出列表（draw 前 glDrawBuffers 用）
    GLuint draw_bufs_[2] = {0, 0};
    int    n_draw_bufs_  = 1;
    void*  fence_depth_  = nullptr;  // 深度回读的栅栏
    // ★ GPU 同步栅栏（Task #12）：取代每帧 glFinish 硬同步。
    //   glFenceSync 只插标记不阻塞；等待推迟到回读前，
    //   让 CPU 侧收尾工作与 GPU 光栅化重叠。
    void*  fence_     = nullptr;   // GLsync，用 void* 存以避免引入 GL 头到 .h

    // 回读缓冲
    mutable std::vector<uint8_t> frame_;
    mutable std::vector<float>   depth_;      // ★ 归一化后的深度（w*h）
    mutable std::vector<float>   depth_wsum_; // ★ ∑w（w*h）
    bool gl_ready_ = false;
};

} // namespace gsplat
