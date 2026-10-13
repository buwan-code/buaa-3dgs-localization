// sorter.h — 板端高斯排序器（M3）
//
// 依据 NOTES.md §2.0 实测 + 附录 A 调研结论：
//  · GLES 3.2 无 compute shader → 排序必须走 CPU
//  · 实测本板（2.2M keys）：qsort 498ms / LSD 8趟 139ms / 仅高32位 68ms
//  · antimatter15 在 PC 上用 16-bit 计数排序排 1M splat ≈ 150ms，
//    与本板数据吻合 → 计数排序作为 baseline，实现成本低一个数量级
//
// key 编码采用 gsplat 官方做法（附录 A.1②）：
//    key = (tile_id << 32) | depth_as_uint32
//  视空间 z> 0 时，正浮点位模式单调递增，按 uint32 排序等价于按深度排序。
//  好处：不需要量化、无量化误差。
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>          // memcpy（depth_to_key / update_view_cache）
#include <thread>
#include <vector>

namespace gsplat {

constexpr int kTileSize = 16;              // 细 tile（光栅化粒度）

struct SortItem {
    uint64_t key;      // 高32=tile_id, 低32=depth bit pattern
    uint32_t gauss_id; // payload：对应的高斯索引
};

// ── ★ 紧凑深度排序项（2026-10-09）───────────────────────────────
// 本项目走「单 draw call + 全局严格序」，**不用 tile 分组**：
// 上面 SortItem 的高 32 位 key 恒为 0，纯属浪费；而且结构体对齐后
// sizeof(SortItem) == 16 B。键与载荷本来正好各 32 位，压成 8 B 后
// 排序两趟的访存量直接减半（本项目排序是仅次于投影的第二大项）。
//
//   key 升序 == 深度降序 == 远 → 近
//   （key = UINT32_MAX - depth_bits；正 float 的位模式随数值单调）
struct SortPair {
    uint32_t key;      // 反转后的深度位模式
    uint32_t id;       // 高斯索引
};
static_assert(sizeof(SortPair) == 8, "SortPair 应为 8 字节（2×uint32）");

// ── 排序器：持有可复用缓冲区，避免每帧分配 ──────────────────────
class Sorter {
public:
    Sorter() : last_view_(16, 0.0f) {}   // 必须 16 个元素，不是 {16, 0}
    ~Sorter() { free_all(); }

    // baseline：16-bit 计数排序（antimatter15 范式，附录 A.1①）
    // 单趟：桶计数 256KB（L2 装得下）→ 前缀和 → 重排
    void sort_counting16(std::vector<SortItem>& items, uint32_t tile_count);

    // ★ 顺序 alpha 混合专用：对 key 的低 32 位做严格升序排序。
    //   2 趟 × 16-bit LSD 计数排序（低位先排）。
    //   ⚠️ 这是**唯一**能保证单一 draw call 内全局 back-to-front 的方法：
    //   tile 分组排序（sort_counting16 / sort_lsd_tile_only）只在
    //   「每 tile 一个 workgroup」的 compute 光栅化下正确，本渲染器
    //   走 instanced 单 draw call，必须用全局序。
    //
    //   nth：并行线程数（<=1 走单线程）。计数与散射阶段按数据分块并行，
    //   前缀和（65536 项）保持单线程（只有 256KB，很快）。
    void sort_depth32(std::vector<SortItem>& items, unsigned nthreads = 1);

    // ★ 紧凑版：与 sort_depth32 完全同序（2 趟 × 16-bit LSD），
    //   但项宽从 16 B 降到 8 B。渲染器走这条路径。
    void sort_depth32_packed(std::vector<SortPair>& items, unsigned nthreads = 1);

    // 进阶：LSD 基数排序，只排高 32 位 tile_id（实测 68 ms）
    // tile 间有序后，tile 内深度近似局部有序
    void sort_lsd_tile_only(std::vector<SortItem>& items, uint32_t tile_count);

    // 进阶：完整 LSD 基数排序 8 趟（实测 139 ms）
    void sort_lsd_full(std::vector<SortItem>& items, uint32_t tile_count);

    // 复用上一帧结果：位姿变化在阈值内时直接返回 false 表示"沿用旧结果"
    // 依据 antimatter15 的视矩阵阈值判定（附录 A.1①）—— O(1) 判定，
    // 不需要知道 tile 表变了多少
    bool should_reuse(const float* new_view, const float* last_view,
                      float eps = 1e-4f);

    void update_view_cache(const float* view);

    size_t last_count() const { return last_count_; }
    const char* last_algo() const { return last_algo_; }
    // ★ 供 render_frame 的排序复用判定使用（Task #11 / M3）：
    //   · last_view_ptr() 返回上帧缓存视矩阵（16 float），只读
    //   · set_last_count() 记录本帧可见集规模，供下帧偏差检查
    const float* last_view_ptr() const { return last_view_.data(); }
    void set_last_count(size_t n) { last_count_ = n; }

    void free_all();

private:
    void alloc_buckets(uint32_t n);

    std::vector<uint32_t> buckets_a_, buckets_b_;  // 计数桶（前缀和）
    std::vector<uint32_t> local_cnt_;              // 多线程：每线程私有计数表
    std::vector<uint32_t> offsets_;                // 多线程：每线程每桶的写入基址
    std::vector<SortItem>  tmp_;                   // 重排缓冲
    std::vector<SortPair>  tmp_pair_;              // 紧凑路径的重排缓冲
    // ⚠️ 花括号初始化陷阱：写 last_view_{16, 0} 会创建「2 个元素、值为 16 和 0」
    //    的 vector，而不是 16 元素的零数组。必须在构造函数里用 (16, 0.0f)。
    std::vector<float>     last_view_;             // 上帧视矩阵（16 个 float）
    size_t                 last_count_ = 0;
    const char*            last_algo_ = "none";
};

// ── 辅助：tile_id 计算 ─────────────────────────────────────────
// row/col 为像素坐标，width 为图像宽
static inline uint32_t tile_id_of(int row, int col, int width) {
    return static_cast<uint32_t>((row / kTileSize) * ((width + kTileSize - 1) / kTileSize)
                                 + (col / kTileSize));
}

// float32 的 bit pattern → uint32（视空间 z>0 保证单调）
static inline uint32_t depth_to_key(float z) {
    uint32_t u;
    memcpy(&u, &z, sizeof(u));
    return u;
}

} // namespace gsplat
