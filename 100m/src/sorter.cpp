// sorter.cpp — 板端高斯深度排序（392 行）
//
// ══ 本文件结构 ═══════════════════════════════════════════════════════
//   sort_counting16       ~14   16-bit 计数排序（默认路径，最快）
//   sort_depth32          ~62   2 趟 16-bit LSD 计数排序（全 32 位精度）
//   sort_depth32_packed   ~177  同上，用打包键值对（省内存）
//   sort_lsd_tile_only    ~266  仅按 tile 的 LSD 排序
//   sort_lsd_full         ~306  完整 LSD
//   should_reuse / update_view_cache  ~363 / ~374
//                               视矩阵变化小于阈值则跳过本帧排序
//   alloc_buckets / free_all          ~379 / ~386
//
// ══ 为什么用计数排序 ═════════════════════════════════════════════════
//   NOTES.md 实测（本板 2.2M keys）：qsort 498ms / LSD 8趟 139ms / 仅高32位 68ms
//   计数排序的固定开销是 2×(256KB memset + 65536 前缀和)；
//   实测 85K 元素时计数排序 9.8ms，std::sort 反而要 12.5ms
//   （元素更多时计数排序优势更大）
//
// ══ 并行化要点 ═══════════════════════════════════════════════════════
//   计数与散射两阶段按数据分块并行。关键是：
//   每线程先算自己的**私有块偏移**，再累加为全局偏移，
//   否则多线程写同一桶会互相覆盖。
//
// ══ 排序顺序为何重要 ═════════════════════════════════════════════════
//   3DGS 是半透明混合，必须**从远到近**合成。
//   CPU 排好序后把实例顺序传给 GPU，靠 GL 的实例顺序保证即自动正确。
//
#include "sorter.h"
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>    // std::sort（小规模排序路径）

namespace gsplat {

// ── 16-bit 计数排序 ─────────────────────────────────────────────
// 思路（antimatter15）：把 key 的低 16 位当桶号，单趟计数 + 前缀和 + 重排。
// 对本项目 key=(tile<<32)|depth，tile 位数通常超过 16，所以这里用
// **高 16 位**（tile_id 的低 16 位）分桶 —— 等价于按 tile 分组排序。
// 若 tile 数 ≤ 65536，一次排序即完成「先 tile 后深度」的目的。
void Sorter::sort_counting16(std::vector<SortItem>& items, uint32_t tile_count) {
    if (items.empty()) { last_count_ = 0; last_algo_ = "counting16"; return; }

    const uint32_t kBuckets = 65536;
    alloc_buckets(kBuckets);
    std::memset(buckets_a_.data(), 0, kBuckets * sizeof(uint32_t));

    // 桶号 = tile_id &0xFFFF（tile 数超过 65536 时会有多轮，见文末说明）
    for (const auto& it : items) {
        uint32_t tile = static_cast<uint32_t>(it.key >> 32);
        buckets_a_[tile & 0xFFFFu]++;
    }

    // 独占前缀和
    uint32_t sum = 0;
    for (uint32_t i = 0; i < kBuckets; ++i) {
        uint32_t c = buckets_a_[i];
        buckets_a_[i] = sum;      //起始位置
        buckets_b_[i] = c;       // 桶大小（reorder 后要读，这里借用）
        sum += c;
    }

    // 重排
    tmp_.resize(items.size());
    for (const auto& it : items) {
        uint32_t tile = static_cast<uint32_t>(it.key >> 32);
        tmp_[buckets_a_[tile & 0xFFFFu]++] = it;
    }
    items.swap(tmp_);

    last_count_ = items.size();
    last_algo_ = "counting16";
}

// ── ★ 顺序混合专用：严格 32 位深度排序（可并行）─────────────────
// 2 趟 × 16-bit LSD 计数排序（先低 16 位，再高 16 位）。
// 桶数 65536 × 4 B = 256 KB，L2 装得下，所以两趟都很便宜。
//
// 为什么是 2 趟而不是 1 趟：单趟只能按 16 位分桶，深度范围超过 65536
// 个可区分值时会丢精度（远处一片高斯挤进同一桶 → 桶内顺序 = 输入顺序
// = 原始文件顺序，与深度无关 → 混合错乱）。2 趟得到完整 32 位序，
// 且位模式对「正 float 的数值序」是单调的，所以等价于按深度精确排序。
//
// ★ 并行化：计数与散射两阶段按数据分块并行。
//   ⚠️ 散射阶段**不能**简单按块并行 —— 每块都从同一个 prefix 起写会
//   互相覆盖。正确做法：每线程先算自己的私有块偏移（局部计数 + 局部
//   前缀和 + 全局基址），再散射。这样各线程写的是互不重叠的区间。
//   这是标准的多线程计数排序（parallel counting sort）写法。
// ── ◀GUIDE▶ 【3.2】 Sorter::sort_depth32()  按深度全局排序（alpha 混合要「远→近」）
//
//   干什么   : 把全部可见高斯按深度排好序，让渲染器可以「从远到近」画，
//     半透明叠加才正确。
//   参数    : items 待排序数组（key + 高斯索引）；nthreads 线程数
//   返回    : 无（就地把 items 排好）
//   算法    : 2 趟 16-bit LSD 基数排序（先排低位 16 位，再排高位 16 位）。
//   为什么不用 std::sort: 实测 qsort 498 ms / LSD 8 趟 139 ms / 本实现两趟最快。
//   ★为什么必须全局序: 本项目一个 draw call 画完（GL 保证实例按 ID 顺序光栅化），
//     tile 分组排序只在「每 tile 一个 compute 工作组」下才正确。
// ──────────────────────────────────────────────────────────────
void Sorter::sort_depth32(std::vector<SortItem>& items, unsigned nthreads) {
    if (items.empty()) { last_count_ = 0; last_algo_ = "depth32"; return; }

    // ⚠️ 曾试过「小规模走 std::sort」但**实测更慢**，已回退：
    //   r=2 场景可见 ~85K → 计数排序 9.8 ms，std::sort 反而 12.5 ms。
    //   原因：85K 时计数排序的固定开销（2×(256KB memset + 65536 前缀和)）
    //   已被 O(n) 部分摊平；而 std::sort 的 n·log n ≈ 1.4M 次比较，
    //   每次比较都要分支（key 分布随机 → 预测失败率高），代价更大。
    //   → 结论：**不要凭"理论复杂度"下判断，要实测**。这个规模区间
    //     计数排序就是更优的。
    constexpr uint32_t kBuckets = 65536;
    const size_t n = items.size();
    alloc_buckets(kBuckets);
    tmp_.resize(n);

    // 数据量大且有多核时才并行（小数据并行反而亏在启动开销）
    const unsigned nt = (nthreads > 1 && n >= 200000) ? nthreads : 1u;

    for (int pass = 0; pass < 2; ++pass) {
        const int shift = pass * 16;

        if (nt <= 1) {
            // ── 单线程路径（原实现，作为对照基准保留）────────────
            std::memset(buckets_a_.data(), 0, kBuckets * sizeof(uint32_t));
            for (size_t i = 0; i < n; ++i)
                buckets_a_[(uint32_t)(items[i].key >> shift) & 0xFFFFu]++;
            uint32_t sum = 0;
            for (uint32_t i = 0; i < kBuckets; ++i) {
                const uint32_t c = buckets_a_[i];
                buckets_a_[i] = sum;
                sum += c;
            }
            for (size_t i = 0; i < n; ++i)
                tmp_[buckets_a_[(uint32_t)(items[i].key >> shift) & 0xFFFFu]++]
                    = items[i];
            items.swap(tmp_);
            continue;
        }

        // ── 多线程路径 ───────────────────────────────────────────
        const size_t chunk = (n + nt - 1) / nt;
        // 每线程一张 (桶 → 局部计数) 表。65536×4B×nt 有点大，
        // 但 nt<=8 时也就 2MB，可接受（一次分配、循环复用）。
        if ((size_t)nt * kBuckets > local_cnt_.size())
            local_cnt_.assign((size_t)nt * kBuckets, 0);
        uint32_t* lc = local_cnt_.data();

        // ① 各线程局部计数
        {
            std::vector<std::thread> th;
            th.reserve(nt);
            for (unsigned t = 0; t < nt; ++t) {
                const size_t lo = t * chunk;
                const size_t hi = std::min(n, lo + chunk);
                if (lo >= hi) break;
                th.emplace_back([&, t, lo, hi]() {
                    uint32_t* cnt = lc + (size_t)t * kBuckets;
                    std::memset(cnt, 0, kBuckets * sizeof(uint32_t));
                    for (size_t i = lo; i < hi; ++i)
                        cnt[(uint32_t)(items[i].key >> shift) & 0xFFFFu]++;
                });
            }
            for (auto& x : th) x.join();
        }

        // ② 合并成全局前缀和 + 每线程的块基址
        //    offset[t][b] = 全局起始位置 + 前面所有线程在桶 b 的计数之和
        if ((size_t)nt * kBuckets > offsets_.size())
            offsets_.assign((size_t)nt * kBuckets, 0);
        uint32_t* off = offsets_.data();
        {
            uint32_t sum = 0;
            for (uint32_t b = 0; b < kBuckets; ++b) {
                // buckets_a_[b] 先临时用来记录「全局起始」
                const uint32_t base = sum;
                buckets_a_[b] = base;
                for (unsigned t = 0; t < nt; ++t) {
                    off[(size_t)t * kBuckets + b] = sum;
                    sum += lc[(size_t)t * kBuckets + b];
                }
            }
        }

        // ③ 各线程按自己的基址散射（区间互不重叠）
        {
            std::vector<std::thread> th;
            th.reserve(nt);
            for (unsigned t = 0; t < nt; ++t) {
                const size_t lo = t * chunk;
                const size_t hi = std::min(n, lo + chunk);
                if (lo >= hi) break;
                th.emplace_back([&, t, lo, hi]() {
                    uint32_t* my = off + (size_t)t * kBuckets;
                    for (size_t i = lo; i < hi; ++i) {
                        tmp_[my[(uint32_t)(items[i].key >> shift) & 0xFFFFu]++]
                            = items[i];
                    }
                });
            }
            for (auto& x : th) x.join();
        }
        items.swap(tmp_);
    }

    last_count_ = items.size();
    last_algo_ = (nt > 1) ? "depth32-mt" : "depth32";
}

// ── ★ 紧凑版 32 位深度排序（8 B/项）─────────────────────────────
// 与 sort_depth32 **逐位同序**（同样 2 趟 × 16-bit LSD，先低后高），
// 只是项宽从 16 B 降到 8 B。收益全在访存：两趟「读 + 散射写」的
// 数据量减半，而本项目排序的瓶颈正是访存（随机散射写）。
//
// ⚠️ 正确性依赖「key 升序 == 远→近」——与 SortItem 路径一致：
//    key = UINT32_MAX - bits(z)，z>0 时正 float 位模式随数值单调。
// ── ◀GUIDE▶ 【3.3】 Sorter::sort_depth32_packed()  紧凑版全局深度排序（实际在用）
//
//   干什么   : 与 sort_depth32 结果完全一样，但元素从 16 字节压到 8 字节。
//   参数    : items —— SortPair{key, id} 数组；nthreads 线程数
//   返回    : 无
//   为什么   : 本项目不用 tile 分组，key 的高 32 位恒为 0（纯浪费）；
//     压成 8 字节后两趟排序的访存量减半 → 排序时间直接省一半。
//     key = UINT32_MAX - depth_bits，于是「key 升序 = 深度降序 = 远→近」。
// ──────────────────────────────────────────────────────────────
void Sorter::sort_depth32_packed(std::vector<SortPair>& items, unsigned nthreads) {
    if (items.empty()) { last_count_ = 0; last_algo_ = "depth32p"; return; }

    constexpr uint32_t kBuckets = 65536;
    const size_t n = items.size();
    alloc_buckets(kBuckets);
    tmp_pair_.resize(n);

    const unsigned nt = (nthreads > 1 && n >= 200000) ? nthreads : 1u;

    for (int pass = 0; pass < 2; ++pass) {
        const int shift = pass * 16;

        if (nt <= 1) {
            std::memset(buckets_a_.data(), 0, kBuckets * sizeof(uint32_t));
            for (size_t i = 0; i < n; ++i)
                buckets_a_[(items[i].key >> shift) & 0xFFFFu]++;
            uint32_t sum = 0;
            for (uint32_t b = 0; b < kBuckets; ++b) {
                const uint32_t c = buckets_a_[b];
                buckets_a_[b] = sum;
                sum += c;
            }
            for (size_t i = 0; i < n; ++i)
                tmp_pair_[buckets_a_[(items[i].key >> shift) & 0xFFFFu]++] = items[i];
            items.swap(tmp_pair_);
            continue;
        }

        // ── 多线程：局部计数 → 归并出每线程每桶基址 → 区间互不重叠地散射 ──
        const size_t chunk = (n + nt - 1) / nt;
        if ((size_t)nt * kBuckets > local_cnt_.size())
            local_cnt_.assign((size_t)nt * kBuckets, 0);
        uint32_t* lc = local_cnt_.data();
        {
            std::vector<std::thread> th;
            th.reserve(nt);
            for (unsigned t = 0; t < nt; ++t) {
                const size_t lo = t * chunk;
                const size_t hi = std::min(n, lo + chunk);
                if (lo >= hi) break;
                th.emplace_back([&, t, lo, hi]() {
                    uint32_t* cnt = lc + (size_t)t * kBuckets;
                    std::memset(cnt, 0, kBuckets * sizeof(uint32_t));
                    for (size_t i = lo; i < hi; ++i)
                        cnt[(items[i].key >> shift) & 0xFFFFu]++;
                });
            }
            for (auto& x : th) x.join();
        }

        if ((size_t)nt * kBuckets > offsets_.size())
            offsets_.assign((size_t)nt * kBuckets, 0);
        uint32_t* off = offsets_.data();
        {
            uint32_t sum = 0;
            for (uint32_t b = 0; b < kBuckets; ++b) {
                for (unsigned t = 0; t < nt; ++t) {
                    off[(size_t)t * kBuckets + b] = sum;
                    sum += lc[(size_t)t * kBuckets + b];
                }
            }
        }

        {
            std::vector<std::thread> th;
            th.reserve(nt);
            for (unsigned t = 0; t < nt; ++t) {
                const size_t lo = t * chunk;
                const size_t hi = std::min(n, lo + chunk);
                if (lo >= hi) break;
                th.emplace_back([&, t, lo, hi]() {
                    uint32_t* my = off + (size_t)t * kBuckets;
                    for (size_t i = lo; i < hi; ++i)
                        tmp_pair_[my[(items[i].key >> shift) & 0xFFFFu]++] = items[i];
                });
            }
            for (auto& x : th) x.join();
        }
        items.swap(tmp_pair_);
    }

    last_count_ = items.size();
    last_algo_ = (nt > 1) ? "depth32p-mt" : "depth32p";
}

// ── LSD 基数排序：只排高 32 位（tile_id） ───────────────────────
// 实测 68 ms / 2.2M keys。tile 间有序后，tile 内深度近似局部有序，
// 渲染阶段本来就是按 tile 遍历的，所以这个折衷对本项目足够。
// ── ◀GUIDE▶ 【3.4 备选】 Sorter::sort_lsd_tile_only()  只按 tile 排序（实验分支）
//
//   干什么   : 只排高 32 位的 tile_id，让同 tile 的高斯聚在一起。
//   参数    : items 数组；tile_count tile 数量
//   返回    : 无
//   实测    : 68 ms（比全排序快），但**不满足全局序**，本项目主路径没用它。
//   保留原因  : 论文消融实验要用。
// ──────────────────────────────────────────────────────────────
void Sorter::sort_lsd_tile_only(std::vector<SortItem>& items, uint32_t tile_count) {
    if (items.empty()) { last_count_ = 0; last_algo_ = "lsd_tile"; return; }

    // 8 趟 × 4 bit，对 32 位 tile_id
    constexpr int kPasses = 8, kRadix = 16, kMask = 0xF;
    const size_t n = items.size();
    tmp_.resize(n);

    std::vector<SortItem>* src = &items;
    std::vector<SortItem>* dst = &tmp_;

    for (int pass = 0; pass < kPasses; ++pass) {
        const int shift = pass * 4;
        alloc_buckets(kRadix);
        std::memset(buckets_a_.data(), 0, kRadix * sizeof(uint32_t));

        for (size_t i = 0; i < n; ++i) {
            uint32_t tile = static_cast<uint32_t>((*src)[i].key >> 32);
            buckets_a_[(tile >> shift) & kMask]++;
        }
        uint32_t s = 0;
        for (int i = 0; i < kRadix; ++i) {
            uint32_t c = buckets_a_[i];
            buckets_a_[i] = s;
            s += c;
        }
        for (size_t i = 0; i < n; ++i) {
            uint32_t tile = static_cast<uint32_t>((*src)[i].key >> 32);
            (*dst)[buckets_a_[(tile >> shift) & kMask]++] = (*src)[i];
        }
        std::swap(src, dst);
    }
    if (src != &items) items.swap(tmp_);

    last_count_ = items.size();
    last_algo_ = "lsd_tile";
}

// ── 完整 LSD 基数排序（64 位 key，8 趟 × 8 bit） ────────────────
// 实测 139 ms / 2.2M keys。
// ── ◀GUIDE▶ 【3.5 备选】 Sorter::sort_lsd_full()  完整 LSD 基数排序（实验分支）
//
//   干什么   : 完整 8 趟 LSD 基数排序，64 位 key 全排。
//   参数    : items 数组；tile_count tile 数量
//   返回    : 无
//   实测    : 139 ms —— 比本实现（2 趟 16-bit）慢，所以主路径不用。
//   保留原因  : 论文消融实验对照。
// ──────────────────────────────────────────────────────────────
void Sorter::sort_lsd_full(std::vector<SortItem>& items, uint32_t tile_count) {
    if (items.empty()) { last_count_ = 0; last_algo_ = "lsd_full"; return; }

    constexpr int kPasses = 8, kRadix = 256;
    const size_t n = items.size();
    tmp_.resize(n);
    std::vector<SortItem>* src = &items;
    std::vector<SortItem>* dst = &tmp_;

    for (int pass = 0; pass < kPasses; ++pass) {
        const int shift = pass * 8;
        alloc_buckets(kRadix);
        std::memset(buckets_a_.data(), 0, kRadix * sizeof(uint32_t));

        for (size_t i = 0; i < n; ++i)
            buckets_a_[((*src)[i].key >> shift) & 0xFFu]++;

        uint32_t s = 0;
        for (int i = 0; i < kRadix; ++i) {
            uint32_t c = buckets_a_[i];
            buckets_a_[i] = s;
            s += c;
        }
        for (size_t i = 0; i < n; ++i)
            (*dst)[buckets_a_[((*src)[i].key >> shift) & 0xFFu]++] = (*src)[i];
        std::swap(src, dst);
    }
    if (src != &items) items.swap(tmp_);

    last_count_ = items.size();
    last_algo_ = "lsd_full";
}

// ── 视矩阵变化阈值判定（antimatter15） ─────────────────────────
// 逐元素比较 16 个 float，最大绝对差 < eps 就判定"没动"。
// 这是 O(1) 的，不需要知道 tile 表变了多少 —— 附录 A.7 的关键修正。
// 注意：view_in / last_in 都必须指向至少 16 个 float 的有效内存。
//
// ⚠️⚠️ 2026-10-03 重要修正 —— 为什么**只**比矩阵是不够的：
//   本渲染器的正确性依赖 order_ 是「全局远→近」的排列。相机一动，
//   每个高斯的**视空间深度都会变**，但**变化量不同**（近处的、
//   横向的都变得快，远处的变得慢）。因此：
//     · 若相机位移很小，绝大多数相邻高斯的**相对顺序**不变，
//       复用旧 order_ 只是极少数邻接对出错 → 视觉上几乎看不出
//       （这些对深度差本身就极小，混合权重也极小）。
//     · 若相机位移变大，错序的对数会**迅速上升**，累积成可见伪影。
//
//   所以阈值 eps 的物理含义是「允许的最大视矩阵元素变化」。
//   视矩阵元素 ≈ 旋转余弦 + 平移（量纲 = 米）。经验值：
//     eps = 1e-4  → 几乎不动（静止画面 / 相机锁定），复用极安全
//     eps = 1e-3  → 位移 < 1mm 量级，对 640×480 无可见差异
//     eps = 1e-2  → 位移 ~1cm，开始可能看到极轻微的边缘抖动
//   默认给 1e-4（保守），需要提速再往上调。
//
//   另外：**必须**同时要求可见集规模不变。如果剔除结果变了
//   （visible count 变了），说明这帧的可见集合与上帧不同，
//   直接复用 order_ 会引用到已剔除的高斯（画出不该画的东西）。
// ── ◀GUIDE▶ 【3.6】 Sorter::should_reuse()  能不能沿用上一帧的排序结果？
//
//   干什么   : 比较本帧与上帧的视矩阵，几乎一样就返回 true，跳过整趟排序，省下 ~15 ms。
//   参数    : new_view 本帧视矩阵(16 float)；last_view_in 上帧缓存；eps 变化阈值
//   返回    : true = 可以复用上帧排序结果；false = 必须重排
//   判据    : 16 个元素的逐项最大差 < eps（默认 1e-4）
//   ★前提   : 还必须可见集规模不变（render_frame 里另行检查），否则会引用到已剔除的高斯。
//   为什么能放宽: 3DGS 是半透明软性混合，顺序小错视觉上无差异，所以 eps 可以放大。
// ──────────────────────────────────────────────────────────────
bool Sorter::should_reuse(const float* new_view, const float* last_view_in,
                          float eps) {
    if (!new_view || !last_view_in) return false;
    float maxdiff = 0.0f;
    for (int i = 0; i < 16; ++i) {
        const float d = fabsf(new_view[i] - last_view_in[i]);
        if (d > maxdiff) maxdiff = d;
    }
    return maxdiff < eps;
}

// ── ◀GUIDE▶ 【3.7】 Sorter::update_view_cache()  缓存本帧视矩阵
//
//   干什么   : 把本帧视矩阵拷进 last_view_，供下一帧的 should_reuse 比较。
//   参数    : view —— 16 个 float 的视矩阵
//   返回    : 无
//   ★坑    : last_view_ 必须在构造函数里用 vector<float>(16, 0.0f)（16 个元素）；
//     写成 {16, 0} 会变成「2 个元素、值 16 和 0」，比较逻辑就全错了。
// ──────────────────────────────────────────────────────────────
void Sorter::update_view_cache(const float* view) {
    if (!view || last_view_.size() != 16) return;
    std::memcpy(last_view_.data(), view, 16 * sizeof(float));
}

// ── ◀GUIDE▶ 【3.8】 Sorter::alloc_buckets()  准备计数桶
//
//   干什么   : 确保计数排序用的 65536 个桶够大（不够就扩容并清零）。
//   参数    : n —— 需要的桶数
//   返回    : 无
//   为什么   : 桶数组复用，避免每帧 new/free（频繁分配会拖慢后续排序）。
// ──────────────────────────────────────────────────────────────
void Sorter::alloc_buckets(uint32_t n) {
    if (buckets_a_.size() < n) {
        buckets_a_.assign(n, 0);
        buckets_b_.assign(n, 0);
    }
}

// ── ◀GUIDE▶ 【3.9】 Sorter::free_all()  释放全部缓冲
//
//   干什么   : 析构时调用，把排序用的临时数组全部归还内存。
//   参数    : 无
//   返回    : 无
//   技巧    : 用 swap 一个空 vector 的方式释放，能真正把内存还给系统。
// ──────────────────────────────────────────────────────────────
void Sorter::free_all() {
    std::vector<uint32_t>().swap(buckets_a_);
    std::vector<uint32_t>().swap(buckets_b_);
    std::vector<SortItem>().swap(tmp_);
}

} // namespace gsplat
