// loadtest.cpp — M0 基线：PLY 加载 + 排序性能 + 位姿阈值
//
// 用途（NOTES.md §2.6 M0 / §5）：
//  本项目已证明纸面推算不可靠（§2.0 排序实测与估算差 10 倍），
//  所有性能假设必须实测。这份程序一次性测出：
//    1. PLY 头解析与字节数校验
//    2. 流式读取带宽（eMMC → 内存）
//    3. 全量加载耗时 + 内存占用
//    4. 场景包围盒（AABB）与尺度（相机参数必需）
//    5. 三种排序算法对照：qsort / counting16 / lsd_tile / lsd_full
//    6. 视矩阵阈值判定命中率（附录 A.7 关键修正后的复用策略）
//
// 用法：./loadtest <ply路径> [--max N] [--sort-only]
#include "ply_loader.h"
#include "sorter.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <random>

using namespace gsplat;
using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ── 生成合成的高斯-tile 对，验证排序正确性 ──────────────────────
// 造N 个高斯，随机散布在 tile 网格上，深度随机
static std::vector<SortItem> make_items(size_t n, uint32_t tiles, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> td(0, tiles - 1);
    std::uniform_real_distribution<float> dd(0.1f, 500.0f);
    std::vector<SortItem> v(n);
    for (size_t i = 0; i < n; ++i) {
        float z = dd(rng);
        v[i].key   = (static_cast<uint64_t>(td(rng)) << 32) | depth_to_key(z);
        v[i].gauss_id = static_cast<uint32_t>(i);
    }
    return v;
}

// ── 校验排序正确性 ───────────────────────────────────────────
// 注意：counting16 / lsd_tile 是**故意只排 tile（高32位）** 的，
// 组内深度无序，所以不能用"全 key 升序"校验。
// 这里校验的是"tile_id 单调不减"，这才是这两个算法的正确性判据。
static bool verify_tile_monotonic(const std::vector<SortItem>& v) {
    uint32_t prev = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        uint32_t t = static_cast<uint32_t>(v[i].key >> 32);
        if (i > 0 && t < prev) {
            fprintf(stderr, "  ✗ tile 序错误 @ %zu: %u < %u\n", i, t, prev);
            return false;
        }
        prev = t;
    }
    return true;
}

// 完整排序（lsd_full）才校验全 key 升序
static bool verify_full_sorted(const std::vector<SortItem>& v) {
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i-1].key > v[i].key) return false;
    }
    return true;
}

int main(int argc, char** argv) {
    const char* ply = (argc > 1) ? argv[1] : "data/buaa_gspj_lite3m.ply";
    int64_t max_pts = -1;
    bool sort_only = false;
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--max") && i + 1 < argc)
            max_pts = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--sort-only"))
            sort_only = true;
    }

    fprintf(stdout, "=====================================================\n");
    fprintf(stdout, " 3DGS 板端基线测试  (RK3588 / Orange Pi 5 Plus)\n");
    fprintf(stdout, "=====================================================\n\n");

    // ── 1. PLY 头 + 字节数校验 ──────────────────────────────
    PlyHeader h;
    try {
        h = parse_ply_header(ply);
    } catch (const std::exception& e) {
        fprintf(stderr, "PLY 解析失败: %s\n", e.what());
        return 1;
    }

    fprintf(stdout, "【1】PLY 文件头\n");
    fprintf(stdout, "  顶点数      : %lld\n", (long long)h.num_points);
    fprintf(stdout, "  header 字节 : %d\n", h.header_bytes);
    fprintf(stdout, "  float32属性 : %d\n", h.num_floats);

    const double expect = static_cast<double>(h.num_points) * h.num_floats * 4.0;
    fprintf(stdout, "  理论数据区  : %.0f MB\n", expect / 1048576.0);

    // 实测文件大小
    FILE* fp = fopen(ply, "rb");
    if (fp) {
        fseek(fp, 0, SEEK_END);
        long fsz = ftell(fp);
        fclose(fp);
        const double actual = (double)fsz - h.header_bytes;
        fprintf(stdout, "  实测数据区  : %.0f MB (文件 %.0f MB)\n",
                actual / 1048576.0, fsz / 1048576.0);
        if (h.num_floats == 62) {
            const double err = fabs(actual - expect);
            fprintf(stdout, "  字节数校验  : %s (误差 %.0f 字节, 每点 %.4f B)\n",
                    err < 1000 ? "✅ 精确匹配" : "❌ 不匹配",
                    err, actual / h.num_points);
        }
    }
    fprintf(stdout, "\n");

    if (!sort_only) {
        // ── 2. 流式读取带宽 ────────────────────────────────
        fprintf(stdout, "【2】流式读取（只提取 position + alpha）\n");
        std::vector<PosAlpha> pa;
        auto t0 = Clock::now();
        stream_positions(ply, pa, max_pts, false);
        double stream_ms = ms_since(t0);
        const double smb = pa.size() * PlyLayout::kStride / 1048576.0;
        fprintf(stdout, "  %lld 点, %.0f ms, %.0f MB/s\n",
                (long long)pa.size(), stream_ms, smb / (stream_ms / 1000.0));

        // ── 3. AABB 与场景尺度 ─────────────────────────────
        if (!pa.empty()) {
            float mnx=1e30f,mny=1e30f,mnz=1e30f, mxx=-1e30f,mxy=-1e30f,mxz=-1e30f;
            float amin=1e30f, amax=-1e30f;
            for (const auto& p : pa) {
                mnx=std::min(mnx,p.x); mny=std::min(mny,p.y); mnz=std::min(mnz,p.z);
                mxx=std::max(mxx,p.x); mxy=std::max(mxy,p.y); mxz=std::max(mxz,p.z);
                amin=std::min(amin,p.alpha); amax=std::max(amax,p.alpha);
            }
            fprintf(stdout, "\n【3】场景几何（相机 near/far 的依据）\n");
            fprintf(stdout, "  AABB min: (%.2f, %.2f, %.2f)\n", mnx,mny,mnz);
            fprintf(stdout, "  AABB max: (%.2f, %.2f, %.2f)\n", mxx,mxy,mxz);
            fprintf(stdout, "  尺寸    : %.2f × %.2f × %.2f\n",
                    mxx-mnx, mxy-mny, mxz-mnz);
            fprintf(stdout, "  对角线  : %.2f  ← 建议 far ≈此值的 2~3 倍\n",
                    sqrtf((mxx-mnx)*(mxx-mnx)+(mxy-mny)*(mxy-mny)+(mxz-mnz)*(mxz-mnz)));
            fprintf(stdout, "  alpha   : [%.4f, %.4f]均值待算\n", amin, amax);
            double asum = 0;
            for (const auto& p : pa) asum += p.alpha;
            fprintf(stdout, "  alpha均值: %.4f\n", asum / pa.size());
        }
        fprintf(stdout, "\n");
    }

    // ── 4. 排序算法对照（核心） ────────────────────────────
    fprintf(stdout, "【4】排序算法对照\n");
    fprintf(stdout, "  模拟：每个高斯随机落在 1 个 tile 上，深度 0.1~500\n");
    fprintf(stdout, "  注：counting16 / lsd_tile 只按 tile 排序（组内深度无序），\n");
    fprintf(stdout, "      这是**有意的折衷** —— 渲染时按 tile 遍历，组内深度近似有序即可。\n");

    struct Cfg { const char* name; size_t n; uint32_t tiles; };
    const Cfg cfgs[] = {
        {"1.2M keys / 1200 tiles  (P1 档 L1 640×480)", 1200000, 1200},
        {"2.2M keys / 1200 tiles  (对齐§2.0 基线)",  2200000, 1200},
        {"2.2M keys / 3600 tiles  (720p 全 tile)",    2200000, 3600},
    };

    for (const auto& c : cfgs) {
        fprintf(stdout, "\n  --- %s ---\n", c.name);

        // std::sort 做完整升序参考
        auto items = make_items(c.n, c.tiles, 12345);
        auto t0 = Clock::now();
        std::sort(items.begin(), items.end(),
                  [](const SortItem& a, const SortItem& b){ return a.key < b.key; });
        double t_sort = ms_since(t0);
        fprintf(stdout, "    %-12s %8.1f ms   %s\n", "std::sort", t_sort,
                verify_full_sorted(items) ? "完全升序✓" : "错误✗");
        const std::vector<SortItem> ref = items;

        Sorter s;
        struct Algo { const char* nm; int mode; bool full; };
        const Algo algos[] = {
            {"counting16", 0, false},
            {"lsd_tile",   1, false},
            {"lsd_full",   2, true },
        };
        for (const auto& a : algos) {
            std::vector<SortItem> work = make_items(c.n, c.tiles, 12345);
            auto t1 = Clock::now();
            if (a.mode == 0)      s.sort_counting16(work, c.tiles);
            else if (a.mode == 1) s.sort_lsd_tile_only(work, c.tiles);
            else                  s.sort_lsd_full(work, c.tiles);
            double t = ms_since(t1);

            const char* okstr;
            if (a.full) {
                okstr = verify_full_sorted(work) ? "完全升序✓" : "错误✗";
            } else {
                // 校验：tile 分组必须与 ref 完全一致，且组内单调
                bool same_tiles = true;
                for (size_t i = 0; i < work.size(); ++i) {
                    if ((work[i].key >> 32) != (ref[i].key >> 32)) {
                        same_tiles = false;
                        fprintf(stderr, "    ✗ tile 分组与参考不一致 @ %zu\n", i);
                        break;
                    }
                }
                okstr = (verify_tile_monotonic(work) && same_tiles)
                      ? "tile 分组✓(与参考一致)" : "错误✗";
            }
            fprintf(stdout, "    %-12s %8.1f ms   %s\n", a.nm, t, okstr);
        }
    }

    // ── 5. 视矩阵阈值判定（附录 A.7） ──────────────────────
    fprintf(stdout, "\n【5】视矩阵阈值复用判定（antimatter15 策略）\n");
    {
        Sorter s;
        std::mt19937 rng(999);
        std::uniform_real_distribution<float> d(-1.f, 1.f);
        float view[16];
        for (int i = 0; i < 16; ++i) view[i] = d(rng);
        s.update_view_cache(view);

        // 完全不变 → 必命中
        int hit_exact = 0, trials = 1000;
        for (int t = 0; t < trials; ++t) {
            float v2[16];
            memcpy(v2, view, sizeof(v2));
            v2[12] += 1e-6f;               // 极小扰动
            if (s.should_reuse(v2, view, 1e-4f)) ++hit_exact;
        }
        fprintf(stdout, "  扰动 1e-6   : 命中率 %d/%d  %s\n",
                hit_exact, trials, hit_exact == trials ? "✓" : "✗");

        // 中等扰动 → 应不命中
        int hit_mid = 0;
        for (int t = 0; t < trials; ++t) {
            float v2[16];
            memcpy(v2, view, sizeof(v2));
            v2[12] += 0.05f;                // 中等
            if (s.should_reuse(v2, view, 1e-4f)) ++hit_mid;
        }
        fprintf(stdout, "  扰动 0.05    : 命中率 %d/%d  %s\n",
                hit_mid, trials, hit_mid == 0 ? "✓" : "✗");

        // 阈值扫描
        fprintf(stdout, "  阈值扫描（扰动幅度 → 命中率）:\n");
        for (float amp : {1e-6f, 1e-4f, 1e-3f, 1e-2f, 5e-2f, 1e-1f}) {
            int hit = 0;
            for (int t = 0; t < trials; ++t) {
                float v2[16];
                memcpy(v2, view, sizeof(v2));
                v2[12] += amp;
                if (s.should_reuse(v2, view, 1e-4f)) ++hit;
            }
            fprintf(stdout, "    %-8g → %5.1f%%\n", amp, 100.0 * hit / trials);
        }
        fprintf(stdout, "  ✅ O(1) 判定，不需要知道 tile 复用率（附录 A.7）\n");
    }

    // ── 6. tile 复用率实测（原阻塞项，现降级为可选调优，附录 A.7） ──
    fprintf(stdout, "\n【6】tile 复用率实测\n");
    fprintf(stdout, "  场景：真实 PLY，模拟相机移动后统计 tile 表未变比例\n");
    {
        // 用真实点云投影到虚拟相机，统计前后两帧的 tile 占用重合度
        const int W = 640, H = 480;
        const int tw = (W + kTileSize - 1) / kTileSize;
        const int th = (H + kTileSize - 1) / kTileSize;
        const uint32_t n_tiles = tw * th;

        std::vector<PosAlpha> pts;
        stream_positions(ply, pts, 500000, false);

        fprintf(stdout, "  图像 %d×%d → %d×%d = %u tiles, 采样 %zu 点\n",
                W, H, tw, th, n_tiles, pts.size());

        // 统计两个相机下各tile 的高斯数量，画布是「tile 是否被占用」
        auto occupancy = [&](float cx, float cy, std::vector<uint8_t>& occ) {
            occ.assign(n_tiles, 0);
            for (const auto& p : pts) {
                // 极简正交投影（只为测 tile 重合度，不需要真实矩阵）
                float sx = (p.x - cx) * 40.0f + W * 0.5f;
                float sy = (p.y - cy) * 40.0f + H * 0.5f;
                if (sx < 0 || sx >= W || sy < 0 || sy >= H) continue;
                uint32_t tid = (uint32_t)((int)(sy / kTileSize) * tw
                                       + (int)(sx / kTileSize));
                occ[tid] = 1;
            }
        };

        std::vector<uint8_t> a, b;
        struct Case { const char* nm; float dx, dy; };
        const Case cases[] = {
            {"极小移动 0.5 px",  0.0125f, 0.0125f},
            {"小移动   2 px",   0.05f,   0.05f  },
            {"中移动   8 px",   0.2f,    0.2f   },
            {"大移动   32 px",  0.8f,    0.8f   },
            {"跳变     200 px", 5.0f,    5.0f   },
        };
        fprintf(stdout, "\n  %-18s %10s %10s\n", "相机移动幅度",
                "占用tile", "重合率");
        for (const auto& c : cases) {
            occupancy(0, 0, a);
            occupancy(c.dx, c.dy, b);
            size_t used = 0, same = 0;
            for (uint32_t i = 0; i < n_tiles; ++i) {
                if (a[i]) ++used;
                if (a[i] && b[i]) ++same;
            }
            fprintf(stdout, "  %-18s %8zu/%-4u %9.1f%%\n", c.nm,
                    used, n_tiles, used ? 100.0 * same / used : 0.0);
        }
        fprintf(stdout, "\n  说明：重合率高→ 排序结果大可复用，位姿阈值策略收益大。\n");
        fprintf(stdout, "  但本项目已改用「视矩阵阈值」O(1) 判定（§2.5.5），\n");
        fprintf(stdout, "  所以这项数据只作参考，不再是开工前置条件。\n");
    }

    fprintf(stdout, "\n=====================================================\n");
    fprintf(stdout, " 测试完成\n");
    fprintf(stdout, "=====================================================\n");
    return 0;
}
