// ply_loader.cpp — PLY 文件解析（310 行）
//
// ══ 本文件结构 ═══════════════════════════════════════════════════════
//   parse_ply_header   ~15   解析 header：属性名/类型/数量 → 列布局
//   sigmoid            ~131  opacity 的反激活（PLY 里存的是 logit）
//   load_ply_splats    ~134  ★主入口：PLY → std::vector<Splat>
//   stream_positions   ~261  只流式读「位置 + 不透明度」（省内存，建索引用）
//
// ══ 三个关键点 ═══════════════════════════════════════════════════════
//   1. **按属性名定位列，不按固定列号**
//      不同训练框架导出的列序不一样。本项目实测遇到过
//      「f_rest 排在 opacity 之前」的非标准列序，只有按名定位才不出错。
//   2. **SH 档位降到 degree 1**（每高斯 12 个系数，3 DC + 9 rest）
//      丢掉高阶系数：PLY 710 MB → 内存 263 MB（省 400+ MB）
//      对定位任务影响小（下视视角变化小、匹配靠结构不靠颜色）
//      ⚠️ 但"降阶对精度的影响"尚未做消融，是待补项。
//   3. **opacity 存的是 logit**，读入时要过 sigmoid 才是真不透明度。
//
#include "ply_loader.h"
#include <algorithm>       // std::min
#include <chrono>
#include <cmath>
#include <cstdlib>

namespace gsplat {

// ── 头部解析 ────────────────────────────────────────────────────
//
// ⚠️ 关键改动（修 BUG-003）：这里必须把**每个属性的真实名字**都存下来，
//    然后在末尾按名字解析出 PlyColumns。历史上这里只存了为数不多的名字
//    且 loader 硬编码列偏移，遇到「f_rest 排在 opacity 之前」的
//    导出格式就会静默读错列（详见 ply_loader.h 的 PlyColumns 注释）。
PlyHeader parse_ply_header(const char* ply_path, uint32_t* out_props,
                           uint32_t* out_scalar_types) {
    FILE* f = fopen(ply_path, "rb");
    if (!f) throw std::runtime_error(std::string("cannot open ") + ply_path);

    PlyHeader h;
    char line[1024];
    bool in_vertex = false;
    int  prop_index = 0;

    while (fgets(line, sizeof(line), f)) {
        char fmt[64] = {0};
        char kw[64] = {0};

        // format 行：必须是 binary_little_endian
        if (sscanf(line, " format %63s", fmt) == 1) {
            if (strncmp(fmt, "binary_little_endian", 20) != 0) {
                fclose(f);
                throw std::runtime_error(
                    std::string("unsupported PLY format: ") + fmt);
            }
            continue;
        }
        // element 行
        int nelem = 0;
        if (sscanf(line, " element %63s %d", kw, &nelem) == 2) {
            in_vertex = (strcmp(kw, "vertex") == 0);
            if (in_vertex) h.num_points = nelem;
            continue;
        }
        // property 行（只在 vertex 元素下统计）
        // 形如 "property float x" 或 "property float x"（无 list，3DGS 不会用 list）
        if (in_vertex && sscanf(line, " property %63s", kw) == 1) {
            if (prop_index >= 4096) { fclose(f); throw std::runtime_error("too many properties"); }
            const bool is_float = (strcmp(kw, "float") == 0 ||
                                   strcmp(kw, "float32") == 0);
            if (out_scalar_types && prop_index < 64)
                out_scalar_types[prop_index] = is_float ? 1 : 0;
            if (is_float) h.num_floats++;

            // ★ 取出属性名（"property float <name>" 的第三个 token）
            char ptype[64] = {0}, pname[128] = {0};
            if (sscanf(line, " property %63s %127s", ptype, pname) == 2) {
                h.property_names.push_back(pname);
                prop_index++;
            }
            continue;
        }
        if (strncmp(line, "end_header", 10) == 0) {
            h.header_bytes = static_cast<int>(ftell(f));
            fclose(f);
            if (h.num_points <= 0) throw std::runtime_error("num_points not found");
            if (h.num_floats <= 0) throw std::runtime_error("no float properties");
            if ((int)h.property_names.size() != h.num_floats)
                throw std::runtime_error(
                    "property_names 数与 float 属性数不符（有非 float 属性？）");

            // ── 按名字建列映射 ────────────────────────────────────
            auto find = [&](const char* nm) -> int {
                for (size_t i = 0; i < h.property_names.size(); ++i)
                    if (h.property_names[i] == nm) return (int)i;
                return -1;
            };
            auto find_prefix = [&](const char* pf, int n) -> int {
                // 找形如 pf+"<i>" 的连续 n 个属性，返回起始列
                for (int i = 0; i < n; ++i) {
                    char nm[64];
                    snprintf(nm, sizeof(nm), "%s%d", pf, i);
                    if (find(nm) < 0) return -1;
                }
                char nm0[64];
                snprintf(nm0, sizeof(nm0), "%s0", pf);
                return find(nm0);
            };

            PlyColumns& c = h.cols;
            c.x = find("x"); c.y = find("y"); c.z = find("z");
            c.f_dc[0] = find("f_dc_0");
            c.f_dc[1] = find("f_dc_1");
            c.f_dc[2] = find("f_dc_2");
            c.opacity = find("opacity");
            c.scale[0] = find("scale_0");
            c.scale[1] = find("scale_1");
            c.scale[2] = find("scale_2");
            for (int i = 0; i < 4; ++i) {
                char nm[64]; snprintf(nm, sizeof(nm), "rot_%d", i);
                c.rot[i] = find(nm);
            }
            c.f_rest = find_prefix("f_rest_", 3);
            if (c.f_rest >= 0) {
                // 数一下 f_rest_N 连续有多少个
                int n = 0;
                for (;;) {
                    char nm[64];
                    snprintf(nm, sizeof(nm), "f_rest_%d", n);
                    if (find(nm) < 0) break;
                    ++n;
                }
                c.n_rest = n;
            }

            c.detect_2dgs();          // ★ 判定 2DGS（缺 scale_2）
            if (!c.valid())
                throw std::runtime_error(
                    "PLY 缺少必需属性（x/y/z/f_dc_*/opacity/scale_*/rot_*）");
            if (c.is_2dgs)
                fprintf(stdout, "[ply] ★ 检测到 2DGS 格式（仅有 scale_0/scale_1）"
                                "—— 第三轴按 0 处理（退化协方差 = 2D 面元）\n");
            return h;
        }
    }
    fclose(f);
    throw std::runtime_error("end_header not found — not a valid PLY?");
}

// ── sigmoid / exp 预变换 ───────────────────────────────────────
static inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// ── 全量加载 ────────────────────────────────────────────────────
// ── ◀GUIDE▶ 【4.8】 load_ply_splats()  ★ 读 3DGS 模型（PLY）
//
//   干什么   : 把 .ply 文件里的几百万个高斯读成内存数组（位置/尺度/旋转/不透明度/颜色）。
//   参数    : ply_path 文件路径；out_splats 输出高斯数组；out_shcs 输出球谐颜色；
//     sh_degree 球谐阶数（本项目降到 1）；quiet 是否少打印
//   返回    : true 成功 / false 失败
//   耗时    : 709 MB，395 ms（约 1800 MB/s，接近内存带宽上限）。
//   ★坑1   : opacity 在 PLY 里存的是 **logit 值**，必须 sigmoid 之后才是 0~1 的不透明度。
//   ★坑2   : scale 存的是 **log 值**，必须 exp 之后才是真实标准差。
//   ★坑3   : **按列名**定位，不能按固定列号 —— 不同导出工具的列顺序会变
//     （本项目实测就有 f_rest 排在 opacity 之前的情况）。
// ──────────────────────────────────────────────────────────────
bool load_ply_splats(const char* ply_path, std::vector<Splat>& out_splats,
                     std::vector<ShCoeffs>& out_sh, int sh_degree, bool verbose, bool* out_is_2dgs) {
    auto t0 = std::chrono::steady_clock::now();

    const int rest_per_degree[4] = {0, 9, 24, 45};
    if (sh_degree < 0) sh_degree = 0;
    if (sh_degree > 3) sh_degree = 3;
    const int n_rest = rest_per_degree[sh_degree];

    PlyHeader h;
    uint32_t props[64] = {0};
    uint32_t types[64] = {0};
    h = parse_ply_header(ply_path, props, types);
    if (out_is_2dgs) *out_is_2dgs = h.cols.is_2dgs;   // ★ 回传 2DGS 标记

    const int NCOL = h.num_floats;          // 每点的 float 数（本 PLY 是 62）
    const PlyColumns& col = h.cols;

    if (verbose) {
        fprintf(stdout, "[ply] 列布局: x=%d f_dc=%d opacity=%d scale=%d rot=%d "
                        "f_rest=%d(+%d)  共 %d 列\n",
                col.x, col.f_dc[0], col.opacity, col.scale[0], col.rot[0],
                col.f_rest, col.n_rest, NCOL);
        // 若 f_rest 出现在 opacity 之前，特别提示 —— 这正是 BUG-003 的场景
        if (col.f_rest >= 0 && col.f_rest < col.opacity)
            fprintf(stdout, "[ply] ⚠️ 检测到非标准列序（f_rest 在 opacity 之前），"
                            "已按名字定位，无需担心\n");
    }

    const int64_t n = h.num_points;
    out_splats.resize(n);
    out_sh.resize(n);

    FILE* f = fopen(ply_path, "rb");
    if (!f) return false;
    fseek(f, h.header_bytes, SEEK_SET);

    // 分块读。⚠️ 关键：块大小**必须是 NCOL 的整数倍**。
    //    否则 fread 读回 float 数不是 NCOL 的倍数，`got / NCOL` 取整会
    //    每块丢掉最多 NCOL-1 个 float（≈1 个点）——丢的点会让后续所有
    //    点整体错位，是比"报错返回 false"更危险的静默数据损坏。
    constexpr size_t kChunkBytes   = 4 * 1024 * 1024;
    const size_t kChunkPoints = kChunkBytes / ((size_t)NCOL * sizeof(float));
    const size_t kChunkFloats = kChunkPoints * (size_t)NCOL;
    if (kChunkPoints == 0) { fclose(f); return false; }
    std::vector<float> buf(kChunkFloats);

    int64_t done = 0;
    int64_t next_report = 500000;
    while (done < n) {
        size_t want = static_cast<size_t>(std::min<int64_t>(
            (int64_t)kChunkFloats, (n - done) * NCOL));
        size_t got = fread(buf.data(), sizeof(float), want, f);
        size_t pts = got / NCOL;
        if (pts == 0) break;

        for (size_t i = 0; i < pts; ++i) {
            const float* p = buf.data() + i * NCOL;
            Splat& s = out_splats[done + i];
            s.x = p[col.x]; s.y = p[col.y]; s.z = p[col.z];
            s.opacity = sigmoid(p[col.opacity]);
            s.scale0 = expf(p[col.scale[0]]);
            s.scale1 = expf(p[col.scale[1]]);
            // ★ 2DGS：缺少 scale_2 时第三轴取 0（col.scale[2] == -1）
            //   → compute_cov3d 里 sc[2]=0 → Σ = R·diag(s0²,s1²,0)·Rᵀ（退化协方差）
            s.scale2 = (col.scale[2] >= 0) ? expf(p[col.scale[2]]) : 0.0f;
            float qw = p[col.rot[0]], qx = p[col.rot[1]];
            float qy = p[col.rot[2]], qz = p[col.rot[3]];
            float qn = sqrtf(qw*qw + qx*qx + qy*qy + qz*qz);
            if (qn > 1e-12f) { qw/=qn; qx/=qn; qy/=qn; qz/=qn; }
            s.rot0 = qw; s.rot1 = qx; s.rot2 = qy; s.rot3 = qz;

            ShCoeffs& c = out_sh[done + i];
            c.dc[0] = p[col.f_dc[0]];
            c.dc[1] = p[col.f_dc[1]];
            c.dc[2] = p[col.f_dc[2]];
            if (n_rest > 0 && col.f_rest >= 0) {
                // 取前 n_rest 个 f_rest。注意 3DGS 的 f_rest 是
                // **按通道分组**存储：f_rest_0..14 = R 的 15 个系数，
                // 15..29 = G，30..44 = B（degree 3 时）。
                // 我们要的「1 阶」是每通道的第 0..2 个，即列
                //   f_rest_0,1,2 (R) / f_rest_15,16,17 (G) / f_rest_30,31,32 (B)
                // 这里做一个「按通道 stride 取 3 个」的映射，
                // 顺带兼容 degree 1（9 个 rest，无 stride）的情况。
                if (col.n_rest >= 45) {
                    const int stride = col.n_rest / 3;   // 45/3 = 15
                    for (int ch = 0; ch < 3; ++ch)
                        for (int k = 0; k < 3 && k < n_rest/3; ++k)
                            c.rest[ch*3 + k] = p[col.f_rest + ch*stride + k];
                } else {
                    memcpy(c.rest, p + col.f_rest, n_rest * sizeof(float));
                }
            }
        }
        done += pts;
        if (verbose && done >= next_report) {
            fprintf(stderr, "  ... %lld / %lld (%.0f%%)\n",
                    (long long)done, (long long)n, 100.0 * done / n);
            while (next_report <= done) next_report += 500000;
        }
    }
    fclose(f);

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const double bytes = static_cast<double>(n) * PlyLayout::kStride;
    if (verbose) {
        fprintf(stdout, "\n=== PLY 加载完成 ===\n");
        fprintf(stdout, "点数        : %lld\n", (long long)n);
        fprintf(stdout, "header      : %d 字节\n", h.header_bytes);
        fprintf(stdout, "属性数      : %d float32\n", h.num_floats);
        fprintf(stdout, "原始数据    : %.1f MB (%.0f B/点)\n",
                bytes / 1048576.0, (double)PlyLayout::kStride);
        fprintf(stdout, "SH 档位     : degree %d (%d 个 rest系数)\n",
                sh_degree, n_rest);
        fprintf(stdout, "内存占用    : Splat %.1f MB + SH %.1f MB = %.1f MB\n",
                n * sizeof(Splat) / 1048576.0,
                n * sizeof(ShCoeffs) / 1048576.0,
                n * (sizeof(Splat) + sizeof(ShCoeffs)) / 1048576.0);
        fprintf(stdout, "耗时        : %.0f ms (%.0f MB/s)\n",
                ms, bytes / 1048576.0 / (ms / 1000.0));
    }
    return done == n;
}

// ── 流式读位置 + alpha（loadtest 用） ──────────────────────────
// ── ◀GUIDE▶ 【工具】 stream_positions()  只流式读「位置+不透明度」
//
//   干什么   : 只读需要的那两列，不解析整个文件（省内存/省时间）。
//   参数    : ply_path 路径；out 输出 PosAlpha 数组；keep 最多读多少个
//   返回    : true 成功
//   用在哪   : 做统计、可视化、粗定位等只需要点云位置的场合。
// ──────────────────────────────────────────────────────────────
bool stream_positions(const char* ply_path, std::vector<PosAlpha>& out,
                      int64_t max_points, bool verbose) {
    auto t0 = std::chrono::steady_clock::now();

    PlyHeader h = parse_ply_header(ply_path);
    const int NCOL = h.num_floats;
    const PlyColumns& col = h.cols;
    const int64_t total = h.num_points;
    const int64_t n = (max_points > 0 && max_points < total) ? max_points : total;
    out.resize(n);

    FILE* f = fopen(ply_path, "rb");
    if (!f) return false;
    fseek(f, h.header_bytes, SEEK_SET);

    // ⚠️ 必须按 NCOL 对齐块大小（同 load_ply_splats），否则丢点在块边界
    constexpr size_t kChunkBytes = 8 * 1024 * 1024;
    const size_t kChunkPoints = kChunkBytes / ((size_t)NCOL * sizeof(float));
    const size_t chunk_floats = kChunkPoints * (size_t)NCOL;
    if (kChunkPoints == 0) { fclose(f); return false; }
    std::vector<float> buf(chunk_floats);

    int64_t done = 0;
    while (done < n) {
        size_t want = static_cast<size_t>(
            std::min<int64_t>((int64_t)chunk_floats, (n - done) * NCOL));
        size_t got = fread(buf.data(), sizeof(float), want, f);
        size_t pts = got / (size_t)NCOL;
        if (pts == 0) break;
        for (size_t i = 0; i < pts; ++i) {
            const float* p = buf.data() + i * NCOL;
            out[done + i] = {p[col.x], p[col.y], p[col.z],
                             sigmoid(p[col.opacity])};
        }
        done += pts;
    }
    fclose(f);

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (verbose) {
        const double bytes = static_cast<double>(done) * PlyLayout::kStride;
        fprintf(stdout, "流式读取 %lld 点: %.0f ms (%.0f MB/s 有效)\n",
                (long long)done, ms,
                bytes / 1048576.0 / (ms / 1000.0));
    }
    return done == n;
}

} // namespace gsplat
