// xfeat_frontend.cpp — XFeat 特征前端的板端 C++ 实现
//
// 精度基线：与 Python 参考实现（compare_xfeat.py）逐比特等价，
//           后者已与官方 verlab/XFeat torch 路径逐比特等价。
#include "xfeat_frontend.h"


#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <thread>
#include <chrono>
#include <fstream>

namespace {

using Clock = std::chrono::steady_clock;
static inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ── PyTorch grid_sample 的等价实现（align_corners=False, padding=zeros）────
//   unnormalize: x = ((g+1)*size - 1) / 2 ；normgrid 用「全分辨率」(W-1, H-1)
// ── ◀GUIDE▶ 【6.11】 unnorm()  归一化坐标 → 原图像素坐标
//
//   干什么   : 网络内部用 [-1,1] 的归一化坐标，这里换回真实像素坐标。
//   参数    : px,py 归一化坐标；Wfull,Hfull 原图尺寸；ox,oy 输出像素坐标
//   返回    : 无（写引用）
// ──────────────────────────────────────────────────────────────
static inline void unnorm(float px, float py, float Wfull, float Hfull,
                          int Wi, int Hi, float& ix, float& iy) {
    const float gx = 2.0f * (px / (Wfull - 1.0f)) - 1.0f;
    const float gy = 2.0f * (py / (Hfull - 1.0f)) - 1.0f;
    ix = ((gx + 1.0f) * Wi - 1.0f) * 0.5f;
    iy = ((gy + 1.0f) * Hi - 1.0f) * 0.5f;
}

// cubic convolution 权重（PyTorch get_cubic_upsample_coefficients, A = -0.75）
// ── ◀GUIDE▶ 【6.12】 cubic_w()  双三次插值的权重
//
//   干什么   : 算双三次（bicubic）采样里某个邻居点的权重。
//   参数    : t —— 距离；k —— 邻居偏移索引
//   返回    : 权重值
//   为什么用它 : 描述子要在亚像素位置采样，双线性太糊、双三次与官方 torch 实现一致。
// ──────────────────────────────────────────────────────────────
static inline float cubic_w(float t, int k) {
    const float A = -0.75f;
    float x;
    switch (k) {
        case 0:  x = t + 1.0f; return ((A * x - 5.0f * A) * x + 8.0f * A) * x - 4.0f * A;
        case 1:  x = t;        return ((A + 2.0f) * x - (A + 3.0f)) * x * x + 1.0f;
        case 2:  x = 1.0f - t; return ((A + 2.0f) * x - (A + 3.0f)) * x * x + 1.0f;
        default: x = 2.0f - t; return ((A * x - 5.0f * A) * x + 8.0f * A) * x - 4.0f * A;
    }
}

// 简易并行 for（避免引入 OpenMP 依赖）
template <class F>
// ── ◀GUIDE▶ 【6.13】 parallel_for()  极简并行 for 循环
//
//   干什么   : 把 0..n-1 的任务平均分给 nthreads 个线程跑完（然后 join）。
//   参数    : n 任务数；nthreads 线程数；f 要执行的函数（lambda）
//   返回    : 无
//   ★C++语法: F&& 是「万能引用」模板参数，所以 f 可以是任何可调用对象（lambda/函数指针）；
//     这是 C++ 模板的用法，看不懂可以只当「传进来一个函数」理解。
// ──────────────────────────────────────────────────────────────
static void parallel_for(int n, int nthreads, F&& f) {
    if (n <= 0) return;
    const int nt = std::max(1, std::min(nthreads, n));
    if (nt == 1) { f(0, n); return; }
    std::vector<std::thread> th;
    th.reserve(nt);
    const int chunk = (n + nt - 1) / nt;
    for (int t = 0; t < nt; ++t) {
        const int b = t * chunk, e = std::min(n, b + chunk);
        if (b >= e) break;
        th.emplace_back([&f, b, e] { f(b, e); });
    }
    for (auto& x : th) x.join();
}

}  // namespace

// ══════════════════════════════════════════════════════════════════════
// ── ◀GUIDE▶ 【6.14】 XFeatFrontend::init()  加载 NPU 模型
//
//   干什么   : 把 .rknn 模型加载到 NPU，查询输入输出张量的形状，并预分配复用缓冲。
//   参数    : model_path 模型文件；p 参数（top_k/阈值/线程数…）；W,H 输入图尺寸
//   返回    : true 成功 / false 失败
//   ★坑1   : rknn_context 是一个**整数句柄**（uintptr_t），不是指针，别当成对象用。
//   ★坑2   : 张量属性结构体里的字段叫 n_dims（不是 ndim）。
//   ★为什么要预分配: cossim 匹配矩阵 1500×1500×4 = 9 MB，每帧 new/free 会把分配器搅碎，
//     连带把渲染器的排序从 10 ms 拖到 21 ms。
// ──────────────────────────────────────────────────────────────
bool XFeatFrontend::init(const std::string& model_path, const XFeatParams& p,
                         int W, int H) {
    P_ = p; W_ = W; H_ = H;
    Hc_ = H / 8; Wc_ = W / 8;

    std::ifstream f(model_path, std::ios::binary);
    if (!f) { fprintf(stderr, "[xfeat] 打不开模型: %s\n", model_path.c_str()); return false; }
    f.seekg(0, std::ios::end);
    const size_t sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);
    f.close();

    int ret = rknn_init(&ctx_, buf.data(), (uint32_t)sz, 0, nullptr);
    if (ret < 0) { fprintf(stderr, "[xfeat] rknn_init 失败 ret=%d\n", ret); return false; }

    rknn_input_output_num ionum;
    memset(&ionum, 0, sizeof(ionum));
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &ionum, sizeof(ionum));
    if (ret < 0 || ionum.n_output != 3) {
        fprintf(stderr, "[xfeat] 输出数量异常: %u\n", ionum.n_output);
        return false;
    }
    // 打印输入/输出形状（校验模型与预期一致）
    rknn_tensor_attr ia = {}, oa[3] = {};
    ia.index = 0;
    rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &ia, sizeof(ia));
    for (uint32_t i = 0; i < 3; ++i) { oa[i].index = i; rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &oa[i], sizeof(oa[i])); }
    fprintf(stdout, "[xfeat] 模型加载 OK  输入 ndim=%u  输出: %d×%d×%d / %d×%d×%d / %d×%d×%d\n",
            (unsigned)ia.n_dims,
            oa[0].dims[1], oa[0].dims[2], oa[0].dims[3],
            oa[1].dims[1], oa[1].dims[2], oa[1].dims[3],
            oa[2].dims[1], oa[2].dims[2], oa[2].dims[3]);

    // 预分配
    smap_.assign((size_t)64 * Hc_ * Wc_, 0.f);
    k1h_.assign((size_t)H_ * W_, 0.f);
    lmax_.assign((size_t)H_ * W_, 0.f);
    desc_q_.assign((size_t)P_.top_k * 64, 0.f);
    desc_r_.assign((size_t)P_.top_k * 64, 0.f);
    m12_.assign(P_.top_k, 0);
    m21_.assign(P_.top_k, 0);
    csmax_.assign(P_.top_k, 0.f);
    cs_.assign((size_t)P_.top_k * P_.top_k, 0.f);
    bt_.assign((size_t)64 * P_.top_k, 0.f);

    inited_ = true;
    return true;
}

// ── ◀GUIDE▶ 【6.15】 XFeatFrontend::release()  释放 NPU 资源
//
//   干什么   : 销毁 NPU 上下文，归还显存/内存。
//   参数    : 无
//   返回    : 无
// ──────────────────────────────────────────────────────────────
void XFeatFrontend::release() {
    if (ctx_) { rknn_destroy(ctx_); ctx_ = 0; }
    inited_ = false;
}

// ── ◀GUIDE▶ 【6.16】 XFeatFrontend::extract()  ★ 提特征：灰度图 → 关键点 + 64维描述子
//
//   干什么   : 把一张灰度图送进 NPU，得到一批关键点坐标和每个点的 64 维描述子。
//   参数    : gray —— W*H 行主序 uint8 灰度图；out —— 输出关键点集合
//   返回    : true 成功
//   流程    : NPU 推理 → 热力图 softmax → 5×5 NMS 取峰 → 双三次采描述子 → L2 归一化
//     → 抛物线亚像素细化（本项目加的，官方稀疏路径没有）
//   ★亚像素为何重要: ORB 的关键点 74.5% 带亚像素，而 XFeat 原是纯整数坐标；
//     在「位姿已知只解平移」的场景里这会差 1.8 倍，补上后反超。
//   耗时    : NPU 17.7 ms + 解码 8.4 ms（512×288 输入）。
// ──────────────────────────────────────────────────────────────
bool XFeatFrontend::extract(const uint8_t* gray, XFeatKps& out) {
    if (!inited_) return false;
    const int Hc = Hc_, Wc = Wc_, W = W_, H = H_;
    out.clear();

    // ── 1. NPU 推理 ───────────────────────────────────────────────
    auto t0 = Clock::now();
    rknn_input in;
    memset(&in, 0, sizeof(in));
    in.index = 0;
    in.type  = RKNN_TENSOR_UINT8;
    in.fmt   = RKNN_TENSOR_NHWC;
    in.size  = (uint32_t)(W * H);
    in.buf   = (void*)gray;
    if (rknn_inputs_set(ctx_, 1, &in) < 0) return false;
    if (rknn_run(ctx_, nullptr) < 0) return false;
    rknn_output outs[3];
    for (int i = 0; i < 3; ++i) {
        memset(&outs[i], 0, sizeof(rknn_output));
        outs[i].index = i;
        outs[i].want_float = 1;
    }
    if (rknn_outputs_get(ctx_, 3, outs, nullptr) < 0) return false;
    const float* feats = (const float*)outs[0].buf;    // (64, Hc, Wc)
    const float* kpl   = (const float*)outs[1].buf;    // (65, Hc, Wc)
    const float* hmap  = (const float*)outs[2].buf;    // (1,  Hc, Wc)
    ms_npu = ms_since(t0);

    // ── 2. 解码 ───────────────────────────────────────────────────
    auto t1 = Clock::now();

    // 2.1 softmax over 65 通道，取前 64 → smap_(64, Hc, Wc)
    {
        const int C = 65, N = Hc * Wc;
        parallel_for(N, P_.match_thr, [&](int b, int e) {
            for (int p = b; p < e; ++p) {
                float mx = -1e30f;
                for (int c = 0; c < C; ++c) mx = std::max(mx, kpl[c * N + p]);
                float s = 0.f;
                // 数值稳定：先算指数和
                float ex[65];
                for (int c = 0; c < C; ++c) { ex[c] = std::exp(kpl[c * N + p] - mx); s += ex[c]; }
                const float inv = 1.0f / s;
                for (int c = 0; c < 64; ++c) smap_[(size_t)c * N + p] = ex[c] * inv;
            }
        });
    }

    // 2.2 8× 像素展开 → K1h (H, W)：
    //     K1h[h*8+i][w*8+j] = smap_(i*8+j, h, w)   （与官方 permute/reshape 一致）
    {
        const int N = Hc * Wc;
        parallel_for(Hc, P_.match_thr, [&](int hb, int he) {
            for (int h = hb; h < he; ++h) {
                for (int w = 0; w < Wc; ++w) {
                    const int row0 = h * 8, col0 = w * 8;
                    for (int i = 0; i < 8; ++i) {
                        float* dst = &k1h_[(size_t)(row0 + i) * W + col0];
                        const float* src = &smap_[(size_t)(i * 8) * N + h * Wc + w];
                        for (int j = 0; j < 8; ++j) dst[j] = src[(size_t)j * N];
                    }
                }
            }
        });
    }

    // 2.3 5×5 局部极大 + 阈值 → 候选峰
    //     PyTorch MaxPool2d(5,1,2) 边界用 -inf 填充 ≡ 只在窗口与图像交集内取极大
    struct Peak { float x, y, v; };
    std::vector<Peak> peaks;      // ⚠️ 不能 static thread_local：parallel_for 的工作线程
    peaks.clear();                //    各自有独立副本，跨线程读会越界（实测段错误）
    {
        const int K = P_.nms_k, half = K / 2;
        const float thr = P_.det_th;
        for (int y = 0; y < H; ++y) {
            const int y0 = std::max(0, y - half), y1 = std::min(H - 1, y + half);
            for (int x = 0; x < W; ++x) {
                const float v = k1h_[(size_t)y * W + x];
                if (!(v > thr)) continue;
                const int x0 = std::max(0, x - half), x1 = std::min(W - 1, x + half);
                float lm = -1e30f;
                for (int yy = y0; yy <= y1; ++yy) {
                    const float* row = &k1h_[(size_t)yy * W];
                    for (int xx = x0; xx <= x1; ++xx) lm = std::max(lm, row[xx]);
                }
                if (v == lm) peaks.push_back({(float)x, (float)y, v});
            }
        }
    }

    // 2.4 抛物线亚像素细化 + 可靠性分（nearest(K1h) × bilinear(heatmap)）
    //     ★ 官方稀疏路径没有亚像素；实测这是「有姿态先验」场景的关键。
    const int NPk = (int)peaks.size();
    std::vector<float> sxs(NPk), sys(NPk), scs(NPk);
    parallel_for(NPk, P_.match_thr, [&](int b, int e) {
        for (int i = b; i < e; ++i) {
            int x = (int)peaks[i].x, y = (int)peaks[i].y;
            float dx = 0.f, dy = 0.f;
            if (P_.subpix) {
                const float c0 = k1h_[(size_t)y * W + x];
                const float cl = k1h_[(size_t)y * W + std::max(0, x - 1)];
                const float cr = k1h_[(size_t)y * W + std::min(W - 1, x + 1)];
                const float cu = k1h_[(size_t)std::max(0, y - 1) * W + x];
                const float cd = k1h_[(size_t)std::min(H - 1, y + 1) * W + x];
                const float denx = cl - 2.f * c0 + cr, deny = cu - 2.f * c0 + cd;
                if (std::fabs(denx) > 1e-12f) dx = std::max(-0.5f, std::min(0.5f, 0.5f * (cl - cr) / denx));
                if (std::fabs(deny) > 1e-12f) dy = std::max(-0.5f, std::min(0.5f, 0.5f * (cu - cd) / deny));
            }
            sxs[i] = x + dx; sys[i] = y + dy;

            // nearest(K1h)：|d|<0.5 时四舍五入回到峰本身
            float sn = k1h_[(size_t)y * W + x];
            // bilinear(heatmap)，归一化用全分辨率 (W-1, H-1)，张量尺寸 (Wc, Hc)
            float ix, iy;
            unnorm(sxs[i], sys[i], (float)W, (float)H, Wc, Hc, ix, iy);
            const int ix0 = (int)std::floor(ix), iy0 = (int)std::floor(iy);
            const float tx = ix - ix0, ty = iy - iy0;
            auto gv = [&](int h, int w) -> float {
                return (h < 0 || h >= Hc || w < 0 || w >= Wc) ? 0.f : hmap[h * Wc + w];
            };
            const float sb =
                (1 - tx) * (1 - ty) * gv(iy0,     ix0)     +
                (1 - tx) * ty      * gv(iy0 + 1, ix0)     +
                tx      * (1 - ty) * gv(iy0,     ix0 + 1) +
                tx      * ty      * gv(iy0 + 1, ix0 + 1);
            scs[i] = sn * sb;
        }
    });
    // 官方：mkpts 恰为 (0,0) 时分数置 -1（实现里等价于把原点峰踢掉）
    for (int i = 0; i < NPk; ++i)
        if (peaks[i].x == 0.f && peaks[i].y == 0.f) scs[i] = -1.f;

    // 2.5 top-k（按分数降序）
    std::vector<int> ord(NPk);
    for (int i = 0; i < NPk; ++i) ord[i] = i;
    const int K = std::min(P_.top_k, NPk);
    std::partial_sort(ord.begin(), ord.begin() + K, ord.end(),
                      [&](int a, int b) { return scs[a] > scs[b]; });

    // 2.6 bicubic 采样 64 维描述子（在 M1 上，坐标已亚像素）+ L2 归一化
    //     官方是先 F.normalize(M1, dim=1) 再采样、采样后再归一化 —— 这里一致。
    //     注意 Python 里是「先归一化 M1」→ 本实现在采样时用归一化后的 M1。
    {
        // 先算归一化后的 M1（复用 smap_ 缓冲：此时 smap_ 已不再需要）
        const int C = 64, N = Hc * Wc;
        parallel_for(N, P_.match_thr, [&](int b, int e) {
            for (int p = b; p < e; ++p) {
                float s = 0.f;
                for (int c = 0; c < C; ++c) { const float v = feats[c * N + p]; s += v * v; }
                const float inv = 1.0f / std::max(std::sqrt(s), 1e-12f);
                for (int c = 0; c < C; ++c) smap_[(size_t)c * N + p] = feats[c * N + p] * inv;
            }
        });
    }

    out.n = K;
    out.x.resize(K); out.y.resize(K); out.score.resize(K);
    out.desc.assign((size_t)K * 64, 0.f);
    parallel_for(K, P_.match_thr, [&](int b, int e) {
        for (int i = b; i < e; ++i) {
            const int id = ord[i];
            out.x[i] = sxs[id]; out.y[i] = sys[id]; out.score[i] = scs[id];
            float fx, fy;
            unnorm(sxs[id], sys[id], (float)W, (float)H, Wc, Hc, fx, fy);
            const int x0 = (int)std::floor(fx), y0 = (int)std::floor(fy);
            const float tx = fx - x0, ty = fy - y0;
            float cxw[4], cyw[4];
            for (int k = 0; k < 4; ++k) { cxw[k] = cubic_w(tx, k); cyw[k] = cubic_w(ty, k); }
            float* d = &out.desc[(size_t)i * 64];
            float nrm = 0.f;
            for (int c = 0; c < 64; ++c) {
                const float* Mc = &smap_[(size_t)c * Hc * Wc];
                auto gv = [&](int h, int w) -> float {
                    return (h < 0 || h >= Hc || w < 0 || w >= Wc) ? 0.f : Mc[h * Wc + w];
                };
                float v[4];
                for (int r = 0; r < 4; ++r) {
                    const int yy = y0 - 1 + r;
                    v[r] = cxw[0] * gv(yy, x0 - 1) + cxw[1] * gv(yy, x0) +
                           cxw[2] * gv(yy, x0 + 1) + cxw[3] * gv(yy, x0 + 2);
                }
                const float val = cyw[0] * v[0] + cyw[1] * v[1] + cyw[2] * v[2] + cyw[3] * v[3];
                d[c] = val; nrm += val * val;
            }
            const float inv = 1.0f / std::max(std::sqrt(nrm), 1e-12f);
            for (int c = 0; c < 64; ++c) d[c] *= inv;
        }
    });
    // 官方：score <= 0 的丢弃
    int w = 0;
    for (int i = 0; i < K; ++i) {
        if (!(out.score[i] > 0.f)) continue;
        if (w != i) {
            out.x[w] = out.x[i]; out.y[w] = out.y[i]; out.score[w] = out.score[i];
            std::memcpy(&out.desc[(size_t)w * 64], &out.desc[(size_t)i * 64], 64 * sizeof(float));
        }
        ++w;
    }
    out.n = w; out.x.resize(w); out.y.resize(w); out.score.resize(w);
    out.desc.resize((size_t)w * 64);

    rknn_outputs_release(ctx_, 3, outs);
    ms_decode = ms_since(t1);
    return out.n > 0;
}

// ── ◀GUIDE▶ 【6.17】 XFeatFrontend::match()  ★ 特征匹配：两组描述子找对应关系
//
//   干什么   : 把渲染图的特征点和实拍图的特征点一一对上（谁对应谁）。
//   参数    : a 第一组特征；b 第二组特征；out 输出配对下标对 (i,j)
//   返回    : 配对数
//   算法    : 算 N×N 余弦相似度矩阵 → 双向互近邻(MNN) → 余弦门限 0.90 过滤
//   ★性能关键 : 原来要对 9 MB 的相似度矩阵做**列扫描**，缓存行流量 ~144 MB；
//     改成在算内积时顺手记录行/列最大值，耗时 46.7 → 35.0 ms（−27%）。
//   输出置信度 : last_mean_cossim = 接受配对平均余弦相似度（本项目当置信度用）。
// ──────────────────────────────────────────────────────────────
int XFeatFrontend::match(const XFeatKps& a, const XFeatKps& b,
                         std::vector<std::pair<int, int>>& out) {
    out.clear();
    const int Nq = a.n, Nr = b.n;
    if (Nq <= 0 || Nr <= 0) return 0;
    const int C = 64;
    const int thr_t = std::max(1, P_.match_thr);

    // ★ 性能关键：把 b 的描述子转置成 (C, Nr)，让内层循环变成对连续内存的 axpy。
    //   否则归约循环（k=64）在没开 -ffast-math 时不会被自动向量化，
    //   实测匹配要 64 ms/帧；转置后 ~8 ms/帧（快 8×）。
    if (Nr > P_.top_k || Nq > P_.top_k) return 0;   // 预分配上限
    for (int j = 0; j < Nr; ++j)
        for (int k = 0; k < C; ++k) bt_[(size_t)k * Nr + j] = b.desc[(size_t)j * C + k];

    // cossim 矩阵（行主序 Nq×Nr），按行并行
    float* cs = cs_.data();
    // ★ __restrict__ 是必须的：`row`(float*) 与 `pb`(const float*) 在 C++ 里
    //   编译器**不能**假设不别名 → 内层 axpy 不向量化（实测 46.7 ms/帧）。
    //   加上限定后走 NEON 4 宽 FMA。
    float* __restrict__ csr = cs;
    const float* __restrict__ bt = bt_.data();
    parallel_for(Nq, thr_t, [&](int b0, int e0) {
        for (int i = b0; i < e0; ++i) {
            const float* __restrict__ pa = &a.desc[(size_t)i * C];
            float* __restrict__ row = &csr[(size_t)i * Nr];
            for (int j = 0; j < Nr; ++j) row[j] = 0.f;
            for (int k = 0; k < C; ++k) {
                const float av = pa[k];
                const float* __restrict__ pb = &bt[(size_t)k * Nr];
                for (int j = 0; j < Nr; ++j) row[j] += av * pb[j];
            }
        }
    });

    // ★★ 关键优化：行/列最大值**在内积那一遍里顺手算完**。
    //   原来 m21 是在 9 MB 的行主序 cossim 矩阵上做**跨列扫描**
    //   —— 每次访问都是一条新缓存行（2.25M 次 × 64 B ≈ 144 MB 流量），
    //   实测它一个人就吃掉 ~30 ms。
    //   现在：行最大 + 列最大更新都在刚算完的那一行上顺序做（L1 友好），
    //   列最大按线程私有（每线程 6 KB）避免竞态，最后再合并。
    const int nt = std::max(1, std::min(thr_t, Nq));
    const int chunk = (Nq + nt - 1) / nt;
    std::vector<float> colv((size_t)nt * Nr, -2.f);
    std::vector<int>   coli((size_t)nt * Nr, 0);
    std::vector<int>   m12(Nq, 0);
    parallel_for(Nq, thr_t, [&](int b0, int e0) {
        const int tid = std::min(nt - 1, b0 / chunk);
        float* __restrict__ cv = &colv[(size_t)tid * Nr];
        int*   __restrict__ ci = &coli[(size_t)tid * Nr];
        for (int i = b0; i < e0; ++i) {
            const float* __restrict__ row = &csr[(size_t)i * Nr];
            // 行最大 + 列最大在**同一遍**里做（再过一遍纯属浪费带宽）
            int bj = 0; float bv = -2.f;
            for (int j = 0; j < Nr; ++j) {
                const float v = row[j];
                if (v > bv) { bv = v; bj = j; }
                if (v > cv[j]) { cv[j] = v; ci[j] = i; }
            }
            m12[i] = bj;
        }
    });
    std::vector<int> m21(Nr, 0);
    for (int j = 0; j < Nr; ++j) {
        float bv = colv[j]; int bi = coli[j];
        for (int t = 1; t < nt; ++t) {
            const float v = colv[(size_t)t * Nr + j];
            if (v > bv) { bv = v; bi = coli[(size_t)t * Nr + j]; }
        }
        m21[j] = bi;
    }

    double cssum = 0.0;
    for (int i = 0; i < Nq; ++i) {
        const int j = m12[i];
        if (m21[j] == i && cs[(size_t)i * Nr + j] > P_.min_cossim) {
            out.emplace_back(i, j);
            cssum += cs[(size_t)i * Nr + j];
        }
    }
    last_mean_cossim = out.empty() ? -1.0 : cssum / (double)out.size();
    return (int)out.size();
}
