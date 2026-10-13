// xfeat_frontend.h — XFeat 特征前端（RKNN NPU 推理 + 官方解码链 + MNN 匹配）
//
// 设计目标：与 Python 参考实现 board_tools/xfeat/compare_xfeat.py **逐比特等价**，
// 已在离线对拍中验证（关键点集合完全一致、描述子余弦 1.000000）。
//
// 解码链（严格对齐官方 verlab/XFeat `xfeat.py::detectAndCompute`）：
//   net(x) → M1=F.normalize(M1,dim=1)
//   K1h = get_kpts_heatmap(K1)                    # softmax(65) 取前 64 → 8× 像素展开
//   mkpts = NMS(K1h, threshold, kernel_size=5)
//   scores = nearest(K1h) * bilinear(heatmap)     # 可靠性分
//   top_k → bicubic 采样描述子 → L2 归一化 → valid = score>0
// 本实现额外加一步 **抛物线亚像素细化**（官方稀疏路径没有）：
//   实测 ORB 关键点 74.5% 带亚像素，而 SP/XFeat 是纯整数 —— 在「固定姿态只解平移」
//   的场景下这个差别让 XFeat 落后 ORB 1.8×；补上后反超（0.326 vs 0.577 m @P0）。
#pragma once
#include "rknn_api.h"
#include <string>
#include <vector>
#include <cstdint>
#include <utility>

struct XFeatParams {
    int   top_k      = 1500;   // 特征预算
    float det_th     = 0.05f;  // NMS 阈值（官方默认）
    int   nms_k      = 5;      // NMS 核尺寸
    float min_cossim = 0.90f;  // MNN 余弦门限（实测 0.90 优于官方默认 0.82）
    bool  subpix     = true;   // 抛物线亚像素细化
    int   match_thr  = 4;      // 匹配矩阵乘的线程数（RK3588 用 4×A76）
};

struct XFeatKps {
    int n = 0;
    std::vector<float> x, y;        // 亚像素坐标
    std::vector<float> score;       // 可靠性分
    std::vector<float> desc;        // n*64 行主序，已 L2 归一化
    void clear() { n = 0; x.clear(); y.clear(); score.clear(); desc.clear(); }
};

class XFeatFrontend {
public:
    // W,H 为输入图尺寸（当前管线 512×288，正好是 32 的倍数，无需 preprocess 缩放）
    bool init(const std::string& model_path, const XFeatParams& p, int W, int H);
    void release();
    bool ready() const { return inited_; }
    // ★ 临时改解码/匹配的线程数。用途：预取线程与渲染的投影（吃满 8 核）并行时，
    //   预取若也开 4 线程会与渲染争核（实测渲染 +21 ms，把预取收益吃光）。
    void set_threads(int n) { P_.match_thr = n < 1 ? 1 : n; }
    int  threads() const { return P_.match_thr; }

    // gray: W*H 行主序 uint8 灰度图
    bool extract(const uint8_t* gray, XFeatKps& out);

    // MNN 互近邻 + cossim 门限（与官方 match() 一致）
    // 返回配对数；out[i] = (a 的索引, b 的索引)
    int match(const XFeatKps& a, const XFeatKps& b,
              std::vector<std::pair<int, int>>& out);

    // 上次 match 的匹配置信度：接受配对的平均余弦相似度（-1 = 无）
    double last_mean_cossim = -1.0;

    // 上次 extract 各阶段耗时（ms）
    double ms_npu = 0, ms_decode = 0;

private:
    bool inited_ = false;
    XFeatParams P_;
    int W_ = 0, H_ = 0;             // 输入尺寸
    int Hc_ = 0, Wc_ = 0;           // 网络输出尺寸 (H/8, W/8)
    rknn_context ctx_ = 0;

    // 复用缓冲（避免每帧分配）
    std::vector<float> smap_;       // (64, Hc, Wc) softmax 后前 64 通道
    std::vector<float> k1h_;        // (H, W) 8× 上采样热力图
    std::vector<float> lmax_;       // (H, W) 5×5 局部极大
    std::vector<float> desc_q_, desc_r_;   // 匹配矩阵复用缓冲
    std::vector<int>   m12_, m21_;
    std::vector<float> csmax_;
    // ★ 匹配用大缓冲必须预分配：cossim 矩阵 1500×1500×4 = 9 MB，
    //   若每帧 new/free 会把分配器搅碎，连带拖慢渲染器的排序（实测 10→21 ms/帧）。
    std::vector<float> cs_;         // (top_k, top_k) cossim 矩阵
    std::vector<float> bt_;         // (64, top_k) 转置后的描述子
};
