// ply_loader.h — 3DGS binary PLY 流式解析（M1）
//
// 设计要点（依据 NOTES.md §2.1 + 附录A.1）：
//  1. 只支持 binary_little_endian，不依赖 numpy / plyfile
//  2. 流式读取：一次 fread 缓冲，逐点解析，避免把 744MB 原始数据整体读进内存
//  3. sigmoid(x) = 1/(1+exp(-x))  预变换不放在这里（由调用方决定）
//  4. 深度用 float32 bit pattern 排序，不量化（附录 A.1② gsplat 官方做法）
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>

namespace gsplat {

// ── 标准 3DGS PLY 的 62 个属性布局 ──────────────────────────────
// 顺序即文件中的列顺序，必须严格对齐，否则整个数据区错位
struct PlyLayout {
    // 位置与基础属性
    float x, y, z;                        //  0..2
    float nx, ny, nz;                     //  3..5   (法线，3DGS 中通常为 0)
    float f_dc_0, f_dc_1, f_dc_2;         //  6..8   DC 球谐（RGB 基）
    float opacity;                        //  9     原始 logit，需 sigmoid
    float scale_0, scale_1, scale_2;      // 10..12  原始 log，需 exp
    float rot_0, rot_1, rot_2, rot_3;     // 13..16  四元数 (w,x,y,z) 顺序

    // f_rest_0 .. f_rest_44  共 45 个 float
    // degree 3 → (3+1)^2 = 16 个系数 × 3 通道 = 48，实际只存 45（舍去高阶）
    float f_rest[45];                     // 17..61

    static constexpr int kFloats = 62;
    static constexpr size_t kStride = kFloats * sizeof(float); // 248 B/点
};

// 单个高斯：渲染真正需要的最小属性集合
//内存布局刻意做成 16 字节对齐的紧凑结构（见 NOTES.md §2.1 纹理组织）
struct Splat {
    float x, y, z;          // 位置 (12 B)
    float opacity;          // 已 sigmoid (4 B)
    float scale0, scale1, scale2;   // 已 exp (12 B)
    float rot0, rot1, rot2, rot3;   // 四元数，已归一化 (16 B)
};   // 共 56 B

// SH 系数（DC + 1 阶 = 12 float），单独存以便按质量档位切换
struct ShCoeffs {
    float dc[3];            //  0..2
    float rest[9];          //  3..11  1 阶：3 组 × 3通道
};   // 48 B

// ── PLY 属性列布局：**运行时从 header 解析** ─────────────────────
//
// ⚠️⚠️ 血泪教训（本项目最严重的一个 bug）：
//   一开始我硬编码了「标准 3DGS」列顺序，假设
//       6..8=f_dc, 9=opacity, 10..12=scale, 13..16=rot, 17..61=f_rest
//   结果本项目的 PLY 实际顺序是
//       6..8=f_dc, 9..53=f_rest_0..44, 54=opacity, 55..57=scale, 58..61=rot
//   即 **f_rest 在 opacity 之前**。硬编码导致把 f_rest 当 opacity/scale 读，
//   3M 个高斯的 scale 全部变成 exp(f_rest)≈1.0，画面上就是一堆巨型同心圆盘。
//
//   不同 3DGS 导出器（INRIA 官方 / gsplat / Postshot / 国产工具链）列顺序
//   都不一样，**唯一正确的做法是按名字查列号**。本结构体即为此而生。
struct PlyColumns {
    int x = -1, y = -1, z = -1;
    int f_dc[3]  = {-1, -1, -1};     // f_dc_0..2
    int f_rest   = -1;               // f_rest_0 的起始列（后续连续）
    int n_rest   = 0;                // f_rest 实际个数
    int opacity  = -1;
    int scale[3] = {-1, -1, -1};     // scale_0..2
    int rot[4]   = {-1, -1, -1, -1}; // rot_0..3

    // ★ 2DGS 标记（2026-10-11）：有 scale_0/scale_1 但**无 scale_2** 时置位。
    //   CityGaussian V2 用 Gaussian2D，导出 PLY 只有 2 个尺度。
    //   板端把缺失的第三轴当 0 处理 —— 由 compute_cov3d 的
    //   Σ = R·diag(s0²,s1²,0)·Rᵀ 自然得到 2D 面元的退化协方差。
    bool is_2dgs = false;

    bool valid() const {
        const bool scale_ok = (scale[0] >= 0 && scale[1] >= 0) &&
                              (scale[2] >= 0 || is_2dgs);   // ★ 2DGS 允许缺 scale_2
        return x >= 0 && y >= 0 && z >= 0 &&
               f_dc[0] >= 0 && f_dc[1] >= 0 && f_dc[2] >= 0 &&
               opacity >= 0 && scale_ok &&
               rot[0] >= 0 && rot[1] >= 0 && rot[2] >= 0 && rot[3] >= 0;
    }

    // 在解析完列号后调用：判定是否 2DGS（缺 scale_2 但有 scale_0/1）
    void detect_2dgs() {
        is_2dgs = (scale[0] >= 0 && scale[1] >= 0 && scale[2] < 0);
    }
};

// ── PLY 文件头解析结果 ──────────────────────────────────────────
struct PlyHeader {
    int64_t num_points = 0;
    int      num_floats = 0;      // 实际 float32 属性数
    int      header_bytes = 0;    // 数据区起始偏移
    std::vector<std::string> property_names;   // 全部属性名（按列序）
    PlyColumns cols;              // 按名字解析出的关键列偏移
};

// 只解析头部，不读数据。ply_path 传nullptr 可用于校验字节数
PlyHeader parse_ply_header(const char* ply_path, uint32_t* out_props = nullptr,
                           uint32_t* out_scalar_types = nullptr);

// 全量加载（会占内存：3M 点 × (56+48) B ≈ 312 MB）
// 调用方需自行保证内存充足；板端 13 GB 可用，OK
bool load_ply_splats(const char* ply_path, std::vector<Splat>& out_splats,
                     std::vector<ShCoeffs>& out_sh, int sh_degree = 1,
                     bool verbose = true, bool* out_is_2dgs = nullptr);

// ── 轻量流式读取：只提取位置和 alpha（loadtest 用，内存占用极小） ──
struct PosAlpha {
    float x, y, z, alpha;
};
bool stream_positions(const char* ply_path, std::vector<PosAlpha>& out,
                      int64_t max_points = -1, bool verbose = true);

} // namespace gsplat
