// ============================================================================
//  splat_loader.h —— 加载 antimatter15 / gsplat 的 32 字节紧凑格式（.splat）
//
//  为什么需要它：
//    训练出的标准 INRIA PLY 是 248 B/高斯，本项目自己的 PLY 更是 504 B/高斯
//    （含 64 维语义列）。而 .splat 只要 **32 B/高斯** ——
//      · 标准 PLY → splat:  7.75× 压缩
//      · 本项目 PLY → splat: 15.75× 压缩
//    实测：point_cloud.ply 1.39 GB → point_cloud.splat 88.6 MB。
//    对「大场景塞进 15 GB 内存的 RK3588」这是关键手段。
//
//  ★ 格式定义来自 3DGS-Playground/tools/export_splat.py（已逐行核对）：
//        rgb   = clip(0.5 + SH_C0 * shs_dc, 0, 1) * 255
//        alpha = clip(sigmoid(opacities) * 255, 0, 255)
//        rot   = normalize(rotations) * 128 + 128      # → [0,255]
//        buf[:, 0:12]  = means            (3 × float32 LE)
//        buf[:, 12:24] = exp(scales)      (3 × float32 LE)   ← 已做 exp
//        buf[:, 24:27] = rgb              (3 × uint8)
//        buf[:, 27]    = alpha            (1 × uint8)
//        buf[:, 28:32] = rot              (4 × uint8)
//    每个高斯恰好 32 字节，无文件头。
//
//  ★ 与板端 Splat 结构的对应（注意板端约定「已激活」）：
//        板端 Splat.opacity 是 **已 sigmoid** 的值   → opacity = alpha / 255
//        板端 Splat.scale*  是 **已 exp**    的值   → 直接用文件里的 float
//        板端 Splat.rot*    是 **已归一化**  四元数 → q = (u8 - 128)/128 再归一化
//    所以本 loader **不做** sigmoid/exp，只做反量化 —— 与 load_ply_splats 的
//    输出语义一致（那边是从 raw 做 sigmoid/exp，这边文件里本就是激活后的值）。
//
//  ⚠️ SH（球谐）限制：
//    .splat 格式只有 DC 项（degree 0），没有 f_rest。本 loader 把
//    ShCoeffs.rest[] 全部置 0。板端渲染若用 sh_degree >= 1，能跑但 rest 贡献为 0
//    （视觉上略平，但结构/边缘仍在，对 ORB 匹配影响很小）。
//    需要高保真时请用 .ply。
//
//  用法（header-only，无需改 Makefile）：
//        #include "splat_loader.h"
//        std::vector<Splat> sp; std::vector<ShCoeffs> sh;
//        if (!gsplat::load_splat_splats("model.splat", sp, sh)) { ... }
// ============================================================================
#pragma once

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#include "ply_loader.h"     // 复用 gsplat::Splat / gsplat::ShCoeffs

namespace gsplat {

// SH 的 0 阶基函数常量（与 export_splat.py 的 SH_C0 一致）
static constexpr double kShC0 = 0.28209479177387814;

// 探测文件是否是 .splat（无魔数，只能靠「大小是 32 的整数倍 && 扩展名」）
inline bool looks_like_splat(const char* path) {
    const char* dot = strrchr(path, '.');
    if (!dot) return false;
    if (strcasecmp(dot, ".splat") != 0) return false;
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fclose(f);
    return sz > 0 && (sz % 32) == 0;
}

// 加载 .splat。成功返回 true。
// out_splats / out_sh 会被 resize 到高斯数。
inline bool load_splat_splats(const char* splat_path,
                              std::vector<Splat>& out_splats,
                              std::vector<ShCoeffs>& out_sh,
                              bool verbose = true) {
    FILE* f = fopen(splat_path, "rb");
    if (!f) {
        fprintf(stderr, "[splat] 打不开 %s\n", splat_path);
        return false;
    }
    fseek(f, 0, SEEK_END);
    const long long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (bytes <= 0 || (bytes % 32) != 0) {
        fprintf(stderr, "[splat] 文件大小 %lld 不是 32 的整数倍，格式不对\n", bytes);
        fclose(f);
        return false;
    }
    const int64_t n = bytes / 32;
    out_splats.clear();
    out_sh.clear();
    out_splats.resize(static_cast<size_t>(n));
    out_sh.resize(static_cast<size_t>(n));   // rest 由 resize 默认初始化为 0

    if (verbose) {
        fprintf(stdout, "[splat] 加载 %s\n", splat_path);
        fprintf(stdout, "[splat]   %.1f MB / %lld 个高斯 (32 B/个)\n",
                bytes / 1e6, (long long)n);
    }

    // 分块读，块大小取 32 的整数倍
    constexpr size_t kChunkGauss = 1 << 18;          // 262144 个/块 = 8 MB
    std::vector<uint8_t> buf(kChunkGauss * 32);

    int64_t done = 0;
    while (done < n) {
        const size_t want = static_cast<size_t>(
            std::min<int64_t>(static_cast<int64_t>(kChunkGauss), n - done));
        const size_t got = fread(buf.data(), 32, want, f);
        if (got == 0) break;

        for (size_t i = 0; i < got; ++i) {
            const uint8_t* p = buf.data() + i * 32;
            Splat& s = out_splats[done + i];

            // 位置：3 × float32 LE
            float pos[3];
            memcpy(pos, p, 12);
            s.x = pos[0]; s.y = pos[1]; s.z = pos[2];

            // 尺度：3 × float32 LE（已 exp）
            float sc[3];
            memcpy(sc, p + 12, 12);
            s.scale0 = sc[0]; s.scale1 = sc[1]; s.scale2 = sc[2];

            // 颜色：3 × uint8 → 反推 SH DC
            //   rgb_u8 = clip(0.5 + SH_C0*f_dc,0,1)*255
            //   ⇒ f_dc = (rgb_u8/255 - 0.5) / SH_C0
            ShCoeffs& c = out_sh[done + i];
            for (int ch = 0; ch < 3; ++ch) {
                const double rgb = static_cast<double>(p[24 + ch]) / 255.0;
                c.dc[ch] = static_cast<float>((rgb - 0.5) / kShC0);
            }
            // rest 全 0（degree 0）
            for (int k = 0; k < 9; ++k) c.rest[k] = 0.f;

            // 不透明度：1 × uint8（已 sigmoid）→ 板端约定就是已 sigmoid
            s.opacity = static_cast<float>(p[27]) / 255.0f;

            // 旋转：4 × uint8 → q = (u8 - 128)/128，再归一化
            float q[4];
            for (int k = 0; k < 4; ++k)
                q[k] = (static_cast<float>(p[28 + k]) - 128.0f) / 128.0f;
            float qn = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
            if (qn > 1e-12f) { q[0]/=qn; q[1]/=qn; q[2]/=qn; q[3]/=qn; }
            // 与 PLY 路径保持同一约定：rot0=w, rot1=x, rot2=y, rot3=z
            s.rot0 = q[0]; s.rot1 = q[1]; s.rot2 = q[2]; s.rot3 = q[3];
        }
        done += static_cast<int64_t>(got);
    }
    fclose(f);

    if (done != n) {
        fprintf(stderr, "[splat] 只读到 %lld / %lld 个高斯\n",
                (long long)done, (long long)n);
        return false;
    }
    if (verbose) {
        fprintf(stdout, "[splat]   ✔ %lld 个高斯加载完成（SH degree 0，rest 全 0）\n",
                (long long)n);
    }
    return true;
}

}  // namespace gsplat
