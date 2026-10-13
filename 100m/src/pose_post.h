// pose_post.h —— 零延迟位姿后处理：倾角约束 + 因果卡尔曼
//
// 设计约束：**实时**。只用当前帧及之前的信息，不引入任何输出延迟。
//   · 倾角约束：逐帧解析校正，利用「云台稳定 ⇒ 光轴铅垂」先验，
//               消除下视相机「水平平移 ↔ 倾斜」的几何退化
//               （实测相关系数 -0.999，err_pos ≈ H·δtilt）
//   · 因果卡尔曼：6 状态匀速模型，吃掉逐帧白噪声。
//               ★ 不用 RTS（离线），也不用固定滞后（有延迟）。
//               噪声参数用**固定标定值**（上线没有真值，不能在线用 GT 估）。
//
// 校正后的位姿会**反馈**给下一帧的渲染先验 → 闭环正反馈，收益比离线阶梯更大。
#pragma once
#include <cmath>
#include <cstring>

struct PoseKF {
    double x[6];        // [px,py,pz, vx,vy,vz]
    double P[36];
    bool   inited = false;
    double dt = 1.0;    // 帧间隔（帧为单位）
    double sig_r = 1.5; // 测量噪声 σ（米）—— 倾角校正后的残差量级
    double sig_a = 0.5; // 过程噪声 σ（米/帧²）—— 平台机动加速度
    int    gated = 0;   // 被门控的帧数（诊断）

    void reset() { inited = false; gated = 0; }

    // z = 倾角校正后的位置测量；out = 滤波后的位置
    void step(const double z[3], double out[3]) {
        const int n = 6;
        if (!inited) {
            for (int i = 0; i < 3; ++i) { x[i] = z[i]; x[i + 3] = 0.0; }
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    P[i * n + j] = (i == j) ? ((i < 3) ? sig_r * sig_r : 100.0) : 0.0;
            inited = true;
            for (int i = 0; i < 3; ++i) out[i] = x[i];
            return;
        }

        // F = [[I, dt·I], [0, I]]
        double F[36], Ft[36];
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                F[i * n + j] = (i == j) ? 1.0 : 0.0;
        F[0 * n + 3] = dt; F[1 * n + 4] = dt; F[2 * n + 5] = dt;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) Ft[i * n + j] = F[j * n + i];

        // Q = G σa² Gᵗ，G = [[0.5dt²·I], [dt·I]]
        double Q[36];
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double s = 0.0;
                for (int a = 0; a < 3; ++a) {
                    const double gi = (i == a) ? 0.5 * dt * dt : ((i == a + 3) ? dt : 0.0);
                    const double gj = (j == a) ? 0.5 * dt * dt : ((j == a + 3) ? dt : 0.0);
                    s += gi * gj;
                }
                Q[i * n + j] = s * sig_a * sig_a;
            }

        double xp[6], FP[36], Pp[36];
        for (int i = 0; i < n; ++i) {
            xp[i] = 0.0;
            for (int j = 0; j < n; ++j) xp[i] += F[i * n + j] * x[j];
        }
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double s = 0.0;
                for (int t = 0; t < n; ++t) s += F[i * n + t] * P[t * n + j];
                FP[i * n + j] = s;
            }
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double s = 0.0;
                for (int t = 0; t < n; ++t) s += FP[i * n + t] * Ft[t * n + j];
                Pp[i * n + j] = s + Q[i * n + j];
            }

        // 创新门控：S 用先验 Pp（R 未放大）算马氏距离
        double innov[3] = { z[0] - xp[0], z[1] - xp[1], z[2] - xp[2] };
        double S0[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                S0[i * 3 + j] = Pp[i * n + j] + ((i == j) ? sig_r * sig_r : 0.0);
        auto inv3 = [](const double S[9], double Si[9]) {
            const double d = S[0] * (S[4] * S[8] - S[5] * S[7])
                           - S[1] * (S[3] * S[8] - S[5] * S[6])
                           + S[2] * (S[3] * S[7] - S[4] * S[6]);
            Si[0] =  (S[4] * S[8] - S[5] * S[7]) / d;
            Si[1] = -(S[1] * S[8] - S[2] * S[7]) / d;
            Si[2] =  (S[1] * S[5] - S[2] * S[4]) / d;
            Si[3] = -(S[3] * S[8] - S[5] * S[6]) / d;
            Si[4] =  (S[0] * S[8] - S[2] * S[6]) / d;
            Si[5] = -(S[0] * S[5] - S[2] * S[3]) / d;
            Si[6] =  (S[3] * S[7] - S[4] * S[6]) / d;
            Si[7] = -(S[0] * S[7] - S[1] * S[6]) / d;
            Si[8] =  (S[0] * S[4] - S[1] * S[3]) / d;
        };
        double Si0[9]; inv3(S0, Si0);
        double md = 0.0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                md += innov[i] * Si0[i * 3 + j] * innov[j];

        // R 放大（离群帧软抑制）
        double Rk[3];
        for (int i = 0; i < 3; ++i) Rk[i] = sig_r * sig_r;
        if (md > 9.0) { for (int i = 0; i < 3; ++i) Rk[i] *= 100.0; ++gated; }

        double S[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                S[i * 3 + j] = Pp[i * n + j] + ((i == j) ? Rk[j] : 0.0);
        double Si[9]; inv3(S, Si);

        // K = Pp[:, :3] · S⁻¹
        double K[18];
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < 3; ++j) {
                double s = 0.0;
                for (int t = 0; t < 3; ++t) s += Pp[i * n + t] * Si[t * 3 + j];
                K[i * 3 + j] = s;
            }

        for (int i = 0; i < 3; ++i)
            x[i] = xp[i] + K[i * 3 + 0] * innov[0] + K[i * 3 + 1] * innov[1] + K[i * 3 + 2] * innov[2];
        for (int i = 0; i < 3; ++i) x[i + 3] = xp[i + 3];

        // P = (I − K·H) · Pp ，(KH)[i][j] = K[i][j] (j<3)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double s = 0.0;
                for (int t = 0; t < 3; ++t) s += K[i * 3 + t] * Pp[t * n + j];
                P[i * n + j] = Pp[i * n + j] - s;
            }

        for (int i = 0; i < 3; ++i) out[i] = x[i];
    }
};

// 倾角约束：把估计位置沿退化方向搬回「光轴铅垂」子流形
//   光轴（世界系）= w2c 的**第三行**。Mat4 列主序：m[j*4+i] = R(i,j)，
//   所以第三行 R(2,j) = m[j*4+2] = (m[2], m[6], m[10])。
//   ⚠️ 踩坑：m[8],m[9],m[10] 是第三**列**（世界 Z 轴在相机系下的表示），
//      用它会施加一个 ~30 m 的错误校正，闭环直接发散（实测 78 m）。
//   校正：C_xy += H · (fwd_xy / |fwd|)
inline void apply_tilt(float* m, float H) {
    const double Cx = -(m[0] * m[12] + m[1] * m[13] + m[2] * m[14]);
    const double Cy = -(m[4] * m[12] + m[5] * m[13] + m[6] * m[14]);
    const double Cz = -(m[8] * m[12] + m[9] * m[13] + m[10] * m[14]);
    double fx = m[2], fy = m[6], fz = m[10];
    const double nl = std::sqrt(fx * fx + fy * fy + fz * fz);
    if (nl < 1e-12) return;
    fx /= nl; fy /= nl; fz /= nl;
    const double Nx = Cx + H * fx, Ny = Cy + H * fy;
    m[12] = (float)(-(m[0] * Nx + m[1] * Ny + m[2] * Cz));
    m[13] = (float)(-(m[4] * Nx + m[5] * Ny + m[6] * Cz));
    m[14] = (float)(-(m[8] * Nx + m[9] * Ny + m[10] * Cz));
}
