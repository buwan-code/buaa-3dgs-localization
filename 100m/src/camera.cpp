// camera.cpp — 相机与 EWA 投影数学（315 行）
//
// ══ 本文件结构 ═══════════════════════════════════════════════════════
//   compute_cov3d       ~13   四元数+尺度 → 3D 协方差 Σ = R·S·Sᵀ·Rᵀ
//   Camera::look_at     ~33   由「注视目标」构造轨道相机（调试/生成参考视图）
//   project_gaussian    ~88   ★核心：3D 高斯 → 2D Splat2D
//                              ① 世界→视空间（刚体，无缩放）
//                              ② 视空间→裁剪空间（透视投影 + 雅可比 J）
//                              ③ Σ3D → Σ2D （EWA splatting）
//                              ④ 剔除（视野外 / 近远平面 / 屏幕外 / 退化）
//   cov2d_to_ellipse    ~298  2D 协方差 → 椭圆长短轴 + 旋转角
//
// ══ EWA splatting 的三段（Zwicker 2001）══════════════════════════════
//   ① 世界 → 视空间：刚体变换（无缩放）
//   ② 视空间 → 裁剪空间：透视投影
//   ③ 3D 协方差 → 2D 协方差：Σ2D = J · W · Σ3D · Wᵀ · Jᵀ
//      其中 J 是透视投影的雅可比矩阵，W 是视旋转
//
// ⚠️ 本文件的 project_gaussian 是**通用路径**（好懂、好验证）；
//    实际渲染跑的是 projfast.cpp 的**快路径**（把 J·W3 折叠成 2×3 矩阵）。
//    想理解数学看这里，想理解性能看 projfast.cpp。
//
#include "camera.h"
#include <cstdio>

namespace gsplat {

// ── 3D 协方差：Σ = R · S · Sᵀ · Rᵀ ────────────────────────────────
// 其中 R 来自四元数，S = diag(sx, sy, sz)。
// 展开后 Σ = M Mᵀ，其中 M = R · S（列缩放），
// 等价于 Σ_ij = Σ_k (R_ik * s_k) * (R_jk * s_k)
//
// 这是 diff-gaussian-rasterization 里 computeCov3D 的标准做法，
// 好处是避免显式构造 S 矩阵再三次相乘。
void compute_cov3d(const Vec3& s3, float qw, float qx, float qy, float qz,
                   float cov[6]) {
    const Mat3 R = quat_to_mat3(qw, qx, qy, qz);
    const float sc[3] = { s3.x, s3.y, s3.z };

    // M[i][k] = R[i][k] * sc[k]
    float M[3][3];
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k) M[i][k] = R.m[i][k] * sc[k];

    // Σ_ij = Σ_k M[i][k] * M[j][k]
    cov[0] = M[0][0]*M[0][0] + M[0][1]*M[0][1] + M[0][2]*M[0][2]; // xx
    cov[1] = M[0][0]*M[1][0] + M[0][1]*M[1][1] + M[0][2]*M[1][2]; // xy
    cov[2] = M[0][0]*M[2][0] + M[0][1]*M[2][1] + M[0][2]*M[2][2]; // xz
    cov[3] = M[1][0]*M[1][0] + M[1][1]*M[1][1] + M[1][2]*M[1][2]; // yy
    cov[4] = M[1][0]*M[2][0] + M[1][1]*M[2][1] + M[1][2]*M[2][2]; // yz
    cov[5] = M[2][0]*M[2][0] + M[2][1]*M[2][1] + M[2][2]*M[2][2]; // zz
}

// ── 相机构造：look_at ──────────────────────────────────────────────
// ── ◀GUIDE▶ 【2.1】 Camera::look_at()  建一个「看向目标」的相机
//
//   干什么   : 站在 eye，看向 target，用 up 定滚转，造出一个相机（OpenGL 习惯 y 朝上）。
//   参数    : eye 相机位置；target 注视点；up 世界上方向；w,h 分辨率；fovy_deg 垂直视场角
//   返回    : Camera
//   用在哪   : 调试/生成参考视图；主定位流程用的是 cam_from_lookdir（OpenCV 约定）。
//   铁律    : 三角函数在 CPU 侧一次算完 —— 放进 GPU 顶点着色器会让帧率掉 14.7 倍。
// ──────────────────────────────────────────────────────────────
Camera Camera::look_at(const Vec3& eye, const Vec3& target, const Vec3& up,
                       int w, int h, float fovy_deg) {
    Camera c;
    c.width = w; c.height = h;

    // 内参：由垂直 FOV 推 fy，fx 用方形像素（fx = fy）
    const float fovy = fovy_deg * 3.14159265358979323846f / 180.f;
    c.cx = w * 0.5f; c.cy = h * 0.5f;
    c.fy = (h * 0.5f) / std::tan(fovy * 0.5f);
    c.fx = c.fy;

    // 视空间基（OpenCV 约定：+x 右，+y 下，+z 前）
    // forward = normalize(target - eye)
    float fx_ = target.x - eye.x, fy_ = target.y - eye.y, fz_ = target.z - eye.z;
    float fl = std::sqrt(fx_*fx_ + fy_*fy_ + fz_*fz_);
    if (fl < 1e-9f) fl = 1e-9f;
    fx_ /= fl; fy_ /= fl; fz_ /= fl;

    // right = normalize(forward × up)
    float rx = fy_*up.z - fz_*up.y;
    float ry = fz_*up.x - fx_*up.z;
    float rz = fx_*up.y - fy_*up.x;
    float rl = std::sqrt(rx*rx + ry*ry + rz*rz);
    if (rl < 1e-9f) { rx = 1; ry = 0; rz = 0; rl = 1; }
    rx /= rl; ry /= rl; rz /= rl;

    // down = forward × right  （因 y 轴向下）
    const float dx = fy_*rz - fz_*ry;
    const float dy = fz_*rx - fx_*rz;
    const float dz = fx_*ry - fy_*rx;

    // 世界 → 相机：(R | t)，R 的行是 right/down/forward，t = -R·eye
    Mat4 W = Mat4::identity();
    W.m[0] = rx;  W.m[4] = ry;  W.m[8]  = rz;  W.m[12] = -(rx*eye.x + ry*eye.y + rz*eye.z);
    W.m[1] = dx;  W.m[5] = dy;  W.m[9]  = dz;  W.m[13] = -(dx*eye.x + dy*eye.y + dz*eye.z);
    W.m[2] = fx_; W.m[6] = fy_; W.m[10] = fz_; W.m[14] = -(fx_*eye.x + fy_*eye.y + fz_*eye.z);
    W.m[3] = 0;   W.m[7] = 0;   W.m[11] = 0;   W.m[15] = 1;
    c.world_to_cam = W;
    return c;
}

// ── EWA 投影（本文件的核心）───────────────────────────────────────
//
// 步骤：
//   1. 把高斯中心转到视空间 v = W · p
//   2. 视空间 z <= 0（在相机后方）→ 剔除
//   3. 算 3D 协方差 Σ_3D = R S Sᵀ Rᵀ
//   4. 算 Σ_3D 在视空间的版本：Σ_v = W₃ Σ_3D W₃ᵀ
//      （注意：严格地说这一步应当用 W 的旋转部分；W 是刚体变换，
//        其 3×3 部分即正交旋转，故正确）
//   5. 算透视雅可比 J（2×3）
//   6. Σ_2D = J Σ_v Jᵀ，得到 [[a,b],[b,c]]
//   7. 加一个小的「低通」项（+0.3 px²）防止高斯在屏幕上退化成像素级尖点
//      —— 这是 3DGS 原论文的做法，对应 Mip-Splatting 之前的常规处理
//   8. 由 Σ_2D 的特征值算 3σ 半径，用于 quad 尺寸
// ── ◀GUIDE▶ 【2.2】 project_gaussian()  ★★★ 全文最核心的数学：3D 高斯 → 屏幕椭圆
//
//   干什么   : 【渲染的地基】把一个三维高斯（位置+大小+朝向）算出它在屏幕上
//     应该是**哪个位置的多大的斜椭圆**，以及它离相机多远。
//   参数    : p 高斯中心(世界)；s3 三轴尺度(已 exp，真实标准差)；
//     qw,qx,qy,qz 朝向四元数；cam 相机；
//     cov6_out 可选，输出 3D 协方差诊断；d_out 可选，输出深度
//   返回    : Splat2D{屏幕中心 cx,cy；2D 协方差 a,b,c；depth 排序键；
//     inv_det 供着色器用；sigma_radius 3σ 半径；valid 是否保留}
//   公式    : Σ_2D = J·W·Σ_3D·Wᵀ·Jᵀ   （EWA splatting, Zwicker 2002）
//     Σ_3D = R·S·Sᵀ·Rᵀ（三维椭球协方差）
//     W = 视矩阵旋转部分； J = 透视投影的雅可比
//   为什么在 CPU: 算特征值要 sqrt、算朝向要 atan2，放进着色器掉速 14.7×，所以批量算好再上传。
// ──────────────────────────────────────────────────────────────
Splat2D project_gaussian(const Vec3& p,
                         const Vec3& s3,
                         float qw, float qx, float qy, float qz,
                         const Camera& cam,
                         float* cov6_out, float* d_out) {
    Splat2D out{};
    out.valid = false;

    // ── 1. 世界 → 视空间 ──────────────────────────────────────────
    const Vec4 vh = cam.world_to_cam.transform(Vec4{p.x, p.y, p.z, 1.f});
    const float vx = vh.x, vy = vh.y, vz = vh.z;

    if (d_out) *d_out = vz;
    out.depth = vz;

    // ── 2. 视锥剔除（相机前方）────────────────────────────────────
    // 留一点余量：z 太靠近近平面时，z² 做分母会数值爆炸
    if (vz < cam.znear) return out;
    if (vz > cam.zfar)  return out;

    const float inv_z  = 1.f / vz;
    const float inv_z2 = inv_z * inv_z;

    // ══════════════════════════════════════════════════════════════
    // ★★ 2026-10-03 代数融合优化（Task #10）★★
    //
    // 原始步骤（教科书写法）：
    //   ① Σ_3D = R·S·Sᵀ·Rᵀ                     ← compute_cov3d
    //   ② Σ_v  = W3·Σ_3D·W3ᵀ                   ← 2 次 3×3 矩阵乘
    //   ③ T    = J·Σ_v          (2×3)          ← 18 mul
    //   ④ Σ_2D = J·Tᵀ          (2×2)           ← 9 mul
    //   共约 27 + 54 + 18 + 9 = 108 次乘加，且中间要建两个 3×3 矩阵。
    //
    // 融合写法（利用 Σ 恒正定、可写成 M·Mᵀ 的性质）：
    //   Σ_2D = J·W3·(R·S)·(R·S)ᵀ·W3ᵀ·Jᵀ
    //        = (J·W3·R·S) · (J·W3·R·S)ᵀ
    //        = A·Aᵀ,   其中 A = J·W3·R·S  是 **2×3** 矩阵
    //
    //   ① A = J·(W3·R·S)      ← 先算 3×3 的 C=W3·R·S（27 mul），
    //                            再算 A=J·C（2×3，只需 2 行 × 3 列，
    //                            且 J 每行只有 2 个非零 → 12 mul）
    //   ② Σ_2D = A·Aᵀ          ← 3 个点积（每个 3 mul）= 9 mul
    //   共 ~48 次乘加，**省掉一半以上**，且不再构造中间 3×3 协方差矩阵。
    //
    //   数学等价性：完全精确，无任何近似 —— 因为
    //     J Σ_v Jᵀ = J W3 Σ_3D W3ᵀ Jᵀ = (J W3) Σ_3D (J W3)ᵀ
    //   而 Σ_3D = (R S)(R S)ᵀ，故 (JW3RS)(JW3RS)ᵀ 就是同一式子。
    //   注意 A 是 2×3，所以 Σ_2D 自动是 2×2 —— 正是我们要的。
    //
    //   ⚠️ 精度：这条路径下 Σ_2D 由 A·Aᵀ 得到，**天然对称正定**
    //      （a·c − b² = |A_0|²|A_1|² − (A_0·A_1)² ≥ 0，柯西-施瓦茨），
    //      比先做矩阵乘再取对称部分的旧路径数值更稳。
    // ══════════════════════════════════════════════════════════════

    // ── 3. C = W3 · R · S （3×3）──────────────────────────────────
    // R 由四元数给出，S = diag(scale)。
    // 直接按列展开：C 的第 k 列 = W3 · (R 的第 k 列 × s_k)
    //   即 C[i][k] = s_k · Σ_j W3[i][j] · R[j][k]
    // 但更省的做法：先把 R 的列按 s_k 缩放，再做 W3· 乘法。
    const Mat3 R = quat_to_mat3(qw, qx, qy, qz);
    const Mat3 W3 = cam.world_to_cam.upper3x3();
    const float s0 = s3.x, s1 = s3.y, s2 = s3.z;

    // RS 的第 k 列 = R 的第 k 列 * s_k  →  存成 rs[3][3]（列为主）
    float rs[3][3];
    rs[0][0] = R.m[0][0]*s0;  rs[1][0] = R.m[1][0]*s0;  rs[2][0] = R.m[2][0]*s0;
    rs[0][1] = R.m[0][1]*s1;  rs[1][1] = R.m[1][1]*s1;  rs[2][1] = R.m[2][1]*s1;
    rs[0][2] = R.m[0][2]*s2;  rs[1][2] = R.m[1][2]*s2;  rs[2][2] = R.m[2][2]*s2;

    // C = W3 · RS  （3×3）
    float C[3][3];
    for (int i = 0; i < 3; ++i) {
        const float w0 = W3.m[i][0], w1 = W3.m[i][1], w2 = W3.m[i][2];
        C[i][0] = w0*rs[0][0] + w1*rs[1][0] + w2*rs[2][0];
        C[i][1] = w0*rs[0][1] + w1*rs[1][1] + w2*rs[2][1];
        C[i][2] = w0*rs[0][2] + w1*rs[1][2] + w2*rs[2][2];
    }

    // ── 4. 透视雅可比 J（2×3）──────────────────────────────────────
    //   J = [ fx/z,    0,   -fx*vx/z² ]
    //       [   0,   fy/z, -fy*vy/z² ]
    // 注意 J 每行只有 2 个非零元 —— 显式写出来避免零元乘法。
    const float j00 = cam.fx * inv_z;  const float j02 = -cam.fx * vx * inv_z2;
    const float j11 = cam.fy * inv_z;  const float j12 = -cam.fy * vy * inv_z2;

    // ── 5. A = J · C  （2×3）──────────────────────────────────────
    //   A[0][j] = j00*C[0][j] + j02*C[2][j]      （j01 == 0）
    //   A[1][j] = j11*C[1][j] + j12*C[2][j]      （j10 == 0）
    float A0[3], A1[3];
    A0[0] = j00*C[0][0] + j02*C[2][0];
    A0[1] = j00*C[0][1] + j02*C[2][1];
    A0[2] = j00*C[0][2] + j02*C[2][2];
    A1[0] = j11*C[1][0] + j12*C[2][0];
    A1[1] = j11*C[1][1] + j12*C[2][1];
    A1[2] = j11*C[1][2] + j12*C[2][2];

    // ── 6. Σ_2D = A Aᵀ  →  [[a,b],[b,c]] ─────────────────────────
    float a = A0[0]*A0[0] + A0[1]*A0[1] + A0[2]*A0[2];
    float b = A0[0]*A1[0] + A0[1]*A1[1] + A0[2]*A1[2];
    float c = A1[0]*A1[0] + A1[1]*A1[1] + A1[2]*A1[2];

    // 若调用方需要 3D 协方差（诊断用），这里单独算一份（不参与快路径）
    if (cov6_out) compute_cov3d(s3, qw, qx, qy, qz, cov6_out);

    // ── 7. 低通项：+0.3 px² ───────────────────────────────────────
    // 目的：防止远距离/极小高斯退化成屏幕上的噪点。3DGS 原实现的做法。
    a += 0.3f;
    c += 0.3f;

    // 对角占优检查（数值健康性）
    if (!(a > 0.f) || !(c > 0.f)) return out;

    // ── 7.5 ★ 椭圆锐度钳制（修「长刺/拉丝」伪影）──────────────────
    //
    // 现象：画面出现大量从物体向外辐射的细长"尖刺"（见 m2_h.png）。
    // 根因：Σ_2D 的两个特征值之比 λ_max/λ_min 可以极大。
    //   薄片状高斯（如墙面/地面的掠射视角）在屏幕上投影成**针状**
    //   椭圆：长轴几十像素、短轴不到 1 像素。
    //   此时 quad 半边长取 3·√λ_max（很大），而 exp(-0.5·d²) 在
    //   长轴方向衰减极慢 → 一条细长高亮线被完整画出来 = 尖刺。
    //
    //   更糟的是这种针状高斯在数值上极不稳定：λ_min 接近 0 时
    //   1/det 爆炸，inv_a/inv_b/inv_c 会溢出成 Inf/NaN。
    //
    // 解法（对齐官方 diff-gaussian-rasterization 的做法）：把
    //   特征值比钳到上限。官方在 CUDA 里对 λ 做 `max(λ, 0.1)` 与
    //   `det` 检查；这里用更直观的「比」形式：
    //     λ_min' = max(λ_min, λ_max / kMaxAnisoRatio)
    //   再重组协方差 a,b,c。
    //
    //   kMaxAnisoRatio 取 15：
    //     - 物理上，真实重建里一个高斯的屏幕长短轴比很少超过 10
    //       （再大说明它是掠射薄片，本身就重建得不可靠）
    //     - 视觉上，15:1 的椭圆已经足够"扁"，但短轴仍有 ~1/4 长轴，
    //       尖刺会变成正常的宽椭圆片
    //     - 调小（如 5）会明显"变糊"，调大（如 50）尖刺复现
    constexpr float kMaxAnisoRatio = 15.0f;
    {
        const float mid0 = 0.5f * (a + c);
        const float rad0 = std::sqrt(std::fmax(0.f, 0.25f*(a-c)*(a-c) + b*b));
        float l_max = mid0 + rad0;
        float l_min = mid0 - rad0;
        if (l_max > 0.f) {
            const float l_min_floor = l_max / kMaxAnisoRatio;
            if (l_min < l_min_floor) {
                // 只抬高 l_min，保持 λ_max 与主轴方向不变：
                //   新 mid' = (l_max + l_min')/2, 新 rad' = (l_max - l_min')/2
                const float mid2 = 0.5f * (l_max + l_min_floor);
                const float rad2 = 0.5f * (l_max - l_min_floor);
                // 由 (mid2, rad2) 反解 a,b,c（保持主轴角度不变）：
                //   设原角度 θ，则 a = mid + rad·cos2θ, c = mid - rad·cos2θ,
                //   b = rad·sin2θ。用 mid0/rad0 求得 cos2θ, sin2θ 再重代入。
                float cos2t = 1.f, sin2t = 0.f;
                if (rad0 > 1e-12f) {
                    cos2t = 0.5f * (a - c) / rad0;
                    sin2t = b / rad0;
                }
                a = mid2 + rad2 * cos2t;
                c = mid2 - rad2 * cos2t;
                b = rad2 * sin2t;
            }
        }
    }

    // ── 8. 屏幕中心 + 半径 ────────────────────────────────────────
    // 视空间 → 像素： px = fx * vx/vz + cx
    const float px = cam.fx * vx * inv_z + cam.cx;
    const float py = cam.fy * vy * inv_z + cam.cy;

    out.cx = px;
    out.cy = py;
    out.a = a; out.b = b; out.c = c;

    const float det = a * c - b * b;
    if (det <= 1e-12f) return out;
    out.inv_det = 1.f / det;

    // 特征值（2×2 对称矩阵）：λ = (a+c)/2 ± sqrt(((a-c)/2)² + b²)
    const float mid = 0.5f * (a + c);
    const float rad = std::sqrt(std::fmax(0.f, 0.25f*(a-c)*(a-c) + b*b));
    const float lam_major = mid + rad;      // 最大特征值
    // 3σ 半径 → quad 半边长（像素）
    // eigenvector 的 σ = sqrt(λ)，3σ 覆盖 99.7%
    out.sigma_radius = 3.f * std::sqrt(std::fmax(lam_major, 0.f));

    // 屏幕外剔除：整个椭圆都在画面外就丢
    if (px + out.sigma_radius < 0.f || px - out.sigma_radius >= cam.width)  return out;
    if (py + out.sigma_radius < 0.f || py - out.sigma_radius >= cam.height) return out;

    // 半径上限：防止单个高斯覆盖全屏导致片元填充爆炸 + 画面出现
    // 横穿全屏的"白色拖尾"（实测：极少数退化的大高斯会糊掉整幅图）。
    // ⚠️ 权衡：
    //   - 太小 → 前景大高斯（地面、近景墙）被误删，画面留洞
    //   - 太大 → 单个高斯覆盖全屏，填充爆炸 + 拖尾伪影
    // 选 1.5× 屏幕对角线：足以容纳任何"合法的大高斯"（一个高斯在
    // 正常焦距下不可能比整幅画面还大 1.5 倍），同时挡住退化样本。
    // 本项目给定位算法生成参考视图，宁可保留也别误删，所以给得宽，
    // 但不要宽到 4× （那是把伪影也一起放进来）。
    const float diag = std::sqrt((float)(cam.width*cam.width + cam.height*cam.height));
    const float max_r = (cam.max_sigma_radius > 0.f)
                            ? cam.max_sigma_radius
                            : 1.5f * diag;
    if (out.sigma_radius > max_r) return out;

    out.valid = true;
    return out;
}

// ── 2D 协方差 → 椭圆轴长与朝向 ────────────────────────────────────
// 用于调试可视化与单元测试：把 Σ_2D 还原成"人能看懂"的几何量。
// ── ◀GUIDE▶ 【2.3】 cov2d_to_ellipse()  2D 协方差 → 椭圆长轴/短轴/倾角
//
//   干什么   : 把协方差矩阵变成人能看懂的「长轴多长、短轴多长、倾角多少度」。
//   参数    : s 投影结果；semi_major/semi_minor/angle_rad 三个输出引用
//   返回    : 无
//   公式    : 特征值 λ = (a+c)/2 ± √(((a-c)/2)²+b²)； 半轴 = √λ； 角 = atan2(2b, a-c)/2
//   用途    : 调试与可视化，主渲染路径用的是 sigma_radius。
// ──────────────────────────────────────────────────────────────
void cov2d_to_ellipse(const Splat2D& s, float* semi_major,
                      float* semi_minor, float* angle_rad) {
    const float a = s.a, b = s.b, c = s.c;
    const float mid = 0.5f * (a + c);
    const float rad = std::sqrt(std::fmax(0.f, 0.25f*(a-c)*(a-c) + b*b));
    const float l1 = mid + rad;     // 大特征值
    const float l2 = mid - rad;     // 小特征值
    if (semi_major) *semi_major = std::sqrt(std::fmax(l1, 0.f));
    if (semi_minor) *semi_minor = std::sqrt(std::fmax(l2, 0.f));
    // 主轴方向：对应特征向量
    if (angle_rad) {
        // 若 b≈0，主轴与坐标轴对齐
        if (std::fabs(b) < 1e-12f) *angle_rad = (a >= c) ? 0.f : 1.5707963f;
        else                        *angle_rad = std::atan2(l1 - a, b);
    }
}

} // namespace gsplat
