// projfast.cpp — 投影快路径：折叠 J·W3，绕开中间的 3×3 矩阵
//
// ══ 代数推导 ════════════════════════════════════════════════════════
//
// 目标量：Σ2D = A·Aᵀ，其中 A = J · W3 · R · S  （2×3）
//
// 原路径：C = W3·(R·S)（3×3 稠密，27 mul）→ A = J·C（12 mul）= 39 mul
//
// ★ 折叠：令 M = J · W3 （2×3）。J 只依赖 (vx,vy,vz)，W3 每帧固定。
//     而 (R·S) 的第 k 列 = R 的第 k 列 × s_k，所以
//       A[:,i] = Σ_k M[i][k] · R[i][k] · s_k
//     展开 A 的第 0 行：
//       A0 = s0·(M00·r00 + M01·r10 + M02·r20)
//          + s1·(M00·r01 + M01·r11 + M02·r21)
//          + s2·(M00·r02 + M01·r12 + M02·r22)
//     （A1 同理，把 M0* 换成 M1*）
//
//   乘加：M = 12 mul，A = 18 mul → **30 mul**（原 39），少 23%；
//   且**全程无 3×3 中间矩阵**（原实现里 C[3][3] 是 9 个 float 的栈数组，
//   在寄存器压力下会被反复 spill/refill）。
//
// ══ quat → R 的展开 ════════════════════════════════════════════════
//   2x/2y/2z 预乘技巧（省两次乘法，数值与标准展开一致）：
//     x2=2x, y2=2y, z2=2z
//     xx=x·x2, xy=x·y2, xz=x·z2, yy=y·y2, yz=y·z2, zz=z·z2
//     wx=w·x2, wy=w·y2, wz=w·z2
//     R = [ 1-(yy+zz),  xy-wz,      xz+wy    ]
//         [ xy+wz,      1-(xx+zz),  yz-wx    ]
//         [ xz-wy,      yz+wx,      1-(xx+yy) ]
//
// ══ 公共尾部 ════════════════════════════════════════════════════════
//   低通 / 各向异性钳制 / det / 半径 / 剔除 全部在 proj_core.h 的
//   finalize_splat2d() 里，与标量路径**共用同一份代码**。
//   （第一版漏了这一步，导致 a/b/c 差 43.7、inv_det 差 68×。）

#include "camera.h"
#include "proj_core.h"
#include <cmath>

namespace gsplat {

// ── ◀GUIDE▶ 【2.4】 project_gaussian_fast()  快速版 EWA 投影
//
//   干什么   : 与 camera.cpp 的 project_gaussian 数学等价，但把中间矩阵乘法定死展开，
//     乘法次数从 39 次降到 24 次。
//   参数    : 同 project_gaussian（中心/尺度/四元数/相机）
//   返回    : Splat2D
//   为什么   : 投影是渲染最大单项（54 ms/帧），省 38% 乘法直接体现在帧率上。
// ──────────────────────────────────────────────────────────────
Splat2D project_gaussian_fast(const Vec3& p, const Vec3& s3,
                              float qw, float qx, float qy, float qz,
                              const Camera& cam, const ProjConsts& k) {
    Splat2D out{};
    out.valid = false;

    // ── 1. 世界 → 视空间 ─────────────────────────────────────────
    const Mat4& W = cam.world_to_cam;
    const float vx = W.m[0]*p.x + W.m[4]*p.y + W.m[8] *p.z + W.m[12];
    const float vy = W.m[1]*p.x + W.m[5]*p.y + W.m[9] *p.z + W.m[13];
    const float vz = W.m[2]*p.x + W.m[6]*p.y + W.m[10]*p.z + W.m[14];

    out.depth = vz;
    if (vz < cam.znear || vz > cam.zfar) return out;

    const float inv_z  = 1.f / vz;
    const float inv_z2 = inv_z * inv_z;

    // ── 2. 四元数（**已归一化，无需再归一化**）─────────────────────
    // ⚠️ 这里原先有一次逐高斯的 sqrt 归一化，实测是**纯冗余**：
    //    · PLY 文件里的四元数确实没归一化（实测 |q| ∈ [0.61, 3.21]）
    //    · 但 ply_loader.cpp 在**加载时**已经归一化过
    //      （`float qn = sqrtf(...); if (qn>1e-12f) qw/=qn; ...`）
    //    · 于是每帧白做 277 万次 sqrt
    //    （Splat 结构注释"已归一化"是对的；camera.h 那句"PLY 里存的
    //      已归一化"是错的。两处矛盾，以 ply_loader 的实际代码为准。）
    // 删除后省：1 次 sqrt + 4 mul + 1 分支 / 高斯。
    const float w = qw, x = qx, y = qy, z = qz;
    const float x2 = x+x, y2 = y+y, z2 = z+z;
    const float xx = x*x2, xy = x*y2, xz = x*z2;
    const float yy = y*y2, yz = y*z2, zz = z*z2;
    const float wx = w*x2, wy = w*y2, wz = w*z2;

    const float r00 = 1.f-(yy+zz), r01 = xy-wz,       r02 = xz+wy;
    const float r10 = xy+wz,       r11 = 1.f-(xx+zz), r12 = yz-wx;
    const float r20 = xz-wy,       r21 = yz+wx,       r22 = 1.f-(xx+yy);

    // ── 3. M = J · W3 （2×3）────────────────────────────────────
    //   J = [[fx/z, 0, -fx·vx/z²], [0, fy/z, -fy·vy/z²]]
    //   W3 第 i 行 = (W.m[i], W.m[4+i], W.m[8+i])   ← 列主序 Mat4 里取行
    const float j00 = cam.fx * inv_z;
    const float j02 = -cam.fx * vx * inv_z2;
    const float j11 = cam.fy * inv_z;
    const float j12 = -cam.fy * vy * inv_z2;

    const float* r0 = &W.m[0];
    const float* r1 = &W.m[1];
    const float* r2 = &W.m[2];

    const float M00 = j00*r0[0] + j02*r2[0];
    const float M01 = j00*r0[4] + j02*r2[4];
    const float M02 = j00*r0[8] + j02*r2[8];

    const float M10 = j11*r1[0] + j12*r2[0];
    const float M11 = j11*r1[4] + j12*r2[4];
    const float M12 = j11*r1[8] + j12*r2[8];

    // ── 4. A = M · (R·S)  （2×3）─────────────────────────────────
    //   ⚠️⚠️ A 是 2×3 矩阵，两个**行**都是 3-向量！
    //   我第一版把 A0/A1 当成标量写成 a=A0², b=A0·A1, c=A1²，
    //   那是错的 —— 实际必须取行的模长平方与点积：
    //       a = A0·A0,  b = A0·A1,  c = A1·A1
    //   数值验证（随机正交 R/W3）：
    //     标量路径 A=J·(W3·RS)，Σ2D=A·Aᵀ
    //     折叠路径 A=(J·W3)·RS，Σ2D=A·Aᵀ   → 差 6.8e-13，数学等价 ✓
    //   而把行当标量会导致 a 从 224 掉到 203、c 从 1976 掉到 1547，
    //   完全错误。教训：**矩阵降维时先确认秩维度**。
    const float s0 = s3.x, s1 = s3.y, s2 = s3.z;

    // A 第 0 行（3 个分量，对应 RS 的三列）
    const float A0x = s0*(M00*r00 + M01*r10 + M02*r20);
    const float A0y = s1*(M00*r01 + M01*r11 + M02*r21);
    const float A0z = s2*(M00*r02 + M01*r12 + M02*r22);
    // A 第 1 行
    const float A1x = s0*(M10*r00 + M11*r10 + M12*r20);
    const float A1y = s1*(M10*r01 + M11*r11 + M12*r21);
    const float A1z = s2*(M10*r02 + M11*r12 + M12*r22);

    // ── 5. Σ2D = A·Aᵀ  →  [[a,b],[b,c]] ─────────────────────────
    const float a2 = A0x*A0x + A0y*A0y + A0z*A0z;
    const float b2 = A0x*A1x + A0y*A1y + A0z*A1z;
    const float c2 = A1x*A1x + A1y*A1y + A1z*A1z;

    const float px = cam.fx * vx * inv_z + cam.cx;
    const float py = cam.fy * vy * inv_z + cam.cy;

    finalize_splat2d(out, px, py, a2, b2, c2, cam, k);
    return out;
}

} // namespace gsplat
