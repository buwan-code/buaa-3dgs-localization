// camera.h — 相机与 EWA 投影数学（M2 核心）
//
// ── 这个模块负责什么 ───────────────────────────────────────────────
// 3DGS 的渲染管线里，"一个高斯怎么变成屏幕上的椭圆"是最核心的一步。
// 它由三段构成：
//   ① 世界 → 视空间：4×4 视矩阵（刚体变换，无缩放）
//   ② 视空间 → 裁剪空间：透视投影矩阵
//   ③ 3D 协方差 → 2D 协方差：**EWA splatting**（Zwicker 2001）
//
// 第③步是本文件的主角。原理：
//   一个 3D 高斯在视空间里是椭球，其等值面 x^T Σ⁻¹ x = const。
//   把这个椭球投影到图像平面（透视除法是非线性的），
//   局部可以用**雅可比矩阵 J** 线性化，于是
//        Σ_2D = J · W · Σ_3D · Wᵀ · Jᵀ
//   其中 W 是视旋转，J 是透视投影在该点处的雅可比。
//   再对 Σ_2D 做特征分解，就得到椭圆的轴长与朝向。
//
// ── 为什么必须在 CPU 侧算 ──────────────────────────────────────────
// ⚠️ M0 实测（NOTES.md §0.1 结论二）：顶点着色器里出现 sqrt/cos/sin 会
//    让帧率掉 14.7 倍。而 Σ_2D 的计算天然包含 sqrt（特征值）与
//    atan2（朝向角）—— 所以**绝对不能在 shader 里做**。
//    本模块在 CPU 批量算好每个高斯的 (中心xy, 轴长xy, 朝向) 再上传。
//
// ── 参考实现 ──────────────────────────────────────────────────────
// gsplat / INRIA 的 diff-gaussian-rasterization 里的
// computeCov2DCUDA / computeCov3D。本文件是其可读的标量版本。
#pragma once

#include <cmath>
#include <cstring>

namespace gsplat {

// ── 极简数学类型（避免拉入 glm/eigen，板端保持零依赖）────────────
struct Vec3 { float x, y, z; };
struct Vec4 { float x, y, z, w; };

// 3×3 矩阵，行主序 m[row][col]
struct Mat3 {
    float m[3][3];
    static Mat3 identity() {
        Mat3 r{}; r.m[0][0] = r.m[1][1] = r.m[2][2] = 1.f; return r;
    }
    static Mat3 zero() { Mat3 r{}; return r; }

    Mat3 operator*(const Mat3& o) const {
        Mat3 r{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                float s = 0.f;
                for (int k = 0; k < 3; ++k) s += m[i][k] * o.m[k][j];
                r.m[i][j] = s;
            }
        return r;
    }
    Mat3 transpose() const {
        Mat3 r{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) r.m[i][j] = m[j][i];
        return r;
    }
};

// 4×4 矩阵，列主序（与 OpenGL 一致，可直接 glUniformMatrix4fv）
struct Mat4 {
    float m[16];

    static Mat4 identity() {
        Mat4 r{}; r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.f; return r;
    }

    // 从列主序数组构造
    static Mat4 from_cols(float m00,float m10,float m20,float m30,
                          float m01,float m11,float m21,float m31,
                          float m02,float m12,float m22,float m32,
                          float m03,float m13,float m23,float m33) {
        Mat4 r{};
        r.m[0]=m00; r.m[1]=m10; r.m[2]=m20; r.m[3]=m30;
        r.m[4]=m01; r.m[5]=m11; r.m[6]=m21; r.m[7]=m31;
        r.m[8]=m02; r.m[9]=m12; r.m[10]=m22; r.m[11]=m32;
        r.m[12]=m03; r.m[13]=m13; r.m[14]=m23; r.m[15]=m33;
        return r;
    }

    Mat4 operator*(const Mat4& o) const {
        Mat4 r{};
        for (int c = 0; c < 4; ++c)
            for (int row = 0; row < 4; ++row) {
                float s = 0.f;
                for (int k = 0; k < 4; ++k) s += m[k*4 + row] * o.m[c*4 + k];
                r.m[c*4 + row] = s;
            }
        return r;
    }

    // 变换点（w 会被算出来，调用方自行做透视除法）
    Vec4 transform(const Vec4& v) const {
        return Vec4{
            m[0]*v.x + m[4]*v.y + m[8] *v.z + m[12]*v.w,
            m[1]*v.x + m[5]*v.y + m[9] *v.z + m[13]*v.w,
            m[2]*v.x + m[6]*v.y + m[10]*v.z + m[14]*v.w,
            m[3]*v.x + m[7]*v.y + m[11]*v.z + m[15]*v.w
        };
    }

    // 取出左上 3×3（用于协方差的旋转部分）
    Mat3 upper3x3() const {
        Mat3 r{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) r.m[i][j] = m[j*4 + i];
        return r;
    }
};

// ── 四元数 → 旋转矩阵 ──────────────────────────────────────────────
// ⚠️ 调用方保证 (w,x,y,z) **已归一化**。
//    ply_loader.cpp 在加载时已归一化（PLY 原始数据是未归一化的，
//    实测 |q| ∈ [0.61, 3.21]）。
//    这里原先有一次防御性归一化，被删掉了 —— 它在热路径上（每帧
//    277 万次）白做一次 sqrt。若将来有新的数据入口（不经过
//    ply_loader），必须在入口处自己归一化。
inline Mat3 quat_to_mat3(float w, float x, float y, float z) {
    Mat3 R{};
    R.m[0][0] = 1 - 2*(y*y + z*z);
    R.m[0][1] =     2*(x*y - w*z);
    R.m[0][2] =     2*(x*z + w*y);
    R.m[1][0] =     2*(x*y + w*z);
    R.m[1][1] = 1 - 2*(x*x + z*z);
    R.m[1][2] =     2*(y*z - w*x);
    R.m[2][0] =     2*(x*z - w*y);
    R.m[2][1] =     2*(y*z + w*x);
    R.m[2][2] = 1 - 2*(x*x + y*y);
    return R;
}

// ── 相机 ───────────────────────────────────────────────────────────
// 本项目定位算法给的是位姿（世界→相机的外参），所以相机以内参+外参表达。
// 坐标约定与 OpenCV/COLMAP 一致（x 右, y 下, z 前），便于与定位算法对齐；
// 上传 GPU 时再转 OpenGL 约定（y 上, z 后）。
struct Camera {
    // 内参（像素）
    float fx = 500.f, fy = 500.f, cx = 320.f, cy = 240.f;
    int   width = 640, height = 480;
    // 近远平面
    float znear = 0.05f, zfar = 200.f;
    // 外参：世界 → 相机
    Mat4  world_to_cam = Mat4::identity();
    // 高斯 3σ 半径的屏幕上限（像素）。超过则视为「退化的巨大高斯」剔除。
    // ⚠️ 这是**质量/性能权衡**参数，不是纯数学：
    //    - 太小：前景大高斯（地面、天空、近景墙面）被误删 → 画面破洞
    //    - 太大：单个高斯覆盖全屏，片元填充量爆炸 → 帧率崩
    //    本项目给定位算法生成参考视图，破洞会破坏特征，所以宁可放宽。
    //    默认取 4× 屏幕对角线（很宽松），真正的大高斯靠自适应下采样处理。
    float max_sigma_radius = 0.f;   // <=0 表示用默认 = 4× 对角线

    // ── 由「注视目标」构造一个轨道相机（方便调试与生成参考视图）──
    // 注意：这里刻意**不在顶点着色器里算三角函数**（M0 铁律），
    // 相机基向量在 CPU 侧一次算好。
    static Camera look_at(const Vec3& eye, const Vec3& target, const Vec3& up,
                          int w, int h, float fovy_deg = 45.f);
};

// ── 一个高斯的投影结果（CPU 侧算好，直接喂 GPU）──────────────────
struct Splat2D {
    float cx, cy;        // 屏幕像素坐标（中心）
    float a, b, c;       // 2D 协方差的上三角：[[a, b], [b, c]]
    float depth;         // 视空间 z（排序键）
    float inv_det;       // 1/(a*c - b*b)，片元着色器直接用
    float sigma_radius;  // 3σ 半径（像素），用于 quad 尺寸与剔除
    bool  valid;         // 是否通过可见性判别
};

// ── EWA 投影：3D 高斯 → 屏幕 2D 椭圆 ──────────────────────────────
// 输入：
//   mean3d   高斯中心（世界坐标）
//   scale3   三个轴的尺度（已 exp，即真实标准差）
//   quat     朝向四元数 (w,x,y,z)
//   cam      相机
//   cov6_out 可选，输出 3D 协方差（诊断/调试用，长度 6，上三角）
//   d_out    可选，输出深度
// 返回：
//   Splat2D，含屏幕中心、2D 协方差、3σ 半径；valid=false 表示应剔除
//
// 数学：Σ_2D = J W Σ_3D Wᵀ Jᵀ
//   - Σ_3D = R S Sᵀ Rᵀ （R 由四元数给出，S = diag(scale)）
//   - W 是视矩阵的 3×3 旋转部分
//   - J 是透视投影雅可比，形式为
//       [ fx/z,    0,   -fx*x/z² ]
//       [   0,   fy/z, -fy*y/z² ]
Splat2D project_gaussian(const Vec3& mean3d,
                         const Vec3& scale3,
                         float qw, float qx, float qy, float qz,
                         const Camera& cam,
                         float* cov6_out = nullptr,
                         float* d_out = nullptr);

// ── 构造 3D 协方差 Σ = R S Sᵀ Rᵀ（供上面的函数与单元测试用）───────
// 输出上三角 6 元组：cov[0]=Σxx, [1]=Σxy, [2]=Σxz, [3]=Σyy, [4]=Σyz, [5]=Σzz
void compute_cov3d(const Vec3& scale3, float qw, float qx, float qy, float qz,
                   float cov[6]);

// ── 2D 协方差 → 椭圆几何量（轴 + 朝向），调试用 ──────────────────
// 返回 (semi_major, semi_minor, angle_rad)
void cov2d_to_ellipse(const Splat2D& s, float* semi_major,
                      float* semi_minor, float* angle_rad);

} // namespace gsplat
