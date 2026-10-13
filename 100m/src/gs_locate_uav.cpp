// gs_locate_uav.cpp — 无人机 3DGS 视觉定位（吃真照片 → 吐位姿）
//
// ══ 与 gs_localize.cpp 的区别 ══════════════════════════════════════
// gs_localize.cpp 是**自证闭环**：用真值位姿渲染一张图当"实拍图"，
// 再扰动初值去解算，验证的是「渲染→匹配→PnP」这条链本身。
//
// 本程序是**真实工况**：
//     输入 = 一个装满真照片的目录 + 一个训练好的 PLY
//     输出 = 每帧的估计位姿（写文件）+ 与真值的误差统计
//
// ══ 数据事实（2026-10-08 实测）══════════════════════════════════════
// 照片 : E:/PiLoT-demo/data_demo/query/images/3dgs_test/  300 张 3840x2160 PNG
// 真值 : E:/PiLoT-demo/data_demo/query/poses/3dgs_test.txt
//          格式 <png> <lon> <lat> <alt> <a1> <a2> <a3>
//          lon≈112.99 lat≈28.29 alt≈143.1  → WGS84 经纬高
//          轨迹 dE=142.3m dN=55.8m，高度几乎恒定 → 悬停环拍
// PLY  : pilot.ply 2767560 高斯，包围盒 span(186.38, 152.68, 42.19)
//          → **局部米制坐标，独立原点**（非经纬高、非投影坐标）
// 内参 : feicuiwan.yaml cam_query.params=[2700, 2700, 1915.7, 1075.1]
//          （对应 3840x2160；UAVD4L 的 960x540 标定 ×4 = 2700/1920/1080）
//
// ══ 坐标系处理（关键设计）══════════════════════════════════════════
// PLY 的原点是 3DGS 训练时自动选的，**我们不知道具体值**。
// 但这不影响定位精度评估，因为误差是**相对量**：
//   只要真值位姿和估计位姿都通过同一个 GeoRef 转到"同一个局部系"，
//   两者相减得到的误差就是正确的。
// 所以本程序：
//   1) 用 --ref 指定一个参考点（默认取真值第一帧），建立 ENU 局部系
//   2) 真值经纬高 → ENU 局部米制 → 作为 GT
//   3) 估计位姿也在 ENU 局部米制里解算（PnP 直接在这个系里做）
//   4) GT 与估计相减 → 误差
// ⚠️ 唯一的前提：**PLY 的局部系与 ENU 局部系之间是一个刚体变换**
//    （平移+旋转，无缩放）。对 3DGS 训练输出而言这个前提成立。
//    如果 PLY 用了 Y 轴朝下（部分实现如此），需要一个翻转 —— 用
//    --ply-y-down 开关处理。

#include "renderer.h"
#include "camera.h"
#include "ply_loader.h"
#include "geo_pose.h"
#include "xfeat_frontend.h"
#include "pose_post.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <dirent.h>

using namespace gsplat;

// ── 计时 ──────────────────────────────────────────────────────────
// ── ◀GUIDE▶ 【0.1】 ms_since()  计时小工具
//
//   干什么   : 算「从 t0 到现在」经过了多少毫秒。用来给每一步掐表。
//   参数    : t0 —— 一个时间戳（由 Clock::now() 得到）
//   返回    : double，毫秒数
//   用法    : auto t0 = Clock::now(); ... ; double ms = ms_since(t0);
// ──────────────────────────────────────────────────────────────
static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}
using Clock = std::chrono::steady_clock;
using ClockA = std::chrono::steady_clock;

// ── 随机扰动（可复现：固定种子）────────────────────────────────────
static std::mt19937 g_rng(42);
// ── ◀GUIDE▶ 【0.2】 u01()  取一个 [0,1) 随机数
//
//   干什么   : 返回 0~1 之间的随机小数。用于首帧位姿扰动。
//   参数    : 无
//   返回    : double ∈ [0,1)
//   为什么   : 全局随机数发生器 g_rng 固定种子 42 → 同一 seed 每次跑结果完全一样，
//     这是科研可复现性的要求（别人能复现你的数字）。
// ──────────────────────────────────────────────────────────────
static inline double u01() {
    return std::uniform_real_distribution<double>(0.0, 1.0)(g_rng);
}

// ══ 欧拉角分解（轴序 = Rz(yaw)·Rx(pitch)·Ry(roll)）═════════════════
// ★★ 重大修正（2026-10-08 实测确定）★★
//   真值文件的 (a1,a2,a3) = (roll, pitch, yaw)，但其姿态矩阵的实际轴序是
//       Rpose = Rz(yaw) · Rx(pitch) · Ry(roll)
//   而不是 geo_pose.h 里 euler_xyz_to_mat 实现的 Rz·Ry·Rx。
//   证据：对估计矩阵用 Rz·Rx·Ry 分解 → (0.4°, 35.7°, 108.2°)，与真值
//        (0°, 35.6°, 108.9°) 三个角全部吻合（误差 <0.7°）；
//        用 Rz·Ry·Rx 分解则得到 (35.7°, 0.3°, 108.5°)（roll/pitch 互换）。
//   这也解释了为何构造初始位姿必须加 --yaw-offset 270（等效 -90°）：
//        两种轴序对同一光轴方位角的解读相差 90°。
//
//   推导（M = Rz(c)·Rx(b)·Ry(a)）：
//       M[2][1] = sin(b)          → pitch = asin(M[2][1])
//       M[2][0] = -cos(b)·sin(a)  → roll  = atan2(-M[2][0], M[2][2])
//       M[0][1] = -sin(c)·cos(b)  → yaw   = atan2(-M[0][1], M[1][1])
//       M[1][1] =  cos(c)·cos(b)
// ── ◀GUIDE▶ 【6.1】 euler_from_R()  旋转矩阵 → 欧拉角
//
//   干什么   : 把 3×3 旋转矩阵拆成 roll/pitch/yaw 三个角（度），用于输出与误差比较。
//   参数    : R —— 3×3 旋转矩阵；roll/pitch/yaw —— 输出用的引用（double&）
//   返回    : 无（结果写进三个引用参数）
//   ★★ 2026-10-13 数值实验修正 ★★
//   原先这里写 Rz(yaw)·Rx(pitch)·Ry(roll)，注释还声称"真值就是这个轴序"。
//   那是**错的**。用 04_make_gt.py 生成的 gt 直接代入两个候选口径算光轴：
//     gt(0187) = (roll 44.244, pitch 0.967, yaw -52.665)
//       标准 ZYX  Rz(y)Ry(p)Rx(r)  ->  方位  51.67°  下倾 45.75°   ← 与 DJI 云台
//       Rz(y)Rx(p)Ry(r)（旧）      ->  方位 -36.34°  下倾 45.75°      真值
//                                                                  (52.0°, 45.0°)
//   只有标准 ZYX 与真值吻合。旧口径把 roll 当成绕 X 轴、pitch 当成绕 Y 轴，
//   导致姿态误差假性放大到 ~173°。故此处统一为标准 ZYX。
//   标准 ZYX 反解： R = Rz(y)·Ry(p)·Rx(r)
//       R[2][0] = -sin(p)          → pitch = asin(-R[2][0])
//       R[2][1] =  cos(p)·sin(r)   → roll  = atan2(R[2][1], R[2][2])
//       R[1][0] =  sin(y)·cos(p)   → yaw   = atan2(R[1][0], R[0][0])
//   （这正是 04_make_gt.py::euler_from_R_pilot 的公式，两端现已一致）
// ──────────────────────────────────────────────────────────────
static void euler_from_R(const geopp::Mat3& R,
                         double& roll, double& pitch, double& yaw) {
    double v = -R.m[2][0];
    v = v < -1.0 ? -1.0 : (v > 1.0 ? 1.0 : v);
    pitch = std::asin(v) * 180.0 / M_PI;
    roll  = std::atan2(R.m[2][1], R.m[2][2]) * 180.0 / M_PI;
    yaw   = std::atan2(R.m[1][0], R.m[0][0]) * 180.0 / M_PI;
}

// 用同一轴序构造姿态矩阵（供真值比较用）
// ── ◀GUIDE▶ 【6.2】 pose_mat_of()  欧拉角 → 旋转矩阵（与上面互逆）
//
//   干什么   : 用与 euler_from_R 相同的轴序构造旋转矩阵，供真值比较使用。
//   参数    : roll/pitch/yaw —— 三个角（度）
//   返回    : 3×3 旋转矩阵
//   ★★ 与 euler_from_R 同步改为标准 ZYX（Rz·Ry·Rx），与渲染路径一致。
// ──────────────────────────────────────────────────────────────
static geopp::Mat3 pose_mat_of(double roll, double pitch, double yaw) {
    const double k = M_PI / 180.0;
    return geopp::rot_z(yaw * k) * geopp::rot_y(pitch * k) * geopp::rot_x(roll * k);
}

// 从 Camera（w2c）提取 camera→world 旋转（行主序）
// ── ◀GUIDE▶ 【6.3】 c2w_of()  从 Camera 取出 camera→world 旋转
//
//   干什么   : Camera 内部存的是 world→camera；差评要用 c2w，这里取出来转置一下。
//   参数    : c —— 相机
//   返回    : 3×3 矩阵，camera→world
//   ★坑    : Camera::world_to_cam 是**列主序**存放（GL 习惯），
//     取元素要用 m[列*4+行]，别写成 m[行*3+列]。
// ──────────────────────────────────────────────────────────────
static geopp::Mat3 c2w_of(const Camera& c) {
    geopp::Mat3 w2c;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            w2c.m[i][j] = c.world_to_cam.m[j * 4 + i];
    return geopp::transpose(w2c);
}

// 旋转矩阵 → 四元数 (w,x,y,z)。Mat3 为 m[行][列]。
// ── ◀GUIDE▶ 【6.4】 quat_from_R()  旋转矩阵 → 四元数
//
//   干什么   : 把旋转矩阵转成四元数 (w,x,y,z)，写进输出文件的 qw qx qy qz 四列。
//   参数    : R —— 3×3 旋转矩阵；qw..qz —— 输出引用
//   返回    : 无
//   算法    : Shepperd 法：看矩阵迹大小分 4 支，选数值最稳的那支，避免除零。
//   ★约定   : 最后强制 qw>0（取「正半球」），否则 q 与 -q 表示同一姿态但符号会跳，
//     导致姿态误差曲线出现 360° 假跳变。
// ──────────────────────────────────────────────────────────────
static void quat_from_R(const geopp::Mat3& R, double& qw, double& qx, double& qy, double& qz) {
    const double tr = R.m[0][0] + R.m[1][1] + R.m[2][2];
    if (tr > 0.0) {
        const double s = std::sqrt(tr + 1.0) * 2.0;
        qw = 0.25 * s;
        qx = (R.m[2][1] - R.m[1][2]) / s;
        qy = (R.m[0][2] - R.m[2][0]) / s;
        qz = (R.m[1][0] - R.m[0][1]) / s;
    } else if (R.m[0][0] > R.m[1][1] && R.m[0][0] > R.m[2][2]) {
        const double s = std::sqrt(1.0 + R.m[0][0] - R.m[1][1] - R.m[2][2]) * 2.0;
        qw = (R.m[2][1] - R.m[1][2]) / s; qx = 0.25 * s;
        qy = (R.m[0][1] + R.m[1][0]) / s; qz = (R.m[0][2] + R.m[2][0]) / s;
    } else if (R.m[1][1] > R.m[2][2]) {
        const double s = std::sqrt(1.0 + R.m[1][1] - R.m[0][0] - R.m[2][2]) * 2.0;
        qw = (R.m[0][2] - R.m[2][0]) / s; qx = (R.m[0][1] + R.m[1][0]) / s;
        qy = 0.25 * s;                   qz = (R.m[1][2] + R.m[2][1]) / s;
    } else {
        const double s = std::sqrt(1.0 + R.m[2][2] - R.m[0][0] - R.m[1][1]) * 2.0;
        qw = (R.m[1][0] - R.m[0][1]) / s; qx = (R.m[0][2] + R.m[2][0]) / s;
        qy = (R.m[1][2] + R.m[2][1]) / s; qz = 0.25 * s;
    }
    const double nn = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
    if (nn > 1e-12) { qw /= nn; qx /= nn; qy /= nn; qz /= nn; }
    if (qw < 0) { qw = -qw; qx = -qx; qy = -qy; qz = -qz; }   // 取正半球，便于比较
}

// 角度差归一到 [-180, 180]
// ── ◀GUIDE▶ 【6.5】 ang_diff()  角度差归一到 [-180,180]
//
//   干什么   : 算 a-b 并把结果绕回 ±180° 以内。
//   参数    : a, b —— 两个角度（度）
//   返回    : 差值（度），∈[-180,180]
//   为什么   : 360° 和 0° 是同一个方向。不归一的话 359-1=358 会被误判成大误差。
// ──────────────────────────────────────────────────────────────
static double ang_diff(double a, double b) {
    double d = a - b;
    while (d > 180.0) d -= 360.0;
    while (d < -180.0) d += 360.0;
    return d;
}

// 旋转矩阵测地距离（度）
// ── ◀GUIDE▶ 【6.6】 rot_geo_deg()  两个旋转之间的测地距离（度）
//
//   干什么   : 算两个 3×3 旋转矩阵「差了多少度」。这是旋转误差 mR 的定义。
//   参数    : A —— 估计旋转；B —— 真值旋转
//   返回    : 夹角（度）
//   公式    : D=AᵀB（相对旋转）→ trace(D)=1+2cosθ → θ=acos((tr-1)/2)
//   为什么   : 比「三个欧拉角分别作差」更严谨：不受万向锁影响，是真正的旋转距离。
// ──────────────────────────────────────────────────────────────
static double rot_geo_deg(const geopp::Mat3& A, const geopp::Mat3& B) {
    const geopp::Mat3 d = geopp::transpose(A) * B;   // 相对旋转
    const double tr = d.m[0][0] + d.m[1][1] + d.m[2][2];
    double cs = (tr - 1.0) * 0.5;
    cs = cs < -1.0 ? -1.0 : (cs > 1.0 ? 1.0 : cs);
    return std::acos(cs) * 180.0 / M_PI;
}

// ── 渲染帧 → 灰度（含 GL 行序翻转，见 gs_localize.cpp 的教训）──────
// ── ◀GUIDE▶ 【5.5】 to_gray()  渲染输出 RGBA → 灰度图
//
//   干什么   : 把 GPU 回读的 RGBA8 缓冲转成 OpenCV 单通道灰度图，喂给特征提取。
//   参数    : rgba —— 长度 W*H*4 的字节数组；W,H —— 宽高
//   返回    : cv::Mat，CV_8UC1（H 行 W 列，行 0 = 图像顶部）
//   ★坑1   : GL 的图像原点是**左下角**，所以内部读的是第 (H-1-y) 行 —— 要上下翻转。
//   ★坑2   : 灰度权重用整数 77/150/29 再右移 8 位（≈0.299R+0.586G+0.113B），
//     比浮点快，且结果与 OpenCV 一致。
// ──────────────────────────────────────────────────────────────
static cv::Mat to_gray(const std::vector<uint8_t>& rgba, int W, int H) {
    cv::Mat g(H, W, CV_8UC1);
    for (int y = 0; y < H; ++y) {
        const uint8_t* s = rgba.data() + (size_t)(H - 1 - y) * W * 4;
        uint8_t* d = g.ptr<uint8_t>(y);
        for (int x = 0; x < W; ++x) {
            d[x] = (uint8_t)((77 * s[0] + 150 * s[1] + 29 * s[2]) >> 8);
            s += 4;
        }
    }
    return g;
}

// ── ◀GUIDE▶ 【5.6】 flip_depth_rows()  深度图上下翻转
//
//   干什么   : 深度图也是 GL 行序（左下角原点），转成「行0=顶部」的常规图像行序。
//   参数    : src 输入 / dst 输出（自动 resize）；W,H 宽高
//   返回    : 无
//   为什么   : 深度要与灰度图逐像素对齐，才能用同一个 (u,v) 反投影出 3D 点。
// ──────────────────────────────────────────────────────────────
static void flip_depth_rows(const std::vector<float>& src,
                            std::vector<float>& dst, int W, int H) {
    dst.resize((size_t)W * H);
    for (int y = 0; y < H; ++y)
        std::memcpy(&dst[(size_t)y * W],
                    &src[(size_t)(H - 1 - y) * W],
                    sizeof(float) * W);
}


// ═══════════════════════════════════════════════════════════════════
//  ★ DUMP-DEPTH-PATCH (2026-10-11)
//  导出渲染深度（float32, W*H, 行主序）+ 配套相机位姿 .txt
//  供离线特征对比实验在 Python 里精确复刻 backproject + solvePnPRansac。
// ═══════════════════════════════════════════════════════════════════
// ── ◀GUIDE▶ 【工具】 dump_depth_bin()  导出深度 .bin + 相机参数 .txt
//
//   干什么   : 把一帧深度写成二进制 float32 数组，并存一份相机参数 txt，供 Python 离线复算。
//   参数    : binpath 输出路径；d 深度数组；W,H 尺寸；cam 该帧相机
//   返回    : 无（写文件）
//   谁的用   : 跑离线特征对比实验（Python 里复刻 backproject + solvePnPRansac）时用。
//   对应开关  : --dump-depth
// ──────────────────────────────────────────────────────────────
static void dump_depth_bin(const char* binpath, const std::vector<float>& d,
                           int W, int H, const Camera& cam,
                           const std::vector<float>* wsum = nullptr) {
    FILE* fp = fopen(binpath, "wb");
    if (!fp) return;
    fwrite(d.data(), sizeof(float), (size_t)W * H, fp);
    fclose(fp);
    // ★ 可选：同时导出「深度权重和 Σαᵢ」（--dump-wsum）
    //   期望深度 ED = Σαᵢzᵢ / Σαᵢ，分母 Σαᵢ 就是这里导出的量。
    //   Σαᵢ 小 → 该像素的射线只擦到高斯边缘（弱覆盖）→ ED 数值不稳、不可信。
    //   渲染器早就算好并存下了它（Renderer::get_depth_weights），
    //   但主程序至今从未调用过 —— 这里是第一次真正用上。
    if (wsum && (int)wsum->size() == W * H) {
        std::string wp(binpath); wp += ".wsum";
        FILE* fw = fopen(wp.c_str(), "wb");
        if (fw) {
            fwrite(wsum->data(), sizeof(float), (size_t)W * H, fw);
            fclose(fw);
        }
    }
    std::string sp(binpath); sp += ".txt";
    FILE* ft = fopen(sp.c_str(), "w");
    if (!ft) return;
    const float* m = cam.world_to_cam.m;      // Mat4 列主序
    for (int r = 0; r < 3; ++r) {             // world_to_cam 的 3x4，行主序
        for (int c = 0; c < 3; ++c) fprintf(ft, "%.9g ", (double)m[c * 4 + r]);
        fprintf(ft, "%.9g\n", (double)m[12 + r]);
    }
    fprintf(ft, "%.9g %.9g %.9g %.9g\n",
            (double)cam.fx, (double)cam.fy, (double)cam.cx, (double)cam.cy);
    fprintf(ft, "%d %d\n", W, H);
    fclose(ft);
}

// ═══════════════════════════════════════════════════════════════════
//  ★ DUMP-COLOR-PATCH (2026-10-11)
//  把渲染器的 RGBA 缓冲转成 BGR（供 --dump-color 导出彩色渲染图）
//  ⚠️ 与 to_gray 一致：缓冲按行**上下翻转**存放（见 to_gray 的 H-1-y），
//     且通道序为 R,G,B,A（to_gray 的 77/150/29 权重即 R/G/B）。
// ═══════════════════════════════════════════════════════════════════
// ── ◀GUIDE▶ 【工具】 to_bgr()  渲染输出 RGBA → 彩色 BGR 图
//
//   干什么   : 与 to_gray 同理，但保留彩色（OpenCV 的 Mat 是 BGR 顺序，所以 R/B 要调换）。
//   参数    : rgba —— W*H*4 字节；W,H 宽高
//   返回    : cv::Mat，CV_8UC3
//   对应开关  : --dump-color（用来出展示图；匹配本身走灰度）
// ──────────────────────────────────────────────────────────────
static cv::Mat to_bgr(const std::vector<uint8_t>& rgba, int W, int H) {
    cv::Mat c(H, W, CV_8UC3);
    for (int y = 0; y < H; ++y) {
        const uint8_t* s = rgba.data() + (size_t)(H - 1 - y) * W * 4;
        uint8_t* d = c.ptr<uint8_t>(y);
        for (int x = 0; x < W; ++x) {
            d[3 * x + 0] = s[2];      // B
            d[3 * x + 1] = s[1];      // G
            d[3 * x + 2] = s[0];      // R
            s += 4;
        }
    }
    return c;
}

// ── ◀GUIDE▶ 【6.7】 K_of()  相机内参 → 3×3 内参矩阵 K
//
//   干什么   : 把 fx,fy,cx,cy 拼成 OpenCV 要的 3×3 内参矩阵 K。
//   参数    : c —— 相机
//   返回    : cv::Mat 3×3 double
//   K 长啥样 : [fx 0 cx; 0 fy cy; 0 0 1] —— 只做解析，不含畸变。
// ──────────────────────────────────────────────────────────────
static cv::Mat K_of(const Camera& c) {
    return (cv::Mat_<double>(3, 3) << c.fx, 0, c.cx,
                                      0, c.fy, c.cy,
                                      0, 0, 1);
}

// ── ◀GUIDE▶ 【6.8】 backproject()  ★ 像素 + 深度 → 世界 3D 点
//
//   干什么   : 【目标定位的核心】给图像上一个像素 (u,v) 和它的深度 z，算出这个像素在
//     世界坐标系里的三维点。这就是「图上点一下 → 告诉你地面/楼顶的经纬度」。
//   参数    : cam 相机；u,v 像素坐标（整数）；z 该像素的相机空间深度（米）
//   返回    : Vec3，世界坐标（此项目里 = PLY/ENU 局部米制）
//   两步    : ① 逆内参：x_c=(u+0.5-cx)/fx*z, y_c=(v+0.5-cy)/fy*z  →相机系点 (x_c,y_c,z)
//     ② 相机系→世界：X = Rᵀ·(P_cam - t)
//   ★坑    : ②里用的是 Rᵀ，代码写成 m[0]*dx+m[1]*dy+m[2]*dz —— 看着像「行乘法」，
//     其实 m[0],m[1],m[2] 是列主序存储下的**第0列**，正好是 Rᵀ 的第0行。
//     这个「看着别扭其实对」的写法是本项目最容易读错的地方之一。
// ──────────────────────────────────────────────────────────────
static Vec3 backproject(const Camera& cam, int u, int v, float z) {
    const float* m = cam.world_to_cam.m;
    const double xc = (u + 0.5 - cam.cx) / (double)cam.fx * z;
    const double yc = (v + 0.5 - cam.cy) / (double)cam.fy * z;
    const double dx = xc - m[12], dy = yc - m[13], dz = z - m[14];
    Vec3 o;
    o.x = (float)(m[0] * dx + m[1] * dy + m[2] * dz);
    o.y = (float)(m[4] * dx + m[5] * dy + m[6] * dz);
    o.z = (float)(m[8] * dx + m[9] * dy + m[10] * dz);
    return o;
}

// ── ◀GUIDE▶ 【8.3.4】 cam_from_pnp()  solvePnP 输出 → Camera
//
//   干什么   : PnP 解出来的 rvec/tvec 换成本项目统一使用的 Camera（含 world_to_cam）。
//   参数    : rvec 旋转向量(3x1)；tvec 平移向量(3x1)；proto 原型相机（提供内参/尺寸）
//   返回    : Camera
//   做什么   : ① cv::Rodrigues(rvec) 把旋转向量转成 3×3 矩阵
//     ② 按列主序写进 world_to_cam.m，平移写进 m[12..14]
// ──────────────────────────────────────────────────────────────
static Camera cam_from_pnp(const cv::Mat& rvec, const cv::Mat& tvec,
                           const Camera& proto) {
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    Camera c = proto;
    c.world_to_cam = Mat4::identity();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            c.world_to_cam.m[j * 4 + i] = (float)R.at<double>(i, j);
    c.world_to_cam.m[12] = (float)tvec.at<double>(0);
    c.world_to_cam.m[13] = (float)tvec.at<double>(1);
    c.world_to_cam.m[14] = (float)tvec.at<double>(2);
    return c;
}

// ── ◀GUIDE▶ 【6.9】 pose_of()  从 Camera 取出「相机中心」和「光轴方向」
//
//   干什么   : 相机中心 C = -Rᵀt；光轴方向 = R 的第 3 行（相机 z 轴在世界系下的方向）。
//   参数    : c 相机；C/fwd 输出引用
//   返回    : 无
//   ★坑    : Mat4 列主序下，光轴是 m[8],m[9],m[10]（= 第2列）没错，
//     但**旋转矩阵的第 3 行**才是光轴 —— 历史上写成「第3列」导致每帧错纠 30 m。
// ──────────────────────────────────────────────────────────────
static void pose_of(const Camera& c, Vec3& C, Vec3& fwd) {
    const float* m = c.world_to_cam.m;
    const double tx = m[12], ty = m[13], tz = m[14];
    C.x = (float)-(m[0] * tx + m[1] * ty + m[2] * tz);
    C.y = (float)-(m[4] * tx + m[5] * ty + m[6] * tz);
    C.z = (float)-(m[8] * tx + m[9] * ty + m[10] * tz);
    // ★★ 2026-10-13 修正 ★★
    //   world_to_cam 是**列主序**存放的 R_wc（世界→相机）。
    //   光轴 = 相机 Z 轴在世界系下的方向 = R_cw 的第 3 列 = R_wc 的第 3 行。
    //   列主序下 R_wc 第 r 行第 c 列在 m[c*4+r]，故第 3 行 = m[2], m[6], m[10]。
    //   原先写 m[8],m[9],m[10]（= R_wc 第 3 列 = R_cw 第 3 行），
    //   只有 z 分量凑巧相同（都含 m[10]），水平方位全错 ——
    //   实测第 1 帧 fwd 方位 -178.7° vs 真值 51.7°，虚假"朝向误差" 78.3°。
    fwd.x = m[2]; fwd.y = m[6]; fwd.z = m[10];
}

// ── ◀GUIDE▶ 【6.10】 ang_deg()  两个三维向量的夹角（度）
//
//   干什么   : 算两个向量夹角。用于判断「渲染用的朝向」与「估计朝向」差了多少。
//   参数    : a, b —— 两个三维向量
//   返回    : 夹角（度）∈[0,180]
//   公式    : cosθ = a·b / (|a||b|)，结果 clamp 到 [-1,1] 防止浮点越界让 acos 返回 NaN。
// ──────────────────────────────────────────────────────────────
static float ang_deg(const Vec3& a, const Vec3& b) {
    auto n = [](const Vec3& v) { return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z); };
    const float na = n(a), nb = n(b);
    if (na < 1e-9f || nb < 1e-9f) return 0.f;
    float cs = (a.x*b.x + a.y*b.y + a.z*b.z) / (na * nb);
    cs = cs < -1.f ? -1.f : (cs > 1.f ? 1.f : cs);
    return (float)(std::acos(cs) * 180.0 / 3.14159265358979);
}

// ══ 从「相机中心 + 朝向」构造 Camera（OpenCV 约定 x右 y下 z前）═══
// 这与 Camera::look_at 不同：look_at 用 OpenGL 习惯的 up 向量推基，
// 这里我们手上有的是**光轴方向**和**机体大致朝上方向**，直接正交化更稳。
//
// 输入：
//   C    相机中心（世界坐标）
//   fwd  光轴方向（世界坐标，指向场景）
//   up   参考"上"方向（世界坐标，用来定滚转；一般取 +Z 天向）
// 输出：world_to_cam 满足 X_cam = R·X_world + t
// ── ◀GUIDE▶ 【8.2】 cam_from_lookdir()  ★ 由「位置 + 光轴 + 上方向」造相机
//
//   干什么   : 【每帧渲染的第一步】给定无人机在哪(C)、镜头朝哪(fwd)、机头朝上大概哪(up)，
//     造出一个相机，告诉渲染器「从这个位置、这个角度去看模型」。
//   参数    : C 相机中心(世界)；fwd 光轴方向(指向场景)；up 参考上方向；proto 原型相机(内参)
//   返回    : Camera（world_to_cam 已填好）
//   怎么造   : zc = normalize(fwd)（相机看的方向）
//     xc = normalize(up × zc)（相机右手方向）
//     yc = zc × xc（相机下方，OpenCV 约定 y 朝下）
//     这三个基向量就是 world→cam 的 3 行；平移 t = -R·C
//   为什么不用 look_at: look_at 是 OpenGL 习惯（y 朝上）；本项目用 OpenCV 约定（y 朝下），
//     而且手上直接有光轴方向，正交化更稳。
// ──────────────────────────────────────────────────────────────
static Camera cam_from_lookdir(const Vec3& C, const Vec3& fwd, const Vec3& up,
                               const Camera& proto) {
    auto norm = [](Vec3 v) {
        float n = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
        if (n < 1e-12f) n = 1.f;
        return Vec3{v.x/n, v.y/n, v.z/n};
    };
    auto cross = [](const Vec3& a, const Vec3& b) {
        return Vec3{a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
    };
    const Vec3 zc = norm(fwd);                 // 相机 z 轴（前）
    Vec3 xc = norm(cross(up, zc));             // 相机 x 轴（右）= up × fwd
    const Vec3 yc = cross(zc, xc);             // 相机 y 轴（下）
    // world_to_cam 的 R 行 = 相机基在世界系下的分量
    Camera c = proto;
    c.world_to_cam = Mat4::identity();
    c.world_to_cam.m[0] = xc.x; c.world_to_cam.m[4] = xc.y; c.world_to_cam.m[8]  = xc.z;
    c.world_to_cam.m[1] = yc.x; c.world_to_cam.m[5] = yc.y; c.world_to_cam.m[9]  = yc.z;
    c.world_to_cam.m[2] = zc.x; c.world_to_cam.m[6] = zc.y; c.world_to_cam.m[10] = zc.z;
    // t = -R·C
    c.world_to_cam.m[12] = -(float)(xc.x*C.x + xc.y*C.y + xc.z*C.z);
    c.world_to_cam.m[13] = -(float)(yc.x*C.x + yc.y*C.y + yc.z*C.z);
    c.world_to_cam.m[14] = -(float)(zc.x*C.x + zc.y*C.y + zc.z*C.z);
    return c;
}

// ══ ★★ PiLoT 官方管线 ★★ ═════════════════════════════════════════
// pilot_c2w_from_lla_euler() 与 struct PilotC2W 已移到 geo_pose.h
// （供 calib_pilot.cpp 共用）。此处仅保留说明。
//
//   rot_pose_in_enu = R.from_euler('xyz', [roll,pitch,yaw], degrees=True)
//   rot_enu_in_ecef = get_rotation_enu_in_ecef(lon, lat)
//   R_c2w           = rot_enu_in_ecef @ rot_pose_in_enu
//   t_c2w           = WGS84_to_ECEF([lon,lat,alt])
//   c2w[:3,:3]     *= diag(1,-1,-1)         # OSG Y/Z 翻转
//
// 之后由调用者用 align 把 ENU 局部系平移到 PLY 局部系：
//   X_ply = X_enu_local + align

// ── 真值行 ────────────────────────────────────────────────────────
struct GtRow {
    std::string name;
    double lon, lat, alt;
    double a1, a2, a3;      // 欧拉角
    bool valid = false;
};

// ══ 欧拉角 → 相机光轴方向 ═════════════════════════════════════════
// 真值实测（3dgs_test.txt 300 帧）：
//   a1 = 0.0     恒定（横滚）
//   a2 = 35.6    恒定
//   a3 = 108.9 → 235.6（跨 ±180）连续变化
// 位置高度 143.09~143.21 m 波动 ±0.06 m → 定高飞行
//
// ★★ 关键修正（看真照片后确认）★★
//   E:\PiLoT-demo\...\3dgs_test\0_0.png 是**近乎垂直的正射俯视图**
//   （楼房顶面/街道路面清晰、无透视倾斜），说明真实光轴 ≈ 垂直向下。
//   所以 a2=35.6 **不是**相机的下倾角，更可能是相机相对 PTZ 云台的
//   安装倾角，或坐标定义里的另一个量。实际渲染必须用**近垂直俯视**。
//
//   因此本函数提供 pitch_override：当 pitch_override > 0 时，
//   用给定的俯仰角（如 88°）代替真值 a2，得到近正射视角。
//
// 坐标系约定：ENU（x=东, y=北, z=天）
//   yaw   a3: 方位角
//   pitch     : 下倾角（90=垂直向下看）
// ── ◀GUIDE▶ 【8.2】 euler_to_fwd()  方位角 + 俯仰角 → 光轴方向
//
//   干什么   : 把「机头朝哪(方位角 yaw)」和「镜头往下压多少(pitch)」换成三维单位向量。
//   参数    : yaw_deg 方位角(度，从北顺时针)；pitch_deg 下倾角(度，90=垂直向下看)
//   返回    : Vec3 单位向量（ENU 系：x=东 y=北 z=天）
//   ★关键   : 真实航片是**近垂直俯视**，所以项目里用 --pitch-override 88 覆盖真值给出的
//     35.6°，否则渲染出来的视角和实拍图对不上，匹配率会崩。
// ──────────────────────────────────────────────────────────────
static Vec3 euler_to_fwd(double yaw_deg, double pitch_deg) {
    const double yr = yaw_deg * M_PI / 180.0;
    const double pr = pitch_deg * M_PI / 180.0;
    const double ch = std::cos(pr);
    return Vec3{
        (float)(ch * std::sin(yr)),
        (float)(ch * std::cos(yr)),
        (float)(-std::sin(pr))
    };
}


// 读真值文件（TAB/空格分隔）
// ── ◀GUIDE▶ 【1.2】 load_gt()  读真值位姿文件
//
//   干什么   : 把 GT 文本读进内存，每行 = 一张照片的真值。
//   参数    : path —— 真值文件路径（--gt 指定）
//   返回    : vector<GtRow>，每个元素含 name/lon/lat/alt/a1/a2/a3
//   文件格式  : <png文件名> <lon> <lat> <alt> <a1> <a2> <a3>，空格分隔
//   注意    : sscanf 返回值必须是 7（恰好 7 个数），否则该行判为无效丢弃。
// ──────────────────────────────────────────────────────────────
static std::vector<GtRow> load_gt(const char* path) {
    std::vector<GtRow> out;
    FILE* f = fopen(path, "r");
    if (!f) return out;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        GtRow r;
        char nm[256];
        if (sscanf(line, "%255s %lf %lf %lf %lf %lf %lf",
                   nm, &r.lon, &r.lat, &r.alt,
                   &r.a1, &r.a2, &r.a3) == 7) {
            r.name = nm;
            r.valid = true;
            out.push_back(r);
        }
    }
    fclose(f);
    return out;
}

// 列目录（*.png / *.jpg），按名字自然排序（0_0, 1_0, 2_0 ... 而非字典序）
// ── ◀GUIDE▶ 【1.1】 list_images()  列出目录里的图片并排序
//
//   干什么   : 扫描照片目录，挑出 .png/.jpg，按文件名里的数字做「自然排序」。
//   参数    : dir —— 目录路径（--query 指定）
//   返回    : vector<string>，已排好序的文件名列表
//   为什么自然排序: 字典序会把 "10_0.png" 排在 "2_0.png" 前面（'1'<'2'），
//     轨迹就乱了。这里取 '_' 前的数字当排序键，0,1,2,...,10 才对。
// ──────────────────────────────────────────────────────────────
static std::vector<std::string> list_images(const char* dir) {
    std::vector<std::string> v;
    DIR* d = opendir(dir);
    if (!d) return v;
    struct dirent* e;
    while ((e = readdir(d))) {
        std::string n = e->d_name;
        if (n.size() < 4) continue;
        std::string ext = n.substr(n.size() - 4);
        for (auto& ch : ext) ch = (char)tolower(ch);
        if (ext == ".png" || ext == ".jpg" || n.size() > 4) {
            if (ext == ".png" || ext.rfind(".jpg") != std::string::npos) {
                v.push_back(n);
            }
        }
    }
    closedir(d);
    // 自然排序：按 '_' 前的数字
    auto key = [](const std::string& s) {
        size_t p = s.find('_');
        return atoi(s.substr(0, p == std::string::npos ? s.size() : p).c_str());
    };
    std::sort(v.begin(), v.end(), [&](const std::string& a, const std::string& b) {
        int ka = key(a), kb = key(b);
        if (ka != kb) return ka < kb;
        return a < b;
    });
    return v;
}

// ── 参数 ──────────────────────────────────────────────────────────
// ── ◀GUIDE▶ 【0.3】 Args  全部命令行参数（参数手册见代码注释或《代码阅读指南》）
//
//   干什么   : 一个「参数篮子」：所有可以通过命令行改的东西都放在这里。
//   怎么读   : 每个字段后面都有中文注释，含义/单位/默认值都在。
//   重点看   : ply / query / gt / out（必填四项），以及 feat_xfeat / pipeline / tilt_h（三条关键路线）。
// ──────────────────────────────────────────────────────────────
struct Args {
    const char* ply    = nullptr;
    const char* query  = nullptr;    // 真照片目录
    const char* out    = nullptr;    // 输出位姿文件
    const char* gt     = nullptr;    // 真值位姿文件（用于对比 + 初始化）
    const char* dump   = nullptr;    // 可视化输出目录
    bool  dump_depth = false;        // ★ 同时导出渲染深度（离线对比用）
    bool  dump_wsum  = false;        // ★ 同时导出深度权重和 Σαᵢ（判断深度可信度）
    bool  dump_color = false;        // ★ 同时导出**彩色**渲染图
    const char* pre_dir = nullptr;   // ★ 预解码图目录（512×288 灰度），加速读图
    int   W = 512, H = 288;
    float fx = 0, fy = 0, cx = 0, cy = 0;   // 内参（原始分辨率下）
    int   src_w = 3840, src_h = 2160;       // 照片原始分辨率
    int   iters = 3;
    int   start_frame = 0;
    int   max_frames = -1;           // -1 = 全部
    // ★ 首帧初值扰动（模拟机载 GPS/IMU 粗定位；用于消融与收敛域测试）
    float pert_pos = 0.0f;           // 位置扰动半径（m），圆盘内均匀采样
    float pert_ang = 0.0f;           // 姿态扰动（度），三轴各自 ±pert_ang
    bool  pert_first_only = false;   // ★ 扰动只在首帧施加（之后用上一帧估计）
    int   seed = 42;                 // 随机种子（固定 → 结果可复现）
    bool  init_gt_perturb = false;   // 首帧用「真值+扰动」初始化（默认随 pert>0 自动开）
    bool  gt_every_frame = false;    // 每帧都用「真值+扰动」初始化（逐帧独立模式）
    float min_alpha_radius = 0.5f;
    int   lod_keep = 0;              // LOD：只保留重要度最高的 N 个高斯（0=全部）
    // ★ 球谐阶数（0..3）。PLY 训练时是 degree 3（45 个 f_rest），
    //   但本机早期为省带宽/算力硬编码成 1（只 9 个 rest）。
    //   代价：视角相关颜色失真 → 渲染图与真图对不上 → 匹配退化。
    //   实测 SH=1 时匹配中位仅 117、reproj_rms 5135px、完整率 20%。
    int   sh_degree = 3;
    // 视觉初始化
    bool  init_from_gt = false;      // 用真值当初始位姿（最简单的验证路径）
    bool  init_from_prev = true;     // 用上一帧结果当初始位姿（真实工况）
    // ★ 用**外部文件里已算好的估计位姿**当每帧初始位姿（仅渲染，不迭代收敛）。
    //   用于成果展示：把主实验 run 的 poses.txt 喂进来，渲染出「估计位姿下的图」。
    //   连续飞行模式在本数据集会发散（组内云台跳变 ~90°），不能用于出图。
    std::string init_from_file;
    // 坐标
    int   geo_mode = 0;              // 0=ENU  1=PROJ
    double ref_lon = 0, ref_lat = 0, ref_alt = 0;  // 手动指定参考点
    bool  ref_auto = true;           // 自动用真值第一帧当参考点
    double lon0 = 114.0, k0 = 0.9996;
    double ox = 401448, oy = 3131258, oz = 0;
    bool  ply_y_down = false;        // PLY 是否 Y 轴朝下
    // ★ PLY 系 ↔ ENU 系 的平移对齐（由 calibrate_align 标定得到）
    //   X_ply = X_enu + (align_x, align_y, align_z)
    float align_x = 0, align_y = 0, align_z = 0;
    bool  yaw_invert = false;        // 若方位角方向反了
    float pitch_override = -1.f;     // >0 时用它代替真值 a2 当俯仰角
    float yaw_offset = 0.f;          // 给 a3 加一个偏置（度）
    // ★★ PiLoT 官方管线模式 ★★
    //   用 euler_angles_to_matrix_ECEF + OSG 翻转精确重建 c2w，
    //   绕开 euler_to_fwd/cam_from_lookdir 的近似。
    bool  pilot_mode = true;
    bool  verbose = false;
    // ★★ 帧间流水线模式（2026-10-09）★★
    //   每帧只做 1 次渲染（而非每轮 1 次），渲染结果滞后 1 帧被消费：
    //     帧开头 collect（收上一帧投递的渲染图，GPU 早已画完，等待≈0）
    //     → 匹配 + PnP → 帧尾用「本帧解算位姿」投递渲染（不等 GPU）
    //   渲染图位姿滞后 1 帧 = 帧间运动量（PiLoT 的 fixed 1-frame lag，
    //   由「用解算结果投递」把老接口的 2 帧滞后压到 1 帧）。
    //   收益：渲染次数 ×2→×1 + 回读等待(40~73ms)全部隐藏。
    bool  pipeline = false;
    // ★ 排序复用阈值（米/弧度量纲的视矩阵元素最大差）。
    //   默认 1e-4 ≈ 静止才命中。3DGS 是半透明软性混合 → 顺序小错
    //   视觉无差异，对 ORB 匹配更不敏感 → 放宽到帧间运动量级即可复用
    //   （逐项复核 valid + 可见集规模 ±2% 双保险，不会画出鬼影）。
    float reuse_eps = 0.0001f;
    int   orb_n = 4000;              // ORB 特征点数（渲染图与 query 共用）
    // ★ ORB 金字塔层数。默认 8 层对 512×288 明显过量：最顶层只剩
    //   4×6 像素，既提不出特征又白白付出整条金字塔的构建+检测成本。
    //   实测 512×288（fx=360）下 8 → 4 层，特征数几乎不变、提取时间减半。
    int   orb_levels = 8;
    // ★★★ 特征前端选择。orb = OpenCV CPU（原实现）；xfeat = RKNN NPU + 官方解码链。
    //   XFeat 已在离线对拍中与官方 verlab/XFeat torch 路径逐比特等价，
    //   并且精度/速度全面优于 ORB 与 SuperPoint（见 2026-10-12 完整实验报告）。
    bool  feat_xfeat = false;
    std::string xf_model  = "models/xfeat_288x512_int8.rknn";
    int   xf_topk    = 1500;      // 特征预算
    float xf_cossim  = 0.90f;     // MNN 余弦门限（0.90 优于官方默认 0.82）
    float xf_det_th  = 0.05f;     // NMS 阈值（官方默认）
    bool  xf_subpix  = true;      // 抛物线亚像素细化（有姿态先验时决定性）
    int   xf_threads = 4;         // 解码/匹配线程数（RK3588 用 4×A76）
    // ★★★ 零延迟位姿后处理（实时可用，无任何输出延迟）
    //   倾角约束：消除下视相机「水平平移 ↔ 倾斜」几何退化（err≈H·δtilt，corr=-0.999）
    //   因果卡尔曼：6 状态匀速模型，吃掉逐帧白噪声。不用 RTS（离线）。
    //   校正后的位姿会反馈给下一帧的渲染先验 → 闭环正反馈。
    // ★★ 2026-10-12 用户决定：**默认关闭倾角约束**。
    //   理由：它假设「光轴严格铅垂」，只在合成数据集（GT roll=pitch=0）成立；
    //   真实航拍有风扰/云台误差，用它会过拟合数据。
    //   实测：开启后全量 1067 帧跑不通（第 314 帧渲染回读不返回；
    //   非流水线模式成功率仅 ~10%）。代码保留供论文消融，默认 0 = 关。
    float tilt_h   = 0.0f;        // 有效高度（米）；0 = 关闭；>0 才启用倾角约束
    bool  prefetch = true;        // ★ 查询图预取：NPU 提取与渲染 CPU 段并行
    // ★ 目标定位探针：--probe u,v [--probe u,v ...] 每帧输出这些像素对应的世界 3D 地理坐标
    std::vector<std::pair<int,int>> probe_px;
    std::string probe_out;
    bool  use_kf   = false;       // 因果卡尔曼（默认关：闭环里会引入 0.4 帧滞后，见注释）
    float kf_sig_r = 2.5f;        // 测量噪声 σ（米）；离线扫参最优 2.5/2.0
    float kf_sig_a = 2.0f;        // 过程噪声 σ（米/帧²）
    // 渲染投影线程数（0 = 自动，吃满全部核）。RK3588 是 4×A76 + 4×A55，
    // 等分块时小核会拖长尾延迟，可试 4/6 看是否比 8 更快。
    int   project_threads = 0;
    // ★★ 可选：单应预筛（--homo-filter，2026-10-13 新增，默认关闭）
    //   动机：本项目是「近垂直俯视 + 近似平面场景」，此时平面单应
    //   x_r = H·x_q 是很强的几何约束 —— 先用 RANSAC 把几何不一致的
    //   错匹配剔掉，再交给 solvePnPRansac，理论上能让 PnP 的输入更干净。
    //   ⚠️ 预期收益有限：3D 点本就由同一帧渲染深度反投影得到、近似共面，
    //      所以单应内点集与 PnP 内点集高度重合（属于冗余筛选）。
    //      保留为对比实验开关，用 --homo-filter 启用。
    bool  homo_filter = false;
    float homo_thresh = 3.0f;     // 单应 RANSAC 重投影阈值（px）
    // ★★ PnP 求解器选择（--pnp-flags，2026-10-13 新增）
    //   本项目是「近垂直俯视 + 近似平面」的**几何退化**配置。
    //   在退化下，不同求解器的数值行为差别很大：
    //     iterative : Levenberg-Marquardt 迭代精化（OpenCV 默认）
    //     epnp      : 闭式解、无迭代，对平面更稳
    //     sqpnp     : 半定规划，抗噪与全局性更好
    //     ippe      : **专门针对平面**（Infinitesimal Plane-Based Pose
    //                 Estimation）—— 名字里就写着 Plane，与本场景最匹配
    //     p3p/ap3p  : 最小 3/4 点解法（RANSAC 最小集正好 4 点）
    std::string pnp_flags = "sqpnp";   // ★ 2026-10-13 默认 sqpnp（iterative 会发散，见下）
};

// ── ◀GUIDE▶ 【0.3】 usage()  打印命令行帮助
//
//   干什么   : 把全部可用参数打印出来。参数漏了或写错时会自动调用。
//   参数    : a0 —— argv[0]，程序名（打印用）
//   返回    : 无
//   怎么用   : 看不懂某个参数就来这里看，或看 --help 输出。
// ──────────────────────────────────────────────────────────────
static void usage(const char* a0) {
    fprintf(stderr,
"用法: %s <ply> --query DIR [选项]\n"
"\n"
"必选:\n"
"  <ply>                 3DGS 模型文件\n"
"  --query DIR           真照片目录（*.png / *.jpg）\n"
"  --pre-dir DIR         预解码图目录（512×288 灰度，同名文件）——\n"
"                        跳过 4K PNG 解码，实测省 ~185 ms/帧（23%% 端到端）\n"
"\n"
"输出:\n"
"  --out FILE            位姿输出文件（<name> <x> <y> <z> <a1> <a2> <a3>）\n"
"  --dump DIR            可视化输出目录（query 灰度 / 每轮 ref）\n"
"\n"
"真值:\n"
"  --gt FILE             真值位姿文件（用于误差统计 & 自动参考点 & 首帧初值）\n"
"  --init-from-gt        每帧都用真值作初始位姿\n"
"  --no-init-from-prev   关闭「用上一帧初始化」\n"
"  --init-from-file F    ★ 每帧用 F 里已算好的估计位姿（仅渲染，不出位姿）\n"
"                        F 格式同 --out 输出：name lon lat alt roll pitch yaw ...\n"
"\n"
"初值扰动（模拟机载 GPS/IMU 粗定位；消融/收敛域测试）:\n"
"  --pert-pos M          首帧位置扰动半径（米，圆盘内均匀采样）\n"
"  --pert-ang D          首帧姿态扰动（度，三轴各 ±D）\n"
"  --pert-first-only    ★ 扰动仅作用于首帧；第2帧起用上一帧估计初始化\n"
"  --seed N              随机种子（默认 42，固定以便复现）\n"
"\n"
"模式说明:\n"
"  默认（不加 --init-from-gt）：首帧用「真值+扰动」初始化，之后逐帧\n"
"  用上一帧结果递推 —— 即「模拟无人机连续飞行」。\n"
"  失败帧不更新上一帧状态，后续帧可自动恢复。\n"
"\n"
"内参（默认取 feicuiwan.yaml: 2700,2700,1915.7,1075.1 @3840x2160）:\n"
"  --fx --fy --cx --cy   相机内参（原始分辨率下）\n"
"  --src-w --src-h       照片原始分辨率（默认 3840 2160）\n"
"\n"
"渲染:\n"
"  --w N --h N           渲染分辨率（默认 512x288）\n"
"  --iters N             每帧 render-and-compare 迭代轮数（默认 3）\n"
"  --pipeline            ★ 帧间流水线：每帧 1 次渲染 + PBO 异步回读，\n"
"                        渲染图滞后 1 帧消费（实测渲染耗时减半、等待归零）\n"
"  --min-alpha-radius R  重要性剔除阈值（默认 0.5）\n"
"  --sh-degree N         球谐阶数 0..3（默认 3）。PLY 是 degree 3（45 个 rest），\n"
"                        降到 1 会丢视角相关颜色，渲染失真 -> 匹配退化\n"
"  --homo-filter         ★ 单应预筛：PnP 前先用平面单应 RANSAC 剔除几何不一致的错匹配\n"
"  --homo-thresh PX      单应 RANSAC 重投影阈值（默认 3.0 px）\n"
"  --pnp-flags NAME      ★ PnP 求解器：iterative(默认) | epnp | sqpnp | ippe | p3p | ap3p\n"
"                         （本项目是平面退化场景，ippe 是平面专用解法，值得对比）\n"
"\n"
"特征前端（默认 orb）:\n"
"  --feat xfeat|orb      xfeat = RKNN NPU + 官方解码链（推荐）；orb = OpenCV CPU\n"
"  --xf-model PATH       XFeat RKNN 模型（默认 models/xfeat_288x512_int8.rknn）\n"
"  --xf-topk N           特征预算（默认 1500；实测加到 4096 无收益）\n"
"  --xf-cossim F         MNN 余弦门限（默认 0.90；实测优于官方默认 0.82）\n"
"  --xf-det-th F         NMS 阈值（默认 0.05，官方默认）\n"
"  --xf-no-subpix        关闭抛物线亚像素细化（默认开启；有姿态先验时决定性）\n"
"  --xf-threads N        解码/匹配线程数（默认 4）\n"
"\n"
"零延迟位姿后处理（实时；校正结果反馈给下一帧渲染先验）:\n"
"  --tilt-h M            倾角约束的有效高度（米）；**默认 0 = 关闭**\n"
"                         （仅在合成数据等「光轴严格铅垂」假设成立时才有意义）\n"
"  --no-kf               关闭因果卡尔曼\n"
"  --kf-sig-r F          测量噪声 σ（米，默认 1.5）\n"
"  --kf-sig-a F          过程噪声 σ（米/帧²，默认 0.5）\n"
"  --lod N               ★ LOD：只保留静态重要度最高的 N 个高斯（0=全部）\n"
"  --dump-depth          ★ 同时导出渲染深度 .bin（离线特征对比用）\n"
"  --dump-wsum           ★ 同时导出深度权重和 Σαᵢ（用 --dump-depth 时生效，\n"
"                         Σαᵢ 小说明该像素深度不可信）\n"
"  --dump-color          ★ 同时导出**彩色**渲染图（灰度只用于匹配）\n"
"\n"
"范围:\n"
"  --start-frame N       从第几帧开始（默认 0）\n"
"  --max-frames N        最多处理帧数（默认全部）\n"
"\n"
"坐标:\n"
"  --ref LON LAT ALT     手动指定 ENU 参考点（默认自动取真值第一帧）\n"
"  --geo-proj            用高斯投影模式（默认 ENU）\n"
"  --lon0 DEG --k0 F     投影中央经线/尺度因子（默认 114 / 0.9996）\n"
"  --offset X Y Z        cgcs_offset（默认 401448 3131258 0）\n"
"  --ply-y-down          PLY 的 Y 轴朝下（需要翻转到 ENU 天向）\n"
"  -v                    详细输出\n",
            a0);
}

// ── ◀GUIDE▶ 【0.4】 parse_args()  解析命令行参数
//
//   干什么   : 把命令行上的一串 --xxx yyy 读进 Args 结构体，供后面使用。
//   参数    : argc/argv —— main 传来的参数个数与字符串数组
//   返回    : Args（含所有参数的默认值或用户指定值）
//   怎么读   : 就是一个大 for 循环 + 一堆 if/else if，逐个匹配字符串，
//     每个分支都是「取下一个字符串，转成数字存起来」。
//     看不懂 C++ 语法可以先跳过这个函数，不影响理解主流程。
// ──────────────────────────────────────────────────────────────
static Args parse_args(int argc, char** argv) {
    Args A;
    for (int i = 1; i < argc; ++i) {
        auto eq = [&](const char* s) { return !strcmp(argv[i], s); };
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if (eq("--query")) { A.query = next(); }
        else if (eq("--pre-dir")) { A.pre_dir = next(); }
        else if (eq("--out"))   { A.out   = next(); }
        else if (eq("--gt"))    { A.gt    = next(); }
        else if (eq("--dump"))  { A.dump  = next(); }
        else if (eq("--w"))     { A.W = atoi(next()); }
        else if (eq("--h"))     { A.H = atoi(next()); }
        else if (eq("--fx"))    { A.fx = (float)atof(next()); }
        else if (eq("--fy"))    { A.fy = (float)atof(next()); }
        else if (eq("--cx"))    { A.cx = (float)atof(next()); }
        else if (eq("--cy"))    { A.cy = (float)atof(next()); }
        else if (eq("--src-w")) { A.src_w = atoi(next()); }
        else if (eq("--src-h")) { A.src_h = atoi(next()); }
        else if (eq("--iters")) { A.iters = atoi(next()); }
        else if (eq("--pipeline")) { A.pipeline = true; }
        else if (eq("--dump-depth")) { A.dump_depth = true; }
        else if (eq("--dump-wsum"))  { A.dump_wsum  = true; }
        else if (eq("--dump-color")) { A.dump_color = true; }
        else if (eq("--reuse-eps")) { A.reuse_eps = (float)atof(next()); }
        else if (eq("--orb-n")) { A.orb_n = atoi(next()); }
        else if (eq("--orb-levels")) { A.orb_levels = atoi(next()); }
        else if (eq("--feat")) {
            const char* v = next();
            A.feat_xfeat = (strcmp(v, "xfeat") == 0);
            if (!A.feat_xfeat && strcmp(v, "orb") != 0) {
                fprintf(stderr, "未知前端: %s（可选 orb / xfeat）\n", v); exit(1);
            }
        }
        else if (eq("--xf-model"))  { A.xf_model = next(); }
        else if (eq("--xf-topk"))   { A.xf_topk = atoi(next()); }
        else if (eq("--xf-cossim")) { A.xf_cossim = (float)atof(next()); }
        else if (eq("--xf-det-th")) { A.xf_det_th = (float)atof(next()); }
        else if (eq("--xf-no-subpix")) { A.xf_subpix = false; }
        else if (eq("--xf-threads")){ A.xf_threads = atoi(next()); }
        else if (eq("--tilt-h"))  { A.tilt_h = (float)atof(next()); }
        else if (eq("--no-tilt")) { A.tilt_h = 0.f; }
        else if (eq("--no-kf"))   { A.use_kf = false; }
        else if (eq("--no-prefetch")) { A.prefetch = false; }
        else if (eq("--probe")) {
            const char* v = next(); int u = 0, vv = 0;
            if (sscanf(v, "%d,%d", &u, &vv) == 2) A.probe_px.emplace_back(u, vv);
            else { fprintf(stderr, "❌ --probe 格式应为 u,v（如 256,144）\n"); exit(1); }
        }
        else if (eq("--probe-out")) { A.probe_out = next(); }
        else if (eq("--kf-sig-r")){ A.kf_sig_r = (float)atof(next()); }
        else if (eq("--kf-sig-a")){ A.kf_sig_a = (float)atof(next()); }
        else if (eq("--threads")) { A.project_threads = atoi(next()); }
        else if (eq("--homo-filter")) { A.homo_filter = true; }
        else if (eq("--homo-thresh")) { A.homo_thresh = (float)atof(next()); }
        else if (eq("--pnp-flags")) {
            const char* v = next();
            if (v) A.pnp_flags = v;
        }
        else if (eq("--start-frame")) { A.start_frame = atoi(next()); }
        else if (eq("--max-frames"))  { A.max_frames = atoi(next()); }
        else if (eq("--pert-pos")) { A.pert_pos = (float)atof(next()); }
        else if (eq("--pert-ang")) { A.pert_ang = (float)atof(next()); }
        else if (eq("--pert-first-only")) { A.pert_first_only = true; }
        else if (eq("--seed"))    { A.seed = atoi(next()); }
        else if (eq("--gt-every-frame")) { A.gt_every_frame = true; }
        else if (eq("--init-gt-perturb")) { A.init_gt_perturb = true; }
        else if (eq("--min-alpha-radius")) { A.min_alpha_radius = (float)atof(next()); }
        else if (eq("--sh-degree")) { A.sh_degree = atoi(next()); }
        else if (eq("--lod")) { A.lod_keep = atoi(next()); }
        else if (eq("--init-from-gt")) { A.init_from_gt = true; }
        else if (eq("--init-from-file")) { A.init_from_file = next(); }
        else if (eq("--no-init-from-prev")) { A.init_from_prev = false; }
        else if (eq("--ref")) {
            A.ref_lon = atof(next()); A.ref_lat = atof(next()); A.ref_alt = atof(next());
            A.ref_auto = false;
        }
        else if (eq("--geo-proj")) { A.geo_mode = 1; }
        else if (eq("--lon0")) { A.lon0 = atof(next()); }
        else if (eq("--k0"))   { A.k0   = atof(next()); }
        else if (eq("--offset")) {
            A.ox = atof(next()); A.oy = atof(next()); A.oz = atof(next());
        }
        else if (eq("--ply-y-down")) { A.ply_y_down = true; }
        else if (eq("--align")) {
            A.align_x = (float)atof(next());
            A.align_y = (float)atof(next());
            A.align_z = (float)atof(next());
        }
        else if (eq("--yaw-invert")) { A.yaw_invert = true; }
        else if (eq("--pitch-override")) { A.pitch_override = (float)atof(next()); }
        else if (eq("--yaw-offset")) { A.yaw_offset = (float)atof(next()); }
        else if (eq("--pilot")) { A.pilot_mode = true; }
        else if (eq("--no-pilot")) { A.pilot_mode = false; }
        else if (eq("-v")) { A.verbose = true; }
        else if (argv[i][0] != '-') { A.ply = argv[i]; }
    }
    return A;
}

// ── ◀GUIDE▶ 【★ STEP 1 ★】 main()  程序入口 —— 阅读从这里开始
//
//   干什么   : 整个程序的主流程：解析参数 → 加载数据 → 逐帧循环 → 汇总指标。
//   参数    : argc/argv —— 命令行参数（交给 parse_args 处理）
//   返回    : 0 正常；1 运行错误；2 参数错误
//   顺序    : 函数内 1150 行，**不要从头硬读**。先读下面这张目录表（第 682 行起），
//     再按 8.1 / 8.2 / 8.3 / 8.4 逐段跳读。
//   配合阅读  : 《代码阅读指南》STEP 1 有逐段讲解。
// ──────────────────────────────────────────────────────────────
int main(int argc, char** argv) {

    // ══════════════════════════════════════════════════════════════════
    //  main() 目录 —— 本函数约 1080 行，**不要从头硬读**，按下面目录跳读
    //  （行号会随改代码漂移，以「── 段落名 ──」的注释为准确锚点）
    // ──────────────────────────────────────────────────────────────────
    //   ── 初始化（一次性，只跑一遍）──────────────────────────────
    //   929   1. 列照片           扫描 --query 目录 → 得到 1067 个文件名
    //   939   2. 读真值           load_gt() → 每张照片的 lon/lat/alt/roll/pitch/yaw
    //   953   3. 建坐标转换器     ★地理配准：定 ENU 参考点 + align 平移（见 geo_pose.h）
    //   988   4. 加载模型         PLY 文件 → 300 万个高斯（约 400 ms）
    //  1013   5. 建渲染器         Renderer::init()
    //                             ⚠️ EGL 上下文在这里创建，从此绑定本线程（见 offscreen.cpp）
    //  1028   6. 内参             3840×2160 的内参按渲染分辨率缩放
    //  1043   7. 输出文件         位姿 / 探针 两个输出文件
    //   ── 主循环（每帧重复，共 1067 遍）──────────────────────────
    //  1092   8. 逐帧处理
    //  1159      └ 查询图预取线程   提前把下一张实拍图提完特征（--prefetch 控制）
    //  1196         8.1 读照片      用预取结果
    //  1251         8.2 初始位姿    首帧 = 真值 + 扰动；之后 = 上一帧解算结果
    //  1367         8.3 render-and-compare  ★★★ 全文核心
    //  1378             if (A.pipeline)  帧间流水线模式：
    //                     ① 交付/收取上一帧渲染图（GPU 早已画完，等待≈0）
    //                     ② 特征匹配（渲染图 ↔ 实拍图）
    //                        ⚠️ 反投影必须用 cam_rb（渲染时用的位姿）
    //                     ③ solvePnPRansac 解出 6DoF 位姿
    //                     ④ 帧尾用「本帧解算位姿」再投递一次渲染，供下一帧用
    //                        → 渲染图只滞后 1 帧（PiLoT 的 fixed 1-frame lag）
    //            else                非流水线模式（每帧现场渲染 + 迭代 iters 轮）
    //  1796         8.4 误差         与真值比，累积统计量
    //  1860         8.5 写输出       6DoF 位姿 + 匹配置信度
    //  1903         8.6 目标定位探针  像素 → 全局 3D 地理坐标（--probe）
    //   ── 收尾 ─────────────────────────────────────────────────
    //  1953   9. 汇总指标卡   mT / mR / Recall@(k m,k°) / Completeness / FPS
    //
    //  ▸ 更细的逐段讲解 + C++ 语法解释，见《代码阅读指南.html》的 STEP 1。
    // ══════════════════════════════════════════════════════════════════
    Args A = parse_args(argc, argv);
    if (!A.ply || !A.query) { usage(argv[0]); return 2; }
    g_rng.seed((unsigned)A.seed);        // 扰动可复现

    // ── 默认内参（feicuiwan.yaml 实测值）──────────────────────────
    if (A.fx == 0) A.fx = 2700.0f;
    if (A.fy == 0) A.fy = 2700.0f;
    if (A.cx == 0) A.cx = 1915.7f;
    if (A.cy == 0) A.cy = 1075.1f;

    // ── 1. 列照片 ─────────────────────────────────────────────────
    std::vector<std::string> imgs = list_images(A.query);
    if (imgs.empty()) {
        fprintf(stderr, "❌ 目录里没有图片: %s\n", A.query);
        return 1;
    }
    fprintf(stdout, "════════ 无人机 3DGS 视觉定位 ════════\n");
    fprintf(stdout, "照片目录  : %s\n", A.query);
    fprintf(stdout, "照片数量  : %zu\n", imgs.size());

    // ── 2. 读真值 ─────────────────────────────────────────────────
    std::vector<GtRow> gt = A.gt ? load_gt(A.gt) : std::vector<GtRow>();
    // ★ 载入「外部估计位姿」文件（--init-from-file）。格式与 --out 相同：
    //   name lon lat alt roll pitch yaw qw qx qy qz nmatch ninl ...  [   # ...]
    //   只取前 7 列（name + 经纬高 + 三欧拉角）构造 GtRow。
    std::vector<GtRow> estfile;
    if (!A.init_from_file.empty()) {
        FILE* ef = fopen(A.init_from_file.c_str(), "r");
        if (!ef) {
            fprintf(stderr, "❌ 打不开 --init-from-file: %s\n", A.init_from_file.c_str());
            return 1;
        }
        char el[2048];
        while (fgets(el, sizeof(el), ef)) {
            if (el[0] == '#') continue;
            GtRow r; char nm[256];
            if (sscanf(el, "%255s %lf %lf %lf %lf %lf %lf",
                       nm, &r.lon, &r.lat, &r.alt,
                       &r.a1, &r.a2, &r.a3) == 7) {
                r.name = nm; r.valid = true;
                estfile.push_back(r);
            }
        }
        fclose(ef);
        fprintf(stdout, "[init] 载入估计位姿 %zu 条 <- %s\n",
                estfile.size(), A.init_from_file.c_str());
    }
    // name → 行 的索引（两文件都按名字匹配）
    auto find_row = [](const std::vector<GtRow>& v, const std::string& nm) -> const GtRow* {
        for (const auto& r : v) if (r.name == nm) return &r;
        return nullptr;
    };
    const bool use_estfile = !estfile.empty();

    if (A.gt) {
        fprintf(stdout, "真值文件  : %s  (%zu 行)\n", A.gt, gt.size());
    }

    // 建立「文件名 → 真值」索引
    std::vector<const GtRow*> gt_by_name;
    for (const auto& r : gt) gt_by_name.push_back(&r);
    auto find_gt = [&](const std::string& name) -> const GtRow* {
        for (const auto* r : gt_by_name) if (r->name == name) return r;
        return nullptr;
    };

    // ── 3. 建坐标转换器 ───────────────────────────────────────────
    geopp::GeoRef geo;
    if (A.geo_mode == 0) {
        geo.mode = geopp::GeoRef::ENU;
    } else {
        geo.mode = geopp::GeoRef::PROJ;
    }
    geo.lon0 = A.lon0; geo.k0 = A.k0;
    geo.ox = A.ox; geo.oy = A.oy; geo.oz = A.oz;

    if (A.ref_auto) {
        // 用真值第一帧当参考点
        const GtRow* r0 = find_gt(imgs[0]);
        if (!r0 && !gt.empty()) r0 = &gt[0];
        if (!r0) {
            fprintf(stderr, "❌ 需要 --gt 或 --ref 来建立局部坐标系\n");
            return 1;
        }
        A.ref_lon = r0->lon; A.ref_lat = r0->lat; A.ref_alt = r0->alt;
    }
    geo.ref_lon = A.ref_lon; geo.ref_lat = A.ref_lat; geo.ref_alt = A.ref_alt;
    geo.prepare_enu();

    // ⚠️ PLY 局部系 → ENU 系 的刚体变换。
    //    因为 PLY 的原点是训练时自选的，我们无法直接知道。
    //    但**PLY 的原点与 ENU 原点之间的平移/旋转是固定的**，
    //    而 render-and-compare 解出来的是「相机在 PLY 系里的位姿」。
    //    要跟 ENU 系真值比误差，必须做这个变换。
    //
    //    ★ 实用解法：先用真值第一帧的 ENU 坐标，减去渲染时用的
    //      「PLY 系初始位姿」，得到偏移量 T。之后所有帧都加 T。
    //      这样误差就是正确的相对量。
    fprintf(stdout, "坐标系    : %s  参考点(%.6f, %.6f, %.2f)\n",
            A.geo_mode == 0 ? "ENU" : "PROJ", A.ref_lon, A.ref_lat, A.ref_alt);

    // ── 4. 加载模型 ───────────────────────────────────────────────
    std::vector<Splat>    splats;
    std::vector<ShCoeffs> shcs;
    if (!load_ply_splats(A.ply, splats, shcs, A.sh_degree, false)) {
        fprintf(stderr, "PLY 加载失败: %s\n", A.ply);
        return 1;
    }
    float mn[3] = { 1e30f, 1e30f, 1e30f }, mx[3] = { -1e30f, -1e30f, -1e30f };
    for (const auto& s : splats) {
        float p[3] = { s.x, s.y, s.z };
        for (int k = 0; k < 3; ++k) {
            if (p[k] < mn[k]) mn[k] = p[k];
            if (p[k] > mx[k]) mx[k] = p[k];
        }
    }
    const Vec3 center{ (mn[0]+mx[0])*0.5f, (mn[1]+mx[1])*0.5f, (mn[2]+mx[2])*0.5f };
    const float radius = 0.5f * std::sqrt((mx[0]-mn[0])*(mx[0]-mn[0]) +
                                          (mx[1]-mn[1])*(mx[1]-mn[1]) +
                                          (mx[2]-mn[2])*(mx[2]-mn[2]));
    fprintf(stdout, "高斯数    : %zu\n", splats.size());
    fprintf(stdout, "包围盒    : min(%.2f %.2f %.2f) max(%.2f %.2f %.2f)\n",
            mn[0],mn[1],mn[2], mx[0],mx[1],mx[2]);
    fprintf(stdout, "中心/半径 : (%.2f %.2f %.2f)  %.2f m\n",
            center.x, center.y, center.z, radius);

    // ── 5. 建渲染器 ───────────────────────────────────────────────
    RenderConfig cfg;
    cfg.width  = A.W;
    cfg.height = A.H;
    cfg.depth  = true;                 // ★ 必须：ED 米制深度是 3D 点的来源
    cfg.min_alpha_radius = A.min_alpha_radius;
    cfg.lod_keep = A.lod_keep;         // ★ LOD 降采样（0 = 全量）
    cfg.sh_degree = A.sh_degree;       // ★ SH 阶数（默认 3，与 PLY 一致）
    cfg.project_threads = A.project_threads;
    cfg.async_readback = A.pipeline;   // ★ 流水线模式启用 PBO 异步回读
    cfg.reuse_eps = A.reuse_eps;       // ★ 排序复用阈值（3DGS 半透明 → 可放宽）
    // ★ 2026-10-09：把 -v 透传给渲染器，才能看到分块剔除/投影的诊断行
    cfg.verbose = A.verbose;
    Renderer r;
    if (!r.init(A.ply, cfg)) { fprintf(stderr, "渲染器初始化失败\n"); return 1; }

    // ── 6. 内参（按渲染分辨率缩放）────────────────────────────────
    const float sx = (float)A.W / (float)A.src_w;
    const float sy = (float)A.H / (float)A.src_h;
    Camera proto;
    proto.width = A.W; proto.height = A.H;
    proto.fx = A.fx * sx; proto.fy = A.fy * sy;
    proto.cx = A.cx * sx; proto.cy = A.cy * sy;
    proto.znear = 0.05f; proto.zfar = radius * 20.f;
    fprintf(stdout, "渲染      : %dx%d\n", A.W, A.H);
    fprintf(stdout, "内参(渲染): fx=%.2f fy=%.2f cx=%.2f cy=%.2f\n",
            proto.fx, proto.fy, proto.cx, proto.cy);
    fprintf(stdout, "视场角    : hFov=%.2f° vFov=%.2f°\n",
            2.f * std::atan(0.5f * A.W / proto.fx) * 180.f / 3.14159265f,
            2.f * std::atan(0.5f * A.H / proto.fy) * 180.f / 3.14159265f);

    // ── 7. 输出文件 ───────────────────────────────────────────────
    FILE* fout = A.out ? fopen(A.out, "w") : nullptr;
    if (fout) {
        fprintf(fout, "# 无人机 6DoF 自位姿（Ego Pose）逐帧输出\n");
        fprintf(fout, "# 列: name lon lat alt roll pitch yaw qw qx qy qz nmatch ninl inl_ratio cossim reproj_rms conf\n");
        fprintf(fout, "# 坐标: WGS84 经纬高 (deg,deg,m) | 姿态: ZYX 欧拉角 (deg) + 四元数 (w,x,y,z, 正半球)\n");
        fprintf(fout, "# 置信度: inl_ratio=内点/匹配, cossim=接受配对平均余弦相似度, reproj_rms=内点重投影RMS(px)\n");
        fprintf(fout, "#         conf = inl_ratio * exp(-reproj_rms/2) * clamp(cossim,0,1)   [0..1]\n");
    }
    FILE* fprobe = A.probe_out.empty() ? nullptr : fopen(A.probe_out.c_str(), "w");
    if (fprobe) {
        fprintf(fprobe, "# 目标定位：图像像素 → 全局 3D 地理坐标（用本帧渲染深度反投影）\n");
        fprintf(fprobe, "# 列: name u v lon lat alt   # ENU e n u  depth\n");
    }
    if (A.out && !fout) {
        fprintf(stderr, "❌ 无法写输出文件: %s\n", A.out);
        return 1;
    }

    // ★ PnP 求解器（--pnp-flags）：在主循环外解析一次，避免每帧做字符串比较
    // ★★ 2026-10-13：默认由 iterative 改为 sqpnp ★★
    //   实测 152 帧 BUAA 数据：iterative 在前 3 帧发散到 148~200 m
    //   （reproj_rms 达 5135 px），而 sqpnp 同样 3 帧全部 < 1.6 m。
    //   iterative(=LM) 需要好初值、易陷局部极小；SQPnP 是全局最优闭式解、
    //   无需初值，代价仅多几毫秒/帧。此处直接用 sqpnp。
    int pnp_flag = cv::SOLVEPNP_SQPNP;
    if      (A.pnp_flags == "epnp")  pnp_flag = cv::SOLVEPNP_EPNP;
    else if (A.pnp_flags == "sqpnp") pnp_flag = cv::SOLVEPNP_SQPNP;
    else if (A.pnp_flags == "ippe")  pnp_flag = cv::SOLVEPNP_IPPE;
    else if (A.pnp_flags == "p3p")   pnp_flag = cv::SOLVEPNP_P3P;
    else if (A.pnp_flags == "ap3p")  pnp_flag = cv::SOLVEPNP_AP3P;
    else if (A.pnp_flags == "iterative") pnp_flag = cv::SOLVEPNP_ITERATIVE;
    else if (A.pnp_flags != "sqpnp")
        fprintf(stderr, "[warn] 未知 --pnp-flags '%s'，回退 sqpnp\n", A.pnp_flags.c_str());
    fprintf(stdout, "PnP 求解器  : %s\n", A.pnp_flags.c_str());

    // ★ levels 可调：512×288 下 8 层金字塔纯浪费（见 Args::orb_levels）
    cv::Ptr<cv::ORB> orb = cv::ORB::create(A.orb_n, 1.2f, A.orb_levels);
    cv::BFMatcher matcher(cv::NORM_HAMMING);

    // ── XFeat 前端（--feat xfeat 时启用）──────────────────────────
    //   NPU 推理 + 官方解码链 + 抛物线亚像素 + MNN 匹配。
    //   离线已与官方 torch 路径逐比特等价。
    XFeatFrontend xfeat;
    XFeatKps xf_q, xf_r;
    // 零延迟位姿后处理状态（倾角约束是纯函数，无需状态；卡尔曼需要）
    PoseKF pkf;
    pkf.sig_r = A.kf_sig_r; pkf.sig_a = A.kf_sig_a; pkf.dt = 1.0;
    if (A.tilt_h > 0.f || A.use_kf)
        fprintf(stdout, "位姿后处理: tilt_h=%.0f m  kf=%d (sig_r=%.2f sig_a=%.2f)\n",
                A.tilt_h, (int)A.use_kf, A.kf_sig_r, A.kf_sig_a);
    if (A.feat_xfeat) {
        XFeatParams xp;
        xp.top_k = A.xf_topk; xp.det_th = A.xf_det_th;
        xp.min_cossim = A.xf_cossim; xp.subpix = A.xf_subpix;
        xp.match_thr = A.xf_threads;
        if (!xfeat.init(A.xf_model, xp, A.W, A.H)) {
            fprintf(stderr, "❌ XFeat 前端初始化失败: %s\n", A.xf_model.c_str());
            return 1;
        }
        fprintf(stdout, "前端: XFeat (NPU)  top_k=%d cossim=%.2f det_th=%.2f subpix=%d threads=%d\n",
                xp.top_k, xp.min_cossim, xp.det_th, (int)xp.subpix, xp.match_thr);
    } else {
        fprintf(stdout, "前端: ORB (CPU)  n=%d levels=%d\n", A.orb_n, A.orb_levels);
    }

    // ── 8. 逐帧处理 ───────────────────────────────────────────────
    const int n_total = (int)imgs.size();
    int f_end = (A.max_frames < 0) ? n_total : std::min(n_total, A.start_frame + A.max_frames);

    // 上一帧结果（PLY 系），用于初始化下一帧
    bool  have_prev = false;
    Vec3  prev_C{}, prev_fwd{};

    // 误差统计
    std::vector<double> epos_all, eang_all;
    std::vector<double> eroll_all, epitch_all, eyaw_all, erot_all;
    std::vector<int>    nmatch_all;
    int n_success = 0, n_fail = 0, n_processed = 0;
    // ★ 单应预筛统计（仅 --homo-filter 时累加）
    long n_homo_kept = 0, n_homo_total = 0;
    const auto t_wall0 = Clock::now();       // 端到端墙钟（算 FPS 用）

    double t_render = 0, t_match = 0, t_pnp = 0, t_io = 0;
    double t_npu = 0, t_dec = 0, t_mnn = 0;   // XFeat: NPU / 解码 / MNN 匹配
    double t_npu_q = 0, t_dec_q = 0;          // 其中「查询图」部分
    // ★ 渲染内部分解诊断（尾部打印）
    double s_proj = 0, s_sort = 0, s_asm = 0, s_up = 0, s_draw = 0, s_rb = 0;
    int    n_reused = 0, n_render_calls = 0;

    // ══ ★★ 查询图预取：让「读图 + 查询图 XFeat 提取（NPU）」与
    //    上一帧帧尾的 3DGS 渲染（CPU 密集，~83 ms）并行 ══
    //
    //  ⚠️ 为什么不是「把渲染搬到线程里」：
    //     EGL 上下文绑定在**创建它的线程**上，从新线程直接调 GL 会死锁
    //     （实测：进程只剩 5 个线程、全部休眠、CPU 39%）。
    //     所以方向反过来 —— 把 NPU 工作搬出去，它与渲染的 CPU/GPU 工作天然可重叠。
    //
    //  时序：
    //    帧 k   开头：join 预取 → 用预取好的 qgray / 查询特征
    //    帧 k   帧尾：**先起预取线程（第 k+1 帧的图像 + 特征）**，再同步渲染
    //                → 预取的 NPU 推理与渲染的 CPU 段并行
    //    帧 k+1 开头：join（此时渲染早已结束，几乎不用等）
    //
    //  线程安全：预取线程只做 imread / resize / xfeat.extract；
    //            主线程在 join 之前不碰这两个东西。XFeatFrontend 的复用缓冲
    //            因此不会被并发访问。
    bool      use_prefetch = A.prefetch && A.feat_xfeat && A.pipeline;
    std::thread pf_th;
    bool        pf_inflight = false, pf_ok = false;
    cv::Mat     pf_gray;
    XFeatKps    pf_kps;
    double      t_pf_wait = 0;
    auto pf_path = [&](const std::string& nm) -> std::string {
        return A.pre_dir ? (std::string(A.pre_dir) + "/" + nm)
                         : (std::string(A.query) + "/" + nm);
    };
    auto join_prefetch = [&]() {
        if (!pf_inflight) return;
        const auto t0 = Clock::now();
        pf_th.join();
        t_pf_wait += ms_since(t0);
        pf_inflight = false;
    };
    auto start_prefetch = [&](int i) {
        const std::string nm = imgs[i];
        const std::string pp = pf_path(nm);
        pf_ok = false; pf_inflight = true;
        pf_th = std::thread([&, pp]() {
            cv::Mat g = cv::imread(pp, cv::IMREAD_GRAYSCALE);
            if (g.empty()) return;
            if (g.cols != A.W || g.rows != A.H)
                cv::resize(g, g, cv::Size(A.W, A.H), 0, 0, cv::INTER_AREA);
            pf_gray = g;
            // ★ 解码降到单线程：渲染的投影正在吃满 8 核，预取再开 4 线程会争核
            //   （实测渲染 97.6 → 118.6 ms）。预取有 ~100 ms 预算，单线程解码(~30 ms)绰绰有余。
            pf_ok = xfeat.extract(g.data, pf_kps);
        });
    };
    std::thread render_th;
    bool        render_inflight = false;
    Camera      cam_render = proto;
    double      t_async_wait = 0;     // 主线程等渲染线程的净时间
    double      t_async_work = 0;     // 渲染线程实际工作时间
    auto join_render = [&]() {
        if (!render_inflight) return;
        const auto t0 = Clock::now();
        render_th.join();
        t_async_wait += ms_since(t0);
        const auto& st = r.stats();
        s_proj += st.ms_project; s_sort += st.ms_sort; s_asm += st.ms_assemble;
        s_up   += st.ms_upload;  s_draw += st.ms_draw;  s_rb  += st.ms_readback;
        n_reused += st.sort_reused ? 1 : 0;
        ++n_render_calls;
        render_inflight = false;
    };

    if (use_prefetch && A.start_frame < f_end) start_prefetch(A.start_frame);

    fprintf(stdout, "\n%-22s %-6s %-7s %-6s %-11s %-9s %-8s %s\n",
            "帧", "匹配", "内点", "轮", "位置误差(m)", "朝向误差(°)", "耗时(ms)", "状态");
    fprintf(stdout, "────────────────────────────────────────────────────────────────────────────\n");

    for (int idx = A.start_frame; idx < f_end; ++idx) {
        const std::string& name = imgs[idx];
        const std::string path = std::string(A.query) + "/" + name;
        const GtRow* g = find_gt(name);
        ++n_processed;

        // ★ 先收上一帧帧尾起的预取线程（渲染期间它已跑完，等待 ≈ 0）
        if (use_prefetch) join_prefetch();

        // ── 8.1 读照片 ────────────────────────────────────────────
        auto tio = Clock::now();
        // ★ 预解码模式（--pre-dir）：直接读离线预转好的 512×288 灰度图。
        //   测速结论：4K PNG 硬解码 ~190 ms/帧，占端到端耗时 23%；
        //   且 OpenCV 的 IMREAD_REDUCED_* 对 PNG **完全无效**（实测 190→190 ms，
        //   因为 PNG 是无损压缩必须整张解码），所以只能离线预转。
        //   预转用与运行时**完全相同**的 INTER_AREA resize 路径 → 精度零差异。
        cv::Mat qgray;
        if (use_prefetch && pf_ok) {
            qgray = pf_gray;                 // ★ 预取线程已读好
        } else if (A.pre_dir) {
            qgray = cv::imread(std::string(A.pre_dir) + "/" + name, cv::IMREAD_GRAYSCALE);
        } else {
            cv::Mat src = cv::imread(path, cv::IMREAD_GRAYSCALE);
            if (!src.empty())
                cv::resize(src, qgray, cv::Size(A.W, A.H), 0, 0, cv::INTER_AREA);
        }
        if (qgray.empty()) {
            fprintf(stdout, "%-22s  读图失败，跳过\n", name.c_str());
            continue;
        }
        if (qgray.cols != A.W || qgray.rows != A.H)
            cv::resize(qgray, qgray, cv::Size(A.W, A.H), 0, 0, cv::INTER_AREA);
        t_io += ms_since(tio);

        std::vector<cv::KeyPoint> kp_q;
        cv::Mat desc_q;
        if (A.feat_xfeat) {
            auto tq = Clock::now();
            bool got_q;
            if (use_prefetch && pf_ok) {
                xf_q.x.swap(pf_kps.x); xf_q.y.swap(pf_kps.y);
                xf_q.score.swap(pf_kps.score); xf_q.desc.swap(pf_kps.desc);
                xf_q.n = pf_kps.n; pf_kps.n = 0;
                got_q = (xf_q.n > 0);
            } else {
                got_q = xfeat.extract(qgray.data, xf_q);
            }
            if (!got_q) {
                fprintf(stdout, "%-22s  XFeat 特征过少(%d)，跳过\n", name.c_str(), xf_q.n);
                continue;
            }
            t_npu_q += xfeat.ms_npu; t_dec_q += xfeat.ms_decode;
            kp_q.resize(xf_q.n);
            for (int i = 0; i < xf_q.n; ++i) kp_q[i] = cv::KeyPoint(xf_q.x[i], xf_q.y[i], 8.f);
            desc_q = cv::Mat(xf_q.n, 64, CV_32F, (void*)xf_q.desc.data()).clone();
            (void)tq;
        } else {
            orb->detectAndCompute(qgray, cv::noArray(), kp_q, desc_q);
        }
        if (kp_q.size() < 40) {
            fprintf(stdout, "%-22s  特征太少(%zu)，跳过\n", name.c_str(), kp_q.size());
            continue;
        }

        // ── 8.2 初始位姿 ──────────────────────────────────────────
        Camera cam_curr = proto;
        Camera cam_out  = proto;   // ★ 用于「输出/误差统计」的位姿（KF 只作用于它）
        bool have_init = false;

        // ★★ --init-from-file：用外部文件里已算好的估计位姿当初始 camera。
        //   只渲染（不进入 PnP 收敛回路），故渲染图 = 「估计位姿下的渲染」。
        if (use_estfile) {
            const GtRow* er = find_row(estfile, name);
            if (!er) {
                fprintf(stdout, "%-22s  估计位姿文件中无此帧，跳过\n", name.c_str());
                continue;
            }
            GtRow gv = *er;                  // 用估计位姿冒充 g 走同一条构造路径
            if (A.pilot_mode) {
                double rr = gv.a1, pp = gv.a2, yy = gv.a3 + A.yaw_offset;
                geopp::PilotC2W pc = geopp::pilot_c2w_from_lla_euler(
                        gv.lon, gv.lat, gv.alt, rr, pp, yy,
                        A.ref_lon, A.ref_lat, A.ref_alt, geo.ref_ecef);
                Vec3 C_ply{ (float)(pc.t.x + A.align_x),
                            (float)(pc.t.y + A.align_y),
                            (float)(pc.t.z + A.align_z) };
                geopp::Mat3 Rw2c = geopp::transpose(pc.R);
                Camera c = proto;
                c.world_to_cam = Mat4::identity();
                for (int r_ = 0; r_ < 3; ++r_)
                    for (int c_ = 0; c_ < 3; ++c_)
                        c.world_to_cam.m[c_ * 4 + r_] = (float)Rw2c.m[r_][c_];
                c.world_to_cam.m[12] = -(float)(Rw2c.m[0][0]*C_ply.x + Rw2c.m[0][1]*C_ply.y + Rw2c.m[0][2]*C_ply.z);
                c.world_to_cam.m[13] = -(float)(Rw2c.m[1][0]*C_ply.x + Rw2c.m[1][1]*C_ply.y + Rw2c.m[1][2]*C_ply.z);
                c.world_to_cam.m[14] = -(float)(Rw2c.m[2][0]*C_ply.x + Rw2c.m[2][1]*C_ply.y + Rw2c.m[2][2]*C_ply.z);
                cam_curr = c;
                have_init = true;
            } else {
                geopp::Vec3d gte = geo.to_local(gv.lon, gv.lat, gv.alt);
                Vec3 fwd = euler_to_fwd(gv.a3 + A.yaw_offset, gv.a2);
                Camera c2 = cam_from_lookdir(Vec3{(float)(gte.x + A.align_x),
                                                  (float)(gte.y + A.align_y),
                                                  (float)(gte.z + A.align_z)},
                                             fwd, Vec3{0.f, 0.f, 1.f}, proto);
                Mat4 w2c = c2.world_to_cam;
                cam_curr = proto;
                cam_curr.world_to_cam = w2c;
                have_init = true;
            }
            cam_out = cam_curr;
        }

        // ★★ pert_first_only：有成功历史帧后，不再用真值重初始化，
        //    转而走下面的 init_from_prev（用上一帧估计）——这才是误差传播的真实工况。
        const bool gt_branch_ok =
            !(A.pert_first_only && have_prev);
        if (!use_estfile && gt_branch_ok &&
            (A.init_from_gt || (!have_prev && A.init_from_prev)) && g) {
            // 验证路径：用真值经纬高 + 真值欧拉角构造初始位姿
            //
            // ★★ 两种构造方式：
            //   [PiLoT 官方管线] pilot_mode=true（默认）
            //       完全复刻 euler_angles_to_matrix_ECEF + OSG 翻转，
            //       姿态直接用真值的 roll/pitch/yaw，不做任何猜测。
            //       之后用 --align 把 ENU 平移拉到 PLY 系。
            //   [旧近似] pilot_mode=false
            //       euler_to_fwd + cam_from_lookdir（yaw/pitch 语义靠自己定义）
            if (A.pilot_mode) {
                // 真值的 [roll,pitch,yaw]（README: 3dgs_test.txt 列序为
                //   name lon lat alt roll pitch yaw）
                double rr = g->a1;
                double pp = g->a2;
                double yy = g->a3 + A.yaw_offset;
                if (A.pitch_override > 0.f) pp = (double)A.pitch_override;
                if (A.yaw_invert) yy = -yy;
                // ★★ 初值扰动 —— 模拟机载 GPS/IMU 粗定位（消融 / 收敛域测试）
                //   位置：pert_pos 圆盘内均匀采样（ENU 东/北）
                //   姿态：三轴各 ±pert_ang 均匀采样
                double lon_p = g->lon, lat_p = g->lat, alt_p = g->alt;
                double pert_used = 0.0;
                // ★ pert_first_only：只在还没有成功历史帧时施加扰动
                const bool do_pert = !(A.pert_first_only && have_prev);
                if (A.pert_pos > 0.f && do_pert) {
                    const double aa = u01() * 2.0 * M_PI;
                    const double rp = (double)A.pert_pos * std::sqrt(u01());
                    const double de = rp * std::cos(aa), dn = rp * std::sin(aa);
                    const double Re = 6378137.0;
                    lat_p += (dn / Re) * 180.0 / M_PI;
                    lon_p += (de / (Re * std::cos(g->lat * M_PI / 180.0))) * 180.0 / M_PI;
                    pert_used = rp;
                }
                if (A.pert_ang > 0.f && do_pert) {
                    rr += (u01() * 2.0 - 1.0) * A.pert_ang;
                    pp += (u01() * 2.0 - 1.0) * A.pert_ang;
                    yy += (u01() * 2.0 - 1.0) * A.pert_ang;
                }
                geopp::PilotC2W pc = geopp::pilot_c2w_from_lla_euler(
                        lon_p, lat_p, alt_p,
                        rr, pp, yy,
                        A.ref_lon, A.ref_lat, A.ref_alt, geo.ref_ecef);
                Vec3 C_ply{
                    (float)(pc.t.x + A.align_x),
                    (float)(pc.t.y + A.align_y),
                    (float)(pc.t.z + A.align_z) };
                // 由 c2w 旋转矩阵构造 world_to_cam：R_w2c = R_c2w^T, t = -R_w2c·C
                // ⚠️ Mat4 是列主序（camera.h upper3x3 用 m[j*4+i]），
                //    所以行主序 Rw2c.m[row][col] 必须写到 m[col*4+row]。
                geopp::Mat3 Rw2c = geopp::transpose(pc.R);
                Camera c = proto;
                c.world_to_cam = Mat4::identity();
                for (int r_ = 0; r_ < 3; ++r_)
                    for (int c_ = 0; c_ < 3; ++c_)
                        c.world_to_cam.m[c_ * 4 + r_] = (float)Rw2c.m[r_][c_];
                c.world_to_cam.m[12] = -(float)(Rw2c.m[0][0]*C_ply.x + Rw2c.m[0][1]*C_ply.y + Rw2c.m[0][2]*C_ply.z);
                c.world_to_cam.m[13] = -(float)(Rw2c.m[1][0]*C_ply.x + Rw2c.m[1][1]*C_ply.y + Rw2c.m[1][2]*C_ply.z);
                c.world_to_cam.m[14] = -(float)(Rw2c.m[2][0]*C_ply.x + Rw2c.m[2][1]*C_ply.y + Rw2c.m[2][2]*C_ply.z);
                c.fx = proto.fx; c.fy = proto.fy; c.cx = proto.cx; c.cy = proto.cy;
                c.znear = proto.znear; c.zfar = proto.zfar;
                cam_curr = c;
                have_init = true;
                if (A.verbose) {
                    // 光轴 = R_c2w 的第三列（相机 z 轴指向）
                    Vec3 fwd{(float)pc.R.m[0][2], (float)pc.R.m[1][2], (float)pc.R.m[2][2]};
                    fprintf(stdout, "  [pilot] C_ply=(%.1f, %.1f, %.1f)  "
                            "fwd=(%.3f, %.3f, %.3f)  pert=%.2f m\n",
                            C_ply.x, C_ply.y, C_ply.z, fwd.x, fwd.y, fwd.z, pert_used);
                }
            } else {
            geopp::Vec3d gte = geo.to_local(g->lon, g->lat, g->alt);
            Vec3 C_enu{ (float)gte.x, (float)gte.y, (float)gte.z };
            // ENU → PLY 系
            Vec3 C_ply{ C_enu.x + A.align_x, C_enu.y + A.align_y, C_enu.z + A.align_z };
            // 光轴：用真值欧拉角（ENU 系）→ 转 PLY 系（轴向一致，直接用）
            const double yaw_use = g->a3 + A.yaw_offset;
            const double pitch_use = (A.pitch_override > 0.f)
                                   ? (double)A.pitch_override : g->a2;
            Vec3 fwd = euler_to_fwd(yaw_use, pitch_use);
            if (A.yaw_invert) fwd = Vec3{ -fwd.x, -fwd.y, fwd.z };
            cam_curr = cam_from_lookdir(C_ply, fwd, Vec3{0.f, 0.f, 1.f}, proto);
            cam_curr.fx = proto.fx; cam_curr.fy = proto.fy;
            cam_curr.cx = proto.cx; cam_curr.cy = proto.cy;
            cam_curr.znear = proto.znear; cam_curr.zfar = proto.zfar;
            have_init = true;
            if (A.verbose)
                fprintf(stdout, "  [init-gt] C_ply=(%.1f, %.1f, %.1f)  fwd=(%.3f, %.3f, %.3f)\n",
                        C_ply.x, C_ply.y, C_ply.z, fwd.x, fwd.y, fwd.z);
            }
        }

        if (!have_init && A.init_from_prev && have_prev) {
            // 真实工况：用上一帧结果（PLY 系），加一点漂移
            Vec3 Ccur{ prev_C.x, prev_C.y, prev_C.z };
            cam_curr = cam_from_lookdir(Ccur, prev_fwd, Vec3{0.f, 0.f, 1.f}, proto);
            cam_curr.fx = proto.fx; cam_curr.fy = proto.fy;
            cam_curr.cx = proto.cx; cam_curr.cy = proto.cy;
            cam_curr.znear = proto.znear; cam_curr.zfar = proto.zfar;
            have_init = true;
        }

        if (!have_init) {
            // 冷启动：从场景上方沿真值方位（若有）俯视
            Vec3 Ccur{ center.x, center.y, mx[2] + 0.5f * radius };
            Vec3 fwd{0.f, 0.f, -1.f};
            if (g) fwd = euler_to_fwd(g->a3, g->a2);
            cam_curr = cam_from_lookdir(Ccur, fwd, Vec3{0.f, 0.f, 1.f}, proto);
            cam_curr.fx = proto.fx; cam_curr.fy = proto.fy;
            cam_curr.cx = proto.cx; cam_curr.cy = proto.cy;
            cam_curr.znear = proto.znear; cam_curr.zfar = proto.zfar;
        }

        // ── 8.3 render-and-compare 迭代 ───────────────────────────
        int    nmatch = 0, ninl = 0, iters_done = 0;
        Vec3   C_c{}, f_c{};
        bool   ok_frame = false;
        double ms_this = 0;
        // ★ 目标定位探针：保留本帧渲染图的深度与配对位姿（仅当有探针时才拷贝）
        Camera cam_rb_keep = proto;
        std::vector<float> rdepth_keep;
        double reproj_rms = -1.0;      // 内点重投影 RMS（px）
        double mean_cossim = -1.0;     // 匹配置信度：接受配对的平均余弦相似度

        if (A.pipeline) {
        // ══ ★ 帧间流水线模式（每帧 1 次渲染，回读等待隐藏在帧间）══
        // 时序（每帧）：
        //   [collect 上帧投递的渲染图] → [ORB 匹配 + PnP] → [投递本帧解算位姿的渲染]
        // 渲染图位姿 = 上一帧解算位姿（滞后 1 帧 = 帧间运动量）。
        // ⚠️ 反投影必须用 frame_camera()（渲染图的真实位姿），
        //    PnP 解出的仍是本帧 query 的位姿 —— 数学上与老路径一致。
        for (int it = 0; it < 1; ++it) {          // 单次外壳：保留 break 语义
            // 1. 收取上一帧投递的渲染图
            //    ★ 先 join 渲染线程：join 前主线程只做过「读图 + 查询图 NPU 提取」，
            //      那段 (~30 ms) 与上一帧的渲染 CPU 工作完全并行。
            auto tr = Clock::now();
            if (!r.collect_async_readback()) {
                // 首帧（无 pending）或 GPU 偶发未完成：投递本帧初值渲染并自旋等待
                // （首帧走这里，一次性多花 ~111 ms，之后每帧只投递一次）
                if (!r.render_frame(cam_curr, /*collect_prev=*/false)) break;
                // ★★ 有界自旋（2026-10-12 修复）：
                //   原来是无条件 `while(!collect){}`，GPU 偶发不返回时会**无限挂死**整个流程
                //   （实测在跑了 ~9 分钟、第 314 帧后永久卡住，单核 100% 空转）。
                //   生产管线必须有超时兜底。
                int spins = 0;
                while (!r.collect_async_readback()) {
                    if (++spins > 3000) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                if (spins > 3000) {
                    fprintf(stderr, "[warn] 渲染回读超时（%d ms），跳过本帧\n", spins);
                    break;
                }
            }
            const Camera cam_rb = r.frame_camera();   // ★ 渲染图的真实位姿
            std::vector<uint8_t> rrgba;
            std::vector<float>   rdepth_raw, rdepth;
            r.get_frame(rrgba);
            if (!r.get_depth(rdepth_raw)) break;
            flip_depth_rows(rdepth_raw, rdepth, A.W, A.H);
            if (!A.probe_px.empty()) { cam_rb_keep = cam_rb; rdepth_keep = rdepth; }
            cv::Mat rgray = to_gray(rrgba, A.W, A.H);
            t_render += ms_since(tr);
            ms_this  += ms_since(tr);

            if (A.dump) {
                char buf[1024];
                snprintf(buf, sizeof(buf), "%s/%s_ref_it%d.png",
                         A.dump, name.c_str(), it);
                cv::imwrite(buf, rgray);
            }
            if (A.dump_depth) {
                // ★ 可选：同时导出 Σαᵢ（--dump-wsum）
                std::vector<float> rw_raw, rw;
                const std::vector<float>* pw = nullptr;
                if (A.dump_wsum && r.get_depth_weights(rw_raw) &&
                    (int)rw_raw.size() == A.W * A.H) {
                    flip_depth_rows(rw_raw, rw, A.W, A.H);
                    pw = &rw;
                }
                char b2[1024];
                snprintf(b2, sizeof(b2), "%s/%s_depth_it%d.bin",
                         A.dump, name.c_str(), it);
                dump_depth_bin(b2, rdepth, A.W, A.H, cam_rb, pw);
            }
            if (A.dump_color) {
                char b3[1024];
                snprintf(b3, sizeof(b3), "%s/%s_refc_it%d.png",
                         A.dump, name.c_str(), it);
                cv::imwrite(b3, to_bgr(rrgba, A.W, A.H));
            }

            // 2. 特征匹配（⚠️ 反投影用 cam_rb —— 渲染图配对位姿）
            auto tm = Clock::now();
            std::vector<cv::KeyPoint> kp_r;
            cv::Mat desc_r;
            std::vector<std::pair<int, int>> pairs;
            if (A.feat_xfeat) {
                if (!xfeat.extract(rgray.data, xf_r)) break;
                t_npu += xfeat.ms_npu; t_dec += xfeat.ms_decode;
                kp_r.resize(xf_r.n);
                for (int i = 0; i < xf_r.n; ++i) kp_r[i] = cv::KeyPoint(xf_r.x[i], xf_r.y[i], 8.f);
                desc_r = cv::Mat(xf_r.n, 64, CV_32F, (void*)xf_r.desc.data()).clone();
                { auto tm2 = Clock::now(); xfeat.match(xf_q, xf_r, pairs); t_mnn += ms_since(tm2); }
            } else {
                orb->detectAndCompute(rgray, cv::noArray(), kp_r, desc_r);
                if (desc_r.empty()) break;
                std::vector<std::vector<cv::DMatch>> knn;
                matcher.knnMatch(desc_q, desc_r, knn, 2);
                for (const auto& v : knn) {
                    if (v.size() < 2) continue;
                    if (v[0].distance > 0.75f * v[1].distance) continue;
                    pairs.emplace_back(v[0].queryIdx, v[0].trainIdx);
                }
            }
            // ★ 可选：单应预筛（--homo-filter）
            //   本项目是「近垂直俯视 + 近似平面场景」，平面单应 x_r = H·x_q
            //   是很强的几何约束。先用 RANSAC 把几何不一致的错匹配剔掉，
            //   再交给 solvePnPRansac。筛过头（<8 点）就放弃本次筛选。
            std::vector<std::pair<int, int>> pairs_f;
            if (A.homo_filter && pairs.size() >= 8) {
                std::vector<cv::Point2f> sq, sr;
                sq.reserve(pairs.size()); sr.reserve(pairs.size());
                for (const auto& pr : pairs) {
                    sq.push_back(kp_q[pr.first].pt);    // 实拍图点
                    sr.push_back(kp_r[pr.second].pt);   // 渲染图点
                }
                std::vector<char> hm;
                cv::Mat Hm = cv::findHomography(sq, sr, cv::RANSAC,
                                                (double)A.homo_thresh, hm, 2000, 0.995);
                if (!Hm.empty() && !hm.empty()) {
                    std::vector<std::pair<int, int>> tmp;
                    tmp.reserve(pairs.size());
                    for (size_t bi = 0; bi < pairs.size(); ++bi)
                        if (hm[bi]) tmp.push_back(pairs[bi]);
                    n_homo_total += (long)pairs.size();
                    n_homo_kept  += (long)tmp.size();
                    if (tmp.size() >= 8) pairs_f.swap(tmp);
                }
            }
            const std::vector<std::pair<int, int>>& PU =
                pairs_f.empty() ? pairs : pairs_f;

            std::vector<cv::Point3f> obj;
            std::vector<cv::Point2f> img;
            for (const auto& pr : PU) {
                const cv::KeyPoint& kr = kp_r[pr.second];
                const int u = (int)kr.pt.x, vv = (int)kr.pt.y;
                if (u < 0 || vv < 0 || u >= A.W || vv >= A.H) continue;
                const float z = rdepth[(size_t)vv * A.W + u];
                if (z <= 0.01f) continue;          // 无高斯覆盖
                const Vec3 Xw = backproject(cam_rb, u, vv, z);
                obj.emplace_back(Xw.x, Xw.y, Xw.z);
                img.push_back(kp_q[pr.first].pt);
            }
            nmatch = (int)obj.size();
            t_match += ms_since(tm);
            ms_this += ms_since(tm);

            if (nmatch < 8) break;

            // 3. solvePnPRansac
            auto tp = Clock::now();
            cv::Mat K = K_of(proto);
            cv::Mat rvec, tvec;
            std::vector<int> inliers;
            const bool ok = cv::solvePnPRansac(
                obj, img, K, cv::Mat(), rvec, tvec,
                /*useExtrinsicGuess=*/false, 200, 3.0, 0.99,
                inliers, pnp_flag);
            t_pnp += ms_since(tp);
            ms_this += ms_since(tp);

            if (!ok || inliers.size() < 8) break;
            ninl = (int)inliers.size();
            // ★ 位姿不确定性：内点重投影 RMS（px）
            {
                std::vector<cv::Point3f> oi; std::vector<cv::Point2f> ii2;
                oi.reserve(inliers.size()); ii2.reserve(inliers.size());
                for (int q2 : inliers) { oi.push_back(obj[q2]); ii2.push_back(img[q2]); }
                std::vector<cv::Point2f> pj;
                cv::projectPoints(oi, rvec, tvec, K, cv::Mat(), pj);
                double ss = 0.0;
                for (size_t q2 = 0; q2 < pj.size(); ++q2) {
                    const double dx = pj[q2].x - ii2[q2].x, dy = pj[q2].y - ii2[q2].y;
                    ss += dx * dx + dy * dy;
                }
                reproj_rms = std::sqrt(ss / std::max<size_t>(1, pj.size()));
            }
            mean_cossim = xfeat.last_mean_cossim;   // ORB 路径保持 -1

            cam_curr = cam_from_pnp(rvec, tvec, proto);
            cam_curr.fx = proto.fx; cam_curr.fy = proto.fy;
            cam_curr.cx = proto.cx; cam_curr.cy = proto.cy;
            cam_curr.znear = proto.znear; cam_curr.zfar = proto.zfar;
            // ── 零延迟位姿后处理：投影到「铅垂姿态 + 等效铅垂位置」──
            //   ★★ 两步缺一不可（2026-10-12 踩坑）：
            //     只改位置不改姿态 → 下一帧的渲染来自「位置对了但还倾斜着的相机」，
            //     其等效铅垂位置又偏了 H·n → 误差逐帧累积（实测 300 帧发散到 241 m）。
            //   ① 等效铅垂位置 P = C + H·n_xy （n_xy 来自光轴水平分量）
            //   ② 姿态重建：roll=pitch=0，保留 yaw（投影回铅垂子流形）
            if (A.tilt_h > 0.f) {
                float* m = cam_curr.world_to_cam.m;
                // 光轴（世界系）= w2c 第三行 R(2,·) = (m[2], m[6], m[10])
                double fx = m[2], fy = m[6], fz = m[10];
                const double nl = std::sqrt(fx * fx + fy * fy + fz * fz);
                if (nl > 1e-9) { fx /= nl; fy /= nl; fz /= nl; }
                // ① 等效铅垂位置：P = C + H·(光轴水平分量)
                Vec3 Cc, fw;
                pose_of(cam_curr, Cc, fw);
                const double Px = Cc.x + A.tilt_h * fx;
                const double Py = Cc.y + A.tilt_h * fy;
                const double Pz = Cc.z;
                // ② 姿态：用**最小旋转**把光轴扳到铅垂 (0,0,-1)，绕开欧拉角约定
                //    R_fix = I + [v]× + [v]×²/(1+c)，v = fwd × target, c = fwd·target
                const double tx = 0, ty = 0, tz = -1;
                const double vx = fy * tz - fz * ty;
                const double vy = fz * tx - fx * tz;
                const double vz = fx * ty - fy * tx;
                const double c = fx * tx + fy * ty + fz * tz;
                const double k = 1.0 / std::max(1e-9, 1.0 + c);
                double F[3][3];
                F[0][0] = 1 - k * (vy * vy + vz * vz);
                F[0][1] = -vz + k * vx * vy;
                F[0][2] =  vy + k * vx * vz;
                F[1][0] =  vz + k * vx * vy;
                F[1][1] = 1 - k * (vx * vx + vz * vz);
                F[1][2] = -vx + k * vy * vz;
                F[2][0] = -vy + k * vx * vz;
                F[2][1] =  vx + k * vy * vz;
                F[2][2] = 1 - k * (vx * vx + vy * vy);
                double R[3][3], Rn[3][3];
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) R[i][j] = m[j * 4 + i];
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) {
                        double t = 0;
                        for (int q = 0; q < 3; ++q) t += F[i][q] * R[q][j];
                        Rn[i][j] = t;
                    }
                const double Ox = Px, Oy = Py, Oz = Pz;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) m[j * 4 + i] = (float)Rn[i][j];
                m[12] = (float)(-(Rn[0][0] * Ox + Rn[0][1] * Oy + Rn[0][2] * Oz));
                m[13] = (float)(-(Rn[1][0] * Ox + Rn[1][1] * Oy + Rn[1][2] * Oz));
                m[14] = (float)(-(Rn[2][0] * Ox + Rn[2][1] * Oy + Rn[2][2] * Oz));
                // ★★ 卡尔曼只作用于**输出**，绝不反馈给渲染先验：
                //   若把 KF 结果喂给下一帧渲染，相当于在闭环里插入滞后 → 环路失稳
                //   （实测：仅倾角 1.06 m → 倾角+KF反馈 23.7 m；改为只作用于输出 → 0.94 m）
                cam_out = cam_curr;
                if (A.use_kf) {
                    const double z[3] = { Px, Py, Pz };
                    double zo[3];
                    pkf.step(z, zo);
                    float* mo = cam_out.world_to_cam.m;
                    mo[12] = (float)(-(mo[0] * zo[0] + mo[4] * zo[1] + mo[8] * zo[2]));
                    mo[13] = (float)(-(mo[1] * zo[0] + mo[5] * zo[1] + mo[9] * zo[2]));
                    mo[14] = (float)(-(mo[2] * zo[0] + mo[6] * zo[1] + mo[10] * zo[2]));
                }
            } else {
                cam_out = cam_curr;
            }
            iters_done = 1;
            ok_frame = true;

            if (A.dump) {
                char buf[1024];
                snprintf(buf, sizeof(buf), "%s/%s_it%d.png", A.dump, name.c_str(), it);
                cv::imwrite(buf, rgray);
            }
        }

        // 4. 帧尾投递本帧（可能已更新的）位姿渲染，供下一帧匹配 —— 不等 GPU。
        //    失败帧也投递（cam_curr 此刻 = 初值 = prev 位姿），下一帧仍有图可匹配。
        auto ti2 = Clock::now();
        // ★★ 起下一帧的预取线程，然后才同步渲染 ——
        //    预取的 NPU 推理与渲染的 CPU 段（投影/排序/组装，~83 ms）在这里重叠
        if (use_prefetch && idx + 1 < f_end) start_prefetch(idx + 1);
        r.render_frame(cam_curr, /*collect_prev=*/false);
        {
            const auto& st = r.stats();
            s_proj += st.ms_project; s_sort += st.ms_sort; s_asm += st.ms_assemble;
            s_up   += st.ms_upload;  s_draw += st.ms_draw;  s_rb  += st.ms_readback;
            n_reused += st.sort_reused ? 1 : 0;
            ++n_render_calls;
        }
        t_render += ms_since(ti2);
        ms_this  += ms_since(ti2);
        }
        else {
        for (int it = 0; it < A.iters; ++it) {
            // a. 渲染参考图（彩色 + ED 深度）
            auto tr = Clock::now();
            if (!r.render_frame(cam_curr)) break;
            std::vector<uint8_t> rrgba;
            std::vector<float>   rdepth_raw, rdepth;
            r.get_frame(rrgba);
            if (!r.get_depth(rdepth_raw)) break;
            flip_depth_rows(rdepth_raw, rdepth, A.W, A.H);
            cv::Mat rgray = to_gray(rrgba, A.W, A.H);
            t_render += ms_since(tr);
            ms_this  += ms_since(tr);

            // ★ 渲染图立即导出（不受后续 break 影响，便于诊断）
            if (A.dump) {
                char buf[1024];
                snprintf(buf, sizeof(buf), "%s/%s_ref_it%d.png",
                         A.dump, name.c_str(), it);
                cv::imwrite(buf, rgray);
            }
            if (A.dump_depth) {
                // ★ 可选：同时导出 Σαᵢ（--dump-wsum），与非流水线分支同一逻辑
                std::vector<float> rw_raw, rw;
                const std::vector<float>* pw = nullptr;
                if (A.dump_wsum && r.get_depth_weights(rw_raw) &&
                    (int)rw_raw.size() == A.W * A.H) {
                    flip_depth_rows(rw_raw, rw, A.W, A.H);
                    pw = &rw;
                }
                char b2[1024];
                snprintf(b2, sizeof(b2), "%s/%s_depth_it%d.bin",
                         A.dump, name.c_str(), it);
                dump_depth_bin(b2, rdepth, A.W, A.H, cam_curr, pw);
            }
            if (A.dump_color) {
                char b3[1024];
                snprintf(b3, sizeof(b3), "%s/%s_refc_it%d.png",
                         A.dump, name.c_str(), it);
                cv::imwrite(b3, to_bgr(rrgba, A.W, A.H));
            }

            // b. 特征匹配
            auto tm = Clock::now();
            std::vector<cv::KeyPoint> kp_r;
            cv::Mat desc_r;
            std::vector<std::pair<int, int>> pairs;
            if (A.feat_xfeat) {
                if (!xfeat.extract(rgray.data, xf_r)) break;
                t_npu += xfeat.ms_npu; t_dec += xfeat.ms_decode;
                kp_r.resize(xf_r.n);
                for (int i = 0; i < xf_r.n; ++i) kp_r[i] = cv::KeyPoint(xf_r.x[i], xf_r.y[i], 8.f);
                desc_r = cv::Mat(xf_r.n, 64, CV_32F, (void*)xf_r.desc.data()).clone();
                xfeat.match(xf_q, xf_r, pairs);
            } else {
                orb->detectAndCompute(rgray, cv::noArray(), kp_r, desc_r);
                if (desc_r.empty()) break;
                std::vector<std::vector<cv::DMatch>> knn;
                matcher.knnMatch(desc_q, desc_r, knn, 2);
                for (const auto& v : knn) {
                    if (v.size() < 2) continue;
                    if (v[0].distance > 0.75f * v[1].distance) continue;
                    pairs.emplace_back(v[0].queryIdx, v[0].trainIdx);
                }
            }
            // ★ 可选：单应预筛（--homo-filter），与流水线分支同一逻辑
            std::vector<std::pair<int, int>> pairs_f;
            if (A.homo_filter && pairs.size() >= 8) {
                std::vector<cv::Point2f> sq, sr;
                sq.reserve(pairs.size()); sr.reserve(pairs.size());
                for (const auto& pr : pairs) {
                    sq.push_back(kp_q[pr.first].pt);
                    sr.push_back(kp_r[pr.second].pt);
                }
                std::vector<char> hm;
                cv::Mat Hm = cv::findHomography(sq, sr, cv::RANSAC,
                                                (double)A.homo_thresh, hm, 2000, 0.995);
                if (!Hm.empty() && !hm.empty()) {
                    std::vector<std::pair<int, int>> tmp;
                    tmp.reserve(pairs.size());
                    for (size_t bi = 0; bi < pairs.size(); ++bi)
                        if (hm[bi]) tmp.push_back(pairs[bi]);
                    n_homo_total += (long)pairs.size();
                    n_homo_kept  += (long)tmp.size();
                    if (tmp.size() >= 8) pairs_f.swap(tmp);
                }
            }
            const std::vector<std::pair<int, int>>& PU =
                pairs_f.empty() ? pairs : pairs_f;

            std::vector<cv::Point3f> obj;
            std::vector<cv::Point2f> img;
            for (const auto& pr : PU) {
                const cv::KeyPoint& kr = kp_r[pr.second];
                const int u = (int)kr.pt.x, vv = (int)kr.pt.y;
                if (u < 0 || vv < 0 || u >= A.W || vv >= A.H) continue;
                const float z = rdepth[(size_t)vv * A.W + u];
                if (z <= 0.01f) continue;          // 无高斯覆盖
                const Vec3 Xw = backproject(cam_curr, u, vv, z);
                obj.emplace_back(Xw.x, Xw.y, Xw.z);
                img.push_back(kp_q[pr.first].pt);
            }
            nmatch = (int)obj.size();
            t_match += ms_since(tm);
            ms_this += ms_since(tm);

            if (nmatch < 8) break;

            // c. solvePnPRansac
            auto tp = Clock::now();
            cv::Mat K = K_of(proto);
            cv::Mat rvec, tvec;
            std::vector<int> inliers;
            const bool ok = cv::solvePnPRansac(
                obj, img, K, cv::Mat(), rvec, tvec,
                /*useExtrinsicGuess=*/false, 200, 3.0, 0.99,
                inliers, pnp_flag);
            t_pnp += ms_since(tp);
            ms_this += ms_since(tp);

            if (!ok || inliers.size() < 8) break;
            ninl = (int)inliers.size();
            // ★ 位姿不确定性：内点重投影 RMS（px）
            {
                std::vector<cv::Point3f> oi; std::vector<cv::Point2f> ii2;
                oi.reserve(inliers.size()); ii2.reserve(inliers.size());
                for (int q2 : inliers) { oi.push_back(obj[q2]); ii2.push_back(img[q2]); }
                std::vector<cv::Point2f> pj;
                cv::projectPoints(oi, rvec, tvec, K, cv::Mat(), pj);
                double ss = 0.0;
                for (size_t q2 = 0; q2 < pj.size(); ++q2) {
                    const double dx = pj[q2].x - ii2[q2].x, dy = pj[q2].y - ii2[q2].y;
                    ss += dx * dx + dy * dy;
                }
                reproj_rms = std::sqrt(ss / std::max<size_t>(1, pj.size()));
            }
            mean_cossim = xfeat.last_mean_cossim;   // ORB 路径保持 -1

            cam_curr = cam_from_pnp(rvec, tvec, proto);
            cam_curr.fx = proto.fx; cam_curr.fy = proto.fy;
            cam_curr.cx = proto.cx; cam_curr.cy = proto.cy;
            cam_curr.znear = proto.znear; cam_curr.zfar = proto.zfar;
            // ── 零延迟位姿后处理：投影到「铅垂姿态 + 等效铅垂位置」──
            //   ★★ 两步缺一不可（2026-10-12 踩坑）：
            //     只改位置不改姿态 → 下一帧的渲染来自「位置对了但还倾斜着的相机」，
            //     其等效铅垂位置又偏了 H·n → 误差逐帧累积（实测 300 帧发散到 241 m）。
            //   ① 等效铅垂位置 P = C + H·n_xy （n_xy 来自光轴水平分量）
            //   ② 姿态重建：roll=pitch=0，保留 yaw（投影回铅垂子流形）
            if (A.tilt_h > 0.f) {
                float* m = cam_curr.world_to_cam.m;
                // 光轴（世界系）= w2c 第三行 R(2,·) = (m[2], m[6], m[10])
                double fx = m[2], fy = m[6], fz = m[10];
                const double nl = std::sqrt(fx * fx + fy * fy + fz * fz);
                if (nl > 1e-9) { fx /= nl; fy /= nl; fz /= nl; }
                // ① 等效铅垂位置：P = C + H·(光轴水平分量)
                Vec3 Cc, fw;
                pose_of(cam_curr, Cc, fw);
                const double Px = Cc.x + A.tilt_h * fx;
                const double Py = Cc.y + A.tilt_h * fy;
                const double Pz = Cc.z;
                // ② 姿态：用**最小旋转**把光轴扳到铅垂 (0,0,-1)，绕开欧拉角约定
                //    R_fix = I + [v]× + [v]×²/(1+c)，v = fwd × target, c = fwd·target
                const double tx = 0, ty = 0, tz = -1;
                const double vx = fy * tz - fz * ty;
                const double vy = fz * tx - fx * tz;
                const double vz = fx * ty - fy * tx;
                const double c = fx * tx + fy * ty + fz * tz;
                const double k = 1.0 / std::max(1e-9, 1.0 + c);
                double F[3][3];
                F[0][0] = 1 - k * (vy * vy + vz * vz);
                F[0][1] = -vz + k * vx * vy;
                F[0][2] =  vy + k * vx * vz;
                F[1][0] =  vz + k * vx * vy;
                F[1][1] = 1 - k * (vx * vx + vz * vz);
                F[1][2] = -vx + k * vy * vz;
                F[2][0] = -vy + k * vx * vz;
                F[2][1] =  vx + k * vy * vz;
                F[2][2] = 1 - k * (vx * vx + vy * vy);
                double R[3][3], Rn[3][3];
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) R[i][j] = m[j * 4 + i];
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) {
                        double t = 0;
                        for (int q = 0; q < 3; ++q) t += F[i][q] * R[q][j];
                        Rn[i][j] = t;
                    }
                const double Ox = Px, Oy = Py, Oz = Pz;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) m[j * 4 + i] = (float)Rn[i][j];
                m[12] = (float)(-(Rn[0][0] * Ox + Rn[0][1] * Oy + Rn[0][2] * Oz));
                m[13] = (float)(-(Rn[1][0] * Ox + Rn[1][1] * Oy + Rn[1][2] * Oz));
                m[14] = (float)(-(Rn[2][0] * Ox + Rn[2][1] * Oy + Rn[2][2] * Oz));
                // ★★ 卡尔曼只作用于**输出**，绝不反馈给渲染先验：
                //   若把 KF 结果喂给下一帧渲染，相当于在闭环里插入滞后 → 环路失稳
                //   （实测：仅倾角 1.06 m → 倾角+KF反馈 23.7 m；改为只作用于输出 → 0.94 m）
                cam_out = cam_curr;
                if (A.use_kf) {
                    const double z[3] = { Px, Py, Pz };
                    double zo[3];
                    pkf.step(z, zo);
                    float* mo = cam_out.world_to_cam.m;
                    mo[12] = (float)(-(mo[0] * zo[0] + mo[4] * zo[1] + mo[8] * zo[2]));
                    mo[13] = (float)(-(mo[1] * zo[0] + mo[5] * zo[1] + mo[9] * zo[2]));
                    mo[14] = (float)(-(mo[2] * zo[0] + mo[6] * zo[1] + mo[10] * zo[2]));
                }
            } else {
                cam_out = cam_curr;
            }
            iters_done = it + 1;
            ok_frame = true;

            if (A.dump && it == A.iters - 1) {
                char buf[1024];
                snprintf(buf, sizeof(buf), "%s/%s_it%d.png", A.dump, name.c_str(), it);
                cv::imwrite(buf, rgray);
            }
        }
        }  // else: 非流水线老路径

        pose_of(cam_out, C_c, f_c);   // 输出/统计用 cam_out（渲染仍用 cam_curr）

        // ── 8.4 误差 ──────────────────────────────────────────────
        // 估计位姿在 PLY 系；真值在 ENU 系。
        // 用 --align 给的平移把估计转到 ENU 系，再与真值比。
        //   X_enu = X_ply - align
        double epos = -1, eang = -1;
        double eroll = -1, epitch = -1, eyaw = -1, erot = -1;
        if (g && ok_frame) {
            geopp::Vec3d gte = geo.to_local(g->lon, g->lat, g->alt);
            const double ex = C_c.x - A.align_x;
            const double ey = C_c.y - A.align_y;
            const double ez = C_c.z - A.align_z;
            epos = std::sqrt((ex-gte.x)*(ex-gte.x) + (ey-gte.y)*(ey-gte.y)
                             + (ez-gte.z)*(ez-gte.z));
            // 朝向误差：真值光轴必须用**与渲染完全一致的 pilot 约定**来算。
            //
            // ⚠️⚠️ 这里原来用的是 `euler_to_fwd(g->a3, g->a2)`，那是**另一套约定**
            //   （fwd = (cos p·sin y, cos p·cos y, -sin p)），而且**完全忽略了
            //   --yaw-offset / --pitch-override**。渲染走的是 pilot 约定
            //   （euler_xyz_to_mat → Rz·Ry·Rx，再乘 OSG 的 diag(1,-1,-1)），
            //   两套口径不一致 → 算出**恒定的 ~51° 虚假朝向误差**
            //   （实测 20 帧 mean 51.297°，RMSE 仅 0.107°，是系统性偏差的典型特征）。
            //
            //   铁证：同一帧渲染图与真照片的 ZNCC 达 **0.85**
            //   （朝向真错 51° 时图像不可能有这么高的相关性）。
            //
            //   正确做法：调同一个 pilot_c2w_from_lla_euler，取 R 的第 3 列
            //   （R 是 camera→world；相机看 -Z(OSG 约定)，故世界系光轴 = R 第 3 列）。
            geopp::PilotC2W pg = geopp::pilot_c2w_from_lla_euler(
                    g->lon, g->lat, g->alt,
                    g->a1,
                    (A.pitch_override > 0.f) ? (double)A.pitch_override : g->a2,
                    g->a3 + A.yaw_offset,
                    A.ref_lon, A.ref_lat, A.ref_alt, geo.ref_ecef);
            Vec3 fgt{ (float)pg.R.m[0][2], (float)pg.R.m[1][2], (float)pg.R.m[2][2] };
            eang = ang_deg(fgt, f_c);

            // ★ 三轴欧拉角误差（同口径）
            //   ⚠️ 关键：pilot_c2w_from_lla_euler 返回的 R = R_e2l·Renu·Rpose·D，
            //     其中 D=diag(1,-1,-1) 已把 OSG 约定转成 OpenCV 约定 —— 所以
            //     pg.R 本身就是 **OpenCV 约定**的 c2w，与 PnP 的结果同空间。
            //     要拿回 Rpose（= Rz(y)Rx(p)Ry(r)）则需乘 D：
            //       Rpose ≈ (c2w)·D  （因 R_e2l·Renu ≈ I，ref 就在场景内）
            geopp::Mat3 Dm; Dm.m[0][0] = 1; Dm.m[1][1] = -1; Dm.m[2][2] = -1;
            const geopp::Mat3 Rc2w_est = c2w_of(cam_curr);      // OpenCV 约定
            const geopp::Mat3 Rpose_est = Rc2w_est * Dm;        // 去掉 D → 姿态矩阵
            double r_e = 0, p_e = 0, y_e = 0;
            euler_from_R(Rpose_est, r_e, p_e, y_e);
            // 评估基准 = 真值文件的原始 (roll,pitch,yaw)，不做任何补偿
            eroll  = ang_diff(r_e, g->a1);
            epitch = ang_diff(p_e, g->a2);
            eyaw   = ang_diff(y_e, g->a3);
            // ★ 旋转测地距离：用同一轴序构造真值姿态矩阵后比较
            const geopp::Mat3 M_gt = pose_mat_of(g->a1, g->a2, g->a3);
            erot = rot_geo_deg(M_gt, Rpose_est);

            epos_all.push_back(epos);
            eang_all.push_back(eang);
            eroll_all.push_back(eroll);
            epitch_all.push_back(epitch);
            eyaw_all.push_back(eyaw);
            erot_all.push_back(erot);
        }
        nmatch_all.push_back(nmatch);

        // ── 8.5 写输出 ────────────────────────────────────────────
        if (fout && ok_frame) {
            // 输出 ENU 局部坐标 + 姿态，**角度口径与输入真值 a1/a2/a3 一致**
            // （都是 pilot 约定的 euler_xyz = roll/pitch/yaw）。
            const double ex = C_c.x - A.align_x;
            const double ey = C_c.y - A.align_y;
            const double ez = C_c.z - A.align_z;
            // ★ 完整 6DoF 姿态输出：从 c2w 旋转矩阵做 ZYX 欧拉角分解
            //   （原实现把 roll 硬编码为 0 —— 因真值 roll 恰好恒为 0 而"蒙对"，
            //     但这不是真正的 6DoF 输出，roll 误差也无从谈起）
            //   约定与真值一致：Rpose = Rz(yaw)·Ry(pitch)·Rx(roll)
            //   （验证：真值 (0, 35.6, 108.9) 经 pilot 重建后分解可还原）
            const geopp::Mat3 Rc2w_out = c2w_of(cam_curr);
            geopp::Mat3 Dm_o; Dm_o.m[0][0] = 1; Dm_o.m[1][1] = -1; Dm_o.m[2][2] = -1;
            double roll_o = 0, pitch_o = 0, yaw_o = 0;
            euler_from_R(Rc2w_out * Dm_o, roll_o, pitch_o, yaw_o);
            // ★ 同时给出经纬高（更有实用价值）：ENU → ECEF → LLA
            double lon = 0, lat = 0, alt = 0;
            {
                const geopp::Mat3 Rl2e = geopp::get_rotation_enu_in_ecef(A.ref_lon, A.ref_lat);
                const geopp::Vec3d d{ ex, ey, ez };
                const geopp::Vec3d e = geopp::Vec3d{
                    geo.ref_ecef.x + Rl2e.m[0][0]*d.x + Rl2e.m[0][1]*d.y + Rl2e.m[0][2]*d.z,
                    geo.ref_ecef.y + Rl2e.m[1][0]*d.x + Rl2e.m[1][1]*d.y + Rl2e.m[1][2]*d.z,
                    geo.ref_ecef.z + Rl2e.m[2][0]*d.x + Rl2e.m[2][1]*d.y + Rl2e.m[2][2]*d.z };
                geopp::ecef_to_lla(e.x, e.y, e.z, lon, lat, alt);
            }
            // 格式与真相位文件一致：<name> <lon> <lat> <alt> <roll> <pitch> <yaw>
            double qw = 1, qx = 0, qy = 0, qz = 0;
            quat_from_R(Rc2w_out * Dm_o, qw, qx, qy, qz);
            const double inl_ratio = (nmatch > 0) ? (double)ninl / (double)nmatch : 0.0;
            const double cossim_v  = (mean_cossim > 0) ? mean_cossim : 0.0;
            const double rr_v      = (reproj_rms > 0) ? reproj_rms : 0.0;
            const double conf = inl_ratio * std::exp(-rr_v / 2.0) *
                                std::min(1.0, std::max(0.0, cossim_v));
            fprintf(fout,
                "%s %.10f %.10f %.4f %.4f %.4f %.4f %.6f %.6f %.6f %.6f %d %d %.3f %.3f %.3f %.4f"
                "   # ENU %.3f %.3f %.3f  err %.3f rot %.3f\n",
                name.c_str(), lon, lat, alt, roll_o, pitch_o, yaw_o, qw, qx, qy, qz,
                nmatch, ninl, inl_ratio, cossim_v, rr_v, conf,
                ex, ey, ez, epos, (erot > 0 ? erot : 0.0));
        }

        // ── 8.6 目标定位探针：像素 → 全局 3D 地理坐标 ──────────────
        if (fprobe && ok_frame && !A.probe_px.empty() &&
            (int)rdepth_keep.size() == A.W * A.H) {
            for (size_t pi = 0; pi < A.probe_px.size(); ++pi) {
                const int u = A.probe_px[pi].first, vv = A.probe_px[pi].second;
                if (u < 0 || vv < 0 || u >= A.W || vv >= A.H) continue;
                const float z = rdepth_keep[(size_t)vv * A.W + u];
                if (z <= 0.01f) continue;
                const Vec3 Xw = backproject(cam_rb_keep, u, vv, z);
                const double px = Xw.x - A.align_x, py = Xw.y - A.align_y, pz = Xw.z - A.align_z;
                double tlon = 0, tlat = 0, talt = 0;
                const geopp::Mat3 Rl2e = geopp::get_rotation_enu_in_ecef(A.ref_lon, A.ref_lat);
                const geopp::Vec3d e{
                    geo.ref_ecef.x + Rl2e.m[0][0]*px + Rl2e.m[0][1]*py + Rl2e.m[0][2]*pz,
                    geo.ref_ecef.y + Rl2e.m[1][0]*px + Rl2e.m[1][1]*py + Rl2e.m[1][2]*pz,
                    geo.ref_ecef.z + Rl2e.m[2][0]*px + Rl2e.m[2][1]*py + Rl2e.m[2][2]*pz };
                geopp::ecef_to_lla(e.x, e.y, e.z, tlon, tlat, talt);
                fprintf(fprobe, "%s %d %d %.10f %.10f %.4f   # ENU %.3f %.3f %.3f  depth %.2f\n",
                        name.c_str(), u, vv, tlon, tlat, talt, px, py, pz, z);
            }
        }

        // ★★ 更新 prev —— 只有**成功帧**才更新！
        //   原实现无条件执行，导致首帧冷启动/扰动失败后，
        //   坏位姿被写进 prev 并传染给后续所有帧（实测 300 帧全崩、匹配数 0）。
        //   失败时保留上一次成功位姿，后续帧仍有恢复机会。
        if (ok_frame) {
            prev_C = C_c; prev_fwd = f_c; have_prev = true;
            ++n_success;
        } else {
            ++n_fail;
        }

        const char* status = ok_frame ? (epos >= 0 ? "OK" : "OK(无真值)") : "失败";
        char ebuf[32] = "-", abuf[32] = "-";
        if (epos >= 0) snprintf(ebuf, sizeof(ebuf), "%.3f", epos);
        if (eang >= 0) snprintf(abuf, sizeof(abuf), "%.3f", eang);
        fprintf(stdout, "%-22s %-6d %-7d %-6d %-11s %-9s %-8.1f %s\n",
                name.c_str(), nmatch, ninl, iters_done, ebuf, abuf, ms_this, status);

        if (A.dump) {
            char buf[1024];
            snprintf(buf, sizeof(buf), "%s/%s_query.png", A.dump, name.c_str());
            cv::imwrite(buf, qgray);
        }
    }

    if (fout) fclose(fout);
    if (fprobe) fclose(fprobe);

    // ── 9. 汇总：完整指标卡 ───────────────────────────────────────
    auto median_of = [](std::vector<double> v) -> double {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    auto pct_of = [](std::vector<double> v, double q) -> double {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        size_t k = (size_t)(v.size() * q);
        if (k >= v.size()) k = v.size() - 1;
        return v[k];
    };
    if (use_prefetch) join_prefetch();
    const double wall_s = ms_since(t_wall0) / 1000.0;

    fprintf(stdout, "\n══════════════════ 指标卡 ══════════════════\n");

    // Completeness（完整率）
    const double completeness = (n_processed > 0) ? 100.0 * n_success / n_processed : 0.0;
    fprintf(stdout, "Completeness           : %.2f %%   (%d / %d 帧输出有效位姿，失败 %d)\n",
            completeness, n_success, n_processed, n_fail);

    if (!epos_all.empty()) {
        // Median Translation Error
        fprintf(stdout, "Median Translation Err : %.3f m   (mean %.3f / rmse %.3f / 90%% %.3f / max %.3f)\n",
                median_of(epos_all),
                std::accumulate(epos_all.begin(), epos_all.end(), 0.0) / epos_all.size(),
                std::sqrt(std::inner_product(epos_all.begin(), epos_all.end(),
                                             epos_all.begin(), 0.0) / epos_all.size()),
                pct_of(epos_all, 0.9), *std::max_element(epos_all.begin(), epos_all.end()));

        // Recall@k —— 分母是全部处理帧（失败帧记为未命中）
        int r1 = 0, r3 = 0, r5 = 0;
        for (double x : epos_all) {
            if (x < 1.0) ++r1;
            if (x < 3.0) ++r3;
            if (x < 5.0) ++r5;
        }
        const double den = (double)std::max(1, n_processed);
        fprintf(stdout, "Recall@1m              : %.2f %%   (%d / %d)\n", 100.0 * r1 / den, r1, n_processed);
        fprintf(stdout, "Recall@3m              : %.2f %%   (%d / %d)\n", 100.0 * r3 / den, r3, n_processed);
        fprintf(stdout, "Recall@5m              : %.2f %%   (%d / %d)\n", 100.0 * r5 / den, r5, n_processed);
        // ★ Recall@(k m, k°) —— 平移 + 旋转双门限（论文口径）
        {
            int q1 = 0, q3 = 0, q5 = 0;
            for (size_t zi = 0; zi < epos_all.size(); ++zi) {
                const double pe = epos_all[zi];
                const double re = (zi < erot_all.size()) ? erot_all[zi] : 1e9;
                if (pe < 1.0 && re < 1.0) ++q1;
                if (pe < 3.0 && re < 3.0) ++q3;
                if (pe < 5.0 && re < 5.0) ++q5;
            }
            fprintf(stdout, "Recall@(1m,1°)         : %.2f %%   (%d / %d)\n", 100.0 * q1 / den, q1, n_processed);
            fprintf(stdout, "Recall@(3m,3°)         : %.2f %%   (%d / %d)\n", 100.0 * q3 / den, q3, n_processed);
            fprintf(stdout, "Recall@(5m,5°)         : %.2f %%   (%d / %d)\n", 100.0 * q5 / den, q5, n_processed);
        }

        // Median Rotation Error（测地距离 + 三轴分解）
        if (!erot_all.empty()) {
            fprintf(stdout, "Median Rotation Err    : %.3f °   (mean %.3f / max %.3f，旋转矩阵测地距离)\n",
                    median_of(erot_all),
                    std::accumulate(erot_all.begin(), erot_all.end(), 0.0) / erot_all.size(),
                    *std::max_element(erot_all.begin(), erot_all.end()));
            fprintf(stdout, "  三轴中位绝对误差     : roll %.3f °   pitch %.3f °   yaw %.3f °\n",
                    median_of(eroll_all), median_of(epitch_all), median_of(eyaw_all));
            fprintf(stdout, "  三轴平均绝对误差     : roll %.3f °   pitch %.3f °   yaw %.3f °\n",
                    std::accumulate(eroll_all.begin(), eroll_all.end(), 0.0,
                                    [](double a, double b) { return a + std::fabs(b); }) / eroll_all.size(),
                    std::accumulate(epitch_all.begin(), epitch_all.end(), 0.0,
                                    [](double a, double b) { return a + std::fabs(b); }) / epitch_all.size(),
                    std::accumulate(eyaw_all.begin(), eyaw_all.end(), 0.0,
                                    [](double a, double b) { return a + std::fabs(b); }) / eyaw_all.size());
        }
    } else {
        fprintf(stdout, "（无真值对比，或所有帧都失败）\n");
    }

    // FPS
    if (n_processed > 0) {
        const double t_algo_ms = (t_render + t_match + t_pnp) / n_processed;
        fprintf(stdout, "FPS (端到端)           : %.2f   (%.0f ms/帧，含 4K 解码)\n",
                n_processed / wall_s, 1000.0 * wall_s / n_processed);
        fprintf(stdout, "FPS (纯算法)           : %.2f   (%.0f ms/帧：渲染+匹配+PnP)\n",
                1000.0 / t_algo_ms, t_algo_ms);
    }

    if (!nmatch_all.empty()) {
        double s = 0, mn_ = 1e9, mx_ = 0;
        for (int x : nmatch_all) { s += x; mn_ = std::min(mn_, (double)x); mx_ = std::max(mx_, (double)x); }
        fprintf(stdout, "匹配数                 : 平均 %.0f  最少 %.0f  最多 %.0f\n",
                s / nmatch_all.size(), mn_, mx_);
    }
    if (n_homo_total > 0) {
        fprintf(stdout, "单应预筛               : 保留 %ld / %ld = %.1f%%（剔除 %.1f%%，阈值 %.1f px）\n",
                n_homo_kept, n_homo_total,
                100.0 * (double)n_homo_kept / (double)n_homo_total,
                100.0 * (1.0 - (double)n_homo_kept / (double)n_homo_total),
                (double)A.homo_thresh);
    }
    fprintf(stdout, "耗时合计               : 渲染 %.0f ms / 匹配 %.0f ms / PnP %.0f ms / 读图 %.0f ms / 墙钟 %.1f s\n",
            t_render, t_match, t_pnp, t_io, wall_s);
    fprintf(stdout, "  分项细分            : 渲染图 NPU %.2f / 渲染图解码 %.2f / MNN 匹配 %.2f / 查询图 NPU+解码 %.2f  ms/帧\n",
            t_npu / std::max(1, n_processed),
            t_dec / std::max(1, n_processed),
            t_mnn / std::max(1, n_processed),
            (t_npu_q + t_dec_q) / std::max(1, n_processed));
    if (use_prefetch)
        fprintf(stdout, "  查询图预取            : 主线程等待 %.0f ms (%.2f ms/帧) → 其余全部被渲染隐藏\n",
                t_pf_wait, t_pf_wait / std::max(1, n_processed));
    if (A.feat_xfeat)
        fprintf(stdout, "  XFeat 分解            : NPU %.0f ms (%.2f ms/帧) / 解码 %.0f ms (%.2f ms/帧) / 匹配含解码 %.0f ms\n",
                t_npu, t_npu / std::max(1, n_processed), t_dec, t_dec / std::max(1, n_processed), t_match);
    if (n_render_calls > 0) {
        fprintf(stdout, "渲染分解(均)           : 投影 %.1f  排序 %.1f  组装 %.1f  上传 %.1f  绘制 %.1f  回读 %.1f  ms\n",
                s_proj / n_render_calls, s_sort / n_render_calls,
                s_asm / n_render_calls,  s_up / n_render_calls,
                s_draw / n_render_calls, s_rb / n_render_calls);
        fprintf(stdout, "排序复用               : %d / %d 帧命中\n", n_reused, n_render_calls);
    }
    if (A.out) fprintf(stdout, "输出文件               : %s\n", A.out);
    return 0;
}
