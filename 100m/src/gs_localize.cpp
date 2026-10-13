// gs_localize.cpp — 3DGS 端到端定位闭环（对标 PiLoT 的 render→match→solve）
//
// ══ 为什么走这条路 ════════════════════════════════════════════════
// PiLoT 的 localization_worker 依赖一个我们拿不到的 ckpt
// （model@mapscape@512@Fourier.ckpt），但它整条链的本质是：
//
//     渲染参考图 + 深度  →  2D-3D 对应  →  位姿解算
//
// 而这三步用 **3DGS 渲染器 + OpenCV(features2d/calib3d)** 就能完成，
// 完全不需要神经网络权重。渲染侧已经在我们手里（含**米制 ED 深度**）。
//
// ══ 闭环怎么验证（不需要外部实拍数据）════════════════════════════
// 用「虚拟拍摄」自造 query，形成自洽闭环：
//   1) 真值位姿 T_gt 渲染一张图，作为「无人机实拍查询图」（只用彩色）
//   2) 给 T_gt 加扰动得到初值 T_0（模拟 GPS/IMU 粗定位误差）
//   3) 迭代 render-and-compare：
//        a. 用 T_curr 渲染参考图（彩色 + ED 米制深度）
//        b. ORB 提取特征，query ↔ ref 匹配（ratio test）
//        c. 用 **ref 的深度**把匹配点反投影成世界坐标 3D 点
//        d. solvePnPRansac(3D 世界点, query 2D 点) → 解出 query 位姿
//        e. 更新 T_curr
//   4) 报位置/旋转误差，与真值对比
//
//   ★ 关键：query 只用彩色（模拟真实相机），3D 点全部来自「渲染深度」，
//     所以这条链真的在验「渲染→定位」，而不是左右手互搏。
//
// ══ 用法 ═════════════════════════════════════════════════════════
//   ./gs_localize <ply> [--w 512 --h 288 --fovy 60]
//                     [--pert-pos 3 --pert-ang 3]  初始位姿扰动幅度
//                     [--iters 5]                 迭代轮数
//                     [--seed 0]                  随机种子（可复现）
//                     [--dump out_dir]            导出可视化

#include "renderer.h"
#include "camera.h"
#include "ply_loader.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

using namespace gsplat;

// ── 计时 ──────────────────────────────────────────────────────────
static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}
using Clock = std::chrono::steady_clock;

// ── 渲染帧 → 灰度 cv::Mat ─────────────────────────────────────────
//
// ⚠️⚠️ **必须垂直翻转**：`frame_` / `depth_` 都是 `glReadPixels` 的原始
//    行序（第 0 行 = **屏幕底部**，GL 约定）；而 cv::Mat / OpenCV 的
//    图像坐标是第 0 行 = **顶部**。
//
//    对 ORB 匹配本身无害（query 与 ref 同样颠倒，特征分布一致），但
//    **反投影会致命**：反投影公式 (v - cy)/fy · z 要求 v 是「自顶向下」
//    的行号。若把「自底向上」的行号喂进去，v-cy 的符号反了 → 反投影出的
//    3D 点沿 y 轴镜像 → solvePnP 求出的是个「y 翻转」的位姿 →
//    下一轮渲染更加偏离 → **迭代单调发散**。
//    实测症状：位置误差 1.55 → 3.07 → 5.43 → 10.17 → 19.03 m，
//    匹配数 1766 → 629 → 0，即典型的「错误累积」轨迹。
//
//    ★ 这条与 depthcheck 里踩的 y 翻转坑**同源**。当时总结的教训是
//      「涉及坐标翻转的代码必须用独立物理量交叉验证」；这里换个马甲
//      又踩了一次，说明教训要变成**默认动作**：任何 GL 回读的数据，
//      进 CPU 算法前先问一句「行序是哪头起的」。
static cv::Mat to_gray(const std::vector<uint8_t>& rgba, int W, int H) {
    cv::Mat g(H, W, CV_8UC1);
    for (int y = 0; y < H; ++y) {
        // 目标第 y 行 ← 源第 (H-1-y) 行（翻转）
        const uint8_t* s = rgba.data() + (size_t)(H - 1 - y) * W * 4;
        uint8_t* d = g.ptr<uint8_t>(y);
        for (int x = 0; x < W; ++x) {
            // Rec.601 灰度（R,G,B 在 0/1/2）
            d[x] = (uint8_t)((77 * s[0] + 150 * s[1] + 29 * s[2]) >> 8);
            s += 4;
        }
    }
    return g;
}

// ── 深度：GL 行序 → 标准图像序（同样必须翻转）─────────────────────
static void flip_depth_rows(const std::vector<float>& src,
                            std::vector<float>& dst, int W, int H) {
    dst.resize((size_t)W * H);
    for (int y = 0; y < H; ++y)
        std::memcpy(&dst[(size_t)y * W],
                    &src[(size_t)(H - 1 - y) * W],
                    sizeof(float) * W);
}

// ── 相机内参 → cv::Mat ────────────────────────────────────────────
static cv::Mat K_of(const Camera& c) {
    return (cv::Mat_<double>(3, 3) << c.fx, 0, c.cx,
                                      0, c.fy, c.cy,
                                      0, 0, 1);
}

// ── 像素 + 视空间深度 z(米) → 世界坐标 ────────────────────────────
// 相机系是 OpenCV 约定（x 右, y 下, z 前），深度就是该系的 Z。
//   X_cam = ((u+0.5-cx)/fx * z, (v+0.5-cy)/fy * z, z)
//   X_world = Rᵀ (X_cam − t)     其中 world_to_cam = [R | t]
static Vec3 backproject(const Camera& cam, int u, int v, float z) {
    const float* m = cam.world_to_cam.m;   // 列主序：R[i][j] = m[j*4+i]
    const double xc = (u + 0.5 - cam.cx) / (double)cam.fx * z;
    const double yc = (v + 0.5 - cam.cy) / (double)cam.fy * z;
    const double dx = xc - m[12], dy = yc - m[13], dz = z - m[14];
    // Rᵀ 的第 0 行 = R 的第 0 列 = (m[0], m[1], m[2])
    Vec3 o;
    o.x = (float)(m[0] * dx + m[1] * dy + m[2] * dz);
    o.y = (float)(m[4] * dx + m[5] * dy + m[6] * dz);
    o.z = (float)(m[8] * dx + m[9] * dy + m[10] * dz);
    return o;
}

// ── solvePnP 的 (rvec,tvec) → Camera ─────────────────────────────
// OpenCV solvePnP 解出的正是 world→camera 的 R,t：X_cam = R·X_world + t
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

// ── 从 Camera 取出相机中心与世界系朝向（用于算误差）──────────────
// X_world = Rᵀ(X_cam − t) ⇒ 相机中心 X_cam=0 ⇒ C = −Rᵀ·t
// 光轴方向 = Rᵀ·(0,0,1)
static void pose_of(const Camera& c, Vec3& C, Vec3& fwd) {
    const float* m = c.world_to_cam.m;
    const double tx = m[12], ty = m[13], tz = m[14];
    C.x = (float)-(m[0] * tx + m[1] * ty + m[2] * tz);
    C.y = (float)-(m[4] * tx + m[5] * ty + m[6] * tz);
    C.z = (float)-(m[8] * tx + m[9] * ty + m[10] * tz);
    fwd.x = m[8]; fwd.y = m[9]; fwd.z = m[10];   // Rᵀ 的第 2 列
}

static float ang_deg(const Vec3& a, const Vec3& b) {
    auto n = [](const Vec3& v) { return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z); };
    const float na = n(a), nb = n(b);
    if (na < 1e-9f || nb < 1e-9f) return 0.f;
    float cs = (a.x*b.x + a.y*b.y + a.z*b.z) / (na * nb);
    cs = cs < -1.f ? -1.f : (cs > 1.f ? 1.f : cs);
    return (float)(std::acos(cs) * 180.0 / 3.14159265358979);
}

// ── 相机基的小扰动：直接在 main 里用 look_at 构造 ─────────────────
//
// ⚠️ 这里曾经有一个函数把 `pose_of()` 返回的**单位方向向量** fwd
//    直接当成 `look_at(eye, target, up)` 的 **target 世界坐标点** 传进去，
//    结果 target 落在相机自己身上（距离 1 m），look_at 算出的朝向是乱的。
//    实测症状：朝向误差恒定在 94.5° 不动，无论迭代多少轮。
//    —— ★ 教训：`target` 是「点」，`fwd` 是「方向」，量纲不同不能混用。
//    现在改为在 main 里用「场景中心 + 偏移」当 target，语义正确。

int main(int argc, char** argv) {
    const char* ply = nullptr;
    int   W = 512, H = 288;
    float fovy = 60.f;
    float pert_pos = 3.0f;      // 初始位置扰动（米，每个轴 ±）
    float pert_ang = 3.0f;      // 初始朝向扰动（度）
    int   iters = 5;
    unsigned seed = 0;
    const char* dump_dir = nullptr;
    float min_alpha_radius = 0.5f;   // ★ 重要性剔除阈值（速度/质量权衡）

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--w")        && i+1<argc) W = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--h")        && i+1<argc) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fovy")     && i+1<argc) fovy = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--pert-pos") && i+1<argc) pert_pos = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--pert-ang") && i+1<argc) pert_ang = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--iters")    && i+1<argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed")     && i+1<argc) seed = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump")     && i+1<argc) dump_dir = argv[++i];
        else if (!strcmp(argv[i], "--min-alpha-radius") && i+1<argc)
            min_alpha_radius = (float)atof(argv[++i]);
        else if (!ply) ply = argv[i];
    }
    if (!ply) {
        fprintf(stderr, "用法: %s <ply> [--w N --h N --fovy DEG --pert-pos M "
                        "--pert-ang DEG --iters N --seed N --dump DIR]\n", argv[0]);
        return 2;
    }

    // ── 1. 加载模型（顺便算包围盒，用于自动摆放相机）──────────────
    std::vector<Splat>    splats;
    std::vector<ShCoeffs> shcs;
    if (!load_ply_splats(ply, splats, shcs, 1, false)) {
        fprintf(stderr, "PLY 加载失败: %s\n", ply);
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

    // ── 2. 建渲染器 ───────────────────────────────────────────────
    RenderConfig cfg;
    cfg.width  = W;
    cfg.height = H;
    cfg.depth  = true;          // ★ 必须开：3D 点全靠它
    cfg.min_alpha_radius = min_alpha_radius;
    cfg.verbose = false;
    Renderer r;
    if (!r.init(ply, cfg)) { fprintf(stderr, "渲染器初始化失败\n"); return 1; }

    // ── 3. 真值位姿 + 相机内参 ────────────────────────────────────
    // 轨道相机：从场景外围上方斜视（贴近无人机视角）
    const float dist = radius * 1.4f;
    const float ang  = 25.0f * 3.14159265f / 180.f;
    const Vec3 eye_gt{ center.x + dist * std::sin(ang),
                       center.y - dist * 0.30f,
                       center.z + dist * std::cos(ang) };
    Camera cam_gt = Camera::look_at(eye_gt, center, Vec3{0.f, -1.f, 0.f},
                                    W, H, fovy);
    cam_gt.zfar  = dist + radius * 4.f;
    cam_gt.znear = std::fmax(0.01f, dist * 0.002f);

    fprintf(stdout, "════════ 3DGS 定位闭环 ════════\n");
    fprintf(stdout, "模型      : %s\n", ply);
    fprintf(stdout, "高斯数    : %zu\n", splats.size());
    fprintf(stdout, "包围盒中心: (%.2f, %.2f, %.2f)  半径 %.2f m\n",
            center.x, center.y, center.z, radius);
    fprintf(stdout, "渲染尺寸  : %dx%d  fovy=%.1f°\n", W, H, fovy);
    fprintf(stdout, "内参      : fx=%.2f fy=%.2f cx=%.2f cy=%.2f\n",
            cam_gt.fx, cam_gt.fy, cam_gt.cx, cam_gt.cy);

    // ── 4. 虚拟拍摄：渲染 query（模拟无人机实拍，只取彩色）────────
    auto t0 = Clock::now();
    if (!r.render_frame(cam_gt)) { fprintf(stderr, "query 渲染失败\n"); return 1; }
    const double ms_q = ms_since(t0);
    std::vector<uint8_t> qrgba;
    r.get_frame(qrgba);
    cv::Mat qgray = to_gray(qrgba, W, H);

    cv::Ptr<cv::ORB> orb = cv::ORB::create(3000);
    std::vector<cv::KeyPoint> kp_q;
    cv::Mat desc_q;
    orb->detectAndCompute(qgray, cv::noArray(), kp_q, desc_q);
    fprintf(stdout, "query     : 渲染 %.1f ms, ORB 特征 %zu 个\n", ms_q, kp_q.size());
    if (kp_q.size() < 30) {
        fprintf(stderr, "⚠️ query 特征太少（%zu），换视角或调 fovy\n", kp_q.size());
        return 1;
    }
    if (dump_dir) {
        cv::imwrite(std::string(dump_dir) + "/query.png", qgray);
    }

    // ── 5. 扰动初值 ───────────────────────────────────────────────
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> upos(-pert_pos, pert_pos);
    std::uniform_real_distribution<float> uang(-pert_ang, pert_ang);
    const Vec3 dEye{ upos(rng), upos(rng), upos(rng) };
    // 朝向扰动用「注视目标点偏移」等效施加：在 dist 距离上偏 tan(θ)
    // ⚠️ 三个轴必须**独立采样**（之前共用一个 tscale，扰动量退化）
    const float tdeg = 3.14159265f / 180.f;
    const Vec3 dTgt{ dist * std::tan(uang(rng) * tdeg),
                     dist * std::tan(uang(rng) * tdeg),
                     dist * std::tan(uang(rng) * tdeg) };

    // ★ target 必须是**世界坐标点**（场景中心 + 偏移），不是方向向量
    const Vec3 eye0{ eye_gt.x + dEye.x, eye_gt.y + dEye.y, eye_gt.z + dEye.z };
    const Vec3 tgt0{ center.x + dTgt.x, center.y + dTgt.y, center.z + dTgt.z };
    Camera cam_curr = Camera::look_at(eye0, tgt0, Vec3{0.f, -1.f, 0.f}, W, H, fovy);
    // look_at 会按 fovy 重算内参，必须覆盖回与 query 完全一致，
    // 否则 PnP 用的内参和渲染用的不一致 → 系统性偏差
    cam_curr.fx = cam_gt.fx; cam_curr.fy = cam_gt.fy;
    cam_curr.cx = cam_gt.cx; cam_curr.cy = cam_gt.cy;
    cam_curr.znear = cam_gt.znear; cam_curr.zfar = cam_gt.zfar;

    Vec3 C_gt, f_gt, C_c, f_c;
    pose_of(cam_gt, C_gt, f_gt);
    pose_of(cam_curr, C_c, f_c);
    fprintf(stdout, "\n初值扰动  : 位置 %.3f m, 朝向 %.3f°\n",
            std::sqrt(dEye.x*dEye.x + dEye.y*dEye.y + dEye.z*dEye.z),
            ang_deg(f_gt, f_c));
    fprintf(stdout, "\n%-4s %-10s %-10s %-12s %-10s\n",
            "轮", "匹配", "内点", "位置误差(m)", "朝向误差(°)");
    fprintf(stdout, "──────────────────────────────────────────────\n");

    // ── 6. render-and-compare 迭代 ────────────────────────────────
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    double ms_render = 0, ms_match = 0, ms_pnp = 0;
    float  first_epos = -1.f, first_eang = -1.f;

    for (int it = 0; it < iters; ++it) {
        // a. 渲染参考图（彩色 + ED 深度）
        auto tr = Clock::now();
        if (!r.render_frame(cam_curr)) { fprintf(stderr, "渲染失败\n"); return 1; }
        std::vector<uint8_t> rrgba;
        std::vector<float>   rdepth_raw, rdepth;
        r.get_frame(rrgba);
        if (!r.get_depth(rdepth_raw)) { fprintf(stderr, "取深度失败\n"); return 1; }
        flip_depth_rows(rdepth_raw, rdepth, W, H);   // ★ GL 行序 → 图像序
        cv::Mat rgray = to_gray(rrgba, W, H);
        ms_render += ms_since(tr);

        // b. ORB
        auto tm = Clock::now();
        std::vector<cv::KeyPoint> kp_r;
        cv::Mat desc_r;
        orb->detectAndCompute(rgray, cv::noArray(), kp_r, desc_r);
        if (desc_r.empty()) { fprintf(stderr, "ref 无特征\n"); break; }

        std::vector<std::vector<cv::DMatch>> knn;
        matcher.knnMatch(desc_q, desc_r, knn, 2);
        std::vector<cv::Point3f> obj;
        std::vector<cv::Point2f> img;
        for (const auto& v : knn) {
            if (v.size() < 2) continue;
            if (v[0].distance > 0.75f * v[1].distance) continue;   // ratio test
            const cv::KeyPoint& kr = kp_r[v[0].trainIdx];
            const int u = (int)kr.pt.x, vv = (int)kr.pt.y;
            if (u < 0 || vv < 0 || u >= W || vv >= H) continue;
            const float z = rdepth[(size_t)vv * W + u];
            if (z <= 0.01f) continue;                 // 无覆盖像素
            const Vec3 Xw = backproject(cam_curr, u, vv, z);
            obj.emplace_back(Xw.x, Xw.y, Xw.z);
            img.push_back(kp_q[v[0].queryIdx].pt);
        }
        const int nmatch = (int)obj.size();
        ms_match += ms_since(tm);

        if (nmatch < 8) {
            fprintf(stdout, "%-4d %-10d %-10s 匹配不足，跳过\n", it, nmatch, "-");
            continue;
        }

        // c. solvePnPRansac
        auto tp = Clock::now();
        cv::Mat K = K_of(cam_gt);
        cv::Mat rvec, tvec;
        std::vector<int> inliers;
        const bool ok = cv::solvePnPRansac(
            obj, img, K, cv::Mat(), rvec, tvec,
            /*useExtrinsicGuess=*/false,
            /*iterationsCount=*/200,
            /*reprojectionError=*/3.0,
            /*confidence=*/0.99,
            inliers, cv::SOLVEPNP_ITERATIVE);
        ms_pnp += ms_since(tp);
        if (!ok || inliers.size() < 8) {
            fprintf(stdout, "%-4d %-10d %-10s PnP 失败\n",
                    it, nmatch, std::to_string(inliers.size()).c_str());
            continue;
        }

        // d. 更新位姿
        cam_curr = cam_from_pnp(rvec, tvec, cam_gt);

        pose_of(cam_curr, C_c, f_c);
        const float epos = std::sqrt((C_c.x-C_gt.x)*(C_c.x-C_gt.x) +
                                     (C_c.y-C_gt.y)*(C_c.y-C_gt.y) +
                                     (C_c.z-C_gt.z)*(C_c.z-C_gt.z));
        const float eang = ang_deg(f_gt, f_c);
        if (first_epos < 0.f) { first_epos = epos; first_eang = eang; }
        fprintf(stdout, "%-4d %-10d %-10zu %-12.3f %-10.3f\n",
                it, nmatch, inliers.size(), epos, eang);

        if (dump_dir) {
            char buf[512];
            snprintf(buf, sizeof(buf), "%s/ref_it%d.png", dump_dir, it);
            cv::imwrite(buf, rgray);
        }
    }

    // ── 7. 收尾 ───────────────────────────────────────────────────
    pose_of(cam_curr, C_c, f_c);
    const float epos = std::sqrt((C_c.x-C_gt.x)*(C_c.x-C_gt.x) +
                                 (C_c.y-C_gt.y)*(C_c.y-C_gt.y) +
                                 (C_c.z-C_gt.z)*(C_c.z-C_gt.z));
    const float eang = ang_deg(f_gt, f_c);

    fprintf(stdout, "──────────────────────────────────────────────\n");
    fprintf(stdout, "★ 最终误差: 位置 %.3f m   朝向 %.3f°\n", epos, eang);
    fprintf(stdout, "耗时合计  : 渲染 %.1f ms / 匹配 %.1f ms / PnP %.1f ms\n",
            ms_render, ms_match, ms_pnp);
    fprintf(stdout, "（初始位置扰动 ±%.1f m，朝向 ±%.1f°）\n", pert_pos, pert_ang);
    if (first_epos >= 0.f)
        fprintf(stdout, "\n收敛过程: 位置 %.3f m → %.3f m,  朝向 %.3f° → %.3f°\n",
                first_epos, epos, first_eang, eang);
    return (epos < 1.0f && eang < 1.0f) ? 0 : 1;
}
