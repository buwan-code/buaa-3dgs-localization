// calib_pilot.cpp — 用 PiLoT 官方管线标定 PLY↔ENU 对齐
//
// 背景（2026-10-08 从 PiLoT 源码确认）：
//   真值位姿 [lon,lat,alt,roll,pitch,yaw] → c2w 的构造是确定的：
//     R_pose = Rz(yaw)·Ry(pitch)·Rx(roll)        (scipy 'xyz' 内旋)
//     R_c2w  = ECEF→ENU(ref) · ENU→ECEF(lon,lat) · R_pose
//     R_c2w *= diag(1,-1,-1)                      (OSG 翻转)
//     t      = ENU(ref) 下的相机中心
//   唯一未知的是 PLY 系与 ENU 系之间的平移 T（3 个数）。
//
// 本工具做的事：
//   固定第 0 帧真值（几何完全确定），在 (align_x, align_y, align_z, yaw_offset)
//   四维网格上渲染并打分，找到让渲染图最像真照片的那组参数。
//
// 打分：ZNCC（零均值归一化互相关），带覆盖率约束，可选梯度域。
//
// 用法:
//   ./build/calib_pilot <ply> --query DIR --gt FILE --ref LON LAT ALT
//        [--ax0 --ax1 --nax] [--ay0 --ay1 --nay] [--az0 --az1 --naz]
//        [--yo0 --yo1 --nyo] [--w 256 --h 144] [--dump BESTDIR] [--topo]
//
// 输出：按分数排序的候选列表（stdout），可存盘。

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <dirent.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "camera.h"
#include "renderer.h"
#include "ply_loader.h"
#include "geo_pose.h"

using Clock = std::chrono::steady_clock;

// ★ 渲染器相关类型（RenderConfig / Renderer / Camera / Mat4）都在 gsplat 命名空间里，
//   geo_pose.h 只提供 geopp::（坐标转换），两者不冲突。少了这行会在
//   "RenderConfig was not declared" 处连环报错。
using namespace gsplat;

static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// ── 灰度（GL 行序翻转）──────────────────────────────────────────
static cv::Mat to_gray(const std::vector<uint8_t>& rgba, int W, int H) {
    cv::Mat out(H, W, CV_8U);
    for (int y = 0; y < H; ++y) {
        const uint8_t* src = rgba.data() + (size_t)(H - 1 - y) * W * 4;
        uint8_t* dst = out.ptr<uint8_t>(y);
        for (int x = 0; x < W; ++x) {
            const uint8_t* p = src + x * 4;
            dst[x] = (uint8_t)((77 * p[0] + 150 * p[1] + 29 * p[2]) >> 8);
        }
    }
    return out;
}

static void flip_rows(const std::vector<float>& in, std::vector<float>& out, int W, int H) {
    out.resize((size_t)W * H);
    for (int y = 0; y < H; ++y)
        memcpy(&out[(size_t)y * W], &in[(size_t)(H - 1 - y) * W], sizeof(float) * W);
}

// ── ZNCC（带 mask，含覆盖率约束）────────────────────────────────
static double zncc(const cv::Mat& a, const cv::Mat& b, const cv::Mat& mask,
                   double& cover) {
    CV_Assert(a.size() == b.size() && a.type() == CV_32F);
    int n = 0; double sa = 0, sb = 0;
    for (int y = 0; y < a.rows; ++y) {
        const float* pa = a.ptr<float>(y);
        const float* pb = b.ptr<float>(y);
        const uint8_t* pm = mask.ptr<uint8_t>(y);
        for (int x = 0; x < a.cols; ++x) if (pm[x]) { sa += pa[x]; sb += pb[x]; ++n; }
    }
    cover = (double)n / (double)(a.rows * a.cols);
    if (n < 200) return -1.0;
    const double ma = sa / n, mb = sb / n;
    double num = 0, da = 0, db = 0;
    for (int y = 0; y < a.rows; ++y) {
        const float* pa = a.ptr<float>(y);
        const float* pb = b.ptr<float>(y);
        const uint8_t* pm = mask.ptr<uint8_t>(y);
        for (int x = 0; x < a.cols; ++x) if (pm[x]) {
            const double u = pa[x] - ma, v = pb[x] - mb;
            num += u * v; da += u * u; db += v * v;
        }
    }
    const double den = std::sqrt(da * db);
    if (den < 1e-9) return -1.0;
    return num / den;
}

// 梯度幅值
static cv::Mat grad_mag(const cv::Mat& g) {
    cv::Mat gx, gy, mag;
    cv::Sobel(g, gx, CV_32F, 1, 0, 3);
    cv::Sobel(g, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);
    return mag;
}

// ── 参数 ────────────────────────────────────────────────────────
struct Args {
    const char* ply = nullptr;
    const char* query = nullptr;
    const char* gt = nullptr;
    const char* dump = nullptr;
    int   W = 256, H = 144;
    int   src_w = 3840, src_h = 2160;
    float fx = 2700, fy = 2700, cx = 1915.7f, cy = 1075.1f;
    double ref_lon = 0, ref_lat = 0, ref_alt = 0;
    bool  ref_set = false;
    // 网格
    float ax0 = -700, ax1 = -400; int nax = 7;
    float ay0 = -120, ay1 = 80;   int nay = 5;
    float az0 = -40,  az1 = 60;   int naz = 3;
    float yo0 = 0,    yo1 = 315;  int nyo = 8;
    bool  use_grad = true;
    float blur_frac = 1.0f / 160.f;
    float min_alpha_radius = 1.5f;
    std::string best_align = "";
};

static void usage(const char* a0) {
    fprintf(stderr,
"用法: %s <ply> --query DIR --gt FILE --ref LON LAT ALT [选项]\n"
"  --ax0/--ax1/--nax     align_x 网格（默认 -700 -400 7）\n"
"  --ay0/--ay1/--nay     align_y 网格（默认 -120 80 5）\n"
"  --az0/--az1/--naz     align_z 网格（默认 -40 60 3）\n"
"  --yo0/--yo1/--nyo     yaw_offset 网格（默认 0 315 8）\n"
"  --w N --h N           渲染分辨率（默认 256x144）\n"
"  --nograd              关闭梯度域\n"
"  --dump DIR            保存最优渲染图\n"
"  --frame N             用第几帧（默认 0）\n", a0);
}

int main(int argc, char** argv) {
    Args A;
    int frame_idx = 0;
    for (int i = 1; i < argc; ++i) {
        auto eq = [&](const char* s) { return !strcmp(argv[i], s); };
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if (eq("--query")) A.query = next();
        else if (eq("--gt")) A.gt = next();
        else if (eq("--dump")) A.dump = next();
        else if (eq("--w")) A.W = atoi(next());
        else if (eq("--h")) A.H = atoi(next());
        else if (eq("--src-w")) A.src_w = atoi(next());
        else if (eq("--src-h")) A.src_h = atoi(next());
        else if (eq("--fx")) A.fx = (float)atof(next());
        else if (eq("--fy")) A.fy = (float)atof(next());
        else if (eq("--cx")) A.cx = (float)atof(next());
        else if (eq("--cy")) A.cy = (float)atof(next());
        else if (eq("--ref")) { A.ref_lon = atof(next()); A.ref_lat = atof(next()); A.ref_alt = atof(next()); A.ref_set = true; }
        else if (eq("--ax0")) A.ax0 = (float)atof(next());
        else if (eq("--ax1")) A.ax1 = (float)atof(next());
        else if (eq("--nax")) A.nax = atoi(next());
        else if (eq("--ay0")) A.ay0 = (float)atof(next());
        else if (eq("--ay1")) A.ay1 = (float)atof(next());
        else if (eq("--nay")) A.nay = atoi(next());
        else if (eq("--az0")) A.az0 = (float)atof(next());
        else if (eq("--az1")) A.az1 = (float)atof(next());
        else if (eq("--naz")) A.naz = atoi(next());
        else if (eq("--yo0")) A.yo0 = (float)atof(next());
        else if (eq("--yo1")) A.yo1 = (float)atof(next());
        else if (eq("--nyo")) A.nyo = atoi(next());
        else if (eq("--nograd")) A.use_grad = false;
        else if (eq("--frame")) frame_idx = atoi(next());
        else if (argv[i][0] != '-') A.ply = argv[i];
    }
    if (!A.ply || !A.query || !A.gt) { usage(argv[0]); return 2; }

    // 读真值
    std::vector<std::string> names; std::vector<double> gt(6 * 400, 0);
    int ngt = 0;
    {
        FILE* f = fopen(A.gt, "r");
        if (!f) { fprintf(stderr, "无法读真值 %s\n", A.gt); return 1; }
        char line[1024];
        while (fgets(line, sizeof(line), f) && ngt < 400) {
            char nm[256]; double v[6];
            if (sscanf(line, "%255s %lf %lf %lf %lf %lf %lf", nm, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) == 7) {
                names.push_back(nm);
                memcpy(&gt[ngt * 6], v, sizeof(v));
                ++ngt;
            }
        }
        fclose(f);
    }
    if (!ngt) { fprintf(stderr, "真值为空\n"); return 1; }
    if (frame_idx >= ngt) frame_idx = 0;
    const double* g = &gt[frame_idx * 6];
    if (!A.ref_set) { A.ref_lon = g[0]; A.ref_lat = g[1]; A.ref_alt = 0.0; }

    // 参考点 ECEF
    geopp::Vec3d ref_ecef = geopp::lla_to_ecef(A.ref_lon, A.ref_lat, A.ref_alt);

    // 读真照片
    std::string qpath = std::string(A.query) + "/" + names[frame_idx];
    cv::Mat q = cv::imread(qpath, cv::IMREAD_GRAYSCALE);
    if (q.empty()) { fprintf(stderr, "读图失败 %s\n", qpath.c_str()); return 1; }
    cv::Mat qs;
    cv::resize(q, qs, cv::Size(A.W, A.H), 0, 0, cv::INTER_AREA);
    cv::Mat qb;
    cv::GaussianBlur(qs, qb, cv::Size(0, 0), A.W * A.blur_frac);
    cv::Mat q_f;
    if (A.use_grad) q_f = grad_mag(qb);
    else            qb.convertTo(q_f, CV_32F);

    // 渲染器
    RenderConfig cfg; cfg.width = A.W; cfg.height = A.H;
    cfg.depth = true; cfg.min_alpha_radius = A.min_alpha_radius; cfg.verbose = false;
    Renderer r;
    if (!r.init(A.ply, cfg)) { fprintf(stderr, "渲染器初始化失败\n"); return 1; }

    const float sx = (float)A.W / A.src_w, sy = (float)A.H / A.src_h;
    Camera proto; proto.width = A.W; proto.height = A.H;
    proto.fx = A.fx * sx; proto.fy = A.fy * sy; proto.cx = A.cx * sx; proto.cy = A.cy * sy;
    proto.znear = 0.05f; proto.zfar = 20000.f;

    fprintf(stdout, "帧 %s  真值 lon=%.6f lat=%.6f alt=%.3f R/P/Y=%.2f/%.2f/%.2f\n",
            names[frame_idx].c_str(), g[0], g[1], g[2], g[3], g[4], g[5]);
    fprintf(stdout, "参考点 (%.6f, %.6f, %.2f)  渲染 %dx%d 网格 %dx%dx%dx%d = %d 组\n",
            A.ref_lon, A.ref_lat, A.ref_alt, A.W, A.H,
            A.nax, A.nay, A.naz, A.nyo, A.nax * A.nay * A.naz * A.nyo);

    struct Hit { double score, cover; float ax, ay, az, yo; };
    std::vector<Hit> hits;
    int done = 0;
    auto t0 = Clock::now();

    for (int ix = 0; ix < A.nax; ++ix) {
        const float ax = (A.nax == 1) ? A.ax0 : A.ax0 + (A.ax1 - A.ax0) * ix / (A.nax - 1);
        for (int iy = 0; iy < A.nay; ++iy) {
            const float ay = (A.nay == 1) ? A.ay0 : A.ay0 + (A.ay1 - A.ay0) * iy / (A.nay - 1);
            for (int iz = 0; iz < A.naz; ++iz) {
                const float az = (A.naz == 1) ? A.az0 : A.az0 + (A.az1 - A.az0) * iz / (A.naz - 1);
                for (int iyo = 0; iyo < A.nyo; ++iyo) {
                    const float yo = (A.nyo == 1) ? A.yo0 : A.yo0 + (A.yo1 - A.yo0) * iyo / (A.nyo - 1);

                    // 构造 c2w
                    geopp::PilotC2W pc = geopp::pilot_c2w_from_lla_euler(
                            g[0], g[1], g[2], g[3], g[4], g[5] + yo,
                            A.ref_lon, A.ref_lat, A.ref_alt, ref_ecef);
                    geopp::Mat3 Rw2c = geopp::transpose(pc.R);
                    // ⚠️ Mat4 是**列主序**（见 camera.h upper3x3: R[i][j]=m[j*4+i]），
                    //    所以行主序的 Rw2c.m[row][col] 要写到 m[col*4+row]。
                    Camera cam = proto;
                    cam.world_to_cam = Mat4::identity();
                    for (int rr = 0; rr < 3; ++rr)
                        for (int cc = 0; cc < 3; ++cc)
                            cam.world_to_cam.m[cc * 4 + rr] = (float)Rw2c.m[rr][cc];
                    const float Cx = (float)pc.t.x + ax;
                    const float Cy = (float)pc.t.y + ay;
                    const float Cz = (float)pc.t.z + az;
                    cam.world_to_cam.m[12] = -(float)(Rw2c.m[0][0]*Cx + Rw2c.m[0][1]*Cy + Rw2c.m[0][2]*Cz);
                    cam.world_to_cam.m[13] = -(float)(Rw2c.m[1][0]*Cx + Rw2c.m[1][1]*Cy + Rw2c.m[1][2]*Cz);
                    cam.world_to_cam.m[14] = -(float)(Rw2c.m[2][0]*Cx + Rw2c.m[2][1]*Cy + Rw2c.m[2][2]*Cz);

                    if (!r.render_frame(cam)) continue;
                    std::vector<uint8_t> rgba; std::vector<float> dr, dd;
                    r.get_frame(rgba);
                    const bool have_d = r.get_depth(dr);
                    if (have_d) flip_rows(dr, dd, A.W, A.H);

                    cv::Mat rg = to_gray(rgba, A.W, A.H);
                    cv::Mat rblur;
                    cv::GaussianBlur(rg, rblur, cv::Size(0, 0), A.W * A.blur_frac);
                    cv::Mat r_f;
                    if (A.use_grad) r_f = grad_mag(rblur);
                    else            rblur.convertTo(r_f, CV_32F);

                    // mask: 渲染有内容 且 深度有效
                    cv::Mat mask(A.H, A.W, CV_8U, cv::Scalar(0));
                    for (int y = 0; y < A.H; ++y) {
                        uint8_t* pm = mask.ptr<uint8_t>(y);
                        for (int x = 0; x < A.W; ++x) {
                            int alpha_ok = rgba[((size_t)(A.H - 1 - y) * A.W + x) * 4 + 3] > 10;
                            int depth_ok = !have_d || (dd[(size_t)y * A.W + x] > 0.01f);
                            pm[x] = (alpha_ok && depth_ok) ? 255 : 0;
                        }
                    }
                    double cover = 0;
                    double sc = zncc(q_f, r_f, mask, cover);
                    if (sc > -0.5 && cover >= 0.30) {
                        sc *= (0.5 + 0.5 * std::min(1.0, cover / 0.8));
                        hits.push_back({sc, cover, ax, ay, az, yo});
                    }
                    ++done;
                    if (done % 20 == 0) {
                        fprintf(stdout, "\r  进度 %d/%d  最好 %.4f   已用 %.0fs",
                                done, A.nax * A.nay * A.naz * A.nyo,
                                hits.empty() ? -1.0 : std::max_element(hits.begin(), hits.end(),
                                    [](const Hit&a, const Hit&b){return a.score<b.score;})->score,
                                ms_since(t0) / 1000.0);
                        fflush(stdout);
                    }
                }
            }
        }
    }
    fprintf(stdout, "\n完成 %d 组，耗时 %.0f s\n", done, ms_since(t0) / 1000.0);

    std::sort(hits.begin(), hits.end(), [](const Hit&a, const Hit&b){ return a.score > b.score; });
    fprintf(stdout, "\n%-9s %-7s %-9s %-9s %-9s %s\n",
            "ZNCC", "覆盖", "align_x", "align_y", "align_z", "yaw_off");
    fprintf(stdout, "──────────────────────────────────────────────────────────────\n");
    for (size_t i = 0; i < hits.size() && i < 25; ++i) {
        const auto& h = hits[i];
        fprintf(stdout, "%-9.4f %-7.3f %-9.1f %-9.1f %-9.1f %.1f\n",
                h.score, h.cover, h.ax, h.ay, h.az, h.yo);
    }
    if (hits.empty()) fprintf(stdout, "（无有效候选）\n");

    if (A.dump && !hits.empty()) {
        const auto& b = hits[0];
        geopp::PilotC2W pc = geopp::pilot_c2w_from_lla_euler(g[0], g[1], g[2], g[3], g[4], g[5] + b.yo,
                                               A.ref_lon, A.ref_lat, A.ref_alt, ref_ecef);
        geopp::Mat3 Rw2c = geopp::transpose(pc.R);
        Camera cam = proto; cam.world_to_cam = Mat4::identity();
        for (int rr = 0; rr < 3; ++rr) for (int cc = 0; cc < 3; ++cc)
            cam.world_to_cam.m[cc*4+rr] = (float)Rw2c.m[rr][cc];
        const float Cx = (float)pc.t.x + b.ax, Cy = (float)pc.t.y + b.ay, Cz = (float)pc.t.z + b.az;
        cam.world_to_cam.m[12] = -(float)(Rw2c.m[0][0]*Cx+Rw2c.m[0][1]*Cy+Rw2c.m[0][2]*Cz);
        cam.world_to_cam.m[13] = -(float)(Rw2c.m[1][0]*Cx+Rw2c.m[1][1]*Cy+Rw2c.m[1][2]*Cz);
        cam.world_to_cam.m[14] = -(float)(Rw2c.m[2][0]*Cx+Rw2c.m[2][1]*Cy+Rw2c.m[2][2]*Cz);
        if (r.render_frame(cam)) {
            std::vector<uint8_t> rgba; r.get_frame(rgba);
            cv::Mat rg = to_gray(rgba, A.W, A.H);
            char buf[1024];
            snprintf(buf, sizeof(buf), "%s/best_x%.0f_y%.0f_z%.0f_yo%.0f.png",
                     A.dump, b.ax, b.ay, b.az, b.yo);
            cv::imwrite(buf, rg);
            cv::imwrite(std::string(A.dump) + "/query.png", qs);
            fprintf(stdout, "最优渲染图 → %s\n", buf);
        }
    }
    return 0;
}
