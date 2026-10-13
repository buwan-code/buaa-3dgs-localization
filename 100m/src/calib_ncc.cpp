// calib_ncc.cpp — 用「归一化互相关 + 相位相关」标定 PLY↔ENU 对齐
//
// ══ 为什么要换指标 ════════════════════════════════════════════════
// 上一版用 ORB 匹配数当目标函数，实测**完全失效**：
//   扫了 12 个 yaw × 3 个高度，匹配数在 5~20 间无规律震荡，无峰。
// 原因（诊断）：
//   1) ORB 是**旋转+尺度敏感**的二进制描述子，渲染图与真照片的
//      尺度层次/光照/阴影差异会直接破坏描述子匹配
//   2) 匹配数还与「视野内纹理总量」正相关，与「对得准不准」弱相关
//      → 相机拉远 = 视野大 = 纹理多 = 匹配多，但可能完全没对准
//
// ══ 换成什么指标 ══════════════════════════════════════════════════
// 用**直接像素比较**，它对「内容是否对应」敏感、对细节差异鲁棒：
//   * 先对两图做高斯模糊（模拟降采样，抹掉细节层次差异）
//   * 计算 ZNCC（零均值归一化互相关），范围 [-1, 1]
//   * 只统计两图都有效的像素（渲染图的黑色空洞要排除）
//   * 用**互信息**作为辅助指标（对光照变化最鲁棒）
//
// 这样得到的是一个**有明确峰值的相似度曲面**，才能用来搜索。
//
// ══ 搜索维度 ══════════════════════════════════════════════════════
// 真照片是近垂直俯视（实测 pitch≈88°），所以：
//   水平位置 (x, y)   —— 相机在 PLY 系的水平坐标
//   高度 z            —— 决定地面采样距离
//   偏航 yaw          —— 图像旋转角
//   （俯仰/横滚固定为近垂直，不搜）
//
// 输出：最优 (x, y, z, yaw) 及对应的 ZNCC 值，并导出最优渲染图供肉眼核验。

#include "renderer.h"
#include "camera.h"
#include "ply_loader.h"
#include "geo_pose.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>

using namespace gsplat;

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}
using Clock = std::chrono::steady_clock;

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

static void make_cam(Camera& c, const Vec3& C, const Vec3& fwd, const Vec3& up) {
    auto norm = [](Vec3 v) {
        float n = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
        if (n < 1e-12f) n = 1.f;
        return Vec3{v.x/n, v.y/n, v.z/n};
    };
    auto cross = [](const Vec3& a, const Vec3& b) {
        return Vec3{a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
    };
    const Vec3 zc = norm(fwd);
    Vec3 xc = norm(cross(up, zc));
    if (std::fabs(xc.x) + std::fabs(xc.y) + std::fabs(xc.z) < 1e-6f)
        xc = norm(cross(Vec3{1.f, 0.f, 0.f}, zc));
    const Vec3 yc = cross(zc, xc);
    c.world_to_cam = Mat4::identity();
    c.world_to_cam.m[0]=xc.x; c.world_to_cam.m[4]=xc.y; c.world_to_cam.m[8] =xc.z;
    c.world_to_cam.m[1]=yc.x; c.world_to_cam.m[5]=yc.y; c.world_to_cam.m[9] =yc.z;
    c.world_to_cam.m[2]=zc.x; c.world_to_cam.m[6]=zc.y; c.world_to_cam.m[10]=zc.z;
    c.world_to_cam.m[12] = -(float)(xc.x*C.x + xc.y*C.y + xc.z*C.z);
    c.world_to_cam.m[13] = -(float)(yc.x*C.x + yc.y*C.y + yc.z*C.z);
    c.world_to_cam.m[14] = -(float)(zc.x*C.x + zc.y*C.y + zc.z*C.z);
}

// ── ZNCC：零均值归一化互相关 ──────────────────────────────────────
// 只统计 mask 内（两图都有效）的像素。范围 [-1,1]，1=完全一致。
static double zncc(const cv::Mat& a, const cv::Mat& b, const cv::Mat& mask) {
    const int n = cv::countNonZero(mask);
    if (n < 500) return -1.0;
    const double ma = cv::mean(a, mask)[0];
    const double mb = cv::mean(b, mask)[0];
    double num = 0, da = 0, db = 0;
    for (int y = 0; y < a.rows; ++y) {
        const uint8_t* pa = a.ptr<uint8_t>(y);
        const uint8_t* pb = b.ptr<uint8_t>(y);
        const uint8_t* pm = mask.ptr<uint8_t>(y);
        for (int x = 0; x < a.cols; ++x) {
            if (!pm[x]) continue;
            const double va = pa[x] - ma, vb = pb[x] - mb;
            num += va * vb; da += va * va; db += vb * vb;
        }
    }
    const double den = std::sqrt(da * db);
    return den < 1e-9 ? -1.0 : num / den;
}

// ── 梯度 ZNCC：对结构（边缘/轮廓）比对更敏感 ──────────────────────
// 灰度 ZNCC 会被大片同色区域（草地/屋顶）主导；
// 用 Sobel 梯度幅值做 ZNCC，让「楼边、路沿」这些结构特征起主导作用。
static cv::Mat grad_mag(const cv::Mat& g) {
    cv::Mat gx, gy, mag;
    cv::Sobel(g, gx, CV_32F, 1, 0, 3);
    cv::Sobel(g, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);
    cv::Mat out;
    mag.convertTo(out, CV_8U, 255.0 / 800.0);   // 截断到 0~255
    return out;
}

int main(int argc, char** argv) {
    const char* ply   = nullptr;
    const char* query = nullptr;
    const char* gt    = nullptr;
    const char* out_dir = nullptr;
    int   W = 512, H = 288;
    float fx = 2700, fy = 2700, cx = 1915.7f, cy = 1075.1f;
    int   src_w = 3840, src_h = 2160;
    float pitch = 88.f;
    // 搜索网格
    float gx0=-800, gx1=-360, gy0=-160, gy1=120;   // 水平
    float gz0=90,  gz1=170;                        // 高度
    float yaw0=-180, yaw1=180;                     // 偏航
    float step_xy = 40.f, step_z = 20.f, step_yaw = 15.f;
    bool  use_grad = true;
    float blur = 0.f;          // 渲染图与照片的高斯模糊 sigma（0=自动）
    float min_alpha_radius = 0.5f;

    for (int i = 1; i < argc; ++i) {
        auto eq = [&](const char* s){ return !strcmp(argv[i], s); };
        auto nx = [&](){ return (i+1<argc)? argv[++i] : nullptr; };
        if (eq("--query")) query = nx();
        else if (eq("--gt")) gt = nx();
        else if (eq("--dump")) out_dir = nx();
        else if (eq("--w")) W = atoi(nx());
        else if (eq("--h")) H = atoi(nx());
        else if (eq("--fx")) fx = (float)atof(nx());
        else if (eq("--fy")) fy = (float)atof(nx());
        else if (eq("--cx")) cx = (float)atof(nx());
        else if (eq("--cy")) cy = (float)atof(nx());
        else if (eq("--src-w")) src_w = atoi(nx());
        else if (eq("--src-h")) src_h = atoi(nx());
        else if (eq("--pitch")) pitch = (float)atof(nx());
        else if (eq("--gx0")) gx0 = (float)atof(nx());
        else if (eq("--gx1")) gx1 = (float)atof(nx());
        else if (eq("--gy0")) gy0 = (float)atof(nx());
        else if (eq("--gy1")) gy1 = (float)atof(nx());
        else if (eq("--gz0")) gz0 = (float)atof(nx());
        else if (eq("--gz1")) gz1 = (float)atof(nx());
        else if (eq("--yaw0")) yaw0 = (float)atof(nx());
        else if (eq("--yaw1")) yaw1 = (float)atof(nx());
        else if (eq("--step-xy")) step_xy = (float)atof(nx());
        else if (eq("--step-z"))  step_z  = (float)atof(nx());
        else if (eq("--step-yaw")) step_yaw = (float)atof(nx());
        else if (eq("--gray")) use_grad = false;
        else if (eq("--blur")) blur = (float)atof(nx());
        else if (eq("--min-alpha-radius")) min_alpha_radius = (float)atof(nx());
        else if (argv[i][0] != '-') ply = argv[i];
    }
    if (!ply || !query || !gt) {
        fprintf(stderr,
          "用法: %s <ply> --query DIR --gt FILE [选项]\n"
          "  搜索维度: 水平(x,y) × 高度z × 偏航yaw\n"
          "  --gx0/gx1 --gy0/gy1 --gz0/gz1 --yaw0/yaw1\n"
          "  --step-xy --step-z --step-yaw\n"
          "  --gray        用灰度ZNCC（默认用梯度ZNCC）\n"
          "  --blur SIGMA  预模糊（自动取图像宽度的1/160）\n"
          "  --dump DIR    导出最优渲染图\n",
          argv[0]);
        return 2;
    }
    if (blur <= 0.f) blur = W / 160.f;

    // ── 读真值第一帧 ──────────────────────────────────────────────
    FILE* fg = fopen(gt, "r");
    if (!fg) { fprintf(stderr, "打不开真值\n"); return 1; }
    char gname[256]; double glon, glat, galt, ga1, ga2, ga3;
    if (fscanf(fg, "%255s %lf %lf %lf %lf %lf %lf",
               gname, &glon, &glat, &galt, &ga1, &ga2, &ga3) != 7) {
        fprintf(stderr, "真值格式不对\n"); fclose(fg); return 1;
    }
    fclose(fg);

    const std::string img_path = std::string(query) + "/" + gname;
    cv::Mat src = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    if (src.empty()) { fprintf(stderr, "读照片失败\n"); return 1; }
    cv::Mat qgray;
    cv::resize(src, qgray, cv::Size(W, H), 0, 0, cv::INTER_AREA);
    cv::Mat qproc = use_grad ? grad_mag(qgray) : qgray.clone();

    printf("════════ NCC 标定（PLY ↔ ENU）════════\n");
    printf("参考照片  : %s\n", gname);
    printf("真值      : %.6f %.6f %.3f  euler %.1f %.1f %.1f\n",
           glon, glat, galt, ga1, ga2, ga3);
    printf("指标      : %s ZNCC   预模糊 sigma=%.2f\n",
           use_grad ? "梯度" : "灰度", blur);
    printf("俯仰      : %.1f° (近垂直俯视)\n", pitch);

    const int nx_ = (int)((gx1-gx0)/step_xy)+1;
    const int ny_ = (int)((gy1-gy0)/step_xy)+1;
    const int nz_ = (int)((gz1-gz0)/step_z)+1;
    const int nyaw= (int)((yaw1-yaw0)/step_yaw)+1;
    const long total = (long)nx_*ny_*nz_*nyaw;
    printf("搜索      : x%d × y%d × z%d × yaw%d = %ld 个候选\n\n",
           nx_,ny_,nz_,nyaw, total);

    // ── 加载 ──────────────────────────────────────────────────────
    std::vector<Splat>    splats;
    std::vector<ShCoeffs> shcs;
    if (!load_ply_splats(ply, splats, shcs, 1, false)) {
        fprintf(stderr, "PLY 加载失败\n"); return 1;
    }
    float mn[3]={1e30f,1e30f,1e30f}, mx[3]={-1e30f,-1e30f,-1e30f};
    for (const auto& s : splats) {
        float p[3]={s.x,s.y,s.z};
        for (int k=0;k<3;++k){ if(p[k]<mn[k])mn[k]=p[k]; if(p[k]>mx[k])mx[k]=p[k]; }
    }
    printf("PLY 盒    : min(%.0f %.0f %.0f) max(%.0f %.0f %.0f)\n\n",
           mn[0],mn[1],mn[2],mx[0],mx[1],mx[2]);

    RenderConfig cfg;
    cfg.width=W; cfg.height=H; cfg.depth=false;
    cfg.min_alpha_radius = min_alpha_radius;
    cfg.verbose=false;
    Renderer r;
    if (!r.init(ply, cfg)) { fprintf(stderr,"渲染器初始化失败\n"); return 1; }

    const float sx=(float)W/(float)src_w, sy=(float)H/(float)src_h;
    Camera proto;
    proto.width=W; proto.height=H;
    proto.fx=fx*sx; proto.fy=fy*sy; proto.cx=cx*sx; proto.cy=cy*sy;
    proto.znear=0.05f; proto.zfar=1000.f;

    struct Hit { double score; float x,y,z,yaw; };
    std::vector<Hit> hits;

    auto t_all = Clock::now();
    long done = 0;

    for (int iy = 0; iy < ny_; ++iy) {
        for (int ix = 0; ix < nx_; ++ix) {
            for (int iz = 0; iz < nz_; ++iz) {
                for (int iw = 0; iw < nyaw; ++iw) {
                    const float cx_ = gx0 + ix*step_xy;
                    const float cy_ = gy0 + iy*step_xy;
                    const float cz_ = gz0 + iz*step_z;
                    const float yaw_ = yaw0 + iw*step_yaw;
                    ++done;

                    // 光轴：俯仰 pitch，方位 yaw
                    const double yr = yaw_ * M_PI/180.0;
                    const double pr = pitch * M_PI/180.0;
                    const double ch = std::cos(pr);
                    const Vec3 fwd{ (float)(ch*std::sin(yr)),
                                    (float)(ch*std::cos(yr)),
                                    (float)(-std::sin(pr)) };
                    Camera cam = proto;
                    make_cam(cam, Vec3{cx_,cy_,cz_}, fwd, Vec3{0.f,0.f,1.f});
                    cam.fx=proto.fx; cam.fy=proto.fy; cam.cx=proto.cx; cam.cy=proto.cy;
                    cam.znear=proto.znear; cam.zfar=proto.zfar;

                    double score = -1;
                    double cover = 0;
                    if (r.render_frame(cam)) {
                        std::vector<uint8_t> rgba;
                        r.get_frame(rgba);
                        cv::Mat g = to_gray(rgba, W, H);
                        // 有效掩码：渲染图非黑（有高斯覆盖）
                        cv::Mat mask;
                        cv::threshold(g, mask, 6, 255, cv::THRESH_BINARY);
                        // ★ 覆盖率约束：有效像素太少时，"局部蒙对高分"会污染结果
                        //   实测教训：(-670,-120,170) 渲染图几乎全黑（只有左上角
                        //   一小块），ZNCC 却给 0.52 —— 因为只对那一小块算相关。
                        //   故覆盖率 < 35% 直接判为无效。
                        cover = (double)cv::countNonZero(mask) / (double)(W*H);
                        if (cover >= 0.35) {
                            cv::Mat ga_, gb;
                            cv::GaussianBlur(g,   ga_, cv::Size(0,0), blur);
                            cv::GaussianBlur(qgray, gb, cv::Size(0,0), blur);
                            cv::Mat pa = use_grad ? grad_mag(ga_) : ga_;
                            cv::Mat pb = use_grad ? grad_mag(gb) : gb;
                            score = zncc(pa, pb, mask);
                            // 用覆盖率加权，偏好"视野填满且相关高"的解
                            score *= (0.5 + 0.5 * std::min(1.0, cover / 0.8));
                        }
                    }
                    hits.push_back({score, cx_, cy_, cz_, yaw_});
                }
            }
        }
        printf("\r  %.0f%% (%ld/%ld)  当前最优 %.4f     ",
               100.0*done/total, done, total,
               hits.empty()?-1:std::max_element(hits.begin(),hits.end(),
                   [](const Hit&a,const Hit&b){return a.score<b.score;})->score);
        fflush(stdout);
    }
    printf("\n\n总耗时 %.1f s\n\n", ms_since(t_all)/1000.0);

    std::sort(hits.begin(), hits.end(),
              [](const Hit&a,const Hit&b){ return a.score > b.score; });
    printf("★ ZNCC 最高的 15 个候选：\n");
    printf("%-5s %-9s %-9s %-9s %-9s %-9s\n","排名","x","y","z","yaw","ZNCC");
    printf("──────────────────────────────────────────────────\n");
    for (int i = 0; i < 15 && i < (int)hits.size(); ++i)
        printf("%-5d %-9.0f %-9.0f %-9.0f %-9.0f %-9.4f\n",
               i+1, hits[i].x, hits[i].y, hits[i].z, hits[i].yaw, hits[i].score);

    // ── 导出最优 ──────────────────────────────────────────────────
    if (out_dir && !hits.empty()) {
        const Hit& b = hits[0];
        const double yr = b.yaw * M_PI/180.0, pr = pitch * M_PI/180.0;
        const double ch = std::cos(pr);
        const Vec3 fwd{ (float)(ch*std::sin(yr)), (float)(ch*std::cos(yr)),
                        (float)(-std::sin(pr)) };
        Camera cam = proto;
        make_cam(cam, Vec3{b.x,b.y,b.z}, fwd, Vec3{0.f,0.f,1.f});
        cam.fx=proto.fx; cam.fy=proto.fy; cam.cx=proto.cx; cam.cy=proto.cy;
        cam.znear=proto.znear; cam.zfar=proto.zfar;
        if (r.render_frame(cam)) {
            std::vector<uint8_t> rgba;
            r.get_frame(rgba);
            cv::Mat g = to_gray(rgba, W, H);
            char buf[1024];
            snprintf(buf,sizeof(buf),"%s/best_ref.png",out_dir);
            cv::imwrite(buf, g);
            snprintf(buf,sizeof(buf),"%s/query.png",out_dir);
            cv::imwrite(buf, qgray);
            printf("\n已导出: %s/best_ref.png  %s/query.png\n", out_dir, out_dir);
        }
    }
    return 0;
}
