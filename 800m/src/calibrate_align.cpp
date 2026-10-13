// calibrate_align.cpp — 标定「PLY 局部系 ↔ ENU 系」的刚体变换
//
// ══ 为什么需要标定 ═════════════════════════════════════════════════
// 事实（实测）：
//   * PLY (pilot.ply) 包围盒 span(188.0, 154.6, 45.7) center(-580.4,-22.4,40.8)
//     → 局部米制，原点由 3DGS 训练器自选（通常是相机质心附近）
//   * 真值 3dgs_test.txt 是 WGS84 经纬高，转 ENU 后轨迹 span(142.4, 56.0, 0.1)
//     → 尺度与 PLY 同量级（同场地），但**原点不同**
//   * 两者差一个刚体变换：X_ply = R·X_enu + t
//
// ══ 标定方法（不解方程，直接搜）═══════════════════════════════════
// 用「渲染-匹配度」当目标函数：
//   对候选的 (R, t)，把真值位姿映射进 PLY 系 → 渲染 → ORB 匹配真照片
//   匹配数（或内点数）最高的 (R, t) 就是对的
//
// 搜索策略（分两步，避免组合爆炸）：
//   Step 1  固定 R = I（先假设两系轴向一致，只差平移）
//           在轨迹中心的邻域里网格搜索 t（水平 ±600m，垂直 0~250m）
//           目标函数 = 单帧 ORB 匹配数
//   Step 2  在 Step 1 最优 t 附近做旋转精修（可选，本轮先做 Step 1）
//
// 为什么假设 R=I 是合理的：
//   3DGS 训练器（gaussian-splatting 官方实现）输出的世界系，
//   其 X/Y 轴由 COLMAP 的初始相机对决定，**通常近似对齐 ENU 的东/北**，
//   Z 轴朝上（因为 COLMAP 用 +Z 向上）。所以 R ≈ I 是很好的先验。
//   若 Step 1 找到高分但匹配数仍低，再启用旋转搜索。

#include "renderer.h"
#include "camera.h"
#include "ply_loader.h"
#include "geo_pose.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
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

// 相机基构造（同 gs_locate_uav）
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
    // 退化保护：up 与 fwd 平行时换一个 up
    if (std::fabs(xc.x) + std::fabs(xc.y) + std::fabs(xc.z) < 1e-6f) {
        xc = norm(cross(Vec3{1.f, 0.f, 0.f}, zc));
    }
    const Vec3 yc = cross(zc, xc);
    c.world_to_cam = Mat4::identity();
    c.world_to_cam.m[0]=xc.x; c.world_to_cam.m[4]=xc.y; c.world_to_cam.m[8] =xc.z;
    c.world_to_cam.m[1]=yc.x; c.world_to_cam.m[5]=yc.y; c.world_to_cam.m[9] =yc.z;
    c.world_to_cam.m[2]=zc.x; c.world_to_cam.m[6]=zc.y; c.world_to_cam.m[10]=zc.z;
    c.world_to_cam.m[12] = -(float)(xc.x*C.x + xc.y*C.y + xc.z*C.z);
    c.world_to_cam.m[13] = -(float)(yc.x*C.x + yc.y*C.y + yc.z*C.z);
    c.world_to_cam.m[14] = -(float)(zc.x*C.x + zc.y*C.y + zc.z*C.z);
}

int main(int argc, char** argv) {
    // ── 参数 ──────────────────────────────────────────────────────
    const char* ply   = nullptr;
    const char* query = nullptr;
    const char* gt    = nullptr;
    int   W = 512, H = 288;
    float fx = 2700, fy = 2700, cx = 1915.7f, cy = 1075.1f;
    int   src_w = 3840, src_h = 2160;
    // 搜索范围（PLY 系里的相机位置）
    float gx0=-1200, gx1=200, gy0=-600, gy1=800;   // 水平网格
    float gz0=40, gz1=260;                          // 垂直网格
    float step_xy = 50.f, step_z = 25.f;
    float min_alpha_radius = 0.5f;
    int   topk = 8;
    const char* out_csv = nullptr;

    for (int i = 1; i < argc; ++i) {
        auto eq = [&](const char* s){ return !strcmp(argv[i], s); };
        auto nx = [&](){ return (i+1<argc)? argv[++i] : nullptr; };
        if (eq("--query")) query = nx();
        else if (eq("--gt")) gt = nx();
        else if (eq("--w")) W = atoi(nx());
        else if (eq("--h")) H = atoi(nx());
        else if (eq("--fx")) fx = (float)atof(nx());
        else if (eq("--fy")) fy = (float)atof(nx());
        else if (eq("--cx")) cx = (float)atof(nx());
        else if (eq("--cy")) cy = (float)atof(nx());
        else if (eq("--src-w")) src_w = atoi(nx());
        else if (eq("--src-h")) src_h = atoi(nx());
        else if (eq("--gx0")) gx0 = (float)atof(nx());
        else if (eq("--gx1")) gx1 = (float)atof(nx());
        else if (eq("--gy0")) gy0 = (float)atof(nx());
        else if (eq("--gy1")) gy1 = (float)atof(nx());
        else if (eq("--gz0")) gz0 = (float)atof(nx());
        else if (eq("--gz1")) gz1 = (float)atof(nx());
        else if (eq("--step-xy")) step_xy = (float)atof(nx());
        else if (eq("--step-z"))  step_z  = (float)atof(nx());
        else if (eq("--min-alpha-radius")) min_alpha_radius = (float)atof(nx());
        else if (eq("--topk")) topk = atoi(nx());
        else if (eq("--out")) out_csv = nx();
        else if (argv[i][0] != '-') ply = argv[i];
    }
    if (!ply || !query || !gt) {
        fprintf(stderr,
            "用法: %s <ply> --query DIR --gt FILE [选项]\n"
            "  --gx0/gx1/gy0/gy1/gz0/gz1   相机位置搜索范围(PLY系)\n"
            "  --step-xy / --step-z        网格步长(默认 50 / 25 m)\n"
            "  --topk N                    报告前 N 个最优候选\n"
            "  --out CSV                   全部打分写 CSV\n",
            argv[0]);
        return 2;
    }

    // ── 读真值第一帧 ──────────────────────────────────────────────
    FILE* fg = fopen(gt, "r");
    if (!fg) { fprintf(stderr, "打不开真值: %s\n", gt); return 1; }
    char gname[256]; double glon, glat, galt, ga1, ga2, ga3;
    if (fscanf(fg, "%255s %lf %lf %lf %lf %lf %lf",
               gname, &glon, &glat, &galt, &ga1, &ga2, &ga3) != 7) {
        fprintf(stderr, "真值格式不对\n"); fclose(fg); return 1;
    }
    fclose(fg);

    const std::string img_path = std::string(query) + "/" + gname;
    cv::Mat src = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    if (src.empty()) { fprintf(stderr, "读照片失败: %s\n", img_path.c_str()); return 1; }
    cv::Mat qgray;
    cv::resize(src, qgray, cv::Size(W, H), 0, 0, cv::INTER_AREA);

    cv::Ptr<cv::ORB> orb = cv::ORB::create(4000);
    std::vector<cv::KeyPoint> kp_q;
    cv::Mat desc_q;
    orb->detectAndCompute(qgray, cv::noArray(), kp_q, desc_q);

    printf("════════ PLY ↔ ENU 标定 ════════\n");
    printf("参考照片  : %s  (%zu 特征)\n", gname, kp_q.size());
    printf("真值经纬高: %.6f %.6f %.3f\n", glon, glat, galt);
    printf("真值欧拉角: %.3f %.3f %.3f\n", ga1, ga2, ga3);
    printf("搜索范围  : x[%.0f,%.0f] y[%.0f,%.0f] z[%.0f,%.0f]  step %.0f/%.0f\n",
           gx0,gx1,gy0,gy1,gz0,gz1,step_xy,step_z);
    const int nx_ = (int)((gx1-gx0)/step_xy)+1;
    const int ny_ = (int)((gy1-gy0)/step_xy)+1;
    const int nz_ = (int)((gz1-gz0)/step_z)+1;
    printf("网格规模  : %d × %d × %d = %d 个候选\n\n", nx_, ny_, nz_, nx_*ny_*nz_);

    // ── 加载模型 ──────────────────────────────────────────────────
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
    printf("PLY 包围盒: min(%.1f %.1f %.1f) max(%.1f %.1f %.1f)\n",
           mn[0],mn[1],mn[2],mx[0],mx[1],mx[2]);

    RenderConfig cfg;
    cfg.width = W; cfg.height = H; cfg.depth = true;
    cfg.min_alpha_radius = min_alpha_radius;
    cfg.verbose = false;
    Renderer r;
    if (!r.init(ply, cfg)) { fprintf(stderr, "渲染器初始化失败\n"); return 1; }

    const float sx = (float)W/(float)src_w, sy = (float)H/(float)src_h;
    Camera proto;
    proto.width=W; proto.height=H;
    proto.fx=fx*sx; proto.fy=fy*sy; proto.cx=cx*sx; proto.cy=cy*sy;
    proto.znear=0.05f; proto.zfar=800.f;

    // ── 策略：把「真值 ENU 坐标 + 候选原点偏移」当相机位置 ────────
    // 更简洁的等价做法：直接遍历「相机在 PLY 系的位置」，
    // 因为 ENU 轨迹的**相对形状**已知（从真值算），我们只需要找
    // 「轨迹放在 PLY 系的哪里」。轨迹的相对偏移用 ENU 值直接加。
    //
    // 即：候选 = 轨迹锚点 A（网格搜索） + 本帧在 ENU 里的相对位移
    geopp::GeoRef geo; geo.mode = geopp::GeoRef::ENU;
    geo.ref_lon = glon; geo.ref_lat = glat; geo.ref_alt = galt;
    geo.prepare_enu();

    // 本帧在 ENU 里的相对位置（相对第一帧，此处第一帧=自身，所以是 0）
    // 若以后要多帧联合标定，这里换成真实相对位移
    geopp::Vec3d self = geo.to_local(glon, glat, galt);
    printf("本帧 ENU 相对位置: (%.3f, %.3f, %.3f)\n\n", self.x, self.y, self.z);

    cv::BFMatcher matcher(cv::NORM_HAMMING);

    struct Hit { int nm; float x,y,z; double ms; };
    std::vector<Hit> hits;
    FILE* fc = out_csv ? fopen(out_csv, "w") : nullptr;
    if (fc) fprintf(fc, "x,y,z,nmatch,ms\n");

    int total = nx_*ny_*nz_, done = 0;
    auto t_all = Clock::now();

    for (int iz = 0; iz < nz_; ++iz) {
        const float cz = gz0 + iz*step_z;
        for (int iy = 0; iy < ny_; ++iy) {
            const float cyy = gy0 + iy*step_xy;
            for (int ix = 0; ix < nx_; ++ix) {
                const float cxx = gx0 + ix*step_xy;
                ++done;

                // 相机位置（PLY 系）。光轴用真值欧拉角（关键修正！）
                // 真值实测: a1=0, a2=35.6(俯仰), a3=108.9→235.6(方位)
                // 早期版本用固定 fwd=(0,0.35,-1)，导致打分面没有判别力
                // （匹配数只反映「视野里有多少纹理」，不反映「对得准不准」）。
                const Vec3 C{ cxx, cyy, cz };
                const double yr = ga3 * M_PI / 180.0;
                const double pr = ga2 * M_PI / 180.0;
                const double ch = std::cos(pr);
                Vec3 fwd{ (float)(ch*std::sin(yr)),
                          (float)(ch*std::cos(yr)),
                          (float)(-std::sin(pr)) };

                Camera cam = proto;
                make_cam(cam, C, fwd, Vec3{0.f, 0.f, 1.f});
                cam.fx=proto.fx; cam.fy=proto.fy; cam.cx=proto.cx; cam.cy=proto.cy;
                cam.znear=proto.znear; cam.zfar=proto.zfar;

                auto t0 = Clock::now();
                int nm = 0;
                if (r.render_frame(cam)) {
                    std::vector<uint8_t> rrgba;
                    r.get_frame(rrgba);
                    cv::Mat rgray = to_gray(rrgba, W, H);
                    std::vector<cv::KeyPoint> kp_r;
                    cv::Mat desc_r;
                    orb->detectAndCompute(rgray, cv::noArray(), kp_r, desc_r);
                    if (!desc_r.empty() && !desc_q.empty()) {
                        std::vector<std::vector<cv::DMatch>> knn;
                        matcher.knnMatch(desc_q, desc_r, knn, 2);
                        for (const auto& v : knn) {
                            if (v.size() < 2) continue;
                            if (v[0].distance > 0.75f * v[1].distance) continue;
                            ++nm;
                        }
                    }
                }
                const double ms = ms_since(t0);
                if (fc) fprintf(fc, "%.0f,%.0f,%.0f,%d,%.1f\n", cxx, cyy, cz, nm, ms);
                hits.push_back({nm, cxx, cyy, cz, ms});

                if (done % 20 == 0 || done == total) {
                    printf("\r  进度 %d/%d (%.0f%%)  当前最优 %d",
                           done, total, 100.0*done/total,
                           hits.empty()?0:std::max_element(hits.begin(),hits.end(),
                               [](const Hit&a,const Hit&b){return a.nm<b.nm;})->nm);
                    fflush(stdout);
                }
            }
        }
    }
    printf("\n\n总耗时 %.1f s\n\n", ms_since(t_all)/1000.0);
    if (fc) fclose(fc);

    // ── 报告 top-K ────────────────────────────────────────────────
    std::sort(hits.begin(), hits.end(),
              [](const Hit&a, const Hit&b){ return a.nm > b.nm; });
    printf("★ 匹配数最高的 %d 个候选：\n", topk);
    printf("%-6s %-10s %-10s %-10s %-8s\n", "排名", "x(PLY)", "y(PLY)", "z(PLY)", "匹配数");
    printf("──────────────────────────────────────────────────\n");
    for (int i = 0; i < topk && i < (int)hits.size(); ++i) {
        printf("%-6d %-10.0f %-10.0f %-10.0f %-8d\n",
               i+1, hits[i].x, hits[i].y, hits[i].z, hits[i].nm);
    }
    // 统计：匹配数为 0 的比例
    int zero = 0, gt50 = 0, gt200 = 0;
    for (const auto& h : hits) {
        if (h.nm == 0) ++zero;
        if (h.nm >= 50) ++gt50;
        if (h.nm >= 200) ++gt200;
    }
    printf("\n分布: 全零 %d/%d (%.0f%%)  |  >=50 %d  |  >=200 %d\n",
           zero, (int)hits.size(), 100.0*zero/hits.size(), gt50, gt200);

    if (!hits.empty() && hits[0].nm < 50) {
        printf("\n⚠️ 最高匹配数仅 %d —— 说明「轴向一致(R=I)」的假设可能不成立，\n"
               "   或者搜索范围没覆盖到正确位置。下一步应启用旋转搜索。\n",
               hits[0].nm);
    }
    return 0;
}
