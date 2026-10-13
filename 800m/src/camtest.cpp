// camtest.cpp — EWA 投影数学的正确性校验（M2 第一步）
//
// 为什么先写这个：EWA 投影是整个 3DGS 渲染里最容易错、也最难 debug 的
// 一段数学（协方差三次矩阵乘 + 雅可比）。如果直接上 GPU，画出来不对时
// 你无法区分是「数学错了」还是「shader 错了」还是「坐标约定错了」。
// 所以先用**解析解可验证的构造用例**把 CPU 侧钉死：
//   ① 单位球高斯 → 各向同性圆，轴长 = fx/z * σ
//   ② 各向异性轴对齐高斯 → 椭圆长短轴比例 = σx:σy
//   ③ 四元数旋转 → 椭圆朝向随之旋转（验证 R 的作用）
//   ④ 透视效应 → 同一高斯在画面边缘应被拉长（验证雅可比）
//   ⑤ 退化工况 → 相机后方/超大高斯应被剔除
//
// 用法：./build/camtest
#include "camera.h"
#include <cstdio>
#include <cmath>

using namespace gsplat;

static int g_pass = 0, g_fail = 0;

static void check(const char* name, float got, float want, float tol) {
    const float err = std::fabs(got - want);
    const bool ok = err <= tol;
    if (ok) ++g_pass; else ++g_fail;
    fprintf(stdout, "  %-46s got=%10.4f want=%10.4f tol=%.4f  %s\n",
            name, got, want, tol, ok ? "✅" : "❌");
}

static void check_bool(const char* name, bool got, bool want) {
    const bool ok = (got == want);
    if (ok) ++g_pass; else ++g_fail;
    fprintf(stdout, "  %-46s got=%-6s want=%-6s %s\n",
            name, got ? "true" : "false", want ? "true" : "false",
            ok ? "✅" : "❌");
}

int main() {
    fprintf(stdout, "=====================================================\n");
    fprintf(stdout, " EWA 投影数学校验\n");
    fprintf(stdout, "=====================================================\n");

    // 统一用一个「正对原点、距离 10」的相机
    // eye=(0,0,-10), target=(0,0,0), up=(0,-1,0)
    // 选这个 up 是为了让相机坐标系的 y 轴朝下（OpenCV 约定），
    // 这样 target 在原点、相机在 -z 方向看向 +z，视线沿 +z 前进。
    const Camera cam = Camera::look_at(
        Vec3{0.f, 0.f, -10.f},   // eye
        Vec3{0.f, 0.f,  0.f},    // target
        Vec3{0.f, -1.f, 0.f},    // up（y 轴朝下）
        640, 480, 45.f);

    fprintf(stdout, "\n相机内参: fx=%.2f fy=%.2f cx=%.1f cy=%.1f\n",
            cam.fx, cam.fy, cam.cx, cam.cy);

    // ── ① 原点处的单位球高斯 ────────────────────────────────────────
    // 数学期望：
    //   视空间 z = 10
    //   屏幕中心 = (cx, cy) = (320, 240)
    //   Σ_2D = diag((fx/z)²σ², (fy/z)²σ²) = diag((fx*σ/z)², ...)
    //   → 轴长 = fx*σ/z = fx/10
    fprintf(stdout, "\n【1】单位球高斯（σ=1）在原点，z=10\n");
    {
        Splat2D s = project_gaussian(Vec3{0,0,0}, Vec3{1,1,1},
                                     1,0,0,0, cam);
        check_bool("valid", s.valid, true);
        check("屏幕中心 x", s.cx, cam.cx, 0.5f);
        check("屏幕中心 y", s.cy, cam.cy, 0.5f);
        check("depth", s.depth, 10.f, 0.01f);

        float maj = 0, mino = 0, ang = 0;
        cov2d_to_ellipse(s, &maj, &mino, &ang);
        // 期望：sqrt(a) 其中 a = (fx/10)²*1² + 0.3（低通项）
        const float expect = std::sqrt(cam.fx*cam.fx/(100.f) + 0.3f);
        check("主轴半长 = sqrt((fx/z)²σ²+0.3)", maj, expect, 0.01f);
        check("长短轴之比 = 1（各向同性）", maj / mino, 1.f, 0.002f);
    }

    // ── ② 各向异性：x 方向拉长 3 倍 ─────────────────────────────────
    // ⚠️ σ 必须取真实 3DGS 尺度（0.01–0.3 世界单位），不能用 3。
    //    用 σ=3 在 z=10 处会得到 3σ 半径 ≈ 492 px，被"巨型高斯"上限剔除 ——
    //    那是正确行为，但会让本测试测不到比例关系。
    fprintf(stdout, "\n【2】各向异性 σ=(0.3,0.1,0.1)，应得 3:1 椭圆且长轴沿 x\n");
    {
        Splat2D s = project_gaussian(Vec3{0,0,0}, Vec3{0.3f,0.1f,0.1f},
                                     1,0,0,0, cam);
        check_bool("valid", s.valid, true);
        float maj = 0, mino = 0, ang = 0;
        cov2d_to_ellipse(s, &maj, &mino, &ang);

        // 中心仍在屏幕中心
        check("中心 x", s.cx, cam.cx, 0.5f);

        // σx:σy = 3:1。考虑低通项后比例略小于 3，用 sqrt((9k+0.3)/(1k+0.3))
        // 其中 k = (fx/z)² * σy² ，因为我们把 σ 整体缩到 0.1 量级
        const float fz = cam.fx / 10.f;              // fx/z
        const float k  = fz*fz*0.1f*0.1f;            // (fx/z)²·σy²
        const float want_ratio = std::sqrt((9.f*k + 0.3f) / (1.f*k + 0.3f));
        check("长短轴之比 = sqrt((9k+.3)/(k+.3))", maj / mino, want_ratio, 0.02f);
        // 长轴应对齐屏幕 x 轴（角度 0）
        check("长轴朝向 ≈ 0 rad（沿 x）", ang, 0.f, 0.02f);
    }

    // ── ③ 四元数旋转 90°：长轴应转到 y 方向 ─────────────────────────
    // 绕 z 轴转 90°，四元数 (cos45, 0,0, sin45) = (0.7071, 0,0, 0.7071)
    fprintf(stdout, "\n【3】把上面高斯绕 z 轴转 90°，长轴应转到 y 方向\n");
    {
        const float s2 = 0.70710678f;
        Splat2D s = project_gaussian(Vec3{0,0,0}, Vec3{0.3f,0.1f,0.1f},
                                     s2, 0, 0, s2, cam);
        check_bool("valid", s.valid, true);
        float maj = 0, mino = 0, ang = 0;
        cov2d_to_ellipse(s, &maj, &mino, &ang);
        // 旋转 90° 后，主轴沿 y。角度应为 ±π/2
        const float a_deg = ang * 180.f / 3.14159265f;
        check_bool("长轴朝向 ≈ ±90°", std::fabs(std::fabs(a_deg) - 90.f) < 1.0f, true);
    }

    // ── ④ 透视：同一高斯移到画面右侧，应被拉长 ──────────────────────
    fprintf(stdout, "\n【4】透视拉长效应（雅可比验证）\n");
    {
        // 中心处
        Splat2D sc = project_gaussian(Vec3{0,0,0}, Vec3{0.1f,0.1f,0.1f}, 1,0,0,0, cam);
        float mc = 0, nc = 0, ac = 0;
        cov2d_to_ellipse(sc, &mc, &nc, &ac);

        // 移到 x=+5（画面右侧，z 变为 10 但横向偏移）
        Splat2D sr = project_gaussian(Vec3{5,0,0}, Vec3{0.1f,0.1f,0.1f}, 1,0,0,0, cam);
        check_bool("右侧高斯 valid", sr.valid, true);
        float mr = 0, nr = 0, ar = 0;
        cov2d_to_ellipse(sr, &mr, &nr, &ar);
        // 右移后屏幕 x 应大于中心
        check_bool("屏幕 x > cx", sr.cx > cam.cx, true);
        // 透视拉伸：长轴应变大（雅可比含 -fx*x/z² 交叉项）
        check_bool("右侧主轴 ≥ 中心主轴（被拉伸）", mr >= mc - 0.01f, true);
        fprintf(stdout, "       （中心主轴=%.4f  右侧主轴=%.4f  比值=%.4f）\n",
                mc, mr, mr/mc);
    }

    // ── ⑤ 剔除工况 ────────────────────────────────────────────────
    fprintf(stdout, "\n【5】剔除边界条件\n");
    {
        // 相机后方
        Splat2D s1 = project_gaussian(Vec3{0,0,-20}, Vec3{0.1f,0.1f,0.1f}, 1,0,0,0, cam);
        check_bool("相机后方 → 剔除", s1.valid, false);

        // 超出远平面
        Splat2D s2 = project_gaussian(Vec3{0,0,1000}, Vec3{0.1f,0.1f,0.1f}, 1,0,0,0, cam);
        check_bool("超出 zfar → 剔除", s2.valid, false);

        // 屏幕外（很远横向）
        Splat2D s3 = project_gaussian(Vec3{10000,0,0}, Vec3{0.1f,0.1f,0.1f}, 1,0,0,0, cam);
        check_bool("屏幕外 → 剔除", s3.valid, false);

        // 极大高斯（σ=50，3σ 半径远超 4× 对角线）→ 应被半径上限剔除
        Splat2D s4 = project_gaussian(Vec3{0,0,0}, Vec3{50,50,50}, 1,0,0,0, cam);
        check_bool("超大高斯 → 剔除", s4.valid, false);

        // 刚好在近平面前
        Splat2D s5 = project_gaussian(Vec3{0,0,-9.99f}, Vec3{0.1f,0.1f,0.1f}, 1,0,0,0, cam);
        check_bool("近平面内 → 剔除", s5.valid, false);
    }

    // ── ⑤b max_sigma_radius 可配（验证"巨型高斯"阈值确实生效）──────
    fprintf(stdout, "\n【5b】max_sigma_radius 阈值可配\n");
    {
        // σ=1 在 z=10 处 → 3σ 半径 ≈ 174 px。默认上限(4×对角线≈3200)不删。
        Splat2D a = project_gaussian(Vec3{0,0,0}, Vec3{1,1,1}, 1,0,0,0, cam);
        check_bool("σ=1, 默认上限 → 保留", a.valid, true);

        // 把上限压到 100 px，同一个高斯应被剔除
        Camera strict = cam;
        strict.max_sigma_radius = 100.f;
        Splat2D b = project_gaussian(Vec3{0,0,0}, Vec3{1,1,1}, 1,0,0,0, strict);
        check_bool("σ=1, 上限=100px → 剔除", b.valid, false);
    }

    // ── ⑥ 与"手算距离缩放"对照：z 加倍，轴长减半 ────────────────────
    fprintf(stdout, "\n【6】距离-尺度关系：z 从 10 → 20，轴长应约减半\n");
    {
        Splat2D s10 = project_gaussian(Vec3{0,0,0},  Vec3{0.1f,0.1f,0.1f}, 1,0,0,0, cam);
        Splat2D s20 = project_gaussian(Vec3{0,0,10}, Vec3{0.1f,0.1f,0.1f}, 1,0,0,0, cam);
        float m10 = 0, m20 = 0, tmp = 0;
        cov2d_to_ellipse(s10, &m10, &tmp, &tmp);
        cov2d_to_ellipse(s20, &m20, &tmp, &tmp);
        // 不含低通项时期望 2:1；含 0.3 低通会略偏离
        const float fz = cam.fx / 10.f;
        const float k  = fz*fz*0.1f*0.1f;
        const float want = std::sqrt((k + 0.3f) / (0.25f*k + 0.3f));
        check("主轴比 = sqrt((k+.3)/(k/4+.3))", m10 / m20, want, 0.02f);
    }

    fprintf(stdout, "\n=====================================================\n");
    fprintf(stdout, " 结果: %d 通过 / %d 失败\n", g_pass, g_fail);
    fprintf(stdout, "=====================================================\n");
    return g_fail == 0 ? 0 : 1;
}
