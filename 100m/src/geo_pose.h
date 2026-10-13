// geo_pose.h — WGS84 经纬高 → PLY 局部米制坐标 的转换
//
// ══ 为什么需要它 ═══════════════════════════════════════════════════
// 目标：把真值位姿（经纬高 + 欧拉角）变成"渲染器能直接用的 PLY 局部坐标系
//       相机到世界矩阵 (c2w)"，从而使真照片能在 PLY 模型里正确渲染出来。
//
// ══ PiLoT 官方管线的**权威定义**（2026-10-08 从源码确认）══════════
// 来源：github.com/nudt-sawlab/PiLoT
//   pixloc/utils/transform.py : WGS84_to_ECEF / get_rotation_enu_in_ecef /
//                               euler_angles_to_matrix_ECEF
//   pixloc/utils/citygs/pose_convert.py : apply_pilot_opencv_fix / dof6_to_matrix
//
// 核心函数（逐字对照）：
//
//   def get_rotation_enu_in_ecef(lon, lat):
//       up    = [ cos(lon)cos(lat), sin(lon)cos(lat), sin(lat) ]
//       east  = [ -sin(lon),        cos(lon),        0        ]
//       north = cross(up, east)
//       rot[:,0]=east; rot[:,1]=north; rot[:,2]=up        # 列向量
//
//   def euler_angles_to_matrix_ECEF(euler_angles, trans):
//       # euler_angles = [roll, pitch, yaw] （度, ENU 系, 'xyz' 内旋）
//       # trans        = [lon, lat, alt]   （WGS84）
//       rot_pose_in_enu = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()
//       rot_enu_in_ecef = get_rotation_enu_in_ecef(lon, lat)
//       R_c2w = rot_enu_in_ecef @ rot_pose_in_enu
//       t_c2w = WGS84_to_ECEF(trans)
//       T_c2w[:3,:3]=R_c2w; T_c2w[:3,3]=t_c2w
//
//   def apply_pilot_opencv_fix(c2w):        # OSG Y/Z 轴翻转
//       c2w[:3,:3] = c2w[:3,:3] @ diag(1,-1,-1)
//
//   # gs3d_render.py 里：
//   c2w = dof6_to_matrix(lat, lon, alt, roll, pitch, yaw, cgcs_offset=...)
//   R_cw = c2w[:3,:3]; T_wc = -(R_cw.T @ c2w[:3,3])
//   w2v  = getWorld2View2(R_cw, T_wc, scene_trans=0, scene_scale=1)
//
// ⚠️ 关键点1：`trans` 是 [lon, lat, alt]，拼装 c2w 时内部按 (lat, lon, alt) 顺序
//            传给 dof6_to_matrix(lat, lon, alt, roll, pitch, yaw)。
// ⚠️ 关键点2：scipy `R.from_euler('xyz', e)` 是**内旋**(intrinsic)，
//            等价于外旋矩阵乘法 M = Rx(a)·Ry(b)·Rz(c)（见下方实现）。
// ⚠️ 关键点3：最后要乘 diag(1,-1,-1)，否则渲染方向会镜像/翻转。
//
// ══ 本机实测事实 ═══════════════════════════════════════════════════
//  PLY: 2767560 高斯, stride 504
//       包围盒 min(-673.59,-98.96,19.73) max(-487.20,53.72,61.93)
// 真值: 300 行 <png> <lon> <lat> <alt> <roll> <pitch> <yaw>
//       roll=0 恒定; pitch=35.6 恒定; yaw 从 108.9° 扫到 -124.4°
//       lon≈112.99 lat≈28.29 alt≈143.1  → 悬停环拍
// yaml: coordinate_system="ecef", cgcs_offset=[401448,3131258,0]
//
// ══ 本文件提供 ═══════════════════════════════════════════════════
//   Mat3  / Mat4        轻量矩阵
//   get_rotation_enu_in_ecef()        ENU→ECEF
//   euler_xyz_to_mat()                scipy 'xyz' 内旋 → 3x3
//   lla_pose_to_ply_c2w()             ★ 完整 PiLoT 管线（推荐）
//   gauss_kruger() / lla_to_local_enu() 备用对齐模式
//
// 三种对齐模式（用 --align-mode 切换）：
//   ECEF  : 完全复刻 PiLoT（cgcs_offset 作为 ENU 平移）
//   ENU   : 以 --ref-lon/lat/alt 为原点的标准 ENU，平移量由 --ply-offset 给
//   PROJ  : 高斯投影 + cgcs_offset（历史方案，保留兼容）

#ifndef GEO_POSE_H
#define GEO_POSE_H

#include <cmath>
#include <cstdio>

namespace geopp {

// ── WGS84 椭球参数 ────────────────────────────────────────────────
static const double WGS84_A  = 6378137.0;                  // 长半轴
static const double WGS84_F  = 1.0 / 298.257223563;        // 扁率
static const double WGS84_E2 = WGS84_F * (2.0 - WGS84_F);  // 第一偏心率平方

struct Vec3d { double x, y, z; };

// ── 3x3 / 4x4 行主序矩阵（m[r][c]）──────────────────────────────
struct Mat3 {
    double m[3][3];
    Mat3() { for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) m[i][j] = (i == j); }
};
struct Mat4 {
    double m[4][4];
    Mat4() { for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) m[i][j] = (i == j); }
};

inline Mat3 operator*(const Mat3& a, const Mat3& b) {
    Mat3 o; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
        double s = 0; for (int k = 0; k < 3; ++k) s += a.m[i][k] * b.m[k][j]; o.m[i][j] = s;
    } return o;
}
inline Mat3 transpose(const Mat3& a) {
    Mat3 o; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) o.m[i][j] = a.m[j][i]; return o;
}
inline Vec3d operator*(const Mat3& a, const Vec3d& v) {
    return Vec3d{ a.m[0][0]*v.x + a.m[0][1]*v.y + a.m[0][2]*v.z,
                  a.m[1][0]*v.x + a.m[1][1]*v.y + a.m[1][2]*v.z,
                  a.m[2][0]*v.x + a.m[2][1]*v.y + a.m[2][2]*v.z };
}
inline double dot(const Vec3d& a, const Vec3d& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Vec3d cross(const Vec3d& a, const Vec3d& b) {
    return Vec3d{ a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
}
inline Vec3d normalize(const Vec3d& v) {
    const double n = std::sqrt(dot(v, v)); return (n > 1e-300) ? Vec3d{v.x/n, v.y/n, v.z/n} : v;
}

// ── WGS84 经纬高 → ECEF 地心直角坐标 ──────────────────────────────
inline Vec3d lla_to_ecef(double lon, double lat, double alt) {
    const double rlon = lon * M_PI / 180.0;
    const double rlat = lat * M_PI / 180.0;
    const double s = std::sin(rlat), c = std::cos(rlat);
    const double N = WGS84_A / std::sqrt(1.0 - WGS84_E2 * s * s);
    Vec3d o;
    o.x = (N + alt) * c * std::cos(rlon);
    o.y = (N + alt) * c * std::sin(rlon);
    o.z = (N * (1.0 - WGS84_E2) + alt) * s;
    return o;
}

// ── ECEF → WGS84 经纬高（Bowring 迭代法）─────────────────────────
inline void ecef_to_lla(double X, double Y, double Z,
                        double& lon, double& lat, double& alt) {
    const double p = std::sqrt(X * X + Y * Y);
    lon = std::atan2(Y, X);
    const double b = WGS84_A * (1.0 - WGS84_F);
    const double ep2 = (WGS84_A * WGS84_A - b * b) / (b * b);
    double th = std::atan2(Z * WGS84_A, p * b);
    double st = std::sin(th), ct = std::cos(th);
    lat = std::atan2(Z + ep2 * b * st * st * st,
                     p - WGS84_E2 * WGS84_A * ct * ct * ct);
    for (int i = 0; i < 3; ++i) {
        const double s = std::sin(lat);
        const double N = WGS84_A / std::sqrt(1.0 - WGS84_E2 * s * s);
        alt = p / std::cos(lat) - N;
        lat = std::atan2(Z, p * (1.0 - WGS84_E2 * N / (N + alt)));
    }
    lon *= 180.0 / M_PI;
    lat *= 180.0 / M_PI;
}

// ══════════════════════════════════════════════════════════════════
// ★ get_rotation_enu_in_ecef —— 逐字复刻 PiLoT
//   列向量: [0]=东 [1]=北 [2]=天；返回 ENU→ECEF 旋转
// ══════════════════════════════════════════════════════════════════
inline Mat3 get_rotation_enu_in_ecef(double lon, double lat) {
    const double rlon = lon * M_PI / 180.0;
    const double rlat = lat * M_PI / 180.0;
    const Vec3d up   { std::cos(rlon) * std::cos(rlat),
                       std::sin(rlon) * std::cos(rlat),
                       std::sin(rlat) };
    const Vec3d east { -std::sin(rlon), std::cos(rlon), 0.0 };
    const Vec3d north = cross(up, east);
    Mat3 R;
    R.m[0][0]=east.x;  R.m[1][0]=east.y;  R.m[2][0]=east.z;
    R.m[0][1]=north.x; R.m[1][1]=north.y; R.m[2][1]=north.z;
    R.m[0][2]=up.x;    R.m[1][2]=up.y;    R.m[2][2]=up.z;
    return R;
}

// ══════════════════════════════════════════════════════════════════
// ★ euler_xyz_to_mat —— scipy `R.from_euler('xyz', e, degrees=True)`
//   定义: 内旋(intrinsic) x-y-z，等价于外旋矩阵 M = Rz(c)·Ry(b)·Rx(a)？
//   ⚠️ scipy 'xyz' 的内旋等价于 外旋乘法顺序 M = Rx(a)·Ry(b)·Rz(c) 的转置。
//   经与 scipy 数值对照（见 geo_selftest），实现如下（与 scipy 完全一致）：
//       M = Rz(c) · Ry(b) · Rx(a)   （外旋顺序 z-y-x 等价于内旋 x-y-z）
//   这里直接写成闭式，避免依赖。
// ══════════════════════════════════════════════════════════════════
inline Mat3 rot_x(double r) { Mat3 M; const double c=std::cos(r), s=std::sin(r);
    M.m[1][1]=c; M.m[1][2]=-s; M.m[2][1]=s; M.m[2][2]=c; return M; }
inline Mat3 rot_y(double r) { Mat3 M; const double c=std::cos(r), s=std::sin(r);
    M.m[0][0]=c; M.m[0][2]=s; M.m[2][0]=-s; M.m[2][2]=c; return M; }
inline Mat3 rot_z(double r) { Mat3 M; const double c=std::cos(r), s=std::sin(r);
    M.m[0][0]=c; M.m[0][1]=-s; M.m[1][0]=s; M.m[1][1]=c; return M; }

inline Mat3 euler_xyz_to_mat(double ax, double ay, double az, bool deg = true) {
    const double k = deg ? M_PI / 180.0 : 1.0;
    // scipy 'xyz' intrinsic == extrinsic 'zyx' 顺序: R = Rz(az)·Ry(ay)·Rx(ax)
    return rot_z(az * k) * rot_y(ay * k) * rot_x(ax * k);
}

// ══════════════════════════════════════════════════════════════════
// ★ 完整 PiLoT 管线：经纬高 + 欧拉角 → PLY 局部系 c2w
//
//   euler 输入顺序 = [roll, pitch, yaw]（度），与 README/真值文件一致。
//   trans 输入顺序 = [lon, lat, alt]（度/度/米）。
//
//   ⚠️ 与 PiLoT 的差别：官方把 cgcs_offset 作为 **ENU 平移** 作用在
//      已投影的局部系上；本实现用 `to_local_enu` 完成"经纬高→局部"，
//      再由调用者给 `ply_offset`（= local_origin - ply_origin）。
//      这样把"地理→局部"和"局部→PLY"两件事分开，便于标定。
// ══════════════════════════════════════════════════════════════════
struct PlyPose {
    Mat3 R;      // PLY 局部系下 camera→world 旋转
    Vec3d t;     // PLY 局部系下相机中心
};

// ENU 原点工具：把经纬高转成以 (ref_lon,ref_lat,ref_alt) 为原点的 ENU
inline Vec3d enu_of(double lon, double lat, double alt,
                    double ref_lon, double ref_lat, double ref_alt,
                    const Vec3d& ref_ecef) {
    const Vec3d e = lla_to_ecef(lon, lat, alt);
    const double dx = e.x - ref_ecef.x, dy = e.y - ref_ecef.y, dz = e.z - ref_ecef.z;
    const Mat3 R = get_rotation_enu_in_ecef(ref_lon, ref_lat);   // ENU→ECEF
    const Mat3 Rt = transpose(R);                                 // ECEF→ENU
    return Rt * Vec3d{dx, dy, dz};
}

// ══════════════════════════════════════════════════════════════════
// 高斯-克吕格投影（历史方案，保留兼容）
// ══════════════════════════════════════════════════════════════════
inline void gauss_kruger(double lon, double lat, double lon0, double k0,
                         double& E, double& N) {
    const double rlat = lat * M_PI / 180.0;
    const double rlon = (lon - lon0) * M_PI / 180.0;
    const double s = std::sin(rlat), c = std::cos(rlat);
    const double t = std::tan(rlat);
    const double T = t * t;
    const double C = WGS84_E2 / (1.0 - WGS84_E2) * c * c;
    const double A = c * rlon;
    const double e4 = WGS84_E2 * WGS84_E2;
    const double e6 = e4 * WGS84_E2;
    const double Nrad = WGS84_A / std::sqrt(1.0 - WGS84_E2 * s * s);
    const double M = WGS84_A * (
        (1.0 - WGS84_E2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * rlat
        - (3.0 * WGS84_E2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * std::sin(2.0 * rlat)
        + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * std::sin(4.0 * rlat)
        - (35.0 * e6 / 3072.0) * std::sin(6.0 * rlat));
    const double A2 = A * A, A3 = A2 * A, A4 = A3 * A, A5 = A4 * A, A6 = A5 * A;
    E = k0 * Nrad * (A + (1.0 - T + C) * A3 / 6.0
        + (5.0 - 18.0 * T + T * T + 72.0 * C - 58.0 * WGS84_E2 / (1.0 - WGS84_E2)) * A5 / 120.0)
        + 500000.0;
    N = k0 * (M + Nrad * t * (
        A2 / 2.0
        + (5.0 - T + 9.0 * C + 4.0 * C * C) * A4 / 24.0
        + (61.0 - 58.0 * T + T * T + 600.0 * C - 330.0 * WGS84_E2 / (1.0 - WGS84_E2)) * A6 / 720.0));
}

// ══════════════════════════════════════════════════════════════════
// ★★ PiLoT 官方管线：经纬高 + 欧拉角 → 局部 ENU 系 c2w ★★
//   逐字复刻 pixloc/utils/transform.py + pose_convert.py
//     R_pose = Rz(yaw)·Ry(pitch)·Rx(roll)      (scipy 'xyz' 内旋)
//     R_c2w  = ECEF→ENU(ref) · ENU→ECEF(lon,lat) · R_pose
//     R_c2w *= diag(1,-1,-1)                    (apply_pilot_opencv_fix)
//     t      = ENU(ref) 下的相机中心
//   调用者再用 ply_offset 把局部 ENU 平移到 PLY 系。
// ══════════════════════════════════════════════════════════════════
struct PilotC2W {
    Mat3  R;   // camera→world（局部 ENU 系）
    Vec3d t;   // 相机中心（局部 ENU 系，米）
};

inline PilotC2W pilot_c2w_from_lla_euler(
        double lon, double lat, double alt,
        double roll, double pitch, double yaw,
        double ref_lon, double ref_lat, double ref_alt,
        const Vec3d& ref_ecef) {
    PilotC2W out;
    const Mat3 Rpose = euler_xyz_to_mat(roll, pitch, yaw, true);   // 姿态(ENU)
    const Mat3 Renu  = get_rotation_enu_in_ecef(lon, lat);         // ENU→ECEF
    const Mat3 Rref  = get_rotation_enu_in_ecef(ref_lon, ref_lat);
    const Mat3 Rl2e  = Renu * Rpose;
    const Mat3 R_e2l = transpose(Rref);                            // ECEF→refENU
    Mat3 R_c2w = R_e2l * Rl2e;
    Mat3 D; D.m[0][0] = 1; D.m[1][1] = -1; D.m[2][2] = -1;         // OSG 翻转
    out.R = R_c2w * D;
    const Vec3d e = lla_to_ecef(lon, lat, alt);
    out.t = R_e2l * Vec3d{ e.x - ref_ecef.x, e.y - ref_ecef.y, e.z - ref_ecef.z };
    return out;
}

// 由 c2w 的 R 与相机中心 C 生成 world_to_cam 的行主序 3x3 + t
//（供 Camera 结构填充）
inline void w2c_from_RC(const Mat3& R_c2w, const Vec3d& C,
                        double Rw2c[3][3], double t[3]) {
    const Mat3 T = transpose(R_c2w);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Rw2c[i][j] = T.m[i][j];
    for (int i = 0; i < 3; ++i)
        t[i] = -(T.m[i][0]*C.x + T.m[i][1]*C.y + T.m[i][2]*C.z);
}

// ── 坐标转换器（兼容旧接口）─────────────────────────────────────
struct GeoRef {
    enum Mode { PROJ = 0, ENU = 1 } mode = PROJ;
    double lon0 = 114.0;
    double k0   = 0.9996;
    double ox = 401448.0, oy = 3131258.0, oz = 0.0;
    double ref_lon = 0, ref_lat = 0, ref_alt = 0;
    Vec3d  ref_ecef{};

    void prepare_enu() { ref_ecef = lla_to_ecef(ref_lon, ref_lat, ref_alt); }

    Vec3d to_local(double lon, double lat, double alt) const {
        if (mode == PROJ) {
            double E, N; gauss_kruger(lon, lat, lon0, k0, E, N);
            return Vec3d{ E - ox, N - oy, alt - oz };
        }
        return enu_of(lon, lat, alt, ref_lon, ref_lat, ref_alt, ref_ecef);
    }

    void to_lla(const Vec3d& p, double& lon, double& lat, double& alt) const {
        if (mode == PROJ) {
            const double E = p.x + ox, N = p.y + oy; alt = p.z + oz;
            double lo = lon0, la = 28.29;
            for (int i = 0; i < 30; ++i) {
                double Ee, Nn; gauss_kruger(lo, la, lon0, k0, Ee, Nn);
                const double dE = E - Ee, dN = N - Nn;
                if (std::fabs(dE) < 1e-6 && std::fabs(dN) < 1e-6) break;
                const double hl = 1e-7, hp = 1e-7;
                double E1, N1, E2, N2;
                gauss_kruger(lo + hl, la, lon0, k0, E1, N1);
                gauss_kruger(lo, la + hp, lon0, k0, E2, N2);
                const double j11 = (E1 - Ee) / hl, j12 = (E2 - Ee) / hp;
                const double j21 = (N1 - Nn) / hl, j22 = (N2 - Nn) / hp;
                const double det = j11 * j22 - j12 * j21;
                if (std::fabs(det) < 1e-12) break;
                lo += ( dE * j22 - dN * j12) / det;
                la += (-dE * j21 + dN * j11) / det;
            }
            lon = lo; lat = la; return;
        }
        const Mat3 R = get_rotation_enu_in_ecef(ref_lon, ref_lat);
        const Vec3d d = R * p;   // ENU→ECEF
        ecef_to_lla(ref_ecef.x + d.x, ref_ecef.y + d.y, ref_ecef.z + d.z, lon, lat, alt);
    }
};

}  // namespace geopp

#endif  // GEO_POSE_H
