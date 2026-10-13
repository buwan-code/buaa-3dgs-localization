// projfast.h — 折叠投影路径的声明
//
// 与 project_gaussian（camera.h 里的标量 EWA）数学等价，但绕开
// 3×3 中间矩阵。真值源：projfast.cpp 顶部的代数推导。
//
// 用 GSPLAT_PROJ_FAST 宏在 renderer.cpp 里切换：
//   不定义 → 走原标量路径
//   定义   → 走折叠快路径
// 这样同一份代码可以编出两个二进制做 A/B，不必改源码。
#pragma once

#include "camera.h"
#include "proj_core.h"

namespace gsplat {

// p    : 世界坐标位置
// s3   : 三轴尺度（已 exp 过）
// q*   : 四元数（w,x,y,z，未归一化也可，内部归一化）
// k    : 逐帧不变量（ProjConsts::from_camera(cam)）
Splat2D project_gaussian_fast(const Vec3& p, const Vec3& s3,
                              float qw, float qx, float qy, float qz,
                              const Camera& cam, const ProjConsts& k);

} // namespace gsplat
