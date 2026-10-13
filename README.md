# BUAA 3DGS 板端视觉定位

在 **Orange Pi 5 Plus（RK3588）** 嵌入式平台上的 3DGS / 2DGS 实时视觉定位系统。

渲染式定位范式：用估计位姿从 3DGS 地图渲染参考图 → 与真实相机图像做特征匹配 →
PnP 解算 6DoF 位姿，闭环迭代。全部算力在板端完成（CPU + Mali-G610 GPU + 6 TOPS NPU）。

本仓库只包含**源代码**。高斯模型（PLY）、数据集、编译产物、运行输出均不入库，
见 `.gitignore`。

## 目录结构

```
100m/     100 m 低空数据集专用工作区（BUAA_location_100）
  src/            渲染器 / 定位主程序 / 特征前端等 C++ 源码
  scripts/        定位运行脚本
  Makefile        板端原生编译（g++ -O3 -march=armv8.2-a+fp16+dotprod）
  README.md       100 m 工作区详细说明（数据、内参、坐标系、实验结论）

800m/     800 m 高空数据集工作区（BUAA_location_800）
  src/            同上，含 800 m 场景的适配与调参
  sh/             频率锁定等系统脚本
  run.sh          800 m 定位入口
  *.sh            校准 / 自检 / 基准脚本
  Makefile
  README.md       800 m 工作区详细说明
```

## 核心组件

| 组件 | 说明 |
|---|---|
| `gs_locate_uav.cpp` | 定位主程序：渲染—匹配—PnP 闭环、时序滤波、指标统计 |
| `renderer.cpp` | 高斯光栅化（EWA splatting），OpenGL ES 3.2 / EGL 离屏渲染 |
| `sorter.cpp` | 高斯深度排序 |
| `xfeat_frontend.cpp` | XFeat 学习型特征前端（RKNN / NPU） |
| `ply_loader.cpp` | INRIA PLY 解析（3DGS 62 列 / 2DGS 37 列自动识别） |
| `camera.cpp` `geo_pose.h` | 相机模型与 ECEF/ENU 坐标转换 |

## 构建

板端原生编译（需 OpenCV 4.5、rknnrt）：

```bash
cd 100m && make build/gs_locate_uav -j4
```

## 运行

```bash
cd 100m && bash scripts/locate.sh
```

## 硬件平台

- CPU：4×Cortex-A76 + 4×Cortex-A55（RK3588）
- GPU：Mali-G610 MP4，1 GHz（OpenGL ES 3.2 / OpenCL / Vulkan 1.3）
- NPU：6 TOPS，librknnrt 2.3.0
- 相机：1566×1176 航拍，云台三视角（−45° / −60° / −90°）循环

---

*本仓库为私有代码库。*
