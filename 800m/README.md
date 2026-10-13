# gsplat —— 嵌入式平台 3DGS 实时视觉定位

> **OrangePi 5 Plus（RK3588）上的 3D Gaussian Splatting 视觉定位系统**
> 吃一张实拍航片 → 渲染参考图 → 特征匹配 → 解出无人机 6DoF 位姿 + 目标地理坐标

> ★★ **工作区定位（2026-10-13）**：本目录（原 gsplat，现 ~/BUAA_location_800）
> **专用于 airzoo 数据集（美国乡村，800 m 高空）**，同时包含 2DGS 与 3DGS 两套 PLY。
> 代码已回滚到「启动 BUAA 新数据集训练之前」的版本（src/gs_locate_uav.cpp 126305 B / 2204 行），
> 并经全量 1067 帧对拍确认与历史结果**逐位一致**。
>
> 100 m 低空 BUAA 任务已独立到 ~/BUAA_location_100，两边完全隔离、互不干扰。

本文档是工程的**唯一入口说明**。包含：怎么编译、怎么跑一次实验、每个参数什么意思、
输出文件怎么看、成果展示在哪、已知问题。

---

## 目录

- [1. 这是什么](#1-这是什么)
- [2. 实测结果](#2-实测结果)
- [3. 硬件平台与算力分工](#3-硬件平台与算力分工)
- [4. 目录结构](#4-目录结构)
- [5. 五分钟跑通](#5-五分钟跑通)
- [6. 命令行参数手册](#6-命令行参数手册)
- [7. 输出文件格式](#7-输出文件格式)
- [8. 成果展示索引](#8-成果展示索引)
- [9. 脚本一览](#9-脚本一览)
- [10. 代码阅读顺序](#10-代码阅读顺序)
- [11. 已知问题与未完成](#11-已知问题与未完成)
- [12. 参考](#12-参考)

---

## 1. 这是什么

一句话：**拿一张实拍航片，跟「用 3DGS 模型从这个大概位置渲染出来的参考图」做特征匹配，
再反解出无人机到底在哪、朝哪，并顺便算出一张图上任意一点对应的经纬度。**

这个思路叫 **Render2Loc**，本项目参考并复刻了 **PiLoT**（NUAA-SAILab）的范式。
巧妙之处在于**不需要把照片和地图做全局配准** —— 只要有一个「大概位置」
（机载 GPS/IMU 给的粗定位），就能渲染出视角相近的参考图，剩下的交给特征匹配 + PnP。

### 系统由四件事构成

| 阶段 | 输入 → 输出 | 在哪跑 |
|---|---|---|
| ① **建图** | 照片 + COLMAP 位姿 → 3DGS 模型（.ply） | 离线，6000D 服务器（CityGaussian 官方流程） |
| ② **渲染** | PLY + 相机位姿 → 参考图 + 深度图 | 板端，**本工程自写的 OpenGL ES 渲染器** |
| ③ **定位** | 参考图 + 实拍图 → 6DoF 位姿 | 板端，NPU 提特征 + CPU 匹配 + PnP |
| ④ **输出** | 位姿 + 目标经纬度 + 置信度 | 板端，与真值比对算指标 |

### 为什么渲染器要自己写

官方 `diff-gaussian-rasterization` 是 **CUDA** 实现，板子的 **Mali GPU 用不了**。
本工程从零实现了一条 **OpenGL ES 3.2** 的 3DGS 光栅化管线：
EWA 投影（CPU）→ 深度排序（CPU）→ 实例化四边形光栅化（GPU）→ 颜色/深度回读。

> 这是本工作的核心贡献点：**首个在嵌入式 ARM 平台实现 3DGS 渲染的实时视觉定位系统**。

---

## 2. 实测结果

### 2.1 精度对比（板端实跑，全量 1067 帧，同 PnP 口径）

| 前端 | mT (m) | mR (°) | R@1m | R@3m | R@5m | 完整率 | FPS |
|---|---|---|---|---|---|---|---|
| **XFeat**（RKNPU） | **7.123** | **0.477** | 0.75% | **14.15%** | **31.30%** | **100%** | 5.53 |
| ORB（OpenCV CPU） | 11.057 | 0.750 | 0.37% | 5.25% | 15.93% | 100% | 7.97 |

- **mT**：中位平移误差（估计相机位置与真值位置的欧氏距离中位数，米）
- **mR**：中位旋转误差（旋转矩阵测地距离中位数，度）
- **Recall@(k m, k°)**：平移 < k m **且** 旋转 < k° 的帧占**全部输入帧**的比例（失败帧计入未命中）
- **Completeness**：能输出有效位姿的帧占比
- **FPS**：端到端单帧速度（含读图）

> 换成 XFeat（NPU）后：mT 11.057 → **7.123 m**，R@5m 15.93% → **31.30%**。

### 2.2 单帧耗时分解（180.7 ms/帧 → 5.53 FPS）

| 阶段 | 耗时 | 说明 |
|---|---|---|
| **渲染** | **113.7 ms** | 投影 54.0 / 排序 15.5 / 组装 6.6 / 上传+绘制 1.1 / 回读等待 0 |
| **定位** | **66.7 ms** | 渲染图 NPU 18.16 / 解码 8.37 / MNN 匹配 35.10 / PnP 4.89 |
| 合计 | **180.7 ms** | 端到端 5.53 FPS |

关键优化（均已实测）：
- **帧间流水线**（`--pipeline`）：每帧只渲染 1 次，渲染图滞后 1 帧消费 → 回读等待（40~73 ms）全部隐藏
- **查询图预取**（`--prefetch`）：主线程等待仅 **0.08 ms/帧**，其余全被渲染隐藏
- **紧凑排序**：元素 16 B → 8 B，访存量减半
- **MNN 匹配缓存优化**：合并行/列最大值计算，46.7 → **35.0 ms**（−27%）

### 2.3 输入数据

| 项 | 值 |
|---|---|
| 帧数 | **1067 帧**（原始视频 3209 帧，每 3 帧取 1） |
| 原始分辨率 | 1600 × 900 → 运行分辨率 **512 × 288 灰度** |
| 3DGS 地图 | CityGaussian 训练，**300 万高斯**，744 MB |
| 相机内参 | fx = fy = **1931.33**，cx = 800，cy = 450（@1600×900） |
| ENU 参考点 | lon −112.066241、lat 41.205416、alt 0 |
| 首帧初值 | 真值 + 扰动（位置 ±5 m、姿态 ±0.3°，seed 42 可复现） |

---

## 3. 硬件平台与算力分工

**OrangePi 5 Plus（RK3588）** · 通过 SSH 免密登录（`~/.ssh/config` 里配置）

| 算力单元 | 规格 | 本工程承担的工作 |
|---|---|---|
| **CPU** | 4×Cortex-A76 + 4×Cortex-A55 | 投影、排序、组装、XFeat 解码链、MNN 匹配、PnP |
| **GPU** | Mali-G610（4 核 @1 GHz，OpenCL + Vulkan 1.3） | 3DGS 光栅化渲染（OpenGL ES 3.2 离屏） |
| **NPU** | RKNPU，6 TOPS，`librknnrt 2.3.0` | XFeat 骨干网络推理 |

**三块算力同时在用** —— 异构协同是本工程的另一卖点。

### 板端环境要求

- OpenGL ES 3.2（EGL 离屏上下文，无需接显示器）
- OpenCV 4（`opencv_core / imgproc / features2d / calib3d / imgcodecs`）
- `librknnrt.so`（瑞芯微 NPU 运行时，在 `/usr/lib/`）
- **GPU/CPU 频率须锁 `performance`**：板卡重启会把 governor 重置回 `ondemand`，
  GPU 掉到 300 MHz 会让渲染从 3.5 FPS 掉到 2.5 FPS。
  已建 systemd 服务 `gsplat-perf.service`（enabled）开机自动锁定。

---

## 4. 目录结构

```
~/gsplat/
├── Makefile                  唯一构建入口（make all / make build/gs_locate_uav）
├── README.md                 本文件
├── src/                      全部源码（详见 §10 阅读顺序）
│   ├── [核心]  camera.*  sorter.*  renderer.*  ply_loader.*  offscreen.*
│   │           geo_pose.h  pose_post.h  projfast.*  proj_core.h  splat_loader.h
│   ├── [主程序] gs_locate_uav.cpp   无人机定位主程序
│   │            gs_localize.cpp     自证闭环（用真值渲染当实拍图）
│   ├── [特征]  xfeat_frontend.*     XFeat NPU 前端
│   ├── [第三方] rknn_api.h          瑞芯微 NPU 官方头文件
│   └── [工具]  depthcheck  calib_*  calibrate_align  raster_bench
│                camtest  proj*  thermal*  loadtest  rbtest
├── build/                    编译产物（.o 与可执行文件）
├── bin/                      部署用可执行文件
├── data/                     数据（11 GB）
│   ├── airzoo_citygs_3dply.ply    ← 本次实验用的 3DGS 地图（710 MB）
│   ├── pilot.ply / buaa_gspj_lite3m.ply / airzoo_gt_clean.ply / ...  其他模型
│   └── query/                原始查询图
├── inputs/                   实验输入（软链到 data/，省 710 MB 重复）
│   ├── models/               → data/ 下的 .ply
│   └── query/airzoo/         实拍图 + images_512/（预解码灰度）+ gt.txt
├── runs/                     带时间戳的实验记录（含 INDEX.md）★
├── outputs/                  输出
├── models/                   小模型文件
├── sh/                       系统级脚本（fan_perm.sh 风扇 / lock_freq.sh 锁频）
├── *.sh                      运行脚本散在顶层（详见 §9）
└── src_bak_*/                历史备份（建议清理，见 §11）
```

---

## 5. 五分钟跑通

### 5.1 编译

```bash
cd ~/gsplat
make build/gs_locate_uav      # 只编译主程序
# 或
make all                      # 编译全部工具
```

### 5.2 环境体检

```bash
bash ~/gsplat/gscheck.sh
```

检查项：GPU/CPU 频率是否为 `performance`、`gsplat-perf.service` 是否 enabled、
Xorg 是否在跑、DISPLAY 是否设置、数据文件是否齐全。
**任何一项 ❌ 都要先修好再跑实验，否则测出来的数据不可信。**

### 5.3 一键跑完整实验（1067 帧，两个前端，约 6 分钟）

```bash
bash ~/run_exp.sh
```

这会依次跑 XFeat 与 ORB，结果落到 `~/results_20261012/`。

### 5.4 手动跑一条命令

```bash
cd ~/gsplat
export DISPLAY=:0

./build/gs_locate_uav data/airzoo_citygs_3dply.ply \
  --query   inputs/query/airzoo/images \
  --pre-dir inputs/query/airzoo/images_512 \
  --gt      inputs/query/airzoo/gt.txt \
  --ref -112.0662410756 41.2054158383 0 --align 0 0 0 \
  --fx 1931.33 --fy 1931.33 --cx 800 --cy 450 \
  --src-w 1600 --src-h 900 --w 512 --h 288 \
  --feat xfeat --pipeline \
  --pert-pos 5 --pert-ang 0.3 --seed 42 --iters 1 \
  --min-alpha-radius 2 --orb-n 1500 \
  --probe 256,144 --probe 64,36 --probe 448,36 \
  --out       /tmp/pose.txt \
  --probe-out /tmp/probe.txt
```

### 5.5 快速冒烟测试（只跑 3 帧，约 40 秒）

在上面命令末尾加 `--max-frames 3`，用于改完代码后的快速验证。

> ⚠️ **跑长任务务必脱离 SSH 会话**：
> ```bash
> setsid nohup ./build/gs_locate_uav ... > /tmp/run.log 2>&1 < /dev/null &
> ```
> 否则 SSH 一断，进程被杀 → **（重要）在 GPU 工作过程中强杀进程会把 Mali 驱动卡死**，
> 之后所有 GL 程序都会永久阻塞在 `ppoll`，**只能重启板子恢复**。

---

## 6. 命令行参数手册

参数定义在 `src/gs_locate_uav.cpp` 的 `struct Args`。
运行 `./build/gs_locate_uav` 不带参数会打印完整帮助。

### 6.1 必填

| 参数 | 含义 |
|---|---|
| `<ply>` | 3DGS 模型文件（位置参数，不带 `--`） |
| `--query DIR` | 实拍照片目录（`.png` / `.jpg`），程序会遍历全部 |
| `--gt FILE` | 真值位姿文件；用途：算误差 + 自动定 ENU 参考点 + 首帧初始化 |

### 6.2 输入输出

| 参数 | 含义 | 默认 |
|---|---|---|
| `--pre-dir DIR` | 预解码图目录（512×288 灰度，同名文件）。**跳过 4K 解码，省 ~185 ms/帧** | 无 |
| `--out FILE` | 位姿输出文件 | 无 |
| `--probe-out FILE` | 目标定位输出文件（配合 `--probe`） | 无 |
| `--dump DIR` | 可视化输出目录（实拍灰度 / 每轮渲染图） | 无 |
| `--dump-depth` | 同时导出渲染深度 `.bin` + 相机参数 `.txt` | 关 |
| `--dump-color` | 同时导出**彩色**渲染图（匹配本身走灰度） | 关 |
| `--probe u,v` | 目标定位探针，可重复。每帧输出该像素的全球经纬高 | 无 |

### 6.3 相机内参

| 参数 | 含义 | 默认 |
|---|---|---|
| `--fx --fy` | 焦距（原始分辨率下） | 2700 / 2700 |
| `--cx --cy` | 主点 | 1915.7 / 1075.1 |
| `--src-w --src-h` | 照片原始分辨率（内参按其缩放） | 3840 / 2160 |

### 6.4 渲染

| 参数 | 含义 | 默认 |
|---|---|---|
| `--w --h` | 渲染分辨率 | 512 / 288 |
| `--iters N` | 每帧迭代轮数（**流水线模式下只渲 1 次，此参数失效**） | 3 |
| `--pipeline` | ★ 帧间流水线：每帧 1 次渲染 + PBO 异步回读，滞后 1 帧消费 | 关（实验开） |
| `--min-alpha-radius R` | 重要性剔除阈值（alpha×屏幕半径），越大剔得越狠 | 0.5（实验 2） |
| `--lod N` | LOD：只保留重要度最高的 N 个高斯（0 = 全部） | 0，**实测不划算** |
| `--reuse-eps F` | 排序复用阈值（视矩阵逐项最大差） | 1e-4 |
| `--threads N` | 投影线程数（0 = 自动） | 0 |
| `--no-prefetch` | 关闭查询图预取线程 | 预取**开** |

### 6.5 特征前端

| 参数 | 含义 | 默认 |
|---|---|---|
| `--feat xfeat\|orb` | ★ 前端。`xfeat` = RKNPU（推荐）；`orb` = OpenCV CPU | orb |
| `--xf-model PATH` | XFeat 的 RKNN 模型 | `models/xfeat_288x512_int8.rknn` |
| `--xf-topk N` | 特征预算（实测提到 4096 无收益） | 1500 |
| `--xf-cossim F` | 互近邻余弦门限（0.90 优于官方默认 0.82） | 0.90 |
| `--xf-det-th F` | NMS 检测阈值（官方默认） | 0.05 |
| `--xf-no-subpix` | 关闭抛物线亚像素细化 | 开 |
| `--xf-threads N` | 解码/匹配线程数（RK3588 用 4×A76） | 4 |
| `--orb-n N` | ORB 特征点数 | 4000（实验 1500） |
| `--orb-levels N` | ORB 金字塔层数（512×288 下 8 层纯浪费，降到 4 层省一半时间） | 8 |

### 6.6 初值与扰动（模拟机载 GPS/IMU）

| 参数 | 含义 | 默认 |
|---|---|---|
| `--pert-pos M` | 首帧位置扰动半径（圆盘内均匀采样），模拟 GPS 误差 | 0 |
| `--pert-ang D` | 首帧姿态扰动（三轴各 ±D），模拟 IMU 误差 | 0 |
| `--seed N` | 随机种子（**固定 → 结果可复现**） | 42 |
| `--init-from-gt` | 每帧都用真值当初始位姿（最简验证路径） | 关 |
| `--no-init-from-prev` | 关闭「用上一帧初始化」 | 上一帧初始化**开** |
| `--gt-every-frame` | 每帧都用「真值+扰动」初始化（逐帧独立模式） | 关 |

### 6.7 坐标系

| 参数 | 含义 | 默认 |
|---|---|---|
| `--ref LON LAT ALT` | 手动指定 ENU 参考点 | 自动取真值第一帧 |
| `--align X Y Z` | ★ PLY 局部系 ↔ ENU 系的平移对齐量 | 0 0 0 |
| `--geo-proj` | 改用高斯投影模式（默认 ENU，精度高得多） | ENU |
| `--lon0 --k0` | 投影中央经线 / 尺度因子（仅 `--geo-proj`） | 114 / 0.9996 |
| `--offset X Y Z` | cgcs_offset（历史方案） | 401448 3131258 0 |
| `--ply-y-down` | PLY 的 Y 轴朝下，需翻转到 ENU 天向 | 关 |
| `--yaw-invert` | 方位角方向反了时用 | 关 |
| `--pitch-override D` | ★ 用给定俯仰角代替真值 a2（**实测须设 88°**，真实航片是近垂直俯视） | −1（不覆盖） |
| `--yaw-offset D` | 给方位角加偏置（两种欧拉轴序差 90°） | 0 |
| `--pilot` / `--no-pilot` | 是否用 PiLoT 官方管线重建 c2w（精确，绕开近似） | 开 |

### 6.8 后处理（**默认全关**）

| 参数 | 含义 | 默认 |
|---|---|---|
| `--tilt-h M` | 倾角约束的「有效高度」（0 = 关闭）。实测开启后 mT 1.06 m 但只在合成数据成立，且全量 1067 帧跑不通 | **0（关）** |
| `--no-tilt` | 显式关闭倾角约束 | 关 |
| `--no-kf` | 关闭因果卡尔曼（默认关：闭环里引入 0.4 帧滞后，mT 反而恶化到 23.74 m） | 关 |
| `--kf-sig-r F` | 卡尔曼测量噪声 σ | 2.5 |
| `--kf-sig-a F` | 卡尔曼过程噪声 σ | 2.0 |

### 6.9 范围

| 参数 | 含义 | 默认 |
|---|---|---|
| `--start-frame N` | 从第几帧开始 | 0 |
| `--max-frames N` | 最多处理多少帧（调试用） | −1（全部） |
| `-v` | 详细输出（透传渲染器，看分块剔除/投影诊断） | 关 |

---

## 7. 输出文件格式

### 7.1 位姿文件（`--out`）—— 核心输出

```
# 列: name lon lat alt roll pitch yaw qw qx qy qz nmatch ninl inl_ratio cossim reproj_rms conf
000000_0.png -112.0662373923 41.2054907833 2137.3130 0.5567 0.0473 88.1778 \
             0.718251 -0.003083 0.003777 0.695767 439 322 0.733 0.933 1.473 0.3276
```

| 列 | 含义 |
|---|---|
| `name` | 照片文件名 |
| `lon lat alt` | **WGS84 经纬高**（度, 度, 米） |
| `roll pitch yaw` | ZYX 欧拉角（度） |
| `qw qx qy qz` | 四元数（**正半球**，避免 q / −q 符号跳变） |
| `nmatch` | 匹配点对总数 |
| `ninl` | PnP 内点数 |
| `inl_ratio` | 内点率 = ninl / nmatch |
| `cossim` | 接受配对平均余弦相似度 |
| `reproj_rms` | 内点重投影 RMS（像素） |
| `conf` | **匹配置信度** = `inl_ratio × exp(−reproj_rms/2) × clamp(cossim,0,1)` ∈ [0,1] |

> `conf` 可用于**异常检测**：数值低 = 该帧匹配不可信。注意它是启发式指标，不是标定过的概率。

### 7.2 目标定位文件（`--probe-out`）

```
# 列: name u v lon lat alt   # ENU e n u  depth
000000_0.png 256 144 -112.0662553261 41.2053963433 1281.8567  # ENU -1.195 -2.166 1281.857  depth 853.50
```

| 列 | 含义 |
|---|---|
| `u v` | 图像像素坐标 |
| `lon lat alt` | 该像素对应的**全球 3D 地理坐标** |
| `ENU e n u` | 局部 ENU 米制坐标（东/北/天） |
| `depth` | 该像素的渲染深度（米） |

### 7.3 指标卡（stdout）

程序结束会打印：mT / mR / Recall@1m,3m,5m / Recall@(1m,1°)… / Completeness / FPS /
匹配数 / 各阶段耗时分解 / 三轴中位与平均绝对误差。

---

## 8. 成果展示索引

成果展示在 **`~/成果展示/`**，共 6 个部分 + 指标汇总 + 运行手册。

| 目录 / 文件 | 内容 | 建议用途 |
|---|---|---|
| `指标.txt` | 全部指标的单一汇总（含定义、硬件占用、版本改动） | 汇报时直接引用数字 |
| `运行手册与流程图.html` | 重启板子后怎么跑实验 + 数据在哪 + 三部分流程图（建图/渲染/定位）含耗时 | 交给别人复现 |
| `01_原始图片/` | 原始航拍图 1600×900、降采样输入 512×288 灰度 | 说明输入 |
| `02_高斯模型/` | 3DGS 模型总览图、特写、与真实图片对照 | 说明建图质量 |
| `03_深度图/` | 深度灰度图 + 三联图 + 深度图例（黑白，非伪彩） | 说明深度来源 |
| `04_特征点匹配/` | XFeat / ORB / SuperPoint 三组匹配可视化 + `匹配统计.csv` | 前端对比 |
| `05_位姿精度/` | `位姿误差总览.png`（3D 东北高轨迹）+ 逐帧卡片（6 张）+ 卡片总表 | 精度展示 |
| `06_demo/` | `demo.mp4` —— **完整 1067 帧、5.53 fps、无加速**、1024×676、192.9 s | 现场演示 |

### 关键文件绝对路径

```
~/成果展示/指标.txt
~/成果展示/运行手册与流程图.html
~/成果展示/06_demo/demo.mp4
~/results_20261012/pose_XFeat.txt       ← 位姿（1067 行）
~/results_20261012/probe_XFeat.txt      ← 目标定位
~/results_20261012/metrics_XFeat.txt    ← 指标卡全文
~/results_20261012/pose_ORB.txt         ← ORB 对照
~/results_20261012/lightglue_cmp.json   ← LightGlue 离线评估结果（另含 lg_1500.json / lg_512.json）
```

> `demo.mp4` 刻意**不做加速**，播放速度 = 实际端到端帧率，避免"演示效果造假"的质疑。

---

## 9. 脚本一览

| 脚本 | 位置 | 用途 |
|---|---|---|
| `run_exp.sh` | `~/` | **一键跑完整实验**：XFeat + ORB 各 1067 帧，结果落 `~/results_20261012/` |
| `gscheck.sh` | `~/gsplat/` | 板端环境一键体检（频率/服务/显示/数据） |
| `run.sh` | `~/gsplat/` | 通用运行脚本 |
| `locate_uav.sh` | `~/gsplat/` | 无人机定位启动脚本 |
| `locate_uav_fast.sh` | `~/gsplat/` | 同上，快速模式 |
| `bench_ab.sh` | `~/gsplat/` | 折叠路径 vs 标量路径 的公平基准 |
| `archive_legacy.sh` | `~/gsplat/` | 归档历史脚本/文件 |
| `sh/lock_freq.sh` | `~/gsplat/sh/` | 手动锁 GPU/CPU 频率为 performance |
| `sh/fan_perm.sh` | `~/gsplat/sh/` | 风扇固定转速 |
| `board_perf_setup.py` | **本机工作区**（不在板子上） | 一键恢复频率设置，支持 `--verify` / `--now` |

### 构建目标（`make`）

`make all` 会编译以下全部可执行文件：

| 目标 | 用途 |
|---|---|
| `build/gs_locate_uav` | ★ 无人机定位主程序（真实工况） |
| `build/gs_localize` | 自证闭环（用真值渲染当实拍图，验证 渲染→匹配→PnP 链） |
| `build/depthcheck` | 深度/渲染质量检查 |
| `build/calib_pilot` | PiLoT 官方管线标定 |
| `build/raster_bench` | 光栅化性能基准 |
| `build/camtest` / `build/rbtest` / `build/projcmp` | 相机 / 回读 / 投影对比测试 |
| `build/gsrender` | 纯渲染演示 |
| `build/loadtest` / `build/thermaltest` | 加载 / 温控测试 |

---

## 10. 代码阅读顺序

> 完整版见 **`~/gsplat/代码阅读指南.html`**、**`~/代码导览.md`**。
> 源码里每个函数上方都有 `◀GUIDE▶` 标记的说明注释块（干什么 / 参数 / 返回 / 谁调用）。

总原则：**先跑通的路径优先，先数据流后数学细节，先头文件后实现。**

| 步 | 读什么 | 时长 | 读完要能回答 |
|---|---|---|---|
| 0 | `camera.h` `renderer.h` `sorter.h` `xfeat_frontend.h` 的文件头 | 30 min | 渲染分几段？为什么必须排序？ |
| 1 | `gs_locate_uav.cpp` 的 `main()`（**按目录表跳读**） | 3 h | 首帧初值从哪来？渲染用哪帧位姿？ |
| 2 | `camera.h` → `camera.cpp` 的 `project_gaussian()` | 1.5 h | 为什么投影不能放 GPU？ |
| 3 | `sorter.h` → `sorter.cpp` 的 `sort_depth32_packed` / `should_reuse` | 1 h | 为什么不能用 tile 分组排序？ |
| 4 | `offscreen.cpp` → `renderer.h` → `renderer.cpp` 的 `render_frame()` | 2.5 h | 一个高斯怎么变成 4 个顶点？ |
| 5 | `xfeat_frontend.h` → `.cpp` 的 `extract()` / `match()` | 1.5 h | 为什么要加亚像素细化？ |
| 6 | `geo_pose.h`（全注释） | 1 h | 为什么要乘 `diag(1,-1,-1)`？ |
| 7 | `ply_loader.h` → `.cpp` 的 `load_ply_splats()` | 45 min | 哪些字段需要「解密」？ |
| 8 | `pose_post.h`、`projfast.cpp`（默认关，按需） | 按需 | — |

**主流程骨架**（`main()` 内，行号会漂移，以 `── 段落名 ──` 注释为准）：

```
初始化： 1 列照片 → 2 读真值 → 3 建坐标转换器 → 4 加载模型
        → 5 建渲染器(⚠️ EGL 上下文绑定本线程) → 6 内参 → 7 开输出文件
每帧：   8.1 读照片 → 8.2 初始位姿 → 8.3 render-and-compare ★核心
        → 8.4 误差 → 8.5 写输出 → 8.6 目标定位探针
收尾：   9 汇总指标卡
```

**最容易读错的一行**：反投影用的深度图是用**上一帧投递时的位姿** `cam_rb` 渲的，
必须写 `backproject(cam_rb, ...)`，不能写当前估计位姿（否则 mT 从 7 m 飙到 30 m+）。

---

## 11. 已知问题与未完成

### 运行注意事项（会踩的坑）

1. **强杀进程会卡死 GPU** —— 在 GPU 工作过程中 `kill -9` 会让 Mali 驱动卡住，
   之后所有 GL 程序永久阻塞在 `ppoll`，**只能重启板子**。
   跑长任务请用 `setsid nohup ... &` 脱离 SSH 会话。
2. **不要用 `pkill -f gs_locate_uav`** —— 会匹配到 SSH 自己的命令行，把会话杀掉。
   用 `pgrep -x gs_locate_uav` + `kill`。
3. **板卡重启会重置频率** —— governor 回到 `ondemand`，GPU 掉到 300 MHz。
   已由 `gsplat-perf.service` 自动处理；手动用 `board_perf_setup.py --now`。
4. **`rknnlite` 会破坏 `logging`** —— `from rknnlite.api import RKNNLite` 会把
   `logging._nameToLevel` 替换成残缺表，导致后续 `setLevel('WARNING')` 抛
   `ValueError`。Python 脚本中**必须先 import torch/kornia，再 import rknnlite**。
5. **本板没有 `/dev/rknpu` 节点** —— RKNPU 注册为 DRM 设备（`/dev/dri/renderD129`），
   不要因为找不到 `/dev/rknpu` 就以为 NPU 没加载。

### 未完成 / 待验证

- [ ] **10 FPS 未达到**。当前 5.53 FPS。剩余路径：把渲染器的仅 CPU 段
      （投影/排序/组装，约 77 ms）拆到工作线程 + 双缓冲，与 NPU 推理重叠。
      理论上限约 8.8 FPS。
- [ ] **倾角约束无法跑通全量 1067 帧**（第 314 帧渲染回读不返回），
      根因未完全定位，疑似渲染器填充率爆炸（`max_sigma_radius` 默认过松）。
      已按你的判断**默认关闭**（真实航拍有风扰/云台误差，假设不成立）。
- [ ] **未在真实飞行数据上验证**（目前是 AirZoo 序列）。
- [ ] **无可用的绝对尺度标定**：单目 SfM 有尺度歧义，GPS 拒止场景下缺少尺度来源。
- [ ] **`conf` 是启发式置信度**，不是标定过的概率；尚无 PnP 解析协方差。
- [ ] **SH 降到 1 阶无消融实验**（官方默认 3 阶）。
- [ ] **LightGlue 评估停在第一步**（离线精度已验证：mT 3.982 → 3.049 m，
      R@3m 33.5% → 49.0%，但 CPU 匹配 709.6 ms 不可用）。
      下一步是 NPU 移植可行性（ONNX → RKNN → 量化 → 测速）。
      决策阈值：NPU < 60 ms 值得做；60~100 ms 边际；> 150 ms 放弃。

### 工程结构待整理（**建议汇报后再做，现在别动**）

- `src/` 下有 8 个 `.bak-*` 文件（约 8.6 k 行，占 src 代码量约 40%），建议移入 `archive/`
- 没有 git —— 这是上一条的根因。建议 `git init` + `.gitignore`
- 顶层与 `src/` 各有一个 Makefile，构建入口不唯一
- 顶层脚本散落，建议收进 `scripts/`
- `data/` 有 3 个约 710 MB 的近似重复 PLY

---

## 12. 参考

### 论文与开源项目

- **3D Gaussian Splatting** — Kerbl et al., *SIGGRAPH 2023*
- **EWA Splatting** — Zwicker et al., *IEEE TVCG 2002*（`Σ_2D = J·W·Σ_3D·Wᵀ·Jᵀ`）
- **PiLoT** — `github.com/nudt-sawlab/PiLoT`（Render2Loc 范式、`pixloc/utils/transform.py`、
  `pixloc/utils/citygs/pose_convert.py` —— 本工程地理配准链逐字复刻自此）
- **CityGaussian** — 3DGS 大场景训练框架（本工程的建图流程）
- **XFeat** — Potje et al., *CVPR 2024*（`verlab/XFeat`）
- **diff-gaussian-rasterization** — INRIA（CUDA 参考实现，`computeCov3D` / `computeCov2DCUDA`）
- **LightGlue / LighterGlue** — Lindenberger et al.（评估中）

### 规范

- **WGS84** — 长半轴 6378137.0 m，扁率 1/298.257223563
- **ENU / ECEF** — 站心东北天 / 地心直角坐标系

---

*最后更新：2026-10-13 · 数据来源：板端实跑 `~/results_20261012`*
