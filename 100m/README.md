# BUAA_location_100 —— BUAA 100 米低空 3DGS 视觉定位（自包含工作区）

> 建立于 2026-10-13。**本目录与 800 米高空数据集（airzoo / feicuiwan）
> 完全隔离，互不干扰。** 所有文件实体存储，无软链接。

---

## 一、这是什么

BUAA_gspj 数据集的 3DGS 视觉定位实验工作区。渲染-比对式定位：
用估计位姿从 3DGS 地图渲染参考图 → 与真实相机图做特征匹配 → PnP 解 6DoF 位姿。

| 项目 | 值 |
|---|---|
| 飞行高度 | **约 100 m AGL（低空）** |
| 影像帧数 | 152 |
| 分辨率 | 1566×1176（去畸变后 PINHOLE） |
| 内参 | fx=1120.0768 fy=1121.4744 cx=783.0 cy=588.0 |
| ENU 原点 | lon=116.08676269 lat=39.41497656 alt=97.037 |
| 坐标系 | 以首图 RTK 点为原点的 ENU（米）⇒ `--align 0 0 0` |
| GPS | RTK，水平精度 0.003~0.004 m |

### ★ 数据集的特殊结构（务必先读）

云台角是**严格 3 帧一循环**：

```
GimbalPitch : -45, -60, -90, -45, -60, -90, ...   ← 俯仰角
GimbalRoll  :   0, 180, 180,   0, 180, 180, ...   ← 万向锁（下倾 90° 时必为 180）
GimbalYaw   :  52, -33, -83,  52, -33, -83, ...   ← 方位角
```

- 无人机**连续前飞**，相邻帧水平位移均匀 6~9 m（不是停在原地）
- 所谓"同一位置点三张"的物理含义：**45°/60°/90° 三张覆盖同一片地面**
  （-90° 拍正下方 80×100 m；-45° 从斜前方 68 m 看，覆盖 274×272 m；两者在正下方重叠）
- **⇒ `--pipeline` 必须关闭**：组内相邻帧是纯旋转大跳变，
  "用上一帧渲染递推下一帧"的前提不成立。脚本默认 `PIPE=0`。

---

## 二、目录结构

```
BUAA_location_100/
├── README.md                     本文件
├── Makefile                      编译规则
├── src/                          源码（含 2026-10-13 的 3 处 bug 修复）
├── build/                        编译产物
│   └── gs_locate_uav             ★ 主程序
├── models/                       NPU 推理模型
│   ├── xfeat_288x512_int8.rknn   ★ 特征前端（固定输入 512×288）
│   └── superpoint_encoder.rknn
├── inputs/
│   ├── models/
│   │   ├── buaa100_3dgs.ply      ★ 3DGS 地图（744 MB，300 万高斯，SH degree 3）
│   │   ├── buaa100_3dgs.splat    （96 MB，备用格式）
│   │   ├── buaa_gspj_3dgs.info.json
│   │   └── buaa100_2dgs.ply      ← 待补（训练完成后导出）
│   └── query/buaa100/
│       ├── images/               152 张查询图（原图）
│       ├── images_pre/           152 张预转灰度（与 images 同名）
│       ├── gt.txt                ★ 真值：name lon lat alt roll pitch yaw
│       ├── name_map.txt          文件名映射
│       └── check.txt
├── scripts/
│   └── locate.sh                 ★ 实验启动脚本（路径全部指向本目录）
├── runs/                         实验输出（一次实验一个目录，自动生成 CONFIG.md）
└── logs/  outputs/               预留
```

---

## 三、怎么用

### 编译（一般不用重编，build/ 已有）

```bash
cd ~/BUAA_location_100
make build/gs_locate_uav -j4
```

### 跑实验

```bash
cd ~/BUAA_location_100

bash scripts/locate.sh                       # 全 152 帧（默认 3DGS）
MAXF=5 bash scripts/locate.sh                # 冒烟测试：只跑前 5 帧
TAG=mytest bash scripts/locate.sh            # 自定义实验标签

# 换 2DGS 模型（训练导出后）
MODEL=$PWD/inputs/models/buaa100_2dgs.ply bash scripts/locate.sh

# 可调项
PIPE=1 bash scripts/locate.sh                # 开递推（本数据集不建议）
PNP=iterative bash scripts/locate.sh         # 换 PnP 求解器
W=1024 H=576 bash scripts/locate.sh          # 改渲染尺寸（注意 XFeat 固定 512x288）
```

每次实验生成 `runs/<时间戳>_<标签>/`：
- `CONFIG.md` — 输入 + 全部参数（自解释、可追溯）
- `run.log` — 完整输出
- `metrics.txt` — 指标卡
- `poses.txt` — 逐帧位姿

---

## 四、★ 2026-10-13 的三处 bug 修复（都在 src/ 里）

### 1. 欧拉角口径统一为标准 ZYX

板卡原有**两套互相矛盾**的口径：
- 渲染路径 `euler_xyz_to_mat` = `Rz(y)·Ry(p)·Rx(r)`（标准 ZYX）← **对的**
- 评价路径 `pose_mat_of`/`euler_from_R` = `Rz(y)·Rx(p)·Ry(r)` ← **roll/pitch 互换，错的**

判别实验：gt(第1帧) = (roll 44.244, pitch 0.967, yaw -52.665)，
DJI 云台真值 = 方位 51.67° / 下倾 45.75°：

```
标准 ZYX  Rz(y)Ry(p)Rx(r)  →  方位  51.67°  下倾 45.75°   ✅ 吻合
旧口径    Rz(y)Rx(p)Ry(r)  →  方位 -36.34°  下倾 45.75°   ❌ 差 88°
```

已把 `euler_from_R` / `pose_mat_of` 统一为标准 ZYX。

### 2. `pose_of()` 取光轴时列/行取错

`world_to_cam` 是**列主序**存 `R_wc`；光轴 = `R_cw` 第 3 列 = `R_wc` **第 3 行**
= `m[2], m[6], m[10]`。原代码写成 `m[8], m[9], m[10]`（第 3 列），
只有 z 分量凑巧相同，**水平方位全错**（实测第 1 帧方位 -178.7° vs 真值 51.7°）。

同一文件 1640/1876 行（tilt 模式）自己写的就是 `m[2],m[6],m[10]` ——
仅 `pose_of()` 一处手误。

### 3. PnP 求解器默认 iterative → sqpnp

`iterative`(=LM) 需好初值、易陷局部极小：前 3 帧发散到 148~200 m
（重投影 RMS 达 5135 px）。`sqpnp` 是全局最优闭式解、无需初值，同样 3 帧 < 1.6 m。
`Args::pnp_flags` 默认值与 `pnp_flag` 初值均已改为 sqpnp。

---

## 五、当前结果（3DGS，152 帧全跑）

```
Completeness           : 100.00 %   (152/152)
Median Translation Err : 0.315 m    (mean 0.360 / rmse 0.416 / max 1.523)
Recall@1m              : 99.34 %    (151/152)
Recall@3m              : 100.00 %   (152/152)
Recall@(3m,3°)         : 100.00 %
Median Rotation Err    : 0.162 °    (mean 0.195 / max 0.623)
三轴中位绝对误差        : roll -0.000°  pitch 0.008°  yaw -0.002°
FPS (端到端)           : 4.25  (235 ms/帧，含 4K 解码)
FPS (纯算法)           : 4.99  (200 ms/帧)
匹配数                 : 平均 721 (364~904)
```

修复前：中位旋转误差 173.4°、中位位置误差 148.5 m。

### 2DGS 模型（2026-10-13 17:22 补入）

`inputs/models/buaa100_2dgs.ply`（444 MB，md5 `8fe7e5fedbbdbeb05f6949ab013c6c38`）已就位。
由 6000D 训练的 `epoch=223-step=30000.ckpt` 经 `tools/export_2dgs_native_param.py` 导出，
原生 37 列（x y z | scale_0 scale_1 | rot_0..3 | opacity | f_dc_0..2 | f_rest_0..23），
无 `scale_2`、SH rest 24 列 —— 板端 `ply_loader.cpp` 靠这两点识别 2DGS。

实测（同 `scripts/locate.sh`，`--yaw-offset 0 --iters 1`，PIPE 关）：

| 帧数 | 中位平移 | 中位旋转 | Recall@1m | 匹配数 |
|---|---|---|---|---|
| 5   | 0.177 m | 0.191° | 100% | 751 |
| 20  | 0.238 m | 0.167° | 100% | 824 |

与 3DGS 那份（0.315 m / 0.162° @152 帧）精度同档，匹配数反而更多。

---

## 六、待办

- [x] 2DGS 训练完成后，导出原生 37 列 PLY → `inputs/models/buaa100_2dgs.ply`
      （2DGS 无 `scale_2`，SH rest 为 24 个而非 45）—— 已完成
- [ ] 跑 2DGS 的完整 152 帧实验，与 3DGS 逐项对比精度 / 速度
- [ ] 可选：重导出 `xfeat_1024x576_int8.rknn` 以支持 4:3 高分辨率

---

## 七、相关但独立的目录

- `~/BUAA_location_800/` —— airzoo（美国乡村，800 m 高空）专用工作区（原 `gsplat`）
