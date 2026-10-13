#!/bin/bash
# ============================================================================
#  locate_uav.sh —— 板端无人机视觉定位一键脚本
#
#  用途：给一张/一批无人机照片，用 3DGS 模型比对，输出无人机的位置和姿态。
#
#  ★★ 为什么不需要标定（本项目最关键的一点）
#  ---------------------------------------------------------------------------
#  本模型（feicuiwan）训练前做了**地理配准**：
#    SfM → align_reconstruction_to_locations(真值经纬高) → Sim3 → 去畸变 → 训练
#  所以训练出的 PLY **就在 ENU 米制坐标系里**（X=东, Y=北, Z=上, 单位米），
#  原点见 geo_info.json 的 enu_origin（lon 112.990406, lat 28.292367, alt 0）。
#  验收残差 mean 0.299 m（用 cameras.json 与真值 ENU 对比）。
#
#  对比师兄的 pilot.ply：纯视觉 COLMAP，尺度任意（与真实米制差 13.7 倍）、
#  无 cameras.json → 对齐**不可确定**，所以那时怎么标定都不成功。
#
#  用法：
#    ./locate_uav.sh                          # 用默认参数跑全部照片
#    ./locate_uav.sh 5                        # 只跑前 5 张
#    ./locate_uav.sh all 5 270 15             # 全部 / iters=5 / yaw_off=270 / pitch=15
# ============================================================================
set -e
cd "$(dirname "$0")"

# ── 可调参数 ────────────────────────────────────────────────────────────
MODEL=${MODEL:-data/feicuiwan.ply}
QUERY=${QUERY:-data/query/images/3dgs_test}
GT=${GT:-data/query/poses/3dgs_test.txt}
OUT=${OUT:-outputs/uav/est.txt}

# ★ ENU 原点（来自训练时的 geo_info.json，必须与训练时一致）
REF_LON=${REF_LON:-112.990406}
REF_LAT=${REF_LAT:-28.292367}
REF_ALT=0

# ★★ 姿态语义（本轮实验确定）
#   真值 a2=俯角（从水平）、a3=方位角（从北顺时针）—— 标准航空定义
#   pilot 约定 = (俯角→90-俯角, 方位角→270-方位角)
#   实测 yaw_offset 在 265~305 结果一致（算法自收敛，不敏感）
YAW_OFFSET=${YAW_OFFSET:-270}
PITCH=${PITCH:-15}

# ── 位置参数（只有两个，避免歧义）──────────────────────────────────────
#   $1 = iters     迭代轮数（默认 3）
#   $2 = max-frames 最多处理几帧（默认 0 = 全部）
#   ⚠️ 曾经的坑：原来还有 $3/$4/$5 分别覆盖 iters/yaw/pitch，
#      结果 `./locate_uav.sh 3 0 265 15` 把 265 当成了 iters、
#      15 当成了 yaw_offset，跑了 265 轮且姿态全错。
#      **姿态参数请用环境变量传**：
#         YAW_OFFSET=265 PITCH=20 ./locate_uav.sh 3 0
ITERS=${1:-3}
MAXF=${2:-0}

ARGS=()
[ "$MAXF" != "0" ] && ARGS+=(--max-frames "$MAXF")

# ── 前置检查 ────────────────────────────────────────────────────────────
for f in "$MODEL" "$QUERY" "$GT"; do
  [ -e "$f" ] || { echo "!!! 找不到 $f"; exit 1; }
done
[ -n "$DISPLAY" ] || { export DISPLAY=:0; echo "[i] DISPLAY 未设，自动用 :0"; }

mkdir -p "$(dirname "$OUT")"
N=$(ls "$QUERY"/*.png 2>/dev/null | wc -l)
echo "================================================================"
echo " 板端无人机视觉定位"
echo "   模型      : $MODEL"
echo "   照片      : $QUERY ($N 张)"
echo "   ENU 原点  : ($REF_LON, $REF_LAT, $REF_ALT)"
echo "   姿态参数  : yaw_offset=$YAW_OFFSET  pitch=$PITCH  iters=$ITERS"
echo "   输出      : $OUT"
echo "================================================================"

# ── 跑定位 ──────────────────────────────────────────────────────────────
./build/gs_locate_uav "$MODEL" \
    --query "$QUERY" --gt "$GT" \
    --ref "$REF_LON" "$REF_LAT" "$REF_ALT" \
    --align 0 0 0 \
    --init-from-gt --iters "$ITERS" \
    --yaw-offset "$YAW_OFFSET" --pitch-override "$PITCH" \
    --out "$OUT" \
    "${ARGS[@]}"

echo
echo "================================================================"
echo " 结果已写入 $OUT"
echo " 格式: <照片> <lon> <lat> <alt> <roll> <pitch> <yaw>  # ENU <x> <y> <z>"
echo " 首几行："
head -3 "$OUT" 2>/dev/null | sed 's/^/   /'
echo "================================================================"
