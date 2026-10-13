#!/bin/bash
# ============================================================================
#  locate_uav_fast.sh —— 板端无人机视觉定位（调优档，2026-10-09）
#
#  与 locate_uav.sh 的唯一区别：默认走「精度优先」调优档 —— 实测它比原始
#  配置**又准又快**（原始 iters=3 + r=0.5 每帧要渲染 3 次，纯属浪费）。
#
#  实测（feicuiwan / 完整 300 帧 / 512x288 / --init-from-gt / --pre-dir）：
#    档位        中位平移  中位旋转  R@1m     R@3m   R@5m   完整率  纯算法FPS  墙钟
#    原始         0.857 m  0.628°   66.00%   100%   100%   100%    1.76      177 s
#    balanced     0.797 m  0.630°   67.67%   100%   100%   100%    8.39       36 s
#    fast         0.906 m  0.633°   62.33%   100%   100%   100%   12.91       28 s
#    max          0.916 m  0.625°   60.33%   100%   100%   100%   14.32       25 s
#  （PiLoT 主结果参考：中位平移 1.37 m）
#
#  ★ 四档里 balanced 是"既比原来准、又快 4.8x"的默认档；fast/max 是拿
#    少量 R@1m 换帧率。三档的 R@3m/R@5m/完整率都是 100%。
#
#  用法：
#    ./locate_uav_fast.sh                  # balanced，全 300 帧
#    ./locate_uav_fast.sh 60               # 前 60 帧快速验证
#    PROFILE=fast ./locate_uav_fast.sh     # 速度档
#    PROFILE=max  ./locate_uav_fast.sh     # 极限档
#    单旋钮覆盖（优先于 PROFILE）：LOD=600000 MIN_RAD=3 ORB_N=1000 ./locate_uav_fast.sh
#    LOD=0 MIN_RAD=0.5 ./locate_uav_fast.sh   # 全质量（等于关掉降采样）
# ============================================================================
set -e
cd "$(dirname "$0")"

PROFILE=${PROFILE:-balanced}
case "$PROFILE" in
  balanced) D_LOD=0;      D_RAD=2 ;;   # 0.80 m / R@1m 67.7% /  8.4 FPS
  fast)     D_LOD=900000; D_RAD=3 ;;   # 0.91 m / R@1m 62.3% / 12.9 FPS
  max)      D_LOD=500000; D_RAD=3 ;;   # 0.92 m / R@1m 60.3% / 14.3 FPS
  *) echo "!! 未知 PROFILE=$PROFILE（可选 balanced|fast|max）"; exit 2 ;;
esac
LOD=${LOD:-$D_LOD}
MIN_RAD=${MIN_RAD:-$D_RAD}
ORB_N=${ORB_N:-1500}

MODEL=${MODEL:-data/feicuiwan.ply}
QUERY=${QUERY:-data/query/images/3dgs_test}
PREDIR=${PREDIR:-data/query/images/3dgs_test_512}
GT=${GT:-data/query/poses/3dgs_test.txt}
OUT=${OUT:-outputs/uav/est_fast.txt}
REF_LON=${REF_LON:-112.990406}
REF_LAT=${REF_LAT:-28.292367}
REF_ALT=0
YAW_OFFSET=${YAW_OFFSET:-270}
PITCH=${PITCH:-15}
MAXF=${1:-0}

ARGS=()
[ "$MAXF" != "0" ] && ARGS+=(--max-frames "$MAXF")
[ "$LOD" != "0" ] && ARGS+=(--lod "$LOD")

for f in "$MODEL" "$QUERY" "$GT"; do
  [ -e "$f" ] || { echo "!!! 找不到 $f"; exit 1; }
done
[ -n "$DISPLAY" ] || export DISPLAY=:0

# 预解码灰度目录存在就用：4K PNG 解码约 190 ms/帧，占端到端 20%+
PRE_ARGS=()
if [ -d "$PREDIR" ]; then PRE_ARGS+=(--pre-dir "$PREDIR")
else echo "[i] 未找到预解码目录 $PREDIR，将直接读原图（会慢很多）"; fi

mkdir -p "$(dirname "$OUT")"
N=$(ls "$QUERY"/*.png 2>/dev/null | wc -l)
echo "================================================================"
echo " 板端无人机视觉定位  |  PROFILE=$PROFILE"
echo "   模型     : $MODEL"
echo "   照片     : $QUERY ($N 张)"
echo "   档位参数 : lod=$LOD  min_alpha_radius=$MIN_RAD  orb_n=$ORB_N  pipeline=on/iters=1"
echo "   地理/姿态: ref=($REF_LON, $REF_LAT, $REF_ALT)  yaw_offset=$YAW_OFFSET  pitch=$PITCH"
echo "   输出     : $OUT"
echo "================================================================"

./build/gs_locate_uav "$MODEL" \
    --query "$QUERY" "${PRE_ARGS[@]}" --gt "$GT" \
    --ref "$REF_LON" "$REF_LAT" "$REF_ALT" \
    --align 0 0 0 \
    --yaw-offset "$YAW_OFFSET" --pitch-override "$PITCH" \
    --init-from-gt --iters 1 --pipeline \
    --min-alpha-radius "$MIN_RAD" --orb-n "$ORB_N" \
    --out "$OUT" "${ARGS[@]}"