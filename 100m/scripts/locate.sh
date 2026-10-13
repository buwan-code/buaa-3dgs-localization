#!/bin/bash
# ============================================================================
#  BUAA_location_100 —— BUAA 100 米低空 3DGS 视觉定位（自包含工作区）
#
#  ★ 本目录与 800 米高空数据集（airzoo / feicuiwan）完全隔离，互不干扰。
#
#  数据：BUAA_gspj，无人机约 100 m AGL 低空，152 帧
#        云台 3 帧一循环：俯仰 -45/-60/-90°，方位 52/-33/-83°
#        坐标系：以首图 RTK 点为原点的 ENU（米）⇒ --align 0 0 0
#
#  用法：
#    bash scripts/locate.sh                    # 全 152 帧（3DGS）
#    MODEL=.../buaa100_2dgs.ply bash scripts/locate.sh
#    MAXF=5 bash scripts/locate.sh             # 冒烟测试
#    PIPE=1 bash scripts/locate.sh             # 开递推（本数据集不建议）
# ============================================================================
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ★ offscreen EGL 需要 DISPLAY（远程/无头下必须显式给）
[ -n "${DISPLAY:-}" ] || export DISPLAY=:0

MODEL=${MODEL:-$ROOT/inputs/models/buaa100_3dgs.ply}
QUERY=$ROOT/inputs/query/buaa100/images
PREDIR=$ROOT/inputs/query/buaa100/images_pre
GT=$ROOT/inputs/query/buaa100/gt.txt

# ★ 本数据集的真实内参（去畸变后 PINHOLE @ 1566x1176）
REF_LON=116.08676269
REF_LAT=39.41497656
REF_ALT=97.037
FX=1120.0768
FY=1121.4744
CX=783.0
CY=588.0
SRC_W=1566
SRC_H=1176

# 渲染尺寸：XFeat 的 RKNN 模型输入固定 512x288，--feat xfeat 时不可改
W=${W:-512}
H=${H:-288}

FEAT=${FEAT:-xfeat}
RAD=${RAD:-2}
MAXF=${MAXF:-0}
INIT=${INIT:-gt}
PERT=${PERT:-0}
# ★ 本数据集「同片地面 / 3 个俯仰角一组 / 组间平移 6~9 m」，
#   组内相邻帧是纯旋转大跳变（yaw 52→-33→-83，Roll 0→180）。
#   --pipeline 的「用上一帧渲染递推下一帧」前提不成立，默认关闭。
PIPE=${PIPE:-0}
# ★ PnP 求解器：sqpnp 已是程序默认（iterative 会发散），此处显式写上便于追溯
PNP=${PNP:-sqpnp}
TAG=${TAG:-buaa100}
OUTDIR=$ROOT/runs/$(date +%Y%m%d-%H%M%S)_$TAG
mkdir -p "$OUTDIR"

BIN=$ROOT/build/gs_locate_uav
if [ ! -x "$BIN" ]; then
  echo "!! 二进制不存在，先编译： make -C $ROOT build/gs_locate_uav -j4"
  exit 1
fi

ARGS=(
  "$MODEL"
  --query "$QUERY" --pre-dir "$PREDIR" --gt "$GT"
  --ref "$REF_LON" "$REF_LAT" "$REF_ALT"
  --align 0 0 0
  --fx "$FX" --fy "$FY" --cx "$CX" --cy "$CY"
  --src-w "$SRC_W" --src-h "$SRC_H"
  --w "$W" --h "$H"
  --yaw-offset 0
  --iters 1
  --feat "$FEAT" --min-alpha-radius "$RAD" --orb-n 1500
  --pnp-flags "$PNP"
  --out "$OUTDIR/poses.txt"
)
[ "$PIPE" != "0" ] && ARGS+=(--pipeline)
[ "$INIT" = "gt" ]   && ARGS+=(--init-from-gt)
[ "$INIT" = "prev" ] && ARGS+=(--pert-pos "$PERT" --pert-ang "$PERT")
[ "$MAXF" != "0" ]   && ARGS+=(--max-frames "$MAXF")

# 记录本次实验配置（自解释、可追溯）
cat > "$OUTDIR/CONFIG.md" <<EOF
# 实验: $TAG

- 时间戳: \`$(date +%Y%m%d-%H%M%S)\`
- 工作区: \`$ROOT\`
- 数据集: BUAA_gspj（约 100 m AGL 低空）

## 输入
\`\`\`
模型      : $MODEL
          $(stat -c '%s B  %y' "$MODEL" 2>/dev/null)
照片目录  : $QUERY
预转灰度  : $PREDIR
真值      : $GT
\`\`\`

## 参数
\`\`\`
geo/pose  : ref=($REF_LON, $REF_LAT, $REF_ALT)  yaw_offset=0
内参      : fx=$FX fy=$FY cx=$CX cy=$CY @ ${SRC_W}x${SRC_H}
初始化    : mode=$INIT  pert_pos=$PERT m  pert_ang=$PERT deg
调度      : iters=1  pipeline=$PIPE  align=0,0,0
渲染      : min_alpha_radius=$RAD  W×H=${W}×${H}
特征      : $FEAT  orb_n=1500
PnP       : $PNP
\`\`\`
EOF

echo "模型 : $MODEL"
echo "影像 : $QUERY"
echo "输出 : $OUTDIR"
echo "------------------------------------------------------------"

set +e
"$BIN" "${ARGS[@]}" 2>&1 | tee "$OUTDIR/run.log"
RC=${PIPESTATUS[0]}
set -e
echo "------------------------------------------------------------"
echo "退出码 $RC；位姿 -> $OUTDIR/poses.txt"
grep -E '^(Completeness|Median|Recall|FPS|匹配数|三轴)' "$OUTDIR/run.log" | tee "$OUTDIR/metrics.txt" || true
