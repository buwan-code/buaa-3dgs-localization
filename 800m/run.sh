#!/bin/bash
# ============================================================================
#  run.sh —— 实验封装器（2026-10-09）
#
#  解决的问题：以前每次实验的输入散落、输出混在一起（outputs/ 下几十个
#  目录与 txt 平铺、/tmp 里堆 50+ 个、参数只体现在文件名里），事后无法
#  判断"这个结果是用哪个模型、哪份数据、什么参数跑出来的"。
#
#  现在：**一次实验 = 一个独立目录**，输入输出全部固定在内，自带说明书。
#
#  ── 用法 ──────────────────────────────────────────────────────────────
#    ./run.sh -t <标签> [-p balanced|fast|max] [-- <额外参数...>]
#
#    ./run.sh -t balanced-300f                    # 默认档，全 300 帧
#    ./run.sh -t fast-300f -p fast                # 速度档
#    ./run.sh -t smoke-60f -- --max-frames 60     # 冒烟测试
#    ./run.sh -t pert3 -- --pert-pos 3 --pert-ang 3 --seed 42
#    ./run.sh -l                                  # 列出全部实验
#    ./run.sh -s <标签或时间戳>                    # 查看某次实验的指标
#
#  ── 目录规范 ──────────────────────────────────────────────────────────
#   inputs/                     ★ 实验输入的唯一来源（软链到 data/，不搬大文件）
#     models/<name>.ply
#     query/<dataset>/{images,images_512,gt.txt}
#
#   runs/<时间戳>_<标签>/        ★ 一次实验 = 一个目录，互不干扰
#     CONFIG.md                 输入清单 + 全部参数 + 指纹（自解释、可追溯）
#     cmd.sh                    实际执行的完整命令（一键复现）
#     run.log                   完整 stdout
#     metrics.txt               指标卡（Med TE/RE、R@1,3,5、FPS…）
#     table.txt                 逐帧表格
#     out/poses.txt             位姿输出
#     dump/                     视觉 dump（仅传 --dump 时非空）
#
#   runs/INDEX.md               ★ 全部实验总索引（一行一次，含关键指标）
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)
RUNS="$ROOT/runs"
INPUTS="$ROOT/inputs"

# ── 自检：初始化 inputs/（幂等）────────────────────────────────────────
init_inputs() {
  mkdir -p "$INPUTS/models" "$INPUTS/query/3dgs_test"
  local L
  L="$INPUTS/models/feicuiwan.ply"
  [ -e "$L" ] || ln -s "$ROOT/data/feicuiwan.ply" "$L"
  L="$INPUTS/query/3dgs_test/images"
  [ -e "$L" ] || ln -s "$ROOT/data/query/images/3dgs_test" "$L"
  L="$INPUTS/query/3dgs_test/images_512"
  [ -e "$L" ] || ln -s "$ROOT/data/query/images/3dgs_test_512" "$L"
  L="$INPUTS/query/3dgs_test/gt.txt"
  [ -e "$L" ] || ln -s "$ROOT/data/query/poses/3dgs_test.txt" "$L"
}

# ── 子命令：列表 / 查看 ────────────────────────────────────────────────
if [ "${1:-}" = "-l" ] || [ "${1:-}" = "--list" ]; then
  [ -f "$RUNS/INDEX.md" ] && cat "$RUNS/INDEX.md" || echo "(还没有实验记录)"
  exit 0
fi
if [ "${1:-}" = "-s" ] || [ "${1:-}" = "--show" ]; then
  key="${2:?用法: ./run.sh -s <标签或时间戳>}"
  d=$(ls -dt "$RUNS"/*"$key"* 2>/dev/null | head -1)
  [ -n "$d" ] || { echo "找不到匹配 '$key' 的实验"; exit 1; }
  echo "=== $d ==="; cat "$d/metrics.txt" 2>/dev/null
  echo; echo "--- CONFIG ---"; cat "$d/CONFIG.md" 2>/dev/null
  exit 0
fi

# ── 参数解析 ───────────────────────────────────────────────────────────
TAG=""; PROFILE=balanced; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    -t|--tag)     TAG="${2:?-t 需要标签}"; shift 2 ;;
    -p|--profile) PROFILE="${2:?-p 需要档位}"; shift 2 ;;
    --)           shift; EXTRA=("$@"); break ;;
    *)  echo "未知参数: $1（额外参数请放在 -- 之后）"; exit 2 ;;
  esac
done
[ -z "$TAG" ] && { echo "必须指定 -t <标签>（用于命名实验目录）"; exit 2; }

case "$PROFILE" in
  balanced) D_LOD=0;      D_RAD=2 ;;   # 0.80 m / R@1m 67.7% /  8.4 FPS
  fast)     D_LOD=900000; D_RAD=3 ;;   # 0.91 m / R@1m 62.3% / 12.9 FPS
  max)      D_LOD=500000; D_RAD=3 ;;   # 0.92 m / R@1m 60.3% / 14.3 FPS
  *) echo "!! 未知 PROFILE=$PROFILE（可选 balanced|fast|max）"; exit 2 ;;
esac
LOD=${LOD:-$D_LOD}; MIN_RAD=${MIN_RAD:-$D_RAD}; ORB_N=${ORB_N:-1500}
# REUSE_EPS：排序复用阈值。默认空 = 用程序内默认（1e-4，几乎静止才复用）。
# 实测帧间真实位移中位仅 7 mm，但 PnP 结果帧间抖动可达 0.4 m；
# 提到 0.15~0.3 可命中约一半帧、省 ~10 ms（3DGS 半透明 → 顺序小错无视觉差异）。
# 需自行验证精度后再作为默认。
REUSE_EPS=${REUSE_EPS:-}

init_inputs
# ★★ 2026-10-13：本工作区已专用于 **airzoo（美国乡村，800 m 高空）**。
#    默认值全部改为 airzoo；改回飞猪湾用 FEICUIWAN=1 或显式传环境变量。
if [ "${FEICUIWAN:-0}" = "1" ]; then
  MODEL=${MODEL:-$INPUTS/models/feicuiwan.ply}
  QUERY=${QUERY:-$INPUTS/query/3dgs_test/images}
  PREDIR=${PREDIR:-$INPUTS/query/3dgs_test/images_512}
  GT=${GT:-$INPUTS/query/3dgs_test/gt.txt}
  REF_LON=${REF_LON:-112.990406}
  REF_LAT=${REF_LAT:-28.292367}
  REF_ALT=${REF_ALT:-0}
  FX=${FX:-2700}; FY=${FY:-2700}
  CX=${CX:-1915.7}; CY=${CY:-1075.1}
  SRC_W=${SRC_W:-3840}; SRC_H=${SRC_H:-2160}
  YAW_OFFSET=${YAW_OFFSET:-270}
  PITCH=${PITCH:-15}
else
  # ── airzoo（美国乡村，800 m）默认 ─────────────────────────────────
  MODEL=${MODEL:-$INPUTS/models/airzoo_citygs_3dply.ply}
  QUERY=${QUERY:-$INPUTS/query/airzoo/images}
  PREDIR=${PREDIR:-$INPUTS/query/airzoo/images_512}
  GT=${GT:-$INPUTS/query/airzoo/gt.txt}
  REF_LON=${REF_LON:--112.0662410756}
  REF_LAT=${REF_LAT:-41.2054158383}
  REF_ALT=${REF_ALT:-0}
  FX=${FX:-1931.33}; FY=${FY:-1931.33}
  CX=${CX:-800}; CY=${CY:-450}
  SRC_W=${SRC_W:-1600}; SRC_H=${SRC_H:-900}
  YAW_OFFSET=${YAW_OFFSET:-0}
  PITCH=${PITCH:--1}
fi
# 初始位姿来源：gt = 每帧用真值（验证路径）；prev = 仅首帧用真值+扰动，之后逐帧递推
INIT_MODE=${INIT_MODE:-gt}
PERT_POS=${PERT_POS:-0}      # 首帧位置扰动半径（米），模拟机载 GPS 粗定位
PERT_ANG=${PERT_ANG:-0}      # 首帧姿态扰动（度），三轴各 ±
THREADS=${THREADS:-}

for f in "$MODEL" "$QUERY" "$GT"; do
  [ -e "$f" ] || { echo "!!! 缺少输入: $f"; exit 1; }
done
[ -n "${DISPLAY:-}" ] || export DISPLAY=:0

# ── 建本次实验目录（唯一写入点）──────────────────────────────────────
TS=$(date +%Y%m%d-%H%M%S)
# ⚠️ 用 printf 而非 echo：echo 会带换行，tr 会把换行也转成 '_'，
#    导致目录名末尾多一个下划线（实测踩过）。
SLUG=$(printf '%s' "$TAG" | tr -c 'A-Za-z0-9._-' '_')
RUNDIR="$RUNS/${TS}_${SLUG}"
mkdir -p "$RUNDIR/out" "$RUNDIR/dump"

ARGS=()
[ "$LOD" != "0" ] && ARGS+=(--lod "$LOD")
[ -d "$PREDIR" ] && ARGS+=(--pre-dir "$PREDIR")
[ -n "$THREADS" ] && ARGS+=(--threads "$THREADS")
[ -n "$REUSE_EPS" ] && ARGS+=(--reuse-eps "$REUSE_EPS")

# ★ POSE_ARGS：PITCH<=0 表示真值俯仰角可用，不加 override
POSE_ARGS=(--yaw-offset "$YAW_OFFSET")
[ "$(printf '%.0f' "$PITCH")" -gt 0 ] 2>/dev/null && POSE_ARGS+=(--pitch-override "$PITCH")
# ★ INIT_ARGS：gt 模式每帧用真值；prev 模式仅首帧用真值+扰动，之后逐帧递推
INIT_ARGS=(--iters 1 --pipeline --pert-pos "$PERT_POS" --pert-ang "$PERT_ANG")
if [ "$INIT_MODE" = "gt" ]; then INIT_ARGS+=(--init-from-gt); fi

BIN="./build/gs_locate_uav"
CMD=( "$BIN" "$MODEL"
      --query "$QUERY" --gt "$GT"
      --ref "$REF_LON" "$REF_LAT" "$REF_ALT"
      --align 0 0 0
      --fx "$FX" --fy "$FY" --cx "$CX" --cy "$CY"
      --src-w "$SRC_W" --src-h "$SRC_H"
      "${POSE_ARGS[@]}"
      "${INIT_ARGS[@]}"
      --min-alpha-radius "$MIN_RAD" --orb-n "$ORB_N"
      --out "$RUNDIR/out/poses.txt"
      "${ARGS[@]}" "${EXTRA[@]}" )

# ── 输入指纹（size + mtime + md5）────────────────────────────────────
fp() { # $1=路径 → "路径 大小 时间戳 md5"
  local p="$1"
  if [ -e "$p" ]; then
    local sz mt md
    sz=$(stat -Lc %s "$p"); mt=$(stat -Lc %y "$p" | cut -d. -f1)
    md=$(md5sum "$(readlink -f "$p")" 2>/dev/null | cut -c1-32)
    echo "$(basename "$p")  ${sz}B  ${mt}  md5=${md:-NA}"
  else echo "$(basename "$p")  <缺失>"; fi
}
NIMG=$(ls "$QUERY"/*.png 2>/dev/null | wc -l)
NPRE=$(ls "$PREDIR"/*.png 2>/dev/null | wc -l)

# ── CONFIG.md（自解释说明书）──────────────────────────────────────────
{
  echo "# 实验: $TAG"
  echo
  echo "- 时间戳: \`$TS\`"
  echo "- 目录: \`runs/${TS}_${SLUG}\`"
  echo "- 档位: **$PROFILE**  (lod=$LOD, min_alpha_radius=$MIN_RAD, orb_n=$ORB_N)"
  echo
  echo "## 输入（唯一来源 inputs/，软链到 data/）"
  echo
  echo '```'
  echo "模型      : $MODEL"
  echo "          $(fp "$MODEL")"
  echo "照片目录  : $QUERY   ($NIMG 张)"
  echo "预转灰度  : $PREDIR   ($NPRE 张)"
  echo "真值      : $GT"
  echo "          $(fp "$GT")"
  echo '```'
  echo
  echo "## 参数"
  echo
  echo '```'
  echo "geo/pose  : ref=($REF_LON, $REF_LAT, $REF_ALT)  yaw_offset=$YAW_OFFSET  pitch=$PITCH"
  echo "内参      : fx=$FX fy=$FY cx=$CX cy=$CY @ ${SRC_W}x${SRC_H}"
  echo "初始化    : mode=$INIT_MODE  pert_pos=$PERT_POS m  pert_ang=$PERT_ANG deg"
  echo "调度      : iters=1  pipeline=on  init-from-gt  align=0,0,0"
  echo "渲染      : ${LOD}(LOD)  min_alpha_radius=$MIN_RAD  W×H=512×288"
  echo "特征      : orb_n=$ORB_N"
  echo "排序复用  : reuse_eps=${REUSE_EPS:-1e-4(程序默认)}"
  echo "额外参数  : ${EXTRA[*]:-（无）}"
  echo '```'
  echo
  echo "## 输出（全部在本目录内）"
  echo
  echo '```'
  echo "out/poses.txt   位姿（name lon lat alt roll pitch yaw # ENU x y z err）"
  echo "run.log         完整 stdout"
  echo "table.txt       逐帧表格"
  echo "metrics.txt     指标卡"
  echo "dump/           视觉 dump（仅传 --dump 时非空）"
  echo '```'
} > "$RUNDIR/CONFIG.md"

# ── cmd.sh（一键复现）────────────────────────────────────────────────
{
  echo '#!/bin/bash'
  echo "# 复现本次实验：由 run.sh 于 $TS 生成"
  echo 'set -e'
  echo 'cd "$(dirname "$0")/../.."'
  printf '%q ' "${CMD[@]}"; echo
} > "$RUNDIR/cmd.sh"
chmod +x "$RUNDIR/cmd.sh"

# ── 执行（stdout 全量落本目录）────────────────────────────────────────
echo "================================================================"
echo " 实验: $TAG   (档位 $PROFILE)"
echo " 目录: runs/${TS}_${SLUG}"
echo " 输入: $(basename "$MODEL")  +  $NIMG 张照片"
echo " 参数: lod=$LOD  rad=$MIN_RAD  orb=$ORB_N  init=$INIT_MODE  pert=$PERT_POS/$PERT_ANG  ${EXTRA[*]:-}"
echo "================================================================"
set +e
"${CMD[@]}" 2>&1 | tee "$RUNDIR/run.log"
RC=${PIPESTATUS[0]}
set -e

# ── 抽取指标卡与逐帧表 ───────────────────────────────────────────────
grep -E '^(Median|Recall@|完整率|FPS|匹配数|耗时合计|渲染分解|排序复用|三轴)' \
     "$RUNDIR/run.log" > "$RUNDIR/metrics.txt" 2>/dev/null || true

# 逐帧表格：从表头行开始，到统计段之前
awk '/^帧 .*匹配 .*内点/{f=1} f{print} /^FPS \(端到端\)/{f=0}' \
    "$RUNDIR/run.log" > "$RUNDIR/table.txt" 2>/dev/null || true

# ── 追加总索引 ───────────────────────────────────────────────────────
[ -f "$RUNS/INDEX.md" ] || cat > "$RUNS/INDEX.md" <<'EOF'
# 实验总索引

每次 `./run.sh` 追加一行。详情看各实验目录的 `CONFIG.md` / `metrics.txt`。

| 时间戳 | 标签 | 档位 | 中位平移 | R@1m | R@3m | 纯算法FPS | 帧数 | 状态 |
|---|---|---|---|---|---|---|---|---|
EOF
g_te=$(grep -m1 'Median Translation Err' "$RUNDIR/metrics.txt" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
g_r1=$(grep -m1 'Recall@1m' "$RUNDIR/metrics.txt" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
g_r3=$(grep -m1 'Recall@3m' "$RUNDIR/metrics.txt" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
g_fps=$(grep -m1 'FPS (纯算法)' "$RUNDIR/metrics.txt" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
nf=$(grep -c . "$RUNDIR/out/poses.txt" 2>/dev/null || echo 0)
st=$([ "$RC" -eq 0 ] && echo OK || echo "FAIL(rc=$RC)")
printf '| %s | %s | %s | %s m | %s%% | %s%% | %s | %s | %s |\n' \
  "$TS" "$TAG" "$PROFILE" "${g_te:-?}" "${g_r1:-?}" "${g_r3:-?}" "${g_fps:-?}" "$nf" "$st" \
  >> "$RUNS/INDEX.md"

echo
echo "================================================================"
echo " 实验完成 (rc=$RC)"
echo "   目录     : runs/${TS}_${SLUG}"
echo "   位姿     : out/poses.txt  ($nf 行)"
echo "   指标     : "; sed 's/^/     /' "$RUNDIR/metrics.txt" 2>/dev/null | head -8
echo "================================================================"
exit $RC
