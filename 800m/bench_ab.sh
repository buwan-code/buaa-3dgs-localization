#!/bin/bash
# 折叠路径 vs 标量路径 的公平基准
#
# 关键控制项（踩过的坑）：
#  • 用 --w/--h，不是 --width/--height（后者被静默忽略，分辨率其实没变）
#  • --orbit 360 --frames 8  → 每帧视矩阵都变，**禁用排序复用捷径**，
#    否则会严重低估排序成本（曾测出 22 ms vs 真实 60 ms）
#  • 两版交替跑，各跑 2 轮取最好，抵消温度漂移
#  • 先跑一轮热身把频率/温度拉到稳态
set -u
cd ~/gsplat
# ★ 基准模型（2026-10-03 用户指令固定）：pilot.ply = PiLoT/翡翠湾
#   2,767,560 高斯 / 1.39 GB。buaa_gspj_lite3m.ply 仅作历史对照。
PLY=data/pilot.ply
W=512; H=288; FR=8

run() {   # run <bin> <tag>
    local bin=$1 tag=$2
    DISPLAY=:0 ./build/$bin $PLY --w $W --h $H --orbit 360 --frames $FR \
        --threads 8 --out /tmp/bench_$tag.ppm 2>&1 \
      | grep -E "^【合计】|^\[帧" | tail -9 \
      | sed "s/^/[$tag] /"
}

echo "=== 热身 ==="
DISPLAY=:0 ./build/gsrender_base $PLY --w $W --h $H --orbit 360 --frames 3 \
    --threads 8 --out /tmp/warm.ppm >/dev/null 2>&1
echo "热身完成，当前温度 $(( $(cat /sys/class/thermal/thermal_zone0/temp) / 1000 ))C"

for round in 1 2; do
    echo "=== 第 $round 轮 ==="
    run gsrender_base base
    run gsrender_fast fast
done

echo "=== 温度 ==="
echo "$(( $(cat /sys/class/thermal/thermal_zone0/temp) / 1000 ))C"
