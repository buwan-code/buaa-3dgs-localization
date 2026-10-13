#!/bin/bash
# gscheck.sh —— 板端环境一键体检
#   用法： bash ~/gsplat/gscheck.sh
#   在 VSCode Remote-SSH 终端里直接跑
echo "══════════════════ 3DGS 环境体检 ══════════════════"
FAIL=0

chk() {   # chk <标签> <实际> <期望>
    if [ "$2" = "$3" ]; then printf "  ✅ %-22s %s\n" "$1" "$2"
    else printf "  ❌ %-22s %s  （期望 %s）\n" "$1" "$2" "$3"; FAIL=1; fi
}

echo "[硬件频率]"
chk "GPU governor"  "$(cat /sys/class/devfreq/fb000000.gpu/governor 2>/dev/null)" "performance"
chk "GPU 频率"      "$(cat /sys/class/devfreq/fb000000.gpu/cur_freq 2>/dev/null)" "1000000000"
chk "CPU4 governor" "$(cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor 2>/dev/null)" "performance"

echo "[服务]"
chk "性能服务"      "$(systemctl is-enabled gsplat-perf.service 2>/dev/null)" "enabled"
chk "更新器已禁用"  "$(test -f ~/.config/autostart/update-notifier.desktop && echo yes || echo no)" "yes"

echo "[显示环境]"
chk "DISPLAY"       "${DISPLAY:-（空）}" ":0"
chk "Xorg 进程数"   "$(pgrep -c Xorg 2>/dev/null || echo 0)" "1"
chk "X11 socket"    "$(test -S /tmp/.X11-unix/X0 && echo 存在 || echo 缺失)" "存在"

echo "[代码与产物]"
chk "源文件数"      "$(ls ~/gsplat/src/*.cpp ~/gsplat/src/*.h 2>/dev/null | wc -l)" "26"
chk "gsrender"      "$(test -x ~/gsplat/build/gsrender && echo 可执行 || echo 缺失)" "可执行"
chk "gs_localize"   "$(test -x ~/gsplat/build/gs_localize && echo 可执行 || echo 缺失)" "可执行"
chk "bin/ 已同步"   "$(cmp -s ~/gsplat/bin/gsrender ~/gsplat/build/gsrender && echo 是 || echo 否)" "是"

echo "[数据]"
chk "pilot.ply"     "$(test -f ~/gsplat/data/pilot.ply && echo 存在 || echo 缺失)" "存在"

echo
printf "[负载] "; uptime | sed 's/.*load average/load average/'
printf "[温度] "; for z in /sys/class/thermal/thermal_zone*/temp; do \
  printf "%s°C " "$(($(cat $z)/1000))"; done; echo

echo "════════════════════════════════════════════════"
if [ $FAIL -eq 0 ]; then
    echo "✅ 全部正常 —— 可直接开跑"
    echo
    echo "试一下："
    echo "  cd ~/gsplat && ./build/gsrender data/pilot.ply \\"
    echo "    --w 512 --h 288 --frames 30 --orbit 360 \\"
    echo "    --min-alpha-radius 2 --depth --stats"
else
    echo "⚠️  有项目异常，见上方 ❌"
    echo "   若是 governor/服务问题 → 在笔记本上跑： python board_perf_setup.py"
    echo "   若是 DISPLAY 问题     → source ~/.bashrc"
fi
echo "════════════════════════════════════════════════"
