#!/bin/bash
# lock_freq.sh — 频率锁定（M0 必做，NOTES.md §1.6 / §2.6）
#
# ⚠️ 必须用 bash 运行：Ubuntu 的 sh 是 dash，不支持 pipefail。
#   用法： bash sh/lock_freq.sh {lock|unlock|status}
#
# 为什么必须做：devfreq governor 默认会在负载变化时立刻降频，
#   实测不锁频性能掉 3–8 倍，所有性能数字都不可信。
#
# ⚠️ 本机内核实测结构（Orange Pi 5 Plus, 6.1.99-rockchip-rk3588）：
#   · devfreq 字段直接挂在 /sys/class/devfreq/<dev>/ 顶层，
#     **没有 user_freq，也没有 devfreq/ 子目录** → 只能设 governor，设不了固定频率
#   · GPU 是 fb000000.gpu（fdab0000.npu 是 NPU，别搞混）
#   · cpufreq 是标准结构：scaling_governor / scaling_{min,max}_freq 可写
#
# sudo 密码：从环境变量 GS_SUDO_PW 读取，不硬编码在仓库里。
#   用法：GS_SUDO_PW=<你的密码> bash sh/lock_freq.sh lock

set -uo pipefail

CPU_LARGE_PATH=/sys/devices/system/cpu/cpufreq/policy4   # A76 大核
CPU_SMALL_PATH=/sys/devices/system/cpu/cpufreq/policy0  # A55 小核
GPU_PATH=/sys/class/devfreq/fb000000.gpu                # Mali-G610
DMC_PATH=/sys/class/devfreq/dmc                        # 内存控制器
NPU_PATH=/sys/class/devfreq/fdab0000.npu               # RKNPU

PW="${GS_SUDO_PW:-}"
if [ -z "$PW" ]; then
    echo "⚠️  未设置 GS_SUDO_PW，需 sudo 的写入会失败。"
    echo "    用法：GS_SUDO_PW=<密码> bash sh/lock_freq.sh lock"
fi
log(){ echo "  → $*"; }

# 用 sudo 写一个文件： echo pw | sudo -S sh -c "echo v > path"
wr(){
    local path="$1" val="$2"
    [ -w "$path" ] || echo "$PW" | sudo -S sh -c "echo '$val' > '$path'" 2>/dev/null
}

show_dev(){
    local name="$1" p="$2"
    echo "--- $name ($p) ---"
    if [ ! -d "$p" ]; then echo "  ⚠️ 不存在"; return; fi
    local gov=$(cat "$p/governor" 2>/dev/null)
    echo "  governor    : ${gov:-<读不到>}"
    echo "  可用策略    : $(cat "$p/available_governors" 2>/dev/null)"
    echo "  当前/目标   : $(cat "$p/cur_freq" 2>/dev/null) / $(cat "$p/target_freq" 2>/dev/null) Hz"
    local mn=$(cat "$p/min_freq" 2>/dev/null) mx=$(cat "$p/max_freq" 2>/dev/null)
    if [ -n "$mn" ] && [ -n "$mx" ]; then
        echo "  可调范围    : $mn ~ $mx Hz"
        echo "  available   : $(cat "$p/available_frequencies" 2>/dev/null | tr '\n' ' ')"
    fi
}

status_all(){
    echo "=================================================="
    echo " 频率与 thermal 状态"
    echo "=================================================="
    echo ""
    echo "--- CPU 大核 A76 ($CPU_LARGE_PATH) ---"
    if [ -d "$CPU_LARGE_PATH" ]; then
        echo "  governor    : $(cat $CPU_LARGE_PATH/scaling_governor 2>/dev/null)"
        echo "  cur/max     : $(cat $CPU_LARGE_PATH/scaling_cur_freq 2>/dev/null) / $(cat $CPU_LARGE_PATH/scaling_max_freq 2>/dev/null) kHz"
        echo "  可用频率    : $(cat $CPU_LARGE_PATH/scaling_available_frequencies 2>/dev/null | tr '\n' ' ')"
    else echo "  ⚠️ 不存在"; fi
    echo ""
    echo "--- CPU 小核 A55 ($CPU_SMALL_PATH) ---"
    if [ -d "$CPU_SMALL_PATH" ]; then
        echo "  governor    : $(cat $CPU_SMALL_PATH/scaling_governor 2>/dev/null)"
        echo "  cur/max     : $(cat $CPU_SMALL_PATH/scaling_cur_freq 2>/dev/null) / $(cat $CPU_SMALL_PATH/scaling_max_freq 2>/dev/null) kHz"
    else echo "  ⚠️ 不存在"; fi
    echo ""
    show_dev "GPU (Mali-G610)" "$GPU_PATH"
    echo ""
    show_dev "DMC 内存控制器" "$DMC_PATH"
    echo ""
    show_dev "NPU" "$NPU_PATH"
    echo ""
    echo "--- 内存 ---"
    free -h | head -2
    echo "--- 温度 ---"
    for z in /sys/class/thermal/thermal_zone*; do
        if [ -f "$z/temp" ]; then
            printf "  %-18s %s C\n" "$(cat $z/type 2>/dev/null)" \
                   "$(awk '{printf "%.1f", $1/1000}' "$z/temp")"
        fi
    done
}

lock(){
    echo "=================================================="
    echo " 锁定频率（性能模式）"
    echo "=================================================="

    # ── CPU 大核 A76 ──
    if [ -d "$CPU_LARGE_PATH" ]; then
        wr "$CPU_LARGE_PATH/scaling_governor" performance && log "CPU大核 governor=performance"
        MAX=$(cat "$CPU_LARGE_PATH/cpuinfo_max_freq" 2>/dev/null)
        if [ -n "$MAX" ]; then
            wr "$CPU_LARGE_PATH/scaling_min_freq" "$MAX" && log "CPU大核 min_freq=$MAX kHz (锁死)"
            wr "$CPU_LARGE_PATH/scaling_max_freq" "$MAX" && log "CPU大核 max_freq=$MAX kHz"
        fi
    fi

    # ── CPU 小核 A55 ──
    if [ -d "$CPU_SMALL_PATH" ]; then
        wr "$CPU_SMALL_PATH/scaling_governor" performance && log "CPU小核 governor=performance"
    fi

    # ── GPU / DMC / NPU ──
    # 本内核无 user_freq，只能设 governor=performance（由内核自动选最高频）
    for item in "GPU:$GPU_PATH" "DMC:$DMC_PATH" "NPU:$NPU_PATH"; do
        name="${item%%:*}"; path="${item##*:}"
        if [ -d "$path" ]; then
            wr "$path/governor" performance && log "$name governor=performance (当前 $(cat $path/cur_freq 2>/dev/null) Hz)"
        fi
    done

    echo ""
    echo "验证："
    status_all
}

unlock(){
    echo "恢复默认 governor..."
    for p in "$CPU_LARGE_PATH" "$CPU_SMALL_PATH"; do
        [ -d "$p" ] && wr "$p/scaling_governor" schedutil
    done
    for p in "$GPU_PATH" "$DMC_PATH" "$NPU_PATH"; do
        [ -d "$p" ] && wr "$p/governor" schedutil
    done
    echo "✅ 已恢复（CPU min_freq 未回退，需要时手动改回 0）"
}

case "${1:-status}" in
    lock)   lock ;;
    unlock) unlock ;;
    status) status_all ;;
    *) echo "用法: bash $0 {lock|unlock|status}"; exit 1 ;;
esac
