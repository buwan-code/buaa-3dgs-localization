#!/bin/bash
# fan_perm.sh — 给 hwmon 风扇节点加组可写权限，避免每次都要 sudo
#
# 背景：/sys/class/hwmon/hwmonN/pwm1 默认 root:root 644，
#       非 root 进程（如平时的渲染测试）无法调速。
#       本项目长测要频繁调风扇，每次 sudo 很烦，且 sudo 会污染计时（见下）。
#
# ⚠️ 为什么不直接用 sudo 跑渲染程序：
#    ① sudo 会让任务的 CPU 亲和性/调度行为异常，污染性能数据；
#    ② 渲染进程会把文件输出成 root 所有，后续普通用户跑不了。
#    所以正确做法是「只把 sysfs 节点放开」，渲染还在普通用户下跑。
#
# 用法：
#   sudo bash sh/fan_perm.sh          # 加权限（当前生效，重启失效）
#   sudo bash sh/fan_perm.sh --persist # 写入 /etc/rc.local 式的开机自启（可选）
#
# 注意：sysfs 属性文件的权限在每次设备重新 probe 时会被重置，
#       所以「重启后失效」是正常的，需要用 systemd/rc.local 固化。
set -u

log() { printf '[fan_perm] %s\n' "$*"; }

# ── 找到 pwmfan（hwmon 编号不固定，必须按 name 查）──────────────────
FAN_DIR=""
for h in /sys/class/hwmon/hwmon*; do
    [ -r "$h/name" ] || continue
    if [ "$(cat "$h/name" 2>/dev/null)" = "pwmfan" ]; then
        FAN_DIR="$h"
        break
    fi
done

if [ -z "$FAN_DIR" ]; then
    log "❌ 未找到 pwmfan 设备（检查风扇是否插好）"
    exit 1
fi

log "找到风扇: $FAN_DIR"

# 取运行脚本的用户（sudo 下为真实调用者）
REAL_USER="${SUDO_USER:-$(id -un)}"
REAL_GROUP="$(id -gn "$REAL_USER" 2>/dev/null || echo "$REAL_USER")"
log "目标用户: $REAL_USER:$REAL_GROUP"

# ── 放开权限 ──────────────────────────────────────────────────────
for f in "$FAN_DIR/pwm1" "$FAN_DIR/pwm1_enable"; do
    if [ -e "$f" ]; then
        if chown "$REAL_USER:$REAL_GROUP" "$f" 2>/dev/null && chmod 664 "$f" 2>/dev/null; then
            log "✅ $f → $(ls -l "$f" | awk '{print $1, $3":"$4}')"
        else
            log "⚠️  $f 权限设置失败（sysfs 可能不允许 chown，见下方说明）"
        fi
    fi
done

# ── sysfs chown 常被拒绝，退路：用 udev 规则或 sudoers ──────────────
cat <<'NOTE'

──────────────────────────────────────────────────────────────
如果上面显示「权限设置失败」，那是因为内核拒绝改 sysfs 属主。
两条替代路径，选一条即可：

【方案 A · 推荐】sudo 白名单（免密码执行 tee 写风扇）
  echo "$USER ALL=(ALL) NOPASSWD: /usr/bin/tee /sys/class/hwmon/*/pwm1, \
/usr/bin/tee /sys/class/hwmon/*/pwm1_enable" | sudo tee /etc/sudoers.d/gsplat-fan
  sudo chmod 440 /etc/sudoers.d/gsplat-fan

【方案 B · udev 规则】（重启后依然生效）
  sudo tee /etc/udev/rules.d/99-pwmfan.rules >/dev/null <<'EOF'
  SUBSYSTEM=="hwmon", ATTR{name}=="pwmfan", RUN+="/bin/chmod 664 /sys%p/pwm1 /sys%p/pwm1_enable"
  EOF
  sudo udevadm control --reload-rules && sudo udevadm trigger

不放开权限也能跑：ThermalGuard 会退化为「只读温度」，程序不崩。
──────────────────────────────────────────────────────────────
NOTE

# ── 验证 ──────────────────────────────────────────────────────────
if [ -w "$FAN_DIR/pwm1" ] && [ "$(id -u)" = "0" ]; then
    : # root 当前可写是当然的
fi
log "当前身份: $(id -un) (uid=$(id -u))"
[ "$(id -u)" = "0" ] && log "✅ 以 root 运行，可直接写风扇" \
                     || log "ℹ️  以非 root 运行，请确认上面的权限是否已放开"
