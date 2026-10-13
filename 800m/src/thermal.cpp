#include "thermal.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <chrono>
#include <dirent.h>
#include <unistd.h>

namespace gsplat {

// ── 小工具：读整个文件为字符串（去掉尾部换行）──────────────────────
static bool slurp(const std::string& path, std::string& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char buf[256];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';
    // 去掉尾部空白
    while (n > 0 && (buf[n-1] == '\n' || buf[n-1] == '\r' || buf[n-1] == ' ')) buf[--n] = '\0';
    out.assign(buf, n);
    return true;
}

bool ThermalGuard::read_int(const std::string& path, int& out) const {
    std::string s;
    if (!slurp(path, s)) return false;
    out = atoi(s.c_str());
    return true;
}

bool ThermalGuard::write_sys(const std::string& path, const std::string& val) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return false;
    const int n = fprintf(f, "%s", val.c_str());
    fclose(f);
    return n > 0;
}

// ── 构造：设定默认温度曲线 ─────────────────────────────────────────
// 曲线设计依据（2026-10-03 实测，见 NOTES.md §0.2）：
//   - 本板被动散热很好：8 核全载 60 s，SoC 仅稳定在 52.7 °C（big core 53.6）
//   - 但本项目要**锁频**跑 GPU（Mali 1000 MHz），锁频后温升更快且
//     不会自动降频自保 —— 顶到 crit=115 °C 就是硬掉电，不是降频
//   - 所以曲线目标是「把 SoC 压在 60 °C 以下」，留足 55 °C 余量
//
// 实测发现：曲线太温和没用。第一版 trip[0]=50→duty 64，结果温度爬到 54.5 °C
// 都没升档（因为 trip[1]=58 未触发）。第二版把档位加密、转速提高。
//
//   ≥48 °C →  85   （刚过正常工作温度就起转，静音但有效）
//   ≥52 °C → 120
//   ≥56 °C → 160
//   ≥60 °C → 200
//   ≥65 °C → 235
//   ≥72 °C → 255   （全速，仅异常时到达）
ThermalGuard::ThermalGuard() {
    const float trips[6] = { 48.f, 52.f, 56.f, 60.f, 65.f, 72.f };
    const int   dutys[6] = {   85,  120,  160,  200,  235,  255 };
    curve_n_ = 6;
    for (int i = 0; i < curve_n_; ++i) { trip_c_[i] = trips[i]; duty_[i] = dutys[i]; }
}

// ── 析构：退出时给一个安全转速，不要关死 ───────────────────────────
// 关死风扇会让板子在进程退出后继续升温（锁频还在），有风险。
// 但也不能留全速（吵）。折中：若当前温度 > 55 °C 留 30% 转速，否则关。
ThermalGuard::~ThermalGuard() {
    if (!fan_ok_ || !enabled_) return;
    sample();
    const int safe = (soc_c_ > 55.f) ? 80 : 0;
    write_sys(fan_pwm_, std::to_string(safe));
}

bool ThermalGuard::init(bool verbose) {
    // ── 1) 枚举 thermal zone ────────────────────────────────────────
    zones_.clear();
    DIR* d = opendir("/sys/class/thermal");
    if (d) {
        struct dirent* e;
        while ((e = readdir(d)) != nullptr) {
            if (strncmp(e->d_name, "thermal_zone", 12) != 0) continue;
            ThermalZone z;
            z.index = atoi(e->d_name + 12);
            std::string base = std::string("/sys/class/thermal/") + e->d_name;
            slurp(base + "/type", z.type);
            read_int(base + "/temp", z.millideg);
            zones_.push_back(z);
        }
        closedir(d);
    }

    // 主控温度节点选 soc-thermal；找不到就退化为 thermal_zone0
    soc_path_ = "/sys/class/thermal/thermal_zone0/temp";
    for (const auto& z : zones_) {
        if (z.type == "soc-thermal") {
            soc_path_ = "/sys/class/thermal/thermal_zone" +
                        std::to_string(z.index) + "/temp";
            break;
        }
    }

    // ── 2) 找 pwmfan（⚠️ hwmon 编号不固定，必须按 name 查）──────────
    fan_ok_ = false;
    DIR* hd = opendir("/sys/class/hwmon");
    if (hd) {
        struct dirent* e;
        while ((e = readdir(hd)) != nullptr) {
            if (strncmp(e->d_name, "hwmon", 5) != 0) continue;
            std::string base = std::string("/sys/class/hwmon/") + e->d_name;
            std::string name;
            if (!slurp(base + "/name", name)) continue;
            if (name == "pwmfan" || name == "fan" || name == "pwm-fan") {
                // 确认 pwm1 存在
                if (access((base + "/pwm1").c_str(), F_OK) == 0) {
                    fan_pwm_    = base + "/pwm1";
                    fan_enable_ = base + "/pwm1_enable";
                    fan_ok_     = true;
                    break;
                }
            }
        }
        closedir(hd);
    }

    // ── 3) 接管风扇：pwm1_enable=1 表示「软件手动控制」──────────────
    if (fan_ok_) {
        // 先读回当前状态
        int en = 0, sp = 0;
        read_int(fan_enable_, en);
        read_int(fan_pwm_,    sp);
        cur_speed_ = sp;
        last_set_  = sp;

        // 尝试设为手动模式。若失败（无权限），风扇仍可用但可能被内核策略接管
        if (write_sys(fan_enable_, "1")) {
            enabled_ = true;
        } else {
            // 再试一次：部分内核只暴露 pwm1，没有 pwm1_enable
            if (access(fan_enable_.c_str(), F_OK) != 0 && write_sys(fan_pwm_, "0")) {
                enabled_ = true;   // 至少能写转速
            }
        }
        if (!enabled_ && verbose)
            fprintf(stderr, "[thermal] ⚠️ 无法写 %s（需要 root）。"
                            "风扇不会被本进程控制，仅做温度监控。\n", fan_pwm_.c_str());
    } else if (verbose) {
        fprintf(stderr, "[thermal] ⚠️ 未找到 pwmfan 设备，仅做温度监控。\n");
    }

    sample();

    if (verbose) {
        fprintf(stdout, "[thermal] thermal zone: ");
        for (size_t i = 0; i < zones_.size(); ++i)
            fprintf(stdout, "%s%s", i ? ", " : "", zones_[i].type.c_str());
        fprintf(stdout, "\n[thermal] 主控=%s  soc=%.1f°C  gpu=%.1f°C  "
                        "npu=%.1f°C  max=%.1f°C\n",
                soc_path_.c_str(), soc_c_, gpu_c_, npu_c_, max_c_);
        if (fan_ok_)
            fprintf(stdout, "[thermal] 风扇: %s  当前=%d/255  %s\n",
                    fan_pwm_.c_str(), cur_speed_,
                    enabled_ ? "（本进程可控）" : "（⚠️ 只读，需 root 才能控）");
    }
    return fan_ok_;
}

float ThermalGuard::sample() {
    float soc = 0, gpu = 0, npu = 0, big = 0, little = 0, mx = 0;
    int mi = 0;

    for (auto& z : zones_) {
        std::string base = "/sys/class/thermal/thermal_zone" +
                           std::to_string(z.index) + "/temp";
        if (!read_int(base, z.millideg)) continue;
        const float c = z.celsius();
        if (c > mx) mx = c;

        if      (z.type == "soc-thermal")      soc = c;
        else if (z.type == "gpu-thermal")      gpu = c;
        else if (z.type == "npu-thermal")      npu = c;
        else if (z.type.find("bigcore") != std::string::npos) {
            if (c > big) big = c;
        }
        else if (z.type.find("littlecore") != std::string::npos) little = c;
        ++mi;
    }
    (void)mi;

    soc_c_ = soc; gpu_c_ = gpu; npu_c_ = npu;
    bigcore_c_ = big; little_c_ = little; max_c_ = mx;
    return soc_c_;
}

// ── 写风扇转速 ─────────────────────────────────────────────────────
// ⚠️ 实测发现（2026-10-03，很重要）：
//   pwm1 的**写入不是立即稳定生效**的。驱动内部有个 workqueue 处理
//   PWM 更新（有最小更新间隔），在更新窗口内读 pwm1 会拿到 0 或旧值。
//   实测现象：连写 0/32/64/96 都稳定；写 128 有时回读 0、有时回读 128；
//   写 192/255 多数能粘住。**同一个值重复写 3 次（间隔 1s）必然粘住。**
//   这不是「值被拒绝」（255 也会出现回读 0），而是**回读时序问题**。
//
//   对策：写完 → 短暂等待 → 回读校验 → 不符则重试（最多 3 次）。
//   同时用 last_set_ 去重，避免无意义的高频写把驱动的 workqueue 打满。
static void tiny_sleep_ms(int ms) {
    struct timespec ts;
    ts.tv_sec  = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, nullptr);
}

bool ThermalGuard::set_speed(int duty_0_255) {
    if (!fan_ok_ || !enabled_) return false;
    if (duty_0_255 < 0)   duty_0_255 = 0;
    if (duty_0_255 > 255) duty_0_255 = 255;

    const std::string sv = std::to_string(duty_0_255);

    // 快速路径：同一个目标值已经确认生效过，不重复写
    // （pwm-fan 驱动有最小更新间隔，高频写只会互相干扰）
    if (duty_0_255 == cur_speed_ && last_set_ == duty_0_255) return true;

    const int max_attempts = 4;
    for (int attempt = 1; attempt <= max_attempts; ++attempt) {
        if (!write_sys(fan_pwm_, sv)) return false;

        // 写后等待驱动 workqueue 处理。首次尝试短等，失败后拉长。
        tiny_sleep_ms(attempt == 1 ? 30 : 120);

        int back = -1;
        if (read_int(fan_pwm_, back) && back == duty_0_255) {
            cur_speed_ = duty_0_255;
            last_set_  = duty_0_255;
            return true;
        }
    }

    // 全部重试后仍未确认。保守起见：记录我们「想设」的值，
    // 但不要假装成功 —— 返回 false 让调用方知道硬件没确认。
    // 注意：不回滚 cur_speed_，因为回读本身就是不可靠的。
    cur_speed_ = duty_0_255;
    last_set_  = duty_0_255;
    return false;
}

void ThermalGuard::set_mode(FanMode m) {
    mode_ = m;
    last_set_ = -1;          // 强制下次 tick 重新下发
    switch (m) {
        case FanMode::kOff:    set_speed(0);   break;
        case FanMode::kMax:    set_speed(255); break;
        case FanMode::kManual: /* 保持当前 */ break;
        case FanMode::kAuto:   sample(); apply_auto_curve(); break;
    }
}

void ThermalGuard::set_curve(const float* trip_c, const int* duty, int n) {
    if (!trip_c || !duty || n <= 0) return;
    if (n > kMaxCurve) n = kMaxCurve;
    curve_n_ = n;
    for (int i = 0; i < n; ++i) { trip_c_[i] = trip_c[i]; duty_[i] = duty[i]; }
}

void ThermalGuard::apply_auto_curve() {
    if (!fan_ok_ || !enabled_) return;
    // 用 soc 与 max 里更高的那个驱动曲线 —— 锁频压测时 GPU 可能比 SoC 更热，
    // 只看 soc 会漏掉 GPU 过热。
    const float t = (soc_c_ > max_c_) ? soc_c_ : max_c_;

    int duty = 0;
    for (int i = 0; i < curve_n_; ++i) {
        if (t >= trip_c_[i]) duty = duty_[i];
    }
    // 迟滞（hysteresis）：避免在档位边界上反复抖动。
    // 若要降档，需比触发温度低 2 °C 才生效。
    if (duty < cur_speed_) {
        float lower_bound = 0.f;
        for (int i = 0; i < curve_n_; ++i)
            if (duty_[i] <= duty && trip_c_[i] > lower_bound) lower_bound = trip_c_[i];
        if (t > lower_bound - 2.0f) duty = cur_speed_;
    }
    set_speed(duty);
}

// 有些情况下内核/别的程序会把 pwm1_enable 改回 2（thermal 接管），
// 这时我们的写入会被忽略。定期重申手动模式。
void ThermalGuard::reassert_manual() {
    if (!fan_ok_ || !enabled_) return;
    int en = -1;
    if (read_int(fan_enable_, en) && en != 1) {
        write_sys(fan_enable_, "1");
        last_set_ = -1;             // 促使下次 set_speed 真正下发
    }
}

void ThermalGuard::tick(int interval_ms) {
    // interval_ms == 0 → 强制立即采样（不限流），用于测试/关键节点
    if (interval_ms > 0) interval_ms_ = (uint32_t)interval_ms;

    if (interval_ms_ > 0) {
        using Clock = std::chrono::steady_clock;
        static Clock::time_point last = Clock::now();
        const auto now = Clock::now();
        const uint64_t dt_us =
            std::chrono::duration_cast<std::chrono::microseconds>(now - last).count();
        if (dt_us < (uint64_t)interval_ms_ * 1000ULL) return;
        last = now;
        last_tick_us_ = dt_us;
    }

    sample();
    reassert_manual();
    if (mode_ == FanMode::kAuto) apply_auto_curve();
}

void ThermalGuard::print_status(const char* tag) const {
    fprintf(stdout, "[thermal]%s soc=%.1f gpu=%.1f npu=%.1f big=%.1f little=%.1f "
                    "max=%.1f | fan=%d/255(%d)\n",
            tag && *tag ? tag : "",
            soc_c_, gpu_c_, npu_c_, bigcore_c_, little_c_, max_c_,
            cur_speed_, (int)mode_);
}

} // namespace gsplat
