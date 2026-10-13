// thermal.h — 板卡温度监控与 PWM 风扇控制（Orange Pi 5 Plus / RK3588）
//
// ── 为什么需要这个模块 ──────────────────────────────────────────────
// 本项目的性能测试全部依赖「GPU/DMC/CPU 锁频在 performance」这个前提
// （见 NOTES.md §0.1）。而锁频的代价是**芯片不会自己降频散热**：
// 一旦温度顶到 critical（本板 115 °C），结果是**硬掉电**而不是降频，
// 长测（2 小时压力测试）中途断电会让整轮数据作废。
// 所以风扇不是「可选优化」，是**长测的前置条件**。
//
// ── 硬件事实（2026-10-03 实测 + 用户手册 §3.37 确认）───────────────
// - 风扇接口：2pin 1.25mm，5V 驱动，PWM 引脚 PWM3_IR_M1
// - 驱动：标准 Linux `pwm-fan`，注册在 hwmon
//   /sys/class/hwmon/hwmon8/pwm1         （0-255，占空比）
//   /sys/class/hwmon/hwmon8/pwm1_enable  （1 = 手动模式，驱动默认值）
//   ⚠️ hwmon 编号**不保证**每次开机都是 8，必须按 `name == "pwmfan"` 查找
// - 该 img 里 DTS 的 `rockchip,temp-trips` 自动温控**未生效**
//   （/sys/class/thermal 下无 trip 节点，cooling_device 里也没有 pwm-fan），
//   所以温度-转速映射必须由本模块在用户态实现。
// - 温度节点：7 个 thermal zone，本模块以 **soc-thermal**（thermal_zone0）
//   作为主控量，因为它最接近 SoC 整体温度；同时记录 gpu/npu/bigcore 便于诊断。
//
// ── 使用方式 ────────────────────────────────────────────────────────
//   gsplat::ThermalGuard  guard;          // 构造时接管风扇（写 pwm1_enable=1）
//   guard.set_curve_auto(true);           // 打开自动曲线
//   guard.tick();                         // 主循环里定期调（内部限流，可每帧调）
//   ...
//   析构时自动把风扇设为「安全值」（不关死，避免退出后过热）
//
// ⚠️ 写 /sys/class/hwmon/**/pwm1 需要 root。若非 root 运行，
//    set_speed() 会失败并返回 false —— 本模块**不会**因此崩溃，
//    只是退化为「只读温度、不控风扇」模式。
//    需要风扇时用 sudo 运行，或预先 chmod（见 sh/fan_perm.sh）。
#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace gsplat {

// 一路温度传感器
struct ThermalZone {
    int         index = -1;      // thermal_zoneN 的 N
    std::string type;            // 如 "soc-thermal"
    int         millideg = 0;    // 原始值（毫摄氏度）
    float       celsius() const { return millideg / 1000.0f; }
};

// 风扇控制模式
enum class FanMode {
    kOff,        // 强制关
    kManual,     // 固定转速（由 set_speed 设定）
    kAuto,       // 按温度曲线自动
    kMax,        // 全速（压测/DEBUG用）
};

class ThermalGuard {
public:
    ThermalGuard();
    ~ThermalGuard();

    // 非拷贝（持有资源且管着硬件状态）
    ThermalGuard(const ThermalGuard&) = delete;
    ThermalGuard& operator=(const ThermalGuard&) = delete;

    // ── 初始化 ────────────────────────────────────────────────────
    // 探测 hwmon 下的 pwmfan 与 thermal zone。返回 false 表示
    // 没找到风扇（只读温度模式仍可用）。
    bool init(bool verbose = true);

    // 读一次所有温度（返回主控温度，摄氏度）
    float sample();

    // ── 查询（不做 I/O，用最近一次 sample 的缓存）────────────────
    float soc()      const { return soc_c_;       }   // 主控温度
    float gpu()      const { return gpu_c_;       }
    float npu()      const { return npu_c_;       }
    float bigcore()  const { return bigcore_c_;   }
    float littlecore()const{ return little_c_;    }
    float max_temp() const { return max_c_;       }   // 7 路里的最高值

    const std::vector<ThermalZone>& zones() const { return zones_; }

    bool  has_fan()   const { return fan_ok_;     }
    int   fan_speed() const { return cur_speed_;  }   // 当前 0-255
    FanMode mode()    const { return mode_;       }
    // 风扇 pwm1 的绝对路径（诊断/回读用，比如别的程序/vmstat 想直接读）
    const std::string& fan_pwm_path() const { return fan_pwm_; }
    // 直接从 sysfs 回读当前占空比（不信任缓存时的校验手段）
    int   read_speed_raw() const {
        int v = -1; read_int(fan_pwm_, v); return v;
    }

    // ── 控制 ──────────────────────────────────────────────────────
    // 直接设占空比 0-255。返回 false 表示写失败（通常是无权限）。
    bool set_speed(int duty_0_255);

    void set_mode(FanMode m);
    void set_curve_auto(bool on) { set_mode(on ? FanMode::kAuto : FanMode::kManual); }

    // 主循环调用。内部按 interval_ms 限流（默认 1000 ms 一次采样）。
    // 在 kAuto 模式下顺带调风扇转速。
    void tick(int interval_ms = 1000);

    // 自定义温度曲线：温度 >= trip[i] 时用 duty[i]（两数组等长、按 trip 升序）
    void set_curve(const float* trip_c, const int* duty, int n);

    // 重申「手动控制」模式（防止内核 thermal 把 pwm1_enable 改回 2 接管风扇）
    void reassert_manual();

    // 打印当前状态（一行）
    void print_status(const char* tag = "") const;

private:
    bool write_sys(const std::string& path, const std::string& val);
    bool read_int(const std::string& path, int& out) const;
    void apply_auto_curve();

    std::string fan_pwm_;          // /sys/class/hwmon/hwmonN/pwm1
    std::string fan_enable_;       // .../pwm1_enable
    std::string soc_path_;         // 主控温度节点

    std::vector<ThermalZone> zones_;
    bool  fan_ok_  = false;
    bool  enabled_ = false;

    FanMode mode_      = FanMode::kManual;
    int     cur_speed_ = 0;
    int     last_set_  = -1;

    float soc_c_ = 0, gpu_c_ = 0, npu_c_ = 0;
    float bigcore_c_ = 0, little_c_ = 0, max_c_ = 0;

    // 温度曲线（默认值在构造里设定）
    static constexpr int kMaxCurve = 8;
    float trip_c_[kMaxCurve] = {};
    int   duty_[kMaxCurve]   = {};
    int   curve_n_ = 0;

    // tick 限流
    uint64_t last_tick_us_ = 0;
    uint32_t interval_ms_  = 1000;
};

} // namespace gsplat
