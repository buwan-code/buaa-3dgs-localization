// thermaltest.cpp — ThermalGuard 自检（M0 补充）
//
// 作用：验证风扇控制链路 + 观察温度曲线行为。三条测试：
//   1) 探测：列出所有 thermal zone、找到 pwmfan、确认可写性
//   2) 阶跃：依次设 0/64/128/192/255，确认回读一致（验证 PWM 真生效）
//   3) 温升：跑满所有 CPU 核心 30 秒，观察自动曲线是否按预期升档
//
// 用法：sudo ./build/thermaltest [--soak 30] [--no-fan]
//   --soak N   负载测试持续 N 秒（默认 30）
//   --no-fan   不控风扇，只看温度（用于对比无风扇时的温升）
#include "thermal.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <chrono>
#include <thread>
#include <atomic>

using namespace gsplat;
using Clock = std::chrono::steady_clock;

static std::atomic<bool> g_stop{false};
static std::atomic<uint64_t> g_spin{0};

// 纯 ALU 负载：避免内存带宽干扰，纯粹给 CPU 发热
// （也避免被编译器优化掉——结果写进 atomic）
static void* burner(void* arg) {
    const long core = (long)arg;
    if (core >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(core, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
    // 固定点迭代，模拟高 IPC 负载
    uint64_t x = 0x9E3779B97F4A7C15ULL ^ (uint64_t)core;
    while (!g_stop.load(std::memory_order_relaxed)) {
        for (int i = 0; i < 100000; ++i) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            x += 0x2545F4914F6CDD1DULL;
            x *= 0xBF58476D1CE4E5B9ULL;
        }
        g_spin.fetch_add(100000, std::memory_order_relaxed);
    }
    return nullptr;
}

int main(int argc, char** argv) {
    int soak = 30;
    bool use_fan = true;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--soak") && i+1 < argc) soak = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-fan")) use_fan = false;
    }

    fprintf(stdout, "=====================================================\n");
    fprintf(stdout, " ThermalGuard 自检  (soak=%ds, fan=%s)\n",
            soak, use_fan ? "on" : "off");
    fprintf(stdout, "=====================================================\n");

    ThermalGuard tg;
    tg.init(true);

    // ── 测试 1：列出所有 zone ────────────────────────────────────────
    fprintf(stdout, "\n【1】温度传感器清单\n");
    for (const auto& z : tg.zones())
        fprintf(stdout, "  zone%-2d  %-18s  %6.1f °C\n",
                z.index, z.type.c_str(), z.celsius());

    // ── 测试 2：PWM 阶跃 ────────────────────────────────────────────
    fprintf(stdout, "\n【2】PWM 阶跃测试（验证写是否真生效）\n");
    if (!tg.has_fan()) {
        fprintf(stdout, "  ⚠️ 未找到风扇，跳过\n");
    } else {
        const int steps[] = {0, 64, 128, 192, 255, 0};
        for (int s : steps) {
            const bool ok = tg.set_speed(s);
            // 从 sysfs 真实回读（不信任缓存），确认 PWM 硬件真的接受了值
            const int back = tg.read_speed_raw();
            fprintf(stdout, "  设 %3d → %s  回读=%3d  %s\n",
                    s, ok ? "OK " : "FAIL", back,
                    (back == s) ? "✅" : "⚠️ 不一致");
            usleep(400000);   // 让风扇有时间响应（能听到转速变化）
        }
        fprintf(stdout, "  （若风扇硬件在场，上面 128/192/255 时应有可听转速变化）\n");
    }

    if (use_fan && tg.has_fan()) {
        tg.set_curve_auto(true);
        fprintf(stdout, "\n风扇已切到【自动曲线】模式\n");
    } else {
        tg.set_mode(FanMode::kOff);
        fprintf(stdout, "\n风扇保持【关闭】（对比组）\n");
    }

    // ── 测试 3：温升 soak ──────────────────────────────────────────
    fprintf(stdout, "\n【3】CPU 满载温升测试（%d 秒）\n", soak);
    const int ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
    fprintf(stdout, "  启动 %d 个烧核线程...\n", ncpu);
    fflush(stdout);

    std::vector<std::thread> th;
    for (int i = 0; i < ncpu; ++i) th.emplace_back(burner, (void*)(long)i);

    // 每 3 秒采样一次
    const auto t0 = Clock::now();
    int elapsed = 0;
    while (elapsed < soak) {
        std::this_thread::sleep_for(std::chrono::seconds(3));
        elapsed += 3;
        // tick 内部有限流；这里传 0 表示「不管限流，强制采样+调曲线」
        tg.tick(0);
        fprintf(stdout, "  t=%3ds  soc=%5.1f°C  gpu=%5.1f°C  big=%5.1f°C  "
                        "max=%5.1f°C  fan=%3d/255\n",
                elapsed, tg.soc(), tg.gpu(), tg.bigcore(), tg.max_temp(),
                tg.fan_speed());
        fflush(stdout);
    }

    g_stop.store(true);
    for (auto& t : th) t.join();
    const double secs = std::chrono::duration<double>(Clock::now()-t0).count();
    const double ops  = (double)g_spin.load();
    fprintf(stdout, "  负载结束：%.1f s，%.2f Gops/s（累计 %.0f M 次迭代）\n",
            secs, ops / secs / 1e9, ops / 1e6);

    // ── 降温观察 ──────────────────────────────────────────────────
    fprintf(stdout, "\n【4】停机后降温观察（15 秒）\n");
    for (int i = 0; i < 5; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(3));
        tg.sample();
        if (use_fan && tg.has_fan()) tg.set_mode(FanMode::kAuto);
        fprintf(stdout, "  +%2ds  soc=%5.1f°C  max=%5.1f°C  fan=%3d/255\n",
                (i+1)*3, tg.soc(), tg.max_temp(), tg.fan_speed());
        fflush(stdout);
    }

    fprintf(stdout, "\n=====================================================\n");
    fprintf(stdout, " 测试完成。（析构时会把风扇交给安全转速）\n");
    fprintf(stdout, "=====================================================\n");
    return 0;
}
