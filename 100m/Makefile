# gsplat — 3DGS 板端渲染器（Orange Pi 5 Plus / RK3588）
#
# 刻意不用 cmake：板子上没装，且裸 Makefile 更透明。
# 交叉编译时把 HOSTCC 传给 rsync 脚本即可。

CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -march=armv8.2-a+fp16+dotprod -fno-math-errno
LDFLAGS  ?=
LDLIBS   ?= -lEGL -lGLESv2 -lm -lpthread

BUILD    := build
SRC      := src

CORE_SRC := $(SRC)/ply_loader.cpp $(SRC)/sorter.cpp $(SRC)/offscreen.cpp \
            $(SRC)/thermal.cpp $(SRC)/camera.cpp $(SRC)/renderer.cpp \
            $(SRC)/projfast.cpp
CORE_OBJ := $(patsubst $(SRC)/%.cpp,$(BUILD)/%.o,$(CORE_SRC))

BINS := $(BUILD)/loadtest $(BUILD)/raster_bench $(BUILD)/thermaltest \
        $(BUILD)/camtest $(BUILD)/gsrender \
        $(BUILD)/rbtest $(BUILD)/projcmp $(BUILD)/depthcheck \
        $(BUILD)/gs_localize $(BUILD)/gs_locate_uav $(BUILD)/calib_pilot

# ── OpenCV（只用于 gs_localize：特征匹配 + PnP）────────────────────
# 板端系统自带 OpenCV 4.5（含 features2d / calib3d），不需要第三方包。
CV_CFLAGS := -I/usr/include/opencv4
CV_LIBS   := -lopencv_core -lopencv_imgproc -lopencv_features2d \
             -lopencv_calib3d -lopencv_imgcodecs

# 板端原生编译
ifeq ($(HOST),1)
CXX      := aarch64-linux-gnu-g++
CXXFLAGS := -O3 -std=c++17 -mcpu=cortex-a76 -mtune=cortex-a76 -ffast-math
endif

.PHONY: all clean run_load run_raster freq deploy

all: $(BINS)

$(BUILD):
	@mkdir -p $(BUILD)

# ⚠️ 头文件依赖必须**全部**列出，且用 -MMD 自动生成补充。
#    踩过的坑：手工列依赖时漏了 renderer.h，改 `RenderConfig` 新增字段后
#    make 不重建依赖它的 .o，结果跑的还是旧二进制 —— 表现为「代码明明
#    改了但行为没变」，白排查很久。用 -MMD -MP 让编译器自己产生 .d。
DEPS := $(CORE_OBJ:.o=.d)

# ★ 把 OpenCV 头路径直接并进全局 CXXFLAGS。
#   为什么不用「目标专属变量」或「二级展开」：
#     • `target.o: CXXFLAGS += ...` 对 **模式规则** 不生效（实测 make -n 无 -Iopencv4）
#     • define/eval 里 $(BUILD) 展开异常（实测生成出 /calib_pilot.o 这种空路径）
#   并进全局只多扫几个头文件，编译时间影响可忽略，但**绝对可靠**。
CXXFLAGS += $(CV_CFLAGS)

$(BUILD)/%.o: $(SRC)/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -MMD -MP -I$(SRC) -c $< -o $@

-include $(DEPS)

$(BUILD)/loadtest: $(CORE_OBJ) $(BUILD)/loadtest.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/raster_bench: $(CORE_OBJ) $(BUILD)/raster_bench.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/thermaltest: $(CORE_OBJ) $(BUILD)/thermaltest.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/camtest: $(BUILD)/camera.o $(BUILD)/camtest.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/gsrender: $(CORE_OBJ) $(BUILD)/gsrender.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

# 分段计时诊断工具（等栅栏 vs 纯读取 分离）
$(BUILD)/rbtest: $(CORE_OBJ) $(BUILD)/rbtest.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

# projcmp：折叠路径 vs 标量路径 的逐点比对
$(BUILD)/projcmp: $(BUILD)/camera.o $(BUILD)/ply_loader.o $(BUILD)/projfast.o $(BUILD)/projcmp.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

# depthcheck：GPU 深度图 vs CPU 独立参考（数值级正确性验证）
$(BUILD)/depthcheck: $(CORE_OBJ) $(BUILD)/depthcheck.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

# gs_localize：3DGS 端到端定位闭环（渲染 → ORB 匹配 → PnP 解算）
# 编译规则由上面带目标专属 CXXFLAGS 的 %.o 模式规则统一处理。
$(BUILD)/gs_localize: $(CORE_OBJ) $(BUILD)/gs_localize.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS) $(CV_LIBS)

# gs_locate_uav：无人机视觉定位（吃真照片目录 → 输出位姿 + 误差统计）
$(BUILD)/gs_locate_uav: $(CORE_OBJ) $(BUILD)/gs_locate_uav.o $(BUILD)/xfeat_frontend.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS) $(CV_LIBS) -lrknnrt

# calibrate_align：标定 PLY 局部系 ↔ ENU 系 的刚体变换（网格搜索 + 渲染匹配打分）
$(BUILD)/calibrate_align: $(CORE_OBJ) $(BUILD)/calibrate_align.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS) $(CV_LIBS)

# calib_ncc：用 ZNCC 相似度标定 PLY↔ENU 对齐（比 ORB 匹配数可靠的指标）
$(BUILD)/calib_ncc: $(CORE_OBJ) $(BUILD)/calib_ncc.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS) $(CV_LIBS)

# calib_pilot：用 PiLoT 官方管线 + ZNCC 标定 PLY↔ENU 平移 + yaw 偏置
$(BUILD)/calib_pilot: $(CORE_OBJ) $(BUILD)/calib_pilot.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS) $(CV_LIBS)

# 部署到板子（默认目标 orangepi5plus，需在 ~/.ssh/config 里配置）
BOARD ?= orangepi5plus
deploy: $(BINS)
	@echo ">>> 部署到 $(BOARD)"
	@ssh -o BatchMode=yes $(BOARD) 'mkdir -p ~/gsplat/bin ~/gsplat/sh' 2>/dev/null || true
	scp -o BatchMode=yes $(BINS) $(BOARD):~/gsplat/bin/
	scp -o BatchMode=yes sh/*.sh $(BOARD):~/gsplat/sh/
	@ssh -o BatchMode=yes $(BOARD) 'chmod +x ~/gsplat/sh/*.sh' 2>/dev/null || true
	@echo "✅ 完成。在板子上运行： ~/gsplat/bin/loadtest ~/gsplat/data/buaa_gspj_lite3m.ply"

freq:
	sh sh/lock_freq.sh status

clean:
	rm -rf $(BUILD)
