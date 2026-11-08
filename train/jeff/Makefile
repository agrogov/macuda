# jefftrain — one source, two builds: real nvcc + cuBLAS on gpubox (`make nv`), the macuda shim on the Mac
# (`make shim`). Design: docs/training/jeff-trainer-design.md §11, plan: docs/training/jeff-trainer-plan.md M0.
#
# Every source is picked up by wildcard, so a new kernel, host file, CPU reference or test appears without editing this
# file (a file appears when the task that owns it writes it; any of them may be absent):
#   kernels/*.cu   -> build/<nv|shim>/<name>.o        device code, extern "C" wrappers only
#   src/*.c        -> build/<nv|shim>/host_<name>.o   C11 host code
#   ref/*.c        -> build/<nv|shim>/ref_<name>.o    CPU references (float; refd_*.o is the -DJT_REAL_DOUBLE build)
#   test/test_*.c  -> build/<nv|shim>/test_<name>     one binary each, linked with every kernel, ref and non-main src object
#   test/k_*.c     -> linked into test_kernels only    (no binary of its own)
#   test/grad_*.c  -> linked into gradcheck only       (no binary of its own)
# build/<nv|shim>/jefftrain is built only when src/main.c exists.

SHELL := /bin/sh
.DEFAULT_GOAL := nv
# objects are built through pattern-rule chains, so keep them instead of treating them as throw-away intermediates
.SECONDARY:

# ---- knobs -------------------------------------------------------------------------------------------------------------
# The main checkout: the shim's libraries and CUDA headers live there (a worktree of the plan's rule 7 has none of them:
# they are git-ignored). Never look for the shim relative to this tree.
MACUDA   ?= $(HOME)/Downloads/vfio/macuda
# The card's architecture. `make nv` defaults to sm_120; `make shim` has no default and says what to pass (§11.1).
ARCH     ?=
SHIM     ?= $(MACUDA)/cuda-shim/build/shim
CUDA_INC ?= $(if $(wildcard $(MACUDA)/cuda-shim/cuda-13/include),$(MACUDA)/cuda-shim/cuda-13/include,/usr/local/cuda/include)
DEMANGLE ?= /opt/homebrew/opt/llvm/lib/libLLVMDemangle.a
SHIMCC   ?= /opt/homebrew/opt/llvm/bin/clang
HOSTCC   ?= gcc
PYTHON   ?= python3

# nvcc is not on a bare ssh PATH on gpubox, so find it the way a login shell would.
ifeq ($(origin NVCC),undefined)
NVCC := $(or $(shell command -v nvcc 2>/dev/null),$(if $(wildcard /usr/local/cuda/bin/nvcc),/usr/local/cuda/bin/nvcc),$(firstword $(wildcard /usr/local/cuda*/bin/nvcc)),nvcc)
endif
NV_CUDA_INC ?= $(if $(filter %/bin/nvcc,$(NVCC)),$(patsubst %/bin/nvcc,%/include,$(NVCC)),/usr/local/cuda/include)
# The macOS SDK this machine can link against; build/sdk.sh answers by linking a one-line program.
SDK := $(shell sh $(MACUDA)/cuda-shim/build/sdk.sh 2>/dev/null)

NV_ARCH   := $(or $(ARCH),sm_120)
# tinycc's own default: only reached when test-null/check-symbols build the kernels without `make shim ARCH=...`
SHIM_ARCH := $(or $(ARCH),sm_120a)
INC       := -Isrc -Ikernels -Iref -Itest
# tinycc rewrites -I<worktree>/... to the directory the container/host sees (/ggml/...), so the device half can include
# any header of this tree, not only ones beside the .cu.
SHIM_INC  := $(addprefix -I$(CURDIR)/,src kernels ref test)
NV_CFLAGS := -std=c11 -O2 -Wall -Wextra -D_GNU_SOURCE $(INC) -I$(NV_CUDA_INC)
SHIM_CFLAGS := -isysroot $(SDK) -std=c11 -O2 -g -Wall -Wextra -D_GNU_SOURCE -DJT_SHIM $(INC) -I$(CUDA_INC)
CPU_CFLAGS  := -std=c11 -O2 -g -Wall -Wextra -D_GNU_SOURCE -DJT_CPU $(INC) -I$(CUDA_INC)
# _GNU_SOURCE: glibc hides clock_gettime()/mmap() under -std=c11 (util.h's jt_now, st.c's mmap); Darwin ignores it. It
# is on every host build so no file has to remember it.
SHIM_LIBS := $(SHIM)/libtinycudart.a $(SHIM)/libtinycublas.a $(SHIM)/nv/libtinynv.a $(DEMANGLE) -lc++ -lm
NV_LIBS   := -lcublas -lm

# ---- what there is (wildcards: every one of these may be empty) --------------------------------------------------------
KERNELS := $(sort $(wildcard kernels/*.cu))
HOSTSRC := $(sort $(wildcard src/*.c))
REFSRC  := $(sort $(wildcard ref/*.c))
TESTSRC := $(sort $(wildcard test/test_*.c))
KTESTS  := $(sort $(wildcard test/k_*.c))
GRADS   := $(sort $(wildcard test/grad_*.c))

# kernel objects keep the kernel's name; host, ref and test objects are prefixed so that src/head.c, kernels/head.cu and
# ref/head.c can all coexist in one flat build directory (which is what check-symbols scans).
KOBJ  = $(patsubst kernels/%.cu,$(1)/%.o,$(KERNELS))
HOBJ  = $(patsubst src/%.c,$(1)/host_%.o,$(HOSTSRC))
HOBJNOMAIN = $(filter-out $(1)/host_main.o,$(call HOBJ,$(1)))
ROBJ  = $(patsubst ref/%.c,$(1)/ref_%.o,$(REFSRC))
RDOBJ = $(patsubst ref/%.c,$(1)/refd_%.o,$(REFSRC))
TOBJ  = $(patsubst test/test_%.c,$(1)/test_%.o,$(TESTSRC))
TBIN  = $(patsubst test/test_%.c,$(1)/test_%,$(TESTSRC))
KTOBJ = $(patsubst test/k_%.c,$(1)/k_%.o,$(KTESTS))
GTOBJ = $(patsubst test/grad_%.c,$(1)/grad_%.o,$(GRADS))

# jefftrain itself exists only once src/main.c does.
ifneq ($(wildcard src/main.c),)
NV_JEFFTRAIN := build/nv/jefftrain
SH_JEFFTRAIN := build/shim/jefftrain
endif
# gradcheck is CPU-only and double precision, but every build compiles it when the task that owns it has written it.
ifneq ($(wildcard test/gradcheck.c),)
NV_GRADCHECK := build/nv/gradcheck
SH_GRADCHECK := build/shim/gradcheck
endif

# ---- nv build: real nvcc, real cuBLAS (gpubox) -------------------------------------------------------------------------
.PHONY: nv
nv:
	@if ! $(NVCC) --version >/dev/null 2>&1; then echo "nv: $(NVCC) not found; real CUDA is needed (gpubox). On the Mac use make shim ARCH=..." >&2; exit 1; fi
	@mkdir -p build/nv; \
	 if [ "$$(cat build/nv/.arch 2>/dev/null)" != "$(NV_ARCH)" ]; then \
	   echo "nv: ARCH=$(NV_ARCH) (device code rebuilds)"; rm -f build/nv/*.o; printf '%s\n' '$(NV_ARCH)' > build/nv/.arch; fi
	@$(MAKE) --no-print-directory nv-build

.PHONY: nv-build
nv-build: $(call KOBJ,build/nv) $(call ROBJ,build/nv) $(call RDOBJ,build/nv) $(call TBIN,build/nv) $(NV_GRADCHECK) $(NV_JEFFTRAIN)

# A different ARCH must rebuild the device code instead of reusing the other card's objects: the `nv` and `shim`
# targets compare ARCH with build/<build>/.arch and drop the objects when it changed (a stamp prerequisite cannot do this:
# make compares mtimes, and an arch switch inside one second would compare equal).
build/nv/%.o: kernels/%.cu $(wildcard kernels/*.h)
	@mkdir -p $(@D)
	$(NVCC) -O2 -arch=$(NV_ARCH) -std=c++17 -Xcompiler -Wall $(INC) -MMD -MF $(@:.o=.d) -c $< -o $@

build/nv/host_%.o: src/%.c
	@mkdir -p $(@D)
	$(HOSTCC) $(NV_CFLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

build/nv/ref_%.o: ref/%.c
	@mkdir -p $(@D)
	$(HOSTCC) $(NV_CFLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

build/nv/refd_%.o: ref/%.c
	@mkdir -p $(@D)
	$(HOSTCC) $(NV_CFLAGS) -DJT_REAL_DOUBLE -MMD -MP -MF $(@:.o=.d) -c $< -o $@

build/nv/test_%.o: test/test_%.c
	@mkdir -p $(@D)
	$(HOSTCC) $(NV_CFLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

build/nv/k_%.o: test/k_%.c
	@mkdir -p $(@D)
	$(HOSTCC) $(NV_CFLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

build/nv/grad_%.o: test/grad_%.c
	@mkdir -p $(@D)
	$(HOSTCC) $(NV_CFLAGS) -DJT_REAL_DOUBLE -MMD -MP -MF $(@:.o=.d) -c $< -o $@

build/nv/gradcheck.o: test/gradcheck.c
	@mkdir -p $(@D)
	$(HOSTCC) $(NV_CFLAGS) -DJT_REAL_DOUBLE -MMD -MP -MF $(@:.o=.d) -c $< -o $@

build/nv/test_%: build/nv/test_%.o $(call KOBJ,build/nv) $(call HOBJNOMAIN,build/nv) $(call ROBJ,build/nv)
	$(NVCC) -o $@ $^ $(NV_LIBS)

# tests that are several translation units: test/k_*.c only into test_kernels, test/grad_*.c only into gradcheck
ifneq ($(KTESTS),)
build/nv/test_kernels: $(call KTOBJ,build/nv)
endif
ifneq ($(GRADS),)
build/nv/gradcheck: $(call GTOBJ,build/nv)
endif
ifneq ($(NV_GRADCHECK),)
build/nv/gradcheck: build/nv/gradcheck.o $(call RDOBJ,build/nv)
	$(NVCC) -o $@ $^ -lm
endif
ifneq ($(NV_JEFFTRAIN),)
build/nv/jefftrain: $(call KOBJ,build/nv) $(call ROBJ,build/nv) $(call HOBJ,build/nv)
	$(NVCC) -o $@ $^ $(NV_LIBS)
endif

# ---- shim build: tinycc device half + clang host half (the Mac) ---------------------------------------------------------
# tinycc names its fatbin /tmp/tinycc_<source basename>.fatbin, so two worktrees (or another macuda session) compiling a
# same-named .cu overwrite each other's device code. Every kernel is therefore compiled from a uniquely named sibling
# copy — same directory, so quoted includes still resolve; unique basename, so the fatbins never meet — and tinycc
# itself is held under one lock. tinycc is not changed.
WT_ID := $(shell printf '%s\n' '$(CURDIR)' | cksum | cut -d' ' -f1)

.PHONY: shim
shim:
	@if [ -z "$(ARCH)" ]; then echo "set ARCH=sm_120a (5090) or sm_89 (Ada, with SHIM=.../shim-sm_89); ask which card" >&2; exit 1; fi
	@if [ -z "$(SDK)" ]; then echo "shim: no macOS SDK (sh $(MACUDA)/cuda-shim/build/sdk.sh failed); set SDKROOT" >&2; exit 1; fi
	@if [ ! -f "$(SHIM)/libtinycudart.a" ]; then echo "shim: $(SHIM)/libtinycudart.a is missing; build the shim in $(MACUDA) or set SHIM=" >&2; exit 1; fi
	@mkdir -p build/shim; \
	 if [ "$$(cat build/shim/.arch 2>/dev/null)" != "$(SHIM_ARCH)" ]; then \
	   echo "shim: ARCH=$(SHIM_ARCH) (device code rebuilds)"; rm -f build/shim/*.o; printf '%s\n' '$(SHIM_ARCH)' > build/shim/.arch; fi
	@$(MAKE) --no-print-directory shim-build

.PHONY: shim-build
shim-build: $(call KOBJ,build/shim) $(call ROBJ,build/shim) $(call RDOBJ,build/shim) $(call TBIN,build/shim) $(SH_GRADCHECK) $(SH_JEFFTRAIN)

build/shim/%.o: kernels/%.cu $(wildcard kernels/*.h)
	@mkdir -p $(@D)
	@{ cp $< kernels/.tc_$(WT_ID)_$*.cu; \
	   TINYCC_GGML_MAC=$(CURDIR) TINYCC_ARCH=$(SHIM_ARCH) TINYCC_SYSROOT=$(SDK) \
	     /usr/bin/lockf /tmp/jefftrain-tinycc.lock \
	       sh $(MACUDA)/cuda-shim/build/tinycc kernels/.tc_$(WT_ID)_$*.cu -o $@ $(SHIM_INC) -DJT_SHIM; \
	   rc=$$?; rm -f kernels/.tc_$(WT_ID)_$*.cu; exit $$rc; }

build/shim/host_%.o: src/%.c
	@mkdir -p $(@D)
	$(SHIMCC) $(SHIM_CFLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

build/shim/ref_%.o: ref/%.c
	@mkdir -p $(@D)
	$(SHIMCC) $(SHIM_CFLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

build/shim/refd_%.o: ref/%.c
	@mkdir -p $(@D)
	$(SHIMCC) $(SHIM_CFLAGS) -DJT_REAL_DOUBLE -MMD -MP -MF $(@:.o=.d) -c $< -o $@

build/shim/test_%.o: test/test_%.c
	@mkdir -p $(@D)
	$(SHIMCC) $(SHIM_CFLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

build/shim/k_%.o: test/k_%.c
	@mkdir -p $(@D)
	$(SHIMCC) $(SHIM_CFLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

build/shim/grad_%.o: test/grad_%.c
	@mkdir -p $(@D)
	$(SHIMCC) $(SHIM_CFLAGS) -DJT_REAL_DOUBLE -MMD -MP -MF $(@:.o=.d) -c $< -o $@

build/shim/gradcheck.o: test/gradcheck.c
	@mkdir -p $(@D)
	$(SHIMCC) $(SHIM_CFLAGS) -DJT_REAL_DOUBLE -MMD -MP -MF $(@:.o=.d) -c $< -o $@

build/shim/test_%: build/shim/test_%.o $(call KOBJ,build/shim) $(call HOBJNOMAIN,build/shim) $(call ROBJ,build/shim)
	$(SHIMCC) -isysroot $(SDK) -o $@ $^ $(SHIM_LIBS)

ifneq ($(KTESTS),)
build/shim/test_kernels: $(call KTOBJ,build/shim)
endif
ifneq ($(GRADS),)
build/shim/gradcheck: $(call GTOBJ,build/shim)
endif
ifneq ($(SH_GRADCHECK),)
build/shim/gradcheck: build/shim/gradcheck.o $(call RDOBJ,build/shim)
	$(SHIMCC) -isysroot $(SDK) -o $@ $^ $(SHIM_LIBS)
endif
ifneq ($(SH_JEFFTRAIN),)
build/shim/jefftrain: $(call KOBJ,build/shim) $(call ROBJ,build/shim) $(call HOBJ,build/shim)
	$(SHIMCC) -isysroot $(SDK) -o $@ $^ $(SHIM_LIBS)
endif

# ---- test-null: every shim test binary, no arguments, on the null device ------------------------------------------------
# The null device reports launches and computes nothing, so a test that needs the numbers must skip its value check
# (README.md). The hardware path needs both TINYNV_HW and TINYNV_SOCKET, so unsetting them is enough.
.PHONY: test-null
test-null: shim-build
	@set -e; n=0; for t in x $(call TBIN,build/shim); do \
	  if [ "$$t" = x ]; then continue; fi; \
	  echo "== $$t"; env -u TINYNV_HW -u TINYNV_SOCKET -u TINYNV_PCI "$$t"; n=$$((n+1)); \
	done; echo "test-null: $$n test(s) passed with no arguments on the null device"

# ---- check-symbols: nothing the shim does not define (design §11.1, same recipe as cuda-shim's coverage) ----------------
.PHONY: check-symbols
check-symbols: shim-build
	@mkdir -p build/shim
	@nm -u build/shim/*.o | awk '{print $$NF}' | grep -E '^_(cuda|cublas|__cuda)[A-Za-z_]' | sort -u > build/shim/need.txt || true
	@nm $(SHIM)/libtinycudart.a $(SHIM)/libtinycublas.a | awk '$$2 ~ /^[A-Z]$$/ {print $$3}' | sort -u > build/shim/have.txt
	@missing=$$(comm -13 build/shim/have.txt build/shim/need.txt); \
	 if [ -n "$$missing" ]; then echo "check-symbols: the shim does not define:"; echo "$$missing" | sed 's/^/  /'; exit 1; fi; \
	 echo "check-symbols: all $$(wc -l < build/shim/need.txt | tr -d ' ') cuda*/cublas* symbol(s) the objects need are defined by the shim"

# ---- test-cpu: the references and their tests, host compiler only, no GPU ------------------------------------------------
# ref/ is built twice: float (what the GPU tests compare against) and double (-DJT_REAL_DOUBLE, for gradcheck).
CPU_RTESTS := $(patsubst test/test_%.c,%,$(filter test/test_ref%.c,$(TESTSRC)))
ifneq ($(wildcard test/test_st.c),)
CPU_RTESTS += st
endif
ifneq ($(wildcard test/test_jtb.c),)
CPU_RTESTS += jtb
endif
CPU_BINS   := $(addprefix build/cpu/test_,$(CPU_RTESTS))
CPU_RFLOAT := $(call ROBJ,build/cpu)
CPU_RDOUBLE := $(call RDOBJ,build/cpu)

ifneq ($(wildcard test/gradcheck.c),)
CPU_GRADCHECK := build/cpu/gradcheck
endif
# the jtb fixture is only meaningful once both its writer and its reader exist
CPU_JTB_BIN := $(if $(and $(wildcard py/make_fixture.py),$(wildcard test/test_jtb.c)),build/cpu/test_jtb,)
CPU_FIXTURE := $(if $(CPU_JTB_BIN),build/cpu/fx.jtb)
# a CPU test that needs the host code named the same as itself (test_st -> src/st.c) gets it; nothing else is linked in.
$(foreach s,$(CPU_RTESTS),$(if $(wildcard src/$(s).c),$(eval build/cpu/test_$(s): build/cpu/c_$(s).o)))

.PHONY: test-cpu
test-cpu: $(CPU_BINS) $(CPU_GRADCHECK) $(CPU_FIXTURE)
	@set -e; \
	grad='$(CPU_GRADCHECK)'; jtb='$(CPU_JTB_BIN)'; fx='$(CPU_FIXTURE)'; \
	if [ -n "$$grad" ]; then echo "== $$grad all"; "$$grad" all; fi; \
	for t in x $(filter-out $(CPU_JTB_BIN),$(CPU_BINS)); do \
	  if [ "$$t" = x ]; then continue; fi; \
	  echo "== $$t"; "$$t"; \
	done; \
	if [ -n "$$jtb" ]; then echo "== $$jtb $$fx"; "$$jtb" $$fx; fi; \
	echo "test-cpu: ok"

build/cpu/c_%.o: src/%.c
	@mkdir -p $(@D)
	$(HOSTCC) $(CPU_CFLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

build/cpu/test_%.o: test/test_%.c
	@mkdir -p $(@D)
	$(HOSTCC) $(CPU_CFLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

build/cpu/ref_%.o: ref/%.c
	@mkdir -p $(@D)
	$(HOSTCC) $(CPU_CFLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

build/cpu/refd_%.o: ref/%.c
	@mkdir -p $(@D)
	$(HOSTCC) $(CPU_CFLAGS) -DJT_REAL_DOUBLE -MMD -MP -MF $(@:.o=.d) -c $< -o $@

build/cpu/test_%: build/cpu/test_%.o $(CPU_RFLOAT)
	$(HOSTCC) -o $@ $^ -lm

ifneq ($(CPU_GRADCHECK),)
build/cpu/gradcheck.o: test/gradcheck.c
	@mkdir -p $(@D)
	$(HOSTCC) $(CPU_CFLAGS) -DJT_REAL_DOUBLE -MMD -MP -MF $(@:.o=.d) -c $< -o $@

build/cpu/grad_%.o: test/grad_%.c
	@mkdir -p $(@D)
	$(HOSTCC) $(CPU_CFLAGS) -DJT_REAL_DOUBLE -MMD -MP -MF $(@:.o=.d) -c $< -o $@

build/cpu/gradcheck: build/cpu/gradcheck.o $(call GTOBJ,build/cpu) $(CPU_RDOUBLE)
	$(HOSTCC) -o $@ $^ -lm
endif

ifneq ($(CPU_FIXTURE),)
$(CPU_FIXTURE): py/make_fixture.py
	@mkdir -p $(@D)
	$(PYTHON) py/make_fixture.py --seed 1 --rows 8 --out $@
endif

.PHONY: clean
clean:
	rm -rf build

.PHONY: help
help:
	@echo "make nv [ARCH=sm_120]   build/nv: nvcc + real cuBLAS (gpubox)"
	@echo "make shim ARCH=sm_120a  build/shim: the macuda shim (ARCH is required; sm_89 for the Ada card)"
	@echo "make test-null          run every build/shim/test_* with no arguments on the null device"
	@echo "make test-cpu           references, gradcheck all, test_ref*/test_st/test_jtb, host compiler only"
	@echo "make check-symbols      fail if build/shim objects need a cuda*/cublas* symbol the shim lacks"
	@echo "make clean              rm -rf build"

-include $(wildcard build/*/*.d)
