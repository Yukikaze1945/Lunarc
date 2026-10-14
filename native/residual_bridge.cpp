/*
 * residual_bridge.cpp - Magpie-style working color downsample and signed residual
 * reconstruction, as a native CPU hot path.
 *
 * The math mirrors the reference kernels in
 * Magpie/src/Magpie.Core/DLSSNRFilter.cpp and must stay numerically identical to the
 * Python reference in pipeline.py, which is what tests/unit/test_native_residual.py
 * measures. Where the two would diverge, this file follows the Python reference.
 *
 * Accumulation is FP32 with the taps summed in ascending tap order, the same order
 * numpy uses in _apply_axis_plan, so results match bit for bit rather than merely
 * within tolerance.
 */

#include "residual_bridge.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER)
#  include <intrin.h>
#  include <immintrin.h>   /* __cpuid, __cpuidex, _xgetbv */
#endif

namespace {

constexpr int32_t kChannels = 3;
constexpr int32_t kRgbaChannels = 4;

/* The store every kernel shares: clamp to the 0..255 range the chain guarantees exactly
 * once, then round. Kept as one function so a second output cannot round differently. */
inline uint8_t round_to_u8(float v)
{
    return static_cast<uint8_t>(std::rint(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v)));
}

/* Doubles are used where the Python reference used doubles: tap construction happens
 * once per plan and its rounding is baked into the stored float32 weights. */
double sinc_kernel(double x)
{
    const double px = 3.14159265358979323846 * x;
    if (std::fabs(px) < 1e-5) {
        return 1.0;
    }
    return std::sin(px) / px;
}

double lanczos2_kernel(double x)
{
    if (std::fabs(x) >= 2.0) {
        return 0.0;
    }
    return sinc_kernel(x) * sinc_kernel(0.5 * x);
}

double catmull_rom_kernel(double x)
{
    x = std::fabs(x);
    if (x < 1.0) {
        return ((1.5 * x - 2.5) * x) * x + 1.0;
    }
    if (x < 2.0) {
        return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
    }
    return 0.0;
}

int clamp_index(int i, int last)
{
    if (i < 0) return 0;
    if (i > last) return last;
    return i;
}

/* One axis of a separable filter: dst_len rows of at most max_taps source samples. */
struct AxisPlan
{
    int src_len = 0;
    int dst_len = 0;
    int max_taps = 0;
    std::vector<int32_t> indices;   /* dst_len * max_taps, clamp-to-edge resolved */
    std::vector<float> weights;     /* dst_len * max_taps, each row sums to 1 */
    std::vector<int32_t> tap_counts;/* dst_len */

    bool build_lanczos2(int src, int dst)
    {
        src_len = src;
        dst_len = dst;
        if (src <= 0 || dst <= 0) {
            return false;
        }
        const double scale = static_cast<double>(dst) / static_cast<double>(src);
        const double support = 2.0 / scale;

        std::vector<std::vector<int32_t>> rows_idx(dst);
        std::vector<std::vector<double>> rows_w(dst);
        max_taps = 0;
        for (int d = 0; d < dst; ++d) {
            const double position = (d + 0.5) / scale - 0.5;
            const int first = static_cast<int>(std::ceil(position - support));
            const int last = static_cast<int>(std::floor(position + support));
            double total = 0.0;
            for (int i = first; i <= last; ++i) {
                rows_idx[d].push_back(clamp_index(i, src - 1));
                const double w = lanczos2_kernel((i - position) * scale);
                rows_w[d].push_back(w);
                total += w;
            }
            if (std::fabs(total) > 1e-6) {
                for (size_t t = 0; t < rows_w[d].size(); ++t) {
                    rows_w[d][t] /= total;
                }
            }
            if (static_cast<int>(rows_w[d].size()) > max_taps) {
                max_taps = static_cast<int>(rows_w[d].size());
            }
        }
        return store(rows_idx, rows_w);
    }

    bool build_catmull_rom(int src, int dst)
    {
        src_len = src;
        dst_len = dst;
        if (src <= 0 || dst <= 0) {
            return false;
        }
        max_taps = 4;
        std::vector<std::vector<int32_t>> rows_idx(dst);
        std::vector<std::vector<double>> rows_w(dst);
        for (int d = 0; d < dst; ++d) {
            const double position = (d + 0.5) * src / static_cast<double>(dst) - 0.5;
            const int center = static_cast<int>(std::floor(position));
            double total = 0.0;
            for (int o = -1; o <= 2; ++o) {
                const int index = center + o;
                rows_idx[d].push_back(clamp_index(index, src - 1));
                const double w = catmull_rom_kernel(position - index);
                rows_w[d].push_back(w);
                total += w;
            }
            if (std::fabs(total) > 1e-6) {
                for (size_t t = 0; t < rows_w[d].size(); ++t) {
                    rows_w[d][t] /= total;
                }
            }
        }
        return store(rows_idx, rows_w);
    }

private:
    bool store(const std::vector<std::vector<int32_t>> &rows_idx,
               const std::vector<std::vector<double>> &rows_w)
    {
        indices.assign(static_cast<size_t>(dst_len) * max_taps, 0);
        weights.assign(static_cast<size_t>(dst_len) * max_taps, 0.0f);
        tap_counts.assign(dst_len, 0);
        for (int d = 0; d < dst_len; ++d) {
            const int count = static_cast<int>(rows_w[d].size());
            tap_counts[d] = count;
            for (int t = 0; t < count; ++t) {
                indices[static_cast<size_t>(d) * max_taps + t] = rows_idx[d][t];
                weights[static_cast<size_t>(d) * max_taps + t] = static_cast<float>(rows_w[d][t]);
            }
        }
        return true;
    }
};


/* Controlled allocation failures, compiled only into the test build.
 *
 * Same translation unit as the production code, so what these interrupt is the real create
 * path rather than a copy of it that could drift. Each point throws the exception type the
 * step it sits on is allowed to throw, which is what makes the evidence worth having: the
 * handlers under test are the ones a genuine std::bad_alloc would reach.
 *
 * A production build defines neither the symbols nor the checks, so a user cannot arm a
 * failure and the shipped export table is unchanged. */
#ifdef RR_TEST_FAULT_INJECTION
enum TestFaultPoint : int32_t {
    kFailNone = 0,
    kFailAfterPlanCreate = 1,          /* Plan itself is built, taps are not */
    kFailDuringAxisPlan = 2,           /* tap vectors are being filled */
    kFailDuringPoolThreadCreate = 3,   /* some workers started, the next one throws */
    kFailAfterRowPoolCreate = 4,       /* pool live, no buffer allocated yet */
    kFailDuringScratchAlloc = 5,       /* pool live, one buffer already filled */
    kFailBeforeRegistryPublish = 6,    /* plan complete, nothing published */
    kFailDuringRegistryPublish = 7,    /* the map node allocation throws */
    kTestFaultPointCount = 8
};

/* Arm/stop is a single int so a test cannot leave the library half configured. kind 0
 * throws std::bad_alloc, kind 1 std::runtime_error: the two are reported differently, and
 * an allocator failure must not be dressed up as a generic internal error. */
std::atomic<int32_t> g_test_fault_point{kFailNone};
std::atomic<int32_t> g_test_fault_kind{0};
/* Hit counts are per point and never reset implicitly, so a test that armed a point the
 * create path does not reach fails on "hits == 0" instead of passing for nothing. */
std::atomic<int64_t> g_test_fault_hits[kTestFaultPointCount];

void test_fault(int32_t point)
{
    if (g_test_fault_point.load(std::memory_order_relaxed) != point) {
        return;
    }
    g_test_fault_hits[point].fetch_add(1);
    if (g_test_fault_kind.load(std::memory_order_relaxed) == 1) {
        throw std::runtime_error("injected failure");
    }
    throw std::bad_alloc();
}

#define RR_TEST_FAULT(point) test_fault(point)
#else
#define RR_TEST_FAULT(point)
#endif

/* Splits a row range over a fixed set of threads that live as long as the plan.
 *
 * The pool exists so a frame does not create, destroy or resize threads: workers are
 * started in rr_create_plan and joined in rr_destroy_plan. Each pass hands out whole
 * row ranges, so workers never touch the same output rows, and a chunk index is passed
 * along for anything that needs private scratch.
 *
 * Bodies must not throw: they only touch preallocated memory. A throw would leave the
 * chunk counter unbalanced, so it is caught, recorded and turned into an error code.
 */
class RowPool
{
public:
    explicit RowPool(int worker_threads)
    {
        const int hardware = static_cast<int>(std::thread::hardware_concurrency());
        int wanted = worker_threads;
        if (hardware > 0) {
            wanted = std::min(wanted, hardware - 1);
        }
        if (wanted < 1) {
            wanted = 0;      /* single caller thread still works, no pool at all */
        }
        created_ = wanted;
        chunks_ = wanted + 1;
        try {
            for (int i = 0; i < wanted; ++i) {
                threads_.emplace_back(&RowPool::worker_main, this);
                /* Only once at least one worker is actually running: the case that matters
                 * is a pool that started threads and then failed to finish, because the
                 * ones already started still have to be joined. */
                if (i + 1 < wanted) {
                    RR_TEST_FAULT(kFailDuringPoolThreadCreate);
                }
            }
        } catch (...) {
            /* Fewer workers than asked for is fine; correctness does not depend on it. */
            created_ = static_cast<int>(threads_.size());
            chunks_ = created_ + 1;
        }
    }

    ~RowPool()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        work_ready_.notify_all();
        for (std::thread &t : threads_) {
            t.join();
        }
    }

    int chunk_count() const { return chunks_; }
    int started_workers() const { return static_cast<int>(threads_.size()); }
    int worker_create_count() const { return created_; }
    int worker_count() const { return created_ + 1; }

    /* Runs body over [0, rows) and returns false if a body threw. Blocks until every
     * worker has reported, so sequential calls act as the barrier between passes. */
    bool run(int rows, const std::function<void(int, int, int)> &body)
    {
        failed_.store(false);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            body_ = &body;
            rows_ = rows;
            /* Only the started workers report back; the caller drains alongside them and
             * is not counted, otherwise this wait can never be satisfied. */
            chunks_left_.store(static_cast<int>(threads_.size()));
            next_chunk_.store(0);
            ++generation_;
        }
        work_ready_.notify_all();
        drain();
        if (!threads_.empty()) {
            std::unique_lock<std::mutex> lock(mutex_);
            chunk_done_.wait(lock, [this] { return chunks_left_.load() <= 0; });
        }
        body_ = nullptr;
        return !failed_.load();
    }

private:
    void drain()
    {
        for (;;) {
            int chunk;
            int rows, total;
            const std::function<void(int, int, int)> *body;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!body_) return;
                chunk = next_chunk_.fetch_add(1);
                if (chunk >= chunks_) return;
                rows = rows_;
                total = chunks_;
                body = body_;
            }
            const int lo = static_cast<int>((static_cast<int64_t>(chunk) * rows) / total);
            const int hi = static_cast<int>((static_cast<int64_t>(chunk + 1) * rows) / total);
            if (hi > lo) {
                try {
                    (*body)(lo, hi, chunk);
                } catch (...) {
                    failed_.store(true);
                    return;
                }
            }
        }
    }

    void worker_main()
    {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_ready_.wait(lock, [this, seen] { return stop_ || seen != generation_; });
                if (stop_) return;
                seen = generation_;
            }
            drain();
            std::lock_guard<std::mutex> lock(mutex_);
            if (chunks_left_.fetch_sub(1) <= 1) {
                chunk_done_.notify_all();
            }
        }
    }

    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable work_ready_;
    std::condition_variable chunk_done_;
    const std::function<void(int, int, int)> *body_ = nullptr;
    std::atomic<int> next_chunk_{0};
    std::atomic<int> chunks_left_{0};
    int chunks_ = 1;
    int rows_ = 0;
    int created_ = 0;
    uint64_t generation_ = 0;
    bool stop_ = false;
    std::atomic<bool> failed_{false};
};

/* Per calling thread, so a failure on one worker can be read back by that worker before
 * another thread's failure overwrites it. rr_get_last_error takes no handle, which is only
 * honest while the storage itself is thread local.
 *
 * A fixed array rather than a std::string: set_error also runs from the handlers that
 * report an allocation failure, and the path that explains a failure must not itself need
 * memory that may be the thing that is missing. */
constexpr int32_t kErrorTextMax = 512;
thread_local char g_last_error[kErrorTextMax] = {0};

void set_error(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    std::vsnprintf(g_last_error, sizeof(g_last_error), format, args);
    va_end(args);
}

void clear_error()
{
    g_last_error[0] = '\0';
}

struct Plan
{
    int src_w = 0, src_h = 0, live_w = 0, live_h = 0;
    AxisPlan lanczos_y, lanczos_x;   /* src -> working, vertical then horizontal */
    AxisPlan catmull_x, catmull_y;   /* working -> src, horizontal then vertical */

    /* One FP32 scratch, (live_h x src_w x 3): the intermediate for both stages. The
     * residual of the working resolution lives in residual_low (live_h x live_w x 3). */
    std::vector<float> scratch;
    std::vector<float> residual_low;
    /* One row accumulator per chunk, so parallel rows never share scratch. */
    std::vector<float> row_acc;
    /* Declared last on purpose: members die in reverse order, so the pool is destroyed -
     * its workers stopped and joined - before the buffers a worker may still be reading
     * are released. Ownership by unique_ptr rather than a raw pointer is what lets a
     * partially built plan be cleaned up by the compiler: every failure path out of
     * rr_create_plan is an ordinary return or an unwound exception, and none of them can
     * forget this member the way a hand-written cleanup list can. */
    std::unique_ptr<RowPool> pool;

    /* Fixed at create, before the handle reaches the registry, then read-only: the parts
     * of this plan's lifetime that cannot change while it exists. Written before the
     * registry mutex publishes the handle, so any thread that finds the handle also sees
     * these values. */
    rr_counters fixed{};

    /* The only counters a frame call moves. Atomics rather than a lock so a diagnostics
     * read never queues behind the frame that is running on this plan. */
    std::atomic<int64_t> downsample_calls{0};
    std::atomic<int64_t> reconstruct_calls{0};
    std::atomic<int64_t> downsample_pixels{0};
    std::atomic<int64_t> reconstruct_pixels{0};
    /* Which of the two ways to prepare the working frame this plan used. The pair is the
     * evidence that a reduced path runs the fused store rather than the Python side
     * claiming it did: downsample_calls says how many frames were downsampled, and these
     * say how each of those downsamples published its result. */
    std::atomic<int64_t> prepare_bgr_only_calls{0};
    std::atomic<int64_t> prepare_bgr_rgba_calls{0};
    std::atomic<int64_t> fused_rgba_store_bytes{0};
    /* The same pair on the other side of the frame: which reconstruct entry point turned
     * the DLSS output back into a full-resolution image. Read together with the Python
     * conversion counters these say whether the RGBA->BGR pass was deleted or merely
     * stopped being reported. */
    std::atomic<int64_t> reconstruct_bgr_only_calls{0};
    std::atomic<int64_t> reconstruct_rgba_input_calls{0};

    /* Serializes everything that touches scratch, residual_low, row_acc and pool: a plan
     * owns exactly one of each, so two frame calls on one plan would overwrite each
     * other's intermediate rows. Different plans never take this lock. */
    std::mutex call_mutex;
};

/* Guards the map only - the lookup copies the shared_ptr out and releases this lock before
 * any compute, and it is never held across a frame. The process-wide totals are atomic for
 * the same reason rr_get_counters must not take registry_mutex while another thread creates. */
std::mutex registry_mutex;
std::unordered_map<int64_t, std::shared_ptr<Plan>> g_registry;
std::atomic<int64_t> g_next_handle{1};
std::atomic<int64_t> g_process_plan_create_total{0};
std::atomic<int64_t> g_process_plan_destroy_total{0};
std::atomic<int64_t> g_process_scratch_alloc_total{0};
std::atomic<int64_t> g_process_scratch_bytes_total{0};
std::atomic<int64_t> g_process_worker_create_total{0};

/* Returns the plan owning the handle, or null. The returned shared_ptr keeps the plan
 * alive for the whole call even if another thread destroys it meanwhile, which is what
 * makes the wait in rr_destroy_plan sufficient instead of racy. */
std::shared_ptr<Plan> lookup_plan(int64_t handle)
{
    if (handle <= 0) {
        return nullptr;
    }
    const std::lock_guard<std::mutex> lock(registry_mutex);
    const auto it = g_registry.find(handle);
    if (it == g_registry.end()) {
        return nullptr;
    }
    return it->second;
}

bool check_axes(const AxisPlan &a, int src_len, int dst_len, const char *name)
{
    if (a.src_len != src_len || a.dst_len != dst_len) {
        set_error("plan %s maps %d->%d but the caller asked for %d->%d",
                  name, a.src_len, a.dst_len, src_len, dst_len);
        return false;
    }
    return true;
}

/* Wraps every exported entry point whose body performs an operation that can throw - the
 * registry lookup and its mutex, the pool's condition variable wait, the std::function a
 * frame call builds from its lambda. An exception leaving an extern "C" function is
 * undefined behaviour, and on MSVC x64 it takes the process down without a diagnostic, so
 * the caller would never learn what failed. These three handlers restore the contract the
 * header states: a negative code plus a message on this thread.
 *
 * rr_create_plan does not go through here because it owns an out parameter whose zeroing
 * is part of the failure contract, and a handler here cannot see the caller's arguments. */
template <class Body>
int32_t rr_guard(const char *what, Body body)
{
    try {
        return body();
    } catch (const std::bad_alloc &) {
        set_error("%s: out of memory", what);
        return RR_ERR_OUT_OF_MEMORY;
    } catch (const std::exception &exc) {
        set_error("%s: %s", what, exc.what());
        return RR_ERR_INTERNAL;
    } catch (...) {
        set_error("%s: unknown exception", what);
        return RR_ERR_INTERNAL;
    }
}

}  // namespace

extern "C" {

RR_API int32_t rr_get_api_version(void)
{
    return RR_API_VERSION;
}

RR_API int32_t rr_get_last_error(char *buf, int32_t buf_len)
{
    if (buf == nullptr || buf_len <= 0) {
        return 0;
    }
    const size_t stored = std::strlen(g_last_error);
    const int32_t copy = static_cast<int32_t>(std::min<size_t>(stored,
                                                              static_cast<size_t>(buf_len - 1)));
    std::memcpy(buf, g_last_error, copy);
    buf[copy] = '\0';
    return copy;
}

RR_API int32_t rr_cpu_has_avx2(void)
{
    /* Runtime capability: can an AVX2 instruction execute on this CPU under this OS right
     * now. Four things have to hold before the AVX2 bit itself means anything - that CPUID
     * leaf 7 exists, that OSXSAVE and hardware AVX are set in leaf 1 ECX, and that the OS
     * has opted into saving YMM state. The replaced version checked none of them and read
     * bit 8 of leaf 7 EBX for the feature bit, which is ERMS: a CPU with ERMS but without
     * AVX2 was reported as AVX2 capable.
     *
     * Deliberately no __AVX2__ shortcut either: that reports what the build was allowed
     * to emit, not what the machine can run, and this DLL is compiled for the baseline
     * x64 set, so the macro is never defined here regardless of the CPU. */
#if defined(_MSC_VER)
    int info[4] = {0, 0, 0, 0};
    __cpuid(info, 0);
    if (info[0] < 7) {
        return 0;                      /* leaf 7 does not exist on this CPU */
    }
    __cpuidex(info, 1, 0);
    const bool osxsave = (info[2] & (1 << 27)) != 0;   /* XGETBV usable */
    const bool hw_avx = (info[2] & (1 << 28)) != 0;    /* AVX in hardware */
    if (!osxsave || !hw_avx) {
        return 0;
    }
    /* The OS must have opted into saving both the upper XMM halves and the YMM state,
     * otherwise AVX/AVX2 instructions fault even though the hardware has them. */
    const unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6) != 0x6) {
        return 0;
    }
    __cpuidex(info, 7, 0);
    return ((info[1] & (1 << 5)) != 0) ? 1 : 0;         /* leaf 7 EBX bit 5: AVX2 */
#elif defined(__GNUC__)
    /* GCC's own runtime detection already walks CPUID plus the OS state. */
    return __builtin_cpu_supports("avx2") ? 1 : 0;
#else
    return 0;
#endif
}

RR_API int32_t rr_create_plan(int64_t *out_handle,
                              int32_t src_w, int32_t src_h,
                              int32_t live_w, int32_t live_h)
{
    clear_error();
    if (out_handle == nullptr) {
        set_error("out_handle is null");
        return RR_ERR_NULL_POINTER;
    }
    *out_handle = 0;
    if (src_w <= 0 || src_h <= 0 || live_w <= 0 || live_h <= 0) {
        set_error("invalid dimensions %dx%d -> %dx%d", src_w, src_h, live_w, live_h);
        return RR_ERR_INVALID_DIMS;
    }

    /* Everything that can fail sits inside one handler set. Each step below allocates, so
     * each can throw bad_alloc, and an exception leaving an extern "C" function is
     * undefined behaviour - on MSVC x64 it terminates the process. Nothing here needs a
     * cleanup list: `plan` is a shared_ptr and its RowPool is a unique_ptr, so unwinding
     * joins the workers that started and releases the buffers, in that order, because the
     * pool is declared after them.
     *
     * The registry insert and the process totals come last, and only after it succeeded,
     * so a create that failed is invisible on all three accounts a caller can check: the
     * handle, the registry, and the counters. */
    try {
        auto plan = std::make_shared<Plan>();
        RR_TEST_FAULT(kFailAfterPlanCreate);
        plan->src_w = src_w;
        plan->src_h = src_h;
        plan->live_w = live_w;
        plan->live_h = live_h;

        RR_TEST_FAULT(kFailDuringAxisPlan);
        const bool ok = plan->lanczos_y.build_lanczos2(src_h, live_h)
                        && plan->lanczos_x.build_lanczos2(src_w, live_w)
                        && plan->catmull_x.build_catmull_rom(live_w, src_w)
                        && plan->catmull_y.build_catmull_rom(live_h, src_h);
        if (!ok) {
            set_error("axis plan construction failed");
            return RR_ERR_INTERNAL;
        }

        /* Allocated here and only here: the per-frame path must never resize these, start
         * threads, or allocate anything at all. */
        plan->pool = std::make_unique<RowPool>(8);
        /* The step the previous version of this function got wrong: the pool was already
         * live here, and the cleanup below it freed that pool by hand and then again from
         * the destructor. */
        RR_TEST_FAULT(kFailAfterRowPoolCreate);
        plan->scratch.assign(static_cast<size_t>(live_h) * src_w * kChannels, 0.0f);
        /* One buffer is already allocated, so a throw here unwinds a partly filled plan. */
        RR_TEST_FAULT(kFailDuringScratchAlloc);
        plan->residual_low.assign(static_cast<size_t>(live_h) * live_w * kChannels, 0.0f);
        plan->row_acc.assign(static_cast<size_t>(plan->pool->chunk_count()) * src_w * kChannels,
                             0.0f);

        const int64_t bytes = static_cast<int64_t>((plan->scratch.size()
                                                    + plan->residual_low.size()
                                                    + plan->row_acc.size()) * sizeof(float))
                              + static_cast<int64_t>(live_h) * live_w * kChannels * 2;  /* uint8 in/out */

        /* Everything below describes this plan alone and never moves again, so a second
         * backend or a capability probe cannot inflate it. */
        plan->fixed.plan_create_count = 1;
        plan->fixed.plan_destroy_count = 0;
        plan->fixed.scratch_alloc_count = 1;
        plan->fixed.scratch_alloc_bytes = bytes;
        plan->fixed.worker_create_count = plan->pool->worker_create_count();
        plan->fixed.worker_count = plan->pool->worker_count();
        plan->fixed.scratch_bytes = bytes;
        plan->fixed.axis_plan_count = 4;

        RR_TEST_FAULT(kFailBeforeRegistryPublish);
        const int64_t handle = g_next_handle.fetch_add(1);
        {
            const std::lock_guard<std::mutex> lock(registry_mutex);
            /* emplace rather than operator[]: the map node allocation is the last thing
             * that can throw, and with it inside this scope a throw leaves the registry
             * exactly as it was - no entry for a handle nobody holds. Throwing here is
             * where a real failing insert would leave the map, before it touched it. */
            RR_TEST_FAULT(kFailDuringRegistryPublish);
            g_registry.emplace(handle, plan);
            g_process_plan_create_total.fetch_add(1);
            g_process_scratch_alloc_total.fetch_add(1);
            g_process_scratch_bytes_total.fetch_add(bytes);
            g_process_worker_create_total.fetch_add(plan->pool->worker_create_count());
        }
        *out_handle = handle;
        return RR_OK;
    } catch (const std::bad_alloc &) {
        /* Reported as what it is: an allocation the plan needed could not be satisfied.
         * Not lumped in with the internal errors below, because a caller deciding whether
         * to retry or to shrink its request needs the distinction. */
        set_error("out of memory creating a plan for %dx%d -> %dx%d",
                  src_w, src_h, live_w, live_h);
        *out_handle = 0;
        return RR_ERR_OUT_OF_MEMORY;
    } catch (const std::exception &exc) {
        set_error("internal error creating a plan: %s", exc.what());
        *out_handle = 0;
        return RR_ERR_INTERNAL;
    } catch (...) {
        set_error("internal error creating a plan: unknown exception");
        *out_handle = 0;
        return RR_ERR_INTERNAL;
    }
}

int32_t destroy_plan_impl(int64_t handle)
{
    clear_error();
    std::shared_ptr<Plan> victim;
    {
        const std::lock_guard<std::mutex> lock(registry_mutex);
        const auto it = g_registry.find(handle);
        if (it == g_registry.end()) {
            set_error("destroy called with an invalid or already destroyed handle");
            return RR_ERR_BAD_HANDLE;
        }
        /* Erase first: every call arriving after this point fails with a bad handle, so no
         * new work can start against a plan that is being taken down. */
        victim = it->second;
        g_registry.erase(it);
        g_process_plan_destroy_total.fetch_add(1);
    }
    /* Then wait for the call already in flight. lookup_plan handed that thread its own
     * shared_ptr, so the pool, the scratch and this lock all still exist while it runs;
     * taking the lock here can only block until it finishes, never touch freed memory. */
    const std::lock_guard<std::mutex> call_lock(victim->call_mutex);
    return RR_OK;
}

/* One Lanczos pass, one or two outputs.
 *
 * `fused` says whether this call publishes the RGBA partner as well. A caller that asked
 * for it with a null buffer gets an error rather than a quietly reduced call, because the
 * counter pair at the bottom is the only way to tell afterwards which path ran. When
 * fused, the three rounded bytes that go into dst are written again in R,G,B order with an
 * opaque alpha: the filter is evaluated once, the channel order is decided while the pixel
 * is still in registers, and the two outputs therefore cannot drift apart the way a second
 * pass over the frame could.
 */
int32_t downsample_impl(int64_t handle,
                        const uint8_t *src, int32_t src_w, int32_t src_h, int64_t src_stride,
                        uint8_t *dst, int32_t dst_w, int32_t dst_h, int64_t dst_stride,
                        uint8_t *dst_rgba, int64_t dst_rgba_stride, bool fused)
{
    clear_error();
    const std::shared_ptr<Plan> plan = lookup_plan(handle);
    if (plan == nullptr) {
        set_error("invalid plan handle");
        return RR_ERR_BAD_HANDLE;
    }
    /* Held for the whole call: this plan owns one scratch and one worker pool, so a second
     * frame call on it would overwrite the rows the first is still reading. A different
     * plan has a different lock and is not delayed by this one. */
    const std::lock_guard<std::mutex> call_lock(plan->call_mutex);
    if (src == nullptr || dst == nullptr || (fused && dst_rgba == nullptr)) {
        set_error("null source or destination pointer");
        return RR_ERR_NULL_POINTER;
    }
    if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        set_error("invalid frame size %dx%d -> %dx%d", src_w, src_h, dst_w, dst_h);
        return RR_ERR_INVALID_DIMS;
    }
    if (!check_axes(plan->lanczos_y, plan->src_h, plan->live_h, "lanczos_y")
        || !check_axes(plan->lanczos_x, plan->src_w, plan->live_w, "lanczos_x")) {
        return RR_ERR_DIM_MISMATCH;
    }
    if (src_w != plan->src_w || src_h != plan->src_h || dst_w != plan->live_w || dst_h != plan->live_h) {
        set_error("plan is %dx%d -> %dx%d, called with %dx%d -> %dx%d",
                  plan->src_w, plan->src_h, plan->live_w, plan->live_h, src_w, src_h, dst_w, dst_h);
        return RR_ERR_DIM_MISMATCH;
    }
    if (src_stride < static_cast<int64_t>(src_w) * kChannels
        || dst_stride < static_cast<int64_t>(dst_w) * kChannels) {
        set_error("stride too small: src %lld for %d columns, dst %lld for %d columns",
                  static_cast<long long>(src_stride), src_w,
                  static_cast<long long>(dst_stride), dst_w);
        return RR_ERR_INVALID_STRIDE;
    }
    if (fused && dst_rgba_stride < static_cast<int64_t>(dst_w) * kRgbaChannels) {
        set_error("stride too small for the RGBA output: %lld for %d columns",
                  static_cast<long long>(dst_rgba_stride), dst_w);
        return RR_ERR_INVALID_STRIDE;
    }

    const int sw = plan->src_w, sh = plan->src_h, lw = plan->live_w, lh = plan->live_h;
    float *inter = plan->scratch.data();

    /* Vertical axis first: (sh x sw) -> (lh x sw), matching DownsampleColorVertical.
     * The tap loop sits outside the column sweep so each sweep is contiguous and can be
     * auto-vectorised. Taps still accumulate in ascending order, exactly like the Python
     * reference, so the FP32 result stays bit identical. */
    const AxisPlan &py = plan->lanczos_y;
    const int row_len = sw * kChannels;
    bool ok = plan->pool->run(lh, [&](int lo, int hi, int) {
        for (int y = lo; y < hi; ++y) {
            const int taps = py.tap_counts[y];
            const int32_t *idx = &py.indices[static_cast<size_t>(y) * py.max_taps];
            const float *w = &py.weights[static_cast<size_t>(y) * py.max_taps];
            float *out_row = inter + static_cast<size_t>(y) * row_len;
            for (int t = 0; t < taps; ++t) {
                const uint8_t *row = src + static_cast<int64_t>(idx[t]) * src_stride;
                const float wt = w[t];
                if (t == 0) {
                    for (int i = 0; i < row_len; ++i) {
                        out_row[i] = wt * static_cast<float>(row[i]);
                    }
                } else {
                    for (int i = 0; i < row_len; ++i) {
                        out_row[i] += wt * static_cast<float>(row[i]);
                    }
                }
            }
        }
    });

    /* Horizontal axis: (lh x sw) -> (lh x lw), then the rounding store. Rows stay
     * independent, so the same pool splits this pass too. */
    const AxisPlan &px = plan->lanczos_x;
    ok = plan->pool->run(lh, [&](int lo, int hi, int) {
        for (int y = lo; y < hi; ++y) {
            const float *inter_row = inter + static_cast<size_t>(y) * row_len;
            uint8_t *dst_row = dst + static_cast<int64_t>(y) * dst_stride;
            uint8_t *rgba_row = fused
                ? dst_rgba + static_cast<int64_t>(y) * dst_rgba_stride : nullptr;
            for (int x = 0; x < lw; ++x) {
                const int taps = px.tap_counts[x];
                const int32_t *idx = &px.indices[static_cast<size_t>(x) * px.max_taps];
                const float *w = &px.weights[static_cast<size_t>(x) * px.max_taps];
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
                for (int t = 0; t < taps; ++t) {
                    const float *p = inter_row + static_cast<size_t>(idx[t]) * kChannels;
                    const float wt = w[t];
                    a0 += wt * p[0];
                    a1 += wt * p[1];
                    a2 += wt * p[2];
                }
                uint8_t *out = dst_row + static_cast<int64_t>(x) * kChannels;
                const uint8_t b = round_to_u8(a0);
                const uint8_t g = round_to_u8(a1);
                const uint8_t r = round_to_u8(a2);
                out[0] = b;
                out[1] = g;
                out[2] = r;
                if (rgba_row != nullptr) {
                    uint8_t *q = rgba_row + static_cast<int64_t>(x) * kRgbaChannels;
                    q[0] = r;
                    q[1] = g;
                    q[2] = b;
                    q[3] = 255;
                }
            }
        }
    }) && ok;
    if (!ok) {
        set_error("kernel threw during downsample");
        return RR_ERR_INTERNAL;
    }

    plan->downsample_calls.fetch_add(1);
    plan->downsample_pixels.fetch_add(static_cast<int64_t>(lh) * lw);
    if (fused) {
        plan->prepare_bgr_rgba_calls.fetch_add(1);
        plan->fused_rgba_store_bytes.fetch_add(
            static_cast<int64_t>(lh) * lw * kRgbaChannels);
    } else {
        plan->prepare_bgr_only_calls.fetch_add(1);
    }
    return RR_OK;
}

/* Catmull-Rom reconstruction of the signed residual, shared by both input shapes.
 *
 * `dlss_is_rgba` says which channel order the working-resolution DLSS output arrives in.
 * Everything after the residual is formed is the same code: one horizontal axis, one
 * vertical axis with the composite fused in, one clamp at the end. Only the first pass
 * reads a different layout, which is the whole reason the conversion it replaces could be
 * deleted - the two paths were never going to diverge anywhere else.
 */
int32_t reconstruct_impl(int64_t handle,
                              const uint8_t *source_full, int32_t sf_w, int32_t sf_h, int64_t sf_stride,
                              const uint8_t *source_low, int32_t sl_w, int32_t sl_h, int32_t sl_stride,
                              const uint8_t *dlss_low, int32_t dl_w, int32_t dl_h, int64_t dl_stride,
                              float strength,
                              uint8_t *dst, int32_t dst_w, int32_t dst_h, int64_t dst_stride,
                              bool dlss_is_rgba)
{
    clear_error();
    const std::shared_ptr<Plan> plan = lookup_plan(handle);
    if (plan == nullptr) {
        set_error("invalid plan handle");
        return RR_ERR_BAD_HANDLE;
    }
    /* Same rule as downsample, and it matters more here: this is the call that writes
     * scratch, residual_low and row_acc, all three of which the next call would reuse. */
    const std::lock_guard<std::mutex> call_lock(plan->call_mutex);
    if (source_full == nullptr || source_low == nullptr || dlss_low == nullptr || dst == nullptr) {
        set_error("null pointer in reconstruct inputs");
        return RR_ERR_NULL_POINTER;
    }
    if (sf_w != plan->src_w || sf_h != plan->src_h
        || sl_w != plan->live_w || sl_h != plan->live_h
        || dl_w != plan->live_w || dl_h != plan->live_h
        || dst_w != plan->src_w || dst_h != plan->src_h) {
        set_error("plan is %dx%d with working %dx%d, got source %dx%d low %dx%d dlss %dx%d dst %dx%d",
                  plan->src_w, plan->src_h, plan->live_w, plan->live_h,
                  sf_w, sf_h, sl_w, sl_h, dl_w, dl_h, dst_w, dst_h);
        return RR_ERR_DIM_MISMATCH;
    }
    if (!std::isfinite(strength)) {
        set_error("residual strength must be finite, got %g", strength);
        return RR_ERR_INVALID_STRENGTH;
    }
    const int64_t low_need = static_cast<int64_t>(plan->live_w) * (dlss_is_rgba ? kRgbaChannels : kChannels);
    const int64_t full_need = static_cast<int64_t>(plan->src_w) * kChannels;
    if (sf_stride < full_need || dst_stride < full_need || sl_stride < static_cast<int64_t>(plan->live_w) * kChannels
        || dl_stride < low_need) {
        set_error("stride too small for the plan dimensions");
        return RR_ERR_INVALID_STRIDE;
    }

    const int sw = plan->src_w, sh = plan->src_h, lw = plan->live_w, lh = plan->live_h;
    float *low = plan->residual_low.data();

    /* Signed residual at working resolution: dlss_low - source_low, never clamped.
     *
     * The two branches read the same three channels and differ only in where they find
     * them. dst of the reconstruction is BGR, so for output channel i the RGBA input keeps
     * that channel at byte 2 - i: an RGBA row is R,G,B,A while a BGR row is B,G,R, and
     * pairing them by byte index would subtract the red residual from the blue channel.
     * The alpha byte is stepped over rather than folded in - it carries no colour, and a
     * fourth term in a three-channel residual would change the image while still looking
     * like a plausible number. */
    bool ok = plan->pool->run(lh, [&](int lo, int hi, int) {
        for (int y = lo; y < hi; ++y) {
            const uint8_t *d = dlss_low + static_cast<int64_t>(y) * dl_stride;
            const uint8_t *s = source_low + static_cast<int64_t>(y) * sl_stride;
            float *out = low + static_cast<size_t>(y) * lw * kChannels;
            if (dlss_is_rgba) {
                for (int x = 0; x < lw; ++x) {
                    const uint8_t *q = d + static_cast<int64_t>(x) * kRgbaChannels;
                    const uint8_t *p = s + static_cast<size_t>(x) * kChannels;
                    float *o = out + static_cast<size_t>(x) * kChannels;
                    o[0] = static_cast<float>(q[2]) - static_cast<float>(p[0]);
                    o[1] = static_cast<float>(q[1]) - static_cast<float>(p[1]);
                    o[2] = static_cast<float>(q[0]) - static_cast<float>(p[2]);
                }
            } else {
                for (int i = 0; i < lw * kChannels; ++i) {
                    out[i] = static_cast<float>(d[i]) - static_cast<float>(s[i]);
                }
            }
        }
    });

    /* Horizontal Catmull-Rom: (lh x lw) -> (lh x sw) into the shared scratch. */
    const AxisPlan &cx = plan->catmull_x;
    const int full_len = sw * kChannels;
    ok = plan->pool->run(lh, [&](int lo, int hi, int) {
        for (int y = lo; y < hi; ++y) {
            const float *row = low + static_cast<size_t>(y) * lw * kChannels;
            float *out_row = plan->scratch.data() + static_cast<size_t>(y) * full_len;
            for (int x = 0; x < sw; ++x) {
                const int taps = cx.tap_counts[x];
                const int32_t *idx = &cx.indices[static_cast<size_t>(x) * cx.max_taps];
                const float *w = &cx.weights[static_cast<size_t>(x) * cx.max_taps];
                float *out = out_row + static_cast<size_t>(x) * kChannels;
                out[0] = w[0] * row[static_cast<size_t>(idx[0]) * kChannels + 0];
                out[1] = w[0] * row[static_cast<size_t>(idx[0]) * kChannels + 1];
                out[2] = w[0] * row[static_cast<size_t>(idx[0]) * kChannels + 2];
                for (int t = 1; t < taps; ++t) {
                    const float *p = row + static_cast<size_t>(idx[t]) * kChannels;
                    const float wt = w[t];
                    out[0] += wt * p[0];
                    out[1] += wt * p[1];
                    out[2] += wt * p[2];
                }
            }
        }
    }) && ok;

    /* Vertical Catmull-Rom with the composite fused in, so no full-resolution FP32
     * residual buffer is ever materialised. Taps accumulate over whole rows, then
     * strength and the single clamp of the chain are applied. The strength guard mirrors
     * the Python reference exactly, quirk included. */
    const bool apply_strength = std::fabs(strength - 1.0) > 1e-4;
    const AxisPlan &cy = plan->catmull_y;
    ok = plan->pool->run(sh, [&](int lo, int hi, int chunk) {
        /* Per-chunk row accumulator: parallel rows must never share one buffer. */
        float *acc_row = plan->row_acc.data() + static_cast<size_t>(chunk) * full_len;
        for (int y = lo; y < hi; ++y) {
            const int taps = cy.tap_counts[y];
            const int32_t *idx = &cy.indices[static_cast<size_t>(y) * cy.max_taps];
            const float *w = &cy.weights[static_cast<size_t>(y) * cy.max_taps];
            for (int t = 0; t < taps; ++t) {
                const float *src_row = plan->scratch.data() + static_cast<size_t>(idx[t]) * full_len;
                const float wt = w[t];
                if (t == 0) {
                    for (int i = 0; i < full_len; ++i) {
                        acc_row[i] = wt * src_row[i];
                    }
                } else {
                    for (int i = 0; i < full_len; ++i) {
                        acc_row[i] += wt * src_row[i];
                    }
                }
            }
            /* Composite this row while its accumulator is still live, then reuse it. */
            const uint8_t *sf_row = source_full + static_cast<int64_t>(y) * sf_stride;
            uint8_t *dst_row = dst + static_cast<int64_t>(y) * dst_stride;
            for (int i = 0; i < full_len; ++i) {
                const float residual = apply_strength ? acc_row[i] * strength : acc_row[i];
                float v = static_cast<float>(sf_row[i]) + residual;
                v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
                dst_row[i] = static_cast<uint8_t>(std::rint(v));
            }
        }
    });
    if (!ok) {
        set_error("kernel threw during reconstruct");
        return RR_ERR_INTERNAL;
    }

    plan->reconstruct_calls.fetch_add(1);
    plan->reconstruct_pixels.fetch_add(static_cast<int64_t>(sh) * sw);
    if (dlss_is_rgba) {
        plan->reconstruct_rgba_input_calls.fetch_add(1);
    } else {
        plan->reconstruct_bgr_only_calls.fetch_add(1);
    }
    return RR_OK;
}

int32_t get_counters_impl(int64_t handle, rr_counters *out)
{
    clear_error();
    const std::shared_ptr<Plan> plan = lookup_plan(handle);
    if (plan == nullptr) {
        set_error("invalid plan handle");
        return RR_ERR_BAD_HANDLE;
    }
    if (out == nullptr) {
        set_error("out is null");
        return RR_ERR_NULL_POINTER;
    }
    /* No plan lock: the fixed half cannot change, and the frame half is read from atomics,
     * so asking for diagnostics never queues behind the frame that is running. */
    rr_counters c = plan->fixed;
    c.downsample_call_count = plan->downsample_calls.load();
    c.downsample_pixel_count = plan->downsample_pixels.load();
    c.reconstruct_call_count = plan->reconstruct_calls.load();
    c.reconstruct_pixel_count = plan->reconstruct_pixels.load();
    c.process_plan_create_total = g_process_plan_create_total.load();
    c.process_plan_destroy_total = g_process_plan_destroy_total.load();
    c.process_scratch_alloc_total = g_process_scratch_alloc_total.load();
    c.process_scratch_bytes_total = g_process_scratch_bytes_total.load();
    c.process_worker_create_total = g_process_worker_create_total.load();
    c.prepare_bgr_only_call_count = plan->prepare_bgr_only_calls.load();
    c.prepare_bgr_rgba_call_count = plan->prepare_bgr_rgba_calls.load();
    c.fused_rgba_store_bytes = plan->fused_rgba_store_bytes.load();
    c.reconstruct_bgr_only_call_count = plan->reconstruct_bgr_only_calls.load();
    c.reconstruct_rgba_input_call_count = plan->reconstruct_rgba_input_calls.load();
    *out = c;
    return RR_OK;
}

/* The exported names are forwarders around the functions above. The bodies did not move:
 * this is the layer that keeps a C++ exception inside the library, so a caller linking the
 * C ABI only ever sees a return code. See rr_guard. */
RR_API int32_t rr_destroy_plan(int64_t handle)
{
    return rr_guard("rr_destroy_plan", [&] { return destroy_plan_impl(handle); });
}

RR_API int32_t rr_downsample_bgr8(int64_t handle,
                                  const uint8_t *src, int32_t src_w, int32_t src_h, int64_t src_stride,
                                  uint8_t *dst, int32_t dst_w, int32_t dst_h, int64_t dst_stride)
{
    return rr_guard("rr_downsample_bgr8", [&] {
        return downsample_impl(handle, src, src_w, src_h, src_stride,
                               dst, dst_w, dst_h, dst_stride, nullptr, 0, false);
    });
}

RR_API int32_t rr_prepare_working_bgr_rgba8(int64_t handle,
                                            const uint8_t *src, int32_t src_w, int32_t src_h, int64_t src_stride,
                                            uint8_t *dst_bgr, int32_t dst_w, int32_t dst_h,
                                            int64_t dst_bgr_stride,
                                            uint8_t *dst_rgba, int64_t dst_rgba_stride)
{
    return rr_guard("rr_prepare_working_bgr_rgba8", [&] {
        return downsample_impl(handle, src, src_w, src_h, src_stride,
                               dst_bgr, dst_w, dst_h, dst_bgr_stride,
                               dst_rgba, dst_rgba_stride, true);
    });
}

RR_API int32_t rr_reconstruct_bgr8(int64_t handle,
                                   const uint8_t *source_full, int32_t sf_w, int32_t sf_h, int64_t sf_stride,
                                   const uint8_t *source_low, int32_t sl_w, int32_t sl_h, int64_t sl_stride,
                                   const uint8_t *dlss_low, int32_t dl_w, int32_t dl_h, int64_t dl_stride,
                                   float strength,
                                   uint8_t *dst, int32_t dst_w, int32_t dst_h, int64_t dst_stride)
{
    return rr_guard("rr_reconstruct_bgr8", [&] {
        return reconstruct_impl(handle, source_full, sf_w, sf_h, sf_stride,
                                source_low, sl_w, sl_h, sl_stride,
                                dlss_low, dl_w, dl_h, dl_stride,
                                strength, dst, dst_w, dst_h, dst_stride, false);
    });
}

/* The same reconstruction with the DLSS output still in the RGBA8 the DLSS host wrote:
 * the residual pass reads R,G,B and steps over A, and nothing in between changes.
 *
 * rr_reconstruct_bgr8 stays for the reference path and for a caller whose DLSS output is
 * already BGR. Which of the two ran is reported per plan, so a build that claims to have
 * deleted the RGBA->BGR conversion is checked against the library rather than against the
 * Python that called it. */
RR_API int32_t rr_reconstruct_from_rgba8(int64_t handle,
                                         const uint8_t *source_full, int32_t sf_w, int32_t sf_h, int64_t sf_stride,
                                         const uint8_t *source_low, int32_t sl_w, int32_t sl_h, int32_t sl_stride,
                                         const uint8_t *dlss_low_rgba, int32_t dl_w, int32_t dl_h,
                                         int64_t dl_stride,
                                         float strength,
                                         uint8_t *dst, int32_t dst_w, int32_t dst_h, int64_t dst_stride)
{
    return rr_guard("rr_reconstruct_from_rgba8", [&] {
        return reconstruct_impl(handle, source_full, sf_w, sf_h, sf_stride,
                                source_low, sl_w, sl_h, sl_stride,
                                dlss_low_rgba, dl_w, dl_h, dl_stride,
                                strength, dst, dst_w, dst_h, dst_stride, true);
    });
}

RR_API int32_t rr_get_counters(int64_t handle, rr_counters *out)
{
    return rr_guard("rr_get_counters", [&] { return get_counters_impl(handle, out); });
}

#ifdef RR_TEST_FAULT_INJECTION
/* Test build only, and only reachable from a DLL compiled with RR_TEST_FAULT_INJECTION:
 * production/residual_bridge.dll exports none of these, which
 * tests/unit/test_residual_bridge_failure_paths.py checks against the shipped binary.
 *
 * The trio is arm / read hits / disarm, and every failure-path assertion in the tests is
 * paired with a hits check, so a test that armed a point the create path never reaches
 * fails loudly instead of passing because nothing was injected. */
RR_API int32_t rr_test_arm_fault(int32_t point, int32_t kind)
{
    clear_error();
    if (point < kFailNone || point >= kTestFaultPointCount) {
        set_error("unknown fault point %d", point);
        return RR_ERR_INVALID_DIMS;
    }
    for (int32_t i = 0; i < kTestFaultPointCount; ++i) {
        g_test_fault_hits[i].store(0);
    }
    g_test_fault_kind.store(kind);
    g_test_fault_point.store(point);
    return RR_OK;
}

RR_API int32_t rr_test_fault_hits(int32_t point)
{
    if (point <= kFailNone || point >= kTestFaultPointCount) {
        return -1;
    }
    return static_cast<int32_t>(g_test_fault_hits[point].load());
}

RR_API void rr_test_clear_fault(void)
{
    g_test_fault_point.store(kFailNone);
    g_test_fault_kind.store(0);
}

/* How many plans the registry is holding right now. A failed create must not move this:
 * a handle the caller never received but that still resolves to a live plan is the ghost
 * entry this whole path is designed to exclude, and no counter read from Python can see
 * it. Reading the size directly is the only way to test for it. */
RR_API int32_t rr_test_registry_size(int32_t *out)
{
    clear_error();
    if (out == nullptr) {
        set_error("out is null");
        return RR_ERR_NULL_POINTER;
    }
    const std::lock_guard<std::mutex> lock(registry_mutex);
    *out = static_cast<int32_t>(g_registry.size());
    return RR_OK;
}
#endif

}  // extern "C"
