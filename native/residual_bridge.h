/*
 * residual_bridge.h - C ABI for the native CPU residual hot path.
 *
 * Implements the same two stages as the Python reference in pipeline.py, read off the
 * Magpie kernels in Magpie/src/Magpie.Core/DLSSNRFilter.cpp:
 *
 *   prepare_working_color -> rr_downsample_bgr8   Lanczos-2, vertical axis then horizontal
 *                        -> rr_prepare_working_bgr_rgba8
 *                                                 the same pass, storing BGR and RGBA from
 *                                                 one set of rounded pixels
 *   reconstruct           -> rr_reconstruct_bgr8  Catmull-Rom (B=0, C=0.5) horizontal axis
 *                            then vertical, with the composite fused into the vertical pass
 *                        -> rr_reconstruct_from_rgba8
 *                                                 the same chain with the DLSS working
 *                                                 output read as RGBA8 instead of BGR8
 *
 * Lifecycle: a plan owns everything reusable - filter taps, the FP32 scratch and the
 * worker pool. Creating one per frame is a bug, and rr_get_counters exists so the Python
 * side can prove it is not happening. No C++ exception may cross this boundary; every
 * failure returns a negative code and a message through rr_get_last_error.
 */

#ifndef RESIDUAL_BRIDGE_H
#define RESIDUAL_BRIDGE_H

#include <stdint.h>

#ifdef _WIN32
#  ifdef RESIDUAL_BRIDGE_EXPORTS
#    define RR_API __declspec(dllexport)
#  else
#    define RR_API __declspec(dllimport)
#  endif
#else
#  define RR_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Minor version in the low 16 bits, major in the next 16. Bumped whenever the
 * semantics below change, so a stale DLL cannot be silently accepted.
 *
 * 2: handles became opaque registry keys instead of cast pointers, rr_counters split into
 * per-plan fields plus process totals, and the last-error text became per calling thread.
 * The struct grew, so a version 1 caller reading a version 2 library would misparse it.
 *
 * 2.1: rr_create_plan became exception-safe on every allocation path. No signature, struct
 * field or existing error value changed, so this is a minor bump: RR_ERR_OUT_OF_MEMORY was
 * added, and a failed create now guarantees the state listed under "create failure" below.
 * A 2.0 library can still return RR_ERR_NO_SCRATCH, which 2.1 never does.
 *
 * 2.2: rr_prepare_working_bgr_rgba8 was added, and rr_counters grew three appended fields
 * reporting the calls it and rr_downsample_bgr8 each made. Still minor because everything
 * a 2.1 caller could observe is unchanged: the existing exports keep their signatures, the
 * existing struct fields keep their offsets, and a caller that never asks for the new
 * export never reads the new fields. A caller that needs the fused store checks the minor
 * version and the export's presence, and uses the BGR-only path when either is missing.
 *
 * 2.3: rr_reconstruct_from_rgba8 was added, taking the working-resolution DLSS output as
 * RGBA8 instead of BGR8, and rr_counters grew two more appended fields reporting which
 * reconstruct entry point ran. Same reasoning as 2.2: nothing a 2.2 caller observes moved,
 * and the two shapes of the DLSS input are now decided by which export is called rather
 * than by a flag that would have changed the meaning of an existing argument. */
#define RR_API_VERSION 0x00020003

#define RR_OK                    0
#define RR_ERR_NULL_POINTER     (-1)
#define RR_ERR_INVALID_DIMS     (-2)
#define RR_ERR_INVALID_STRIDE   (-3)
#define RR_ERR_DIM_MISMATCH     (-4)
#define RR_ERR_BAD_HANDLE       (-5)
#define RR_ERR_NO_SCRATCH       (-6)  /* only versions <= 2.0 return this; superseded */
#define RR_ERR_INTERNAL         (-7)
#define RR_ERR_INVALID_STRENGTH (-8)
#define RR_ERR_OUT_OF_MEMORY    (-9)  /* an allocation failed; nothing was published */

/* Thread safety, in one place:
 *
 *   - Handles are opaque ids into an internal registry guarded by a mutex. A handle from
 *     another process, a reused or a destroyed one yields RR_ERR_BAD_HANDLE; nothing is
 *     derived from the integer itself.
 *   - rr_create_plan and rr_destroy_plan are safe to call from any thread, concurrently
 *     with calls on other plans.
 *   - The frame calls take a per-plan lock. Two different plans run their kernels at the
 *     same time; two calls on one plan are serialized, because a plan owns one scratch,
 *     one residual buffer, one row accumulator set and one worker pool.
 *   - rr_destroy_plan removes the handle first, so new calls fail immediately, then waits
 *     for the call already in flight and only then releases the pool and the scratch. The
 *     plan outlives that call by construction, so no thread can touch freed memory.
 *   - The registry mutex is held only for the lookup and the erase, never across a frame.
 *
 * Create failure, in the same place: rr_create_plan builds the whole plan before anything
 * is published, so any failure at or after allocation leaves *out_handle at 0, adds no
 * entry to the registry, moves no process_* counter, starts no surviving worker and frees
 * everything it had already built. The process_* totals are raised only after the registry
 * insert succeeded, which is the point at which the plan becomes visible to other threads.
 * No allocation is attempted by a caller-supplied size: scratch is bounded by the plan's
 * own dimensions, so a failing create is a real allocation failure, not an input trick.
 */

/* Fixed layout, mirrored by residual_native.py.
 *
 * The first twelve fields describe the plan the handle refers to and nothing else: a plan
 * reports its own lifetime, so a second backend, or a capability probe that created and
 * destroyed its own plan, cannot move these numbers.
 *
 * The process_* fields are the running totals for the whole library instance, which is
 * where a probe's plan does belong.
 *
 * The fields after those are per-plan too, and they sit at the end on purpose: a
 * struct only ever grows by appending, so a caller built against an older minor version
 * still reads every field it knows at the offset it knows. */
typedef struct rr_counters {
    int64_t plan_create_count;          /* this plan: always 1 while it exists */
    int64_t plan_destroy_count;         /* this plan: 0 until it is destroyed */
    int64_t scratch_alloc_count;        /* this plan: always 1, allocated at create */
    int64_t scratch_alloc_bytes;        /* this plan: bytes its scratch holds */
    int64_t downsample_call_count;      /* this plan */
    int64_t reconstruct_call_count;     /* this plan */
    int64_t downsample_pixel_count;     /* this plan */
    int64_t reconstruct_pixel_count;    /* this plan */
    int64_t worker_create_count;        /* workers this plan started */
    int64_t worker_count;               /* threads this plan runs frame work on, workers + caller */
    int64_t scratch_bytes;              /* this plan */
    int64_t axis_plan_count;            /* this plan */
    int64_t process_plan_create_total;  /* plans created in this process */
    int64_t process_plan_destroy_total; /* plans destroyed in this process */
    int64_t process_scratch_alloc_total;/* scratch allocations in this process */
    int64_t process_scratch_bytes_total;/* bytes allocated for scratch in this process */
    int64_t process_worker_create_total;/* workers started in this process */
    int64_t prepare_bgr_only_call_count;/* rr_downsample_bgr8 calls that made BGR alone */
    int64_t prepare_bgr_rgba_call_count;/* fused rr_prepare_working_bgr_rgba8 calls */
    int64_t fused_rgba_store_bytes;     /* RGBA bytes those fused calls wrote */
    int64_t reconstruct_bgr_only_call_count;   /* rr_reconstruct_bgr8 calls */
    int64_t reconstruct_rgba_input_call_count; /* rr_reconstruct_from_rgba8 calls */
} rr_counters;

RR_API int32_t rr_get_api_version(void);

/* Copies the last error message into buf (NUL terminated, truncated if needed).
 * Returns the number of bytes written excluding the terminator, or 0 if none.
 *
 * The message is per calling thread, not per plan: this function takes no handle, so a
 * thread that gets a failure code back must call this from the same thread and before any
 * other bridge call on that thread. One thread's error never replaces another's. */
RR_API int32_t rr_get_last_error(char *buf, int32_t buf_len);

RR_API int32_t rr_create_plan(int64_t *out_handle,
                              int32_t src_w, int32_t src_h,
                              int32_t live_w, int32_t live_h);
RR_API int32_t rr_destroy_plan(int64_t handle);

/* src must be src_w x src_h BGR8 from the plan, dst must be live_w x live_h BGR8. */
RR_API int32_t rr_downsample_bgr8(int64_t handle,
                                   const uint8_t *src, int32_t src_w, int32_t src_h, int64_t src_stride,
                                   uint8_t *dst, int32_t dst_w, int32_t dst_h, int64_t dst_stride);

/* The same single Lanczos pass, writing two outputs from one store: dst_bgr is the
 * live_w x live_h BGR8 working frame, dst_rgba the same frame as RGBA8 with alpha 255.
 *
 * The reason this exists rather than the caller converting afterwards is that the channel
 * order is decided while the rounded pixel is still in registers, so the second output
 * costs a write. It is not a different filter: for the same source, dst_bgr is what
 * rr_downsample_bgr8 writes and dst_rgba is what cv2.cvtColor(dst_bgr, COLOR_BGR2RGBA)
 * would give, byte for byte, and both are gated on that.
 *
 * dst_rgba must be at least live_w x 4 bytes per row and may not be null - a caller that
 * wants BGR alone calls rr_downsample_bgr8. */
RR_API int32_t rr_prepare_working_bgr_rgba8(int64_t handle,
                                            const uint8_t *src, int32_t src_w, int32_t src_h, int64_t src_stride,
                                            uint8_t *dst_bgr, int32_t dst_w, int32_t dst_h,
                                            int64_t dst_bgr_stride,
                                            uint8_t *dst_rgba, int64_t dst_rgba_stride);

/* source_full is src_w x src_h, source_low and dlss_low are live_w x live_h,
 * dst is src_w x src_h. All BGR8. strength multiplies the signed residual before
 * the single clamp of the chain. */
RR_API int32_t rr_reconstruct_bgr8(int64_t handle,
                                   const uint8_t *source_full, int32_t sf_w, int32_t sf_h, int64_t sf_stride,
                                   const uint8_t *source_low, int32_t sl_w, int32_t sl_h, int64_t sl_stride,
                                   const uint8_t *dlss_low, int32_t dl_w, int32_t dl_h, int64_t dl_stride,
                                   float strength,
                                   uint8_t *dst, int32_t dst_w, int32_t dst_h, int64_t dst_stride);

/* The same chain with the DLSS output read as RGBA8: source_full is src_w x src_h BGR8,
 * source_low is live_w x live_h BGR8, dlss_low_rgba is live_w x live_h RGBA8 as the DLSS
 * host wrote it, dst is src_w x src_h BGR8.
 *
 * residual_C = dlss_low_rgba.C - source_low.C for C in B,G,R, then Catmull-Rom back to full
 * resolution, then source_full + residual * strength with the one clamp of the chain. The
 * channels are paired by meaning, not by byte index: an RGBA row is R,G,B,A and a BGR row
 * is B,G,R, so output channel i reads byte 2 - i of the RGBA pixel. Alpha is stepped over,
 * never folded into the residual - it carries no colour, and a fourth term would change the
 * picture while still looking like a number.
 *
 * For a dlss_low_rgba that is cv2.cvtColor(x, COLOR_BGR2RGBA) of some BGR frame, this
 * returns exactly what rr_reconstruct_bgr8 returns for x, byte for byte. That equivalence
 * is what lets the caller stop converting the DLSS output before reconstructing from it. */
RR_API int32_t rr_reconstruct_from_rgba8(int64_t handle,
                                         const uint8_t *source_full, int32_t sf_w, int32_t sf_h, int64_t sf_stride,
                                         const uint8_t *source_low, int32_t sl_w, int32_t sl_h, int32_t sl_stride,
                                         const uint8_t *dlss_low_rgba, int32_t dl_w, int32_t dl_h,
                                         int64_t dl_stride,
                                         float strength,
                                         uint8_t *dst, int32_t dst_w, int32_t dst_h, int64_t dst_stride);

RR_API int32_t rr_get_counters(int64_t handle, rr_counters *out);

/* Not part of this ABI: a build with RR_TEST_FAULT_INJECTION defined additionally exports
 * rr_test_arm_fault, rr_test_fault_hits, rr_test_clear_fault and rr_test_registry_size,
 * used by tests/unit/test_residual_bridge_failure_paths.py to interrupt the create path at
 * a named step. The shipped library defines none of them. */

/* 1 when an AVX2 instruction would execute on this CPU under this OS right now, else 0.
 * Decided at call time from CPUID leaf 0 (does leaf 7 exist), leaf 1 ECX bits 27 and 28
 * (OSXSAVE and hardware AVX), XGETBV(0) bits 1 and 2 (the OS saves XMM and YMM state) and
 * leaf 7 subleaf 0 EBX bit 5 (AVX2). No compile-time macro participates: this library is
 * built for the baseline x64 instruction set, so a macro would describe the build, not
 * the machine. Informational only - the kernels rely on compiler auto-vectorisation, so
 * no path appears or disappears when this is 0. */
RR_API int32_t rr_cpu_has_avx2(void);

#ifdef __cplusplus
}
#endif

#endif /* RESIDUAL_BRIDGE_H */
