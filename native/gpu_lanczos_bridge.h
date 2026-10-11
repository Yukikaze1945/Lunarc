#ifndef GPU_LANCZOS_BRIDGE_H
#define GPU_LANCZOS_BRIDGE_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
#ifdef _WIN32
#  ifdef GPU_LANCZOS_EXPORTS
#    define GPU_LANCZOS_API __declspec(dllexport)
#  else
#    define GPU_LANCZOS_API __declspec(dllimport)
#  endif
#endif
/* 2 adds GpuLanczosTransferCounters and gpu_lanczos_get_transfer_counters. Every
 * ABI 1 entry point keeps its signature and behaviour, so an ABI 1 caller that
 * never reads the counters is unaffected. */
#define GPU_LANCZOS_ABI_VERSION 2u
/* The scaled contract is a new export under the same ABI number, not a new ABI:
 * nothing ABI 2 declared changes signature, layout or behaviour, so an ABI 2
 * caller loads and runs against this DLL exactly as before. A caller that wants
 * to resample to a size other than the half asks for
 * gpu_lanczos_open_scaled by name and treats its absence as "this build cannot
 * do it", the way the NVOF device exports are already probed. */
typedef struct GpuLanczosBridge GpuLanczosBridge;

/* Exact 2x downsample contract: full-size CUDA RGBA8 -> borrowed working-size
 * CUDA RGBA8. The filter is separable Lanczos-2, vertical then horizontal,
 * with the native CPU reference's float32 weights and clamp-to-edge taps.
 * Process is synchronous; output remains valid until the next process/close.
 * Close before destroying the input CUDA context. */
GPU_LANCZOS_API uint32_t gpu_lanczos_abi_version(void);
GPU_LANCZOS_API int32_t gpu_lanczos_open(void* cuda_context,
    uint32_t full_width, uint32_t full_height, GpuLanczosBridge** out_bridge);
GPU_LANCZOS_API int32_t gpu_lanczos_process(GpuLanczosBridge* bridge,
    uint64_t full_rgba_ptr, size_t full_pitch,
    uint64_t* out_working_ptr, uint32_t* out_working_pitch);

/* Every field is incremented by the calls that ran, so a zero is a measurement
 * of a boundary that was not crossed rather than a default. One process call
 * copies the full-size frame into the shared D3D12 source buffer on the device,
 * writes the vertical pass's float4 separations, then writes the RGBA8 working
 * frame; all three are payload bytes the code determines per call. Reads,
 * cache-line traffic and PCIe traffic are not observable from here and are not
 * counted, and no host copy occurs in any of these steps. */
typedef struct GpuLanczosTransferCounters {
    uint32_t struct_size;
    uint64_t process_count;              /* calls that returned 0 */
    uint64_t source_device_copy_bytes;   /* full-size RGBA8 device-to-device copy
                                          * into the shared source buffer */
    uint64_t scratch_device_write_bytes; /* vertical pass output: every float4
                                          * element written once, full_w*low_h*16
                                          * bytes per call */
    uint64_t output_device_write_bytes;  /* horizontal pass output: one packed
                                          * RGBA8 pixel per thread, low_w*low_h*4
                                          * bytes per call */
} GpuLanczosTransferCounters;

GPU_LANCZOS_API int32_t gpu_lanczos_get_transfer_counters(
    GpuLanczosBridge* bridge, GpuLanczosTransferCounters* counters);
GPU_LANCZOS_API void gpu_lanczos_close(GpuLanczosBridge* bridge);
GPU_LANCZOS_API const char* gpu_lanczos_last_error(void);

/* Any downscale ratio the caller can hand a tap plan for. GPU_LANCZOS_MAX_TAPS
 * is the widest padded window the kernels run; a plan with a shorter window is
 * read up to MAX_TAPS with the extra entries carrying weight 0.0f, which is how
 * the CPU reference's own AxisPlan padding works, so the sum is the CPU's sum
 * rather than an approximation that drops the padding.
 *
 * 24 is the bound on what a plan can ask for, not a measured ratio's width. The
 * caller builds its tables from a Lanczos-2 window of ceil(p - 2/s) .. floor(p +
 * 2/s) at s = working/full, so no row is wider than floor(4/s) + 1 taps, and the
 * ratio it divides by is the one 8-alignment resolved, not the one asked for.
 * Rounding an axis to the nearest multiple of 8 can pull it 4 pixels under a
 * clean quarter and the floor at 16 pulls a tiny axis further still, which bounds
 * 4/s at 20 for every source size and every whole percentage from 25% up: 21
 * taps, which a 80-pixel axis at 25% really needs. Three columns are spare.
 *
 * Every array is [dst_len][taps] row-major, clamp-to-edge resolved by the
 * caller, float32 weights already normalized. The bridge copies these bytes to
 * the device once, at open, and checks that every index lies inside
 * [0, src_len) and every weight is finite before it does; a plan that fails is
 * refused and nothing runs. The full->working ratio is not constrained to 2,
 * but working must not exceed full on either axis and both axes must be at
 * least 2.
 *
 * The plan upload is filter coefficients, not a frame: it is not part of
 * GpuLanczosTransferCounters, which counts per-process payload bytes only. */
#define GPU_LANCZOS_MAX_TAPS 24u
/* What this loaded DLL's plan kernels actually walk, so a caller can ask the
 * binary instead of trusting a header it read at build time. A build that has
 * never heard of the question is not answered by this header's figure: it added
 * gpu_lanczos_open_scaled under ABI 2 while the window was still 16, so the
 * caller measures such a build against the 16 its own shader was compiled with
 * and refuses the plans it cannot read, rather than assuming the widening that
 * came with this getter. */
GPU_LANCZOS_API uint32_t gpu_lanczos_max_taps(void);
typedef struct GpuLanczosTapTables {
    uint32_t struct_size;
    uint32_t taps;
    uint32_t vertical_src_len;      /* full height    */
    uint32_t vertical_dst_len;      /* working height */
    const int32_t* vertical_indices;
    const float* vertical_weights;
    uint32_t horizontal_src_len;    /* full width     */
    uint32_t horizontal_dst_len;    /* working width  */
    const int32_t* horizontal_indices;
    const float* horizontal_weights;
} GpuLanczosTapTables;

/* Opens the table-driven pair of kernels. Same borrowed CUDA context, same
 * synchronous process(), same borrowed working CUDA surface out, same counters
 * with the same meaning - every field of GpuLanczosTransferCounters is computed
 * from the extents this call was given. */
GPU_LANCZOS_API int32_t gpu_lanczos_open_scaled(void* cuda_context,
    uint32_t full_width, uint32_t full_height,
    uint32_t working_width, uint32_t working_height,
    const GpuLanczosTapTables* plan, GpuLanczosBridge** out_bridge);
#ifdef __cplusplus
}
#endif
#endif
