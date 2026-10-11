#ifndef GPU_RESIDUAL_BRIDGE_H
#define GPU_RESIDUAL_BRIDGE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#  ifdef GPU_RESIDUAL_EXPORTS
#    define GPU_RESIDUAL_API __declspec(dllexport)
#  else
#    define GPU_RESIDUAL_API __declspec(dllimport)
#  endif
#endif

#define GPU_RESIDUAL_ABI_VERSION 1u
typedef struct GpuResidualBridge GpuResidualBridge;

/* This first contract supports exactly a 2x Catmull-Rom residual upscale.
 * The borrowed D3D12 device and DLSS texture must be the same device. The
 * DLSS texture is RGBA8 in COPY_SOURCE state. Both CUDA inputs are RGBA8 and
 * stay valid until process returns. Close before destroying either context. */
GPU_RESIDUAL_API uint32_t gpu_residual_abi_version(void);
GPU_RESIDUAL_API int32_t gpu_residual_open(void* d3d12_device,
    uint32_t working_width, uint32_t working_height,
    GpuResidualBridge** out_bridge);
GPU_RESIDUAL_API int32_t gpu_residual_process(GpuResidualBridge* bridge,
    void* dlss_texture, void* dlss_fence, uint64_t dlss_fence_value,
    void* full_cuda_context, uint64_t full_rgba_ptr, size_t full_pitch,
    void* working_cuda_context, uint64_t working_rgba_ptr,
    size_t working_pitch, float strength, void** out_texture);
/* Explicit correctness readback. It is not part of process and copies
 * full_width*full_height*4 bytes of the composited output from GPU to host. */
GPU_RESIDUAL_API int32_t gpu_residual_readback(GpuResidualBridge* bridge,
    uint8_t* rgba, size_t row_stride);
typedef struct GpuResidualTransferCounters {
    uint32_t struct_size;
    uint64_t full_device_copy_bytes;
    uint64_t working_device_copy_bytes;
    uint64_t output_readback_bytes;
} GpuResidualTransferCounters;
GPU_RESIDUAL_API int32_t gpu_residual_get_transfer_counters(
    GpuResidualBridge* bridge, GpuResidualTransferCounters* counters);
GPU_RESIDUAL_API void gpu_residual_close(GpuResidualBridge* bridge);
GPU_RESIDUAL_API const char* gpu_residual_last_error(void);

/* The scaled contract is a new export under an unchanged ABI number, the same
 * way gpu_lanczos_open_scaled extends the Lanczos bridge: nothing ABI 1
 * declared changes signature, layout or behaviour, so an ABI 1 caller loads and
 * runs against this DLL exactly as before. A caller that needs an upscale other
 * than the exact double asks for gpu_residual_open_scaled by name and treats
 * its absence as "this build cannot do it".
 *
 * GPU_RESIDUAL_MAX_TAPS is the tap window the plan kernel walks; the Catmull-Rom
 * axis plan the CPU reference builds is exactly this wide, and a narrower row is
 * padded by the bridge with weight 0.0f at a valid index rather than dropped.
 *
 * GpuResidualTapTable is GpuLanczosTapTables field for field on purpose, so one
 * mirror in the Python caller covers both bridges. Here the axes point the other
 * way: the source of each axis is the working extent and the destination is the
 * full extent, which is what an upscale plan is. Clamp-to-edge is already
 * resolved in the indices, and every array is [dst_len][taps] row-major.
 *
 * The plan upload is filter coefficients, not a frame: it is not part of
 * GpuResidualTransferCounters, which counts per-process payload bytes only. */
#define GPU_RESIDUAL_MAX_TAPS 4u
typedef struct GpuResidualTapTable {
    uint32_t struct_size;
    uint32_t taps;
    uint32_t vertical_src_len;      /* working height */
    uint32_t vertical_dst_len;      /* full height    */
    const int32_t* vertical_indices;
    const float* vertical_weights;
    uint32_t horizontal_src_len;    /* working width  */
    uint32_t horizontal_dst_len;    /* full width     */
    const int32_t* horizontal_indices;
    const float* horizontal_weights;
} GpuResidualTapTable;

/* Opens the table-driven upscale. Same borrowed device, same process(), the same
 * three counters with the same meaning - each one is computed from the extents
 * this call was given, so a 4:3 pair moves a 4:3 amount of pixel data. */
GPU_RESIDUAL_API int32_t gpu_residual_open_scaled(void* d3d12_device,
    uint32_t full_width, uint32_t full_height,
    uint32_t working_width, uint32_t working_height,
    const GpuResidualTapTable* plan, GpuResidualBridge** out_bridge);

#ifdef __cplusplus
}
#endif
#endif
