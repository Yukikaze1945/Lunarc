#ifndef GPU_LOW_RESIDUAL_BRIDGE_H
#define GPU_LOW_RESIDUAL_BRIDGE_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
#ifdef _WIN32
# define GPU_LOW_RESIDUAL_API __declspec(dllexport)
#endif

typedef struct GpuLowResidualBridge GpuLowResidualBridge;
/* The returned signed RGBA32F texture is borrowed and COPY_SOURCE. RGB is in
 * byte-valued units (DLSS byte minus original byte), alpha is zero. The
 * resource and producer fence live until the next process call or close. */
GPU_LOW_RESIDUAL_API uint32_t gpu_low_residual_abi_version(void);
GPU_LOW_RESIDUAL_API int32_t gpu_low_residual_open(void* device,
    uint32_t width, uint32_t height, GpuLowResidualBridge** out);
GPU_LOW_RESIDUAL_API int32_t gpu_low_residual_process(
    GpuLowResidualBridge* bridge, void* dlss_texture, void* dlss_fence,
    uint64_t dlss_fence_value, void* cuda_context, uint64_t working_rgba_ptr,
    size_t working_pitch, void** out_texture, void** out_fence,
    uint64_t* out_fence_value);
typedef struct GpuLowResidualCounters {
    uint32_t struct_size;
    uint64_t process_count;
    uint64_t working_device_copy_bytes;
    uint64_t output_readback_bytes;
} GpuLowResidualCounters;
GPU_LOW_RESIDUAL_API int32_t gpu_low_residual_get_transfer_counters(
    GpuLowResidualBridge* bridge, GpuLowResidualCounters* counters);
GPU_LOW_RESIDUAL_API void gpu_low_residual_close(GpuLowResidualBridge* bridge);
GPU_LOW_RESIDUAL_API const char* gpu_low_residual_last_error(void);
#ifdef __cplusplus
}
#endif
#endif
