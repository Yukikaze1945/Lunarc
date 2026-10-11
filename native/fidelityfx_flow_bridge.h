#pragma once
#include <stdint.h>
#ifdef _WIN32
#define FFX_FLOW_API __declspec(dllexport)
#else
#define FFX_FLOW_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct FfxFlowHandle FfxFlowHandle;
typedef struct FfxFlowCounters {
    uint64_t process_calls;
    uint64_t successful_dispatches;
    uint64_t reset_dispatches;
    uint64_t scd_scene_change_bits;
    uint32_t width, height, grid_width, grid_height;
    uint32_t output_pitch_bytes, flow_format_float32x2;
    uint32_t scd_history_bits, scd_workgroups;
} FfxFlowCounters;
FFX_FLOW_API uint32_t ffx_flow_abi_version(void);
FFX_FLOW_API const char* ffx_flow_last_error(void);
FFX_FLOW_API int32_t ffx_flow_open(void* borrowed_d3d12_device, void* cuda_context,
    uint32_t width, uint32_t height, FfxFlowHandle** out_handle);
/* mode: 0 = half-resolution analysis, 1 = full-resolution analysis.
 * Output always uses the original input extent and input-pixel displacement. */
FFX_FLOW_API int32_t ffx_flow_open2(void* borrowed_d3d12_device, void* cuda_context,
    uint32_t width, uint32_t height, uint32_t mode, FfxFlowHandle** out_handle);
/* Input is borrowed CUDA RGBA8. Output is bridge-owned CUDA float32x2 in full
 * resolution and remains valid until close or the next process call. */
FFX_FLOW_API int32_t ffx_flow_process(FfxFlowHandle* handle,
    uint64_t input_cuda_rgba8, uint32_t input_pitch_bytes, int32_t reset,
    uint64_t* out_cuda_float32x2, uint32_t* out_pitch_bytes);
FFX_FLOW_API int32_t ffx_flow_get_counters(FfxFlowHandle*, FfxFlowCounters*);
FFX_FLOW_API void ffx_flow_close(FfxFlowHandle*);
#ifdef __cplusplus
}
#endif
