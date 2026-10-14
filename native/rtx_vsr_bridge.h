#pragma once
#include <stdint.h>

#ifdef _WIN32
#define RTX_VSR_API __declspec(dllexport)
#else
#define RTX_VSR_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RtxVsrHandle RtxVsrHandle;
typedef struct RtxVsrCounters {
    uint64_t process_calls;
    uint64_t successful_runs;
    uint32_t input_width;
    uint32_t input_height;
    uint32_t output_width;
    uint32_t output_height;
    uint32_t quality_level;
    uint32_t model_loaded;
} RtxVsrCounters;

RTX_VSR_API uint32_t rtx_vsr_abi_version(void);
RTX_VSR_API const char* rtx_vsr_last_error(void);
RTX_VSR_API int32_t rtx_vsr_open(void* cuda_context, uint32_t input_width,
    uint32_t input_height, uint32_t output_width, uint32_t output_height,
    uint32_t quality_level, const wchar_t* runtime_directory, RtxVsrHandle** out_handle);
RTX_VSR_API int32_t rtx_vsr_process(RtxVsrHandle* handle, uint64_t input_cuda_rgba,
    uint32_t input_pitch_bytes, uint64_t output_cuda_rgba, uint32_t output_pitch_bytes,
    void* cuda_stream);
RTX_VSR_API int32_t rtx_vsr_open2(void* cuda_context, uint32_t input_width,
    uint32_t input_height, uint32_t output_width, uint32_t output_height,
    uint32_t quality_level, uint32_t component_type,
    const wchar_t* runtime_directory, RtxVsrHandle** out_handle);
/* Additive API; old open/open2 preserve SDK-default Strength. */
RTX_VSR_API int32_t rtx_vsr_open3(void* cuda_context, uint32_t input_width,
    uint32_t input_height, uint32_t output_width, uint32_t output_height,
    uint32_t quality_level, uint32_t component_type, float strength,
    const wchar_t* runtime_directory, RtxVsrHandle** out_handle);
RTX_VSR_API int32_t rtx_vsr_get_configuration(RtxVsrHandle* handle,
    uint32_t* mode, float* strength, uint32_t* known);
RTX_VSR_API int32_t rtx_vsr_get_counters(RtxVsrHandle* handle, RtxVsrCounters* out_counters);
RTX_VSR_API void rtx_vsr_close(RtxVsrHandle* handle);
// The VFG handle uses the same counter/lifetime ABI; pixel buffers are CUDA.
RTX_VSR_API int32_t vfx_framegen_open(void* context, uint32_t width, uint32_t height,
    uint32_t multiplier, uint32_t mode, uint32_t packed10,
    const wchar_t* runtime_directory, RtxVsrHandle** out_handle);
RTX_VSR_API int32_t vfx_framegen_process(RtxVsrHandle* handle, uint64_t previous,
    uint32_t previous_pitch, uint64_t current, uint32_t current_pitch,
    uint64_t output, uint32_t output_pitch, uint32_t index, uint32_t reset);
RTX_VSR_API int32_t vfx_framegen_open2(void* context, uint32_t width, uint32_t height,
    uint32_t multiplier, uint32_t mode, uint32_t packed10, uint32_t scene_detection,
    const wchar_t* runtime_directory, RtxVsrHandle** out_handle);
RTX_VSR_API int32_t vfx_framegen_process_time(RtxVsrHandle* handle, uint64_t previous,
    uint32_t previous_pitch, uint64_t current, uint32_t current_pitch,
    uint64_t output, uint32_t output_pitch, float timestep, uint32_t reset);

RTX_VSR_API int32_t vfx_truehdr_open(void* context, uint32_t width, uint32_t height,
    uint32_t contrast, uint32_t saturation, uint32_t middle_gray, uint32_t luminance,
    uint32_t debanding_off, const wchar_t* runtime_directory, RtxVsrHandle** out_handle);
RTX_VSR_API int32_t vfx_truehdr_update(RtxVsrHandle* handle, uint32_t contrast,
    uint32_t saturation, uint32_t middle_gray, uint32_t luminance, uint32_t debanding_off);
RTX_VSR_API int32_t vfx_truehdr_process(RtxVsrHandle* handle, uint64_t input_rgba8,
    uint32_t input_pitch, uint64_t output_rgb10a2, uint32_t output_pitch);

#ifdef __cplusplus
}
#endif
