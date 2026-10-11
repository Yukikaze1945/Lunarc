#ifndef GPU_TEMPORAL_BRIDGE_H
#define GPU_TEMPORAL_BRIDGE_H

#include <stddef.h>
#include <stdint.h>
#include "gpu_post_control_bridge.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
# define GPU_TEMPORAL_API __declspec(dllexport)
#endif

#define GPU_TEMPORAL_ABI_VERSION 1u
typedef struct GpuTemporalBridge GpuTemporalBridge;

enum GpuTemporalSceneClass {
    GPU_TEMPORAL_FIRST = 0,
    GPU_TEMPORAL_DUPLICATE_STATIC = 1,
    GPU_TEMPORAL_EXPOSURE_FLASH = 2,
    GPU_TEMPORAL_TRANSLATION = 3,
    GPU_TEMPORAL_CUT = 4,
    GPU_TEMPORAL_AMBIGUOUS = 5,
    GPU_TEMPORAL_DISABLED = 6
};

/* Experimental sampled continuity report, not a semantic content classifier.
 * The guide search is bounded to +/-4 cells; large motion, sparse scenes, and
 * unusual flashes can remain ambiguous or resemble a cut. Cut thresholds are
 * conservative heuristics and are not a general content guarantee.
 * All counters except scene_reset_count describe this call. Shifts are in
 * fixed 32x18 guide cells (offset used to sample the previous guide), not
 * source-image pixels. scalar_readback_bytes counts one 32-byte GPU stats
 * readback and no image data. */
typedef struct GpuTemporalObserveReport {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t classification;
    uint32_t scene_cut;
    float similarity;
    float normalized_delta;
    float chroma_delta;
    float exposure_ratio;
    float shift_x;
    float shift_y;
    uint32_t scene_reset_count;
    uint32_t reserved;
    uint64_t source_transport_bytes;
    uint64_t scalar_readback_bytes;
} GpuTemporalObserveReport;

/* history_used/rejected count pixels. This is static low-frequency residual
 * stabilization, not motion reprojection; history_weight=0 is exact passthrough.
 * Transport and scalar-readback byte counts
 * describe this call; scene_reset_count is cumulative for this bridge handle.
 * The shader reads and writes signed scRGB FP32 intermediates while preserving
 * the caller's RGBA8/FP16 output format. */
typedef struct GpuTemporalProcessReport {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t history_used;
    uint32_t history_rejected;
    uint32_t unrepresentable_pixels;
    uint32_t reserved;
    float effective_history_weight;
    uint32_t scene_reset_count;
    uint64_t source_transport_bytes;
    uint64_t candidate_transport_bytes;
    uint64_t scalar_readback_bytes;
} GpuTemporalProcessReport;

GPU_TEMPORAL_API uint32_t gpu_temporal_abi_version(void);
GPU_TEMPORAL_API int32_t gpu_temporal_open(void* d3d_device,
    void* cuda_context, uint32_t width, uint32_t height, uint32_t pixel_bytes,
    GpuTemporalBridge** out_bridge);
GPU_TEMPORAL_API int32_t gpu_temporal_observe(GpuTemporalBridge* bridge,
    uint64_t source_cuda_ptr, size_t source_pitch, uint32_t reset,
    uint32_t scene_enabled, GpuTemporalObserveReport* report);
GPU_TEMPORAL_API int32_t gpu_temporal_process(GpuTemporalBridge* bridge,
    uint64_t source_cuda_ptr, size_t source_pitch,
    const GpuPostCandidate* candidate, float history_weight, uint32_t reset,
    void** out_texture, void** out_fence, uint64_t* out_fence_value,
    GpuTemporalProcessReport* report);
GPU_TEMPORAL_API void gpu_temporal_close(GpuTemporalBridge* bridge);
GPU_TEMPORAL_API const char* gpu_temporal_last_error(void);

#ifdef __cplusplus
}
#endif
#endif
