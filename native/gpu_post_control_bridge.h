#ifndef GPU_POST_CONTROL_BRIDGE_H
#define GPU_POST_CONTROL_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
# define GPU_POST_CONTROL_API __declspec(dllexport)
#endif

#define GPU_POST_CONTROL_ABI_VERSION 1u
#define GPU_POST_CANDIDATE_CUDA 1u
#define GPU_POST_CANDIDATE_D3D12 2u
#define GPU_POST_CANDIDATE_D3D12_COMPLETED 3u
#define GPU_POST_ROI_NONE 0u
#define GPU_POST_ROI_RECT 1u
#define GPU_POST_ROI_ELLIPSE 2u
#define GPU_POST_DEBUG_OFF 0u
#define GPU_POST_DEBUG_RAW 1u
#define GPU_POST_DEBUG_CONTROLLED 2u
#define GPU_POST_DEBUG_WEIGHT 3u

typedef struct GpuPostControlBridge GpuPostControlBridge;

/* ABI 1 is the independently derived GPU-only post-fusion residual controller.
 * The source and CUDA candidate use interop copies; a D3D12 candidate is copied
 * from a producer-fenced COPY_SOURCE texture into the same bridge-owned input
 * transport. Output is bridge-owned RGBA8 UNORM (pixel_bytes=4) or linear
 * RGBA16F (pixel_bytes=8), COPY_SOURCE, behind the returned producer fence. */
typedef struct GpuPostControlOptions {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t width;
    uint32_t height;
    uint32_t pixel_bytes;
    float total_gain;
    float lightness_gain;
    float chroma_gain;
    float low_frequency_gain;
    float high_frequency_gain;
    float hue_protection;
    float dark_protection;
    float highlight_protection;
    float soft_compression;
    uint32_t roi_kind;
    float roi_bounds[4]; /* normalized x0,y0,x1,y1 */
    float roi_feather;   /* normalized fraction of min(width,height) */
    uint32_t debug_mode;
} GpuPostControlOptions;

/* On x64 this is exactly 48 bytes. For CUDA kind, d3d_state must be zero and
 * only cuda_ptr/pitch are populated. Fenced D3D12 kind requires the CUDA fields
 * to be zero, d3d_texture/fence/value populated and d3d_state equal to
 * D3D12_RESOURCE_STATE_COPY_SOURCE (0x800). Completed D3D12 kind is reserved
 * for producers whose API synchronously waits its own queue before returning;
 * it requires texture/state and null fence/value 0. It is not a generic missing-
 * fence path. D3D12 has no resource-state query; the bridge validates the
 * declared state by copying the resource as COPY_SOURCE. */
typedef struct GpuPostCandidate {
    uint32_t kind;
    uint32_t d3d_state;
    uint64_t cuda_ptr;
    size_t cuda_pitch;
    void* d3d_texture;
    void* producer_fence;
    uint64_t producer_fence_value;
} GpuPostCandidate;

/* Lifetime counters. scalar_readback_bytes counts the two 16-byte diagnostic
 * structures read per process; no image pixels cross to the host. */
typedef struct GpuPostControlCounters {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t process_count;
    uint64_t source_transport_bytes;
    uint64_t candidate_transport_bytes;
    uint64_t scalar_readback_bytes;
    uint64_t unrepresentable_pixels;
} GpuPostControlCounters;

GPU_POST_CONTROL_API uint32_t gpu_post_control_abi_version(void);
GPU_POST_CONTROL_API int32_t gpu_post_control_open(
    void* d3d_device, void* cuda_context, uint32_t width, uint32_t height,
    uint32_t pixel_bytes, GpuPostControlBridge** out_bridge);
GPU_POST_CONTROL_API int32_t gpu_post_control_process(
    GpuPostControlBridge* bridge, uint64_t source_cuda_ptr,
    size_t source_pitch, const GpuPostCandidate* candidate,
    const GpuPostControlOptions* options, void** out_texture,
    void** out_fence, uint64_t* out_fence_value,
    GpuPostControlCounters* counters);
GPU_POST_CONTROL_API void gpu_post_control_close(GpuPostControlBridge* bridge);
GPU_POST_CONTROL_API const char* gpu_post_control_last_error(void);

#ifdef __cplusplus
}
#endif
#endif
