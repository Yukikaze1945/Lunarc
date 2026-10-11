#ifndef DLSS_GPU_RESOURCES_H
#define DLSS_GPU_RESOURCES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#  ifdef DLSS_GPU_RESOURCES_EXPORTS
#    define DLSS_GPU_API __declspec(dllexport)
#  else
#    define DLSS_GPU_API __declspec(dllimport)
#  endif
#else
#  define DLSS_GPU_API
#endif

#define DLSS_GPU_RESOURCES_ABI_VERSION 1u

/* All pointers are borrowed COM interfaces from a trusted native caller.
 * A caller retains them until GPU processing has completed. This is the input
 * contract for a future DLSS GPU host; validation does not evaluate DLSS. */
typedef struct DlssGpuFrameResources {
    uint32_t struct_size;
    void* device;          /* ID3D12Device* */
    void* color;           /* ID3D12Resource*, R8G8B8A8_UNORM */
    void* motion;          /* ID3D12Resource*, R16G16_FLOAT */
    void* depth;           /* ID3D12Resource*, R32_FLOAT */
    void* output;          /* ID3D12Resource*, R8G8B8A8_UNORM, UAV capable */
    void* producer_fence;  /* ID3D12Fence*, optional when queue-ordered */
    uint64_t producer_value;
    uint32_t width;
    uint32_t height;
} DlssGpuFrameResources;

typedef enum DlssGpuValidationResult {
    DLSS_GPU_VALID = 0,
    DLSS_GPU_BAD_ARGUMENT = 1,
    DLSS_GPU_DEVICE_MISMATCH = 2,
    DLSS_GPU_BAD_RESOURCE = 3,
    DLSS_GPU_BAD_FORMAT = 4,
    DLSS_GPU_BAD_EXTENT = 5,
    DLSS_GPU_BAD_FENCE = 6
} DlssGpuValidationResult;

DLSS_GPU_API uint32_t dlss_gpu_resources_abi_version(void);

/* Validate shape, format, ownership and the producer fence. No GPU work is
 * submitted and no resource state is changed. The caller remains responsible
 * for correct D3D12 state transitions and queue synchronization. */
DLSS_GPU_API int32_t dlss_gpu_validate_frame(
    const DlssGpuFrameResources* frame);

#ifdef __cplusplus
}
#endif

#endif
