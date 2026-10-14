#ifndef NVOF_D3D12_INTEROP_H
#define NVOF_D3D12_INTEROP_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#  ifdef NVOF_D3D12_INTEROP_EXPORTS
#    define NVOF_D3D12_API __declspec(dllexport)
#  else
#    define NVOF_D3D12_API __declspec(dllimport)
#  endif
#endif

#define NVOF_D3D12_INTEROP_ABI_VERSION 1u

typedef struct NvofD3D12FlowBridge NvofD3D12FlowBridge;
typedef struct CudaD3D12ColorBridge CudaD3D12ColorBridge;

/* The NVOF session owns cuda_context and flow_ptr. The bridge owns a shared
 * D3D12 buffer and fence on the same physical adapter. The flow buffer stores
 * tightly packed float32x2 rows. DLSS needs R16G16_FLOAT texture conversion
 * after this handoff; the buffer is not a DLSS motion resource by itself. */
NVOF_D3D12_API uint32_t nvof_d3d12_abi_version(void);
NVOF_D3D12_API int32_t nvof_d3d12_create(
    void* d3d_device, void* cuda_context, uint32_t width, uint32_t height,
    NvofD3D12FlowBridge** out_bridge);
NVOF_D3D12_API int32_t nvof_d3d12_copy_flow(
    NvofD3D12FlowBridge* bridge, uint64_t flow_ptr, size_t flow_pitch,
    uint64_t* out_fence_value);
NVOF_D3D12_API int32_t nvof_d3d12_get_resources(
    NvofD3D12FlowBridge* bridge, void** out_buffer, void** out_fence);
/* Record float32x2 buffer -> R16G16_FLOAT texture conversion on a D3D12
 * command list. The caller must queue Wait(fence, value) before executing it.
 * The returned texture is borrowed and ends in NON_PIXEL_SHADER_RESOURCE. */
NVOF_D3D12_API int32_t nvof_d3d12_record_motion(
    NvofD3D12FlowBridge* bridge, void* command_list, void** out_texture);
NVOF_D3D12_API void nvof_d3d12_destroy(NvofD3D12FlowBridge* bridge);
NVOF_D3D12_API const char* nvof_d3d12_last_error(void);

/* Borrowed CUDA RGBA8 device rows -> D3D12 shared buffer. row_pitch is the
 * D3D12 placed-footprint pitch (256-byte aligned). The caller completes CUDA
 * production before copy_color and waits on the returned fence value on its
 * D3D12 queue before CopyTextureRegion. Destroy before the CUDA context. */
NVOF_D3D12_API int32_t cuda_d3d12_color_create(
    void* d3d_device, void* cuda_context, uint32_t width, uint32_t height,
    uint32_t row_pitch, CudaD3D12ColorBridge** out_bridge);
/* Create the same shared buffer with D3D12 UAV capability so a compute shader
 * can write RGBA rows and hand the mapped CUDA pointer to a downstream stage. */
NVOF_D3D12_API int32_t cuda_d3d12_color_create_output(
    void* d3d_device, void* cuda_context, uint32_t width, uint32_t height,
    uint32_t row_pitch, CudaD3D12ColorBridge** out_bridge);
NVOF_D3D12_API int32_t cuda_d3d12_color_get_device_pointer(
    CudaD3D12ColorBridge* bridge, uint64_t* out_ptr,
    uint32_t* out_pitch);
NVOF_D3D12_API int32_t cuda_d3d12_color_copy(
    CudaD3D12ColorBridge* bridge, uint64_t rgba_ptr, size_t rgba_pitch,
    uint64_t* out_fence_value);
NVOF_D3D12_API int32_t cuda_d3d12_color_get_resources(
    CudaD3D12ColorBridge* bridge, void** out_buffer, void** out_fence);
NVOF_D3D12_API void cuda_d3d12_color_destroy(CudaD3D12ColorBridge* bridge);

/* The payload width a colour bridge transports is a property of the bridge, not
 * of the entry point: an R16G16B16A16_FLOAT row is width * 8 bytes and the
 * width * 4 that every older entry point moved is only right for RGBA8. These
 * additions let a caller say which it means and make the bridge check that the
 * source pitch, the destination pitch and the row it copies all agree, so a
 * wider sample is transported whole instead of having its tail half dropped. */
#define CUDA_D3D12_COLOR_ROW_RGBA8 0x1u
#define CUDA_D3D12_COLOR_ROW_RGBA16F 0x2u
/* Which payload widths this build can move. A caller discovers the capability
 * from this symbol rather than from the ABI number, which is unchanged: an
 * interop DLL that predates it exports neither the query nor create2, and a
 * caller that cannot resolve them must refuse the wide format rather than
 * quietly transport it as RGBA8. */
NVOF_D3D12_API uint32_t cuda_d3d12_color_row_capabilities(void);

typedef struct CudaD3D12ColorLayout {
    uint32_t struct_size;
    uint32_t width;
    uint32_t height;
    /* The 256-byte aligned row pitch of the shared buffer, which is also the
     * pitch a consumer's placed-footprint copy has to use. */
    uint32_t row_pitch;
    /* The bytes of real payload in every row of every copy this bridge makes. */
    uint32_t bytes_per_row;
    uint32_t pixel_bytes;
    /* row_pitch * height: the whole allocation, tail padding of the last row
     * included. */
    uint64_t plane_bytes;
} CudaD3D12ColorLayout;

/* pixel_bytes is 4 or 8 and nothing else; the bridge derives bytes_per_row from
 * it and refuses a row_pitch that cannot hold that row, so a mismatched format
 * fails at creation instead of being reinterpreted on the way through. */
NVOF_D3D12_API int32_t cuda_d3d12_color_create2(
    void* d3d_device, void* cuda_context, uint32_t width, uint32_t height,
    uint32_t row_pitch, uint32_t pixel_bytes, int32_t allow_uav,
    CudaD3D12ColorBridge** out_bridge);
NVOF_D3D12_API int32_t cuda_d3d12_color_get_layout(
    CudaD3D12ColorBridge* bridge, CudaD3D12ColorLayout* out_layout);

#ifdef __cplusplus
}
#endif

#endif
