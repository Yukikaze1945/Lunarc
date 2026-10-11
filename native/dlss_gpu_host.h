#ifndef DLSS_GPU_HOST_H
#define DLSS_GPU_HOST_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#  ifdef DLSS_GPU_HOST_EXPORTS
#    define DLSS_GPU_HOST_API __declspec(dllexport)
#  else
#    define DLSS_GPU_HOST_API __declspec(dllimport)
#  endif
#endif

#define DLSS_GPU_HOST_ABI_VERSION 3u

typedef struct DlssGpuHost DlssGpuHost;

typedef struct DlssGpuHostOptions {
    uint32_t struct_size;
    int32_t preset;
    int32_t style;
    float intensity;
    float local_tone;
    float local_structure;
    float skin_structure;
    int32_t auto_mask;
    int32_t ui_correction;
    float motion_scale_x;
    float motion_scale_y;
} DlssGpuHostOptions;

DLSS_GPU_HOST_API uint32_t dlss_gpu_host_abi_version(void);
/* Serialized with processing by the owning thread. Preset stays create-only. */
DLSS_GPU_HOST_API int32_t dlss_gpu_host_update_options(
    DlssGpuHost* host, const DlssGpuHostOptions* options);
DLSS_GPU_HOST_API int32_t dlss_gpu_host_open(
    uint32_t width, uint32_t height, const wchar_t* signed_snippet_path,
    const DlssGpuHostOptions* options, DlssGpuHost** out_host);
/* Official SuperSampling feature, PerfQuality=DLAA, equal input/output extent.
 * SDR only. Uses pixel-space optical-flow motion and zero depth/jitter because
 * prerecorded video has no engine depth or jitter; quality needs video testing. */
DLSS_GPU_HOST_API int32_t dlss_gpu_host_open_dlaa(
    uint32_t width, uint32_t height, const wchar_t* signed_dlss_path,
    const DlssGpuHostOptions* options, DlssGpuHost** out_host);
/* Independent DLAA feature on an existing NR device, for GPU post-processing.
 * Close this feature before its NR anchor. The signed modules remain separate. */
DLSS_GPU_HOST_API int32_t dlss_gpu_host_open_dlaa_layer(
    DlssGpuHost* anchor, uint32_t width, uint32_t height,
    const wchar_t* signed_dlss_path, const DlssGpuHostOptions* options,
    DlssGpuHost** out_host);
DLSS_GPU_HOST_API int32_t dlss_gpu_host_dlaa_counters(uint64_t* three_counts);
typedef struct DlssGpuHostOpenInfo DlssGpuHostOpenInfo;
DLSS_GPU_HOST_API int32_t dlss_gpu_host_open_dlaa_ex(
    uint32_t width, uint32_t height, const wchar_t* signed_dlss_path,
    const DlssGpuHostOptions* options, const DlssGpuHostOpenInfo* info,
    DlssGpuHost** out_host);
DLSS_GPU_HOST_API int32_t dlss_gpu_host_open_dlaa_layer_ex(
    DlssGpuHost* anchor, uint32_t width, uint32_t height,
    const wchar_t* signed_dlss_path, const DlssGpuHostOptions* options,
    const DlssGpuHostOpenInfo* info, DlssGpuHost** out_host);

/* Experimental, opt-in per-session colour format.
 *
 * `dlss_gpu_host_open` and every older entry point keep the colour they always
 * had - R8G8B8A8_UNORM, four bytes a pixel - and a session opened without an
 * open info is exactly such a session. `dlss_gpu_host_open_ex` lets one session
 * be R16G16B16A16_FLOAT, eight bytes a pixel, for the HDR chain: its Feature 18
 * Color and Output textures, every row it copies or reads back, and the format
 * it reports are that width, while the rest of the process keeps RGBA8 sessions
 * untouched. Colour is chosen at open, never per frame, and a session never
 * converts: an RGBA8 caller on an FP16 session is refused by the row-size check
 * rather than quietly reinterpreted.
 *
 * The capability is a property of the running build, so a caller asks for it
 * rather than assuming it. An older dlss_gpu_host.dll exports neither
 * `dlss_gpu_host_query_capabilities` nor `dlss_gpu_host_open_ex` (the ABI number
 * stays 3 because nothing an existing caller binds changed), and an FP16 open
 * against an nvof_d3d12_interop.dll that cannot transport an eight-byte row
 * refuses with code 23 instead of falling back to an 8-bit picture. */
#define DLSS_GPU_HOST_COLOR_RGBA8 0u
#define DLSS_GPU_HOST_COLOR_RGBA16F 1u
#define DLSS_GPU_HOST_COLOR_FORMAT_BIT(format) (1u << (format))

typedef struct DlssGpuHostOpenInfo {
    uint32_t struct_size;
    uint32_t color_format; /* DLSS_GPU_HOST_COLOR_* */
    uint32_t flags;        /* none defined; must be zero */
} DlssGpuHostOpenInfo;

typedef struct DlssGpuHostCapabilities {
    uint32_t struct_size;
    /* Every colour format this host can open a session with. */
    uint32_t color_formats;
    /* The row payload widths the nvof_d3d12_interop.dll actually loaded here
     * can transport, as cuda_d3d12_color_row_capabilities reports them: 1 for a
     * four-byte row, 2 for an eight-byte row. Zero when that DLL predates the
     * query, which is how an old bridge is told apart from a missing export. */
    uint32_t row_bridge_capabilities;
    /* Non-zero when the wide-row entry points resolved, so an FP16 open will
     * not be refused for lack of a transport. */
    uint32_t fp16_row_bridge_ready;
} DlssGpuHostCapabilities;

/* Open codes 22 and 23 are the two new refusals: 22 an unknown colour format or
 * a bad open info, 23 a colour format this build cannot transport. Codes 1 to
 * 21 mean what they meant; nothing about an RGBA8 open changed. */
DLSS_GPU_HOST_API int32_t dlss_gpu_host_query_capabilities(
    DlssGpuHostCapabilities* out_capabilities);
DLSS_GPU_HOST_API int32_t dlss_gpu_host_open_ex(
    uint32_t width, uint32_t height, const wchar_t* signed_snippet_path,
    const DlssGpuHostOptions* options, const DlssGpuHostOpenInfo* info,
    DlssGpuHost** out_host);
DLSS_GPU_HOST_API int32_t dlss_gpu_host_open_layer_ex(
    DlssGpuHost* anchor, uint32_t width, uint32_t height,
    const DlssGpuHostOptions* options, const DlssGpuHostOpenInfo* info,
    DlssGpuHost** out_host);

/* The D3D12 device the session made its Feature 18 textures on - the same
 * pointer the device output struct already reports, available before the first
 * frame so a producer or consumer bridge that has to share the device (the HDR
 * scRGB bridge insists on it) can be opened against a session that exists.
 * Borrowed: the host owns it and releases it at close. */
DLSS_GPU_HOST_API void* dlss_gpu_host_d3d12_device(DlssGpuHost* host);

/* What colour this session actually is, and what its row transport actually
 * moved, so a gate can read the format and pitch off the running session
 * instead of off the setting that asked for them. `bridge_bytes_per_row` and
 * `bridge_row_pitch` are the live color bridge's own layout (zero before the
 * first device-colour frame creates it); `last_copy_*` describe the most recent
 * device-colour copy. */
typedef struct DlssGpuHostColorInfo {
    uint32_t struct_size;
    uint32_t width;
    uint32_t height;
    uint32_t dxgi_format;   /* DXGI_FORMAT_R8G8B8A8_UNORM or _R16G16B16A16_FLOAT */
    uint32_t bytes_per_pixel;
    uint64_t row_bytes;     /* width * bytes_per_pixel: the payload of one row */
    uint32_t color_footprint_pitch;
    uint32_t output_footprint_pitch;
    uint32_t bridge_row_pitch;
    uint32_t bridge_bytes_per_row;
    uint64_t last_copy_src_pitch;
    uint64_t last_copy_row_bytes;
    uint64_t last_copy_fence_value;
    uint64_t device_color_copy_count;
} DlssGpuHostColorInfo;
DLSS_GPU_HOST_API int32_t dlss_gpu_host_get_color_info(
    DlssGpuHost* host, DlssGpuHostColorInfo* out_info);

/* Create an independent Feature 18 session on an existing host's NGX core and
 * D3D12 device. The anchor may be closed first; the shared core is released
 * only after the final session closes. Each layer owns its own parameters,
 * feature handle, textures, command resources and temporal state. */
DLSS_GPU_HOST_API int32_t dlss_gpu_host_open_layer(
    DlssGpuHost* anchor, uint32_t width, uint32_t height,
    const DlssGpuHostOptions* options, DlssGpuHost** out_host);
/* Monotonic opaque ID for diagnostics; zero is invalid. This never exposes an
 * NGX handle or D3D12 pointer. */
DLSS_GPU_HOST_API uint64_t dlss_gpu_host_session_id(DlssGpuHost* host);
/* Flow pointer/context are borrowed from a live NVOF session and must stay
 * valid until this synchronous call returns. The host copies flow between GPU
 * allocations and never downloads it to CPU. Color/output remain host
 * buffers in this phase. Depth is an immutable zero GPU texture. */
DLSS_GPU_HOST_API int32_t dlss_gpu_host_process(
    DlssGpuHost* host, const uint8_t* rgba, size_t rgba_stride,
    void* cuda_context, uint64_t flow_ptr, size_t flow_pitch, int32_t reset,
    uint8_t* output_rgba, size_t output_stride);
/* Additive GPU-color entry point. RGBA8 and float2 pointers are borrowed from
 * live CUDA allocations, which may belong to different contexts on the same
 * adapter. Their producers must complete before this synchronous call. Output
 * still returns to CPU for the current residual stage. */
DLSS_GPU_HOST_API int32_t dlss_gpu_host_process_device_color(
    DlssGpuHost* host, void* color_cuda_context, uint64_t rgba_ptr,
    size_t rgba_pitch, void* flow_cuda_context, uint64_t flow_ptr,
    size_t flow_pitch, int32_t reset, uint8_t* output_rgba,
    size_t output_stride);

/* The borrowed D3D12 output belongs to the host device and is in COPY_SOURCE
 * state. The returned fence value has completed before this call returns.
 * Consume the resource before the next process call or close; do not Release
 * the borrowed COM pointers. A consumer using another queue must wait on the
 * returned fence before accessing the texture. */
typedef struct DlssGpuHostDeviceOutput {
    uint32_t struct_size;
    uint32_t width;
    uint32_t height;
    uint32_t dxgi_format; /* the session's colour: RGBA8, or RGBA16F on a
                             session opened with DLSS_GPU_HOST_COLOR_RGBA16F */
    void* d3d12_device;
    void* d3d12_resource;
    void* d3d12_fence;
    uint64_t fence_value;
} DlssGpuHostDeviceOutput;
DLSS_GPU_HOST_API int32_t dlss_gpu_host_process_device_color_resource(
    DlssGpuHost* host, void* color_cuda_context, uint64_t rgba_ptr,
    size_t rgba_pitch, void* flow_cuda_context, uint64_t flow_ptr,
    size_t flow_pitch, int32_t reset, DlssGpuHostDeviceOutput* output);

/* Guidance without motion vectors. DLSSNR.MVec is bound to a texture the host
 * created at open and zeroed once: same extent, DXGI_FORMAT_R16G16_FLOAT and
 * NON_PIXEL_SHADER_RESOURCE state as the flow conversion produces, but written
 * by nobody after open. Neither entry point touches CUDA for flow, so no NVOF
 * session, flow bridge, fence wait or per-frame motion upload is involved.
 * DLSSNR.Reset keeps the meaning it has on the flow entries. */
DLSS_GPU_HOST_API int32_t dlss_gpu_host_process_zero_motion(
    DlssGpuHost* host, const uint8_t* rgba, size_t rgba_stride, int32_t reset,
    uint8_t* output_rgba, size_t output_stride);
/* The unguided form of the device-color device-output entry; the output
 * contract, including the completed fence value, is identical. */
DLSS_GPU_HOST_API int32_t dlss_gpu_host_process_device_color_zero_motion_resource(
    DlssGpuHost* host, void* color_cuda_context, uint64_t rgba_ptr,
    size_t rgba_pitch, int32_t reset, DlssGpuHostDeviceOutput* output);
/* Guided host-pointer motion: a CPU motion plane in place of a borrowed NVOF
 * flow descriptor, for a caller that already holds motion in host memory (the
 * old dlssnr_host.dll Live.process contract) and must not load CUDA or an NVOF
 * session to run one guided frame. Color and output are host buffers, as on the
 * host-pointer entries above.
 *
 * `motion` points at rows of float32x2 pixel-space displacement vectors, exactly
 * the shape the legacy host took (an HxWx2 plane): full DLSSNR.Color extent
 * (width x height), row-major with x fastest and a top-left origin, so texel
 * (x, y) carries the (dx, dy) displacement of color pixel (x, y). The host stages
 * the plane to the GPU and converts it to the bound R16G16_FLOAT motion texture
 * with the same compute pass the guided NVOF conversion runs, so a host frame and
 * a flow frame hand DLSS byte-identical motion for the same float32 vectors. The
 * host never reinterprets, scales, flips or zeroes the values. motion_stride is
 * the row pitch in bytes and must be at least width * 8. Depth stays the
 * immutable zero texture. MVecScaleX/Y keep the meaning they have on the flow
 * entries: the scale is applied by DLSS, not baked into the plane. */
DLSS_GPU_HOST_API int32_t dlss_gpu_host_process_host_motion(
    DlssGpuHost* host, const uint8_t* rgba, size_t rgba_stride,
    const uint8_t* motion, size_t motion_stride, int32_t reset,
    uint8_t* output_rgba, size_t output_stride);
/* Explicit readback of the resident motion texture the host-pointer and
 * zero-motion entries bind, in its DXGI_FORMAT_R16G16_FLOAT placed-footprint
 * row order. This is a D2H transfer for a correctness check that a host-motion
 * frame wrote the plane DLSS will read, not zeros; it is not on any frame path
 * and does not touch the flow bridge's own motion resource. dst_stride is the
 * row pitch in bytes and must be at least width * 4. */
DLSS_GPU_HOST_API int32_t dlss_gpu_host_readback_motion(
    DlssGpuHost* host, uint8_t* dst, size_t dst_stride);
/* Explicit readback of the most recent device output, for CPU consumers and
 * parity checks. This is a D2H transfer and increments the readback counters.
 * The plane is the session's own colour, so output_stride is at least
 * width * bytes_per_pixel: four on an RGBA8 session, eight on an FP16 one. */
DLSS_GPU_HOST_API int32_t dlss_gpu_host_readback_output(
    DlssGpuHost* host, uint8_t* output_rgba, size_t output_stride);

typedef struct DlssGpuHostTransferCounters {
    uint32_t struct_size;
    uint64_t host_color_stage_count;
    uint64_t host_color_stage_bytes;
    uint64_t device_color_copy_count;
    uint64_t device_color_copy_bytes;
    uint64_t output_readback_count;
    uint64_t output_readback_bytes;
} DlssGpuHostTransferCounters;
DLSS_GPU_HOST_API int32_t dlss_gpu_host_get_transfer_counters(
    DlssGpuHost* host, DlssGpuHostTransferCounters* out_counters);
DLSS_GPU_HOST_API void dlss_gpu_host_close(DlssGpuHost* host);
DLSS_GPU_HOST_API const char* dlss_gpu_host_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
