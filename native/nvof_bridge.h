/*
 * nvof_bridge.h — Clean C ABI Bridge for NVIDIA Hardware Optical Flow SDK
 */

#ifndef _NVOF_BRIDGE_H_
#define _NVOF_BRIDGE_H_

#include <stdint.h>

#if defined(_WIN32)
#  if defined(NVOF_BRIDGE_EXPORTS)
#    define NVOF_API __declspec(dllexport)
#  else
#    define NVOF_API __declspec(dllimport)
#  endif
#else
#  define NVOF_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Bumped whenever the exported C ABI changes in a way a caller cannot detect from the
 * symbol names alone. Version 1 had no NvOFBridgeTiming.dependency_wait_ms and carried a
 * convert_ms field that no kernel writes any more; a caller that read version 1 layout
 * out of a version 2 library would get silently wrong numbers, so the caller must check
 * this before trusting the struct.
 *
 * 3 adds NvOFBridgeTiming.input_convert_wait_ms, three appended NvOFBridgeCounters fields
 * and the nvof_bridge_execute_bgr_sequential export. Both structs grow at the end only, so
 * every field an ABI 2 caller knows keeps its offset - but nvof_bridge_get_counters and the
 * timing output write sizeof(their ABI 3 type), which is larger than an ABI 2 caller's
 * buffer. That is why the check is not optional: a caller must compare
 * nvof_bridge_get_abi_version() against the version it was built for and refuse the library
 * otherwise, which is what optical_flow.py does before it reads either struct. */
#define NVOF_BRIDGE_ABI_VERSION 3u

/* A stage this build path does not time arrives as this value. Callers must report it as
 * unavailable; a zero below means the span was measured and took no measurable time. */
#define NVOF_TIMING_UNMEASURED (-1.0)

NVOF_API uint32_t nvof_bridge_get_abi_version(void);

typedef struct NvOFBridgeProbeInfo {
    int32_t  is_available;             /* 1 if NVOF hardware and driver are ready, 0 otherwise */
    char     gpu_name[128];            /* GPU model name, e.g. "NVIDIA GeForce RTX 5060 Ti" */
    uint32_t driver_version;           /* Driver version (raw) */
    uint32_t max_api_version;         /* Max API version supported by driver, e.g. 0x50 for 5.0 */
    uint32_t min_width;                /* Minimum width supported, e.g. 32 */
    uint32_t min_height;               /* Minimum height supported, e.g. 32 */
    uint32_t max_width;                /* Maximum width supported, e.g. 8192 */
    uint32_t max_height;               /* Maximum height supported, e.g. 8192 */
    uint32_t supported_grid_mask;      /* Bitmask of supported grid sizes: bit 1 (1x1), bit 2 (2x2), bit 4 (4x4) */
    char     error_message[256];       /* Diagnostic message if unavailable */
} NvOFBridgeProbeInfo;

/**
 * Probe the system for NVIDIA Optical Flow capabilities.
 */
NVOF_API int32_t nvof_bridge_probe(NvOFBridgeProbeInfo* out_info);

/**
 * Create an NVOF hardware session with ABGR8 input format.
 *
 * @param width       Frame width in pixels
 * @param height      Frame height in pixels
 * @param grid_size   Output vector grid size: 1 (1x1 full res), 2 (2x2), or 4 (4x4)
 * @param perf_level  Performance level: 5 (SLOW), 10 (MEDIUM), 20 (FAST)
 * @param out_session Pointer to receive session opaque pointer
 * @return 0 on success, non-zero error code on failure.
 */
NVOF_API int32_t nvof_bridge_create(
    uint32_t width,
    uint32_t height,
    uint32_t grid_size,
    uint32_t perf_level,
    void** out_session
);

/* Each field is the duration of exactly one measured span, timed inside
 * nvof_bridge_execute_timed / nvof_bridge_execute_sequential with QueryPerformanceCounter.
 * A span that the bridge cannot observe on its own is not represented here at all; the
 * caller reports it as unavailable rather than as a zero or as a copy of another field. */
typedef struct NvOFBridgeTiming {
    double upload_ms;                     /* host->device cuMemcpy2D call(s) feeding this execute:
                                           * two after a temporal reset, one in steady state on the
                                           * sequential path */
    double submit_cpu_ms;                 /* nvOFExecute() alone: CPU submit/return, no GPU wait */
    double dependency_wait_ms;            /* cuCtxSynchronize() after nvOFExecute: NVOF hardware
                                           * completion. Unmeasured on the sequential path, which
                                           * folds that wait into the postprocess span below */
    double download_ms;                   /* device->host cuMemcpy2D of the float32 flow tensor */
    double postprocess_wait_inclusive_ms; /* postprocess cuLaunchKernel + its stream synchronize.
                                           * Inclusive by construction: the wait covers everything
                                           * still queued on that stream, and on the sequential
                                           * path that includes the NVOF hardware itself. The
                                           * S10.5->float32 conversion and the upscale are one
                                           * fused kernel inside this span and have no span of
                                           * their own */
    double total_native_ms;               /* wall clock across the whole native execute, including
                                           * context push/pop and the session critical section */
    /* Appended, so every ABI 2 field above keeps the offset an ABI 2 caller reads it at. */
    double input_convert_wait_ms;         /* ABI 3, BGR input path only: the cuLaunchKernel of the
                                           * byte-map kernel that turns the uploaded BGR24 staging
                                           * into the ABGR8 surface NVOF reads, plus the stream
                                           * synchronize needed for that surface to be safe to
                                           * consume. Nothing else lives in this span - not the
                                           * upload before it, not the flow postprocess after it.
                                           * Unmeasured (-1) on every RGBA-input entry point,
                                           * which never runs that kernel. */
} NvOFBridgeTiming;

/**
 * Execute hardware optical flow computation between current frame and reference frame.
 * Produces backward flow (cur -> prev) in full-resolution pixel displacement.
 *
 * @param session               Valid session handle
 * @param cur_abgr              Host pointer to current frame (cur, t), width*height*4 bytes (ABGR8/RGBA32)
 * @param prev_abgr             Host pointer to reference frame (prev, t-1), width*height*4 bytes (ABGR8/RGBA32)
 * @param out_flow              Host pointer to output buffer for (height * width * 2) float32 values
 * @param flow_stride_bytes     Stride of output flow in bytes (pass 0 for contiguous width*2*sizeof(float))
 * @param reset_temporal_hints  1 to reset/disable temporal hints (seek/jump/first-frame), 0 to use temporal hints
 * @return 0 on success, non-zero error code on failure.
 */
NVOF_API int32_t nvof_bridge_execute(
    void* session,
    const uint8_t* cur_abgr,
    const uint8_t* prev_abgr,
    float* out_flow,
    uint32_t flow_stride_bytes,
    int32_t reset_temporal_hints
);

/**
 * Execute hardware optical flow computation with detailed phase timing breakdown.
 *
 * @param session               Valid session handle
 * @param cur_abgr              Host pointer to current frame (cur, t), width*height*4 bytes (ABGR8/RGBA32)
 * @param prev_abgr             Host pointer to reference frame (prev, t-1), width*height*4 bytes (ABGR8/RGBA32)
 * @param out_flow              Host pointer to output buffer for (height * width * 2) float32 values
 * @param flow_stride_bytes     Stride of output flow in bytes (pass 0 for contiguous width*2*sizeof(float))
 * @param reset_temporal_hints  1 to reset/disable temporal hints (seek/jump/first-frame), 0 to use temporal hints
 * @param out_timing            Pointer to receive NvOFBridgeTiming (optional, can be NULL)
 * @return 0 on success, non-zero error code on failure.
 */
NVOF_API int32_t nvof_bridge_execute_timed(
    void* session,
    const uint8_t* cur_abgr,
    const uint8_t* prev_abgr,
    float* out_flow,
    uint32_t flow_stride_bytes,
    int32_t reset_temporal_hints,
    NvOFBridgeTiming* out_timing
);

/**
 * Execute hardware optical flow computation in sequential stream mode.
 * In steady-state continuous stream (when prev_abgr is NULL or reset_temporal_hints is 0),
 * reuses the previously uploaded frame on GPU as reference frame (hRef <-> hIn handle swap)
 * and uploads ONLY cur_abgr to the GPU (1 H2D upload per frame).
 *
 * @param session               Valid session handle
 * @param cur_abgr              Host pointer to current frame (cur, t), width*height*4 bytes (ABGR8/RGBA32)
 * @param prev_abgr             Host pointer to reference frame (prev, t-1), or NULL if sequential continuous
 * @param out_flow              Host pointer to output buffer for (height * width * 2) float32 values
 * @param flow_stride_bytes     Stride of output flow in bytes (pass 0 for contiguous width*2*sizeof(float))
 * @param reset_temporal_hints  1 if first frame or seek/jump, 0 for continuous sequential stream
 * @param out_timing            Pointer to receive NvOFBridgeTiming (optional, can be NULL)
 * @return 0 on success, non-zero error code on failure.
 */
NVOF_API int32_t nvof_bridge_execute_sequential(
    void* session,
    const uint8_t* cur_abgr,
    const uint8_t* prev_abgr,
    float* out_flow,
    uint32_t flow_stride_bytes,
    int32_t reset_temporal_hints,
    NvOFBridgeTiming* out_timing
);

/**
 * Execute hardware optical flow from BGR24 host frames, ABI 3.
 *
 * NVOF reads an ABGR8 surface, so a caller holding BGR24 frames used to reorder them into
 * RGBA on the CPU first: a whole extra pass that read width*height*4 bytes and wrote
 * width*height*3, and then uploaded 4 bytes per pixel over PCIe for a kernel to use the
 * fourth one as a constant. Here the 3-byte rows are uploaded as they lie, into a persistent
 * device staging buffer, and a byte-map kernel writes the surface instead. It permutes three
 * byte addresses into four and stores 255 in the fourth - no colour matrix, no gamma, no
 * range conversion - so for a cur_bgr whose RGBA form is cv2.cvtColor(cur_bgr,
 * COLOR_BGR2RGBA), the surface this fills holds exactly the same bytes the RGBA path would
 * have uploaded, and the flow it produces is bit-identical.
 *
 * The upload count is unchanged from the sequential RGBA path: one H2D per frame in steady
 * state (the previous frame is already on the GPU, and the input/reference handles swap), and
 * two on a reset frame. The steady-state payload is width*height*3 rather than width*height*4.
 *
 * @param session               Valid session handle
 * @param cur_bgr               Host pointer to current frame (cur, t), BGR24
 * @param cur_stride            Row stride of cur_bgr in bytes; 0 for width*3
 * @param prev_bgr              Host pointer to reference frame (prev, t-1) as BGR24, or NULL
 *                              to use cur_bgr as the reference the way the sequential path does
 * @param prev_stride           Row stride of prev_bgr in bytes; 0 for width*3
 * @param out_flow              Host pointer to output buffer for (height * width * 2) float32 values
 * @param flow_stride_bytes     Stride of output flow in bytes (pass 0 for contiguous width*2*sizeof(float))
 * @param reset_temporal_hints  1 if first frame or seek/jump, 0 for continuous sequential stream
 * @param out_timing            Pointer to receive NvOFBridgeTiming (optional, can be NULL).
 *                              input_convert_wait_ms is measured here and unmeasured on the
 *                              RGBA-input entry points; upload_ms is the BGR24 H2D alone.
 * @return 0 on success, non-zero error code on failure.
 */
NVOF_API int32_t nvof_bridge_execute_bgr_sequential(
    void* session,
    const uint8_t* cur_bgr,
    uint32_t cur_stride,
    const uint8_t* prev_bgr,
    uint32_t prev_stride,
    float* out_flow,
    uint32_t flow_stride_bytes,
    int32_t reset_temporal_hints,
    NvOFBridgeTiming* out_timing
);

/* Execute the same BGR24 sequential path but leave the float2 result on device.
 * No D2H copy is made. The result remains owned by the session and is overwritten
 * by the next execute or invalidated by destroy. Query its address and row pitch
 * with nvof_bridge_get_output_deviceptr after success. A consumer must use the
 * same CUDA context or establish explicit cross-API sharing and synchronization;
 * a CUdeviceptr cannot be passed to the existing host-pointer DLSSNR API.
 * This additive export keeps the ABI 3 timing/counter struct layout unchanged.
 * download_ms stays NVOF_TIMING_UNMEASURED on this path. */
NVOF_API int32_t nvof_bridge_execute_bgr_sequential_device(
    void* session,
    const uint8_t* cur_bgr,
    uint32_t cur_stride,
    const uint8_t* prev_bgr,
    uint32_t prev_stride,
    int32_t reset_temporal_hints,
    NvOFBridgeTiming* out_timing
);

/* Feed RGBA8 device rows from another CUDA context on the same GPU. This is a
 * device-to-device copy into the existing ABGR8 NVOF input surface, not an H2D
 * upload. The caller must finish producing the source before this call and keep
 * its allocation/context alive through the call. The float2 result stays on
 * device and is queried with nvof_bridge_get_output_deviceptr. The first or
 * discontinuous frame uses itself as the reference frame. */
NVOF_API int32_t nvof_bridge_execute_rgba_device_sequential(
    void* session, uint64_t cur_device_ptr, uint32_t cur_pitch_bytes,
    void* source_cuda_context, int32_t reset_temporal_hints,
    NvOFBridgeTiming* out_timing
);

/* Number and payload bytes of completed device-to-device input copies. */
NVOF_API int32_t nvof_bridge_get_device_input_counters(
    void* session, uint64_t* out_copy_count, uint64_t* out_bytes
);

/**
 * Retrieve the timing breakdown of the most recent execute operation.
 */
NVOF_API int32_t nvof_bridge_get_last_timing(
    void* session,
    NvOFBridgeTiming* out_timing
);

/**
 * Destroy the NVOF hardware session and release all GPU buffers and context.
 */
NVOF_API int32_t nvof_bridge_destroy(void* session);

typedef struct NvOFBridgeCounters {
    uint64_t execute_count;
    uint64_t h2d_copy_count;
    uint64_t h2d_bytes;
    uint64_t d2h_copy_count;
    uint64_t d2h_bytes;
    uint64_t reset_count;
    /* ABI 3, appended for the same reason as the timing field: a struct only ever grows at
     * the end. These three describe the BGR input path, and they are counted inside the
     * library because the claim they support - that this session took 3 bytes per pixel over
     * PCIe rather than 4, and converted it on the GPU rather than on the CPU - is one the
     * caller must not be asked to vouch for itself. */
    uint64_t bgr_staging_alloc_count;      /* device BGR staging buffers cuMemAlloc'd by this
                                            * session: 2, once each at create, never per frame */
    uint64_t bgr_input_call_count;         /* nvof_bridge_execute_bgr_sequential calls that got
                                            * their surfaces onto the GPU */
    uint64_t input_convert_launch_count;   /* byte-map kernel launches: one per surface filled,
                                            * so one in steady state and two on a reset frame */
} NvOFBridgeCounters;

/**
 * Retrieve the native diagnostic counters for session copy operations and executions.
 */
NVOF_API int32_t nvof_bridge_get_counters(
    void* session,
    NvOFBridgeCounters* out_counters
);

/**
 * Query session ownership and context details (device index, CUcontext pointer, thread owner).
 */
NVOF_API int32_t nvof_bridge_get_session_info(
    void* session,
    uint32_t* out_device_idx,
    void** out_ctx,
    uint32_t* out_thread_owner_id
);

/**
 * Query output float2 flow GPU device pointer and row pitch.
 *
 * This reports where the session's persistent flow tensor sits in GPU memory.
 * The opt-in D3D12 GPU host consumes it through explicit CUDA/D3D12 interop.
 * Never pass this device pointer to the old host-pointer dlssnr_process API.
 */
NVOF_API int32_t nvof_bridge_get_output_deviceptr(
    void* session,
    uint64_t* out_dptr,
    size_t* out_pitch_bytes
);

/**
 * Get the last error string for the current thread / session.
 */
NVOF_API const char* nvof_bridge_get_last_error(void);

/* Not part of this ABI: a build with NVOF_TEST_DEBUG_EXPORTS defined additionally exports
 *
 *   int32_t nvof_bridge_test_read_input_surface(void* session, int32_t which,
 *                                               uint8_t* dst, uint32_t dst_stride);
 *
 * which is 0 for the input surface (frame t) and 1 for the reference surface (frame t-1),
 * dst_stride 0 means the surface's own pitch, and dst receives width*height*4 bytes of the
 * ABGR8 surface as it sits on the GPU. It exists so a test can D2H the surface the byte-map
 * kernel filled and compare it against cv2.cvtColor(frame, COLOR_BGR2RGBA) byte for byte:
 * the claim that the GPU conversion is a pure permutation is then measured on the bytes
 * NVOF actually reads, not inferred from the kernel's source. A production build does not
 * export it, and no path in the pipeline reads an input surface back to the host. */

#ifdef __cplusplus
}
#endif

#endif /* _NVOF_BRIDGE_H_ */
