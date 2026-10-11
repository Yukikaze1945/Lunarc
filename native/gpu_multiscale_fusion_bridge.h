#ifndef GPU_MULTISCALE_FUSION_BRIDGE_H
#define GPU_MULTISCALE_FUSION_BRIDGE_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
#ifdef _WIN32
# define GPU_FUSION_API __declspec(dllexport)
#endif

typedef struct GpuFusionBridge GpuFusionBridge;
/* These arrays are the existing pipeline AxisPlan, laid out destination row
 * major. Every tap, including repeated clamp-to-edge indices, is retained. */
typedef struct GpuFusionAxisPlan {
    uint32_t src_len, dst_len, taps;
    const int32_t* indices;
    const float* weights;
} GpuFusionAxisPlan;
typedef struct GpuFusionLayerPlan {
    uint32_t width, height;
    GpuFusionAxisPlan horizontal, vertical;
    float weight;
} GpuFusionLayerPlan;
typedef struct GpuFusionOptions {
    uint32_t struct_size;
    uint32_t full_width, full_height, layer_count;
    GpuFusionLayerPlan layers[3];
    GpuFusionAxisPlan baseline_horizontal, baseline_vertical;
    float beta, residual_strength, relative_tolerance;
    uint32_t max_iterations;
} GpuFusionOptions;
typedef struct GpuFusionObservation {
    void* signed_texture;
    void* producer_fence;
    uint64_t producer_fence_value;
} GpuFusionObservation;
typedef struct GpuFusionSolveReport {
    uint32_t struct_size;
    uint32_t iterations;
    float initial_residual_norm;
    float final_residual_norm;
    float relative_residual;
    uint32_t converged;
} GpuFusionSolveReport;
typedef struct GpuFusionTransferCounters {
    uint32_t struct_size;
    uint64_t process_count;
    uint64_t full_device_copy_bytes;
    uint64_t scalar_readback_bytes;
    uint64_t output_readback_bytes;
} GpuFusionTransferCounters;

GPU_FUSION_API uint32_t gpu_fusion_abi_version(void);
GPU_FUSION_API int32_t gpu_fusion_open(void* d3d_device,
    const GpuFusionOptions* options, GpuFusionBridge** out);
GPU_FUSION_API int32_t gpu_fusion_process(GpuFusionBridge* bridge,
    void* cuda_context, uint64_t full_rgba_ptr, size_t full_pitch,
    const GpuFusionObservation* observations, uint32_t observation_count,
    void** out_rgba_texture, GpuFusionSolveReport* report);
GPU_FUSION_API int32_t gpu_fusion_get_transfer_counters(
    GpuFusionBridge* bridge, GpuFusionTransferCounters* counters);
GPU_FUSION_API void gpu_fusion_close(GpuFusionBridge* bridge);
GPU_FUSION_API const char* gpu_fusion_last_error(void);

/* --- Typed HDR (scRGB) path --------------------------------------------- */

/* The optional HDR group, versioned on its own. It does not change
 * GPU_FUSION_ABI_VERSION, GpuFusionOptions, GpuFusionObservation,
 * GpuFusionSolveReport, GpuFusionTransferCounters or any entry above: a bridge
 * opened with gpu_fusion_open still runs the whole RGBA8 chain, and a bridge
 * opened with gpu_fusion_open_hdr never can. A caller probes for these symbols
 * by name and reads gpu_fusion_hdr_abi_version(); it must not infer HDR support
 * from the ABI number above, which is unchanged.
 *
 * What the group adds is a colour contract, not a new solver. The objective,
 * the taps, the adjoint, the baseline, the CG and the failure reports are the
 * ones above, run through the same internal path, on the same float32 plans.
 * Three things differ, and each of them is a place where the SDR path assumed
 * four bytes a sample:
 *
 *   - the full-resolution source plane is a pristine linear scRGB
 *     R16G16B16A16_FLOAT plane (8 bytes a sample), transported at width * 8 and
 *     a 256-byte row pitch rather than at width * 4, so no row loses its tail;
 *   - each observation is the signed working residual an HDR layer's own
 *     gpu_hdr_observe handed back as a CUDA plane, widened device-side into the
 *     solver's FP32 work precision rather than reinterpreted as an R32 texture;
 *   - the fused output is a full-resolution R16G16B16A16_FLOAT texture in
 *     COPY_SOURCE state plus the fence and value that name its completion, so
 *     gpu_hdr_set_composite and then the existing PQ/P010 encode can consume it.
 *
 * Units. The source and the observations of an HDR bridge are linear scRGB with
 * 1.0 == 80 cd/m2, so a component is legitimately negative (out of gamut) and
 * legitimately above 1.0 (headroom). Nothing in this path scales, normalises or
 * clamps them: the SDR observation is in 8-bit pixel units and the HDR one is
 * not, which is why the two entries are separate rather than one entry with a
 * factor in it. */
#define GPU_FUSION_HDR_ABI_VERSION 1u

/* How this bridge was opened. struct_size must be sizeof of the caller's copy
 * and abi_version must be GPU_FUSION_HDR_ABI_VERSION; a caller that guessed the
 * struct out of a stale header is refused instead of read past its end. */
typedef struct GpuFusionHdrOptions {
    uint32_t struct_size;
    uint32_t abi_version;
    /* Bytes of one sample of the source plane and of each observation. 8 is
     * R16G16B16A16_FLOAT; no other value is a half plane, and nothing here
     * accepts 4: an HDR bridge that took RGBA8 would be an SDR bridge wearing
     * the wrong contract. */
    uint32_t pixel_bytes;
    uint32_t reserved;
} GpuFusionHdrOptions;

/* One layer's signed working residual, as gpu_hdr_observe returned it: a CUDA
 * device pointer into that bridge's own plane, at that layer's working extent,
 * RGB in linear scRGB and alpha 0.0 padding. This is not a proxy difference and
 * not an R32 texture, and it is not reinterpreted as one.
 *
 * Completion: gpu_hdr_observe waits its own pass fence before it hands the
 * plane back, so the samples are device-visible when this struct is filled.
 * The solver copies each plane with a device-to-device copy that it completes
 * inside the call, so the pointer only has to survive the call; the plane's
 * *contents* belong to the observing bridge and are rewritten by that bridge's
 * next observe() or reconstruct(), which is why each layer needs its own
 * bridge. A producer that has not retired its own work must be made complete by
 * its own caller first: this entry takes no fence because a CUDA device pointer
 * has no fence to name, and it will not pretend to wait one. */
typedef struct GpuFusionHdrObservation {
    uint64_t residual_ptr;
    uint32_t residual_pitch;   /* bytes; at least width * pixel_bytes */
    uint32_t width, height;    /* must equal this layer's plan extent */
} GpuFusionHdrObservation;

/* The solve evidence of one HDR frame, plus the store's own accounting. The
 * four CG fields mean exactly what GpuFusionSolveReport says they mean; an
 * unconverged or indefinitive solve is an error return, not a field here. */
typedef struct GpuFusionHdrReport {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t iterations;
    float initial_residual_norm;
    float final_residual_norm;
    float relative_residual;
    uint32_t converged;
    /* Components the half grid could not hold, counted by the composite itself
     * and read back as a scalar. Non-zero says the fused picture had pixels fall
     * back to the pristine master; it is reported, never smoothed over. */
    uint32_t unrepresentable_components;
    /* The store the caller gets: its DXGI_FORMAT, its extent, and the
     * D3D12_RESOURCE_STATES value it is left in, all as the enums the caller
     * already has spell them and all read back off the texture rather than
     * asserted. This build stores R16G16B16A16_FLOAT in
     * D3D12_RESOURCE_STATE_COPY_SOURCE; compare against your own enum, do not
     * hard-code the numbers. */
    uint32_t output_format;
    uint32_t output_width, output_height;
    uint32_t output_state;
    /* This call's own traffic, in bytes: the full-resolution source plane, the
     * per-layer observation planes, and the scalar reductions (the CG dot
     * products and the composite's counted fallback). The lifetime totals of the
     * same three are in GpuFusionHdrCounters. */
    uint64_t source_device_copy_bytes;
    uint64_t observation_device_copy_bytes;
    uint64_t scalar_readback_bytes;
} GpuFusionHdrReport;

/* Lifetime totals of the typed path, in a struct of its own so
 * GpuFusionTransferCounters keeps the layout the SDR callers read. */
typedef struct GpuFusionHdrCounters {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t process_count;
    uint64_t source_device_copy_bytes;
    uint64_t observation_device_copy_bytes;
    /* The CG dot reductions and the composite's unrepresentable-count readback:
     * scalars, never pictures. No field here claims host traffic is zero. */
    uint64_t scalar_readback_bytes;
    uint64_t unrepresentable_components;
} GpuFusionHdrCounters;

GPU_FUSION_API uint32_t gpu_fusion_hdr_abi_version(void);
/* Opens the solver for the scRGB contract. options is the same GpuFusionOptions
 * the SDR entry takes - the same float32 Lanczos taps, centres, clamp edges and
 * Catmull baseline, with the same explicit weights, beta, tolerance and
 * iteration budget - and nothing in it means anything different. What the HDR
 * options add is the sample width, checked against what the transport can
 * actually move on this build. */
GPU_FUSION_API int32_t gpu_fusion_open_hdr(void* d3d_device,
    const GpuFusionOptions* options, const GpuFusionHdrOptions* hdr_options,
    GpuFusionBridge** out);
/* Solves the objective and writes the fused scRGB composite.
 *
 * full_scrgb_ptr/full_scrgb_pitch name the pristine full-resolution linear
 * scRGB FP16 master the composite is built on (the plane gpu_hdr_decode or
 * gpu_hdr_import_master returned). It must already be complete and
 * device-visible, for the same reason and with the same rule as the
 * observation planes above.
 *
 * On success *out_texture is the bridge's own full-extent
 * R16G16B16A16_FLOAT texture, left in COPY_SOURCE state, and out_fence plus
 * out_fence_value name the bridge's own queue and the value that signals that
 * texture's writes have retired. The fence is real: it is the ID3D12Fence the
 * solver's queue signals, not a promise about a copy the solver did not make.
 * A consumer on another queue waits it and then takes the texture, which is
 * what gpu_hdr_set_composite does.
 *
 * The texture, the fence and the report belong to the bridge; the texture's
 * contents are overwritten by the next process_hdr on the same bridge.
 *
 * A fused component the half grid cannot hold falls back to the pristine master
 * sample and is counted in the report, exactly as the single-layer HDR composite
 * counts it; nothing is clamped to make it fit. If even that fallback sample is
 * outside the grid - a master that already carries a non-finite component - then
 * no finite picture exists, and the call fails instead of storing one. Any
 * non-zero return is a failure of the whole call: no texture, fence or partial
 * picture is handed back, and the bridge is marked failed, so every later frame on
 * it is refused at stage 1 until gpu_fusion_close retires it. The stage codes are
 * stable: 1 arguments or contract, 2 an observation that does
 * not describe its layer's plane, 3 a transport this build cannot make, 4 a CUDA
 * context that changed, 5 a source copy, 6 an observation copy, 7 a producer
 * wait, 8 to 12 the solver's own stages exactly as gpu_fusion_process numbers
 * them (dispatch, initial norm, CG step, lost positive definiteness, iteration
 * budget), 13 the store dispatch, 14 an unreadable counted fallback, 15 a store
 * that could only hold a value the half grid cannot, 16 a failed flush. */
GPU_FUSION_API int32_t gpu_fusion_process_hdr(GpuFusionBridge* bridge,
    void* cuda_context, uint64_t full_scrgb_ptr, size_t full_scrgb_pitch,
    const GpuFusionHdrObservation* observations, uint32_t observation_count,
    void** out_texture, void** out_fence, uint64_t* out_fence_value,
    GpuFusionHdrReport* report);
GPU_FUSION_API int32_t gpu_fusion_get_hdr_counters(
    GpuFusionBridge* bridge, GpuFusionHdrCounters* counters);
#ifdef __cplusplus
}
#endif
#endif
