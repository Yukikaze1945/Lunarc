# Feature 18 HDR gate audit (2026-09-29)

Scope: isolated format/number probes and source audit. The production GPU
Transfer path still rejects P010/HDR. Performance is `NOT_CHARACTERIZED`.

## Feature 18 color-domain contract

The installed NVIDIA DLSS SDK (`F:/dlss/DLSS-310.5.3`) documents the general
DLSS SR `NVSDK_NGX_DLSS_Feature_Flags_IsHDR`, `ExposureTexture`,
`DLSS.Pre.Exposure`, and `DLSS.Exposure.Scale` controls. Its programming guide
describes linear HDR before tone mapping for general DLSS. These declarations
do **not** establish that Feature 18 (`DLSSNR.Color` / `DLSSNR.Output`) accepts
linear scRGB with native HDR headroom. Neither the local SDK headers nor the
audited Magpie Feature 18 bindings document a Feature 18-specific
`Exposure`, `PreExposure`, `HDR`, or `ColorSpace` parameter. No official
Feature 18 PQ/HLG or linear-scRGB semantic contract was found.

Magpie's full experimental route is reconstructed in
`MAGPIE_HDR_SCRGB_REFERENCE.md`. It uses an *external* bounded normalization,
not a documented Feature 18 exposure parameter.

## Isolated FP16 probe

`native/build_dlss_gpu_fp16_probe.bat` builds a separate diagnostic DLL under
`build/fp16_probe/`. Only that DLL defines `DLSS_GPU_HOST_FP16_PROBE`, making
the Feature 18 color/output textures `DXGI_FORMAT_R16G16B16A16_FLOAT`. Run
`python native/dlss_gpu_fp16_probe.py`. The probe submits a uniform 640x360
FP16 RGB patch with alpha 1, zero motion and `Reset=1` for each call. Its
probe-only readback copies the actual color texture **before** NGX evaluation.
Every pre-NGX pixel exactly matched the submitted half values.

| Submitted RGB | Pre-NGX texture | Feature 18 central median | Output max |
|---:|---:|---:|---:|
| 0.1 | 0.0999756 | 0.1011963 | 0.1038818 |
| 0.5 | 0.5 | 0.5039063 | 0.5078125 |
| 1.0 | 1.0 | 0.9995117 | 1.0 |
| 1.5 | 1.5 | 1.0 | 1.0 |
| 2.0 | 2.0 | 1.0 | 1.0 |
| 4.0 | 4.0 | 1.0 | 1.0 |

Thus >1.0 survives host staging, D3D12 upload and the FP16 input texture.
The clamp occurs at or within Feature 18 evaluation/output. A private NGX
internal stage cannot be isolated by this probe.

An **external**, diagnostic input x0.25 followed by output x4 produced central
medians 0.10022, 0.50537, 1.00586, 1.51367, 2.01563 and 3.99805 for the
six original levels. It numerically recovers headroom on flat patches only.
It is not exactly reversible through an image processor and does not establish
spatial, temporal, colorimetric or HDR correctness.

The isolated probe also set the *generic DLSS SR* scalar keys to
`(PreExposure,ExposureScale)=(0.25,1),(1,0.25),(4,1)`. Tested outputs for raw
inputs 0.5, 2.0, 4.0 stayed at central medians 0.5039063, 1.0, 1.0 for every
setting. These keys did not restore headroom in this Feature 18 configuration;
that observation does not establish behavior of other configurations or DLLs.

## Independent NVOF guidance design

Retain the decoded HDR main color as GPU-resident FP16. From the same frame,
make a *separate* 8-bit guidance surface at each layer's resolved working
extent, solely for NVOF. For limited-range P010 Y, read its 10-bit code as
`word >> 6` and map `round(255 * clamp((Y-64)/876, 0, 1))` to GRAYSCALE8.
Alternately define an explicit luminance mapping from FP16 color; in either
case, make that mapping independent of the main color texture and record its
range/transfer semantics. Query the installed NVOF driver for GRAYSCALE8
support, or use a separately tested ABGR8 guidance surface. NVOF outputs
motion only; the FP16 main color never travels through the guidance format.
Synchronize guidance and main color by frame ID, extent, layer and reset, with
no host pixel copy. This path is a design only; the present bridge still uses
ABGR8 and the present Python device input requires `rgba8_cuda`.

## Decision

**Blocked for production HDR export.** A direct Feature 18 color/exposure
contract is missing, and this local Feature 18 clamps raw FP16 RGB >1.0.
External normalization warrants a separate research proposal with structured
image, temporal and color tests; it is not authorization to enable HDR export.
