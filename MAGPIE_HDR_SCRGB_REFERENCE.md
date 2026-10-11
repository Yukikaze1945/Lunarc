# Magpie HDR/scRGB and DLSSNR reference trace

Read-only reconstruction of the bundled `Magpie/` reference project on
2026-09-29. This describes what Magpie implements; it is **not** a NVIDIA
Feature 18 HDR contract or a proposal to enable our production HDR export.

## End-to-end path

```
capture texture + declared/inferred color metadata
  -> HdrCaptureProcessor: decode transfer, create linear scRGB RGBA16F
  -> HdrFrame canonical surface (reference white: 80 nit = 1.0)
  -> EffectDrawer: bounded normalization on GPU for DLSSNR
  -> DLSSNRFilter: D3D11/D3D12 shared RGBA16F input -> Feature 18 -> RGBA16F output
  -> EffectDrawer: inverse bounded normalization -> canonical RGBA16F
  -> renderer shared publication / presenter: canonical FP16
```

**Capture and semantics.** `Magpie/src/Magpie.Core/HdrCaptureProcessor.cpp`
decodes sRGB, PQ and HLG in its `ToCanonical` shader. PQ values decode to
absolute nits, then divide by 80; sRGB/HLG relative values are embedded at
`sdrWhiteNits / 80`. Linear input passes through apart from the metadata's
exposure multiplier. The output resource is `R16G16B16A16_FLOAT`. In the
Graphics Capture case, Magpie treats a WGC FP16 surface as already linear
scRGB. `Magpie/src/Magpie.Core/DlssnrAutoHdr.h` explicitly says that FP16
storage alone does not imply this interpretation. `HdrFrame` carries transfer,
primaries, range, white point and source format separately from texture bits.

**Route selection.** `GroupBHdrRoutes.cpp` selects `DLSSNR/sdr-r8` normally.
With experimental HDR enabled and scale exactly 1, 2 or 4.5, it selects
`experimental-fp16-scale`: FP16 input/output, linear transfer, scene-linear
range, `BoundedHDR` adapter, `LocalValidation` evidence, and `hdrNative=false`.
`GroupBEffectProtocol.h::DlssnrExperimentProtocol` holds those three scales.
`DLSSNRFilter.cpp::ParseDLSSNRSettings` defaults the automatic HDR scale to 1.
`Renderer.cpp::_BuildEffects` enables the automatic route only when HDR
compatibility and FP16 support are present with valid canonical input.

**Bounded mapping.** `EffectDrawer.cpp::PrepareHdrInput` calls
`HdrSurfaceAdapter::ConvertHdrToBounded`, and `CompleteHdrOutput` calls
`ConvertBoundedToHdr`. The shader in `HdrSurfaceAdapter.cpp` implements

```
S = sdrWhiteNits / 80
k = normalizationScale
boundedRGB = max(canonicalRGB, 0) * k / max(S, 0.0001)
restoredRGB = max(featureOutputRGB, 0) * S / max(k, 0.0001)
```

These are `mode=2` and `mode=3`; they are **simple linear scaling**, not the
logarithmic `MapHdrToSdr`/`MapSdrToHdr` used by other adapter modes. The shader
does not explicitly clamp `boundedRGB` to 1, but Feature 18 may do so, as our
isolated probe shows. The domain is information-preserving only while every
relevant nonnegative channel satisfies `canonicalRGB * k / S <= 1`. For an
80-nit SDR white (`S=1`) and default `k=1`, any channel >1 reaches Feature 18
above its observed ceiling. For `S=4.5,k=4.5`, the same is true. Negative
scRGB channels are discarded by `max(...,0)` even before Feature 18. A fixed
scale cannot protect arbitrary HDR peaks.

**Feature 18 input/output.** In `DLSSNRFilter.cpp::Initialize`, the FP16
experiment requires same-sized FP16 input and output; it disables the SDR
resolution-scaling/residual branch. It creates D3D11/D3D12 shared FP16 input
and output resources with fences. `PrepareInput` copies the FP16 input into
the shared input without an 8-bit conversion. Create parameters set Feature
18 extent/scale/preset and generic NGX dimensions. Evaluate parameters bind
`DLSSNR.Color`, `DLSSNR.Output`, `DLSSNR.MVec`, `DLSSNR.Depth`, reset and
style controls. There is no Feature 18-specific HDR, color-space, exposure
or pre-exposure parameter in these two binding functions. `Draw` waits for
the guidance/input fence, evaluates Feature 18 once, waits for output, then
copies the shared FP16 output to the effect output; the boundary adapter does
the inverse normalization outside `DLSSNRFilter`.

**Optical flow is an auxiliary.** `NvidiaOpticalFlowProvider.cpp` accepts a
canonical FP16 frame in HDR mode but runs its separate `HDR_TO_NVOF_HLSL`
conversion: `saturate(max(rgb,0)/4.5)` into an 8-bit NVOF input texture.
It queries driver formats, preferring ABGR8, then GRAYSCALE8/NV12 where
available. Its output is motion (`R16G16_FLOAT`) and confidence, not color.
The 4.5 guidance divisor is a local motion-contrast policy; highlights above
that guidance range saturate. It does not reduce the main FP16 color surface.

**Output.** `Renderer.cpp::_PublishBackendTexture` requires canonical FP16
for ordinary HDR publication and carries color metadata. The separate
XeSS-FG terminal path converts canonical values to HDR10/Rec.2020/PQ R10;
that branch is not the DLSSNR output contract.

## Gaps before borrowing this design

1. Magpie itself labels this DLSSNR FP16 route `LocalValidation`,
   `BoundedHDR`, `hdrNative=false`; the docs explicitly say it is not vendor
   HDR proof (`docs/experimental/HDR_EFFECT_PROTOCOL_AUDIT_20260907.md`).
2. Magpie's normalization is pairwise algebraic but Feature 18 need not be
   scale-equivariant. Our flat-patch probe finds direct >1.0 clipping and only
   approximate numeric recovery with external scaling.
3. The capture shader shown here decodes PQ but does not visibly apply the
   Rec.2020-to-Rec.709 primary matrix that the separate HDR10 presentation
   shader applies in reverse. A source-primary/colorimetric audit is required
   before using its PQ path as a template.
4. Magpie's FP16 NVOF guidance policy is reusable as an architectural pattern,
   not as a validated mapping for P010 video. Our design should define P010
   range, transfer, frame synchronization and driver format capability first.

The corresponding local probe and decision are in
`HDR_FP16_FEATURE18_AUDIT.md`.
