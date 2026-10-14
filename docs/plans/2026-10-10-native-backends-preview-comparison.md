# Native backends, multiscale preview and export comparison

Source: user task attachment abd56f8f-675c-4e06-85fd-0dd44d778e05 plus explicit AMD FidelityFX Optical Flow request, 2026-10-10.

## Scope and boundaries

Preserve Lunarc branding and Swiss UI, existing independent multiscale observations and CG fusion, SDR10/HDR10 colour mathematics, remux and all existing working-tree changes. No Qoder. No serial Magpie pass substitution, fake inference or silent fallback. Commit and push related changes only after acceptance; exclude unrelated local files and generated evidence.

## Order

1. Reproduce preview routing: today render_preview reads BGR8 and invokes one Live, ignoring reconstruction_mode. Seek reset currently only happens at frame zero. Reuse Chain/MultiScaleObservationChain and HdrChain/HdrMultiScaleChain with an explicit display consumer instead of NVENC. Preserve native GPU input, independent sessions and guidance histories. Display conversion/readback belongs only to preview and must be counted. Use one persistent execution thread for CUDA ownership. Reset on nonsequential requests, source/settings changes and scene cuts. Report actual sessions, dimensions, inference/fusion calls, processing and presentation rates, and dropped/waiting state.
2. Official NGX DLAA: inspect SDK enum and create/evaluate/release ABI, implement equal input/output extent on shared D3D12 resource infrastructure, initially SDR only. Hardware test fine lines/text/animation/movement and resets; verify actual NGX calls and reject unsupported HDR.
3. Actual VideoSuperRes: inspect official SDK/model/runtime licensing and reference code licenses. Implement quality values verified against ABI, native resources and synchronisation. Hardware test four qualities, 720p to 1080p and 1080p to 4K. Refuse missing runtime/models and unverified HDR/processing combinations.
4. AMD FidelityFX OF: official SDK implementation, pinned source/license, D3D12 dispatch and actual vector units/direction/grid verification. Independent histories, device motion integration and scene cut/reset; no migration to DIS for an explicit FidelityFX request.
5. Comparison page: decode original and final files, never recompute AI. Common media-time alignment using actual PTS, side-by-side/slider/A-B, seek/play/step/speed, fit/100%/200%, one audio source, honest frame-alignment and HDR display limitations. Hook completed jobs/history/manual files without affecting export. Real packaged Electron interaction screenshots.

## Verification

Short hardware fixtures; no sustained benchmark, Performance=NOT_CHARACTERIZED until explicitly measured. Actual Single/Dual/Triple preview evidence and matching export frames, lifecycle/reset tests and visible parameter updates. DLAA Create/Evaluate/Release; VSR actual model and quality/extent evidence; FidelityFX shifted-frame direction/reference tests. Units and frontend build, real packaged UI. Final SDR8/SDR10/HDR Single/Dual/Triple short export and remux regression. Missing externally provided SDK/model/driver capability is reported precisely; continue independent work. Existing passing gates are not repeated without affected production changes.

## Research provenance

Inspect actual code in SAOG0721/Magpie v0.6.9-experimental, Likely7/Veyra-NRVideo and maanHimself/OpenDLSS-NR only after reading licenses; implementation derived from official NVIDIA and AMD interfaces. Store downloaded research and raw hardware evidence under ignored bench_out, record source revision/license in acceptance documentation. Never vendor proprietary binaries without confirmed redistribution rights.
