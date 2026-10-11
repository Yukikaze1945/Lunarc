# AMD FidelityFX Optical Flow dependency

The vendored SDK source is FidelityFX SDK `v1.1.4`, pinned at commit
`c6efa6bf7f2027b3ec94f28578bb5965eabb9e55` in
`native/vendor/fidelityfx/`. It retains AMD's MIT license in
`native/vendor/fidelityfx/LICENSE.txt`; no GPL code is included.
At this revision the Optical Flow effect's own public version macros report
`1.1.2`.

## Optical Flow API contract

The public entry points are `ffxOpticalflowContextCreate`,
`ffxOpticalflowGetSharedResourceDescriptions`,
`ffxOpticalflowContextDispatch`, and `ffxOpticalflowContextDestroy` in
`sdk/include/FidelityFX/host/ffx_opticalflow.h`. Dispatch consumes a color
resource and writes two resources:

* `opticalFlowVector`: `R16G16_SINT`, grid dimensions
  `ceil(width / 8) × ceil(height / 8)`. Each cell estimates one non-overlapping
  8×8 image block. The two signed components are displacement in input/display
  pixels; shader search samples the previous image around the current block,
  so the vector points from the current block toward its previous-frame match.
  The maximum documented tracking displacement is 512 pixels.
* `opticalFlowSCD`: 3×1 `R32_UINT`. Slots 0, 1, and 2 carry scene-change
  value bits, rolling scene-history bits, and completed workgroup count. Slot 0
  is written from a float scene-change value using float/uint bit reinterpret;
  consumers must use that contract rather than treating it as an arbitrary
  normalized scalar.

The SDK maintains frame history internally; each dispatch supplies the current
`color` resource, and the context compares it with its prior input. The vector
direction/unit note above is derived from the pinned shader's previous-image
search and vector write, since the API guide does not name a coordinate
convention. The SDK has no user-facing “quality” setting. It uses a seven-level pyramid,
3×3 vector median filtering, a 24×24 search window (8-pixel radius around an
8×8 block), and SCD from nine image regions. Input transfer-function and
min/max luminance fields are part of dispatch for HDR-aware luma preparation.

## Build and third-party tools

Run `native/build_fidelityfx_optical_flow.bat`. It builds only the static
`ffx_opticalflow_x64.lib` effect and `ffx_backend_dx12_x64.lib` backend with
shader permutations from FidelityFX_SC. Artifacts/logs go under
`bench_out/amd_flow_20261010/ffx_sdk` and `cmake_build`.

The SDK source is vendored; build-time binary tools are kept outside the vendor
tree. The captured official compiler distribution is in
`bench_out/amd_flow_20261010/compiler`. Set `FFX_SC_EXECUTABLE` to another
official compiler location. This tool is a build dependency, not bundled.
The only SDK source adaptation adds `FFX_DISABLE_PIX_CAPTURES` around PIX
instrumentation and its CMake dependency, including unused-parameter annotations
for the upstream warnings-as-errors build. The application enables this flag.
The resulting bridge has no WinPixEventRuntime import and needs no PIX DLL.
No optical-flow shader or algorithm is modified.

The bridge builds and installs with `native/build_fidelityfx_flow_bridge.bat`.
Each processing layer owns a separate SDK context/history. Official startup
suppression returns zero vectors for the first six frames after creation/reset;
the scene-history suppression spans four frames. These states are reported.
The bridge expands the native 8×8 block vectors to full-resolution float32x2
device memory; it does not claim subpixel accuracy. Input is an independent
8-bit guidance image; HDR/SDR10 picture data stays in its existing FP16 chain.
SCD reads back 12 scalar bytes per completed dispatch. Image pixels are not
read back by the flow provider. Preview display and diagnostic test downloads
are separate and must not be counted as zero total host traffic.
