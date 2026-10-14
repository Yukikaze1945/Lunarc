# GPU transfer phase 2: implementation contract

Status: the project-owned GPU-flow backend now evaluates NGX Feature 18 with
D3D12 resources. NVOF flow stays on the GPU, crosses into a shared D3D12
buffer through CUDA external-memory interop, and becomes an R16G16_FLOAT
motion texture. Select it explicitly with `dlss_backend="gpu_flow_d3d12"` in
job settings; the default host-pointer backend remains available. *Both halves of
that sentence were true when written and neither is now: `gpu_flow_d3d12` is the
default a frame that names no host gets, and what used to remain available was the
explicit `host_pointer` name, now gone with the DLL that served it — see
"A service frame that names no host now names this one" for the caller map that
closed it, and "Retiring the explicit legacy host: one DLSS host ships and the old
DLL leaves the tree" for the retirement itself.* The GPU host
creates one zero depth texture at session start and reuses it for every frame;
real depth guidance and its provider have been removed. Production export still
uploads color from the CPU, reads DLSS output back for CPU residual processing,
and uses FFmpeg decode/encode pipes — of the export backends a job may name, that
describes the default (`export_backend="auto"`/`"legacy"`); the opt-in
`export_backend="gpu_transfer"` chain keeps the frame in VRAM instead, and it
refuses rather than falling back when it cannot. A separate native smoke now exercises
CUDA device-color input and a borrowed D3D12 output texture without changing
the production selection. The same-process chain is now reachable from a real
export through `export_backend="gpu_transfer"`, which is opt-in and refuses
rather than silently degrading; the legacy FFmpeg pipe remains the default. See
`## Production opt-in: export_backend="gpu_transfer"`.

## Baseline at `5783c3e`

The 15-frame 4K, 50% working-scale smoke run established functional correctness,
reduced NVOF input upload, and matching demuxed AV1 bitstreams. It did not
characterize performance. Its steady-state byte counts were:

| Boundary | Bytes/frame | Current owner |
| --- | ---: | --- |
| NVDEC hardware frame to CPU NV12 | 12,441,600 | FFmpeg `hwdownload` and stdout |
| CPU BGR24 to NVOF | 6,220,800 | `nvof_bridge_execute_bgr_sequential` |
| NVOF float2 to CPU | 16,588,800 | `execute_after_inputs` D2H |
| CPU BGR24 to FFmpeg encoder process | 24,883,200 | `export_dlss_stream` stdin |

The encoder pipe count is host bytes handed to the subprocess, not a measurement
of the later FFmpeg-to-NVENC device upload. Likewise, pipe traffic and PCIe
traffic must not be added together as if they were distinct GPU transfers.

`dlssnr_host.dll` exposes only `dlssnr_process` with host pointers for color,
motion, depth, and output (see `ABI_CAPABILITIES.md`). The repository contains
the DLL, not its C++ source. This does not mean the repository lacks a GPU
DLSSNR implementation: `Magpie/src/Magpie.Core/DLSSNRFilter.cpp` and `.h` form
a distinct D3D11/D3D12 backend. It passes D3D12 color, motion, depth, and output
resources to NGX and uses shared fences with its D3D11 producer. Its guidance
interop is in `FrameGuidanceD3D12Interop.cpp`; it is not an export of
`dlssnr_host.dll` and cannot be called through the current Python ctypes ABI.
Magpie's `EnableDLSSNR` build option defaults to false and requires `DLSSSdkDir`
for NGX headers and its link library. It is reference code, not the standalone
runtime or a dependency to modify. The standalone implementation must establish
which SDK installation and signed-snippet runtime are available before building.
On this workstation, an NGX SDK header and link library were found at
`F:/dlss/DLSS-310.5.3/include/nvsdk_ngx.h` and
`F:/dlss/DLSS-310.5.3/lib/Windows_x86_64/x64/nvsdk_ngx_s.lib`.
The independent backend now initializes and evaluates Feature 18 on this
workstation; its build and smoke tests are hardware-specific.
`nvof_bridge_get_output_deviceptr` is diagnostic only. Passing that pointer to
the current host call has previously faulted; it must never be used as an
implicit GPU ABI.

## Decision: one process and one explicit GPU resource contract

Create a source-buildable host API before switching the production path. Keep
the existing host-pointer path available as a fallback until a full GPU export
passes correctness checks. The resource API must have a new name and version;
do not reinterpret `dlssnr_process` arguments based on configuration.

The new API should specify, at minimum:

- Resource type and handle: CUDA device allocation or D3D12 resource. State
  precisely which API owns each allocation and whether CUDA/D3D12 interop is
  supported. A raw pointer from a different CUDA context is insufficient.
- Device identity and context: validate adapter LUID/device index and reject
  cross-adapter resources. State whether calls require a particular thread.
- Per-plane format, dimensions, row pitch, color space, and motion-vector
  coordinate convention for color, flow, depth, and output.
- Lifetime: who retains input/output resources, when the caller may reuse or
  free them, and how resize/reset invalidates them.
- Synchronization: an explicit producer completion signal and consumer
  completion signal (fence/stream/event). The flow pointer must not be read
  before NVOF postprocessing finishes.
- Error behavior: version/capability query, unsupported-resource response,
  device loss, and deterministic fallback before processing the first frame.

The interface can use CUDA only if the DLSS host can consume CUDA resources
correctly. If it operates on D3D12 resources, implement and verify CUDA/D3D12
interop or produce flow directly in a compatible resource. Do not claim zero
copy solely because both sides use the same physical GPU.

## Work packages and order

1. **DLSS GPU backend API** (prerequisite). Implement a new source-buildable
   backend in this project. Use Magpie's `DLSSNRFilter` and
   `FrameGuidanceD3D12Interop` to understand the D3D11/D3D12 resource and fence
   contract, without making Magpie part of the standalone build. Do not assume it has the same
   parameters or pixels as the existing host: compare the two paths explicitly.
   Add a versioned resource process entry point plus capability query. Prove
   one frame with host-input and GPU-input parity before wiring the exporter.
   The first resource-contract library is now in `native/dlss_gpu_resources.*`:
   it validates D3D12 device ownership, 2D extents, RGBA8 color/output,
   R16G16_FLOAT motion, R32_FLOAT depth, output UAV capability, and producer
   fence shape. A native D3D12 test creates real textures and checks acceptance
   plus format, size, fence, and cross-device rejection. This validation library
   does not initialize or evaluate DLSS and is not advertised as an available
   inference backend.
   A separate `native/dlss_gpu_init_probe.cpp` checks the local signed runtime
   without changing its code or the existing host DLL. On this workstation,
   direct `NVSDK_NGX_D3D12_Init_Ext` returned `0xBAD00002`. An independently
   written PE import adapter scoped to the loaded signed DLL then allowed
   `Init_Ext` to return success (`0x00000001`, no exception). The adapter is
   restored before unloading the DLL. The same probe also initializes NGX Core
   using this project's own ID, allocates a parameter map, creates a 64x64
   Feature 18 handle, and submits and waits for its initialization command list.
   All four calls succeeded locally. `native/dlss_gpu_host.*` now provides a
   versioned opt-in C ABI and persists the D3D12/NGX session across frames.
   A two-frame 1920x1080 synthetic test compared its RGBA output byte-for-byte
   with the existing host. The 12-frame 4K export at 50% working scale also
   produced identical demuxed AV1 bitstreams. The standalone host source was
   not found in this checkout, `F:/DLSS5Tool`, or the local SDK tree; this GPU
   host was written independently.
2. **NVOF flow D2H**. The additive
   `nvof_bridge_execute_bgr_sequential_device` leaves postprocessed flow on
   device. `native/nvof_d3d12_interop.*` imports a shared D3D12 buffer and
   fence into the same CUDA context, checks adapter identity, copies the flow
   device-to-device, and records the float32-to-float16 motion conversion.
   A 64x64 native test checks flow-buffer bitwise parity and motion-texture
   values. The two-frame DLSS test checks RGBA bytewise parity. A 12-frame 4K
   export through `native/gpu_flow_export_smoke.py` checks packet count,
   demuxed AV1 stream identity, actual backend selection, and zero NVOF D2H
   copies/bytes on every device-path frame. The old host execute path remains.
3. **NVDEC `hwdownload`**. `native/nvdec_bridge.*` is now a source-buildable
   same-process FFmpeg/CUDA decoder ABI. It returns borrowed NV12 Y/UV device
   addresses, their row pitches, CUDA context, PTS/time base, and source color
   metadata. ABI 2 added GPU-only NV12 scaling through `scale_cuda`, which ABI 3
   still drives unchanged.
   It refuses CPU decoder fallback and non-NV12 output. Its 12-frame 4K smoke
   checks both full-size and 1920x1080 device surfaces, stable context, PTS
   parity with `ffprobe`, and cancellation. A separate one-frame test-only
   readback confirms that the scaled NV12 bytes exactly match FFmpeg's existing
   `scale_cuda` path. ABI 3 adds `NvdecTransferCounters` and a counted
   diagnostic readback, so the decode boundary is a measured figure instead of
   an absence: the smoke reads 12 device surfaces with zero host copies, counts
   exactly the one readback it asks for and finds it byte-identical to FFmpeg's
   own `hwdownload`, and sees the scaler's 37,324,800 device-written bytes in
   the scaled pass. Production export never calls the readback. Build with
   `FFMPEG_SDK_DIR=<FFmpeg development tree>`; on this workstation the headers,
   import libraries, and runtime DLLs are under `F:/WhisperJAV/Library`.
   `native/nv12_rgba_bridge.*` then converts a borrowed CUDA NV12 surface to
   persistent RGBA8 device memory in the decoder's CUDA context. Its BT.601
   limited-range integer result matches OpenCV's `COLOR_YUV2RGBA_NV12` byte
   for byte on a 1920x1080 frame; the test alone reads the GPU result back.
   The conversion uses a context synchronization before returning its borrowed
   output pointer. The additive `nvof_bridge_execute_rgba_device_sequential`
   copies those RGBA device rows into the existing ABGR8 NVOF surface through
   one pitched cross-context CUDA copy per steady-state frame. A two-frame
   fixture compares its float2 flow byte-for-byte with the BGR host-input
   NVOF path using identical scaled NV12 pixels; the device-input path reports
   zero NVOF H2D and D2H copies. `dlss_gpu_host_process_device_color` then
   copies the borrowed RGBA CUDA allocation into a D3D12 shared color buffer,
   with adapter LUID validation and a CUDA-to-D3D12 fence. Two consecutive
   inference calls match the host-color entry point byte-for-byte, and the
   Python wrapper's device-color call matches too. Native counters show zero
   host color staging, one 8,294,400-byte GPU color copy per call, and an
   unchanged 8,294,400-byte DLSS output readback per call at 1920x1080 when
   the host-output entry point is used. The additive
   `dlss_gpu_host_process_device_color_resource` entry point instead returns
   a borrowed RGBA8 D3D12 output texture on the host's device. It is in
   COPY_SOURCE state, and the returned fence value has completed before the
   call returns. The consumer must finish with the texture before the next
   process call or close. Two consecutive calls had zero output readback at
   process return. An explicit readback after each call matched both existing
   output paths byte for byte and counted 8,294,400 bytes per check; the
   Python resource wrapper passed the same check. This establishes the GPU
   output handoff, not a GPU residual or encoder path.
   The earlier `scale_cuda` path is only exercised by native smoke, not
   production export. It does not reproduce the CPU residual downsample: at
   1920x1080 on the fixture,
   `scale_cuda` color differs from the production Lanczos-2 working color in
   386,514 channel values (maximum absolute difference 223). This is an input
   contract mismatch, not evidence that the reconstruction shader is wrong.
   `native/gpu_lanczos_bridge.*` now performs the exact 50% working-color
   preparation from full-size CUDA RGBA8, vertical then horizontal with the
   CPU plan's float32 weights. It returns a borrowed CUDA RGBA8 allocation.
   Two consecutive 4K fixture frames matched the native CPU Lanczos-2
   preparation byte for byte. The second frame then fed NVOF and DLSS without
   a color H2D or flow D2H copy.
   `native/gpu_residual_bridge.*` now implements a source-buildable 2x
   Catmull-Rom GPU residual stage. It accepts borrowed CUDA RGBA8 full and
   working frames plus the D3D12 DLSS output texture, checks device ownership,
   waits for CUDA/DLSS fences, and leaves a full-size RGBA8 D3D12 texture on
   the GPU. Its independent shader matches the native CPU residual byte for
   byte on the 4K fixture with strengths 1.0 and 0.5 when both receive the
   same full, working, and DLSS pixels. Each call makes one GPU-to-GPU copy of
   the 4K full frame and one of the 1080p working frame; process itself has
   zero output readback. The smoke performs a separate, counted 4K readback
   solely to compare output bytes. With GPU Lanczos-2 as its working input,
   the full isolated NVDEC -> NVOF -> DLSS -> residual chain matched the CPU
   residual output byte for byte on the second 4K fixture frame.
   These bridges are not selected by production export yet. The current
   pipeline still uses CPU BGR for working-color preparation, NVOF input and
   residual reconstruction. A 10-20 frame full export gate, PTS and mux/audio
   equivalence, and an encoder GPU-surface path remain before the old decode
   download and encode pipe can be removed. Any required residual readback
   must be counted instead of described as zero transfer.
4. **FFmpeg stdin to NVENC**. Move residual/output composition to GPU first,
   then feed an in-process encoder with a GPU surface. Keep muxing, audio
   mapping, container/codec settings, timestamp behavior, and cancellation
   equivalent to `build_transcode_command`. Removing stdin while still handing
   the encoder a CPU frame does not remove the upload.
   An isolated proof now exists in `native/nvenc_bridge.*` and
   `native/nvenc_gpu_smoke.py`. The bridge copies the final D3D12 RGBA texture
   into a CUDA-importable linear buffer on the GPU, converts RGBA to NV12 in a
   CUDA kernel, and submits CUDA frames to in-process `av1_nvenc` through
   libavcodec. The 15-frame 4K smoke traverses NVDEC, working-color conversion,
   NVOF, DLSSNR, GPU residual, and this encoder. It produced a 3840x2160,
   60 fps AV1 file with 15 packets and source-aligned PTS (maximum error
   0.00067 s from Matroska timebase rounding). Bridge host-pixel traffic and
   residual output readback were zero; a separate 640x384 synthetic conversion
   check read back 368,640 bytes solely for diagnostics. The texture-to-linear
   copy is 33,177,600 GPU bytes per 4K frame, so this is not zero-copy.
   Production still used its existing FFmpeg stdin path at the time of the
   isolated proof. The color mismatch it left open resolved toward the
   conversion that actually runs: the chain decodes NV12 with the BT.601
   limited-range matrix and encodes with `NVENC_COLOR_MATRIX_BT601`, so the
   output file declares `smpte170m` primaries/transfer/matrix instead of the
   BT.709 the smoke had stamped onto BT.601 pixels. The chain is now selected
   by production export under `export_backend="gpu_transfer"`; audio mapping,
   cancellation, production encoder options and full-export output comparison are
   gated there and recorded in the section below.

The order of packages 3 and 4 may change after package 1, but both depend on
GPU-resident color/output processing. Do not introduce a same-process decoder
or encoder as a performance claim while frames still cross the CPU for DLSS or
residual processing.

## Short correctness gates for each switch

- Compile Python/native code and run `pytest -q tests/unit`.
- Run one 10–20 frame 4K export at 50% working scale, NVOF 4M, with unchanged
  DLSS and encoder settings. Confirm output frame count, dimensions, codec,
  timestamps, and successful demux.
- Compare the new path against the host-pointer path on the same fixture.
  Prefer decoded frame comparison for format changes; require demuxed bitstream
  identity only when the encoder configuration and input bytes are unchanged.
- Report actual selected backend, fallback reason, and per-boundary copy bytes
  and counts. A claimed removal requires a measured zero at that boundary.
- Keep performance as `NOT_CHARACTERIZED` after a single smoke run. Do not add
  long benchmark or desktop-app isolation as a correctness gate.

## Production opt-in: `export_backend="gpu_transfer"`

`gpu_transfer_export.py` drives the verified chain inside `pipeline.py`, and
`EXPORT_BACKENDS = ("auto", "legacy", "gpu_transfer")` names the switch. It is
opt-in: `auto` and `legacy` both record
`export_backend_actual="legacy_pipe"` with
`export_backend_fallback_reason="gpu_transfer_is_opt_in: set export_backend=gpu_transfer to request it"`,
so the default path is unchanged and says so. `dlss_service` rejects an unknown
name and rejects `gpu_transfer` unless `dlss_backend="gpu_flow_d3d12"` is
already selected, because the chain needs the flow provider's device RGBA entry
point. A refusal raises instead of dispatching the pipe: the caller asked for a
GPU-resident flow and a silent fallback would report a transfer profile the job
did not get. Refusal reasons are accumulated whole in
`export_backend_refusal_reasons`, and the residual diagnostics survive the
failure because `job["residual_backend_diagnostics"]` is written in the `finally`.

The gate is a negotiation, not a capability hope. It accepts limited-range
8-bit 4:2:0 SDR with a BT.601-family or BT.709 matrix, selecting the same
matrix for device NV12→RGBA conversion and NVENC output. Unspecified matrix
retains BT.601 behavior. It refuses other matrices, explicit full range, HDR,
that needs a dst resize, that asks for a codec outside
`av1_nvenc/hevc_nvenc/h264_nvenc`, or an encoder option the real encoder
rejects. Options become same-named AVOptions through the ABI-2 `extra_options`
blob and are preflighted with `av_opt_find` against the encoder that will run,
so `libx265` answers `has no option spatial-aq` here rather than after 4K of
work. The legacy `uhq -> hq` rule for `h264_nvenc` is honored rather than
silently dropped. Container-level keys (muxer, chapters, metadata, attachments)
never enter the encoder gate. Audio, subtitle, metadata and chapter mapping
follow `build_transcode_command`, and when nothing encoder-side needs changing
the intermediate `.mkv` is remuxed with `-c:v copy` so the container rules stay
FFmpeg's. Both backends end in the same `verify_export_output`, so a GPU export
answers for frame count and truncation exactly as the pipe does.

Every transfer figure below is a bridge counter read after the run, per
boundary, or `null` where no bridge counts it. None is a constant.

| Boundary, 14 frames of 4K | Measured | Source |
| --- | ---: | --- |
| NVOF device RGBA input | 15 copies / 124,416,000 B | `nvof_device_input_copy_count_total`, `..._bytes_total` |
| DLSS host color staging | 0 B | `dlss_gpu_host` diagnostics |
| DLSS device color copy | 116,121,600 B | same |
| DLSS output readback | 0 B | same |
| Residual full-size GPU copies | 464,486,400 B | `gpu_residual_bridge` diagnostics |
| Residual working-size GPU copies | 116,121,600 B | same |
| Residual readback | 0 B | same |
| Encoder staging texture | 464,486,400 B | `nvenc_bridge` diagnostics |
| Encoder RGBA→NV12 conversion | 174,182,400 B | same |
| Encoder host pixels | 0 B | same |
| Decode host copy | 0 B over 14 frames decoded | `nvdec_bridge` ABI 3 counters |
| NV12→RGBA and working color device writes | `null` in this run | no bridge counted these at the time; the counters added below cover them |

The 15 copies across 14 frames are one steady-state copy per frame plus one on
the reset the first frame triggers. The residual and encoder staging figures
agree with the 33,177,600 B/frame 4K copies the native smoke already reported,
and the working figures with 8,294,400 B/frame at 1920x1080: this path moves
frames around the GPU and never through the process for DLSS, residual or
encode. The decode row is the decoder bridge's own count:
`decode_frames_decoded_measured` and `decode_device_frame_count_measured` are
both 14, `decode_host_copy_count_measured` and `decode_host_copy_bytes_measured`
are 0, and `decode_scale_device_output_bytes_measured` is 0 because production
opens the unscaled decoder. The same counters report
`decode_bitstream_submitted_bytes_measured` = 1,938,968 for those 14 frames,
which is compressed packet traffic handed to the decoder and not frame traffic;
it may be read next to, never added to, the frame figures. A zero in the host
fields is a counter's value with the frames it counted over beside it, not a
placeholder: the readback entry point that can raise those fields is the only
one that does, and this module does not name it.

Conversion and working-color counters. `nv12_rgba_bridge` ABI 3 adds
`Nv12RgbaTransferCounters` and `gpu_lanczos_bridge` ABI 2 adds
`GpuLanczosTransferCounters`; `gpu_transfer_export.Chain.counters` reads both and
`transfer_report` publishes them. Every field is written only after the call that
produced it returned 0, so a zero is a measurement of a step that did not run,
and every figure is a payload the code determines: the RGBA8 bytes the NV12
kernel wrote, the full-size device-to-device copy the working-color bridge makes
into its shared source buffer, the vertical pass's float4 scratch writes and the
RGBA8 working-frame writes. None is an allocation size, and reads, cache-line
traffic and PCIe traffic are not observable from either bridge, so they are not
reported. A 5-frame 4K export at 50% working scale through
`pipeline.export_dlss_stream` reported `nv12_to_rgba_convert_count_measured` 5
with `nv12_to_rgba_device_bytes_measured` 165,888,000 (5 x 33,177,600),
`working_color_process_count_measured` 5, `working_color_device_bytes_measured`
41,472,000 (5 x 8,294,400),
`working_color_source_device_copy_bytes_measured` 165,888,000 and
`working_color_scratch_device_write_bytes_measured` 331,776,000 (5 x
66,355,200), next to the decode bridge's 5 decoded frames with zero host copies
and the encoder's 5 packets. The two bridge smokes check the counters against
their own arithmetic: `nv12_rgba_bridge_smoke.py` asserts a fresh converter
counts zero, one convert counts 1 / 8,294,400 bytes and a refused convert counts
nothing while the BT.601 and BT.709 readbacks stay byte-identical as before, and
`gpu_lanczos_smoke.py` asserts 2 processes / 66,355,200 copied bytes /
132,710,400 scratch bytes / 16,588,800 written bytes over its two 4K frames with
the byte-identical parity unchanged.

Gates run. A 14-frame 4K export at 50% working scale with NVOF 4M through
`pipeline.export_dlss_stream` produced 14 packets at 3840x2160 `av1` yuv420p
tv, PTS first 0.021 s / last 0.455 s with deltas of 0.033 and 0.034 s, and the
source AAC stream copied; `fps` reached the encoder as the rational `30000/1001`
rather than the rounded float. The legacy export of the same fixture reports the
same packet count, dimensions and timestamps, and no color metadata at all, so
the 21 ms start offset is muxer behavior shared by both paths and `smpte170m` is
the GPU file's own declaration. Picture check against that file: per-frame PSNR
40.2-42.2 dB, mean channel values `169.22 120.60 169.48` (legacy) against
`170.47 120.60 171.01` (GPU), i.e. identical green and ~1.2 on red/blue, which
is encoder loss at this cq and not a matrix error. That PSNR is an upper bound on
the export-path difference rather than an isolation of it: the two gate passes
build different `Live` objects, so the DLSS backend is a second variable, and the
channel means - not the PSNR - are the color evidence, because a wrong YUV matrix
moves every channel by tens of code values. A cancellation after 4 frames
raised `JobCancelled`, left no deliverable and no
`dlss_gpu_transfer_*` intermediate. `tests/unit/gpu_transfer_export_gate.py`
reproduces all of it on hardware - a fourth argument names a legacy output path
and adds the comparability pass - and the 54 tests in
`tests/unit/test_gpu_transfer_export_backend.py` cover the negotiation, the
mux clause order, the refusal-not-fallback rule, the decode and
conversion/working-color counter contracts with their header mirrors, and the
shared verifier without a GPU. Performance is `NOT_CHARACTERIZED`.

BT.709 extension gate: `nv12_rgba_bridge` ABI 2 keeps the old BT.601 open
entry point and adds `nv12_rgba_open_matrix`. The 1920×1080 one-frame device
readback remains byte-identical to OpenCV for BT.601 and differs by at most
2 code values from an independent floating-point BT.709 limited-range
reference. A separate 10-frame tagged BT.709 4K H.264 source completed the
opt-in production export with NVOF 4M; the output has 10 AV1 packets at
3840×2160 and ffprobe reports `bt709` matrix, primaries and transfer.
Cancellation after four frames left no deliverable or temporary intermediate.
This short gate does not characterize performance or output identity.

100% working-scale gate. `tests/unit/gpu_transfer_export_fullscale_gate.py`
runs the identity chain the 50% gate cannot reach: 8 frames of the same 4K
fixture (h264, `yuv420p` tv, `smpte170m`, 30000/1001) at
`working_scale=1.0` with NVOF 4M and `av1_nvenc` through
`pipeline.export_dlss_stream`, then holds the run to what the 1.0 contract
claims. The two exact-2x bridges answer for nothing — every
`working_color_*_measured` and `residual_*_measured` field is `null`, because
`gpu_lanczos_bridge` and `gpu_residual_bridge` were never opened rather than
because a counter failed — while the stages that did run keep their own
figures: 8 frames decoded over 0 host copies (1,444,872 B of compressed
bitstream submitted, which is packet traffic and not frame traffic),
`nv12_to_rgba_device_bytes_measured` 265,420,800 (8 × 33,177,600), the same
265,420,800 for the DLSS host's device colour copy and for the encoder's
staging copy, 99,532,800 B of encoder RGBA→NV12 output, and 0 for
`dlss_host_color_stage_bytes_measured`, `dlss_output_readback_bytes_measured`
and `encoder_host_pixel_bytes_measured`. NVOF read 9 full-size planes,
298,598,400 B: one per frame plus the reset the first frame asks for, which is
the residue the engine's own cumulative total leaves when the other seven rows
each report exactly one 33,177,600 B plane (the first row reports no delta
because the engine has nothing to say before its first call, and the gate
requires that absence rather than a guessed value). Every frame's ledger row
says `export_backend_actual=gpu_transfer`, `transfer_mode_actual=gpu_transfer`,
`dlss_backend_actual=gpu_flow_d3d12`, `decode_actual=cuda_nvdec_bridge` and
`residual_backend=none`, times `nv12_to_rgba`, `nvof_flow`, `dlss_inference`
and `nvenc_submit` and no column for either skipped stage, with a `flow_pts`
that steps 0, 1001 … 7007 monotonically. The file is 8 packets at 3840×2160
`av1` `yuv420p` tv with `smpte170m` matrix, primaries and transfer, 8 distinct
presentation timestamps from 0.021 s to 0.255 s, every gap within 0.63 ms of
one frame at 30000/1001, and the source AAC copied; no
`dlss_gpu_transfer_*` intermediate was left behind and the DLSS session the
chain borrowed was closed by `Chain.close`. The gate asserts container and
counter facts only — no legacy picture comparison at 1.0 — and performance
stays `NOT_CHARACTERIZED`.

Fixing that run surfaced one real defect, in the collector rather than in the
pixels: `export_gpu_transfer` took its caller's ledger with
`transfer_collector or TransferProfiler()`, and because `TransferProfiler`
defines `__len__` an empty collector the caller had just handed over was
falsy, so the export silently recorded into a private one — which is also why
`native/gpu_flow_export_smoke.py` had nothing to read out of the collector it
passed. It is compared against `None` now, and
`test_the_export_runs_and_reports_the_shape_the_gate_classified` guards the
form of that line.

Contracts still not covered: none of the accepted scales is a benchmark claim,
and the chain now serves every whole percentage from 25% to 100%. Which of the
three shapes a percentage runs on is decided by the size it resolves to, not by
which ratios have been measured: at 100% the full RGBA plane goes directly to
DLSS and NVENC with no working downsample and no residual stage, at 50% on a
source whose axes it halves exactly the GPU color and residual stages are the
exact-2x pair on constant taps, and every other percentage is those same two
stages opened through their plan-driven exports with the CPU plan's own tap
tables at the resolved size. What still refuses is what a build or a source
cannot do rather than what nobody has run: a ratio that is not a whole
percentage, a percentage whose 8-aligned size leaves no axis below the source, a
plan-driven percentage on a build whose bridges were compiled without the
plan-driven exports, and a plan wider than the loaded kernel walks. The last of
those is the reason `GPU_LANCZOS_MAX_TAPS` is 24: a Lanczos-2 window is
`floor(4·src/dst) + 1` taps and `resolve_working_dims` never leaves a served
percentage below `dst = 0.2·src` on an even axis, so 21 is the deepest real
window (the banker's-rounding accident at an 80-pixel axis) and 24 carries it
with three columns spare. A 3842×2162 input at a quarter is 17 taps, which the
widened kernel serves and a 16-wide one would have refused - and the refusal a
narrower build gets is read out of that build's own
`gpu_lanczos_max_taps`, not out of this document. A build that exports no such
getter is the pre-widening one (it added the plan-driven open under the same ABI
2 and got the getter with the widening), so it is measured at the 16 its own
shader was compiled with - `LEGACY_LANCZOS_MAX_TAPS`, whose figure is the
`#define TAPS 16u` in the DLL still shipped in `dist/` - and never at the 24
this checkout's mirror describes. Only the three NVENC codecs
and single-pass are accepted, so `multi_pass` refuses; cancel latency is one
decoded frame and the watchdog cannot cover a decode that never returns because
the decoder runs on the producing thread; and a custom `color_preset` remap
refuses rather than applies. The DLSS page now exposes `export_backend`, with
`auto` retaining the legacy default and `gpu_transfer` requiring an explicit
selection. The Deliver page sets `dlss_backend=gpu_flow_d3d12` on jobs that
select GPU Transfer; other jobs retain their existing backend choice. The UI
states the NVOF, 25%-100% whole-percentage, 8-bit SDR, and NVENC conditions,
steps its slider over the served range one percent at a time, and prints the
working size through the same clause-for-clause mirror of
`pipeline.resolve_working_dims` the gate resolves with;
`server.py` passes the job settings through unchanged.

## A third chain shape: plan-driven resampling for the percentages a constant table cannot describe

The scales first served were served by stages that carry their filter in the
shader: `gpu_lanczos_bridge` walks eight taps of a constant table and
`gpu_residual_bridge` walks four, both of which describe exactly a 2x change,
and negotiation therefore refused anything else. `working_scale=0.75` is the
first scale served by a stage that reads its taps from the caller instead, and
0.67 and 0.63 joined it the same way by the same two exports; 0.25 followed on
the two byte-identical 4K comparisons below, not on the fit of its window. Every
other whole percentage was refused until the range replaced the list: the
served set is now
`gpu_transfer_export.WORKING_PERCENT_MIN..WORKING_PERCENT_MAX`, every percentage
in it resolves through
`pipeline.resolve_working_dims`, and every tap table comes from
`pipeline._build_*_axis_plan` for that resolved size.

What makes that safe rather than a second resampler is that the plan is the CPU
plan's own. `gpu_transfer_export.working_color_plan` and `.residual_plan`
serialize `pipeline._build_lanczos2_axis_plan` and
`pipeline._build_catmull_rom_axis_plan` into `GpuLanczosTapTables` /
`GpuResidualTapTable`, in the vertical-then-horizontal order both kernels run,
and `test_the_half_plan_the_serializer_builds_is_the_half_table_in_the_kernel`
parses the `asfloat(0x…)` constants out of the 2x shader and requires the
serialized 8→4 table to equal them bit for bit. So the exact-2x path and the
plan-driven path are the same filter, and the 0.5 fixtures stay the evidence
for it.

Two contracts the widened scope had to keep:

* The new exports are `gpu_lanczos_open_scaled` and
  `gpu_residual_open_scaled`, added under the ABI numbers already published
  (2 and 1). An existing caller is untouched; a build without the exports still
  loads and still serves 0.5 and 1.0, and refuses each plan-driven scale with
  that scale and the bridges named, because `load_bridges` records
  `lanczos_scaled` / `residual_scaled` as capabilities and `negotiate` gates the
  scale on them.
* The counter structs did not move. Every plan-driven figure is the same
  extents-driven formula the 0.5 run already reported, evaluated at the size the
  ratio resolves to; the tap-table upload is filter coefficients, not a frame,
  so no byte field counts it. A plan wider than the kernel's window is refused
  by `tap_tables` (`GPU_LANCZOS_MAX_TAPS` 24, `GPU_RESIDUAL_MAX_TAPS` 4) rather
  than truncated, because dropping taps would build a different filter. The
  figure the request is measured against is the loaded bridge's own
  `gpu_lanczos_max_taps`, recorded by `load_bridges` as the
  `lanczos_max_taps` capability; a build that reports nothing is the
  pre-widening one, and is measured at `LEGACY_LANCZOS_MAX_TAPS` - the 16 its
  own kernel walks - because this module's `LANCZOS_MAX_TAPS` mirrors the
  header this source is compiled against and says nothing about a binary that
  predates it. The
  width negotiation compares against comes from
  `pipeline.lanczos2_axis_taps`, which shares `_lanczos2_axis_windows` with the
  plan builder and returns the same number as the built plan's `max_taps`
  without building it — so asking the question never moves
  `filter_plan_build_count`, which is the per-job re-plan detector.

The working size is what `pipeline.resolve_working_dims(3840, 2160, 0.75)`
returns: **2880×1616**, not the 2880×1620 the ratio alone suggests (8-alignment
and the CPU plan's own rounding). The GPU chain follows that rule rather than
inventing a second answer, and negotiation refuses a caller that hands
2880×1620 with the text naming it. The two added ratios resolve to **2576×1448**
and **2416×1360** on the same rule, and their plans are built for those extents:
6 and 7 Lanczos taps respectively, where 0.75 needs 6, against the 16 the
working-colour kernel walks.

Hardware evidence, in order. Both smokes take the working scale as an argument
and check it against `gpu_transfer_export.plan_driven_scale` before opening a
bridge, so a ratio the backend does not serve fails there rather than measuring
a chain that could never be selected; the default of 0.75 is what was run first
and is still what re-running with no argument reproduces.

* `native/gpu_lanczos_scaled_smoke.py` — two decoded 4K frames, GPU working
  colour at the resolved size versus `NativeResidualBackend.prepare_working_color_rgba`
  with `assert np.array_equal`: `max_difference=0`, `differing_channels=0`, and
  the three device counters pinned to `2·3840·2160·4`, `2·3840·H·16`,
  `2·W·H·4`. Run as `gpu_lanczos_scaled_4k_to_2880x1616=BYTE_IDENTICAL` (taps=6),
  `…_2576x1448=BYTE_IDENTICAL` (taps=6), `…_2416x1360=BYTE_IDENTICAL` (taps=7)
  and `…_960x544=BYTE_IDENTICAL` (taps=16 — the widest window the kernel walks,
  and the comparison that opened 0.25 rather than the fit);
  `bench_out/_gpu_scaled_smoke_0.67.txt`,
  `bench_out/_gpu_scaled_smoke_0.63.txt`,
  `bench_out/_gpu_scaled_smoke_0.25_lanczos.txt`.
* `native/gpu_residual_scaled_smoke.py` — the whole device chain at the same
  scale, decode → convert → scaled Lanczos → NVOF device flow → DLSS host →
  scaled residual, against `cpu.reconstruct_from_rgba`:
  `residual_4k_composite_max_difference=0`, `differing_channels=0` at 0.75, 0.67,
  0.63 and 0.25 (`bench_out/_gpu_scaled_smoke_0.25_residual.txt`, at
  lanczos_taps=16 / residual_taps=4). The NVOF stage reports `(h2d, d2h) == (0, 0)`
  copies and the DLSS host reports `host_color_stage_bytes == 0`, so no frame
  crossed to the host on
  the way.
* `tests/unit/gpu_transfer_export_plan_driven_gate.py`, which takes the scale as
  its fourth argument, and one run of it per ratio. Besides classifying the
  scale as plan-driven and requiring `export_backend_actual=gpu_transfer`, the
  gate checks the size and color contracts (resolved working dims, 4K output
  extent, ffprobe's own `color_space`/`color_range` read back equal to
  `gpu_transfer_color_matrix` and `tv`), the packet/frame counts (submitted and
  written both equal the frames exported), that no image frame crossed a host
  copy (the six host-side counters at zero), the per-frame presentation-timestamp
  ledger, and cleanup (no `dlss_gpu_transfer_*` temp intermediate left behind).
  The last two assertions were added when `0.63` was first exported through this
  entry point, so the `0.75` and `0.67` logs below predate them and were not
  re-run.
  * `0.75` — a 4-frame production export at 2880×1616
    (`bench_out/_gpu075_gate.txt`): `gpu_transfer_resampling=plan_driven`,
    `gpu_transfer_working_dims=[2880,1616]`,
    `decode_host_copy_count_measured` 0 over 4 frames decoded,
    `dlss_host_color_stage_bytes_measured` 0,
    `dlss_output_readback_bytes_measured` 0,
    `residual_output_readback_bytes_measured` 0,
    `encoder_host_pixel_bytes_measured` 0. The device figures are the same
    formulas at the resolved extents: 74,465,280 B of working-color output (4 ×
    2880·1616·4), 397,148,160 B of scratch writes (4 × 3840·1616·16), 132,710,400 B
    each for the full-plane copies at three boundaries, and NVOF 5 device copies /
    93,081,600 B — one 2880×1616 plane per frame plus the reset the first frame
    asks for. The deliverable is 4 packets at 3840×2160 `av1`.
  * `0.67` — one 5-frame production export at 2576×1448
    (`bench_out/_gpu067_gate.txt`), the first run of the newly accepted ratio
    through the real entry point and not repeated. `gpu_transfer_resampling=plan_driven`,
    `gpu_transfer_working_scale=0.67`, `gpu_transfer_working_dims=[2576,1448]`,
    `export_backend_actual=gpu_transfer` with no fallback reason, and every
    host-side field above 0. The device side is the same formula evaluated at
    the new extents: 74,600,960 B of working-color output (5 × 2576·1448·4),
    444,825,600 B of scratch writes (5 × 3840·1448·16), 165,888,000 B each for
    the full-plane copies at three boundaries, and NVOF 6 device copies /
    89,521,152 B. `encoder_frames_submitted_measured` and
    `encoder_packets_written_measured` are both 5, the deliverable is 5 frames at
    3840×2160 `av1` `yuv420p` marked `tv` / `smpte170m` — ffprobe's own reading
    of the file, matching `gpu_transfer_color_matrix`.
  * `0.63` — one 5-frame production export at 2416×1360
    (`bench_out/_gpu063_gate.txt`), the first run of the last newly accepted
    ratio through `pipeline.export_dlss_stream` and not repeated.
    `gpu_transfer_resampling=plan_driven`, `gpu_transfer_working_scale=0.63`,
    `gpu_transfer_working_dims=[2416,1360]`, `export_backend_actual=gpu_transfer`
    with no fallback reason, and every host-side field at 0 over 5 frames
    decoded. The device side is the same formula at the resolved extents:
    65,715,200 B of working-color output (5 × 2416·1360·4), 417,792,000 B of
    scratch writes (5 × 3840·1360·16), 165,888,000 B each for the full-plane
    copies at three boundaries, and NVOF 6 device copies / 78,858,240 B (one
    2416×1360 plane per frame plus the reset). `encoder_frames_submitted_measured`
    and `encoder_packets_written_measured` are both 5; the five ledger rows carry
    `flow_pts` 0, 1001, 2002, 3003, 4004 — one uniform 30000/1001 step per frame,
    strictly increasing from 0; no `dlss_gpu_transfer_*` intermediate was left
    behind; and the deliverable is 5 frames at 3840×2160 `av1` `yuv420p` marked
    `tv` / `smpte170m`, matching `gpu_transfer_color_matrix`.
  * `0.25` — one 4-frame production export at 960×544
    (`bench_out/_gpu025_gate.txt`), the first run of the deepest accepted ratio
    through `pipeline.export_dlss_stream` and not repeated.
    `gpu_transfer_resampling=plan_driven`, `gpu_transfer_working_scale=0.25`,
    `gpu_transfer_working_dims=[960,544]`, plan widths
    `working_color=16` against a `LANCZOS_MAX_TAPS=16` window and `upscale=4`, so
    this ratio is accepted at exactly the width and not inside a margin. (The
    window is 24 now, and this row records the run as it was made against the
    kernel of the day; the 17-tap case that used to refuse is served.)
    `export_backend_actual=gpu_transfer` with no fallback reason, and all six
    host-side counters at 0 over 4 frames. The device side is the same formula at
    the resolved extents: 8,355,840 B of working-color output (4 × 960·544·4),
    133,693,440 B of scratch writes (4 × 3840·544·16), 132,710,400 B each for the
    full-plane copies at three boundaries, and NVOF 5 device copies / 10,444,800 B
    (one 960×544 plane per frame plus the reset).
    `encoder_frames_submitted_measured` and `encoder_packets_written_measured` are
    both 4; the four ledger rows carry `flow_pts` 0, 1001, 2002, 3003 — one
    uniform step, strictly increasing from 0; no `dlss_gpu_transfer_*`
    intermediate was left behind; and the deliverable is 4 frames at 3840×2160
    `av1` `yuv420p` marked `tv` / `smpte170m`, matching
    `gpu_transfer_color_matrix`.

No run was repeated for logs; performance stays `NOT_CHARACTERIZED`.

## `guidance_mode=0` on a machine that cannot start NVOF

The unguided mode takes its motion from the D3D12 texture `dlss_gpu_host.dll`
zeroes once at `open` and nobody writes again
(`process_device_color_zero_motion_resource`, host ABI 3), so no frame in such a
run asks an optical flow engine for anything. What this section records is that
the *loading and setup* path now agrees with the frame path, so the mode is
independent of NVOF rather than merely unused by it.

`load_bridges()` used to import `optical_flow` and call
`probe_nvof_capabilities()` unconditionally and raise on any negative answer,
which made a mode 0 job unrecoverable on a machine with no optical flow
hardware: `export_backend=gpu_transfer` refuses rather than falling back, and
the refusal named a library the run never touches. It now takes
`requires_flow`, and `gpu_transfer_export.guidance_requires_flow(job_settings)`
— the helper `pipeline.export_gpu_transfer_stream` asks it through — answers the
question from the same setting the gate reads, so the libraries and the refusal
cannot disagree about what was requested. `guidance_mode=1` keeps every
validation word for word: a missing module, `available=False`, a missing device
flow output export and a missing device RGBA input export each still refuse,
each with its own message. `guidance_mode=0` does not import the module at all,
so nothing is asked of it, and a mode it cannot read asks for the flow bridges
anyway rather than accepting a guided job it has no engine for.
`Chain` already routed on the negotiated `uses_flow`; the counters it reports
(`gpu_transfer_uses_flow`, `gpu_transfer_dlss_flow_calls`,
`gpu_transfer_dlss_zero_motion_calls`) are unchanged, and
`gpu_transfer_nvof_bridge_loaded` now records the loading fact next to them,
because zero flow calls on their own would also describe a guided run that
decoded nothing.

The native side was audited rather than assumed, since a load-time import is
exactly the dependency a Python-side skip cannot remove. Read off the PE import
tables: `native/dlss_gpu_host.dll` imports `ADVAPI32`, `KERNEL32`, `USER32`,
`d3d12` and `nvof_d3d12_interop`; `native/nvof_d3d12_interop.dll` imports
`D3DCOMPILER_47`, `KERNEL32`, `VCRUNTIME140`, the CRT shards and `d3d12` — no
NVOF and no CUDA import anywhere in the chain. Despite its name that library
carries no optical flow code: it is the CUDA↔D3D12 interop shim, and it reaches
`nvcuda.dll` through a `LoadLibraryW` on the `nvof_d3d12_create` /
`cuda_d3d12_color_create` paths only. NVOF itself is reachable in exactly one
place — `native/nvof_bridge.cpp` loads `nvofapi64.dll` at runtime — and that
library is loaded by `optical_flow._load_native_bridge()`, which a mode 0 run
never enters. Inside the host, every `nvof_d3d12_create`, `nvof_d3d12_copy_flow`
and `nvof_d3d12_record_motion` call is behind `if (!use_zero_motion)`
(`dlss_gpu_host.cpp:615`, `:639`, `:708`), and the motion resource defaults to
the host's own `h->motion` at `:638`. No native change was needed.

What this does *not* claim:

* **NVOF-free is not NVIDIA-free.** A mode 0 run still needs the D3D12 device,
  `nvngx_dlssnr.dll`, NVDEC/NVENC, and `nvcuda.dll` — the colour plane reaches
  the host as CUDA device rows from NVDEC and is imported by that same shim. The
  verified claim is that no *optical flow* component is loaded or probed.
* `nvof_d3d12_interop.dll` remains a load-time static import of
  `dlss_gpu_host.dll` and must ship beside it. It is a repository build
  artifact, not a driver component, and it pulls in no flow library; loosening
  the name from the flow work is a separate, cosmetic change.
* `dlss_service.py:32` imports the `optical_flow` module at service start. It
  creates no engine and no session unless the mode asks (`need_flow` at
  `:936`, `:484`), and the module has no import-time NVOF side effect, so this
  costs an `import` — with its `cv2` and `torchvision` weight — rather than a
  dependency on flow hardware. Left alone deliberately: it is outside the
  loading path this gate covers.
* `server.py:268` still reports `nvof_probe` in the capabilities readout. That
  is an informational answer about the machine and gates no job.
* Scope is the GPU-transfer loading and setup path. The preview path, the image
  path and the `export_backend=legacy` flow-disabled path are untouched, and
  `dlssnr_host.dll` stays a dependency for them.

Evidence, neither run repeated: the unit gate
`tests/unit/test_nvof_independence.py` (21 checks, no GPU) asserts the module is
never looked for when the mode does not need it — an import blocker that counts
attempts sees zero — that the guided set still refuses on each of the four
answers, that a refused load closes every `add_dll_directory` cookie it opened,
and that the report field reads off a real `export_gpu_transfer` call in both
modes. The hardware gate `scratch/_mode0_nvof_free_4k_gate.py` ran one 5-frame
4K `guidance_mode=0` export through `pipeline.export_dlss_stream` with
`nvof_bridge.dll` and `nvofapi64.dll` refused at `ctypes.CDLL` and
`probe_nvof_capabilities` replaced by a function that raises, so a run touching
NVOF failed loudly instead of being talked around it: 19/19 checks passed. In
the finished process `nvof_bridge.dll`, `nvofapi64.dll` and `nvofnvml64.dll` are
absent from the module list while `nvof_d3d12_interop.dll`, `nvcuda.dll` and
`dlss_gpu_host.dll` are present, and the report reads
`gpu_transfer_nvof_bridge_loaded=false`, `gpu_transfer_uses_flow=false`,
`gpu_transfer_dlss_flow_calls=0`, `gpu_transfer_dlss_zero_motion_calls=5`,
`nvof_device_input_copy_count_total=null`,
`nvof_device_input_bytes_total=null`. The transfers stay on the device over
those 5 frames: 0 B of decode host copy, 0 B of DLSS host colour staging with
41,472,000 B of device colour copies, 0 B of DLSS output readback, 0 B of
encoder host pixels against 5 frames submitted and 5 packets written. ffprobe
answers 5 frames at 3840×2160 `av1` `yuv420p` marked `tv` / `smpte170m` with
unique PTS spanning 133,000 µs against the 133,467 µs the 30000/1001 plan
implies, and no `dlss_gpu_transfer_*` intermediate was left behind. In the same
blocked process the identical job at `guidance_mode=1` is refused with the NVOF
message, so the mode 0 result is a routing decision and not a machine that lost
flow for everybody. Full unit suite: 477 passed.

Performance remains `NOT_CHARACTERIZED`: this gate counted transfers and
loading, not throughput, and five frames is a correctness sample.

## `guidance_mode=0` preview on the GPU host's zero-motion entry

The NVOF-independence gate above left the preview rendering through
`dlssnr_host.dll`, which is item 1 of the next-work list. This stage closes the
preview half of that item. The legacy export and the single-image path are not
touched here and still ran on the old host at this point; each was already excluded
from this change by the task boundary. The legacy export has since moved too — an
unguided export on `dlss_backend=gpu_flow_d3d12` reaches the GPU host on either
chain, which is its own section below — so the two sentences in this file about the
export are true of different moments, and the later one is the current behavior.

What an unguided preview now does. `DLSSService` decides which host backs a
preview frame from the settings the frame is drawn under, in
`_preview_live_kind`: with `guidance_mode=0` **and** `dlss_backend=gpu_flow_d3d12`
it opens `dlss_gpu_engine.Live` and drives it through
`process_zero_motion(rgba, reset=...)` — host RGBA in, the host's own resident
zero-motion texture, host RGBA out. Every other preview — a guided frame, or the
default host-pointer backend — still goes through `dlss_engine.Live`
(`dlssnr_host.dll`), exactly as before. **That last clause is the state at this
stage, not the current one**: `dlss_engine.py` and `dlssnr_host.dll` were removed
afterwards, `host_pointer` is a retired backend that is refused by name, and
`_preview_live_kind` now answers only `gpu_zero_motion` (mode 0) or
`gpu_host_motion` (mode 1, the host-memory motion entry). No preview is drawn by
the old host. The decision is one predicate, and both
the flow-array allocation and the entry-point routing read it, so a zero-motion
frame never allocates a host `HxWx2` flow array, never builds a flow provider,
and never instantiates NVOF.

The migration reuses the session slot the preview and the render queue already
share, because `dlssnr_host.dll` cannot survive a second `dlssnr_init` in one
process and a guided preview is expected to keep the export's host session. The
cached session now carries a `kind` ("legacy", "gpu_zero_motion", "gpu_flow")
that names the type actually constructed, so a request never drives a session
through the wrong host's entry point: an ABI-3 host session is not asked to
`process(rgba, flow, ...)` and the host-pointer session is not asked for
`process_zero_motion`. The GPU host reads its options once, at open, and exposes
no live-update entry, so a preview session is reused only while the baked option
tuple and the dimensions are unchanged; either one moving tears down and
reopens the one session — the same rebuild rule the export already enforces,
rather than the legacy host's live-`update`-in-place.

Everything around the routing was held fixed, and it is held fixed by tests
rather than by argument:

* `orig` and `flow` views do not enter the DLSS branch and open no session.
* While a job owns the GPU (`active_job_id` set) the preview serves the source
  with the collision indicator and touches neither host — this happens before
  any session lookup, so the new backend collides with the queue exactly the way
  the old one did, and the `_live_lock` timeout branch is untouched.
* `split` and `dual` still composite the source against the DLSS result and the
  two side by side, using whatever the routed host returned.
* The residual pyramid path reconstructs from the working-size result exactly as
  the export does, which is what `test_preview_export_parity.py` keeps pinning:
  the unguided pyramid now runs `process_zero_motion` at the working size and
  the unguided full-native branch runs it at source size, both against the same
  residual backends.
* Seek semantics are unchanged: `reset=(frame_idx == 0)` reaches whichever entry
  the frame used, so the host's temporal state is cleared on a seek to zero and
  carried forward otherwise.
* Settings changes flow through `update_settings` as before; for a GPU session a
  moved option rebuilds it rather than updating it in place.
* Failure reporting is still the red "[DLSS 增强失败: reason]" banner over the
  source, never a frame presented as a successful enhancement. The GPU host
  raises on a failed frame where the legacy host returned `None`; that raise is
  caught and surfaced through the same path, so a per-frame inference error is
  a banner instead of a 500 that stops the MJPEG stream.

Unsupported configurations fail rather than migrate sideways. A GPU host that
binds no zero-motion texture — an ABI-2 build of `native/dlss_gpu_host.dll` —
is refused in `_open_gpu_zero_motion_live` before any frame: the session is
closed and the reason is carried to the banner. Nothing in the unguided GPU
branch creates a `dlss_engine.Live`, so a refused configuration cannot
silently switch hosts, and the source is never served as a DLSS result.

What this does *not* claim:

* It is not NVIDIA-free, in the same sense the export gate is not: the run still
  needs the D3D12 device, `nvngx_dlssnr.dll`, and the CUDA↔D3D12 interop shim.
  The verified claim is that no optical-flow component is loaded, created, or
  asked for on this path.
* It is not a throughput result. Three frames counted entry points, not
  performance; see `Performance` below.
* The desktop GUI (`gui.py`) has its own DLSS session factory
  (`get_dlss_session`) and still drives `dlssnr_host.dll` for both its preview and
  its `--selftest`; the task scope is the service's preview, so that path was
  left alone deliberately.
* The preview UI could not yet ask for this route when this section was written.
  `dlss_backend` was sent by the Deliver page on the job, not by the settings POST
  the preview page writes, so in the shipped frontend an unguided preview still
  landed on the host-pointer path and this routing was exercised only by the
  gate below, through `POST /api/settings`
  (`{"guidance_mode": 0, "dlss_backend": "gpu_flow_d3d12"}`). Wiring the preview
  page to send it was a frontend change, not a backend one; it is done and
  verified in the section below, "The shipped preview page can now ask for the
  zero-motion route".
* A `guidance_mode=1` preview on `dlss_backend=gpu_flow_d3d12` is still served by
  the legacy host with a CPU flow plane. Moving guided preview onto the host's
  device-flow entries is separate work — the task pinned guided behaviour, and
  the routing tests assert it stayed pinned. *True of this stage, and no longer of
  the tree: the guided frame on that backend now runs on the host's **host-memory**
  motion entry — see the section "The guided preview on the GPU host's host-motion
  entry", which rewrote the pin this bullet describes. The device-flow
  entries are still not what a guided preview uses: it holds host memory, and a
  CUDA descriptor is not a CPU array.*

Evidence, neither run repeated. The unit gate
`tests/unit/test_preview_zero_motion.py` (15 checks, no GPU) replaces both
engines with fakes whose wrong-entry methods raise, and asserts: the
`guidance_mode=0` + `gpu_flow_d3d12` combination opens exactly one
`dlss_gpu_engine.Live` and never a `dlss_engine.Live`; the same settings without
`dlss_backend` and a `guidance_mode=1` frame (even with `dlss_backend` set) stay
on the legacy host with their flow plane intact *(the guided-preview section rewrote
that parenthetical: the same file now asserts the opposite for the frame that names the
GPU host, and keeps the no-`dlss_backend` half of it)*; the pyramid branch opens the
session at the resolved working size; a `numpy.zeros` spy sees no `HxWx2` array
allocated on the zero-motion branch; four consecutive frames open one session
with `reset` true only at frame zero, and a seek back to zero re-asserts it; a
geometry change and a baked-option change each close the old session and open
one; flipping the backend closes the legacy session instead of driving it
through `process_zero_motion`; an ABI-2 host is refused without constructing a
legacy session, without calling the zero-motion entry, and with the reason
visible; `orig`/`flow` views and a queue-colliding `active_job_id` open no
session at all. Full unit suite: 489 passed (477 before, plus these 15), no
failures, `test_preview_export_parity.py` among them.

The hardware gate `scratch/_zero_motion_preview_4k_gate.py` loaded
`bench_out/fixture_4k_gpu15.mp4` into `DLSSService` and rendered five real
preview frames on an RTX 5060 Ti — three at `working_scale=0.5` with the
residual pyramid, then two at `working_scale=1.0` — with `guidance_mode=0` and
`dlss_backend=gpu_flow_d3d12`, `dlss_engine.Live` replaced by a factory that
raises, and the GPU host wrapped in a counter whose flow entries also raise. 10/10
checks passed: `process_zero_motion` answered 5 of 5 frames with 0 flow
dispatches and 0 inference errors; the host opened exactly twice, once per
geometry; the returned RGBA was 1920x1080x4 for the pyramid frames and
3840x2160x4 for the full-native ones, all `uint8`, and every served JPEG decoded
to 3840x2160x3 `uint8`; `reset` read `[true, false, true, false, false]`, so the
seek to frame zero reset temporal state and the other frames did not;
`optical_flow._ENGINES` stayed empty and the NVOF execute counter did not move;
`GetModuleHandleW` reports `dlssnr_host.dll`, `nvof_bridge.dll`, `nvofapi64.dll`
and `nvofnvml64.dll` absent from the finished process while `dlss_gpu_host.dll`
is present. `dlss_gpu_host.dll` itself was already ABI 3 —
`supports_zero_motion()` answered true on this machine, and the
zero-motion-capable session was the one that ran.

Performance remains `NOT_CHARACTERIZED`: five frames at two geometries is a
correctness sample of entry points and module residency, not a throughput
measurement.

## The shipped preview page can now ask for the zero-motion route

The routing above was reachable only through a hand-written `POST /api/settings`:
the Deliver page named `dlss_backend=gpu_flow_d3d12` on the *job*, while the
settings write the preview page performs sent `export_backend` and nothing else.
In the shipped frontend an unguided preview therefore still landed on
`dlssnr_host.dll`, and the branch the previous section describes had no UI caller.
This closes item 5. No backend migration happened here — nothing moved from one
host to another, no native code changed, and `dlssnr_host.dll` is still present
and still used by every path that was using it.

One control, one derivation. `dlss_backend` is not a second choice an operator
makes: it names which native DLSS host a session opens on, and the GPU transfer
chain is the only thing that runs a device session. So
`frontend/src/services/api.ts` derives it in one function —
`dlssBackendForExport`: `gpu_flow_d3d12` for `gpu_transfer`, `host_pointer`
otherwise — and the preview page's single settings writer
(`DLSSPage.handleParamChange`, the only caller that ever changes `export_backend`)
sends that pair with the choice. The frontend types the field optional
(`dlss_backend?: DlssBackend` in `types/index.ts`) because absent and
`host_pointer` name the same host and older settings blobs say nothing.

Written in both directions on purpose. `update_settings` merges, so a patch that
only ever *added* the key would leave `gpu_flow_d3d12` in the service after the
operator moved back to Auto, and the next frame would still be drawn by the GPU
host — a hidden choice the panel no longer shows. The page therefore posts
`dlss_backend=host_pointer` on the way out, and
`test_omitting_the_host_key_would_strand_the_gpu_preview` pins the merge behaviour
that makes the explicit write necessary rather than leaving it to a comment.

Guided preview semantics are unchanged, and now stated out loud. A
`guidance_mode=1` frame on `dlss_backend=gpu_flow_d3d12` is still served by the
legacy host with a CPU flow plane, because `_preview_live_kind` routes only the
unguided case; that is asserted at the settings level by
`test_a_guided_preview_keeps_the_host_pointer_route_under_gpu_transfer` on top of
the frame-level pinning `test_preview_zero_motion.py` already had. The
export-backend panel, which used to promise the zero-motion entry to exports only,
now says the preview shares that session, says a guided preview does not, and says
switching back hands the preview to the host-pointer session. *This stage moved no
frame, and its panel copy was accurate as written then; the guided-preview section is
the one that has to correct it. That test name is now
`test_a_guided_preview_routes_to_the_gpu_host_motion_entry_under_gpu_transfer`, and the
panel says GPU Transfer takes **both** preview modes onto the GPU host — mode 1 through
the host's host-memory motion entry rather than the resident zero texture. The one claim
of this paragraph that survives is the one the panel still makes: the guided frame does
not take the zero-motion entry, it takes the other one, under its own session kind.*

The job payload keeps its shape. Because the shared settings can now carry
`dlss_backend`, `DeliverPage` derives a job's host from that job's own
`export_backend` instead of letting it inherit whatever the preview left behind:
an `auto` job that inherited `gpu_flow_d3d12` used to be refused at
`dlss_service.py:964-968` for a pairing nobody asked for on the Deliver page. That
refusal is gone — see the section below on the unguided export on the GPU host — and
what the rule now prevents is quieter than an error: an inherited key would silently
run an `auto` export on the GPU host through the legacy pipe, naming a host the
operator never picked on that page. A
non-GPU job therefore still carries no `dlss_backend` key at all, which is what
E2E 14.1 and the revert step of 14.2 assert on the wire, unchanged.

Two settings-boundary facts this work surfaced:

* `update_settings` now validates `dlss_backend` against
  `pipeline.DLSS_BACKENDS = ("host_pointer", "gpu_flow_d3d12")`, the same way it
  already validates `export_backend`. The settings POST is an operator-facing
  writer of this key now, and a value that is neither host — a typo, a
  `gpu-flow-d3d12` — used to read as "host pointer", leaving the preview on the
  old host with no message anywhere to act on.
* **Switching back from the GPU host was broken, and is fixed.** The first run of
  E2E 14.5 got the GPU preview right and then failed once the operator chose Auto
  again: the service logged `[DLSS5 Warning] Preview session update failed: 'Live'
  object has no attribute 'update'`, and the frames after it were served with the
  `[DLSS 增强失败: Feature 18 推理失败或会话未初始化]` banner instead of an
  enhancement. The cause was in `_get_or_create_live`: its legacy reuse branch was
  gated on the *requested* kind alone, so a cached GPU session at unchanged
  dimensions was handed the legacy host's live-`update` entry, that call raised,
  and the branch returned no session instead of falling through to the
  close-and-reopen below it. The gate now also requires
  `self._live_session_kind == "legacy"` — the ledger that already exists precisely
  to keep the two hosts from being driven through each other's entries — so the GPU
  session is closed and `dlss_engine.Live` is reopened. Only one direction of that
  flip had been pinned before: `test_config_flip_between_hosts_rebuilds` covered
  legacy→GPU, never GPU→legacy.

Evidence. The full unit suite is **499 passed**, no failures (489 before this
change). In it:

* `tests/unit/test_export_backend_ui_contract.py` — 19 checks, was 11. New: the
  preview page's settings writer pairs `dlss_backend` with `export_backend` and
  both values come from the one shared derivation; the panel copy no longer limits
  zero motion to export and states the guided case and the switch-back; the
  settings pair the page posts actually moves `_preview_live_kind`
  (`legacy` → `gpu_zero_motion` → `legacy`) with no GPU required; a guided frame
  under GPU Transfer stays on the legacy route; omitting the host key strands the
  GPU preview (the merge behaviour); and three unknown `dlss_backend` values are
  refused by name without being written into the settings.
* `tests/unit/test_preview_zero_motion.py` — 17 checks, was 15. New: GPU→legacy at
  unchanged dimensions closes the GPU session, reopens `dlss_engine.Live`, drives
  one motion call and returns a frame; and a session the export left in the shared
  slot is still *reused* by a legacy preview rather than reopened — the
  same-host check added to fix the first of those must not cost the handoff that
  exists because `dlssnr_host.dll` cannot survive another init/shutdown cycle in
  one process, so the test pins both the reuse (with its `update` call) and the
  refusal to reuse a session the export's GPU host left behind. The GPU fake also
  gained a raising `update`, mirroring the real host's missing entry: without the
  fix the two new tests fail and the other fifteen pass.

`npx tsc -b` exits 0 and `npm --prefix frontend run build` rebuilds the renderer
(1873 modules) and the Electron main/preload bundles. The browser gate is the
existing Playwright harness, and this went into the group that already owns the
opt-in: `npx playwright test tests/e2e/13_gpu_transfer_optin.spec.ts` → **5 passed
(1.1m)** against the real backend, real Electron window and real decode on the
RTX 5060 Ti (`PLAYWRIGHT_RUN_TAG=ui-preview-route-final`, recorded in
`test-results/e2e-ui-preview-route-final.json`: 5 expected, 0 unexpected). That is
the run against the final bundle; the same group passed on the run before it, and
`test-results/` doubles as Playwright's attachment directory, so it holds only the
newest run's artifacts. The new test, 14.5, drives the real
control and reads the real service: selecting GPU Transfer on the DLSS page puts
exactly `{"export_backend":"gpu_transfer","dlss_backend":"gpu_flow_d3d12"}` on the
`POST /api/settings` body; `GET /api/settings` reads back the pair; real preview
frames (`view=dlss`, `frame=0` then `frame=1` after the reload) answer 200
`image/jpeg` at over 1 KB; the service's own ledger says
`kind=gpu_zero_motion @ 1280x720` for the session that served them, with no
`kind=legacy` line after it and no `Preview` ERROR entry; after a page reload the
select still reads `gpu_transfer` and the service still names `gpu_flow_d3d12`;
and choosing Auto posts `{"export_backend":"auto","dlss_backend":"host_pointer"}`,
reopens `kind=legacy @ 1280x720`, serves `frame=2` with no error, and the job
added afterwards carries no `dlss_backend` key. Two harness notes: the group's `dlssTab` helper had
to become an exact-name click, because the DLSS page itself carries a
`DLSS 5 增强` view button and the ambiguous matcher only became reachable once a
test navigated while already on that page; and 14.5 runs with **no** `/api/frame`
stub, unlike 14.1-14.4, because a mocked frame would answer nothing about which
host rendered.

The 4K preview hardware gate was not repeated. It counted entry points, transfer
residency and module residency across five frames at two geometries, and none of
those is a claim this change touches; what it does change is which session the
service reuses when the host kind flips, and 14.5 exercises exactly that on real
hardware, real frames and both hosts. Performance remains
`NOT_CHARACTERIZED`: this gate counted requests, session kinds and one JPEG size
per frame, not throughput.

What this does *not* claim:

* It is not a new GPU capability. Every route here already existed behind
  `POST /api/settings`; this makes the UI write those settings.
* It does not move the guided preview, the image path, the legacy or
  host-pointer export, or the desktop GUI off `dlssnr_host.dll`. What still
  constructs the old host is `dlss_service.py:463` (preview: guided frames, and
  any preview whose export backend is not `gpu_transfer`), `dlss_service.py:1044`
  (a job session that did not ask for `gpu_flow_d3d12`), `dlss_service.py:834`
  (the job's `get_log_tail` readout), and `gui.py:393`, `:399`, `:1351` (the
  desktop GUI's preview, its `process_images` image path and its `--selftest`).
  `pipeline.py:43` still names `HOST_DLL`. Deleting the library would still break
  all of them. *Of that list, the GUI's image path has since moved — the section on
  it, below, is where a still image runs now; the export and the selftest still do
  not. The preview's guided frame has moved too, but only once the backend names the
  GPU host — see the guided-preview section — so a guided preview still reaches `dlssnr_host.dll`
  on the default backend. Those line numbers were read off the file at this stage; the
  current ones are in the boundary section.*
* E2E 14.5 proves what the UI writes, that a real preview frame comes back, and
  which host the service says it opened. It does not re-prove the entry point, the
  flow-array absence or the module residency list — those are the hardware gate's
  claims, and they are unchanged.
* An ABI-2 GPU host that binds no zero-motion texture is still refused before any
  frame (`_open_gpu_zero_motion_live`), and that refusal still reaches the operator
  as a banner rather than a silent host switch.
* There is no JavaScript unit-test runner in this tree, so the page-level pairing
  is checked by reading the shipped source in the Python contract test — the
  pattern `test_export_backend_ui_contract.py` already used — and the HTTP wire
  contract is checked by the Playwright run rather than by a fetched-mock unit
  test.

Files this change touches, and what each one is for:

* `frontend/src/types/index.ts` — `DlssBackend` type, and `dlss_backend?:` on
  `DLSSSettings`.
* `frontend/src/services/api.ts` — `dlssBackendForExport`, the one derivation both
  writers use.
* `frontend/src/pages/DLSSPage.tsx` — the settings writer pairs the host with
  `export_backend`; the panel copy describes preview routing.
* `frontend/src/pages/DeliverPage.tsx` — a job derives its host from its own
  `export_backend` instead of inheriting the preview's.
* `dlss_service.py` — `dlss_backend` validation in `update_settings`; the
  same-host gate on the legacy preview reuse; one `INFO` line naming the host a
  preview session opened.
* `pipeline.py` — `DLSS_BACKENDS` next to `EXPORT_BACKENDS`.
* `tests/unit/test_export_backend_ui_contract.py` (+8), `tests/unit/test_preview_zero_motion.py`
  (+2), `tests/e2e/13_gpu_transfer_optin.spec.ts` (+1 test, 4 harness helpers, the
  exact-name tab click).

No native code, no `native/**`, no build scripts, no `dlssnr_host.dll`. Nothing is
committed or pushed, and no cleanup was run; the worktree still carries the prior
uncommitted work alongside this.

## `guidance_mode=0` on the legacy FFmpeg pipe, on the GPU host's zero-motion entry

This is the last of the four items the 0.5 export work left in `dlss_service`, and
it is the one that was still *refused by name* rather than merely unprobed. An
unguided job that asked for the GPU host by name — `dlss_backend=gpu_flow_d3d12`,
`guidance_mode=0` — was rejected at `dlss_service.py:964-968` unless its
`export_backend` was `gpu_transfer`, and `ProductionExportCore` then rejected the
same session a second time, in its constructor, with
`"GPU-flow DLSS requires an NVOF device flow provider"`
(that message is still in `pipeline.py`, now on the guided branch's own guard)
before dispatching `process_device_flow` on every frame.
The host already had the entry the pipe needed: `process_zero_motion` takes host
RGBA and returns host RGBA, which is exactly the shape this chain has at the DLSS
seam. So the entry point moved and no boundary did.

Wiring, at the three places the decision is observable:

* `dlss_service._execute_job` no longer refuses the pairing on the chain. It still
  refuses a `guidance_mode` outside `{0, 1}` and a non-NVOF flow engine on a
  guided GPU job, both unchanged. What replaces the chain refusal is a question to
  the session: a mode-0 GPU job records its session as
  `self._live_session_kind = "gpu_zero_motion"` and asks
  `supports_zero_motion()`. An ABI-2 host closes the session, clears the slot and
  fails the job naming the library and the ABI — with the words
  `不回退到 dlssnr_host.dll` in the message, because a fallback here would denoise
  the frame on a host the job never named. The queue log line now states which
  chain the zero-motion session was opened for.
* `ProductionExportCore` decides dispatch from the session *and* the mode, not
  from the mode alone: `gpu_zero_motion = gpu host and guidance_mode != 1`. On
  that branch `self.flow_provider is None` — the `ZeroFlowProvider` the
  host-pointer path builds is not built, and a caller that hands a provider to an
  unguided GPU job is refused rather than ignored, because it is an engine nobody
  asked for. `_run_dlss` calls `process_zero_motion(rgba, reset=reset, out=...)`
  inside the same timed `_live_lock` window; the guided GPU branch and the
  host-pointer branch keep their old calls.
* `FrameWorkspace` reserves the CPU motion plane only when something will write it
  (`with_flow_plane`), so the unguided GPU frame allocates nothing. At the
  1920x1080 working size the gate below runs, that is 16,588,800 bytes of zeros
  never allocated, and on a 4K working frame it would have been 66,355,200. Every
  host-pointer frame still gets the plane: that ABI has no entry that takes no
  motion, so the array is its `out=` argument. The capability that the reduced
  transfer order reads for flow input is answered vacuously on this branch —
  there is no flow input to copy.

Byte and entry accounting follows the frame: `dlss_input_flow_bytes_logical` reads
`0` (not `null`: there were no motion bytes to count), and each record plus the
run diagnostics carry `dlss_entry`, `flow_provider` and `flow_plane_allocated`.

**Supported scope, exactly.**

| `dlss_backend` | `guidance_mode` | `export_backend` | What runs |
| --- | --- | --- | --- |
| `gpu_flow_d3d12` | 0 | `legacy` | pipe + `process_zero_motion`, no provider, no plane |
| `gpu_flow_d3d12` | 0 | `auto` | the same: `auto` is the legacy chain |
| `gpu_flow_d3d12` | 0 | `gpu_transfer` | unchanged, device-resident entry |
| `gpu_flow_d3d12` | 1 | any it already accepted | unchanged, NVOF device flow |
| absent / `host_pointer` | 0 or 1 | `legacy` / `auto` | unchanged, `dlssnr_host.dll` |

`auto` is honoured deliberately. `dlss_backend` selects a *host* and
`export_backend` selects a *chain*; `auto` has always resolved to the legacy pipe
(`export_backend_actual: "legacy_pipe"`, with the reason recorded), so refusing
the host name under `auto` would make one word mean two configurations. `auto`
still never selects `gpu_transfer` — that remains opt-in by name, and the
`gpu_transfer` branch is untouched, including its own ABI-3 negotiation. A job
that carries no `dlss_backend` key at all is unaffected: nothing here infers the
GPU host from `auto`.

**Evidence.** `tests/unit/test_export_zero_motion_pipe.py` is new (21 checks, no
GPU): routing through `process_zero_motion` with guided and legacy entries as
traps; `flow_provider is None` *and* `workspace.live_flow is None` *and* zero flow
bytes on the branch; `ZeroFlowProvider.get_flow` never reached, monkeypatched to
record; the contrast test that a host-pointer unguided frame still gets its plane,
its zero provider and its motion call; the guided GPU frame still on
`process_device_flow`; reset flags `[True, False, False, True]` across a decode
discontinuity; an ABI-2 host and a host that cannot answer the probe both refused
before frame zero; a provider handed to an unguided GPU job refused; an inference
failure propagated with the core still resetting on the next frame; the host
handed the prepared RGBA (channel order and opaque alpha checked) writing into the
workspace buffer, with the red bump it makes visible in the frame that reaches the
encoder. At the queue: the explicit pairing completes on a GPU session with
`flow_provider=None` at the export seam, `auto` routes the same way,
`gpu_transfer` still does its own thing, a guided job still opens the flow engine
and is recorded as `kind="gpu_flow"`, an ABI-2 host fails the job by name with no
`dlss_engine.Live` constructed, a GPU job neither reuses nor leaves behind a
host-pointer session, a host-pointer job still reuses and `update()`s its own, and
across three jobs the GPU host opens once and closes while two host-pointer jobs
share one session.

`tests/unit/test_export_backend_ui_contract.py` lost the test that pinned the old
refusal (`test_unguided_on_a_chain_without_the_zero_motion_entry_names_that_chain`)
and gained two that pin the routing instead, including that the refusal text does
not come back; it is now 20 checks. `test_the_service_no_longer_refuses_guidance_off_on_the_gpu_transfer_chain`
is unchanged and still passes. The full unit suite is
**521 passed**, no failures, in 215.58s (499 before this change; the 22-test net gain
is the new file plus the two replacements, and no test now waits on a GPU).

The hardware gate is new: `scratch/_zero_motion_pipe_4k_gate.py`, one 4K export of
5 frames through `pipeline.export_dlss_stream` with the explicit
`gpu_flow_d3d12 + guidance_mode=0 + export_backend=legacy` triple, `working_scale=0.5`
(1920x1080 working on a 3840x2160 source). `ctypes.CDLL` is sabotaged from outside
so `dlssnr_host.dll`, `nvof_bridge.dll`, `nvofapi64.dll` and `nvofnvml64.dll` raise
on load, and `dlss_engine.Live` raises on construction; the GPU host's guided
entries raise too, so a frame that asked for motion is a failure rather than a
slower success. **14/14 checks passed**: `process_zero_motion` answered 5 of 5
frames with 0 guided calls and 0 errors, one session opened at the working size,
each result written into the buffer it was handed, `reset` true on frame 0 only,
the core that ran holding no provider and no motion plane and reporting
`dlss_entry=process_zero_motion`, zero flow and zero depth bytes on every frame,
`export_backend_actual=legacy_pipe` with `flow_plane_allocated=false`, no NVOF
engine and a zero execute delta, an empty blocked-load list with
`dlss_gpu_host.dll` the only host resident in the process, 5 frames at 3840x2160
av1/yuv420p with unique PTS, and no leaked intermediate file. The run took the
**reduced** transfer order (`transfer_mode_actual: "reduced"`,
`prepare_bgr_rgba_call_count: 5`, `reconstruct_rgba_input_call_count: 5` on
`cpu_reference_native`), so this also gates the fused-store + in-place-RGBA-read
order working against the unguided GPU host, which no earlier run covered.
Output: `bench_out/zero_motion_4k_legacy_pipe.mkv`, 4,576,079 bytes; the run's own
record is `bench_out/zero_motion_4k_legacy_pipe.log`.

**Re-verified against the working copy afterwards.** The suite was re-run from the
tree as it stands and reported **521 passed**, no failures, in 219.46s; the two
files re-collected at 21 and 20 checks. The 4K export was *not* re-run: each of the
gate's 14 verdicts was re-derived from the recorded log alone, and `ffprobe`
re-read the output as 5 frames of av1/yuv420p at 3840x2160 with a format size of
4,576,079 bytes. The queue-to-seam wiring was read rather than executed: at
`guidance_mode=0` the job opens no guidance service, so `dlss_service.py:1109`
builds `flow_provider=None` and passes that to `export_dlss_stream` — the argument
the new branch requires and the one it refuses anything else to be given. No code
change came out of that re-check.

**Limitations of that evidence.** The gate drives `export_dlss_stream` directly
with the arguments the queue builds for this pairing; the queue-to-seam step is
proven by the unit tests above with fakes, not by the 4K run. The chain still
crosses CPU memory twice per frame — decode pipe in, encoder pipe out — which is
what makes it the legacy chain; removing the flow upload does not remove the
pipe, and only `export_backend=gpu_transfer` does that. The reset rule is the pipe
loop's own (frame 0 and any index discontinuity, no scene-cut detection). The
matrix tag is still absent from this chain's output; the gate pins that it *stayed*
absent rather than letting the branch claim a colour space it never wrote, and the
4K legacy baseline `bench_out/gpu15_legacy.mkv` reads the same way. Performance for
this pairing is `NOT_CHARACTERIZED`: five frames is a correctness sample.

**One correction, for the record.** An earlier revision of this document claimed
that the queue-gate refusal of `guidance_mode=0` on the pipe chain "no longer exists
in the tree". It did exist: at `dlss_service.py:964-968` of the working copy going
into the change this section describes, in a queue gate that had itself been written
but not committed, with a unit test asserting on the text by name. What was true of
that claim is only that the *gpu_transfer* pairing was no longer refused, which is a
different sentence. This change removes the refusal rather than this document
recording it, so those line numbers are quoted from the file it was deleted from and
are not findable in any commit; `git diff` for this tree shows no trace of either the
added refusal or its removal, which is exactly what a same-session add-then-delete
looks like.

## The desktop GUI's still images on the GPU host's zero-motion entry

Every `guidance_mode=0` item above lives in the service. The still-image path never
did: it is `gui.py`'s, and the document said so repeatedly, because the window carried
no `dlss_backend` control and asked `dlss_engine.Live` for every frame it ran, guided
or not. A still image is the one frame in this tree that cannot use motion — image
mode already forces `guidance_mode = 0`, and the path then handed the old host a zero
plane to ignore and a zero depth plane to read. So this is the migration the previous
sections kept pointing at, done in the one place that had no routing decision to make.

Wiring, all of it inside `gui.py`:

* `get_dlss_session(w, h, settings, kind="legacy")` (`gui.py:393`) names the host a
  caller needs, and one slot serves both kinds
  (`_get_or_create_session_locked`, `gui.py:413`). One slot, not two, because
  `dlss_gpu_host_open` refuses a second session while one is active and
  `dlssnr_host.dll` is not safe to init twice either; a request for the other host
  closes the cached session instead of driving it through entries it does not have.
* The image kind cannot live-update. The GPU host reads its ten options once, at
  open, so the session records `_gpu_options_key` (`gui.py:466`) and reuse compares
  it (`gui.py:425-429`); the legacy kind keeps its existing `update(settings)`.
* `_open_image_zero_motion_live` (`gui.py:449`) refuses an ABI-2 host *before* frame
  zero and does not fall sideways: running the picture on `dlssnr_host.dll` instead
  would report a GPU-host run that did not happen, and that DLL has no entry that
  takes no motion in the first place.
* `_dlss_image` (`gui.py:929`) is now the single implementation of pad-to-8, the RGBA
  assembly, the session lookup and the call, with the lookup and the call inside one
  `_dlss_lock` hold so a concurrent video rebuild cannot evict the session between
  them. `reset=True` every time — each still is its own sequence. `_image_dlss`
  (`gui.py:920`) turns an unusable host into a logged `None`; `_do_images`
  (`gui.py:1207`) claims the slot for the run (`_image_batch_running`), parks the
  preview worker so a video frame and a folder cannot evict each other per image, and
  aborts a batch once on a host that will not open rather than once per remaining
  file. Neither path allocates a motion or a depth plane any more.

**Evidence — no GPU.** `tests/unit/test_gui_image_zero_motion.py` is new: 28 checks
covering the routing with both hosts' foreign entries as traps, the pad-to-8 /
trim-back geometry including that the ring is *replicate* padding and not a black
border the host would then denoise, channel order and opaque alpha, `reset` on every
independent picture, reuse while geometry and the baked options hold and a rebuild
when either moves, the batch's existing naming and skip and per-image-failure and
cancellation behaviour, the preview-worker pause/resume pairing on the host-fails,
cancelled and raising paths, that no `np.zeros` motion or depth plane is built, that
`optical_flow` is neither imported nor entered, and that an ABI-2 host is refused
rather than swapped. That run is the operator's, on this tree, and it reported
**28 passed** with `gui.py` and `dlss_gpu_engine.py` byte-compiling; the gate below
changed no source file, so it was not re-run here. Getting to those 28 required one
correction in the test file itself: the lock probe in the last check released
`app._dlss_lock` unconditionally, including on the path where it had not acquired it —
which is the hold the frame is running inside, not the probe's to give back. It now
releases only what it took. The two neighbouring files that pin the same one-slot
factory and the same flow independence were run on this tree for the record:
`tests/unit/test_preview_zero_motion.py` and `tests/unit/test_nvof_independence.py`,
**38 passed** in 4.35 s.

**Evidence — real GPU.** `scratch/_gui_image_zero_motion_gpu_gate.py` is new and is
the first still-image hardware gate: **18/18 checks passed** on an RTX 5060 Ti
(driver 616.92) against the ABI-3 `native/dlss_gpu_host.dll`. It drives the real
`gui.App._image_dlss`, `_dlss_image` and `_do_images` on an `App` built with `__new__`
from a worker thread, wraps — does not replace — the real `dlss_gpu_engine.Live`, and
sabotages `ctypes.CDLL` from outside so `dlssnr_host.dll`, `nvof_bridge.dll`,
`nvofapi64.dll` and `nvofnvml64.dll` raise on load and `dlss_engine.Live` raises on
construction. Sizes are WxH throughout. Six frames in four sessions:

| opened at | frames | host's own `host_color_stage_count` |
| --- | --- | --- |
| 1288x728 (a 1282x722 still, neither dimension /8) | 2 | 2 |
| 1288x728 again, `intensity` moved 1.0 → 0.5 | 1 | 1 |
| 1280x720 (batch files 0 and 1) | 2 | 2 |
| 808x456 (an 806x454 batch file) | 1 | 1 |

Three single-image requests opened two sessions: the first two ran on one, which is the
reuse, and the third opened the next, which is the baked-option rebuild. The batch then
reused across its two same-size files and rebuilt at the new geometry. Every frame
reported `reset=True`, `alpha_min=255`, a contiguous `uint8 (h, w, 4)` in and an out of
the same shape, and zero calls to any flow entry. The staging column is the host's own
counter, read through `transfer_counters()` before each close, and it equals the frames
each session was given — the colour moved on the card, not around in Python.
`process_zero_motion` returned in 7.2–10.9 ms per frame; that is a liveness record, not
a rate. Results came back at the *original* sizes (1282x722 and 806x454) and every
batch PNG re-read through `pipeline.imread_u` as `uint8` at its source size. Each
fixture carries a hard step edge at a chosen, off-centre x and y; the edge stayed
within 0.53 px of where it was put, on all four geometries and on both axes, which is
what a pad/trim that misplaced the frame would have broken. The blocked-load list is
empty, the four libraries above are absent from the process by `GetModuleHandleW`,
`optical_flow._ENGINES` stayed empty with a zero execute delta, and `dlss_gpu_host.dll`
is resident with the `nvof_d3d12_interop.dll` it links at load time — a CUDA↔D3D12
shim, not flow, which is why item 3 below still wants it renamed. Teardown closed all
four sessions and left `_dlss_session` and `_dlss_session_kind` empty. No check
reported an unhandled exception. The record is
`bench_out/gui_image_zero_motion_gpu_gate.json` with stdout in
`bench_out/gui_image_zero_motion_gpu_gate.log`, and it carries its own card and driver
(`NVIDIA GeForce RTX 5060 Ti, 616.92`) rather than a hardcoded label. The script was
then edited twice — that label became a query, and one more flow entry joined the
refusal list — and the gate was run again: **18/18 again**, with the same four
`opened_at`/frame/staging rows, the same edge shifts, the same mean absolute difference
and the same repeat verdict. Only the per-frame milliseconds differ between the two
runs (the first read 5.6–8.6), which is exactly why neither figure is quoted as a rate.
The two artifact files hold the second run; the first run's stdout was overwritten by
it, and its figures above are quoted from the console record of that run rather than
from a surviving file.

Two things the run observed and does *not* claim. Repeating an identical still on one
session came back byte-identical (`repeat_bit_identical: true`), and the frame is not a
copy of its input either (mean absolute difference from that input: 1.725 levels). Both
are determinism and liveness inside this host, measured twice. Neither is parity with
the old host: `dlssnr_host.dll` is refused in this process, so this gate structurally
cannot compare the pictures either host makes.
Image-mode output equivalence across the migration is therefore **not measured**, and
nothing here should be read as saying the denoised still looks like it used to.

**Limitations of that evidence.** Six frames under a megapixel each is a correctness
sample; still image performance on the GPU host is `NOT_CHARACTERIZED`. The gate builds
its `App`
with `__new__`, so no Tk root, no widgets, no button handlers, no file dialog and no
`display_view()`; `_collect_settings` is reached through the cached-settings branch
that the real worker thread takes, and nobody clicked anything or looked at a window.
The batch ran with no `_preview_worker` on the app at all — the folder-pick-without-
video case — so the pause/resume of a *live* preview worker around a batch is
unit-tested only. `intensity` stands in for the ten baked options, one reuse and one
rebuild rule each. The odd sizes tested are 1282x722 and 806x454; the `max(8, ...)`
clamp for a sub-8 image, very large stills, non-8-multiples in only one axis, EXIF
orientation and 16-bit or RGBA sources are not covered. A folder that mixes an
unreadable file, an existing output and a cancellation is likewise pinned by the unit
file, not by this run. And the GUI's *video* paths still run on the old host, so this
closes the image half of the GUI and not the window.

## A host-memory motion entry on the GPU host, validated and not yet routed

*The heading describes this stage; the guided preview has since been routed onto this
entry, which is the section below.*

The GPU host could already take a frame from CPU colour and either NVOF device flow
or its own resident zero texture. The one shape every guided preview and the desktop
GUI still hand to `dlssnr_host.dll` — colour and motion both arriving as host
pointers, no CUDA — had no GPU-host entry, and the previous two sections named that
missing entry as the single reason those callers could not leave the old DLL. This
section adds the entry and proves it works. It does **not** route any caller onto it:
per the boundary of this change, no GUI or service routing moved, and
`dlssnr_host.dll` was untouched and still the host those callers used — the guided
preview being the one of them the section below takes off it.

The new exports on `native/dlss_gpu_host.dll` are additive at ABI 3 (no version bump;
capability is gated by symbol presence, exactly as `zero_motion_supported` is):

* `dlss_gpu_host_process_host_motion(host, rgba, rgba_stride, motion, motion_stride,
  reset, output_rgba, output_stride)` — host RGBA8 in, a host `float32x2` motion plane
  in, host RGBA8 out. `motion` is the H×W×2 pixel-space displacement the legacy
  `Live.process` took: full `DLSSNR.Color` extent, row-major, x fastest, top-left
  origin, `motion_stride` at least `width * 8`. Depth stays the immutable zero texture;
  `MVecScaleX/Y` keep their flow-path meaning (the scale is applied by DLSS, not baked
  into the plane).
* `dlss_gpu_host_readback_motion(host, dst, dst_stride)` — an explicit D2H copy of the
  resident R16G16_FLOAT motion texture, for a gate that wants to prove the plane landed
  rather than being zeroed. It is not on any frame path.

The conversion is on the card, not in numpy. The host stages the caller's rows into a
flat upload buffer, copies them into a default-heap buffer, and runs a compute pass —
byte-identical HLSL to the guided NVOF conversion — that loads each `float32x2` and
writes the `RWTexture2D<float2>`. This is the design decision the earlier half-plane
attempt forced: `numpy`'s `float32→float16` narrows through a `float64` intermediate
and can differ by 1 ULP from a single-rounded cast, and the temporal filter amplifies a
1-ULP motion difference into visible frame-to-frame divergence. Doing the cast in the
same compute the flow path uses removes numpy from the loop and hands DLSS, for the same
float32 vectors, the same half2 texture whether the motion arrived from the host or from
NVOF. `process_impl` branches on `host_motion` / `use_zero_motion` / `guided_flow` so a
host frame touches no CUDA and loads no flow bridge, and the zero-motion and NVOF-device
paths bind exactly as before.

Verified, not asserted. The hardware gate
`native/dlss_gpu_host_motion_smoke.py` (three 1280×720 frames) checks:

* **the plane reaches the bound resource** — after each frame
  `readback_motion()` equals the caller's `float32` displacement cast to `float16`, and
  is not zero; on integer fixtures this is an exact byte comparison, and the reset frame
  proves the values are the ones DLSS read;
* **parity with the old host** — each frame, run on a reset so it is the deterministic
  spatial reconstruction rather than a history the two independent sessions seed
  differently, is **byte-identical** to `dlssnr_host.dll` for the same colour and motion
  (`frame0/1/2_vs_legacy_host_motion = identical`); the non-reset temporal run is exercised
  but is not byte-claimed, because a per-frame divergence there is two separate DLSS
  histories, not a wrong motion vector;
* **malformed input is refused** — wrong extent, wrong channel count, `float64`/`float16`
  dtype, a non-contiguous slice, and a non-contiguous RGBA each raise before a native
  call;
* **the paths that already worked still bind** — a zero-motion frame on the same session
  still produces a real image;
* **no resource leak** — close returns the handle to dead and a reopen at a different
  extent still evaluates.

The CPU-only gate `tests/unit/test_gpu_host_motion_wrapper.py` (11 checks, no GPU) pins
the wrapper contract so a machine without a card still catches the regression the GPU
gate needs a card to see: the `float32` plane reaches the native entry **uncast** at a
`width*8` row pitch (a wrapper that narrowed to `float16` first would pass the GPU gate's
integer fixtures and drift on real sub-pixel motion), malformed planes are refused before
any native call, an old host that exposes no symbol is refused with a reason instead of
running the frame on zeroed motion, and a nonzero native result is surfaced. The existing
gates `native/dlss_gpu_host_smoke.py` (guided NVOF) and
`native/dlss_gpu_zero_motion_smoke.py` (unguided) both still pass unchanged, so the new
entry did not displace the paths that already worked.

What this does **not** claim:

* **No caller moved.** `dlss_service.py`, `gui.py`, `server.py` and the preview/export
  routing are unchanged; every guided host-pointer frame still runs on
  `dlssnr_host.dll`. This is the entry point the boundary said was missing; reaching it
  from a route is the next step, and this change deliberately stopped short of it.
  *Which is what happened next, and only for that one route: the section below drives
  the guided preview on `dlss_backend=gpu_flow_d3d12` through this entry. `gui.py`,
  `server.py`, every export path and every guided frame on the host-pointer backend are
  still exactly where this bullet puts them.*
* **Performance is `NOT_CHARACTERIZED`.** The build carries `/W4 /WX` and the gate runs a
  real card, but no benchmark was run and no timing claim is made here or in the ABI
  report.
* The compute conversion narrows `float32→float16` because the bound MVec resource *is*
  `DXGI_FORMAT_R16G16_FLOAT`; that is a property of what DLSS reads, not of this entry.
* Magpie's `DLSSNRFilter` was read only as a reference for the convention; its GPLv3
  implementation was not copied into this tree.

Next migration steps, in the order they unblock themselves: route the preview's guided
branch (`dlss_service.py:463`) onto `process_host_motion` behind a capability check on
`supports_host_motion()`, keeping the zero-motion and device-flow branches intact and
refusing rather than silently switching when the host predates the entry (the pattern
`test_preview_zero_motion.py` already enforces for zero motion); then the desktop GUI's
video worker and its own export (`gui.py:444` / `gui.py:393`); only after both, and after
the `optical_flow` import edge, does `dlssnr_host.dll` removal (item 4 below) clear its
last host-motion blocker. *The first of those is done — see the section below.*

## The guided preview on the GPU host's host-motion entry

The section above added `dlss_gpu_host_process_host_motion`, proved it on a card, and
left every caller where it was. This moves one of them: a preview that names
`dlss_backend=gpu_flow_d3d12` **and** `guidance_mode=1` now runs on that entry instead
of `dlssnr_host.dll`. No native code changed, no DLL was rebuilt for this, and no
export, image or device-flow path changed — the whole change is one routing decision,
the session bookkeeping that had to follow it, and UI copy that had started describing
the old answer.

The route. `_preview_live_kind` (`dlss_service.py:368`) now asks which host first and
what for second: the D3D12 backend names the GPU host for **both** guidance modes, mode
0 taking `process_zero_motion` as before and mode 1 taking a new, separately named kind,
`gpu_host_motion`. Separately on purpose: the two are different entry points with
different capability gates, and one kind covering both would let a session opened for
one be driven through the other. `_open_gpu_live` (`dlss_service.py:413`) opens the
session, calls `supports_host_motion()` on it, and closes and refuses with the reason if
the answer is no — so a stale `dlss_gpu_host.dll` produces a visible failure, never a
frame quietly denoised by `dlssnr_host.dll`. `_preview_apply_live`
(`dlss_service.py:499`) hands the guided frame the plane guidance already made: the same
contiguous `float32` HxWx2 object the provider returned, at the session's own extent.
`process_device_flow` is untouched by this route, because the preview holds host memory
and the device entries read a CUDA descriptor; passing one as the other is exactly the
confusion the wrapper's own argument checks exist to refuse.

What had to survive, and was pinned rather than assumed:

* **geometry** — the session still opens at the pyramid working size when
  `residual_pyramid and working_scale < 1.0` and at full native otherwise, and the motion
  plane is asked for at the identical extent (`flow_dims` in `render_preview`), which is
  what `process_host_motion`'s shape check enforces on every frame;
* **baked-option reuse** — `_gpu_preview_options_key` is unchanged, so a session is
  reused only for the identical request and reopened when intensity or a motion scale
  moves (the GPU host has no live-update entry);
* **reset and seek** — `reset=(frame_idx == 0)` on every entry, so scrubbing back to
  frame zero replays the temporal seed rather than continuing a history;
* **the shared session slot** — flipping `guidance_mode` at unchanged dimensions now
  closes the guided session and opens a zero-motion one, and back; flipping
  `dlss_backend` closes the GPU session and reopens `dlss_engine.Live`. The reuse branch
  requires `_live_session_kind == kind`, which is what the new kind makes meaningful;
* **the lock and the banner** — the 2s `_live_lock` timeout still serves the source with
  the occupancy indicator, and a GPU frame that raises `DLSSInferenceError` still serves
  the source with the red `[DLSS 增强失败: …]` overlay and the host's own reason through
  `_preview_live_error_detail`;
* **everything not asked to move** — a preview with no `dlss_backend` or with
  `host_pointer` runs the same code it ran before, `view=orig` and `view=flow` open no
  DLSS session, an active export job still gets the source frame, and `split`/`dual`
  still composite the enhanced half;
* **guided NVOF/DIS/RAFT configuration** — the preview consumes the plane, so it never
  needed the *export's* `flow_engine=nvof` rule (`dlss_service.py:1005`), which is about
  a device-resident descriptor. No engine was refused here and no real incompatibility
  surfaced: the hardware gate below runs the shipped NVOF default, and the unit file
  pins DIS and RAFT feeding the same entry with the same plane contract.

Verified, not asserted. `tests/unit/test_preview_host_motion.py` (25 checks, no GPU)
drives the real `render_preview` against injected hosts and proves the routing by
tripping, not by reading a call ledger: the GPU host's `process()`/`process_zero_motion`
/`process_device_flow`/`update` all raise, the old host's `process_host_motion` raises,
and `dlss_engine.Live` is optionally replaced by a factory that raises a
**non-`Exception`** trap, because `_get_or_create_live` turns any `Exception` from a
session create into a failure banner and an `AssertionError` trap would have been
swallowed by the code under test and reported as a served frame. It pins that the plane
reaching the host `is` the object guidance returned (dtype, shape, `c_contiguous` and a
`width*8` row pitch checked, non-flatness checked); that a guided frame on the default
backend still gets its flow on the old host; that reuse, the two rebuild triggers, the
reset sequence, the backend flip, the guidance flip, and an export's cached
`gpu_flow` session all behave as the zero-motion branch already required; that a host
without the symbol is refused with `rebuild` in the message and the old host never built;
that a failing frame yields a banner rather than a crash; and that the `orig`/`flow`
views, the lock timeout and the queue collision touch neither host. One test in
`tests/unit/test_preview_zero_motion.py` — `test_guidance1_with_gpu_backend_keeps_legacy_flow_path`
— asserted the old answer and was rewritten to the new one rather than left to rot; the
file keeps the guided case as a cross-check that the unguided branch's guarantees did not
leak into it. `tests/unit/test_export_backend_ui_contract.py` gained the settings-level
pair (`guidance_mode=1` + `gpu_flow_d3d12` ⇒ `gpu_host_motion`, and the write back to
`host_pointer` ⇒ `legacy`). Its own stale pin —
`test_a_guided_preview_keeps_the_host_pointer_route_under_gpu_transfer`, which read the
panel copy and the derivation and asserted the guided frame stayed on the old host —
became
`test_a_guided_preview_routes_to_the_gpu_host_motion_entry_under_gpu_transfer`.

On hardware: `scratch/_guided_preview_host_motion_gate.py` (16/16 checks,
`bench_out/fixture_4k_gpu15.mp4`, 3840x2160 @ 29.97, four frames — 0, 1, 0 at the
pyramid working size 1920x1080, then 1 at full native). `dlss_engine.Live` is sabotaged
and `dlss_gpu_engine.Live` counts:

* `host_motion=4`, `zero_motion=0`, and every other GPU entry `0` — the guided preview
  ran on `process_host_motion` and nowhere else; `opens=2`, one per geometry;
* `dlssnr_host.dll` is **absent** from the process by `GetModuleHandleW`, and
  `dlss_gpu_host.dll` present — the absence is read off the module list, not inferred
  from which factory was called;
* the planes the host received were real NVOF output: `(1080,1920,2)` `float32`
  contiguous at pitch 15360 for the three working frames and `(2160,3840,2)` at pitch
  30720 for the native one, with `abs_mean` 0.031 / 6.13 / 0.031 / 12.26 pixels,
  `abs_max` up to 115.7, and 3.7M / 14.7M nonzero pixels — the near-zero first and third
  rows are the two resets, where the service hands NVOF the frame against itself;
* after each frame `readback_motion()` agrees with the caller's plane: worst deviation
  across all four frames was 0.42 of the `float16` allowance (and exactly 0 for the two
  near-flat frames), with nonzero texels every frame — so the displacement reached the
  `R16G16_FLOAT` resource DLSS binds rather than being dropped, zeroed or reinterpreted;
* the host's own counters, read as per-frame deltas: `host_color_stage_count` +1 and
  `output_readback_count` +1 every frame (8,294,400 bytes at 1080p working, 33,177,600 at
  4K) and `device_color_copy_count` +0 — this entry stages colour from host memory and
  carries the result back to a host buffer, which is what a host-pointer frame is;
* `reset` flags `[True, False, True, False]` across the seek, every served JPEG decoded
  to 3840x2160x3, and no ERROR-level preview diagnostic was written.

What this does **not** claim:

* **Performance is `NOT_CHARACTERIZED`.** Four frames on one clip are a correctness
  sample; no timing, no A/B against the old host on this route, no throughput figure.
  The per-frame output readback above is a real PCIe cost, stated because it is
  measurable, not because it was compared to anything.
* **Nothing else moved.** A guided export on either chain, a host-pointer preview, and
  the desktop GUI's video preview and export still construct `dlssnr_host.dll`; the
  parity evidence for a guided *frame* is the earlier `native/dlss_gpu_host_motion_smoke.py`
  reset-frame byte comparison, not a new one, and this gate does not re-run it.
  *One half of that sentence is now history: the section below this one routed a guided
  export whose motion arrives from host memory onto this same entry, on the legacy pipe
  chain. A guided export on the GPU Transfer chain still takes the device descriptor it was
  already taking, and a host-pointer frame of either kind still constructs the old DLL.*
* The preview still pays for a CPU flow plane: NVOF computes on the GPU, `optical_flow`
  downloads it, and the host uploads it again. This route removes the old **host** from
  the guided preview, not the D2H/H2D round trip — the zero-copy statement in
  `ABI_CAPABILITIES.md` §3.2 is unchanged.
* `tests/e2e/13_gpu_transfer_optin.spec.ts` was not re-run: 14.5 pins the
  `guidance_mode=0` pairing, which this does not touch, and the guided pairing has no
  Playwright gate to update. The shipped frontend changed only panel copy and a comment;
  `npm run build` (tsc -b + vite) passed, `npm run lint` reports the same five pre-existing
  warnings on lines this change did not touch.
* No `dlss_backend` control exists in the Tk desktop window, so its guided video frames
  cannot name this host yet; that is item 1 below, not this section.

## A CPU flow provider in the export, on the GPU host's host-motion entry

The section above routed one caller onto `dlss_gpu_host_process_host_motion`, and said the
guided *export* had not moved. This moves it: `pipeline.export_dlss_stream` /
`ProductionExportCore` on an explicitly named GPU host now takes a flow provider whose
motion can only arrive as a host array, and runs every frame on that entry. No native code
changed, no DLL was rebuilt, the service's GPU Transfer negotiation and device path are
untouched, and no GUI routing moved. The whole change is one routing decision in the core,
one new diagnostics key, and the refusals that keep a frame from reaching an entry that
cannot read it.

Why it was a decision rather than a call. The core detected a GPU host by the presence of
`process_device_flow` (`pipeline.py:2383`) and then required the provider to answer
`get_flow_device`; anything else was refused outright. That is the right rule for an NVOF
frame, whose vectors *are* a CUDA descriptor, and the wrong rule for
`pipeline.CachedFloFlowProvider` — the desktop GUI's cached `.flo` route — which reads
vectors off disk and can hand over nothing but a `float32` HxWx2 array. The two are not
interchangeable at the ABI boundary: the first is bound where it lies, the second is staged
into an upload buffer by the host. So the choice is now made once, before frame zero, from
what the session reports about itself and what the provider can actually be made to return
(`_flow_delivery` at `pipeline.py:2492`, `_route_guided_flow` at `:2508`) — probed as
methods implemented, never off a class name:

| frame shape | entry | `flow_transport` |
| --- | --- | --- |
| `guidance_mode=0` on the GPU host | `process_zero_motion` | `resident_zero_texture` |
| device descriptor on the GPU host | `process_device_flow` | `nvof_device_descriptor` |
| CPU `float32x2` plane on a GPU host that reports the symbol | `process_host_motion` | `host_plane_upload` |
| CPU plane on a host-pointer session (unchanged route) | `process` | `host_plane_legacy_session` |

A provider that can do both is sent the descriptor (`:2530`), because a caller that can keep
the vectors resident must not be made to round-trip them just to satisfy a host entry —
which is what keeps the opt-in GPU Transfer chain's `nvof_d2h_copy_count_delta == 0`
guarantee true after this change rather than merely unfalsified. The pairings that cannot be
served are refusals with the reason in them: a CPU plane on a GPU host whose loaded DLL
predates the entry (`_require_host_motion_host`, `:2545`, naming
`native/dlss_gpu_host.dll`, `dlss_gpu_host_process_host_motion` and `dlssnr_host.dll`), and
a device-descriptor-only provider on a host-pointer session (`:2536`), where no entry can
read it. The one message that used to cover both cases — "GPU-flow DLSS requires an NVOF
device flow provider" — is gone, because it was wrong about half of them.

What had to survive, and was pinned rather than assumed:

* **the two guided GPU routes stay distinct in the ledger.** `dlss_entry` gained its fourth
  value and `flow_transport` is new, written into the run diagnostics (`:2630`) and into
  every frame record (`:2818`); `dlss_input_flow_bytes_logical` (`:2789`) is `0` for a
  resident zero texture, the descriptor's *extent arithmetic* for a device frame (those
  vectors were never in host memory, so no H2D of them exists), and the plane's `nbytes` for
  a host upload. A CPU-flow frame reports a nonzero figure and **no NVOF number at all**
  (`nvof_d2h_copy_count is None`), where a resident NVOF frame reports a measured `0` —
  absent versus zero is the difference between "no flow was downloaded because no flow ran"
  and "flow stayed on the card";
* **no zero-copy claim.** The loaded host counts colour staging and has no motion counter, so
  the plane's crossing is unmeasured and `dlss_gpu_transfer_bytes_measured` stays `None` on
  both guided GPU routes; the byte figure above is the size of a buffer somebody handed a
  function, not a measurement of PCIe;
* **geometry, reset, reuse, and the workspace plane** — the plane is asked for at the
  session's own extent and filled into the core's reused `workspace.live_flow`
  (`pipeline.py:2686`), `reset` follows the first frame and every decode discontinuity, and
  the device branch still reserves a plane it never reads (`flow_plane_requested = not
  gpu_zero_motion`, `:2438`): pre-existing, pinned by an existing test, and deliberately not
  weakened while I was in the file;
* **the transfer order is orthogonal.** A cached provider satisfies
  `flow_input_needs_no_rgba_copy`, so `transfer_mode=auto` on this route negotiates `reduced`
  whenever the residual backend can fuse the working stores — the hardware run below did — and
  falls back to `legacy` with the missing capability named when it cannot (pinned with
  `residual_backend: "python"`). The provider is handed `input_format` either way and a
  cached plane ignores it;
* **everything not asked to move** — the NVOF device path, the zero-motion path on both
  export chains, the host-pointer export, `gpu_transfer_export.py`'s own frame path, and
  `dlss_service`'s guided GPU job, which still hands that chain a device provider
  (`tests/unit/test_export_cpu_flow_gpu_host.py` pins exactly that so this file cannot pass
  by quietly moving the service too).

Verified, not asserted. `tests/unit/test_export_cpu_flow_gpu_host.py` (23 checks, no GPU, no
flow library) drives the real core — and in one test the real `export_dlss_stream`, against
a fake Python encoder — with fakes whose *wrong* entries raise and whose argument checks
mirror the real wrapper's. It proves the route by tripping, not by reading a call ledger:
the plane reaching `process_host_motion` is the real nonzero vectors, at the session extent
rather than the source extent, contiguous `float32` at a `width*8` pitch; the object handed
over `is` the workspace buffer; reset follows frame zero and a decode discontinuity; a
missing cached frame degrades to zeros and is *counted*; an inference failure propagates and
never reaches the legacy host; a host that cannot answer `supports_host_motion()` — or whose
answer is no — is refused with `rebuild` in the message and `dlss_engine.Live` never built; a
device descriptor still takes the device entry, a both-capable provider is not downloaded to
the host entry, and the guidance-off branch still hands the host no motion at all. The
diagnostics assertions above are each pinned on both guided routes side by side. The full
unit suite is green at **608 passed** (`python -m pytest tests/unit -q -p no:randomly`,
run twice with the same result), 23 of them this file's.

On hardware, in two processes on purpose:
`scratch/_cpu_flow_export_flo_prep.py` writes `bench_out/cpu_flow_gate_flow/00000{0..4}.flo`
from OpenCV DIS at the 3840x2160 source extent — the half that cannot live in the gate,
because the gate refuses to load any NVIDIA flow library — with frame 0 written as the exact
zero plane, which is what the project's own `generate_flow` does for a first frame and what
turns the readback below into a per-frame claim. Then
`scratch/_cpu_flow_export_gate.py` (**22/22 checks**, one run, no retries,
`bench_out/fixture_4k_gpu15.mp4`, five frames, `guidance_mode=1`,
`dlss_backend=gpu_flow_d3d12`, `export_backend=legacy`, `working_scale=0.5` → 1920x1080
session, `residual_pyramid`, av1_nvenc p7 through the FFmpeg stdin pipe):

* `host_motion=5`, `other_entry=0`, `errors=0`, `opens=1` — and every GPU entry *except*
  `process_host_motion` sabotaged, so a frame that took a device or zero-motion entry is a
  failure rather than a different flavour of success;
* `dlssnr_host.dll`, `nvof_bridge.dll`, `nvofapi64.dll` and `nvofnvml64.dll` were never
  requested through `ctypes.CDLL` and are absent from the process by `GetModuleHandleW`,
  while `dlss_gpu_host.dll` is present; `dlss_engine.Live` raises; `optical_flow._ENGINES`
  is still empty and the NVOF execute count moved by 0 — the legacy host and NVOF are proved
  absent, not inferred;
* every plane was `(1080, 1920, 2)` `float32`, contiguous, pitch 15360, and was the core's
  `workspace.live_flow` object itself: mean displacement 11.13 / 8.22 / 8.23 / 11.43 px,
  `abs_max` up to 49.7, all 4,147,200 vector components nonzero, five frames read off disk
  with zero misses;
* `readback_motion()` after each frame agrees with the plane that frame was handed — worst
  deviation 0.48 of the `float16` allowance — and frame 0, whose plane was exactly zero,
  reads back exactly zero, so the agreement is *this frame's* upload rather than one good
  copy sitting in a resource; 4,147,194 of 4,147,200 components survive the narrowing
  nonzero on the sparsest frame, i.e. the handful that round to zero in half precision;
* the run's own diagnostics say `dlss_entry=process_host_motion`,
  `flow_transport=host_plane_upload`, `export_backend_actual=legacy_pipe`,
  `flow_plane_allocated=true`, `transfer_mode_actual=reduced`, and all five frame records
  report `dlss_input_flow_bytes_logical=16,588,800` with `nvof_d2h_copy_count` and
  `nvof_h2d_copy_count` `None` and `dlss_gpu_transfer_bytes_measured` `None`;
* the host's own counters, read before the gate's readback: one colour stage and one output
  readback per frame, `device_color_copy_count` 0 — a host-pointer frame by its own
  accounting;
* the finished 4,579,589-byte MKV decodes 5 frames at 3840x2160, `av1`/`yuv420p`, unique
  PTSs, limited range with no matrix tag (that chain's long-standing tagging, unchanged by a
  guided frame), and nothing was left behind as an intermediate file.

What this does **not** claim:

* **Performance is `NOT_CHARACTERIZED`.** Five frames of one clip on one card is a
  correctness sample: no timing, no A/B against `dlssnr_host.dll` on this route, no
  throughput figure.
* **No bit-exactness figure for this route.** The parity evidence for a host-motion frame
  remains the earlier `native/dlss_gpu_host_motion_smoke.py` reset-frame byte comparison;
  this gate does not re-run it, and a rendered-pixel A/B between the two hosts on the
  cached-`.flo` export is still open work.
* **No caller moved in the GUI.** `gui.py` still builds its session through
  `gui.py:393`/`:444` with no `dlss_backend` to name, so the window's cached-`.flo` export
  (`gui.py:1345`, provider at `gui.py:1353`) is still a host-pointer frame and still runs on
  `dlssnr_host.dll`. What changed is that such a job *can* be routed: the missing thing is
  the choice, not the entry point or the core's willingness.
* The gate's `readback_motion()` calls are D2H copies this gate asked for, off the
  production path, as the wrapper's own docstring says.
* A `.flo` stored at a different extent than the session is resampled to it by the provider
  with `cv2.INTER_LINEAR` and its vector *magnitudes are not rescaled* — long-standing
  cached-route behaviour, unchanged here and not exercised by this pairing beyond the
  2:1 resample the gate ran through. Whoever moves the GUI's export should decide that on
  purpose rather than inherit it.
* No Playwright run: this section touches no UI, no HTTP surface and no frontend file.

## The desktop GUI's video preview on both GPU host motion entries

*The `gui.py` line numbers in this section and in "Current boundary and next work" below
are the current ones. The earlier sections cite the file as each was written, and the file
has grown since — `get_dlss_session` was at `:393` when the still-image section was
written, at `:402` at `92833e3`, and is at `:460` now — so following an old number lands
somewhere nearby and not on the definition.*

The window is the place this document kept naming as untouched: every video frame it
drew asked `dlss_engine.Live` whatever the guidance control said, and an unguided frame
was handed a `np.zeros((h, w, 2))` plane for the old host to ignore. Both halves of that
are gone. A `guidance_mode=0` preview frame now runs on
`dlss_gpu_host_process_zero_motion` and no motion plane is built at all; a
`guidance_mode=1` frame runs on `dlss_gpu_host_process_host_motion` and is handed the
cached `.flo` itself. The preview moved **without a new selector**, because the entry the
previous sections kept calling the missing piece now exists:
`App.video_preview_kind(settings)` (`gui.py:450`) reads the entry straight off the
guidance control that was already in the panel. What did not move is the window's video
export, which is the next gate and is still on the old host.

Wiring, all of it inside `gui.py`:

* `_PreviewWorker._live_dlss_image` (`gui.py:89`) is the route. It resolves the kind,
  then holds `app._dlss_lock` across **both** the session lookup and the entry call
  (`gui.py:135`), because the two kinds a preview can name now live on different hosts
  and a lookup that released the lock would let another route take the slot and close
  the session this frame is about to draw on. The choice is one expression
  (`gui.py:142`): the plane read from disk goes to `process_host_motion`, nothing goes
  to `process_zero_motion`. A `.flo` cached at another extent is resampled to the frame
  size before the call (`gui.py:130-131`), which the entry requires and which the old
  host never did.
* `reset` (`gui.py:141`) is a seek, a jump, **or** a session that is not the object the
  previous frame drew on. The second clause is new and closes a hole that predates the
  migration: another route could rebuild the cached session underneath the worker and
  it would have continued a temporal sequence on a host with no history.
* The GPU host raises where `dlssnr_host.dll` returned `None`, so a failed frame is now
  named rather than merely absent. The worker keeps `last_error` (`gui.py:60`), clears
  it per request (`gui.py:101`), sets it and drops the preview cache on failure
  (`gui.py:147`), and `_render` puts it on the canvas behind a `DLSS 预览失败：` prefix
  (`gui.py:271`) instead of the "generate flow first" hint, which means nothing when
  the failure was elsewhere.
* `App._GPU_HOST_KINDS` (`gui.py:432`) is one table of kind → capability, refusal and
  ready line, and `_open_gpu_live` (`gui.py:524`) — which replaces the image-only
  `_open_image_zero_motion_live` — serves all three GPU kinds from it. A host that does
  not expose the entry is closed and refused before frame zero (`gui.py:540`); drawing
  the frame on `dlssnr_host.dll` instead would report a GPU-host run that did not
  happen, since which host made a picture is invisible in the picture.
* Reuse (`gui.py:498-504`) is same kind, same `_w`/`_h`, and for a GPU kind the same
  `_gpu_options_key` — the host bakes its ten options at open and has no live-update
  entry, so `session.update(settings)` stays on the legacy arm only.
* `_video_task_kind` (`gui.py:316`, enforced at `gui.py:490`) is the new guard, and the
  migration is the reason it exists. While preview and render shared one kind, a
  preview frame that slipped past `pause()` was merely wasted work; with the preview on
  the GPU host the same frame would close the `dlssnr_host.dll` session the render loop
  or the stream export is mid-frame on and leave it calling a closed handle for the rest
  of the run. `_run_dlss_gui` (`gui.py:1217`, claims at `:1236`, releases in `finally`
  at `:1272`) and `_do_dlss_stream_export` (`gui.py:1420`, claims at `:1425`, releases
  at `:1484`) own the slot on the host they opened, and a preview that names the other
  one drops its frame. The pre-existing `_image_batch_running` guard (`gui.py:483`) and
  the still-image paths are unchanged, and `close_live()` (`gui.py:79`) now also drops
  the worker's reference to the session it no longer has.
* A video frame is still **not** padded to a multiple of 8, as it never was: a still
  image is, because the window pads and trims one, while `dlss_gpu_host_open` requires
  only nonzero dimensions, so the preview session is opened at the decoded frame size.

**Evidence — no GPU.** `tests/unit/test_gui_preview_gpu_host.py` is new: **29 passed**.
It drives a real `gui._PreviewWorker` against an `App` built with `__new__` and both
engines replaced by fakes that validate the arguments the real entries validate, and it
runs each frame on a thread of its own. It pins: the routing in both modes with the
other host's entries as traps (`dlss_engine.Live` raises a `BaseException`, because the
worker turns any `Exception` into a reported failure and a swallowable trap would prove
nothing); that an unguided frame allocates no CPU zero plane at all, by spying
`np.zeros`; that the cached `.flo` object reaches the host untouched, and that one
stored at another size arrives at the session extent; reset on a seek, on a jump, and on
a session someone else rebuilt; reuse across a sequence and a rebuild per geometry and
per baked option; the guidance flip rebuilding rather than sharing a slot; the preview
cache invalidating on a refresh and a settings change; that a refused or failing frame
is reported on the canvas, clears the cache and forces a reset on the retry; that a
missing capability, a missing `dlss_gpu_host.dll` and an ABI-2 host are each refused with
`dlssnr_host.dll` never constructed; that an image batch and a running video render each
own the slot and the preview drops its frame instead of evicting them; that the render
loop, the stream export and the still-image path keep the hosts they had; that teardown
closes whichever session is cached; that `optical_flow` is neither imported nor entered;
and that a paused worker draws nothing until the batch ends. Reaching 29 required one
correction in the test file, not in `gui.py`: two cases consumed the single failure the
`FailsOnce` fake offers before asserting on it, and a flip test guided a frame whose
`.flo` it had never written.

Two harnesses needed one line each (`tests/unit/test_gui_image_zero_motion.py::_app`,
`scratch/_gui_image_zero_motion_gpu_gate.py::_app_shell`): an `App` built with `__new__`
does not run `__init__`, so the new `_video_task_kind` field was simply missing. With it,
the pre-existing image suite is **28 passed** unchanged, and the six files that share the
session contract — the new one, the two image/preview routing files,
`test_preview_export_parity.py`, `test_export_cpu_flow_gpu_host.py` — are **128 passed**
in 9.15 s. The whole unit directory is **637 passed** in 228.19 s; it was 608 before this
file existed.

**Evidence — real GPU.** `scratch/_gui_preview_gpu_host_gate.py` is new and is the
window's video preview on hardware: **28/28 checks passed, twice**, on an
`NVIDIA GeForce RTX 5060 Ti` with driver 617.14, against a private working copy of
`bench_out/fixture_4k_gpu15.mp4` (3840x2160). It builds an `App` with `__new__` around a
real `gui._PreviewWorker` — whose thread starts, as in the window — and asks for each
frame the way the canvas does: `worker.request(...)` on the queue, the answer taken off
`post_ui`. It wraps, not replaces, the real `dlss_gpu_engine.Live`, and sabotages
`ctypes.CDLL` from outside so `dlssnr_host.dll`, `nvof_bridge.dll`, `nvofapi64.dll` and
`nvofnvml64.dll` raise on load while `dlss_engine.Live` raises on construction. Every
CUDA-descriptor entry, `process` and `update` raise too. Four frames, two sessions, both
at 3840x2160:

| guidance | frames | entry that answered | reset flags | host's own stage / device-colour / output-D2H per frame |
| --- | --- | --- | --- | --- |
| 0 | 0, 1 | `process_zero_motion` | `True, False` | 1 / 0 / 1, 33,177,600 bytes back |
| 1 | 2, 3 | `process_host_motion` | `True, False` | 1 / 0 / 1, 33,177,600 bytes back |

One geometry, two sessions: the kind flip rebuilt and the sequence did not, which is the
only way those two numbers can come out this shape. The motion is the clip's own — a
Farneback estimate between each frame and the one before it, written as the `.flo` files
the window reads — and the plane the host was handed digests (`sha256`, first 16 hex) to
the same value as the payload of the file on disk: `1c9727e7bdf9735c` and
`c5a90755bd8ed1db`, contiguous `float32 (2160, 3840, 2)` at row pitch 30720, per-axis
mean magnitudes 6.64/3.13 and 5.99/2.71 pixels, 1,013,302 and 901,534 elements at or
above half a pixel. `readback_motion()` after each guided frame shows that plane bound:
elementwise agreement within float16 to a worst-case ratio of 0.485 of tolerance, and a
nonzero element count of 2,481,579 and 2,368,041 against the 2,494,864 and 2,377,792 the
caller's own array keeps once narrowed to half precision — the fixture's flat regions
produce vectors below half-precision range, and they are zero on the card exactly as
they are on the CPU, which is a sharper agreement than "not zero". Both runs returned
identical counts and identical digests. `pipeline.read_flo` was called twice in the whole
gate, for `000002.flo` and `000003.flo` only: the unguided frames read no flow. Every
frame came back `uint8 (2160, 3840, 4)` from the host and `(2160, 3840, 3)` BGR to the
worker, with a per-frame mean absolute difference from its own source frame of
35.7–36.5 levels and a canvas result at the 480x270 the widget asked for. Alpha was 255
and every input contiguous. The blocked-load list is empty, the four libraries above are
absent from the process by `GetModuleHandleW`, `dlss_gpu_host.dll` is present,
`optical_flow._ENGINES` stayed empty with a zero execute delta, the window logged one
`[预览 DLSS] GPU 宿主零运动会话已就绪` and one `…运动向量会话已就绪` line, and teardown
closed both sessions and left `_dlss_session`, `_dlss_session_kind` and
`_video_task_kind` empty. `process_zero_motion` answered in 44.6–52.5 ms and
`process_host_motion` in 59.9–70.7 ms; those are liveness records, and the two runs
differ in them and only in them. The record is
`bench_out/gui_preview_gpu_host_gate.json` with stdout in
`bench_out/gui_preview_gpu_host_gate.log`, and it carries its own card and driver rather
than a hardcoded label.

Two things the run observes and does *not* claim. It is not parity: `dlssnr_host.dll` is
refused in this process, so the window's *video* output equivalence across the migration
is **not measured**, and nothing here says a preview looks the way it used to. And the
`.flo` files are Farneback's, not NVOF's — the clip's own pixels, cached in the format
the window reads, but a different estimator; a real NVOF plane reaching this entry is the
service's guided-preview gate, not this one.

**Limitations of that evidence.** Four frames is a correctness sample; desktop-GUI
preview performance is `NOT_CHARACTERIZED`, and no rate is quoted. The gate drives the
worker and the session factory, not Tk: no root window, no `_collect_settings` (the
settings dict is written out by hand, so a key the panel would add is not covered here),
no button handler and no `display_view()`. One geometry, one clip, one preset and one
`intensity`, with `guidance_mode` as the only thing moved, so the ten baked options
still stand or fall on the one reuse and one rebuild rule each that the unit file pins.
The slot-ownership guard is never exercised concurrently — the gate runs no render and no
export alongside the preview, so `_video_task_kind` is unit-tested only, as is the
pause/resume pairing around a batch. A frame whose `.flo` is missing, a cancelled or
failing source, a seek backwards mid-sequence on hardware, a video shorter than the flow
cache, a portrait or extreme-aspect clip, a second monitor and a window resized under a
live preview are all outside this sample. And the window's *export* still runs on the old
host, so this closes the preview half of the GUI and not the window.

## The desktop GUI's streaming video export on both GPU host motion entries

This is the window's last video task that named `dlssnr_host.dll`, and it moved without a
new native symbol, without a pipeline change and without a `dlss_backend` control. Every
piece the route needed already existed and had already been gated on hardware: the GPU
host's `process_zero_motion` and `process_host_motion` entries, and
`ProductionExportCore`'s willingness to run a `CachedFloFlowProvider` — a CPU plane of
float32x2 vectors read off disk — on a session whose host stages that plane itself. What
was left was the call site, and the call site turned out to hold three defects of its own,
each of which is now a test:

* `App.video_export_kind` (`gui.py:477`) is the selector. It reads the entry off the
  guidance control exactly as `video_preview_kind` (`gui.py:466`) does — `guidance_mode=1`
  asks for the host-memory plane, anything else asks for the resident zero texture — and it
  is a *second* function only because the session slot is shared and named by kind.
* `_GPU_HOST_KINDS` (`gui.py:432`) gained `export_zero_motion` and `export_host_motion`
  (`gui.py:453`, `gui.py:458`): the same capability probes as the preview's two kinds, the
  same "rebuild the DLL" refusal shape, and an export-specific ready line. Borrowing the
  preview's names would have been the bug: `_video_task_kind` (declared `gui.py:316`,
  enforced `gui.py:521`) refuses a request that names a *different* kind, so an export
  running under the preview's name would have handed the slot to a preview frame that
  slipped past `pause()` — and that frame would have reset the temporal sequence the encoder
  is being fed from, or closed the session it is being fed through.
* `_do_dlss_stream_export` (`gui.py:1452`) claims the slot at `gui.py:1467-1468`, **before**
  `get_dlss_session`, so no other route can take it between the claim and the open; names
  the kind at `gui.py:1462`; hands `flow_provider` only when guidance is on
  (`gui.py:1480-1481`); and passes `job_settings` (`gui.py:1482`) instead of nothing.
  Before this change the call site passed a `ZeroFlowProvider` for an unguided run and no
  `job_settings` at all, so `ProductionExportCore` read `guidance_mode=0` whatever the panel
  said and the guidance control moved nothing on this path.
* Handing `None` on the unguided arm is not a simplification, it is the route's
  precondition: the core refuses *any* provider on `gpu_zero_motion`, because the host
  answers motion from the texture it bound at open. The unit file pins the sharper version of
  that claim — an unguided run allocates no `np.zeros((h, w, 2))` plane anywhere.
* What the window hands the pipe is preserved because the export core's own behaviour depends
  on it: no `out_path` and no `transcode_settings`, so the destination stays
  `<dir>/<stem>_dlss.mkv` and the encoder chain stays the shipped default; `cq` and
  `with_audio` still come from `_export_crf()` and `_export_audio()`; `lock` is still
  `self._dlss_lock`, the same lock the session bookkeeping takes; `frames`/`fps` still come
  from the panel; `cancel` is still the live `_is_cancelled` predicate and not a snapshot.
* A cancelled run is now reported as an abort. The pipe raises `JobCancelled` rather than
  returning, and the only handler here was `except Exception`, so a Stop click used to
  produce a 导出错误 dialog and a traceback for a failure the operator had asked for. The
  arm at `gui.py:1524` says 已由用户中止, and the `finally` at `gui.py:1535` still resets the
  bar, releases the slot (`gui.py:1538`) and resumes the worker (`gui.py:1539`).
* The run now says which host answered (`gui.py:1503-1508`): `[流式导出 DLSS] 宿主入口
  <entry>，运动来源 <transport>`, read off the core's own diagnostics rather than from the
  routing the call site intended, because which host made a file is invisible in the file.

Unmoved, deliberately: `_run_dlss_gui` (`gui.py:1248`), which still claims
`kind="legacy"` at `gui.py:1271` and — apart from the stale `--selftest` probe at
`gui.py:1588`, named in the boundary section below — is the window's only remaining
old-host caller; the export-only mux (`gui.py:1541`), which builds no session at all; and
every image and preview path.

**Evidence — no GPU.** `tests/unit/test_gui_stream_export_gpu_host.py` is new: **29
passed**. It drives the real `_do_dlss_stream_export` on a thread of its own against an
`App` built with `__new__`, with both engines replaced by fakes that validate what the real
entries validate (contiguous uint8 RGBA at the session extent, opaque alpha, a contiguous
float32 HxWx2 plane at row pitch W·8, an `out=` buffer the host must write into) and
`dlss_engine.Live` replaced by a factory raising a `BaseException`, because the export
turns any `Exception` into a reported failure and a swallowable trap would read as a run
that merely degraded. Most cases run the **real** `pipeline.export_dlss_stream` over a real
pipe with a `sys.executable` stand-in for the encoder, so "the frames the host rendered are
the bytes that were delivered" is checked by file size (N·W·H·3) rather than asserted; a
smaller set pins what the window hands that function, because that is the contract this
stage had to preserve. It pins: the routing in both modes with the other entry as a trap;
the cached `.flo` vectors per frame, and a plane cached at another extent arriving at the
session extent; `video_export_kind` for both modes; that an unguided run allocates no CPU
motion plane; a missing `.flo` degrading and being reported; the slot claimed before the
session opened; reuse across two runs with `reset_flags == [True, False, False] * 2`; a
rebuild on the guidance flip and on a baked-option change; each of the three refusals
(ABI-2 host, no host-motion entry, no `dlss_gpu_host.dll`) with the old host never
constructed and the bar, slot and worker restored; a failing frame reported once and as an
error; the recorded `live`/`lock`/`cancel`/`cq`/`with_audio`/`frames`/`fps` identity
contract and the absence of `out_path` and `transcode_settings`; progress fields; the
preview parked for the run with all three other kinds refused while the export's own kind
re-enters; a preview frame after the run replacing the session; an image batch holding the
slot; a `JobCancelled` abort with no error dialog; a Stop flag set mid-run stopping the real
export after exactly one frame; the symmetric guard (the export refuses to evict a running
legacy render); and that the cached-frame render still opens the old host and still gets its
live `update` on reuse.

Four mutations of `gui.py` were run against that file, restored byte for byte each time
(md5 verified), because a routing suite that passes both ways proves nothing: naming
`kind="legacy"` in `video_export_kind` failed **25** of the 29; handing a `ZeroFlowProvider`
on the unguided arm failed **9**; dropping the slot claim failed **2** (the two ownership
tests, and only those); dropping `guidance_mode` from `job_settings` failed **7**.

The six files that share the session and export contract — this one,
`test_gui_preview_gpu_host.py`, `test_gui_image_zero_motion.py`,
`test_preview_export_parity.py`, `test_export_cpu_flow_gpu_host.py`,
`test_export_zero_motion_pipe.py` — are **136 passed** in 9.68 s. The whole unit directory
is **666 passed** in 219.20 s; it was 637 before this file existed.

**Evidence — real GPU.** `scratch/_gui_stream_export_gpu_host_gate.py` is new: **34/34
checks passed, twice**, on an `NVIDIA GeForce RTX 5060 Ti` with driver 617.14. It presses
the button, not the pipeline: it builds an `App` with `__new__` around a real
`gui._PreviewWorker` whose thread is running, and calls `App._do_dlss_stream_export()` on a
worker thread of its own. `ctypes.CDLL` is sabotaged from outside so `dlssnr_host.dll`,
`nvof_bridge.dll`, `nvofapi64.dll` and `nvofnvml64.dll` raise on load, `dlss_engine.Live`
raises on construction, and every CUDA-descriptor entry plus `process` and `update` raise.
`dlss_gpu_engine.Live` is wrapped, not replaced. Both runs use the same `App`, so the second
is the guidance flip an operator makes between two clicks.

| guidance | frames | entry that answered | reset flags | host stage / device-colour / output D2H, per frame | host ms |
| --- | --- | --- | --- | --- | --- |
| 0 | 5 | `process_zero_motion` | `True, False ×4` | 1 / 0 / 1 × 33,177,600 bytes | 37.1–44.0 |
| 1 | 5 | `process_host_motion` | `True, False ×4` | 1 / 0 / 1 × 33,177,600 bytes | 43.6–60.2 |

Two sessions, one geometry (3840x2160 each), so the flip rebuilt and nothing else did. The
motion is the known cache: the five `.flo` files `scratch/_cpu_flow_export_flo_prep.py`
prepared at the source extent, hardlinked into each working copy's own `<stem>_flow` so the
window's naming convention is what finds them. `pipeline.read_flo` was called zero times in
the unguided run and five times, in frame order, in the guided one. Each plane the host
motion entry received digests to the payload of the file on disk —
`4935700cc5284b55`, `67db3560a4943b34`, `daee15acc89b883e`, `4ef2e17153c5bd6d`,
`1bcd53ba66d0ca2b` — contiguous `float32 (2160, 3840, 2)` at row pitch 30720, and all five
share one buffer id, so the run uploaded through one reused plane. Frame 0's cached plane is
exactly zero, as it is in that cache; frames 1-4 carry the clip's own displacement at
abs-mean 8.22–11.43 px, abs-max 37.7–49.7 px, 11.0M–12.1M of 16,588,800 elements at or above
half a pixel. `readback_motion()` after every frame agrees with the plane that frame was
handed, at a worst-case float16 tolerance ratio of **0.48**, and the nonzero element count
the card reports (16,588,779–16,588,797) is the caller's own plane narrowed to half precision
(16,588,789–16,588,799); the per-axis means it reads back (x 1.81–2.75, y 14.2–20.4 px) are
that motion, not a flat stand-in. Every unguided frame read back an all-zero motion texture.

The rest of the claims are read off the run rather than asserted from the code: the exact
route line the operator saw (`…process_zero_motion，运动来源 resident_zero_texture` and
`…process_host_motion，运动来源 host_plane_upload`), the two `[导出 DLSS] GPU 宿主…会话已就绪`
lines, `subprocess.Popen` recorded so the encoder command line is this window's own default
chain (`-f rawvideo -pix_fmt bgr24 -s 3840x2160 -r 29.970030 -i - -i clip.mp4 -map 0:v:0
-map 1:a:0? -c:v av1_nvenc -preset p7 -tune hq -multipass qres -rc:v vbr -cq:v 24 -b:v 0 -g
60 -pix_fmt yuv420p -c:a copy -shortest clip_dlss.mkv`) and its output is the default
destination beside the clip, ffprobe on both finished files (av1, yuv420p, tv range,
3840x2160, five frames, unique PTS, one aac stream), one `导出成功` line and one info dialog
per run with no error dialog and no degrade line, the worker's `pause`/`resume` pairing with
`_paused` true on every one of the five progress ticks, `_video_task_kind` back to `None`
and the bar reset once per run, both sessions closed and the slot empty at teardown, the
blocked-load list empty with all four libraries absent by `GetModuleHandleW` and
`dlss_gpu_host.dll` present, `optical_flow._ENGINES` empty with a zero execute delta, and
nothing left in the workspace but the declared clips, caches and outputs. The two runs agree
on every one of those numbers and on the encoded byte counts (1,683,210 and 1,669,082); they
differ only in the milliseconds above. The record is
`bench_out/gui_stream_export_gpu_host_gate.json` with stdout in
`bench_out/gui_stream_export_gpu_host_gate.log`, and the first run's record is
`bench_out/gui_stream_export_gpu_host_gate_run1.json`; it carries its own card and driver
rather than a hardcoded label.

**What the run does not claim.** It is not parity: `dlssnr_host.dll` is refused in this
process, so nothing here says the exported picture looks the way it used to, and the
rendered-pixel A/B this document has kept asking for on this pairing is still outstanding.
And it is not a transfer reduction. The colour the host staged and read back is
33,177,600 bytes per frame in each direction at 3840x2160 — the same two copies the old host
made — and the decode and encode pipes are both still there, recorded as
`-hwaccel cuda … -pix_fmt nv12` in and `rawvideo`/`bgr24` out. What changed is who runs the
frame; the numbers only became countable because the host that now runs it can count them.
`export_backend` stays `legacy` for this call site: the `gpu_transfer` chain is a different
stage's work and the GUI does not ask for it.

**Limitations of that evidence.** Five frames per run is a correctness sample; desktop-GUI
export performance is `NOT_CHARACTERIZED` and no rate is quoted. The gate drives the export
method and the session factory, not Tk: no root window, no button handler, no
`_collect_settings` (the settings dict is written out by hand), and `_export_crf` /
`_export_audio` stand in for the panel at the shipped defaults (24, audio on), so a key the
panel would add is not covered here. One geometry, one clip, one preset and one `intensity`,
with `guidance_mode` as the only thing moved, so the ten baked options still stand or fall on
the reuse and rebuild rules the unit file pins. The slot guard is never exercised
concurrently on hardware — nothing renders or previews alongside these runs — so
`_video_task_kind`, the pause/resume pairing, the refusals and the cancel-mid-run path are
unit-tested only. A missing `.flo` in the real cache, a decode failure, a source shorter
than the cache, a portrait or extreme-aspect clip, a working scale other than 1.0 (which the
window never asks for), a second monitor and a multi-pass or `gpu_transfer` export are all
outside this sample. The flow planes are OpenCV DIS's, not NVOF's. And the window's
*cached-frame* render — the PNG loop the export-only mux consumes — is still on the old host,
so this closes the streaming half of the GUI's video tasks and not the render.

## The desktop GUI's cached-frame video render on both GPU host motion entries

The window's third video caller of the same two GPU host entries, and it moved with the same
three absences as the two before it: no new native symbol, no pipeline change, no
`dlss_backend` control. `gui.py` is the only source file that changed — five hunks, **89
insertions and 41 deletions**, md5 `ea2e0a19863f39fe5020ceaf7b49caa5` — because everything
the route needs had already been gated on hardware for the preview and the streaming export.
What the audit of `_run_dlss_gui` (`gui.py:1284`) found was that the route needed nothing new
and that the loop carried **four defects** the old host had been hiding. The first two bullets
below are the route; the four after them are the defects, each now a test:

* `App.video_render_kind` (`gui.py:506`) reads the entry off the guidance control exactly as
  `video_preview_kind` (`gui.py:482`) and `video_export_kind` (`gui.py:493`) do, and
  `_GPU_HOST_KINDS` (`gui.py:432`) gained `render_zero_motion` (`gui.py:469`) and
  `render_host_motion` (`gui.py:474`): the same two capability probes, the same "rebuild
  `native/dlss_gpu_host.dll`" refusal shape, and a ready line that names the task —
  `[DLSS 渲染] GPU 宿主零运动会话已就绪` / `…运动向量会话已就绪`. Named apart for the same
  reason the export was named apart from the preview: the one session slot is keyed by kind
  and refuses a request that names a different one (`gui.py:551`), so a render that borrowed
  the export's name would let a single session serve two loops that each believe they own its
  temporal sequence.
* The slot is claimed under that kind at `gui.py:1303-1304`, **before** `get_dlss_session`
  (`gui.py:1306`), and released in the `finally` at `gui.py:1351-1352`. Another route can take
  `_dlss_lock` between those two holds, but it cannot take the slot: it reads
  `_video_task_kind` already set and is refused (`gui.py:551`).
* `_guidance_inputs` (`gui.py:1251`) now returns `(rgba, flow, degraded)` and produces **no
  motion plane at all** when guidance is off. At 3840x2160 the array it used to build per
  frame — `np.zeros((h, w, 2), np.float32)`, 33,177,600 bytes — filled an argument slot the
  zero-motion entry does not have; on this host the unguided frame's motion is the texture it
  bound at open. Guidance on keeps the cached-`.flo` semantics exactly as they were: the same
  `<dir>/<stem>_flow/{idx:06d}.flo` convention, the same resize to the extent the frame is
  drawn at, the same zero plane for a frame with no cache or one that will not parse. The
  helper only adds the flag, so the loop can say what it actually fed.
* The degrade report stopped claiming a future it could not see. The old code announced, once
  per run through a flag on `self`, `第 {idx} 帧起未检测到光流数据，自动平滑降级为无光流模式继续处理`
  — a statement about *every later frame* that nothing in the loop supported, since a cache
  can come back on frame N+3 — and the count of degraded frames was never kept. Now the first
  degraded frame is named (`gui.py:1324`) and the run closes with the number of frames it fed
  zero vectors (`gui.py:1338-1341`), in wording that describes them as what they were:
  `这些帧以零运动向量送入渲染`, not a guided result.
* The reset flag was wrong in a way the old host made invisible. The loop tracked
  `first_proc`, so exactly one frame per run asked for a new temporal sequence. A render that
  resumed past cached frames (断点续算) therefore joined the sequence of the frame before the
  hole, and a second run into a session the slot kept continued a sequence the *previous* run
  had accumulated on the card. `reset = rendered_through is None or idx != rendered_through + 1`
  (`gui.py:1331`) asks for a new sequence whenever the frame drawn is not the successor of the
  frame this session drew — the rule the preview and the streaming export already used.
* A failed frame now ends the run and says so. The old loop wrote the PNG only
  `if o is not None`, so a host that returned `None` left a **hole** in the sequence the mux
  reads while the progress bar still advanced; `dlss_gpu_engine` raises instead, the handler at
  `gui.py:1345` reports it, and because the result is guaranteed the write at `gui.py:1336`
  needs no `o.shape[-1] == 4` conditional over the colour conversion any more.

Preserved because the mux depends on it: the PNG directory and names
(`pipeline.out_dirs(video)[1]`, `{idx:06d}.png`), BGR at full native size through
`pipeline.imwrite_u`, the skip test (`exists and size > 0`) that makes 断点续算 work, the
progress ticks for both rendered and skipped frames, the cancel check at the top of the loop
and 已由用户中止 in `_do` (`gui.py:1212-1215`), the 启动/完成 lines, `view_var.set("DLSS")`, and
the generate-only versus generate-then-mux choice itself — `_do`'s `_pending_export` arm
(`gui.py:1216`) and `_do_export_only` (`gui.py:1589`) both call the same
`pipeline.export_video` on the same PNG sequence and build no session, so neither needed a
line changed and neither got one.

Unmoved, deliberately: the export-only mux; the stale `--selftest` probe at `gui.py:1636`,
whose four-positional `Live.process` call (`gui.py:1638`) this stage's smoke never used and
which blocks nothing — it is named in the boundary section below and left as it was found; the
service's host-pointer defaults (`dlss_service.py:476`, `dlss_service.py:1072`), which still
read the host the operator named or did not name; and every image, preview and streaming-export
path. `get_dlss_session` still keeps its `kind="legacy"` default argument (`gui.py:518`); after
this stage no call site in `gui.py` uses it, so the factory's legacy arm at `gui.py:580` is
reached by no frame the window draws.

**Evidence — no GPU.** `tests/unit/test_gui_video_render_gpu_host.py` is new: **31 passed**.
It drives the real `_run_dlss_gui` against an `App` built with `__new__`, with both engines
replaced by fakes that validate what the real entries validate (contiguous uint8 RGBA at the
frame extent, opaque alpha, a contiguous float32 HxWx2 plane at row pitch W·8) and
`dlss_engine.Live` replaced by a factory raising a `BaseException` the window's
`except Exception` cannot swallow into a degraded run. It pins: both entries with the other
one as a trap; the cached `.flo` vectors per frame, and a plane cached at another extent
arriving at the frame size; `video_render_kind` for both modes, and that the three video
callers of these entries do not share a kind; that an unguided render builds no CPU motion
plane **and** reads no flow file although its cache is complete; a missing `.flo` and a
corrupt `.flo` each degrading like the other, announced once and counted at the end; cached
frames skipped with the run continuing past them; a hole in the middle starting a new temporal
sequence; a second run on the reused session starting one; a run that skipped every frame
rendering nothing; a guidance flip and a baked-option change each rebuilding; the PNG sequence
the mux reads, byte-identical to what the host returned; each of the three refusals (ABI-2
host, no host-motion entry, no `dlss_gpu_host.dll`) with the old host never constructed; a
frame the host fails ending the run; a frame the decoder **grew** shutting the run down
instead of drawing it wrong; the slot claimed before the session opened; the preview refused
the slot for the whole run and replacing the session after it; an image batch holding the slot
with the render reporting that; cancel mid-run and cancel before the first frame; and the
progress the render reports being the frames it walked.

Four mutations of `gui.py` were run against that file and `gui.py` restored byte for byte each
time (md5 verified), because a routing suite that passes both ways proves nothing: naming
`kind="legacy"` in `video_render_kind` failed **30** of the 31; making the unguided arm build
`np.zeros((h, w, 2))` and hand it to the motion entry failed **11**; dropping the slot claim
failed **2** (the two ownership tests, and only those); reverting the reset rule to "the first
frame this run processed" failed **1**. The four GUI files that share the session and slot
contract — this one, `test_gui_stream_export_gpu_host.py`, `test_gui_preview_gpu_host.py`,
`test_gui_image_zero_motion.py` — are **117 passed** in 5.47 s. The whole unit directory is
**697 passed** in 216.24 s; it was 666 before this file existed.

**Evidence — real GPU.** `scratch/_gui_video_render_gpu_host_gate.py` is new: **45/45 checks
passed, twice**, on the card the record reads off `nvidia-smi` itself:
`NVIDIA GeForce RTX 5060 Ti, 617.14`. It presses the button path, not the loop:
`app.run_worker("dlss")` → `_in_thread` → `_do("dlss")` → `_run_dlss_gui`, on the window's own
worker-thread bookkeeping, around a real `gui._PreviewWorker` whose thread is running, and then
presses `_do_export_only` separately for the mux. `ctypes.CDLL` is sabotaged from outside so
`dlssnr_host.dll`, `nvof_bridge.dll`, `nvofapi64.dll` and `nvofnvml64.dll` raise on load,
`dlss_engine.Live` raises on construction, and every CUDA-descriptor entry plus `process` and
`update` raise; `dlss_gpu_engine.Live` is wrapped, not replaced. Three presses rather than two,
because the third is the case this loop's own cache semantics create: `zero` (guidance off,
complete flow cache), `guided` (guidance on, frame 4's `.flo` deleted from the cache) and
`resume` (the same guided clip with `000002.png` deleted and the full `.flo` cache restored).

| guidance | frames | entry that answered | reset flags | host stage / device-colour / output D2H, per frame | host ms |
| --- | --- | --- | --- | --- | --- |
| 0 | 5 | `process_zero_motion` | `True, False ×4` | 1 / 0 / 1 × 33,177,600 bytes | 39.5–43.5 |
| 1 | 5 | `process_host_motion` | `True, False ×4` | 1 / 0 / 1 × 33,177,600 bytes | 45.7–57.5 |
| 1, resumed | 1 (frame 2 only) | `process_host_motion` | `True` | 1 / 0 / 1 × 33,177,600 bytes | 46.0 |

Eleven inferences, two sessions, both opened at 3840x2160 and both carrying the **same**
`baked_option_key` (`1, 0, 1.0, 1.0, 1.0, 1.0, 1, 0, 1.0, 1.0`): the flip rebuilt because
`guidance_mode` is not a baked option — guidance picks the entry, and the entry is the kind,
and the kind is what keys the slot. The close order says the rebuild sequenced itself the way
the factory intends: the zero-motion session was released when one session had *ever* been
opened, so the flip closed it before constructing the next, and the motion session was released
at teardown when two had, with no third opened when the resume press returned into it. Session 0
answered 5 `process_zero_motion` and no motion call; session 1 answered 6 `process_host_motion` —
the guided press's five and the resume's one — with 0 traps and 0 errors.

The motion is the known cache: the `.flo` files `scratch/_cpu_flow_export_flo_prep.py` prepared
at the source extent, hardlinked into each working copy's own `<stem>_flow` so the window's
naming convention is what finds them. `pipeline.read_flo` was called **zero** times by the
unguided press, whose cache was complete, four times in frame order by the guided press
(`000000`–`000003`, stopping at the hole) and once by the resume press. Every plane the
host-motion entry received is contiguous `float32 (2160, 3840, 2)` at row pitch 30720 and
digests to the payload of the file on disk — `67db3560a4943b34`, `daee15acc89b883e`,
`4ef2e17153c5bd6d` — plus `4935700cc5284b55` twice: frame 0, which that cache holds as exactly
zero, and frame 4, the frame with no cache, whose plane is the zero array the helper builds for
the degrade. The three dense frames carry the clip's own displacement at abs-mean 8.22–11.13 px,
abs-max 37.7–49.7 px and 11.02M–11.69M of 16,588,800 elements at or above half a pixel.
`readback_motion()` after every guided frame agrees with the plane that frame was handed at a
worst-case float16 tolerance ratio of **0.48**, the nonzero element count the card reports
(16,588,779–16,588,797) is the caller's own plane narrowed to half precision
(16,588,789–16,588,799), and the per-axis means it reads back match the caller's to
≈0.006 px (x 1.809→1.808, 2.222→2.221, 1.979→1.978; y 20.455→20.447, 14.217→14.212,
14.483→14.478). Both all-zero planes bound zero **inside the same run** as the dense ones,
which is what makes the agreement per frame rather than one upload sitting in a resource; every
unguided frame bound an all-zero motion texture.

The pixels are checked as well as the routing. Each press wrote five PNGs (one for the resume
press), each 8.96–9.53 MB, each decoding back to shape `(2160, 3840, 3)` `uint8` whose digest
equals the BGR the host returned for that frame — no stale buffer, since every frame of a press
is a distinct picture. Two of those comparisons are worth the bytes they cost: frame 0 of the
unguided press and frame 0 of the guided press produced the **same** picture
(result `5203ff4bfd7e…`, PNG `a4bd7c3b69eb…`) while frame 1 onward differ, so the resident zero
texture and an uploaded all-zero plane reach DLSS alike; and the resume press's re-render of
frame 2 (`33717047cfd1…`, PNG `71fa6e49b2b0…`) differs from the same frame drawn inside the
guided sequence (`ee4a70618342…`, PNG `11d44733e251…`), so the reset flag reached the output
rather than only the call. The four cached PNGs the resume press skipped were left byte for byte
as the press that drew them wrote them.

The rest is read off the run rather than asserted from the code: both ready lines naming the
session a render's, `subprocess.Popen` recorded so the **render press spawned no encoder at all**
— it is generate-only, and its only FFmpeg is `pipeline.iter_frames`' own auto decode route,
recorded here as `-hwaccel cuda -hwaccel_output_format cuda -i clip.mp4 -vf
scale_cuda=format=nv12,hwdownload,format=nv12 -f rawvideo -pix_fmt nv12 -` — while each clip's
mux ran the shipped default PNG→NVENC chain (`-f rawvideo -pix_fmt bgr24 -s 3840x2160 -r
29.970030 -i - -c:v av1_nvenc -preset p7 -tune hq -multipass qres -rc:v vbr -cq:v 24 -b:v 0 -g
60 -pix_fmt yuv420p` to `<dir>/<stem>_dlss.mkv`) and built **zero** sessions; ffprobe on both
finished files (av1, yuv420p, tv range, 3840x2160, five frames, unique PTS,
1,648,350 and 1,666,278 bytes); one 导出成功 line and one info dialog per mux with no error
dialog; the exact degrade pair the guided press showed the operator; `_video_task_kind` back to
`None` after each of the three presses; the bar reset once per press; `pause`/`resume` paired
with the worker `_paused` on all six progress samples of every press; a probe during each press
asking the window's real `get_dlss_session` for `gpu_zero_motion`, `gpu_host_motion`, `image`,
`legacy` and `export_host_motion` and getting `None` five times with the session count unchanged
and the render's own kind still holding the slot; alpha 255 and contiguous uint8 colour on every
frame; `out_identity` false on every frame, which is what a caller that passes no `out=` buffer
must produce; the blocked-load list empty with all four libraries absent by `GetModuleHandleW`
and `dlss_gpu_host.dll` present; `optical_flow._ENGINES` empty before and after with a zero
execute delta; both sessions closed and the slot empty at teardown; and nothing left in the
workspace but the declared clips, caches, frames and muxes. The two runs agree on every one of
those numbers, PNG digests and mux byte counts; they differ only in the milliseconds above. The
record is `bench_out/gui_video_render_gpu_host_gate.json` with stdout in
`bench_out/gui_video_render_gpu_host_gate.log` and the first run's record in
`bench_out/gui_video_render_gpu_host_gate_run1.json`.

**What the run does not claim.** It is not parity: `dlssnr_host.dll` is refused in this process,
so nothing here says the rendered picture looks the way it used to, and the rendered-pixel A/B
this document has kept asking for is now outstanding on **all three** of the window's video
callers rather than two. And it is not a transfer reduction. The colour the host stages and reads
back is 33,177,600 bytes per frame in each direction at 3840x2160 — the same two copies the old
host made — and this chain still pays what a streaming chain does not: a 12,441,600-byte/frame
nv12 decode pipe into CPU memory, a PNG encode per frame to disk, and a PNG decode per frame out
of it into a 24,883,200-byte/frame `rawvideo` encode pipe. What changed is who runs the frame and
what the frame's motion actually is; the numbers only became countable because the host that now
runs it can count them.

**Limitations of that evidence.** Five frames per clip and eleven inferences in total is a
correctness sample; desktop-GUI render performance is `NOT_CHARACTERIZED` and no rate is quoted.
The gate drives `_do("dlss")` on the window's own thread bookkeeping but not Tk: no root window,
no real widgets, no `_collect_settings` (the settings dict is written out by hand), and
`_export_crf` / `_export_audio` standing in for the panel at 24 and **off**. Audio off is a
sample artifact, not a claim: a five-frame trim's audio track runs 0.149 s and
`pipeline.export_video`'s `-c:a copy -shortest` truncates the *video* to it, so the mux was
handed a silent `-an` clip and the audio-bearing mux arm of `export_video` is not covered on
hardware by this gate — the streaming-export gate covered its own `-c:a copy -shortest` with one
aac stream. Neither file presses `_do`'s generate-then-mux arm (`gui.py:1216`): the gate pressed
generate-only and the standalone mux, and the unit file pins the PNG contract both arms read, so
that arm's *composition* is untested rather than changed. One geometry, one clip, one preset and
one `intensity`, with `guidance_mode` and the flow cache as the only things moved, so the ten
baked options still stand or fall on the reuse and rebuild rules the unit file pins. The slot
guard is never exercised concurrently on hardware — nothing previews or exports alongside these
presses — so `_video_task_kind`, the pause/resume pairing, the three refusals and both cancel
paths are unit-tested only. A source longer than its flow cache, a hole at frame 0, a decode
failure, a portrait or extreme-aspect clip, a working scale other than 1.0 (which the window
never asks for), a second monitor, the image-folder batch and a multi-pass or `gpu_transfer`
export are all outside this sample, and the flow planes are OpenCV DIS's, not NVOF's. The
`--selftest` probe is still on the old host and this stage neither used it nor fixed it.

## A service frame that names no host now names this one

*What moved here is a default, not an entry point.* Every entry the previous
sections validated already existed, already had its own hardware gate and already had
callers; what no caller did was reach for one without being told to. From this stage,
`native/dlss_gpu_host.dll` is what the service answers with when nothing named a host:
`DEFAULT_DLSS_BACKEND = "gpu_flow_d3d12"` at `dlss_service.py:42` feeds the settings
default at `:179`, the preview router at `:410` and the job resolver at `:344`, so the
default preview takes `gpu_zero_motion` or `gpu_host_motion` off `guidance_mode` and the
default job — on *either* export chain — opens its session through
`self._import_gpu_host().Live` at `:1146` instead of `dlss_engine.Live` at `:1148`.
`export_backend` is untouched as a choice: `auto` and `legacy` still pick the chain and
`gpu_transfer` still picks the device-resident one, which is why the two keys stay two
separate controls rather than collapsing into one.

**What stayed reachable, and by name.** `host_pointer` is still a host an operator can
pick and still the only thing that gets a frame onto `dlssnr_host.dll`: `_job_dlss_backend`
(`:344`) lets an explicit value win over the default in both directions, and
`_preview_live_kind` (`:422`) returns `"legacy"` for anything that is not
`gpu_flow_d3d12`. What the default moved *off* is omission: a client built before the key
existed, or a hand-written `POST /api/queue/add` with no `dlss_backend`, used to mean the
old DLL by accident and now means the host the project ships. `gpu_transfer`'s two
requirements are unchanged and still refuse by name — a non-NVOF flow engine at `:1078`
and a host-pointer session at `:1086`, the first of them now pointing the operator at the
host-motion entry ("若要用 DIS/RAFT 的 CPU 运动平面，请把 export_backend 设为 auto 或
legacy：该链路的 GPU 宿主有宿主内存运动入口可以直接吃它") rather than at the old host.
An unknown host is refused rather than read as the other one
(`_validated_dlss_backend`, `:328`, reached from `update_settings` at `:370` and from both
job paths), and an unbuilt GPU host raises a message that names the module, names the
setting and states in words that there is no automatic fallback
(`_import_gpu_host`, `:455`: "不会自动回退到 dlssnr_host.dll"). There is no on-disk
settings file and the queue is in memory, so there was no state to migrate; what a job
carries is resolved and *written back* at `add_job` (`:825`) so the queue row shows the
host it will run on, and re-read off the job rather than off the live setting at
`_execute_job` (`:1040`), so a job queued before the operator moved the control still runs
on the host it was queued with.

**The audit's one real hazard, and why it does not bite.** A method table cannot be the
capability check here, because one adapter class (`pipeline.GuidanceServiceFlowProvider`)
wraps *every* flow engine: it exposes `get_flow_device` for DIS exactly as it does for
NVOF, so a routing decision read off `hasattr` would send OpenCV DIS's motion to
`process_device_flow` and the first frame would either download vectors nobody asked for
or fail inside the bridge. `_flow_delivery` (`pipeline.py:2522-2528`) therefore asks the
provider, and the provider answers for its engine: `supports_device_flow`
(`pipeline.py:1135`) delegates to `optical_flow.OpticalFlowEngine.supports_device_bgr_flow`
(`optical_flow.py:1778`), which is true only for `engine_type in ("nvof", "nvidia")`
*and* only when `compute_flow_bgr24_device` is actually callable. A provider that answers
neither way is `host_plane`, and on the GPU host that routes to `process_host_motion`
behind `_require_host_motion_host` (`:2567`), which refuses a host that predates the entry
rather than quietly running the frame on the old one. `guidance_mode != 1` short-circuits
both checks to false, so a provider that will not be consulted cannot be mistaken for one
that will. This stage did not need a code change here — the delegation was already in the
partial diff — it needed a run that could tell DIS from NVOF by what the card did, which
is the gate below.

**Frontend.** `DEFAULT_SETTINGS` carries `dlss_backend: 'gpu_flow_d3d12'`
(`frontend/src/App.tsx:35`), and the export panel stopped writing a host on every chain
change: `dlssHostRequiredByExport` (`frontend/src/services/api.ts:14`) returns
`'gpu_flow_d3d12'` for `gpu_transfer` and `null` otherwise, `DLSSPage.tsx:351-352` applies
the key only in the first case, and `DeliverPage.tsx:638` / `:651` derive the same way when
a job is built, so an `auto` or `legacy` job reaches the queue with the *user's* host
choice — including the explicit `host_pointer` the DLSS-host select can still be moved to —
instead of a host the chain did not ask for. The panel copy and the select's display
fallback (`DLSSPage.tsx:1049`) match. One divergence is left standing on purpose:
`App.tsx:23` starts the renderer at `guidance_mode: 1` while the service defaults to `0`,
which is a pre-load flash only — `GET /api/settings` replaces the whole object — and
fixing a cosmetic default is not a reason to move a routing control.

**Unit tests.** 699 pass (`python -m pytest tests/unit -q`, 101.9 s), and the four files
whose contracts this stage moved pass 70 between them. The renames are the record of what
changed: `test_preview_zero_motion.py`'s `test_guidance0_default_backend_stays_on_legacy_host`
became `test_guidance0_explicit_host_pointer_backend_stays_on_legacy_host` and grew a new
`test_guidance0_default_backend_routes_to_the_gpu_host` beside it, because the old test was
pinning "nothing named means the old host" — which is precisely the behaviour this stage
removed — and the same file's host-flip and session-reuse cases now *name* `host_pointer`
to keep testing the pair they tested. `test_preview_host_motion.py`'s route table moved
`(None, "legacy")` to `(None, "gpu_host_motion")` and its flow-plane case is now
`test_explicit_host_pointer_guided_preview_still_gets_its_flow_plane`.
`test_export_backend_ui_contract.py` gained
`test_the_auto_and_legacy_chains_carry_the_default_gpu_host_to_the_queue` and reworded the
settings-POST case to assert the *conditional* write, so a future change that puts the host
back on every chain change fails. `test_preview_export_parity.py` names `host_pointer` in
its `_service()` helper: its fake Live is a `process`-only session and the parity it pins is
the residual backend's, not the entry's, so naming the host keeps that file answering its
own question.

**E2E.** `tests/e2e/13_gpu_transfer_optin.spec.ts` — 5 passed (57.8 s), including 14.5 on
real hardware. 14.1 now reads a default `auto` job as a GPU-host job, 14.2 asserts that
reverting the chain to `auto` reverts *only* the chain (`dlss_backend` stays on the GPU
host, which is the sentence the panel copy makes), and 14.5 drives the DLSS-host select from
its new default through a real preview (`kind=gpu_zero_motion`), a reload and an explicit
`host_pointer` (`kind=legacy`). 14.3 needed the honest version of itself: it used to prove
"gpu_transfer without its host is refused" by *omitting* the key, and omission now supplies
the host, so that job would have completed rather than failed and the case would have passed
its assertion on the wrong premise. It names `host_pointer` to reach the refusal, which is the
same claim with the word for the host in it — and the fact that leaving the word out no longer
triggers it is the evidence that the default moved.

**One short real routing gate, with the old host sabotaged.**
`tests/unit/default_gpu_routing_gate.py` (`bench_out/default_gpu_routing_gate.log`) replaces
`dlss_engine.Live` with a raiser before anything opens, so a wrong default is a failed job
naming the mistake rather than a frame quietly drawn by `dlssnr_host.dll`. The source is
`bench_out/default_gate_src.mp4`: 8 frames @ 120 fps, 1280x720, cut from
`tests/fixtures/fixture_120fps.mkv` — a routing gate, so it asks which entry answered, not
how fast. Three service jobs, each submitted the way a **client that predates the key**
submits: the service's own defaults with `dlss_backend` removed, so nothing in the job names
a host and only the default can answer. The record's `host_key_absent` is that assertion and
`resolved_host` is what `add_job` wrote back for it; each job then runs through
`DLSSService._execute_job` on the legacy FFmpeg pipe:

| job | entry answered | flow transport | bytes | ffprobe |
| --- | --- | --- | --- | --- |
| guidance off | `process_zero_motion` | `resident_zero_texture` | 504,901 | h264 1280x720 yuv420p, 8 packets |
| guidance on, DIS 1F | `process_host_motion` | `host_plane_upload` | 506,324 | h264 1280x720 yuv420p, 8 packets |
| guidance on, NVOF 1F | `process_device_flow` | `nvof_device_descriptor` | 510,225 | h264 1280x720 yuv420p, 8 packets |

All three report `dlss_backend_actual: gpu_flow_d3d12` and `export_backend_actual:
legacy_pipe` beside the `resolved_host` above, and the DIS row is the audit's answer: the
same provider class, the engine underneath refusing the descriptor, the frame taking the
upload entry. The preview half runs with the same sabotage and gets `kind=gpu_zero_motion` at
guidance 0 and `kind=gpu_host_motion` at guidance 1 with DIS, both `status=ok`, JPEGs of
59,310 and 58,419 bytes decoding back to 720x1280 and `_preview_live_error` null. The
first version of this gate spread the service's own settings into each job, which named a
host on the job's behalf and left its own "no host named" field both misread and unasserted;
the key is stripped from what it submits and checked in what it reports now, and the run
below is that version. 32 checks, 0 failures, `RESULT: PASS`.

**What the run does not claim.** It is 720p and eight frames, so it says nothing about a
scale that no earlier section's contract changed — the 4K claims belong to the gates above
it and were not re-run, because the routing they pinned is the routing this stage kept. It
is not a pixel A/B: the old host is refused in this process on purpose, so no number here
says the GPU host's picture matches `dlssnr_host.dll`'s, and that comparison is still
outstanding on every path that moved. It is not a `gpu_transfer` run — the device chain and
its cancellation stay under `tests/unit/gpu_transfer_export_gate.py` — and it is not a
performance sample; throughput after the default moved is `NOT_CHARACTERIZED`. The old host
is not exercised by this gate at all, which is the point of blocking it: what still runs
there does so because a caller named it, and those paths are covered by the unit files, not
by this run.

## A caller map for `dlssnr_host.dll`, taken at the load seam

Deletion-readiness turns on a count this document has never gathered in one place: how
many call sites can still put the old host's image into a process. An import is not that
event, so every reference below is sorted into one of three kinds — a name read, an
import, or a load — and loads are counted at `ctypes.CDLL`.

**The seam is one function.** `dlss_engine._load` (`dlss_engine.py:52-69`) holds the only
`ctypes.CDLL` on `HOST_DLL` (`:57`; path from `:25`, resolved by the frozen-aware
`_res_dir()` at `:17-21`). It is lazy, and three things reach it: `probe` (`:77`),
`run_dlss` (`:98`) and `Live.__init__` (`:150`, whose `_open` then calls `dlssnr_init` at
`:182`). Nothing else in the module touches `_lib`, so a reference to `dlss_engine` that
constructs no `Live`, calls no `run_dlss` and calls no `probe` cannot load the file.

**Two of those three routes are dead code.** `probe` and `run_dlss` each appear exactly
once in the tree, at their own `def`. The two names that read like callers are not:
`_run_dlss()` inside `pipeline.ProductionExportCore` (`pipeline.py:2721`) is a closure over
`self.primary_live`, whichever session somebody injected, and `gui.py:1211`'s
`_run_dlss_gui` is the window's own render entry. So in this whole tree the old host enters
a process by one event only: a `dlss_engine.Live` construction.

**The four construction sites, and what reaches them.**

| Site | Reached when | State |
| --- | --- | --- |
| `dlss_service.py:537` | `_preview_live_kind()` (`:410`) answers `"legacy"`, which after `:422`, the seeding at `:179` and the validation at `:328-342`/`:370-375` means exactly one thing: a preview whose settings say `dlss_backend="host_pointer"` | live |
| `dlss_service.py:1148` | `gpu_flow_backend` (`:1041`) is false — a job whose host `_job_dlss_backend` (`:344`) resolved to `host_pointer`, whether its own, the service's, or `DEFAULT_DLSS_BACKEND`, which is no longer that value | live |
| `gui.py:581` | `on_gpu_host` is false, i.e. `kind == "legacy"` — and no call site passes it. The window's four session entries are `gui.py:137` (preview worker, `video_preview_kind`), `gui.py:1087` (`"image"`), `gui.py:1307` (`video_render_kind`) and `gui.py:1518` (`video_export_kind`); every selector returns a GPU name in both guidance modes, and the `kind="legacy"` default at `gui.py:519` has no caller | unreachable |
| `gui.py:1636`, before this stage | `python gui.py --selftest` | migrated, below |

Both live sites are therefore one condition: a caller that says `host_pointer` out loud.
Nothing reaches them by omission any more, which is what `_validated_dlss_backend`'s
docstring (`:329-335`) is written to keep true — an unknown host name is an error out loud,
never a reading of "the other one".

**The export core's fourth arm is a call shape, not a dependency.** `pipeline.py:2735`
ends `ProductionExportCore`'s per-frame chain with `self.primary_live.process(rgba, flow,
...)` and `pipeline.py:2449` labels that route's transport `host_plane_legacy_session`.
`pipeline.py` imports no `dlss_engine` and never calls `ctypes.CDLL`: the DLL is already in
the process because `dlss_service.py:1148` put it there, and the core is being told how to
speak to it. Removing the host would retire that arm's only tenant, not an import edge.

**Reads of the name that are not reads of this file.** `DLSSNR_DLL` is
`nvngx_dlssnr.dll` — NVIDIA's runtime, not this repository's host — and it is the one
genuine import edge between the two wrappers: `dlss_gpu_engine.py:18` takes `DLSSNR_DLL`
and `DLSSInferenceError` from `dlss_engine`, hands the first to `dlss_gpu_host_open`
(`:165`) and shares the second so one `except` covers both hosts. `pipeline.py:44` keeps
its own copy of the runtime path; `pipeline.py:43` keeps a `HOST_DLL` that has no other use
in that file. `server.py:41`, `dlss_service.py:31` and `gui.py:37` import the module for
its exception class, for `get_log_tail`, or for nothing at all: the name appears once in
`server.py`, at the import.

**`get_log_tail` is a file read, and it now usually reads another host's history.**
`dlss_engine.py:37-49` tails `LOG_PATH` (`:27`), which only the old host writes — the path
is passed into `dlssnr_init` at `:78`, `:106` and `:182`. The GPU host takes no log
argument (`dlss_gpu_host_open(w, h, dlssnr_dll, options, &handle)`, `dlss_gpu_engine.py:123-125`)
and `native/dlss_gpu_host.cpp` opens no log file, so `dlss_service.py:925`, which attaches
`job["dlss_log_tail"]` to every failed job, attaches either `""` (no such file) or the tail
of the last `host_pointer` run that process performed. `server.py:427` passes it out,
`DeliverPage.tsx:1927-1931` renders it under a `[DLSS Log]` heading, and `:1945` folds it
into the copy-out diagnostic report — so a GPU-host failure report can carry the other
host's log, unlabelled. The strings at `dlss_service.py:471`, `:1158` and `:1169-1184` are
a different kind of reference: refusal and remedy text, and correct about the host they
name. `transfer_profiler.py:243` says only that the old ABI reports no transfer counter,
which is why that figure is `unavailable`.

**Frontend.** Three places, all of them the compatibility option this job keeps:
`frontend/src/types/index.ts:23` (`DlssBackend`), `DLSSPage.tsx:1054` (the `host_pointer`
`<option>`, whose copy at `:1057` already says the old host survives only for jobs that
still choose it and is not deleted in this stage) and `api.ts:11-14`
(`dlssHostRequiredByExport`, which answers `null` for `auto`/`legacy` so an export-chain
change cannot move a preview back onto the old host behind the operator).

**Native and packaging.** No C/C++ source references the DLL;
`native/dlss_gpu_host.h:95` and `native/dlss_gpu_host.cpp:1009` name it only to describe
the plane format the old `Live.process` contract took. `DLSS5Tool_standalone.spec:16-17`
ships `dlssnr_host.dll` and `nvngx_dlssnr.dll` and ships nothing from `native/` — no
`dlss_gpu_host.dll`, no `nvof_d3d12_interop.dll`, no CUDA runtime — while
`dlss_gpu_engine._LIBRARY` (`:21`) is `dirname(__file__)/native/dlss_gpu_host.dll` with no
`sys.frozen` branch, unlike `dlss_engine._res_dir()`. The packaged window therefore has no
GPU host to reach and still carries the old one. That is packaging work, recorded in the
boundary below and deliberately untouched here.

**Who else would notice a deletion.** Eleven scripts construct the real `dlss_engine.Live`:
the four native smokes (`native/dlss_gpu_host_smoke.py:15`,
`native/dlss_gpu_host_motion_smoke.py:134`, `native/dlss_gpu_zero_motion_smoke.py:212`,
`native/gpu_flow_export_smoke.py:40`), two scratch probes
(`scratch/_legacy_guidance_mode_probe.py:10`, `scratch/_torch_rounding_parity.py:41`),
four uncollected hardware scripts under `tests/unit/`
(`benchmark_4k_scenarios.py:72`, `benchmark_e2e.py:157`, `collect_transfer_ledger.py:126`,
`gpu_transfer_export_gate.py:115`) and one collected test file. That last is the only
pytest-time dependency, and it is a soft one:
`tests/unit/test_cancellation_e2e.py:40` opens a legacy session inside a module fixture
that turns a failure into `pytest.skip` (`:57-62`), so the DLL's absence skips that file
rather than reddening it. Every other `dlss_engine.Live` match in `tests/unit/` is a fake
or a trap — the GUI files replace it with a factory that raises, which is how "the old
host was never called" is a failure mode there rather than a line in a log.

**The count.** Two reachable `dlss_engine.Live` constructions in production, both gated on
an explicitly named `host_pointer`; one unreachable arm of the window's own factory; no
load from `probe` or `run_dlss`, because nothing calls either; and, after the section
below, none from `gui.py`. Nothing in this tree needs the old host to start, to route a
frame, or to serve a request that names no host. What still needs it is a caller that asks
for it by name, and the comparison scripts above that hold it up as the reference.


## The last probe in the window moves to the zero-motion entry, and says what answered

`python gui.py --selftest` was the only reachable thing left in the desktop window that
asked `dlssnr_host.dll` for a frame, and it had not worked since depth guidance was
removed. `dlss_engine.Live.process` takes
`(self, rgba, motion, reset=False, raise_on_error=False, out=None)` (`dlss_engine.py:219`);
the probe handed it a colour frame, a motion plane **and a depth plane** positionally
(`gui.py:1638-1639` before this stage) plus a `reset=True` keyword. The depth array bound
to `reset`, the keyword then collided with it, and every run ended in
`TypeError: Live.process() got multiple values for argument 'reset'` — confirmed against
the live signature without touching a DLL. The probe had therefore drawn no frame in the
whole time its callers were being moved off the host it named.

Why nobody noticed is worth recording, because it is the same reason a stale probe is
worth migrating rather than deleting: no script, batch file, test, spec or document in the
tree reads `_selftest.txt` or passes `--selftest`, so the only reader was a human who had
to open a file to discover that the failure was the probe and not the machine; and the
probe reported every exception as one undifferentiated `DLSS_ERR` line, which is exactly
what a signature mismatch looks like when nothing distinguishes the stages.

**The move.** Guidance is off in a probe, so the frame asks for the zero-motion entry —
the same one `App._GPU_HOST_KINDS["image"]` describes and the window's own still pictures
run on. The capability it checks and the refusal it prints are *read off that table*
(`gui.py:1644`) rather than restated, so the probe cannot keep describing an entry the
window has since moved; that is how it rotted the first time. `dlss_gpu_engine.Live` is
opened at 320x240 (`gui.py:1654`) and one reset frame goes through
`process_zero_motion` (`gui.py:1666`), with `live.close()` in a `finally`. There is no
fallback: an ABI-2 host is a stale native build and is reported as one, in the window's own
words.

**The reporting.** Each failure is tagged with the stage that produced it —
`[host-load]`, `[capability]`, `[frame]`, `[output]` — so "the host is not built" cannot be
read as "the host said no". A missing `native/dlss_gpu_host.dll` is rewritten from a
winerr into a sentence naming the file and saying the probe will not fall back. Success now
names the host, the entry, the reset flag, the returned shape, a short digest of the pixels
and the torch line it always carried, and it prints as well as writing the file. A
`report_path` parameter exists so the tests can point it at a temp directory; the default
is still `_selftest.txt`, and the `DLSS_OK`/`DLSS_ERR` prefixes are unchanged.

**What did not move.** Nothing about the window's rendering: no default route, no session
slot, no selector, no control, and no call to `dlss_engine.Live` from the GUI is
added or removed by this — the factory's unreachable arm at `gui.py:581` is still there,
still unreachable, and the map above is why deleting it is a decision rather than a
cleanup.

**Tested without a card.** `tests/unit/test_gui_selftest.py`, 12 tests, all passing, none
of which needs a GPU. The three worth naming: one seals `dlss_engine._load` — the seam the
map counts — and asserts the probe never reaches it, distinguishing the module import the
probe still does from the file load it must not; two read the window's table back at it,
patching the refusal string and then the capability name to prove both come from the table
(a probe with a hard-coded capability would have drawn a frame on a host whose table row
said otherwise); and one pins the dead call shape, asserting that
`Live.process` still has exactly `(self, rgba, motion, reset, raise_on_error, out)` and
that the old depth positional still raises, so nobody "fixes" this probe by restoring the
argument that killed it.

**Run for real, with the old host held out of the process.**
`scratch/_gui_selftest_gpu_gate.py`, **27/27, twice**, on the same RTX 5060 Ti the other
GUI gates used. `ctypes.CDLL` is sabotaged from outside so `dlssnr_host.dll`,
`nvof_bridge.dll`, `nvofapi64.dll` and `nvofnvml64.dll` raise on load,
`dlss_engine.Live` raises on construction and `dlss_engine._load` raises on the call; the
real `dlss_gpu_engine.Live.process_zero_motion` is wrapped rather than replaced, so the
tally is the wrapper's own view of a call that really happened. It records, for one run:

```
DLSS_OK (240, 320, 4) | host=native/dlss_gpu_host.dll entry=process_zero_motion reset=1
rgb_mean=0.233 rgb_max=1 alpha_mean=255.000 sha1=cd5095247ece | torch=2.13.0+cpu cuda=False
```

The entry was entered exactly once, with a reset, by a session opened at 320x240, handed a
contiguous uint8 240x320x4 frame; `GetModuleHandleW` reports `dlssnr_host.dll` absent
before and after and `dlss_gpu_host.dll` present; no NVOF library was loaded, because the
zero-motion entry needs neither CUDA nor a flow bridge — which is also why the probe passes
on this box's CPU-only torch, and why its `cuda=False` is a reported fact rather than a
reason to skip. The digest repeated identically across both runs. The gate's own negative
arms run with the loader still shut: an ABI-2 host is refused with the window's message and
draws nothing, and a missing host file names the file, says it did not fall back, and draws
nothing. Its evidence is `bench_out/gui_selftest_gpu_gate.json`.

## The picture A/B every migration gate refused to run

`scratch/_host_pixel_ab.py`. The boundary has asked for this comparison since the preview
moved — "the A/B that this item has kept asking for is still outstanding", then
"outstanding on three now" — and every gate written since then refused `dlssnr_host.dll`
on purpose, which is the right call for a routing test and a structural impossibility for a
parity test. So this is the first place in the document that puts the two hosts' decoded
pictures side by side. It is four frames at one working resolution, twice, and nothing
else: no timing, no throughput, no re-running of a gate that already passed.

**One host per process, because the old DLL cannot share one.** `dlssnr_host.dll` keeps a
singleton D3D12/NGX state and is not safe through a second `dlssnr_init` in one process
(`ABI_CAPABILITIES.md`, and `dlss_engine.Live._teardown`'s note that a double shutdown is
an access violation Python cannot catch). Each arm therefore runs in its own child: the
child opens one session, draws four frames, writes them as `.npy`, and exits. Two children
per arm, plus two more for the control below.

**The fixture.** 1920x1080, which is the working resolution a 4K source at 0.5 scale
produces and a 1080p source at 1.0 produces — one representative scale, not a sweep. Four
frames of one synthetic scene carrying a step edge, a diagonal, a 97-pixel stripe pattern
and ±14 levels of noise, each frame the previous one moved by (0.6, 0.2) px, so the flow
that describes it is both correct and sub-pixel. Settings are the panel's defaults
(`preset 1, style 0, intensity 1, local_tone 1, local_struct 1, skin_struct 1,
use_auto_mask 1, ui_correction 0, motion scales 1`), given to each host through its own
option contract. The two arms are the two shapes production actually sends:

* **guidance off** — old host: `process(rgba, plane)` with an all-zero float32x2 plane and
  `guidance_mode=0`, which is what a host-pointer job with no flow has always sent; GPU
  host: `process_zero_motion(rgba)` with **no motion argument at all**, its host binding a
  resident zero texture. Those are different call shapes by design, and this is the only
  place they have been compared.
* **guided CPU flow** — both hosts get the *same* float32x2 plane, uncast, at the same
  extent: `process` and `process_host_motion`. Plane convention as stated and used: the
  displacement of scene content from the previous frame, in pixels, +x right, +y down.
  Depth is zero on both sides by construction: a zero plane under the fixed
  `depth_convention=2` on the old host, a persistent zero texture on the new one.

**The result, stated as numbers.** All eight frames — four per arm — are **byte-identical**
across the two hosts, on RGB and on alpha: `rgb_mae=0.0`, `rgb_max=0`, PSNR infinite,
matching SHA-1 per pair. `bench_out/host_pixel_ab/ab_report.json` keeps every figure and
hash.

That sentence is worth nothing on its own, so two controls come with it.

*The agreement is not two hosts doing nothing.* Each frame's distance from the frame it was
handed is reported per row: 18.36, 19.83, 20.80, 20.93 MAE in the unguided arm and 18.36,
20.84, 22.02, 22.32 in the guided one, with **100.0%** of pixels differing from the input in
every frame of both arms. Whatever the two hosts agreed on, it was a picture each of them
had rewritten.

*The agreement is not two hosts ignoring the plane.* A third pair of children re-runs the
guided arm on both hosts with the plane's y negated. On **both** hosts the reset frame is
unchanged and frames 1-3 move, by identical amounts: MAE 1.466 / 1.679 / 1.630, max 70 /
51 / 45, PSNR 41.87 / 40.90 / 41.12 dB — and the same divergence profile on both sides,
down to the hashes. Flipped-old against flipped-GPU is then byte-identical again on all
four frames. So the plane reaches DLSS through both hosts, it reaches it under the same
convention, and the agreement is not an artefact of a dropped input.

**What the control also says about the reset frame.** A frame drawn with `reset=1` came out
the same whether its host was handed no motion, an all-zero plane or the fixture plane —
the unguided and guided reset frames share the hash `82fe4ec27c74`. On Feature 18 the first
frame of a sequence is the spatial reconstruction; the motion vector only starts to matter
when there is a history to reproject. That narrows an existing claim rather than adding one:
`native/dlss_gpu_host_motion_smoke.py` compares frame-for-frame *on a reset* and is right
to say so, but its parity is parity of the spatial path only, and it could not have
detected a host that ignored motion. The temporal half — where the plane demonstrably acts —
is what this A/B adds, and it holds.

**What this does not cover.** One working scale, one settings tuple, four frames, one
fixture; the guided case is a CPU flow plane, so the NVOF device-descriptor entry
(`process_device_flow`) is not in this comparison at all, and neither is the
`gpu_transfer` in-process chain, an extent that is not a multiple of 8, a preset other than
1, real depth guidance (which no host in this tree takes any more), or any second card.
Byte identity on this fixture is not a parity theorem about the two binaries, and no
threshold was invented to grade the result: the frames either matched or they did not, and
the largest divergence this run measured anywhere was the 1.68 MAE the plane-flip control
produced identically on both hosts. Performance is `NOT_CHARACTERIZED` — nothing in this
section timed anything, and the two hosts ran in separate processes for a reason that has
nothing to do with fairness.

**The tree this stage leaves.** `tests/unit` is **711 passed** in 99s — the 699 the default
 moved with, plus the 12 this stage added — with nothing deselected and nothing skipped for
 want of a card. `tests/unit/default_gpu_routing_gate.py` is **32/32**, re-run rather than
 assumed, because a probe that names a host is one more thing that could have moved a
 default. `scratch/_gui_selftest_gpu_gate.py` is **27/27 twice**. The five Playwright specs
 were not re-run, and the reason is the changed-code test: this stage touched one function
 in `gui.py`, added one stdlib import to that file, and wrote two scripts under `scratch/`
 and one unit file under `tests/unit/`. No service route, no server endpoint, no frontend
 source, no pipeline or native code and no ABI changed, so the browser paths have nothing
 new to exercise — the last stage's 5/5 stands, and re-running it would be a gate repeated
 rather than a gate widened. No DLL was deleted, no native source or header was edited, and
 nothing was committed, pushed or cleaned up: `dlssnr_host.dll` is still in the tree, still
 shipped by the spec, still reachable by the one name that asks for it, and the dirty tree
 this stage inherited is as dirty as it was.


*One honesty note about this stage itself, because the section above is the one that reads
best: the routing and the default were already in the partial diff it inherited from a run
that died before recording anything, and the only line of production code this stage changed
is in a gate script. Its test files and this document were what was missing.*

## The packaged window draws on the same host as the source tree

The item below this section's boundary had been left as "recorded, not fixed", on the grounds
that no PyInstaller build had ever been run. One has now been run, so the prediction it rested
on can be replaced with a result. What the build found is not one bug but one missing answer:
six modules each guessed, differently, at where a frozen bundle keeps its files.

**The two roots, named once.** A PyInstaller ONEDIR bundle has a launcher directory
(`<dist>/DLSS5Tool/`, what `sys.executable` is inside) and a contents directory
(`<dist>/DLSS5Tool/_internal/`, what `sys._MEIPASS` names). In a source tree there is one
directory and it is both. Every module that resolves a shipped binary used to pick a
different stand-in for those two: `dlss_gpu_engine._LIBRARY` was
`dirname(__file__)/native/dlss_gpu_host.dll` with no frozen branch, `dlss_engine` had a
`_res_dir()` that did handle the question, `pipeline._bundle_ffmpeg()` looked for its tool at
the `sys._MEIPASS` root, `optical_flow` and `residual_native` rebuilt `native/` from
`__file__` again, and `pipeline.py` carried a second, unreferenced pair of `HOST_DLL` /
`DLSSNR_DLL` constants that disagreed with `dlss_engine`'s about which root they meant.
`bundle_paths.py` is that one answer: `contents_dir()` (`:69`) and `app_dir()` (`:76`),
`native_dll()` / `collected()` over them, and the shipping manifest itself
(`CONTENTS_BINARIES` `:33`, `NATIVE_BINARIES` `:40`, `TOOL_BINARIES` `:55`, `NEVER_BUNDLE`
`:61`). Torch, model and settings behaviour are untouched; nothing in this section's diff
changes a number, a transfer or a preset.

**Two machine literals went out with it.** The committed spec's `binaries` list was
`dlssnr_host.dll`, `nvngx_dlssnr.dll` and `E:\games\dlss5\DLSS5Tool_build\ffmpeg.exe` — a path
from a machine that no longer answers for this tree — with nothing from `native/` and no
`ffprobe.exe` at all. `gpu_transfer_export.ffmpeg_dll_directories()` named
`F:\WhisperJAV\Library\bin` outright. Both are now derived: `resolve_sources()` (`:156`)
returns `(source, destination)` pairs built from the manifest and the FFmpeg installation the
build machine actually has (`DLSS5_FFMPEG_DIR`, else a directory on `PATH` holding both tools
and preferring one that also carries `avcodec-*.dll`), and it **raises** naming the file and
the layout rather than emitting a bundle that fails on the first frame. A test asserts no
non-portable drive-letter string survives in any production module or in the spec.

**What the bundle is now, measured.** PyInstaller 6.22.3, one build, `--distpath` and
`--workpath` inside `C:\Users\18392\AppData\Local\Temp\dlss5_pkg_smoke\`, exit 0,
"Build complete!", 1,170 MB.

* `_internal/native/` holds exactly the nine bridges the manifest names, and each one is
  byte-identical (SHA-1) to the file in this working tree — `dlss_gpu_host.dll` `c54fda99e647`,
  `nvof_d3d12_interop.dll` `01d3cefabcc4`, `nvdec_bridge.dll` `3415156533a8`,
  `nvenc_bridge.dll` `0469ca317f88`. The bundle carries *this* host, not an older copy.
* `_internal/` holds `nvngx_dlssnr.dll`, `dlssnr_host.dll`, `ffmpeg.exe` and `ffprobe.exe`.
  The FFmpeg shared libraries the decoder and encoder bridges import at load time
  (`avcodec-61`, `avformat-61`, `avutil-59`, `avfilter-10`, `avdevice-61`, `swresample-5`,
  `swscale-8`) are there too, collected by PyInstaller's own walk of the bridges' import
  tables — which is what makes the frozen branch of `ffmpeg_dll_directories()` (which now
  includes the contents directory) answer instead of coming up empty.
* Zero copies of `nvcuda.dll`, `nvofapi64.dll`, `nvencodeapi64.dll`, `nvencodeapi.dll`,
  `nvml.dll`, `d3d12.dll`, `d3dcompiler_47.dll` or `dxgi.dll`. Those belong to the installed
  driver and the OS; a bundle copy would be searched first and cannot be newer than the thing
  it shadows. A spec-level filter (`_keep`) drops them and anything else under the Windows
  directory, and drops the flattened root-level duplicate of any bridge already declared into
  `native/` — `dlss_gpu_host.dll` imports `nvof_d3d12_interop.dll` at load time and does not
  look at the contents root first.

**The one frozen run.** `--selftest` in that output, with `_internal/dlssnr_host.dll` renamed
to `.disabled` first — the old host removed *only inside the isolated dist*, nothing in the
repository touched — and from that directory as the working directory:

    DLSS_OK (240, 320, 4) | host=native/dlss_gpu_host.dll entry=process_zero_motion reset=1
    rgb_mean=0.233 rgb_max=1 alpha_mean=255.000 sha1=cd5095247ece | torch=2.13.0+cpu cuda=False

exit 0, and the same line in `_selftest.txt`. The same probe from the source tree, writing to a
temp path so no repository file is created, gives the **identical** `sha1=cd5095247ece`. The
packaged window and the source tree put the same host through the same entry and got the same
bytes back, with the legacy DLL unavailable to either.

**What that frame does and does not prove.** Its input is an all-zero 320x240 RGBA buffer, so
it establishes that the frozen layout resolves, that `dlss_gpu_host.dll` loads inside the
bundle (which means `native/nvof_d3d12_interop.dll` resolved beside it and
`nvngx_dlssnr.dll` resolved from the contents directory), that the zero-motion capability
check answered and that the return is the shape and dtype `gui._selftest` validates. It is not
picture evidence; the eight-frame byte-identity A/B in the section above this one is that, and
it ran in the source tree. `cuda=False` is this venv's CPU-only torch, not a packaging fault:
the frozen DLSS path never calls torch. `rgb_mean=0.233` on a black input is the reconstruction
of a black input.

**Not verified, stated plainly.** No GPU transfer export ran inside the bundle. The frozen
window has no CLI for one — `gui.py:1690` branches on `--selftest` and otherwise starts Tk —
so a real frozen export would need a new entry point, which is outside this stage. What was
checked instead is the thing that fails first on that path: `ctypes.CDLL` on each of the nine
bundled bridges, with the bundle's own two directories registered, loads all nine, including
`nvdec_bridge.dll` and `nvenc_bridge.dll` against the bundle's collected `av*` set. That is
the binary closure of the bundle, loaded from outside the frozen interpreter; it is not an
export and no frame was drawn with it. No NVOF capture ran in the bundle either — the bridge
loads, and `nvofapi64.dll` is deliberately the driver's, so a machine without an NVOF-capable
driver behaves as the source tree does. No benchmark was run and no timing was taken;
performance after repackaging is `NOT_CHARACTERIZED`.

**Not portable either.** The build ships the build machine's FFmpeg — that is the design, a
bundle cannot carry a tool the source tree found on `PATH` without carrying its DLLs too, and
it is why `DLSS5_FFMPEG_DIR` exists. It also carries torch: 518 MB of this 1,170 MB, because
`pipeline` and `gui._selftest` contain `import torch` and the spec's own comment forbids
excluding torch's own runtime dependencies. That is size, not correctness, and trimming it is a
separate decision with a separate risk (a bundle that cannot import torch is a bundle whose
depth and RAFT buttons have to say so honestly). And `native/build_*.bat` plus
`native/CMakeLists.txt` still name prior-machine SDK locations (`E:\Visual Studio`, `F:\dlss`).
Those are build-time only, ship nothing, and were left alone because rebuilding the bridges was
not this stage's job — but they are the reason a fresh clone cannot rebuild what this spec
insists on collecting.

**The tests.** `tests/unit/test_packaged_binaries.py`, 11 tests. They do not restate the
manifest: the collection test reads the native library names out of the loader *sources* with
a regex per module and demands each one is collected into `native/`, so a tenth bridge added
without a manifest entry fails the build rather than shipping broken. Another exec's the real
spec against stubbed `Analysis`/`COLLECT` and asserts every declared source exists on disk,
survives the spec's own filter, and lands where the loaders will look; another asserts the
filter refuses a flattened duplicate and a System32-sourced `nvngx_dlssnr.dll` while keeping
`python311.dll`; another runs a child interpreter with `sys.frozen` and `sys._MEIPASS` set and
asks each production loader, in that state, which path it resolves — the four native ones must
answer under `_internal/native`, the snippet and legacy host under `_internal`, `ffmpeg.exe`
under `_internal`. `bundle_paths.missing_binary()` is tested for saying different, actionable
things per layout: "rebuild it from the current source tree and spec" to a bundle, "rebuild it
from native/" to a source tree.

**The tree this stage leaves.** `tests/unit` is **722 passed** in 124.76s — the 711 the picture
A/B left plus the 11 here, nothing deselected, nothing skipped for want of a card, run once. No
native source, header, ABI, model, preset or transfer-semantics line was touched; no Playwright
spec was re-run, because no route, endpoint or frontend file changed; no benchmark. Nothing was
committed or pushed. `dlssnr_host.dll` is in the tree under its own name, unmodified, and
`git status` counts the same 98 dirty-or-untracked entries it did before this stage began.

**Can `dlssnr_host.dll` go now? No.** The packaged window no longer needs it — that half is
now a measurement rather than an inference. But `dlss_engine.Live` is still the implementation
behind the explicit `dlss_backend=host_pointer` name, which `dlss_service.py` still accepts,
still advertises in its own refusal text (`:470`) and still treats as the only host for a job
that asks for it, and the spec still ships it. Removing the file is that routing decision made
final, not a packaging cleanup, and it stays open.

## Current boundary and next work

The opt-in production backend removes 16,588,800 bytes/frame of NVOF flow D2H
at 1920x1080. It does not remove the 12,441,600-byte/frame decode pipe or the
24,883,200-byte/frame encoder pipe, and it still uploads color and reads
back the DLSS result for CPU residual composition. That description is the
`legacy` path. The `gpu_transfer` path in the section above replaces those two
pipes with the in-process decoder and encoder for the cases it accepts, and
keeps the pipe for the cases it refuses. The decode boundary is now counted
rather than unknown: `nvdec_bridge` ABI 3 reports the frames it handed out, the
host copies it made and the compressed bytes it submitted, so the 12,441,600
bytes/frame the pipe download used to cost reads as a measured zero on this
path instead of `null`. The NV12→RGBA and working-color device writes are
counted now too, by the `nv12_rgba_bridge` ABI 3 and `gpu_lanczos_bridge` ABI 2
getters above, so the figures at those boundaries are the bridges' own
code-determined payload numbers rather than `null`; the reads around them and
any cache-line or PCIe figure stay unmeasured because neither bridge can observe
them. What still blocks removing the decode pipe is the chain's
working-scale restriction and the remaining correctness gates, not an
uninstrumented boundary. Performance impact remains
`NOT_CHARACTERIZED`. Magpie remains a
read-only reference; its GPLv3 implementation was not copied into this tree.

The old `dlssnr_host.dll` is still a runtime dependency for the host-pointer preview —
in **both** guidance modes —, the host-pointer export, and the desktop GUI's *video
render and video export*. *The last clause is now history rather than a blocker: the
desktop GUI's video render moved in the section above this one, and its streaming export
moved in the one before that, so the window draws no frame on this host any more. What the
file still contains of it is a factory arm no call site reaches (`gui.py:580`, behind
`gui.py:518`'s unused `kind="legacy"` default) and the stale `--selftest` probe at
`gui.py:1636`.*
*Both halves of that survivor are now one fewer. The probe moved to the GPU host's
zero-motion entry in the section three above this one, so `gui.py` reaches
`dlss_engine.Live` nowhere at all; the factory arm stays, still unreached, and the line
numbers in the sentence above this one drifted again by one because the migrated probe's
file gained an `import hashlib` — `gui.py:581` and `gui.py:519` today. And the question
"what exactly still needs this file" is no longer answered from six paragraphs of
corrections: it has its own section, two above the probe's, which counts the loads at
`ctypes.CDLL` instead of at the name.* It is no longer the only way to run a still image, and it is no
longer the only way to draw a guided preview frame — in either program. The GUI is still
the only place in
this tree that denoises one, and that path now runs on the GPU host's
`process_zero_motion` — see the section on it — while a preview that names
`dlss_backend=gpu_flow_d3d12` with `guidance_mode=1` runs on that host's
`process_host_motion`: the entry the *A host-memory motion entry* section added, the
*guided preview* section routed onto it, and the *CPU flow provider in the export* section
above this one opened to a second caller. And the desktop window reaches both entries on
its own, in either guidance mode and with no `dlss_backend` control to move, which is the
section above this one.
Six routes have come off it: the unguided preview on
`dlss_backend=gpu_flow_d3d12` — see the section on it, and note it is now
selectable in the shipped UI rather than only by a hand-written
`POST /api/settings` — the guided preview on that same backend, the unguided export on
the same backend, on *either* export chain, the still-image path, a guided export
on that backend over the legacy FFmpeg pipe, for a provider whose motion is host memory
rather than a CUDA descriptor, and the desktop GUI's video preview in both guidance modes.
*That list is seven with the section above this one, which adds the desktop window's own
streaming export, in both guidance modes and again with no `dlss_backend` control to move.
*And eight with the section above that one, which adds the same window's cached-frame render
loop — the PNG sequence the export-only mux reads — on the same two entries, named
`render_zero_motion` and `render_host_motion` because the slot it claims is the third of three
sharing it.*
What is left
of the service side is keyed on the host, not on the chain: `dlss_backend` chooses who
runs the frame, and every configuration that still chooses the old host chooses it
because the operator named that host or named none, not because the GPU host is missing
an entry point for the frame or the export refuses to take a plane. The entry the sections above kept calling the last missing
piece now exists, is validated and has callers
(`dlss_gpu_host_process_host_motion`, ABI 3), as does the zero-motion entry beside it
(`dlss_gpu_host_process_zero_motion`). What keeps the DLL is a backend named by the
operator — `host_pointer`, or nothing, which reads as the same host — and a window that
has no `dlss_backend` control to move.
*Both of those paragraphs' second half is what the section above this one removed: naming
none stopped reading as the same host. The sentence's remaining subject is a frame whose
operator said `host_pointer` — in the panel, in a hand-written job, or in a client built
before the key existed and still sends it, which is the compatibility that paragraph's
"named that host" now covers on its own — plus the same probe no operator presses.*
*The last clause still holds and now explains less than it used to: the window has no
such control, and it turned out not to need one for either of its two video tasks that
name an entry, because guidance mode alone says which of the host's two entries may draw
the frame. The one video task that
does keep the DLL — the cached-frame render loop — is kept by the call site that
names `kind="legacy"`, not by a missing choice.*
*Both halves of that amendment are now history, and by the same mechanism: the render loop
needed no control either, because it asks the same two entries the preview and the streaming
export ask, on `App.video_render_kind` (`gui.py:506`) under the third pair of kind names
(`gui.py:469`, `gui.py:474`). So the window has no `dlss_backend` control and no
`kind="legacy"` call site, and the sentence's remaining subject — a video task kept by a
call site rather than by a missing choice — no longer describes anything in `gui.py`.*
Read off the call sites, what still constructs it is
`dlss_service.py:476` (the preview's legacy branch — keyed on `dlss_backend` and
`guidance_mode` at `:368`, so it now takes every frame on the default host-pointer
backend, guided or not, and no frame that names the GPU host), `dlss_service.py:1072` (a job
session whose `dlss_backend` is not `gpu_flow_d3d12`), `dlss_service.py:860` (the
job's `get_log_tail` readout), and `gui.py:550` — the legacy arm of the one session
factory at `gui.py:512`, reached now only through an explicit `kind="legacy"`: the
window's cached-frame DLSS render (`gui.py:1248`, claim at `:1269`, release at `:1304`).
Its streaming export is no longer in that list: it names
`App.video_export_kind` (`gui.py:477`) and reaches the GPU host through the same factory,
in either guidance mode (`gui.py:1462`, `gui.py:1469`). Both of those video tasks own the
single session slot on the host they opened for the length of their run
(`_video_task_kind`, declared at `gui.py:316`, enforced at `gui.py:521`), and the export now
owns it under its *own* kind names (`export_zero_motion`, `export_host_motion`,
`gui.py:453`, `gui.py:458`) rather than the preview's, which is what keeps a preview frame
that slips past `pause()` from resetting or evicting the session the encoder is being fed
through — the hazard the preview migration created and this one closed.
*The render is no longer in that list either, and it was the last thing in it that named the
legacy arm on purpose rather than by default: `App._run_dlss_gui` (`gui.py:1284`) now claims
`video_render_kind(s)` before it opens (`gui.py:1302-1306`) and releases it in its `finally`
(`gui.py:1351-1352`). Read off the call sites today, `gui.py` contributes **no** construction of
`dlss_engine.Live`: the factory's legacy arm is at `gui.py:580` behind the `kind` default of
`gui.py:518`, no call site uses that default, and the only `dlss_engine.Live` the file still
names is the stale probe two entries further down.*
*Read off the same call sites after the default moved, the service's three entries in that
list are two and a half. The preview's legacy arm is at `dlss_service.py:535-537` (the key is
read at `:422`, and `guidance_mode` no longer decides between hosts there, only between the
GPU host's two entries), the job's session arm is `live_type = dlss_engine.Live` at `:1148`,
reached only when the `gpu_flow_backend` test at `:1145` fails — so both now take a frame
because `dlss_backend` said `host_pointer`, and neither takes one because the setting was
left alone, which is what the "default host-pointer backend" in the sentence above means
after this stage. The third is not a host at all: `:925` reads
`dlss_engine.get_log_tail(30)`, a symbol on the wrapper module rather than a session, and
what the default moved for it is that a default job's log tail now comes from a process whose
frames were drawn by the other DLL.*
Neither the GUI's image paths, its video preview nor its streaming export is among its
callers any more:`_image_dlss`/`_dlss_image` (`gui.py:1026`, `gui.py:1035`) and
`_do_images` (`gui.py:1323`) go through the image kind of `_open_gpu_live`
(`gui.py:555`), `_live_dlss_image` (`gui.py:89`) goes through that helper's two
preview kinds, and `_do_dlss_stream_export` (`gui.py:1452`) through its two export kinds.
The window still has no `dlss_backend` control, so the render loop's frames cannot name the
GPU host — for them the missing thing is the choice, not the entry point, and the
service's remaining old-host frames are the ones where that choice was already made the
other way. *That sentence used to cover the export too, and it was true for a different
reason: the export had no entry to name and no provider the core would accept on this host.
Both halves are gone, so what is left of it is a call site that asks its own question.*
*That last clause has now answered itself and dissolved: the render's call site asked its own
question, `kind="legacy"`, and the answer is `App.video_render_kind` (`gui.py:506`), so the
window is out of this paragraph's subject entirely. Every frame the operator can ask the
window to draw is now on the GPU host; what still draws on the old one is a service frame whose
`dlss_backend` named that host or named none, plus a `--selftest` probe no operator presses.*
*The "or named none" clause of that sentence is the whole of what the section above this one
changed, and it changed it in the service rather than in the window: naming none is now naming
`gpu_flow_d3d12`, so what draws on the old host is a frame whose caller said so, plus the same
probe. A pre-migration client that keeps sending `host_pointer` on every control change is a
caller saying so, which is why that compatibility is kept this stage rather than collapsed.*
One more old-host caller is worth naming precisely because it is stale
rather than merely unmigrated: `gui.py:1590`, `--selftest`, still passes a depth plane
as a third positional *and* `reset=True` to a `Live.process` whose third positional
**is** `reset` (`dlss_engine.py:219`), so that probe can only ever write `DLSS_ERR
TypeError`. Read, not executed; a leftover of the ABI change that dropped the depth
argument, and not this migration's doing.
Deleting the DLL now would still break every one of those. Its GPU-host entry points now
exist for host motion as well as zero motion, each with its short correctness check and
each with routed callers on both sides of the tree — a service preview, a service export,
a desktop preview and now a desktop streaming export on the motion entry, a desktop still,
a desktop preview and a desktop unguided streaming export on the
zero-motion one — so no frame is waiting on a native symbol any more: what is left is the
window's cached-frame render, which names the legacy arm of its own factory, and
the frames whose operator picked this host.
*Of that pair only the second half survives: the render names `render_zero_motion` /
`render_host_motion` now, so it joins the list this sentence was counting — five callers on the
host-motion entry (service preview, service CPU-flow export, desktop preview, desktop streaming
export, desktop render) and four on the zero-motion one (a desktop still, a desktop preview, a
desktop streaming export and a desktop render, each unguided). The DLL is kept today by operator
choice and by one probe, not by a GUI call site.* The NVDEC bridge does not change that
dependency.

What is left after the preview-route work above, in the order it unblocks
itself:

1. ~~`guidance_mode=0` on the paths that are not GPU-transfer.~~ *The preview and
   both export chains are done, and so are the still-image path and the desktop GUI's
   video preview and streaming export — the last two in **both** guidance modes. What is
   left of the desktop GUI on the old host is its cached-frame render loop (`gui.py:1248`,
   `kind="legacy"` at `gui.py:1271`), the export-only mux that consumes the PNGs that
   render writes (`gui.py:1541`) and builds no session at all, and the stale `--selftest`
   probe at `gui.py:1588`.*
   `dlss_service` routes an unguided `dlss_backend=gpu_flow_d3d12` preview onto the
   GPU host's `process_zero_motion` entry, and the entry now answers an unguided
   export on either chain as well — the two sections above, each with its own unit
   gate (`tests/unit/test_preview_zero_motion.py`,
   `tests/unit/test_export_zero_motion_pipe.py`) and its own hardware gate
   (`scratch/_zero_motion_preview_4k_gate.py`,
   `scratch/_zero_motion_pipe_4k_gate.py`). Both gates prove the old host is absent
   by sabotaging the load, not by reporting which entry answered.
   The half this item kept pointing at is now closed: the HTTP API still has no
   still-image endpoint, so the image path remained `gui.py`'s, and `gui.py` is the
   section above — its own unit file (`tests/unit/test_gui_image_zero_motion.py`) and
   its own hardware gate (`scratch/_gui_image_zero_motion_gpu_gate.py`), both proving
   the old host absent the same way. It did not need a `dlss_backend` control, because
   an image has no motion to route and image mode already forces guidance off; the
   window names the host at the one seam that needed naming.
   ~~What is left inside that window is its video preview worker and its own export.
   Both ask for `dlss_engine.Live` whatever the guidance control says, so an
   unguided video frame is still on the old host today~~ *Both halves of that sentence
   are now one half: the preview worker stopped asking for `dlss_engine.Live` in either
   guidance mode — the section above yours — and ~~only the export still does~~ the
   streaming export followed it, in both modes, in its own section a little further above.
   Moving the
   worker also disproved the last claim of this paragraph: the routing needed no control,
   because the two GPU entries are distinguished by the guidance mode the panel already
   had, so `App.video_preview_kind` reads the entry off it and the window names the host
   at the same one seam the still image did.*
   The entry point that entry needed is no longer a blocker either: the
   guided-preview section routed a guided frame onto
   `dlss_gpu_host_process_host_motion` — in the service, which is where the choice
   could already be made. The window cannot make it, so the routing left in this
   item is a control as much as a call site.
   The one case this item never covered is untouched by all of the above: an
   unguided job on a *host-pointer* backend still runs on `dlssnr_host.dll`, which
   is what it asked for by name or by default, and it is still not refused at the
   gate.
   *"By name or by default" lost its second half: the default is the GPU host now, so a job
   that names nothing is not on that DLL any more, and the case left is the one that says
   `host_pointer` out loud. Not refused at the gate, and not supposed to be — it asked for
   what it got.*

   *What is left of the "next step" sketched below is its second half. The preview
   migrated without a `dlss_backend` control, so the control is no longer a precondition
   for the window to reach the GPU host — it is a choice the operator would have to be
   given for the *export*, whose frames are drawn by `pipeline.ProductionExportCore`
   and whose flow comes from a `CachedFloFlowProvider`. That pairing is already a route
   and not a rejection.*
   *The next step inside that window, now that the entry and the core route both exist,
   is the video migration rather than the still one: give `gui.py` a `dlss_backend`
   control and let its cached-`.flo` export (`gui.py:1420`, provider at `gui.py:1441`, a
   session from the one factory at `gui.py:481`) name `gpu_flow_d3d12`. Nothing in the core
   would refuse that pairing any more — a `CachedFloFlowProvider` on that host is a route,
   not a rejection — so the work left is the control, the session-reuse and geometry keys
   around it, a unit file in the shape `test_gui_image_zero_motion.py` took, a gate in the
   shape `_cpu_flow_export_gate.py` took, and the rendered-pixel A/B against the old host on
   this pairing that neither of the two sections above produced.
   The GUI's video preview now has its own unit file
   (`tests/unit/test_gui_preview_gpu_host.py`) and its own hardware gate
   (`scratch/_gui_preview_gpu_host_gate.py`), both proving the old host absent the same
   way, and the export was the last path in the window that still named
   `dlssnr_host.dll`.*
   *That last clause is now the one that moved: the streaming export has its own unit file
   (`tests/unit/test_gui_stream_export_gpu_host.py`, 29 tests) and its own hardware gate
   (`scratch/_gui_stream_export_gpu_host_gate.py`, 34/34 twice), it needed no
   `dlss_backend` control either, and the sketch two paragraphs above — "give `gui.py` a
   `dlss_backend` control and let its cached-`.flo` export … name `gpu_flow_d3d12`" — turned
   out to be one control short of the truth and one control too many: guidance mode already
   says which entry a frame runs on, and the window had to name the *kind* apart from the
   preview's (`gui.py:453`, `gui.py:458`) rather than name a host. What is left of the
   window on the old host is its cached-frame render loop and the export-only mux that
   consumes the PNGs that render writes; neither is a DLSS frame the GPU host cannot serve.*
   *Both of those are now one, and then zero: the render moved, on the same two entries and with
   the same absence of a control, under `tests/unit/test_gui_video_render_gpu_host.py` (31) and
   `scratch/_gui_video_render_gpu_host_gate.py` (45/45, twice, on the same RTX 5060 Ti), and the
   export-only mux that reads its PNG sequence builds no session and never needed a host. What
   the window keeps of `dlssnr_host.dll` after all three video tasks and its image paths is the
   unreached legacy arm of its own factory (`gui.py:580`) and the stale `--selftest` probe, which
   the boundary text above places at `gui.py:1588`/`gui.py:1590` and which is at
   `gui.py:1636`/`gui.py:1638` because this stage's net 48 lines moved it — those two numbers
   were right when they were written and drifted, which is the usual fate of a line reference in
   this document.*
   *The drift stopped meaning anything for the probe, because the probe is gone: it is now the
   GPU host's `process_zero_motion` at `gui.py:1654`/`gui.py:1666`, and what the window keeps of
   the old host is exactly the one factory arm this item has been carrying all along —
   unreached, and no longer the last executable thing in the file.*
   The A/B that this item has kept asking for is still outstanding on *both* halves: no
   section in this document, the preview and the streaming export ones included, has
   compared the pictures the two hosts make side by side.
   *It is outstanding on three now, and each of the three has a gate that would have to be the
   A/B and is not: the render's own picture on the old host was never compared to the PNGs
   `scratch/_gui_video_render_gpu_host_gate.py` recorded, because that gate refuses
   `dlssnr_host.dll` on purpose. What the render gate does establish about pixels is narrower
   and checkable — that the unguided frame and the frame handed an all-zero cached plane come
   out of the two entries identically, and that a reset flag changes the picture — and it says
   nothing about what the old host made.*
   *The comparison now exists — `scratch/_host_pixel_ab.py`, in the section above this list —
   and it answers the question this item meant, in the narrowest way that is honest to claim:
   byte-identical pictures from the two hosts on all four frames of both the unguided pairing
   (no-motion entry against an all-zero plane) and the guided one (the same float32x2 plane
   through both entries), at 1920x1080, one host per process, with a plane-flip control that
   moves frames 1-3 by the same MAE on both hosts and so proves neither side dropped its
   motion input. It is not the three comparisons this paragraph named. It does not drive the
   window's preview, streaming export or cached render, so it says nothing about what those
   paths' own decode, scale, residual and encode stages do with a frame once a host returns
   it, and the render gate's PNGs are still uncompared against the old host for exactly the
   reason this paragraph gave. What closed is the host-to-host half — the sentence "no section
   in this document has compared the pictures the two hosts make side by side" is no longer
   true — and what stays open is per-path, per-stage pixel parity, which is a larger ask than
   this item ever spelled out.*
2. The `optical_flow` module import at `dlss_service.py:32`. Once every caller
   reaches it lazily, that becomes an import inside the engine factory and an
   NVOF-free install stops paying `cv2`/`torchvision` start-up cost for a job
   that computes no flow.
3. The `nvof_d3d12_interop` name. Nothing in it is flow-specific, so splitting
   the CUDA↔D3D12 shim from `nvof_d3d12_*` symbols — or renaming the library —
   is what would let a mode 0 build carry the colour bridge without carrying a
   misleading dependency edge. Cosmetic, and deliberately not done under a
   loading-path change.
4. `dlssnr_host.dll` removal, which needs items 1 and 2 finished first: today it
   is still the only host for a host-pointer preview in either guidance mode, for a
   host-pointer export on either chain, and for the standalone desktop GUI's video
   render and export. The unguided GPU preview, the guided GPU preview, the
   unguided GPU export on both
   chains are off it, and so is every still image the window denoises — which
   removes the "the GUI owns image processing too" clause from this item's list of
   blockers. What kept the DLL was the same thing in both halves of the tree: a
   frame that carries host motion, and the `optical_flow` import edge in item 2. The
   first half of that has changed since this section was written — the GPU host now
   answers a host-motion frame (`dlss_gpu_host_process_host_motion`, ABI 3, with its
   hardware and CPU-only gates) — and a caller has been routed onto it, the guided
   preview, so today the
   DLL is kept by the *backend choice* of every host-pointer preview and export and
   by the GUI's video paths, plus the item 2 import edge. Item 1 is the routing this
   removal now waits on.
   *The other half of "a frame that carries host motion" has changed too, and it was the
   half this item kept for last: the export now takes a CPU flow provider on that entry as
   well (`pipeline.ProductionExportCore._route_guided_flow`, with
   `tests/unit/test_export_cpu_flow_gpu_host.py` and
   `scratch/_cpu_flow_export_gate.py`), so a guided `.flo` job is on the DLL only because it
   names no GPU host. Two callers, then, rather than one — and the item's remaining blockers
   are unchanged by it: the GUI's video paths and the item 2 import edge.*
   *Both of those clauses are now smaller than they were. The GPU host has three routed
   callers on its host-motion entry — the service's guided preview, its CPU-flow export,
   and the desktop window's guided preview — and one on its zero-motion entry for the
   window's own unguided frames (`scratch/_gui_preview_gpu_host_gate.py`, 28/28 on an
   RTX 5060 Ti), so "the GUI's video paths" in the sentence above is now its render loop
   and its stream export only. What is left of this item is therefore the item 2 import
   edge plus one GUI call site pair (`gui.py:1239`, `gui.py:1433`), and the alignment
   question the preview settled by keeping raw frame dimensions: `dlss_gpu_host_open`
   asks only for nonzero dims, so nothing on the video path needs the image path's
   pad-to-8.*
   *The export half of "the GUI's video paths" has moved as well, and it was the last frame
   in the window this item called a blocker by name: `App._do_dlss_stream_export`
   (`gui.py:1452`) now routes on `App.video_export_kind` (`gui.py:477`) in either guidance
   mode, over `tests/unit/test_gui_stream_export_gpu_host.py` (29) and
   `scratch/_gui_stream_export_gpu_host_gate.py` (34/34, twice, on the same RTX 5060 Ti), and
   the host-motion entry is handed the window's own cached `.flo` plane. "Three routed
   callers on its host-motion entry" is therefore four, and its zero-motion entry now has the
   window's unguided preview, its unguided streaming export and every still image the window
   denoises. So "the standalone desktop GUI's video render and export", in this item's first
   sentence, is one of those two tasks. What is left of the item is the item 2 import edge,
   one GUI call site (`gui.py:1271`, the cached-frame render's `kind="legacy"`), the service's
   two host-pointer branches (`dlss_service.py:476`, `dlss_service.py:1072`) and the stale
   `--selftest` probe at `gui.py:1590` — no missing entry point, and no frame the GPU host
   refuses.*
   *That GUI call site is the last blocker this item named in `gui.py`, and it is gone: the
   cached-frame render routes on `App.video_render_kind` (`gui.py:506`) in either guidance mode,
   over `tests/unit/test_gui_video_render_gpu_host.py` (31) and
   `scratch/_gui_video_render_gpu_host_gate.py` (45/45, twice, on the same RTX 5060 Ti), so
   "one GUI call site pair" and then "one GUI call site" in the sentences above are now zero and
   the item's list is the item 2 import edge, the service's two host-pointer branches
   (`dlss_service.py:476`, `dlss_service.py:1072`) plus its `get_log_tail` readout
   (`dlss_service.py:860`), and the stale `--selftest` probe — which moved from `gui.py:1590` to
   `gui.py:1638` with this stage's net 48 lines and is still read, not executed, and still the
   only thing in the window that asks the old host for a frame.*
   *The service's half of that list got smaller and its numbers moved with the default: the two
   host-pointer branches are `dlss_service.py:535-537` and `dlss_service.py:1148` today, and each
   of them now takes a frame only for a caller that named `host_pointer` — the line that used to
   read "or named none" is the one the section above this boundary deleted, so the item's
   remaining blockers are the item 2 import edge, an explicitly named host, and a probe.*
   *The probe is gone, and with it the last thing in `gui.py` that asked the old host for a
   frame, so the list is two: the item 2 import edge and a caller that names `host_pointer`
   out loud. The count is no longer taken on faith — the section three above this list walks
   every reference in the tree at `ctypes.CDLL` and finds exactly two reachable
   `dlss_engine.Live` constructions, both in the service, both gated on that name, plus one
   unreachable factory arm, two dead wrapper entry points (`probe`, `run_dlss`) that nothing
   calls, and eleven scripts that hold the old host up as a comparison reference. Deletion is
   now a decision with a number attached rather than a hunt; it is still not taken in this
   job, and items 6 and 7 below are the two things that would have to be settled first for
   reasons that have nothing to do with whether any route still needs the file.*
5. ~~Making the unguided-preview route reachable from the shipped UI.~~ *Done.*
   The preview page now writes `dlss_backend` next to `export_backend` on its
   `POST /api/settings`, derived from the one control rather than chosen twice,
   and E2E 14.5 reads the pairing off the request the page actually sends and
   shows a real preview frame served by the GPU host's zero-motion session. It
   departed from the sketch above in one way: the pairing is with `export_backend`
   alone, not with `guidance_mode=0` as well, because the route needs both keys and
   the router is the thing that reads the mode — deriving the host from a single
   control keeps the guided frame on the host it was already using, and it is why
   the key has to be written on the way back out too. Doing that surfaced the
   GPU→legacy session-reuse bug fixed in the section above, which had no unit test
   because no UI path could reach it. *The derivation and the double write are
   unchanged and still correct for the reason given; what the guided-preview section
   changed is
   the consequence this sentence draws from them. One control now moves both preview
   modes onto the GPU host, and the write back out hands both of them back.*
   *The write back out is the half the default moved on. "Has to be written on the way back out"
   was a statement about a default: reverting the chain to `auto` had to name `host_pointer`
   explicitly, because naming nothing used to mean that host. Naming nothing means the GPU host
   now, so the page writes the key only where the chain requires a host —
   `dlssHostRequiredByExport` (`frontend/src/services/api.ts:14`) answers `gpu_flow_d3d12` for
   `gpu_transfer` and `null` for `auto` and `legacy`, `DLSSPage.tsx:351-352` and
   `DeliverPage.tsx:638`/`:651` apply it, and an unconditional write would overwrite an explicit
   `host_pointer` the operator had just picked in the DLSS-host select. E2E 14.2 is the version of
   14.5's pairing that reads the new rule, and it matches the panel's own copy — 切回 Auto 或
   Legacy 只换导出链路，不会把预览交还给旧宿主 — reverting the chain takes back the chain and
   leaves `dlss_backend` on the GPU host, because that is what `auto` means now. The route item 5
   made reachable is unchanged — a page that names no host still lands on the GPU one, which is
   the same frame served by the same entry — and `test_export_backend_ui_contract.py` now fails
   if the write goes back to unconditional.
6. ~~The standalone build ships the wrong host.~~ *Fixed, built and run — see the packaged
   correctness section above this boundary list.* What it was: `DLSS5Tool_standalone.spec:16-17`
   put `dlssnr_host.dll` and `nvngx_dlssnr.dll` in the bundle and put nothing from `native/` in
   it, while `dlss_gpu_engine._LIBRARY` resolved
   `dirname(__file__)/native/dlss_gpu_host.dll` with no `sys.frozen` branch at all — unlike
   `dlss_engine._res_dir()`, which was written for exactly this question. So the packaged
   window had no GPU host to reach, on a tree whose every default route now names that host.
   That half was *predicted from the collection list* when this item was written, and the
   prediction was right: the build that first tested it produced a bundle whose every DLSS call
   would have failed to find its host.
   *The fix is not a frozen branch in one module. Six modules had each guessed at the two roots
   of an ONEDIR bundle separately and differently, so the guess is now in one place
   (`bundle_paths.py`), the spec collects from the same manifest the loaders read, the two
   prior-machine paths (`E:\games\dlss5\DLSS5Tool_build\ffmpeg.exe` in the spec,
   `F:\WhisperJAV\Library\bin` in `gpu_transfer_export`) are derived from `DLSS5_FFMPEG_DIR` /
   `PATH` instead, driver and OS DLLs are refused at spec level, and one frozen `--selftest`
   with the legacy DLL renamed out of the way draws the same bytes as the source tree.*
   Whether `dlssnr_host.dll` stays in the shipped list is the item 4 question again, asked about
   a bundle that can now answer it: the answer so far is that the bundle does not need it and
   `host_pointer` still does.
7. `dlss_engine.get_log_tail` reports another host's log. `dlss_service.py:925` attaches it to
   every failed job as `dlss_log_tail`; `server.py:427` passes it out and
   `DeliverPage.tsx:1927-1931`/`:1945` render it under `[DLSS Log]` and fold it into the
   copy-out diagnostic. The file it tails is written only by `dlssnr_init`, which the GPU host
   is never given (its `dlss_gpu_host_open` takes no log path), so for a job on the default
   route the field is either empty or the tail of some earlier `host_pointer` run in the same
   process — a failure report that can quote the host that did not fail. The fix is not to
   delete the call: the GPU host's own error channel exists
   (`dlss_gpu_host_last_error`, surfaced per frame by `dlss_gpu_engine.Live._error`), it just
   has no equivalent of a tailable log behind it, so a truthful field has to say which host it
   came from. Left for the next stage; this one only records what the field means.*

## Retiring the explicit legacy host: one DLSS host ships and the old DLL leaves the tree

Items 6 and 7 above are closed here, and with them the file they were about. `host_pointer`
was the last setting that could put a frame on `dlssnr_host.dll`: every route a user could
click had already moved to `native/dlss_gpu_host.dll`, so what kept the old binary in the
distribution was one selectable backend name, one GUI factory arm nobody reached, and a
diagnostic field that read another host's log.

### The name is refused, in four places, with the migration in the message

`pipeline.DLSS_BACKENDS == ("gpu_flow_d3d12",)` (`pipeline.py:2962`) and
`pipeline.RETIRED_DLSS_BACKENDS == ("host_pointer",)` (`pipeline.py:2969`). A retired name is
never reinterpreted as the surviving one, because a caller holding a pre-migration script can
only act on an answer that names what to change:

* `dlss_service._validated_dlss_backend` (`dlss_service.py:341`) raises
  `pipeline.retired_dlss_backend_message` (`pipeline.py:2972`) before a settings merge and
  before a job is queued, so `add_job` refuses synchronously and the queue stays empty;
* `POST /api/settings` turns that `ValueError` into HTTP 400 whose `detail` is the message
  (`server.py:365-373`). Until this stage it was an uncaught `ValueError` and the caller
  received a 500 with no body at all — the migration was inside the exception and none of it
  reached the wire;
* `POST /api/queue/add` already took that route (`server.py:389`), and keeps it;
* `App._GPU_HOST_KINDS` (`gui.py:434`) has no key for the old host, so a window caller asking
  for a kind that is not one of the GPU host's entries is refused before any DLL is loaded.

The refusal text contains `host_pointer`, `dlssnr_host.dll`, `dlss_engine.py`,
`native/dlss_gpu_host.dll` and `gpu_flow_d3d12`; that set is pinned by
`tests/unit/test_gui_stream_export_gpu_host.py`, so a message that loses the migration fails a
test rather than losing a word.

`export_backend` is untouched and still three distinct choices — `auto`, `legacy`,
`gpu_transfer` — because it names how frames travel, not who denoises them. One host now draws
all of them.

### What the single host answers, on hardware, with the old DLL deleted

`tests/unit/default_gpu_routing_gate.py` ran from the source tree after both
`dlss_engine.py` and `dlssnr_host.dll` were removed: 32 checks, 0 failures, `RESULT: PASS`,
exit 0 (`bench_out/routing_gate_after_retire.log`). Its old-host tripwire no longer patches a
live module — there is none to patch — it installs one that raises, so a path still reaching
for the retired wrapper would be reported as such.

| route | entry the frame went through | motion transport | chain |
|---|---|---|---|
| job, guidance 0, host unnamed | `process_zero_motion` | `resident_zero_texture` | `legacy_pipe` |
| job, guidance 1, DIS | `process_host_motion` | `host_plane_upload` | `legacy_pipe` |
| job, guidance 1, NVOF | `process_device_flow` | `nvof_device_descriptor` | `legacy_pipe` |
| preview, guidance 0 | `gpu_zero_motion` | — | — |
| preview, guidance 1, DIS | `gpu_host_motion` | — | — |

Each job named no `dlss_backend` in the payload the way a pre-migration client does, resolved
to `gpu_flow_d3d12`, completed, and produced an ffprobe-readable deliverable (8 packets,
h264, 1280x720, yuv420p). The two previews returned real JPEG frames with no host error and
`host=dlss_gpu_host.dll abi=3` in the service's own log line.

### The diagnostics say which host they came from

`dlss_engine.get_log_tail` is gone from `dlss_service.py`, `server.py` and the front end: no
production module names it, and `grep` over those three plus `gui.py` and `pipeline.py` returns
nothing. A job record now carries `dlss_host` — `{"backend", "dll", "abi"}`, read off the
session that ran or refused it (`dlss_service.py:143`, written at `:1127`) — and
`dlss_host_error`, the GPU host's own `last_error` (`:1233`). `/api/diagnostics/job/{id}`
(`server.py:424`) returns those two and no log field, which
`tests/unit/test_export_backend_ui_contract.py` checks by asserting that neither `log_tail`
nor `dlss_log` nor the string `dlssnr_run.log` appears in the body. The host writes no log
file; the stale `dlssnr_run.log` still sitting at the repository root is untracked residue from
before the removal and nothing reads it.

### The distribution and the source tree

`bundle_paths.CONTENTS_BINARIES == ("nvngx_dlssnr.dll",)` — the retired host was dropped from
the manifest that the spec collects from, so the bundle cannot gain it back through a private
list. Two files left the tree:

* `dlss_engine.py` (10,836 bytes), the ctypes wrapper that resolved and loaded the old host;
* `dlssnr_host.dll` (54,272 bytes).

`nvngx_dlssnr.dll` (165,840,496 bytes) stays: it is NVIDIA's signed runtime, and the GPU host
opens it by path at session start. Driver-owned DLLs remain refused at spec level
(`bundle_paths.NEVER_BUNDLE`). `tests/unit/test_packaged_binaries.py` holds the three separate
facts that have to stay false — the manifest naming it, the spec keeping it, a production
module importing the wrapper — and adds that no production module may build a *path* to the
file, which is the difference between documentation that explains the retirement and a second
loader.

`DLSSInferenceError` no longer lives in the deleted wrapper: `dlss_errors.py` (2 exports,
imports nothing from the retired module) is what `dlss_service.py`, `dlss_gpu_engine.py`,
`pipeline.py` and `gui.py` raise, so a session refusal does not pull a DLL into the process to
say no.

### Scripts that measured the old host

Kept, and made honest about what they can no longer run:

* `native/dlss_gpu_host_smoke.py`, `native/dlss_gpu_zero_motion_smoke.py`,
  `native/dlss_gpu_host_motion_smoke.py` — the byte-parity arm against `dlssnr_host.dll` is
  now optional and *reports itself*: the first two print the arm as unavailable when the
  wrapper is absent, the third already caught any failure of that arm and skipped it. Their
  GPU assertions are unchanged and still hard.
* `native/gpu_flow_export_smoke.py` — its whole subject was a device-vs-old-host export A/B,
  and the A/B is now structurally impossible: the export core refuses any session without
  `process_device_flow`. It is a device-only gate (12 packets, `gpu_flow_d3d12` selected, zero
  NVOF D2H) that says the peer is gone.
* `tests/unit/gpu_transfer_export_gate.py` — the fourth argument was a *legacy-pipe*
  comparison and it was building the old host to make it; it now builds the GPU host, which is
  what the legacy pipe actually runs on.
* `tests/unit/benchmark_4k_scenarios.py`, `benchmark_e2e.py`, `collect_transfer_ledger.py`,
  `collect_transfer_abi.py` — archival. Their docstrings now say the figures describe the
  retired session and that re-running needs the old DLL back. `run_transfer_stability_round.py`
  is an evidence-audit script over an old commit's declared file list; untouched, and it names
  the deleted path as part of that commit's state, which is still true.
* `scratch/` — untouched historical gate logs and baselines.

### Evidence, limits, and what is next

Tests: `tests/unit` is **747 passed, 0 failed** (`bench_out/unit_suite_after_retire.log`,
175.95 s) with the two files deleted. The focused set this stage rewrote or re-anchored is 251
checks across 12 files — `test_export_backend_ui_contract` 24, `test_export_zero_motion_pipe`
21, `test_preview_host_motion` 23, `test_gui_image_zero_motion` 29 (was 28),
`test_packaged_binaries` 12, `test_preview_export_parity` 6, `test_export_pipeline` 9,
`test_gui_preview_gpu_host` 40, `test_gui_video_render_gpu_host` 39,
`test_gui_stream_export_gpu_host` 35, `test_gui_selftest` 12, `test_cancellation_e2e` 1. That
last one failed once, on a run taken while a PyInstaller build was competing for the GPU and
the disk: the decoder producer thread had not finished draining into the encoder pipe when
cancellation was asserted (`test_cancellation_e2e.py:167`). It passes in the full-suite run
above and on a quiet machine.

Front end: `DlssBackend` is a one-member type, `dlssHostRequiredByExport` is gone, the export
panel no longer writes a host, and the host row is a stated fact rather than a control
(`frontend/src/pages/DLSSPage.tsx:1039-1049`). `npx tsc --noEmit` is clean, and `frontend/dist`
was rebuilt so the copy the Electron shell actually loads matches the source the assertions
quote — the bundle in `dist/` predated the panel rewrite and would have failed its own test.
The Playwright
specs were rewritten to match — `tests/e2e/13_gpu_transfer_optin.spec.ts` is 5 tests, `--list`
resolves them — and **were not executed in this stage**; they need the Electron shell and a
real device session, which is the next verifiable item along with a frozen `gpu_transfer`
export.

Frozen build: `python -m PyInstaller --noconfirm DLSS5Tool_standalone.spec` completed in
508 s (`bench_out/frozen_build_after_retire.log`), and the bundle's own `--selftest` answers
`host=native/dlss_gpu_host.dll entry=process_zero_motion reset=1 rgb_mean=0.233
sha1=cd5095247ece` — the same frame bytes the source tree reports from the same probe, on a
bundle that carries neither `dlssnr_host.dll` nor the wrapper that loaded it. That is read off
the build, not inferred: `find` over the collected tree returns no such file, all five PyInstaller
TOCs (`Analysis`, `PYZ`, `COLLECT`, `PKG`, `EXE`) contain zero occurrences of either name, and
`native/` still ships its nine DLLs with `nvngx_dlssnr.dll` at the contents root.

Limits that this stage did not close: the packaged window's `gpu_transfer` chain is still not
exercised inside a bundle — `--selftest` draws a still and a preview, and never opens the
device transfer path — so "the frozen build exports on GPU transfer" remains unproven rather
than proven-and-fast. Performance remains `NOT_CHARACTERIZED`: nothing here measured a rate,
and the opt-in chains are still sold on function, not on speed.
*Both halves of that limit are now measurements rather than limits: the five browser tests
were executed, and the frozen bundle exported six frames on the device chain. The two
sections below this one are those runs, and the sentence above stays exactly as it was
written because it is what the stage before it left.*

## The five opt-in browser tests, executed against the rebuilt front end and real hardware

The stage above left `tests/e2e/13_gpu_transfer_optin.spec.ts` at "5 tests, `--list` resolves
them — **were not executed in this stage**". They are executed here, and the reason they had
never run is the reason a listing is not a result: one of them was wrong in a way only a run
could show.

**What the run actually attached to.** The real Electron shell
(`tests/helpers/electron_runner.js` launches `frontend/node_modules/electron/dist/electron.exe`
with `cwd=frontend`, and `frontend/electron/main.ts:189` loads the renderer from
`RENDERER_DIST = frontend/dist`), the renderer and shell both from the 27 Sep 22:23 rebuild
(`frontend/dist/index.html`, `frontend/dist/assets/index-COrjgDhS.js`,
`frontend/dist-electron/main.js` — two hours before this run, not a stale copy), and the real
Python backend it spawns (`tests/helpers/paths.js:12` → `F:\WhisperJAV\python.exe`, torch
2.10.0+cu128, `cuda=True`) on an RTX 5060 Ti, driver 617.14. No request is stubbed except
where a test says so below.

**The result.** `5 passed` in 1.3 m; `test-results/e2e-spec13-final-20260928.json` carries
`stats: {expected: 5, unexpected: 0, skipped: 0, flaky: 0}` and `errors: []`, with per-test
durations 16.32 / 17.36 / 16.07 / 13.75 / 13.81 s. Console logs:
`scratch/e2e_spec13_{,rerun_,final_}20260928.log`. The spec was invoked three times, and the
honest reading of that is:

1. **first run — `1 failed`, `4 passed`**: 14.3 asserted
   `expect(settled.status).toBe('failed')` and got `completed`;
2. **second run, after the fix — `5 passed`**, but with `--reporter=list` on the command line,
   which *replaces* the config's reporter array, so no JSON artifact was written and
   `tests/test-results.json` still held the 27 Sep 21:57 content (5 skipped). A passing run
   with no artifact is a passing run nobody can audit;
3. **third run, no reporter override — `5 passed`** and the JSON above. That is the run this
   section quotes.

**14.3 was the test's bug, not the product's.** The job it queues is meant to be a
`gpu_transfer` export handed a CPU flow plane, which the device chain refuses because it wants
its flow descriptor on the card. It built that settings blob as
`{flow_engine: 'dis', ...GPU_READY_SETTINGS}` — and `GPU_READY_SETTINGS` carries
`flow_engine: 'nvof'`, so the spread last over-wrote the very field under test. The queue
received a fully valid device-chain job and **exported it for real**, which is why the status
was `completed`. The fix is the spread order (`tests/e2e/13_gpu_transfer_optin.spec.ts:221-228`),
and no production line was changed for this section: nothing on the service side was wrong.

**The accident is worth keeping, because it is evidence.** That refused-to-be-refused job ran
through the browser dialog, the queue API, `dlss_service`, and the device chain on hardware and
left `tests/fixtures/gpu_transfer_cpu_flow.mkv` behind: 12,704,588 bytes, `av1` Main,
1280x720, `yuv420p`, `tv` range, `smpte170m`, 600 packets over 5.015 s at 120/1. Its size is
identical to `gpu_transfer_unpaired.mkv` that the previous stage's run of the same fixture
produced. So a real front-end-clicked `gpu_transfer` export does complete on this machine —
and after the fix, no browser test in this file does one any more, because 14.3 now correctly
demands the refusal. That is the trade this section records rather than hides.

**What the five tests do exercise**, per test: 14.1 and 14.2 assert the request body the
Deliver page actually puts on the wire and the job the service then stores (`auto` +
`gpu_flow_d3d12`; `gpu_transfer` + `gpu_flow_d3d12`; the opt-in panel naming NVOF, 50 %, 100 %,
8-bit SDR, NVENC, "拒绝", "不会自动回退", and none of 更快/加速/性能提升/零拷贝) — they queue
and never render. 14.3 queues a real job and reads its failure reason both from the queue API
and from the row text and diagnostics panel. 14.4 asserts a 400 from the real service for
`export_backend=gpu-transfer` (the hyphen) with the reason in `detail`, the queue still empty,
and then — the one deliberately stubbed half — that whatever `detail` says reaches the operator
verbatim through a dialog. 14.5 draws a **real** preview frame through the real backend and
reads `kind=gpu_zero_motion` out of the service's own log, asserts no `kind=legacy` line and no
ERROR line, that the host select is gone, and that a `dlss_backend=host_pointer` write is a 400
naming `host_pointer`, `dlssnr_host.dll` and `gpu_flow_d3d12` without merging.

**Limits of this section.** Browser opt-in wiring plus one real preview route, not a browser
export: after the fix no Playwright test in this file drives a `gpu_transfer` job to a
deliverable, and none drives a `legacy` export end to end either (that is what the queue gates
in `tests/unit` cover). One card, one fixture (`fixture_120fps.mkv`), one worker. No new test
was needed to find the 14.3 defect, and none was added here: the change is one spread reordered
and three lines of comment saying why, all inside the spec, and the spec's own artifact is the
proof.

## The frozen bundle exports on the device chain, and everything it loaded came from itself

The limit the previous stage ended on — "`--selftest` draws a still and a preview, and never
opens the device transfer path" — was not a missing test but a missing **entry**: `export_backend`
is a queue setting, the queue lives in `dlss_service`, and the standalone bundle does not ship
the service. So the packaged window had no way to reach `gpu_transfer` at all, and "the frozen
build exports on GPU transfer" could not be asked, let alone answered.

**The smallest bounded way in** (`gui.py:1689-1958`): a CLI entry,
`DLSS5Tool.exe --gpu-transfer-selftest [frames=6] [video=path] [out=path] [report=path]`,
dispatched in `main()` (`gui.py:1964`) before any Tk root exists, that calls
**`pipeline.export_dlss_stream`** with `job_settings.export_backend = "gpu_transfer"` — the
production entry, the same one the service drives, with no second pipeline, no re-implemented
loop and no re-derived counters. Its settings are the chain's precondition set
(`gui.py:1697`): `guidance_mode 1`, `flow_engine nvof` at `flow_quality 4M`, `working_scale
0.5`, `residual_pyramid True`, `preset 1` and the panel's defaults otherwise,
`dlss_backend gpu_flow_d3d12`. It is bounded by construction: frames are clamped into
`GPU_TRANSFER_SELFTEST_FRAMES = (3, 8)` (`gui.py:1692`), the fixture is 1280x720/24 fps, the
work directory is `tempfile.mkdtemp`, and the only file it ever writes into a caller-visible
place is the JSON report it was told to write. `transcode_settings={}` is deliberate: the
encoder plan is the chain's default, not an invention of the probe.

**The build.** PyInstaller 6.22.3 on Python 3.11.15 (`hermes` venv, torch 2.13.0+cpu),
`DLSS5_FFMPEG_DIR=F:\WhisperJAV\Library\bin`, `--distpath F:\tmp\pkgdist2 --workpath
F:\tmp\pkgbuild2` — outside the repository, so the checkout's `dist/` and `build/` are
untouched — exit 0 in 316 s (`F:\tmp\frozen_build_gpu_transfer_selftest_v2.log`). The bundle is
1,208,978,812 bytes over 7,395 files; `_internal/native/` holds exactly the nine bridges;
`_internal/native/dlss_gpu_host.dll` is `c54fda99e6475845b609bf77fcbcdaf9758247dd`, the same
SHA-1 as the working tree's; `_internal/` carries `ffmpeg.exe`, `ffprobe.exe` (both 7.1.1, the
build machine's), the seven `av*`/`sw*` libraries and `nvngx_dlssnr.dll`. `find` over all 7,395
files for `dlssnr_host` returns **zero**, and the probe's own whole-bundle walk reports the same
(`retired_host_files: []`). One earlier build of a nearly identical tree sits superseded at
`F:\tmp\pkgdist` — 1,208,978,614 bytes against the 1,208,978,812 above, the gap being the two
renamed report keys; it was not inspected for this section and is left there unclaimed, because
deleting it is not this stage's business.

**The one frozen run** (`bench_out/frozen_gpu_transfer_selftest_20260928.log` and the `.json`
beside it; run from `cwd = _internal`, 6 frames):

    GPU_TRANSFER_OK frames=6 packets=6 report=F:\tmp\gts_frozen_v2.json

exit 0, 28 named checks, `failed_checks: []`. The deliverable, ffprobe'd with the bundle's own
`ffprobe.exe`: `av1`, `1280x720`, `yuv420p`, `tv`, `smpte170m`, **6 packets**, pts
`0.000000…0.208000`, 235,604 bytes — the packet count is what says six frames left the encoder,
and it is read from the container rather than from what the exporter claimed. The chain's own
report line: `export_backend_requested/actual = gpu_transfer/gpu_transfer`,
`export_backend_fallback_reason = null`, `gpu_transfer_working_dims [640, 360]` from 1280x720 at
`working_scale 0.5` with `gpu_transfer_resampling = exact_two_x`, `gpu_transfer_copy_pass =
false`, `gpu_transfer_encoder_codec = av1_nvenc`, `gpu_transfer_residual_backend =
gpu_residual_bridge_d3d12`, `gpu_transfer_nvof_bridge_loaded = true`, per-frame records 6/6
`export_backend_actual = gpu_transfer` on host `gpu_flow_d3d12`.

**The transfer claim is the counters, and they are of two kinds.** Device-side counts that have
to equal the frame count — `decode_frames_decoded`, `decode_device_frame_count`,
`nv12_to_rgba_convert_count`, `working_color_process_count`, `encoder_frames_submitted`,
`encoder_packets_written`, `gpu_transfer_dlss_flow_calls` — all 6. Boundaries that a frame on
this chain may not cross, all exactly 0 — `decode_host_copy_bytes`, `decode_host_copy_count`,
`dlss_host_color_stage_bytes`, `dlss_output_readback_bytes`, `residual_output_readback_bytes`,
`encoder_host_pixel_bytes`. `gpu_transfer_dlss_zero_motion_calls = 0` and the flow entry served
6, so motion reached DLSS. NVOF's own device-input counters moved —
`nvof_device_input_copy_count_total = 7`, `nvof_device_input_bytes_total = 6,451,200` — and are
recorded, not asserted zero: they count uploads the flow engine makes **on the card**, and a
guided run that did not move them would be the suspicious one. They are excluded from the zero
list at `gui.py:1706-1710` with that reason.

**Packaged resolution, checked two ways.** The probe's way: `ffmpeg_path`, `ffprobe_path`,
`native_dir` and the `gpu_transfer_bridge_dll_directory` the chain reported are all under
`_internal` (`binaries_outside_contents: []`), and all five production modules that ran —
`pipeline`, `gpu_transfer_export`, `optical_flow`, `transfer_profiler`, `dlss_gpu_engine` —
have a `__file__` under `_internal` (`module_paths_outside_contents: []`). So no part of the run
read a tool or a source file out of the checkout; the fixture and the deliverable lived in
`C:\Users\18392\AppData\Local\Temp\dlss5_gpu_transfer_selftest_*`.

The probe's answers are resolution facts, though, and one thing it cannot know is which
`avcodec-61.dll` Windows actually *mapped* — the bundle's, or a machine install's, when both are
reachable. `scratch/_frozen_bridge_imports.py` asks the loader: it registers `_internal` then
`_internal/native` in that order, loads the eight libraries the export loads by name (seven
bridges plus `dlss_gpu_host`), and enumerates mapped modules
with `EnumProcessModules` — with `F:\WhisperJAV\Library\bin`, which carries the **same seven
FFmpeg file names**, on the process PATH (`bench_out/frozen_bridge_imports_20260928.log`, exit
0). All seven `av*`/`sw*` mapped from `_internal`, as did every one of the eight and the
`nvof_d3d12_interop` they pull in; `d3d12.dll` and `D3DCOMPILER_47.dll` from `C:\WINDOWS\SYSTEM32`, which is the
design; no `dlssnr_host` mapped. `nvcuda`/`nvofapi64`/`nvencodeapi` are absent from that list
because no session or engine was started in that process, not because they were refused. It runs
in the build interpreter, outside the frozen one — the same boundary the packaging stage
declared for its CDLL check.

**A regression the still frame still guards.** `--selftest` on this same bundle, after the
export probe went into the file: `DLSS_OK (240, 320, 4) | host=native/dlss_gpu_host.dll
entry=process_zero_motion reset=1 rgb_mean=0.233 sha1=cd5095247ece | torch=2.13.0+cpu
cuda=False`, exit 0 (`bench_out/frozen_selftest_after_gpu_transfer_probe_20260928.log`) — the
same `sha1` the packaging stage recorded and the same the source tree reports. The new entry
displaced nothing.

**Source tree, same probe, same verdict**: `GPU_TRANSFER_OK`, 28/28 checks, identical ffprobe
view down to 6 packets and 235,604 bytes. **Not byte-identical output** (deliverable
`c9cb5af96d56` frozen vs `227ddca93b44` source) and the reason is worth stating: the two layouts
resolve different FFmpeg builds — the bundle carries 7.1.1, `pipeline._bundle_ffmpeg()` in the
checkout answers scoop's 8.1.2 — so their generated fixtures differ (`940cb4abc3a0` vs
`67e0f8d5d521`). And the fixture is not bit-reproducible even against one tool: the same
1280x720/24 fps/6-frame command returned 78,874 bytes twice with two different SHA-1s. The probe
hashes what it made and reports it; no gate depends on it being constant.

**`tests/unit/test_gui_gpu_transfer_selftest.py`, 12 tests, no card needed.** The fake chain
answers with the fields `export_gpu_transfer` writes, so every test is a decision the probe
makes about a report it was handed: a frame that crossed to host memory is not a pass; a silent
fallback to the pipe is not a pass; a run whose flow-call count says the motion never arrived is
not a pass; five packets for six frames is not a pass; a deliverable that came back resized is
not a pass; a returned path that is not the output is not a pass; a refusal is written down with
the chain's own reason and the session still closes; the frame count stays in 3…8; the flag is
dispatched before any window exists. Writing them found one real fault: `facts["ffprobe"]` held
the resolved tool path on the way in and the probe result later, so on **any** failure the
closing `print` raised `AttributeError` and the probe could not report its own error — the
rename to `ffmpeg_path`/`ffprobe_path` is `gui.py:1791`. Focused set for a `gui.py` edit: 179
tests across the seven gui/packaged unit files, `179 passed` in 12.32 s, run once. `tests/unit`
as a whole was not re-run: this stage changed no native source, no ABI, no pipeline arithmetic
and no service route, and the only production edit is a CLI entry plus two report keys in a file
whose behavior is covered by that focused set.

**What this does not close.** Six frames at 1280x720, one settings tuple, one working scale
(0.5, the `exact_two_x` branch — a plan-driven scale and the `copy_pass = true` branch are still
unexercised in a bundle), guidance 1 only, `av1_nvenc` only, CQ 24 default, no audio, no 4K, no
cancellation, no queue: the bundle still has no service, so a packaged operator still cannot
click this path, and the window's own menus do not reach it. `NOT_CHARACTERIZED` still: the run
printed a profiler table (6 frames, `FPS: 6.61`, `frame_total` mean 96.66 ms, max 545.06 ms) and
that is a cold-start artifact of one six-frame process — shader compile, NVOF init, encoder
open — not a rate, nothing was compared and no throughput claim is being made from it.

**The next smallest verifiable gap** is therefore the one parameter this probe never moved:
run the same frozen entry at a plan-driven `working_scale` (0.63 or 0.67) and at 1.0, so the
resampling plan and the `gpu_transfer_copy_pass = true` branch are exercised inside a bundle
rather than only in the source-tree gate. Everything it needs already exists; it is one setting
on one line of the probe plus the two fixture extents that make those branches reachable.

## The frozen bundle serves the other two branches, and the delivery panel reaches the device chain

**The probe now moves the one parameter it never moved.** `--gpu-transfer-selftest` takes
`scale=` (or `working_scale=`) and still defaults to 0.5 (`gui.py:1700`, `gui.py:2128`); the 3…8
frame window is unchanged. A named ratio is checked twice before anything is opened: against the
range this build serves, which is read out of
`gpu_transfer_export.WORKING_PERCENT_MIN`/`_MAX` and `working_percent` rather than restated
(`_gpu_transfer_served_range`, `gui.py:1752`), and against the source extents, through
`_gpu_transfer_selftest_branch` (`gui.py:1770`). Both answers are derived from the production
functions the gate itself uses — `pipeline.resolve_working_dims`, `plan_driven_scale`,
`plan_tap_widths` against `LANCZOS_MAX_TAPS`/`RESIDUAL_MAX_TAPS` — so a drift between the probe's
expectation and the gate is a refusal or a failed check, never a pass. The extent check raises a
`ValueError` **before** `dlss_gpu_engine.Live` is constructed, which is the whole reason it can
refuse without a device session.

Six new report keys state the branch instead of trusting it: `reported_working_scale`,
`reported_resampling`, `reported_working_dims`, `reported_residual_backend`,
`per_frame_residual_backend`, `reported_copy_pass`, plus `no_leaked_intermediate` over
`dlss_gpu_transfer_*` in the temp directory. `reported_copy_pass` compares against
`gpu_transfer_export.needs_copy_pass(probe, transcode_settings, with_audio, out_path)` — the same
call the chain makes. **The earlier note that 1.0 "pairs with `copy_pass = true`" was wrong and is
corrected here**: `copy_pass` is a container and mux clause (non-`.mkv` output, audio, subtitles,
attachments, metadata, chapters, extra args), not a property of a scale. The identity run reaches
it honestly by naming a `.mp4` deliverable. The identity branch also does not carry the two stages
the downscaled branches measure, so its check list swaps
`counter:working_color_process_count_measured` and `zero:residual_output_readback_bytes_measured`
for `unmeasured:working_color_process_count_measured` and
`unmeasured:residual_output_readback_bytes_measured`. That is the actual contract: a bridge that
never opened reports `None`, and a measured `0` is a different claim, so the probe says which one
it saw rather than flattening them.

**Two real frozen exports, four frames each, from one bundle.** The bundle was built once into
isolated output — `DLSS5Tool_standalone.spec` with `--distpath F:\tmp\pkgdist3 --workpath
F:\tmp\pkgbuild3`, `DLSS5_FFMPEG_DIR=F:\WhisperJAV\Library\bin`, PyInstaller 6.22.3 / Python
3.11.15, exit 0 in 460.069 s (`F:\tmp\frozen_build_scale_gates_v3.log`). `--selftest` on it still
answers `DLSS_OK (240, 320, 4) | host=native/dlss_gpu_host.dll entry=process_zero_motion reset=1
… sha1=cd5095247ece` (`bench_out/frozen_selftest_pkgdist3_20260928.log`) — the same hash the
packaging and opt-in stages recorded, so the widened probe displaced nothing. Both runs were
started from `_internal`, so every path in their reports is a layout claim.

- **plan-driven** — `GPU_TRANSFER_OK frames=4 scale=0.63 packets=4`, 33 checks, 0 failed
  (`bench_out/frozen_gpu_transfer_selftest_063_20260928.{json,log,mkv}`). `resampling =
  "plan_driven"`, `working_dims = [808, 456]` from 1280x720, `residual_backend =
  "gpu_residual_bridge_d3d12"` on every frame, `copy_pass = false`, `gpu_transfer_dlss_flow_calls
  = 4` against `gpu_transfer_dlss_zero_motion_calls = 0`, per-frame host `gpu_flow_d3d12`,
  `nvof_bridge_loaded = true`, and the six host-crossing counters — `decode_host_copy_bytes`,
  `decode_host_copy_count`, `dlss_host_color_stage_bytes`, `dlss_output_readback_bytes`,
  `residual_output_readback_bytes`, `encoder_host_pixel_bytes` — all exactly `0`. Resolution came
  from the bundle: `ffmpeg_path`/`ffprobe_path` under `_internal`, `native_dir = _internal\native`,
  `binaries_outside_contents = []`, `module_paths_outside_contents = []`,
  `retired_host_files = []`, `leaked_intermediates = []`. ffprobe, read back independently with
  the bundle's own tool: matroska, `av1`, 1280x720, `yuv420p`, `tv`/`smpte170m`, 4 packets,
  192,942 bytes.
- **identity + copy pass** — `GPU_TRANSFER_OK frames=4 scale=1 packets=4`, 33 checks, 0 failed
  (`bench_out/frozen_gpu_transfer_selftest_ws100_20260928.{json,log,mp4}`). `resampling =
  "identity"`, `working_dims = [1280, 720]`, `residual_backend = "none"` and per-frame
  `["none"]`, `copy_pass = true` because the deliverable was named `.mp4`, the same flow/zero-motion
  split (4/0), the same four measured crossing counters at `0`, and the two `unmeasured:` checks
  above in place of counters. Independent ffprobe: `mov,mp4,m4a` container, **one** stream
  (`0,video`), `av1` 1280x720 `yuv420p` `tv`/`smpte170m`, 4 packets, 196,927 bytes — the rewrap
  happened and produced a real MP4.

The already-passing 0.5 run was **not** repeated. `elapsed_sec` (2.777 and 2.519, fixture
generation included) is a bounded probe's wall clock; nothing was compared and no rate is claimed.

**Refusals, same bundle, no frame decoded.** `scale=0.3` exits 1 with
`working_scale=0.3 is not a ratio this backend serves; ask for 0.25, 0.5, 0.63, 0.67, 0.75, 1`
and writes no deliverable (`bench_out/frozen_gpu_transfer_refusal_03_20260928.json`,
`branch = null`). `scale=0.25` is in the served set but not at these extents: it exits 1 with
`working_scale=0.25 resolves to [320, 176] from 1280x720, which is no chain this backend serves
for these source extents; the gate would refuse the job before it decoded a frame`
(`…_refusal_025_20260928.json`) — which is the second half of the same guard, and the reason the
message names the extents as well as the ratios.

*Both of those messages are the record of the bundle as it was built that day, and the range
replaced the list the work on this page describes.* Neither refusal is what the current tree
prints: `0.3` is a served percentage now, so the surviving refusal is the one a non-integer
percentage gets — `working_scale=0.333 is not a whole percentage between 25 and 100, which is the
range this backend serves` — and `0.25` at 1280x720 is accepted, its 17-tap window fitting the
widened kernel. The extents half of the guard is unchanged in kind: a ratio whose resolved size
the job did not declare, or which leaves no axis below the source once 8-alignment is applied,
still refuses before a session opens.

**A positive path through the interface, and two faults it found.**
`tests/e2e/14_gpu_transfer_positive_export.spec.ts` (test `15.1`) runs the real service and the
real card against the new 6-frame/1280x720/24 fps silent fixture
(`tests/fixtures/fixture_gpu_transfer_short.mkv`, 78,871 bytes, one video stream — not the
600-frame 120 fps fixture): it *clicks* the working-scale preset, the guidance select and the
export-backend select, polls `/api/settings` until all three are stored together, captures the
`/api/queue/add` body, clicks 全部渲染, and lets the finished job's own record answer. `1 passed
(22.5 s)`, test duration 21.5 s, job `completed` at `total_frames = 6`
(`bench_out/e2e15_gpu_transfer_positive_20260928.json`, backend line in
`bench_out/e2e15_stdout_20260928.log`): `export_backend_requested`/`actual` both `gpu_transfer`,
`fallback_reason = null`, `gpu_transfer_working_scale = 0.5` → `[640, 360]` on
`resampling = "exact_two_x"`, `residual_backend = gpu_residual_bridge_d3d12`, flow calls 6 and
zero-motion calls 0, encoder `av1_nvenc` with `weighted_pred=0, spatial-aq=1, temporal-aq=0,
aq-strength=8`, `copy_pass = true` (the panel's `copy_attachments`), six crossing counters at `0`,
bridge DLL directory the one installation, and the deliverable
`tests/fixtures/fixture_gpu_transfer_short_dlss.mkv` — 519,066 bytes, one video stream, 6 packets,
0.249 s, `av1` 1280x720 `yuv420p` `tv`/`smpte170m`. The same test asserts
`native/dlssnr_host.dll` is absent and the recorded host is `dlss_gpu_host.dll`.

Getting there took two fixes, because the panel's payload had never been negotiated:

1. `gpu_transfer_export.py:92`/`gpu_transfer_export.py:104` — the gate refused
   `transcode_settings['custom_factor']`, `['suffix']` and `['finish_action']`, and every job the
   delivery page builds carries all three. The opt-in was therefore **unreachable from the UI**:
   the job died at negotiation before a pixel moved. `custom_factor` joins the resolution family
   whose effect is already refused by the resolved size (`cannot be resized to WxH`), and
   `suffix`/`finish_action` are named `QUEUE_ONLY_KEYS` — an output name and a power action, read
   before the export and after the queue, never by either chain. A key that really is an encoding
   choice is still refused.
2. `gpu_transfer_export.py:185` with `pipeline.py:3036` — the tune question was asked of
   `_bundle_ffmpeg()` while the encoder is a library this process bound from
   `ffmpeg_dll_directories()[0]`. On this machine those are two different FFmpeg installs, and
   they disagree: PATH's 8.1.2 lists `uhq` as an `av1_nvenc` tune, the loaded build does not
   (`-tune … (from 1 to 4)`). So the gate approved `uhq` and NVENC refused it once the frame had
   already been decoded, converted, resampled, driven through DLSS and residual —
   `nvenc_last_error reported -1: tune: Invalid argument (-22)`, exactly the mid-run failure the
   up-front gate exists to prevent. `encoder_cli()` puts the question to the installation that
   will do the encoding; the plan then sends `hq`, which is that encoder's own default.

Five tests were added to `tests/unit/test_gpu_transfer_export_backend.py` (85 → 90 passed,
1.57 s) so neither fix is a one-machine answer: the panel's default payload verbatim negotiates
with no unsupported-key reason; an unknown encoding key is still a refusal; a `custom_factor` that
`resolution_mode` actually acts on is still refused through its resolved 7680x4320; `encoder_cli()`
prefers the CLI beside the loaded libraries and falls back to the tool this process would run; and
`tune_available` false sends `hq` while true sends `uhq`. The UI needed one attribute to be
clickable at all — the preview toolbar has its own “50%” button, which only zooms the canvas, so
the working-scale presets are named `工作分辨率缩放 50%` etc. (`frontend/src/pages/DLSSPage.tsx:831`,
matching the existing `aria-label` convention on the two selects) and the front end was rebuilt
(`frontend/dist/assets/index-DjPL-Myq.js`). The spec also deletes a stale deliverable first: the
panel sends `skip_existing`, and a run that skipped had `status = "completed"` with **no** backend
record at all — which the `export_backend_requested` assertion now catches instead of passing on.

**Gates run for this stage.** `tests/unit/test_gui_gpu_transfer_selftest.py` — 25 passed in 7.74 s
(the branch table pins 0.5/0.63/0.75/1.0 with their dims and shape, `0.25` at 1280x720 as `None`,
the three refusal inputs as *no export call and no Live session*, `copy_pass` differing between
`.mkv` and `.mp4` at the same scale, and a leaked `dlss_gpu_transfer_*` file as a failure). The
three files this stage's edits touch — 136 passed in 8.16 s. `tests/unit` as a whole — **777
passed** in 132.25 s. `npx tsc --noEmit` — exit 0. `npx playwright test --list` — 28 tests in 14
files. Spec 13's five opt-in tests were not re-run, per instruction; the only thing this stage
changed under them is one added `aria-label`, and the assertions they make read panel copy, not
button names.

**What this does not close.** One bundle, one settings tuple, `guidance_mode = 1` only,
`av1_nvenc` only, 1280x720 sources only, 4–6 frames, no 4K inside a bundle, no audio in the
frozen probe, no cancellation, no multi-job queue, and the identity branch reached its
`copy_pass = true` through a container name rather than through audio or metadata clauses. `0.25`
remains extent-dependent: served at 4K, refused at 720p because its window needs 17 taps.
`pkgdist3` was built at 01:37, so it contains **neither** gate fix nor the `aria-label` — those are
proven in the source tree, and the two frozen runs are unaffected by them by construction (the
probe negotiates with an empty `transcode_settings`, and in a bundle `encoder_cli()` and
`_bundle_ffmpeg()` are the same `_internal\ffmpeg.exe`, which both reports confirm: the first
bridge DLL directory *is* `dirname(ffmpeg_path)`). `NOT_CHARACTERIZED` still holds: the profiler
tables these runs printed are cold-start artifacts of bounded processes, and no throughput, speed
or parity claim is being made from them.

**The next smallest verifiable gap** is the ordering just described: no packaged artifact has ever
carried the panel's own default transcode payload. One rebuild from the current tree plus test
`15.1` pointed at the frozen `DLSS5Tool.exe` instead of the checkout closes it — same spec, same
fixture, one setting in the helper — and it would make the two gate fixes above claims about the
distribution rather than about the source tree.

## The packaged build takes the delivery panel's own payload, verbatim

The previous stage ended on a sentence that cannot be executed: it proposed pointing test
`15.1` at the frozen `DLSS5Tool.exe`. **That is not possible and is retracted here.** The
standalone bundle ships no service (`export_backend` is a queue setting and the queue lives in
`dlss_service`), so Electron Playwright has no backend to talk to through a packaged window —
the browser test can only ever drive the checkout. What *was* reachable is the other half of the
same claim: `pkgdist3` was built at 01:37, before both gate fixes, and its frozen runs
negotiated with `transcode_settings={}`. So no packaged artifact had ever carried the 51-key
dictionary the Delivery page emits on every job. This stage rebuilds and hands the probe that
dictionary.

**The payload was captured, not composed.** `tests/e2e/14_gpu_transfer_positive_export.spec.ts`
already reads the `POST /api/queue/add` body its own click produces; it now also writes it,
env-gated (`DLSS_E2E_TRANSCODE_ARTIFACT`) and only when a file is named, so the test's behavior is
unchanged. One bounded run — the 6-frame fixture, never the 600-frame one — `1 passed (25.8s)`
(`bench_out/e2e15_ui_payload_run_20260928.log`), job `30661660` `completed`, `total_frames = 6`.
The artifact is `F:\tmp\ui_payload_20260928\transcode_settings.json`: the envelope carries
`captured_at`, the spec and test names, the endpoint, the wire body's byte count and `sha256`
(`fdc3c34c23b5…`), the job's `input_path`/`output_path` and its `settings` alongside the
`transcode_settings` object itself (envelope sha1 `f583be809823`). The bare object the probe
consumes is `transcode_settings.ui_payload.json` — 1,305 bytes, sha1 `f70a7bbc5392` — and
`provenance.json` records the canonical-JSON sha256 (`4541b6f1064e…`) of the object as extracted
from the envelope and as re-read from the file, which are the same: **51 keys, round-trip
identical, no hand-written defaults dictionary anywhere in this stage.** It is the panel's default
state: `codec av1_nvenc`, `container .mkv`, `cq 16`, `preset p7`, **`tune uhq`**,
`multipass fullres`, `rate_mode cq`, `aq_mode spatial`/`aq_strength 8`,
`weighted_prediction false`, `resolution_mode original`, `copy_attachments true`, and the three
always-present keys the last stage unblocked (`suffix _AV1_P7_UHQ_Q16`, `finish_action none`,
`custom_factor 1`).

**The probe moved its second parameter.** `--gpu-transfer-selftest` takes
`transcode_settings=path` (`gui.py`): the file is read by
`_gpu_transfer_selftest_transcode_settings`, which requires a JSON **object** and refuses
anything else — non-JSON, a list, non-UTF-8, an empty file, a missing path — and it refuses
*before* the fixture is generated and before `dlss_gpu_engine.Live` is constructed, so a bad
payload leaves no artifact of a run that never happened. The dictionary is handed unchanged to the
same `pipeline.export_dlss_stream` the queue would hand it to; the probe filters, fills or renames
nothing, because `gpu_transfer_export.negotiate` is the gate that decides what a key means and a
probe that pre-filtered would be a weaker copy of it. The report gained
`transcode_settings_file` (path/bytes/sha1), `transcode_settings_accepted` and `encoder_options`,
and four checks that exist **only** for a run that was handed a payload —
`requested_encoder_codec`, `requested_encoder_cq`, `requested_encoder_tune_reported`,
`encoder_tune_matches_encoder` — so a default run still reports the same 33 checks it always did.
The 3…8 frame window, the scale gates, the bundle containment lists, the branch checks and the
refusal paths are untouched.

**The chain now says what it handed the encoder.** `gpu_transfer_export.py` reports
`gpu_transfer_encoder_options` beside the codec and AVOptions it already reported: the resolved
struct (`preset`, `tune`, `multipass`, `rate_control`, `constant_quality`, `bitrate`, `gop`,
`color_matrix`) as the bridge received it, through `_encoder_option`. This exists because a
request and a resolution are different facts — a tune the loaded build does not have is downgraded
in `plan_encoder`, and the probe's own `cq` default is 24 while the panel asks for 16 — and the
container cannot tell anybody which one reached NVENC. The probe's tune check does not re-derive
that downgrade; `_gpu_transfer_selftest_tune_held` re-asks the *encoder's* own predicate
(`pipeline._supports_tune_uhq` of `gpu_transfer_export.encoder_cli()`) and requires the reported
value to be consistent with the answer.

**The build.** PyInstaller 6.22.3 / Python 3.11.15 (`hermes` venv),
`DLSS5_FFMPEG_DIR=F:\WhisperJAV\Library\bin`, `--distpath F:\tmp\pkgdist4 --workpath
F:\tmp\pkgbuild4` — one build of the current working tree, into isolated output, the checkout's
`dist/` and `build/` untouched. Exit 0 in 436.884 s
(`bench_out/frozen_build_ui_payload_v4_20260928.log`). 7,394 files / 1,208,986,349 bytes, and the
file **name** set is identical to `pkgdist3`'s (`diff` of the two sorted lists is empty), so
nothing was dropped or gained but content. `_internal/native` holds the nine bridges;
`_internal/dlss_gpu_host.dll` is `c54fda99e6475845b609bf77fcbcdaf9758247dd`, the working tree's
own; `ffmpeg.exe`/`ffprobe.exe` are 7.1.1 with the seven `av*`/`sw*` libraries beside them. A
`find` over the whole bundle for `dlssnr_host` returns **zero**. Asked directly, this bundle's
`av1_nvenc` advertises `-tune` as `hq/ll/ull/lossless` and **has no `uhq`** — the authoritative
answer for the encoder that ships, from the same installation the bridges bind.

**One frozen run, first attempt, no refusal to fix.** Started from `_internal`, 6 frames at the
supported `0.5`, the captured payload, and a fresh deliverable path
(`F:\tmp\gts_frozen4_payload.mkv`, removed beforehand): `GPU_TRANSFER_OK frames=6 scale=0.5
packets=6`, exit 0, **37 named checks, `failed_checks: []`**, `elapsed_sec 3.002`
(`bench_out/frozen_gpu_transfer_selftest_payload_20260928.{log,json,mkv}`). The requested
dictionary arrived whole — `transcode_settings_file.sha1 = f70a7bbc5392`, 51 accepted keys — and
what the encoder received was `av1_nvenc`, `preset p7`, `rate_control vbr`,
**`constant_quality 16`** (the panel's `cq`, not the probe's 24), `gop 48`, AVOptions
`weighted_pred=0, spatial-aq=1, temporal-aq=0, aq-strength=8`, and **`tune hq`**: the shipped
encoder does not have `uhq`, the plan said so, `encoder_tune_matches_encoder` passed, and the
value was accepted by evidence rather than by assertion — `nvenc` opened, six frames submitted,
six packets written, no bridge error anywhere in the run.

**The container and the copy pass.** The panel's `copy_attachments: true` makes this payload a
copy-pass request even on a `.mkv` target, and the run is on that branch:
`gpu_transfer_copy_pass = true`, `reported_copy_pass` and `no_leaked_intermediate` both pass, and
`leaked_intermediates: []` over `dlss_gpu_transfer_*` in the temp directory — independently
confirmed afterwards by a directory listing that finds none. `skip_existing: true` is in the
payload and is a `QUEUE_ONLY` key the chain never reads (the queue acts on it before the export),
and the fresh output path means nothing could have been skipped anyway: the deliverable is a real
file, probed with the **bundle's own** `ffprobe.exe` — matroska, exactly **one** stream
(`0, video`), `av1` 1280x720 `yuv420p` `tv`/`smpte170m`, **6 packets**, 528,409 bytes.

**Packaged resolution, checked two ways again.** `ffmpeg_path` and `ffprobe_path` are
`_internal\ffmpeg.exe`/`_internal\ffprobe.exe`, `native_dir = _internal\native`, and the
`gpu_transfer_bridge_dll_directory` the run reported is
`["F:\\tmp\\pkgdist4\\DLSS5Tool\\_internal"]` — one entry, the bundle, not the machine install
that `ffmpeg_dll_directories()` also lists second. `binaries_outside_contents: []`,
`module_paths_outside_contents: []`, `retired_host_files: []`; per-frame records 6/6 on
`export_backend_actual = gpu_transfer`, `dlss_backend_actual = gpu_flow_d3d12`,
`residual_backend = gpu_residual_bridge_d3d12`; `gpu_transfer_dlss_flow_calls = 6` against
`gpu_transfer_dlss_zero_motion_calls = 0`; the six host-crossing counters —
`decode_host_copy_bytes`/`count`, `dlss_host_color_stage_bytes`, `dlss_output_readback_bytes`,
`residual_output_readback_bytes`, `encoder_host_pixel_bytes` — all measured `0`, while NVOF's
device-input counters moved (`7` copies) and are recorded, not asserted.

**The interface's record and the packaged record are the same record.** Three-way comparison of
the chain's own report lines: the UI job (48 fields) against the source-tree run with the same
payload — **48 of 48 identical**; that run against the frozen run — **48 of 49 identical**; and
the UI job against the frozen bundle — **47 of 48 identical**. The single difference in every
pairing is `gpu_transfer_bridge_dll_directory`: the checkout binds
`F:\WhisperJAV\Library\bin`, the bundle binds its own `_internal`. That is the one field that is
*supposed* to differ, and it is the containment claim. Two caveats stated rather than smoothed
over: the fixtures are not the same file (the panel job rendered
`tests/fixtures/fixture_gpu_transfer_short.mkv`, 78,871 bytes `2967f5e3ff49`; the bundle generated
its own 1280x720/24 fps/6-frame testsrc2 with its own FFmpeg, 78,871 bytes `2ccecb18638b` — same
content and the same 78,198 bitstream bytes submitted, a different muxing hash), and the
deliverables are not byte-identical (519,066 in the source tree against 528,409 in the bundle,
the AV1 encoders being 8.1.2 and 7.1.1).

**Refusals and regressions, same bundle.** `--selftest` still draws the still frame: `DLSS_OK
(240, 320, 4) | host=native/dlss_gpu_host.dll entry=process_zero_motion reset=1 rgb_mean=0.233 …
sha1=cd5095247ece` (`bench_out/frozen_selftest_pkgdist4_20260928.log`) — the same hash the
packaging, opt-in and scale-gate stages recorded, and the only file the bundle gained since
collection is that probe's own `_selftest.txt`. `scale=0.3` still exits 1 naming the served ratios
and writes no deliverable (`bench_out/frozen_gpu_transfer_refusal_03_v4_20260928.{log,json}`). A
payload file holding a JSON list exits 1 with `transcode_settings file … holds a list, not the
JSON object the queue carries`, no deliverable, and — visible in its report — no fixture at all:
the refusal precedes the FFmpeg call (`bench_out/frozen_gpu_transfer_badpayload_refusal_20260928.{log,json}`).

**Gates run for this stage.** `tests/unit/test_gui_gpu_transfer_selftest.py` — **37 passed** in
9.34 s (the 25 existing tests unchanged in what they assert, plus 12 new: verbatim pass-through
and file hash, the empty object asking for no extra checks, the resolved-options report, `uhq`
kept when the encoder has it, `uhq` claimed when it does not as a failure, a dropped `cq` and a
dropped `tune` each as failures, four bad-file refusals and a missing file — all five asserting no
export call and no `Live` session — and the CLI token beside its default).
`test_gpu_transfer_export_backend.py` + `test_packaged_binaries.py` — 102 passed; the sixteen
gui/export/transfer/packaged files these edits touch — **397 passed** in 38.64 s. `tests/e2e` was
run once, for the capture, and `npx playwright test --list` still collects 28 tests in 14 files.
`tests/unit` as a whole was **not** re-run, on purpose: this stage changed no native source, no
ABI, no pipeline arithmetic and no service route — the production edits are one report key in
`gpu_transfer_export.py` and one CLI argument in `gui.py`, both covered by the focused set above.
No frontend application source changed (only `tests/`), so no front-end rebuild and no `tsc` run.
No benchmark, no long fixture, no commit, and nothing in the working tree that was already there
was cleaned up.

**Verdict.** The GPU Transfer migration's **verifiable correctness goals are met**: a packaged
artifact built from the current tree takes the delivery panel's own default payload — the one that
carries `custom_factor`/`suffix`/`finish_action` and asks for `uhq` — runs it on the device chain
through the production entry point, keeps every frame on the card (six measured-zero host
boundaries), drives DLSS from NVOF motion on every frame, produces a container that says six
frames left the encoder, resolves its encoder settings against the FFmpeg it actually ships, runs
the copy-pass branch and cleans up after it, and refuses — before any device session — the inputs
it should refuse. The two gate fixes of the previous stage are now claims about the distribution
rather than about the source tree. **Performance remains `NOT_CHARACTERIZED`**: the profiler table
this run printed is a six-frame cold start, `elapsed_sec = 3.002` is a bounded probe's wall clock,
nothing was compared, and no throughput, speed or parity claim is being made from any of it.

**What this does not close.** One bundle and one payload — the panel's *default* state; an
operator who changes codec, `resolution_mode` or `audio_mode` is making a different request, and
`hevc_nvenc`/`h264_nvenc`, a resize and an audio-bearing source are still unproven inside a
bundle. `guidance_mode = 1` only, 1280x720 sources only, 4–6 frames, no 4K, no cancellation, no
multi-job queue, and no packaged route to any of it from the window: the bundle still ships no
service, so a packaged operator still cannot click GPU Transfer — what is proven is that the
packaged *chain* accepts the panel's payload, not that the packaged *UI* can send it. The frozen
run's `job_settings` are the probe's precondition set, which differs from the panel's in one field
(`flow_quality 4M` against the UI job's `1F`); `transcode_settings` is the part this stage was
asked to match, and it matches exactly. `0.25` remains extent-dependent (served at 4K, refused at
720p), and the bundle's scale/branch coverage is this 0.5 `exact_two_x` run plus the earlier
`plan_driven` 0.63 and `identity` 1.0 runs, which came from a *superseded* bundle.

**The next smallest verifiable gap** is that last caveat: `pkgdist3`'s two branch runs predate the
fixes this stage rebuilt, so the 0.63 `plan_driven` and 1.0 `identity` branches have not been
re-run against a bundle that carries them. One frozen run at each ratio from `pkgdist4`, with the
same captured payload, would make the branch coverage as current as the payload coverage.

## The four static-review findings: the queue refuses to overwrite its own source, says "skipped" when it skipped, retries only on request, and stops describing GPU Transfer as a command line

A static review of the delivery path found four claims the code was making that it could not
keep. None of them are algorithmic: no DLSS, NVOF, NVENC or resampling behaviour changed, and
performance remains `NOT_CHARACTERIZED`.

**1. A job could name its own source as its destination.** `pipeline.build_transcode_command`
sends `-y` whenever `skip_existing` is false and `overwrite_output` is true, so an export asked
for `in.mkv → in.mkv` truncated the material while still decoding it. The answer is
`pipeline.assert_output_not_input`, placed at every entry that receives a destination —
`build_transcode_command`, `export_dlss_stream` (before the probe), `export_gpu_transfer_stream`
and `gpu_transfer_export.export_gpu_transfer` / `build_mux_command`, plus
`DLSSService._execute_job` before `video_info` opens a decoder — so the refusal lands before any
process is spawned. Identity is resolved with `os.path.realpath` (case, 8.3 short names, trailing
dots, `.`/`..` and `\?\` prefixes fold), then compared again on `(st_dev, st_ino)` so a hard link
or a symlink to the same file is caught; two genuinely different files are never refused, a
different container suffix included. `OutputOverwritesInputError` carries a Chinese reason that
names the file and the fix, and it is answered at both surfaces: `add_job` rejects the payload, so
`POST /api/queue/add` answers HTTP 400 with that reason and no row enters the queue, and a job that
already had the row is refused at the job boundary as `status = "failed"` with `error` set,
broadcast on `/ws/queue` like any other failure. The panel pre-checks the same composition
(`sameOutputAsInput` in `frontend/src/services/exportPreview.ts`) and refuses the add with a
visible error, but the backend check is the one that is load-bearing: a browser cannot see a link.

**2. A skipped job used to report itself as an export that succeeded.** `_execute_job` returned
early when the destination existed under `skip_existing`/`overwrite_output=false`, and the queue
then wrote `completed`, counted it for `finish_action` and left the progress bar alone. It now
returns `JOB_SKIPPED`, and `_run_queue` records `status = "skipped"` with
`skipped_reason = "output_exists"`, counters untouched. Because `_handle_finish_action` reads
`finish_action` only off `completed_this_run`, a skipped-only run cannot sleep or shut the machine
down — the list it is handed is empty, which the new tests capture rather than infer. Cancellation
and failure keep their previous states; the file-preservation behaviour that spec 9.1 pins is
unchanged. `queue_running` was added to every queue message (`_queue_state()` in `server.py`, the
`/ws/queue` `init` frame, `GET /api/queue`) because a row's own status cannot express the window
where the worker is still alive and nothing is running — which is exactly when a retry must be
refused. DeliverPage renders 已跳过 · 未写入 in amber, counts it separately in the queue header,
and does not clear the "start" button on it.

**3. Pressing 全部渲染 used to re-run yesterday's failures.** The queue loop now skips every
status but `pending`. Retry is an explicit action: `POST /api/queue/retry` calls
`DLSSService.retry_job`, which decides under `_queue_lock`, refuses the active job, refuses while
the worker thread is alive, is idempotent for a job already pending, and resets only
failed/cancelled/skipped rows — progress, frame counters, error, host record and residual
diagnostics included, so a reset row describes no attempt until the next one fills them in. The
row's 重试 button is disabled while the queue runs or the job is running, and a new pending job
added next to a failed one starts only the pending one.

**4. The GPU Transfer preview showed a command that does not exist.** The panel's preview box
built one FFmpeg command line and called it the export, which was true only of the legacy pipe.
The builder moved whole into `frontend/src/services/exportPreview.ts` — a client-side *mirror* of
`pipeline.build_transcode_command`, which e2e 7.1 still compares token by token — and gained a
second form: `export_backend=gpu_transfer` renders the device stages (NVDEC → NVOF or the resident
zero-motion texture → DLSS → GPU Lanczos + residual → NVENC), the requested codec/preset/tune/rate/
GOP/pixel-format values, the single-stream `.mkv` plus `-c:v copy` remux rule, the no-automatic-
fallback admission line and the `NOT_CHARACTERIZED` performance line, with no command string and
the copy button disabled. The tune line does not promise `uhq`: for `h264_nvenc` it says the
downgrade to `hq` is certain, otherwise that it depends on what the backend probes the *shipped*
encoder to support, and that the diagnostic log for the job is the authoritative answer. The
preview recomputes from `dlssSettings` in the same render, so switching chains is immediate.

**Tests.** New: `tests/unit/test_output_overwrite_guard.py` (18 — the refusal precedes
`_bundle_ffmpeg`, `subprocess.Popen`, `video_info`, `load_bridges` and a mux, case/8.3/`\?\`/
dot-suffix aliases are caught, distinct files and different containers are not, hard links are),
`tests/unit/test_queue_skip_and_retry_contract.py` (20 — skipped status/counters, the finish-action
list captured as `[([], False)]`, pending-only round, retry under lock and its refusals),
`tests/unit/test_render_queue_ui_contract.py` (15 — the panel's status, count and retry wiring read
from source) and `tests/node/export_preview.test.ts` (15, `npm --prefix frontend run test:preview`)
for the two preview forms and `sameOutputAsInput`. Adjusted for the new contract:
`tests/unit/test_cancellation_e2e.py` hand-built its job as `status = "queued"`, a state the queue
rightly ignores, and now builds it as `pending`; `tests/helpers/backend_client.js` counts `skipped`
as terminal in `waitForJob`, otherwise spec 9.1 would wait out its timeout on a job that had
already been truthfully answered. New e2e spec `tests/e2e/16_preview_and_selfoverwrite_ui.spec.ts`
drives both halves in the real window without rendering a frame.

**Gates run.** `tests/unit` — **2 failed, 840 passed** in 121.75 s; both failures are
`tests/unit/test_nvof_independence.py`, whose `_negotiation()` stub hands `plan["struct"] = None`
while the report path at `gpu_transfer_export.py:1717` reads `plan["struct"].get(name)` — a defect
in that untracked test's stub, untouched by this stage. The three new Python files — 53 passed in
5.42 s. The six adjacent contract suites (export-backend UI, zero-motion pipe, cancellation,
gpu_transfer backend, preview parity, export pipeline) — 161 passed in 20.17 s.
`tests/unit/test_cancellation_e2e.py` — 1 passed in 9.75 s. Frontend: `npm run build` exit 0,
`npm run lint` at the 4-warning baseline (all four are pre-existing `set-state-in-effect` notes on
lines this stage did not write), `npm run test:preview` 15 passed. Playwright: spec 7.1 —
1 passed in 25.8 s, the legacy preview still token-identical to `build_transcode_command`; spec 16 —
2 passed in 38.5 s. `npx playwright test --list` now collects 30 tests in 15 files, the two added
ones being the only change under it. The GPU Transfer six-frame gate was not re-run: nothing in the chain it
exercises changed, only the panel's description of it.

**What this does not close.** The guard compares the destination against the source as the job
names them; a destination that becomes the source only after the run starts (a directory swap, an
atomic rename by another process) is still a race no check here can see. `retry_job` resets one
row and cannot start the queue: an operator still presses 全部渲染 afterwards, which is intended.
The preview's encoder parameters are the *requested* values, not what the encoder accepted — the
job's own diagnostic record remains the authority. And the skipped state is only as honest as the
`os.path.exists` it is built on: a destination that exists but is unwritable still skips.

## Experimental independent multi-scale DLSSNR observations

The sealed 25–100% single-layer scale contract remains the default. The DLSS
settings UI now groups experimental reconstruction controls by layer, following
Magpie's parameter-group layout without adopting its repeated-pass algorithm.
Selecting Dual/Triple switches the export backend to GPU Transfer and exposes
distinct layer scales, observation weights, and a baseline/fusion output choice;
the CG solver controls appear only for experimental fusion. Preview and image
rendering remain single-layer, and the delivery page rejects inconsistent layer
vectors before queueing. A `gpu_transfer` request can opt into
`reconstruction_mode=dual|triple`, with explicit,
distinct `layer_scales` and positive `layer_weights`. The first scale equals
`working_scale`; all extents come from `pipeline.resolve_working_dims`. Every layer
downsamples the same full source with the existing Lanczos-2 axis plans, owns a
separate Feature 18 handle, parameter map, temporal state and reset flag, and
produces a signed working-resolution residual. Extra guided layers own separate
NVOF services. NGX calls share the initialized device/core and are serialized;
closing one Feature does not shut down its siblings. A new export recreates all
sessions after a layer-scale change or resize; an in-job source extent change is
rejected before processing with old textures. The first frame and any backward,
duplicate or missing-frame PTS reset all active layers together. Cancel closes
every layer.

`fusion_mode=catmull_baseline` collects the independent observations while
encoding the original layer-zero Catmull result. Opt-in
`fusion_mode=data_consistency_experimental` solves on the GPU
`(Σ w_i D_iᵀD_i + βI)r = Σ w_i D_iᵀr_i + βr₀`, where `β>0`, `r₀` is the
existing Catmull reconstruction and each `D_i` uses the production Lanczos-2
tap indices, float32 weights, sampling centers and edge clamps. The transpose
is built from every tap, including duplicate clamped edge taps. The linear
operator acts on signed float residuals; the production RGBA8 downsample's
final quantization is outside that linear operator and has no linear adjoint.
CG has explicit iteration and relative-residual limits. Nonconvergence fails
the export with a reason. Final composition remains full source plus
`residual_strength * r`, clamped and rounded to RGBA8.

Short validation on the current RTX 5060 Ti: two independent sessions at
50/75% processed three frames; closing and reopening one left the other usable.
A five-frame dual fusion export produced five full-resolution packets with
monotonic PTS and no host pixel readback; a three-frame 40/67/85% triple fusion
export did the same with three distinct sessions. A two-frame 50/100% dual
export exercised the full-resolution identity layer. One 14-frame 4K single
export had the same demuxed AV1 bitstream SHA-256 as its sealed baseline
(`7adf30de18b05a710a31a7a80ea218bc3c45f65a1aee2d296dbc6b9af0e5b471`).
The synthetic CPU-reference data-consistency sum fell from 1207.730804 to
15.674310; this measures agreement with the observations, not image quality.
GPU dot products read back scalar partial sums, recorded separately as
`fusion_scalar_readback_bytes_measured`; host pixel transfer counters remain zero.
Two unguided fusion frames also completed without loading NVOF.
4K multi-layer memory use, visual quality and performance are not characterized.
