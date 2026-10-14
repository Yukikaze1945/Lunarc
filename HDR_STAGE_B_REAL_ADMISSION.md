# HDR Stage B — one real P010 frame admitted by the native bridge

Workspace `F:\dlss5standaloneV2`, following the accepted binding repair
`qbg_20261001_e3bea564` (`HDR_STAGE_B_BINDING_AUDIT_FIX.md`). That repair changed
`kSpaceBt2020Ncl` from 10 to 9 in `native/gpu_hdr_scrgb_bridge.cpp` and
`AV_COLOR_SPACE_BT2020_NC` from 10 to 9 in `hdr_gpu_engine.py`, and proved the pair
only against CPU tests and a rebuilt DLL. Its own last line said the missing piece:
**"True single-frame real admission through this rebuilt DLL: NOT YET VERIFIED."**

This task is that one piece. One borrowed real P010 frame, the decoder's own colour
fields, the current production wrapper and the current production DLL. No pixel ever
came to the host. No Feature 18 call, no NVOF, no NVENC, no export, no UI, no
orchestrator. The 66 CPU tests and the build were not re-run.

## Result

| Question | Answer |
| --- | --- |
| Does the rebuilt bridge admit a real decoder's `color_space`? | **YES.** `gpu_hdr_decode` returned success on the decoder's own `range 1 primaries 9 transfer 16 space 9`. |
| Does it still refuse BT.2020 constant-luminance? | **YES.** The same live frame, only `color_space` changed to 10, was refused: rc −1, message naming the expected NCL/ST 2084 domain and reporting `space 10`. |
| Did any frame pixel reach the host? | **NO.** Decoder `host_frame_copy_count == 0`, `host_frame_copy_bytes == 0`; bridge `report_readback_bytes == 0`; no readback entry is bound in the process. |
| Verdict | `PASS` — 42 checks, 0 failed. Gate ran as `native/hdr_real_admission_gate.py`, exit 0. |

The 66 CPU tests and the rebuild still stand as recorded; this adds the first real
decoder meeting the repaired enumerator.

## Commands actually run

| Command | Exit | Note |
| --- | --- | --- |
| `python native/hdr_real_admission_gate.py` | **0** | the recorded run; console in `bench_out/hdr_real_admission/gate_stdout_attempt3.txt` |
| `python native/hdr_real_admission_gate.py` | 1 | attempt 2 — admission passed, the *declared domain* did not (see below) |
| `python native/hdr_real_admission_gate.py` | 1 | attempt 1 — a bug in this gate's own binding, not in production code |
| `python -m py_compile native/hdr_real_admission_gate.py` | 0 | after each edit |
| `ffprobe … -show_entries stream=… -of json build/hdr_probe/hdr_src.mkv` | 0 | the only external command the gate itself ran; recorded in `evidence.json → commands` |

Both failed artifacts are preserved, not overwritten:
`bench_out/hdr_real_admission/evidence_attempt1_failed.json` and
`evidence_attempt2_ceiling250_failed.json` (with their console logs). The gate
records them in `evidence.json → declared_domain.history`, so the passing file names
its own predecessors.

## What the decoder actually returned

`build/hdr_probe/hdr_src.mkv` — hevc Main 10, `yuv420p10le`, tv range, bt2020nc,
smpte2084, bt2020 primaries (container claim, from ffprobe). Through
`nvdec_bridge_open_hdr` (ABI 3), exactly one `nvdec_bridge_next` frame:

| Field | Value as the decoder reported it |
| --- | --- |
| width × height | 1920 × 1080 |
| software_format | 2 = P010 (`AV_PIX_FMT_P010LE`) |
| y / uv device pointers | `0xd0b600000` / `0xd0ba38000` |
| y / uv pitch | 4096 / 4096 bytes (rows need 3840; NVDEC's surface is aligned) |
| cuda_context | `0x22022107740` |
| pts | 0 |
| **color_range / primaries / transfer / space** | **1 / 9 / 16 / 9** |

Those four values were sent to `decode` as the decoder gave them. The gate asserts
them against the HDR10 contract; it never substitutes `hdr_gpu_engine`'s constant
for the decoder's field — the wrapper constant happens to read 9 now, and that
coincidence is recorded as `space_constant_sent`, not used as the input.

Source HDR metadata as the bridge reports it: `has_mastering 1` (primaries and
luminance both present, white 15635/50000, 16450/50000, min 1000/10000, max
10000000/10000 = 1000 cd/m²), `has_content_light 1`, **MaxCLL 1000, MaxFALL 400**.

## Session, bridge and descriptor

`pipeline.resolve_working_dims(1920, 1080, 0.5)` → **960 × 544** (the production
rule, not a chosen number). One `dlss_gpu_engine.Live(960, 544, color_format="rgba16f")`
on `dlss_gpu_host.dll` ABI 3, opened only to borrow its `d3d12_device()`
(`0x2202f7629c0`); **no frame was ever given to it**. The decoder's CUcontext was
pushed current on the wrapper thread, then
`hdr_gpu_engine.GpuHdrBridge(device, context, full 1920×1080, working 960×544,
paper_white_scrgb=8.0)` with the CPU `pipeline` plans — opened, configured paper
white 8.0, bridge-reported effective paper white 8.0, ABI 1.

`decode` returned:

```
format rgba16f_cuda · pixel_format R16G16B16A16_FLOAT (scRGB) · 1920 × 1080
pitch_bytes 15360 (= 1920 × 8) · bytes_per_pixel 8 · device_ptr 0x205200000
cuda_context 2337033713472 (= 0x22022107740) → equal to the decoder's own context
```

Source report for that one decode: `pixels 2073600` (the full frame),
`nonfinite_samples 0`, `above_ceiling_components 0`, `negative_components 3098461`,
`min_component −614.1239013671875`, `max_component 9420.1640625`. Negative channels
are legal signed wide-gamut values and are reported, not treated as a failure.

## The declared domain: why 250 scRGB failed and 10000 was used

Attempt 2 declared 250.0 scRGB and measured `max_component 9420.164`, so
`above_ceiling_components` came back **629189** — the gate's own ceiling check
failed. That is a fixture property, not a native mismatch, and saying so needs the
bridge's own scalars:

- this bridge's units: 1.0 scRGB = 80 cd/m², so the ST 2084 top (10 000 cd/m²) is
  125.0 scRGB, and the source's MaxCLL 1000 cd/m² would be 12.5 scRGB;
- PQ's inverse is asymptotic. With the contract's `pq_c2/pq_c3` the denominator
  vanishes at coded input **1.99206**;
- the measured 9420.164 scRGB corresponds to a coded PQ input of **1.39762** —
  **70.2 % of the way to the pole**, i.e. a channel a few code units past its
  limited-range top, which a synthetic pattern with extreme chroma excursions
  produces. Past that point a small code error is a huge scRGB number, which is
  exactly why the ceiling must be a declared domain rather than a metadata copy.

So the domain is **10000.0 scRGB**: finite, per-channel, above the fixture's
measured max with the margin recorded, and explicitly not `MaxCLL`-derived (12.5
scRGB would have made the check circular and would have refused this source).
Attempt 3 passed with `above_ceiling_components 0` against that declaration.
Two checks now pin it: the domain must exceed the measured max, and must not equal
the MaxCLL equivalent.

## Zero host pixel traffic, from the counters and the code path

| Boundary | Value | Meaning |
| --- | --- | --- |
| decoder `frames_decoded` / `device_frame_count` | 1 / 1 | one frame, borrowed; the negative call did not advance the decoder |
| decoder `host_frame_copy_count` / `host_frame_copy_bytes` | 0 / 0 | the exposed host-image counter is zero, as the ABI reports it — no invented "measured 0" field |
| decoder `bitstream_bytes_submitted` | 95999 | packet traffic, which the decoder uploads itself |
| decoder `scale_device_output_bytes` | 0 | `open_hdr` builds no scaler |
| bridge `source_device_copy_bytes` | 6220800 | the P010 payload moved device-to-device |
| bridge `source_device_write_bytes` | 16588800 | 1920×1080×8 — the FP16 master plane written on device |
| bridge `working/guidance/composited/encode_*_bytes` | 0 / 0 / 0 / 0 | later stages never ran; this is admission only |
| bridge `report_readback_bytes` | 0 | scalar reports are counted separately, and none of them is an image |

The stronger claim is structural: `hdr_gpu_engine._BRIDGE` binds no
`gpu_hdr_readback_*` entry (`gpu_hdr_readback_source_p010`,
`gpu_hdr_readback_encoded_p010`, `gpu_hdr_readback_scrgb` are absent from the bound
list), and `nvdec_bridge_readback_last_frame` was never resolved in the process — the
gate asserts both. No `gpu_hdr_plane_bytes`/`pitch` either. There is no code path in
this run that could copy a pixel to the host.

## Negative admission, verbatim

Sent on the same live borrowed frame, with only `color_space` changed:

```
{'color_range': 1, 'color_primaries': 9, 'color_transfer': 16, 'color_space': 10}
```

Native answer, captured before any cleanup could overwrite the bridge's last error:

```
HdrBridgeError: gpu_hdr_decode failed (-1): HDR decode wants limited-range
BT.2020 NCL with ST 2084, got range 1 primaries 9 transfer 16 space 10
```

It was not converted into a fallback and not recorded as a success: the bridge is
still open after it (`is_alive() == True`), the descriptor of the earlier positive
decode is untouched, and `BT2020_CL` was read out of the installed
`F:\WhisperJAV\Library\include\libavutil\pixfmt.h` (`AVCOL_SPC_BT2020_NCL = 9`,
`AVCOL_SPC_BT2020_CL = 10`) rather than typed in, so the refusal is checked against
the same header the bridge was compiled against.

## Loaded binaries, hashed after the FFmpeg load

| File | SHA-256 |
| --- | --- |
| `native/gpu_hdr_scrgb_bridge.dll` (253 440 B, Oct 1 19:26) | `e016fce6e1b3cbc01b4acd751db71144c1546dcd57e362cdb0486a6a3a269e47` — **matches** the accepted build, asserted by a check |
| `native/nvdec_bridge.dll` (155 648 B) | `263c4397876e729fd964e1457c2bc5365de3032c77ba481961af69b85f801ea0` |
| `native/dlss_gpu_host.dll` (285 696 B) | `5640d4fe0fcb03f5a3e7b50ffe287acb18bd7b1c5cfd6f4342af318e1b799cfd` |
| `F:\WhisperJAV\Library\bin\avutil-59.dll` | `1e6ff943de8f95ec…` (full value in `evidence.json`) |
| `F:\WhisperJAV\Library\bin\libcrypto-3-x64.dll` | `e5a134bc7562c8f279b5f88612c7d9140ee52ee2f6368d92c383ec587366d972` |

The libcrypto line is the reason for the loader order: `F:\WhisperJAV\Library\bin`
is registered and `nvdec_bridge.dll` is loaded before anything imports `hashlib`, so
the FFmpeg chain and `_hashlib` resolve the *same* copy instead of colliding. The
gate asks `GetModuleHandleW` for the already-loaded module rather than loading a
second handle, which is the only way to see which copy won.

Wrapper provenance: `hdr_gpu_engine.py`
`7f8c5734dbfb3978357606e5d7bac87adc32f81174772e5bba1903d992bc7ff5` (ABI 1,
`AV_COLOR_SPACE_BT2020_NC = 9`), `native/nvdec_bridge.h` and `pipeline.py` /
`dlss_gpu_engine.py` hashed in `evidence.json → wrapper_provenance`, together with
this gate's own source hash `c1118307d88a5b466e742f8b46e78a35cc0ff91fb2c7292af935d6738a581d4a`.

## Cleanup

Closed in order, asserted in the evidence and executed the same way on a failure
path: `gpu_hdr_bridge` (`is_alive() == False` after) → `dlss_gpu_engine.Live`
(`is_alive() == False`) → decoder CUcontext popped → `nvdec_bridge_close`. All on
the one thread that opened them, which is the wrapper's documented rule.
`cleanup_order == ["gpu_hdr_bridge", "dlss_gpu_engine.Live", "nvdec_bridge"]`.

## Not claimed by this run

- **Full pipeline: NOT DONE.** `downscale`, `proxy`, `reconstruct` and `encode` were
  never called; their bridge counters read zero because those stages did not run.
- **Production orchestrator and UI: NOT DONE.** Nothing in `dlss_service.py`,
  `gpu_transfer_export.py` or `gui.py` calls `GpuHdrBridge`.
- **Final 4–6 frame real-source gate: NOT DONE.** This was one frame by design.
- **Performance: NOT_CHARACTERIZED.** One decode of one frame is not a measurement.
- **Formal Feature 18 HDR acceptance: unproven here.** No Feature 18 call was made.
- **FP16 loss: unchanged.** The Stage A numeric record
  (`HDR_PROXY_AB_STAGE_A.md:49,428,463-465`) is neither restated nor superseded;
  no math suite was re-run.
- No second `hdr_gpu_engine` truth was invented: the wrapper and the bridge are the
  production ones, unmodified by this task.

## Files

- `native/hdr_real_admission_gate.py` — new, the gate. Narrow local mirrors of
  `NvdecDeviceFrame`, `NvdecTransferCounters`, `NvdecHdrMetadata`; it imports the
  production `hdr_gpu_engine` / `dlss_gpu_engine` / `pipeline` and binds no readback.
- `bench_out/hdr_real_admission/evidence.json` — the passing run, 42 checks.
- `bench_out/hdr_real_admission/evidence_attempt1_failed.json`,
  `evidence_attempt2_ceiling250_failed.json`, `gate_stdout*.txt` — preserved failures.
- `HDR_STAGE_B_REAL_ADMISSION.md` — this record.

The gate carries 5 cosmetic `ruff` findings (import order, implicit string
concatenation in the ffprobe argv, an explicit `check=`, two deliberate
`except BaseException` in the cleanup path that report rather than swallow). They
were left unfixed on purpose: `evidence.json → wrapper_provenance.gate.sha256`
records the exact source that produced this run, and editing it would either
invalidate that hash or require a fourth GPU run, which "run the gate once" does
not sanction. The pre-existing `native/nvenc_hdr_p010_smoke.py` has 12 such
findings, so this is the local convention for gate scripts, not an oversight.

No commit, push, reset, stash or clean. No pre-existing file was edited: the only
additions are the gate and its artifacts listed above, production `hdr_gpu_engine.py`,
`native/gpu_hdr_scrgb_bridge.{cpp,dll,h}` and `native/nvdec_bridge.*` are untouched,
and no math or Feature 18 suite was re-run. The previously-dirty tree and all earlier
artifacts are as they were.
