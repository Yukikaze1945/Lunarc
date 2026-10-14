# HDR Stage B — binding audit fix (recovering `qbg_20261001_32eda8a1`)

Workspace `F:\dlss5standaloneV2`. The crashed worker left **no** partial edits: the
three defects below were still in the tree exactly as the audit found them, so this
task is the implementation, not a re-audit. Scope was the binding only — no matrix,
PQ, proxy, guidance, resampling or memory-layout change. Code and CPU only.

## 1. BT.2020 colour-space enumerator (the real admission defect)

**Local evidence, primary source first.**

- `F:\WhisperJAV\Library\include\libavutil\pixfmt.h:620-621` (the installed FFmpeg
  this build links against):
  `AVCOL_SPC_BT2020_NCL = 9,  ///< ITU-R BT2020 non-constant luminance system`
  `AVCOL_SPC_BT2020_CL = 10, ///< ITU-R BT2020 constant luminance system`
- `native/gpu_hdr_scrgb_bridge.cpp:2278-2279` — decode accepts a source only when
  `color_space != kSpaceBt2020Ncl` is false, i.e. this constant *is* the gate.
- `native/gpu_hdr_scrgb_bridge.cpp:2016-2018` + `:2021` — the four `constexpr` mirrors,
  of which `kRangeLimited = 1`, `kPrimariesBt2020 = 9`, `kTransferPq = 16` were already
  right and only the space was wrong.

Both sides of the binding had agreed on **10**, so every fake-library test passed
while the pair was wrong together: a real HDR10 source, which decoders report as
`AVCOL_SPC_BT2020_NCL = 9`, would be refused by `gpu_hdr_decode` as a
constant-luminance space. The internal agreement was exactly what hid it.

| File | Before | After |
| --- | --- | --- |
| `native/gpu_hdr_scrgb_bridge.cpp:2019` → `:2021` | `constexpr int32_t kSpaceBt2020Ncl = 10;` | `constexpr int32_t kSpaceBt2020Ncl = 9;` plus a two-line comment naming 10 as `AVCOL_SPC_BT2020_CL` |
| `hdr_gpu_engine.py:59` → `:60` | `AV_COLOR_SPACE_BT2020_NC = 10` | `AV_COLOR_SPACE_BT2020_NC = 9` plus the same `pixfmt.h` citation |
| `tests/unit/test_hdr_gpu_engine.py:278,366,468` | literal `color_space=10` | `color_space=hdr.AV_COLOR_SPACE_BT2020_NC` (the wrong literal is gone, not re-pointed) |
| `tests/unit/test_hdr_gpu_engine.py:470-472` | `assert (args[5], args[6], args[7], args[8]) == (1, 9, 16, 10)` | asserted against the four named wrapper constants, so the pass-through test tests pass-through and no longer re-states a number |

Not touched, deliberately: the reconstruction matrix, PQ, proxy, guidance and
resampling code and constants; `DXGI_R16G16B16A16_FLOAT = 10`, which is a different
enum that also happens to be 10 and is now pinned by its own test. `native/gpu_hdr_scrgb_bridge.h`
documents the space only in prose ("non-constant-luminance") and carries no numeric
value, so it needed no edit and is now consistent.

**New regression, grounded rather than mirrored:**
`test_color_enums_match_the_bridge_constants_they_are_checked_against` (test:503)
parses the four `constexpr int32_t` values out of `native/gpu_hdr_scrgb_bridge.cpp`
and asserts the wrapper constants equal what the compiled bridge compares against;
`test_bt2020_ncl_is_the_libavutil_value_not_the_constant_luminance_one` (test:516)
parses `AVCOL_SPC_BT2020_NCL` and `AVCOL_SPC_BT2020_CL` out of the installed
`pixfmt.h` (overridable with `FFMPEG_INCLUDE_DIR`), asserts
they differ, and asserts the wrapper and the native constant both equal NCL — so a
regression back to 10 fails against the header, not against another copy of itself.
`test_dxgi_fp16_format_keeps_its_own_ten` (test:535) keeps the unrelated DXGI 10 pinned.
Historical probe artifacts were neither run nor modified.

## 2. Encode chroma descriptor

`hdr_gpu_engine.py:853-861` (`GpuHdrBridge.encode`): the chroma plane was declared
`bytes_per_pixel=2` at `width = full_width // 2`. P010 is semi-planar — one
interleaved Cb,Cr **pair** of 10-bit codes per luma 2x2 cell — so half-width entries
carry two components and each entry is 4 bytes.

- `uv.bytes_per_pixel`: `2` → `4`; `y` stays `2`.
- `uv.pixel_format`: `"P010LE chroma (4:2:0)"` → `"P010LE chroma pairs (Cb/Cr interleaved, 4:2:0)"`.
- `uv.height` stays `full_height // 2`.
- Stated invariant, now asserted: `uv.width * uv.bytes_per_pixel == full_width * 2 ==
  y.width * y.bytes_per_pixel` — the chroma row is the same byte width as the luma row.

No native change: the bridge sizes and copies its own planes by pitch and never
reads this field. Under-sizing the descriptor by half was a consumer-side buffer
error (an encoder handed 2 bytes/entry would read half a row and alias Cb as Cr).
New test: `test_encode_chroma_bpp_is_the_interleaved_pair_not_one_component` (test:445).

## 3. Owner lifetime and open-failure injection

`hdr_gpu_engine.py:657-678`:

- `close()` now refuses the wrong thread **before** the native call. The bridge's
  CUDA context is current on the thread that opened it, so a foreign `gpu_hdr_close`
  would destroy device allocations through a context that does not own them. The
  handle survives a refused close.
- An already-closed `close()` returns immediately — idempotent, including from
  another thread, because there is nothing left to protect.
- `_release_handle()` (657) is now the single `gpu_hdr_close` call site, reused by
  the failed-open path (618) and both constructor refusals below.

`hdr_gpu_engine.py:621-643` — the constructor now refuses a bad open instead of
adopting it, and cleans up whatever it acquired:

- `rc == 0` with a null out-slot → refused, `gpu_hdr_paper_white` never probed, and
  nothing closed because nothing was acquired.
- effective paper white that is non-numeric, non-finite, or different from the
  requested value → the acquired handle is released, the open is refused, and the
  requested value is left exactly as it was sent to `gpu_hdr_open` (`_configured_paper_white`
  is never rewritten to match the bridge). `_effective_paper_white` is published only
  after the check.

New tests: `test_successful_open_with_a_null_handle_is_refused_not_adopted` (568),
`test_a_changed_or_non_finite_effective_paper_white_refuses_the_open` (579, parametrized
`nan`/`inf`/`2.5`, also asserting `close_count == 1` and `open_args[8] == 3.0`),
`test_a_non_numeric_paper_white_report_refuses_the_open` (593), and the lifecycle test
`test_close_from_another_thread_is_refused_before_the_native_call` (676).
`FakeLibrary` gained one injection knob, `handle=` (0 simulates a success-with-null
out-slot). No generalized pointer hardening, no refactor.

## Verification actually run (code + build + CPU only)

| Command | Result | Exit |
| --- | --- | --- |
| `python -m py_compile hdr_gpu_engine.py tests/unit/test_hdr_gpu_engine.py` | clean | 0 |
| `python -m pytest tests/unit/test_hdr_gpu_engine.py -q` | **66 passed** in 1.60s | 0 |
| `python -m pytest tests/unit/test_hdr_gpu_engine.py -q -rs -k "ncl or enums or fp16 or paper_white_refuses or null_handle or another_thread or chroma"` | **12 passed, 54 deselected**, zero skips — the `pixfmt.h`-grounded test really read the installed SDK header | 0 |
| `cmd //c "build_gpu_hdr_scrgb_bridge.bat"` (existing script, one run, VS 2022 x64 v17.14.40, `cl /LD /O2 /EHsc /std:c++17 /W4 /WX`) | compiled and linked, no warning under `/WX` | 0 |

DLL, `native/gpu_hdr_scrgb_bridge.dll` (253 440 bytes both sides):

- before: `41a38ce99b022ff43676d5c4169babe477ca42d3894e0ec47ef0ea28dab2fbd5` (Sep 30 19:22)
- after: `e016fce6e1b3cbc01b4acd751db71144c1546dcd57e362cdb0486a6a3a269e47` (Oct 1 19:26)

Side artifacts regenerated by that build, all git-ignored: `gpu_hdr_scrgb_bridge.obj`,
`.exp`, `.lib`. No build error needed fixing, so no extra edit was made.

Baseline preserved: no commit, push, reset or clean. Everything this task touched
(`hdr_gpu_engine.py`, its test, `native/gpu_hdr_scrgb_bridge.cpp`, the rebuilt DLL)
was already untracked, and the previously-dirty tracked native files
(`dlss_gpu_host.*`, `nvdec_bridge.*`, `nvenc_bridge.*`, `nvof_d3d12_interop.*`) were
left untouched.

## Not run, on purpose

No real export, no GPU job, no pixel readback. The native numerical gate
(`gpu_hdr_scrgb_bridge_test.exe` / `build_gpu_hdr_scrgb_bridge_test.bat`), Feature 18
gates, HDR gate scripts and every benchmark were **not** run, and no historical probe
artifact was read-modified or regenerated.

**Corroboration from elsewhere, explicitly not this task's evidence.** A companion
background job running the same repair text (`hdr stage b binding audit fix`) finished
afterwards reporting that it had rebuilt and run the CPU-only native gate: 416
assertions passed / 0 failed, including decode accepting the non-constant-luminance
space and refusing a primaries value of 10 — against **its** DLL
`1036673d58203137d9b5d3a0d860a2be8b30d4558790ca09fbc875afdc7420d8`. The binary on disk
in this workspace is `e016fce6e1b3cbc01b4acd751db71144c1546dcd57e362cdb0486a6a3a269e47`,
a different build, so the two must not be conflated and `1036673d…` is not this
workspace's artifact. Whoever runs the admission frame should re-hash the DLL they
actually load and re-run whatever gate they cite.

**True single-frame real admission through this rebuilt DLL: NOT YET VERIFIED.** That
is the next short task, and it is the first moment the corrected NCL enumerator meets
a decoder's actual `color_space` instead of a fake. This job is not HDR acceptance.

## Remaining

1. One-frame admission: decode → downscale → proxy → Feature 18 → reconstruct →
   encode on a real frame, with the bridge's own reports, proving the NCL gate opens
   and the P010 pair descriptor matches the encoder's buffer maths.
2. Product wiring: nothing calls `GpuHdrBridge` yet — no decoder, NVOF, Feature 18 or
   NVENC orchestration in this layer.
3. UI: no surface, control or status reads this binding.
4. Final HDR10 export acceptance: SEI/MaxCLL, container flags, the encode ceiling
   policy and end-to-end measurement on a display.
5. Descriptor consumers: `p010_cuda` encode planes still have no consumer in-tree to
   check against the new 4-byte chroma entry claim.

## Statements preserved unchanged by this repair

- `HDR_STAGE_B_PYTHON_BINDING.md:76-77`: "This substage does **not** claim HDR is
  complete or lossless. Performance through this layer is **NOT_CHARACTERIZED**."
  Still true; nothing here characterizes performance.
- The Stage A numeric loss record stands as measured and is neither restated nor
  superseded: at `paperWhite 4.0`, `color_sub_and_super` ch0 the true quantized slope
  move is −0.12875 stops (−8.5%) where the stale math read −0.45982 stops (−27.3%),
  and the stale headline figure was ~7.9% too generous
  (`HDR_PROXY_AB_STAGE_A.md:49,428,463-465`). The FP16 and guidance semantics, the
  reconstruction matrix, PQ constants and tap plans were not modified, so no prior
  number can shift.

## Files modified

- `hdr_gpu_engine.py` — `AV_COLOR_SPACE_BT2020_NC = 9` (:60); encode chroma descriptor
  `bytes_per_pixel=4` and pair-named `pixel_format` (:853-861); `_release_handle()`
  (:657), thread-refusing `close()` (:665), open-admission checks for null handle and
  effective paper white (:615-643).
- `native/gpu_hdr_scrgb_bridge.cpp` — `kSpaceBt2020Ncl = 9` with its citation
  (comment :2019-2020, constant :2021; the decode comparison that uses it is :2278-2279).
- `tests/unit/test_hdr_gpu_engine.py` — 8 new tests, 4 corrected expectations,
  `FakeLibrary(handle=…)` injection knob; imports `os`, `pathlib`, `re`.
- `native/gpu_hdr_scrgb_bridge.dll` — rebuilt (hash above).
- `HDR_STAGE_B_BINDING_AUDIT_FIX.md` — this record.
