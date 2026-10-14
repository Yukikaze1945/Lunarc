# HDR Stage B — FP16 Feature 18 device-colour host connect (recovery of `qbg_20260930_c4a53cf9`)

Date: 2026-09-30. Owner: Qoder (implementation), Codex (audit).
Evidence: `bench_out/hdr_fp16_host_connect/` — `chain.json`, `recovery_summary.json`,
`chain_run1.log`, plus the three SDR baselines the crashed job already left there.
Original task: `build/qoder_hdr_fp16_host_connect.txt`.

## Verdict

The interrupted job died at 59m21s with no result (`qoder-bg`: worker pid 20736 and Qoder pid
7564 gone). Its saved code was kept and not restarted. The native host test it had written had
never run, and it had three defects that would each have invalidated the evidence it claimed to
produce. All three are repaired, and **the 3-frame chain now runs and passes 48 explicit checks
(exit 0)**.

What is proven, and nothing more:

* an FP16 Feature 18 session (`DXGI_FORMAT_R16G16B16A16_FLOAT`, 8 bytes/pixel) draws frames from
  **device** colour with **no host pixel staging at all** (`host_color_stage_count = 0`) and
  **no output readback on the processing path** (`output_readback_count = 0` at the snapshot);
* the CUDA/D3D12 row bridge moves a **`width * 8`** row, including the tail half of every row and
  every row after the first, from a source whose pitch is wider than the payload;
* the 8-bit surface in the chain is the **guidance plane only** (`guidance_pitch_bytes = 1024 =
  256 * 4`), which only NVOF reads; master, proxy and output stay FP16 end to end;
* reconstruction uses the **production** Lanczos-2 down / Catmull-Rom up plans, not an invented
  filter;
* mismatched descriptors, undersized pitches, unknown colour names and a second independent core
  are each refused by result rather than by assertion, a layer session can coexist with an FP16
  anchor, the anchor survives the layer's close, and a fresh session opens after the final close.

**This is not a production HDR10 export. No P010 encode, NVENC or UI path ran, so no HDR export
has passed.** Colour loss is *not* measured here — the genuine FP16/PQ losses stay on the record
in `HDR_STAGE_B_ROUNDTRIP_FIX.md` (fixture Y19/UV23, extreme sweep Y47/UV33); a host gate PASS
does not make them disappear. The FP16 session is an **experimental opt-in mode**: the original
formal colour contract is still not proven. Performance is **NOT_CHARACTERIZED**.

## The three defects, repaired before anything was run

### 1. The fabricated 2-tap plan (wrong resampler, wrong in the upscale direction)

`tap_plan()` built a 2-tap box for **both** axes in **both** directions with
`source = min(2 * destination, src_len - 1)`. For the reconstruct plan that maps 256 → 512, every
destination past 128 clamped onto the last two source rows — the composite the test then reported
was not a reconstruction, and a fixed 2-tap box is not the production contract in either
direction (`pipeline.py:2038-2042` documents the Lanczos window widening as the scale deepens).

Now the test builds the same `AxisPlan` objects the export job builds and serializes them into the
bridge's `[dst_len][taps]` layout:

| plan | builder | mapping | taps uploaded | row weight sum |
| --- | --- | --- | --- | --- |
| downscale (colour) | `pipeline._build_lanczos2_axis_plan` | 512 → 256 | **8** | 1.0 … 1.0 |
| reconstruct (residual) | `pipeline._build_catmull_rom_axis_plan` | 256 → 512 | **4** | 1.0 … 1.0 |

The bridge reads exactly `taps` columns per row (`gpu_hdr_scrgb_bridge.cpp:1668-1720`) and shares
one stride between both axes, so the narrower axis is padded to the wider one with weight `0.0` at
an already clamp-to-edge index. A plan wider than the table stride **raises instead of cropping**,
and `taps > GPU_HDR_MAX_TAPS` (24) refuses at `gpu_hdr_open`. The hard-coded `downscale_taps: 2`
and `reconstruct_taps: 2` claims are gone; `chain.json` records each axis's kernel, source/dest
lengths, real tap-count range, and the SHA-256 of the indices and weights actually handed to the
bridge (down weights `2cd6389d…`, up weights `195ce893…`). `filter_plan_builds_during_chain = 0`
confirms no plan was rebuilt per frame. Working extent is `pipeline.resolve_working_dims(512, 512,
0.5) = 256 x 256` by rule, no longer a literal.

### 2. The report read a closed session

`main()` closed the host and *then* evaluated `host.color_info()` inside the return expression, so
the run could only ever raise a closed-session error — and any exception path leaked the session,
the bridge and the CUDA context.

Now every session number is snapshotted while the session is open, and the release happens once,
in a `finally`, in dependency order: **bridge → host session → NVOF session**. The bridge is a
consumer of both the host's `ID3D12Device` (which it `AddRef`s inside `gpu_hdr_open`,
`gpu_hdr_scrgb_bridge.cpp:2135`) and the NVOF session's `CUcontext`, so it must die before either
owner; `dlss_gpu_host_d3d12_device` hands back an un-AddRef'd raw pointer, valid only for that
call. CUDA allocations and the pushed context keep their own `finally` blocks, and the report is
built exclusively from snapshots — by construction nothing can read a closed object.

### 3. Two independent cores, and a "reopen" that was not one

`_refusals()` opened an FP16 session and then a *second bare* `engine.Live()` with no
`shared_with`. `dlss_gpu_host_open` gates on one active core (`active.exchange(true)`,
`dlss_gpu_host.cpp:789`), so that call is refused — the test would have died on its own evidence
step. The gate was **not** relaxed to make the call work; instead:

* the bare second open is now itself a refusal case, and it is refused:
  `GPU-flow host open failed (2): host already active`;
* the SDR session joins the FP16 anchor through `engine.Live(..., shared_with=anchor)`, i.e. the
  existing `dlss_gpu_host_open_layer_ex` path — separate sessions, separate colour:
  anchor `session_id 1`, DXGI **10** (RGBA16F, 8 bytes) and layer `session_id 2`, DXGI **28**
  (RGBA8, 4 bytes);
* closing the layer leaves the anchor alive and still RGBA16F (`anchor_alive_after_layer_close =
  true`);
* after the final close a fresh independent FP16 session opens (`session_id 1`, DXGI 10), which is
  what "reopen" means here. `reopened_after_close` is now evidence, not `None`.

## What the chain measured (3 frames, 512×512 master, 256×256 working)

Snapshot taken **before** any pixel readback:

| counter | value | reading |
| --- | --- | --- |
| `host_color_stage_count` / `_bytes` | **0 / 0** | no host pixel staging on the processing path |
| `output_readback_count` / `_bytes` | **0 / 0** | nothing read back to draw the frames |
| `device_color_copy_count` | **3** | one device colour copy per frame |
| `device_color_copy_bytes` | **1 572 864** | `3 × 256 rows × 2048 B` = `width*8`, not `width*4` |
| bridge `process_count` | **3** | decode+downscale+proxy+reconstruct per frame |
| bridge `encode_device_write_bytes` | **0** | no encode ran — correctly, it is out of scope |

Per frame (all three identical in shape, different in content): reset `[1, 0, 0]`, Feature 18
output DXGI 10 at `[256, 256]`, fence values advancing to 3, master pitch 4096 B, working master
2048 B, proxy 2048 B, **guidance 1024 B (8-bit)**, NVOF flow 2048 B, composited 4096 B.

Session as the host itself reports it (`dlss_gpu_host_get_color_info`, read back rather than
assumed because the loaded interop DLL implements `cuda_d3d12_color_get_layout`): `row_bytes
2048`, `color_footprint_pitch 2048`, `output_footprint_pitch 2048`, `bridge_row_pitch 2048`,
`bridge_bytes_per_row 2048`, `last_copy_src_pitch 2048`, `last_copy_row_bytes 2048`,
`last_copy_fence_value 3`.

Capabilities/ABI: host ABI **3**, `color_formats 3` (RGBA8 | RGBA16F), `row_bridge_capabilities 3`,
`fp16_row_bridge_ready 1`, HDR bridge ABI **1**.

Colour reports, as measured and not smoothed: source `nonfinite_samples 0`,
`above_ceiling_components 0`, `negative_components 13544`, `max_component 8.945` scRGB against a
declared ceiling of 12.5; proxy `pinned_components 0`, `above_saturation_components 0`,
`dead_zone_components 35928`, `max_code 0.9653`; reconstruct `nonfinite_components 0`,
`composite_nonfinite_components 0`, `composite_out_of_domain_components 18900`, composite range
`-0.3035 … 8.2943`. The out-of-domain and negative counts are reported, not asserted to be zero.

## Row-tail transport (labelled diagnostic, not part of the frame path)

A `width * 8` FP16 row is copied by `cuda_d3d12_color_copy` out of a source whose pitch is
deliberately **2304 B for a 2048 B payload**, into a 256-byte-aligned plane that was wiped first,
and read back whole:

* `rows_whole_row_matches_source = 128 / 128`
* `rows_tail_half_matches_source = 128 / 128` — the second half of every row arrives
* `rows_tail_half_not_left_wiped = 128 / 128` — it arrived because the copy put it there
* `rows_whose_tail_is_their_head = 0` — it is not the first half repeated

Four malformed `create2` requests (row pitch below `width * pixel_bytes`, pitch not a 256 multiple,
`pixel_bytes = 5`, `width * pixel_bytes` overflowing `uint32`) are all refused. This proves the
transport, **not** what the network thought about those pixels.

## Refusals that were tested rather than assumed

| case | result |
| --- | --- |
| `color_format="rgba32i"` | refused: `unknown color_format … this host speaks ['rgba16f', 'rgba8']` |
| `rgba8_cuda` descriptor on an FP16 session | refused before any native call |
| FP16 session given a `width*4` pitch | refused (`pitch < width * bytes_per_pixel`) |
| second independent core while one is active | refused, `open failed (2): host already active` |
| FP16 open against an interop DLL with no wide-row transport | refused at open, code **23**, no SDR fallback |

## What this interface cannot tell you (stated, not oversold)

A raw 8-bit colour pointer handed to an FP16 session **with an FP16-shaped padded pitch cannot be
distinguished by row length**. The C entry points receive a device pointer, a pitch and an extent —
the format is settled once at open and is never a per-call argument
(`dlss_gpu_host.cpp:1095-1101`). The same contract runs the other way too: an FP16 pointer given to
an RGBA8 session is **not** refused, it is silently transported as `width*4` bytes per row. So:

* the raw C API is a **trusted typed contract** — the caller asserts the type;
* the enforcement that exists is in the Python wrapper: the descriptor name
  (`_COLOR_DESCRIPTOR`/`_check_color`, `dlss_gpu_engine.py:101-103, 401-409`) plus numpy dtype in
  `_check_plane`, which is why both refusal rows above raise without touching the DLL;
* it is **wrong** to claim the pitch check detects every format mismatch, and this report does not
  make that claim. No API redesign was needed for this stage; the honest gap is documented.

## SDR parity: kept, re-read, not re-run

The crashed job captured the SDR gate before and after rebuilding the production DLLs. Those
artifacts are preserved and were **only re-read** — a passing gate was not run again to collect a
label.

| file | host DLL | interop DLL | aggregate output SHA-256 |
| --- | --- | --- | --- |
| `sdr_pre_rebuild.json` | `7393a5fb…` | `80e5da43…` | `948fc30aeb2dd6ea…fd15b` |
| `sdr_pre_rebuild_r2.json` | `7393a5fb…` | `80e5da43…` | `948fc30aeb2dd6ea…fd15b` |
| `sdr_post_rebuild.json` | `5640d4fe…` | `da2db27e…` | `948fc30aeb2dd6ea…fd15b` |

* the two pre-rebuild files are equal **byte for byte**, so the original pre artifact matches the
  repeat rather than being a second opinion;
* all three agree on every per-frame **input** SHA (`d1d28629…`, `84a9fb6c…`, `eb65a82c…`), every
  per-frame **output** SHA (`2268f26f…`, `c3897b21…`, `8f15b1a1…`), extent, options and entry point;
* the binaries really did change (`7393a5fb… → 5640d4fe…`, `80e5da43… → da2db27e…`) while the
  drawn pixels did not;
* `sdr_post_rebuild.json` records exactly the hashes of the DLLs loaded by the FP16 run above
  (`5640d4fe…`, `da2db27e…`, nvof `7185767e…`), so the parity claim and this chain share one build;
* the FP16 session's `device_color_copy_bytes` is exactly **2×** the SDR figure
  (1 572 864 vs 786 432) at the same extent and frame count — the eight-byte row, visible in the
  counters.

Honest label, as instructed: that baseline reads its own output back to hash it
(`output_readback_count = 3`), so it is a **parity test, not a zero-readback production run**. The
zero-readback statement belongs to the FP16 processing path above, and only for the shipped
non-probe DLL — under `DLSS_GPU_HOST_FP16_PROBE` the host would add an uncounted D2H copy
(`dlss_gpu_host.cpp:1240-1246`), and this run's DLL exposes no probe symbols.

## Commands and exits

| command | exit |
| --- | --- |
| `python -m py_compile native/hdr_fp16_host_connect_smoke.py native/hdr_fp16_host_sdr_baseline.py dlss_gpu_engine.py` | 0 |
| `python -m ruff check native/hdr_fp16_host_connect_smoke.py` | 0, `All checks passed!` |
| `python native/hdr_fp16_host_connect_smoke.py bench_out/hdr_fp16_host_connect/chain.json` | 0 — `PASS: 48 checks hold` |
| native rebuild `/W4 /WX` | **not run**: no native source changed in this recovery, so a rebuild would only invalidate the DLL hashes the SDR parity evidence is pinned to |

Loaded binaries: `dlss_gpu_host.dll 5640d4fe…`, `nvof_d3d12_interop.dll da2db27e…`,
`nvof_bridge.dll 7185767e…`, `gpu_hdr_scrgb_bridge.dll 41a38ce9…`.

## Audit of the saved native API against the task spec

Requirements 1–4 of `build/qoder_hdr_fp16_host_connect.txt` are met: the additions are additive and
ABI 3 is unchanged; every copy, validation and counter path uses the runtime row width with 64-bit
overflow guards (`nvof_d3d12_interop.cpp:468, 593, 605`); an FP16 open against a host that cannot
transport it returns an explicit capability error (code 23) instead of drawing an 8-bit picture;
the FP16 device path never enters the host-staging branch (`dlss_gpu_host.cpp:1143, 1211`). The
untouched `* 4` sites are the R16G16 **motion** plane (`:1496, :1529`), which is correct.

Six findings are open and deliberately **not** patched in this bounded job (each is a comment, a
hardening gap or a teardown-order change; none changes what the gate measured):

1. `dlss_gpu_host.h:84-86` claims "codes 1 to 21 mean what they meant" — code **21** is new on the
   legacy path (`abort_open(21)`, `dlss_gpu_host.cpp:911`).
2. `dlss_gpu_host.cpp:1163-1174`: with an interop DLL that has no `get_layout`,
   `bridge_row_pitch`/`bridge_bytes_per_row` are assumed, contradicting the comment beside them.
   Not this run — the loaded DLL implements `get_layout`.
3. `dlss_gpu_host.cpp:733-748`: legacy close now destroys parameters before snippet shutdown;
   behaviour differs from the preserved order even though the gates pass.
4. `dlss_gpu_host.cpp:1152`: `bridge.create2` is dereferenced without a null check (safe today only
   because the FP16 open refuses unless `wide_ready`).
5. `nvof_d3d12_interop.cpp:588-605`: the **source** extent is never validated, only
   `pitch >= bytes_per_row`; the spec's "allocated buffer dimensions" is enforced on the
   destination only. A raw device pointer cannot close this gap by itself.
6. `dlss_gpu_host.cpp:938`: `dlss_gpu_host_d3d12_device` returns an un-AddRef'd pointer, valid only
   inside `gpu_hdr_open` — the reason the test's release order is bridge-first.

## Not established / remaining

* no HDR10 export, no P010 encode, no NVENC, no metadata — **do not call the HDR10 goal complete**;
* colour fidelity untouched by this gate; measured FP16/PQ losses remain in
  `HDR_STAGE_B_ROUNDTRIP_FIX.md`, and this gate's PASS says nothing about them;
* the formal colour contract is unproven; FP16 is an experimental opt-in session mode;
* the trusted-typed-contract gap above is documented, not closed;
* performance **NOT_CHARACTERIZED**.

Next bounded task: P010 encode + HDR10 static metadata and the NVENC handoff.

Nothing was committed, pushed, reset or cleaned; every prior artifact and all unrelated dirty
multiscale/UI edits are untouched.
