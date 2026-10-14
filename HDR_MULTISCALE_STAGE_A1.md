# HDR10 multiscale foundation - stage A1

Date: 2026-10-02. Owner of review: Codex. Implementation: Qoder (this document).

A1 is the native + wrapper foundation only. It adds three GPU-only abilities to the
experimental HDR10 scRGB bridge so that a second and third Feature 18 layer can be
fed, observed and delivered without a pixel ever coming to the host. It does **not**
implement fusion, does **not** wire multiscale into the HDR production path or the UI,
and does **not** claim any numerical or hardware result. Those are A2, B and C.

## 1. Files actually changed

| File | Change |
|---|---|
| `native/gpu_hdr_scrgb_bridge.h` | purely additive: `GpuHdrMultiscaleCounters`, `GPU_HDR_MULTISCALE_ABI_VERSION`, the `--- Multiscale foundation` section (5 entries + contracts), one clarified comment on `GpuHdrTransferCounters.process_count`. Zero deleted lines. |
| `native/gpu_hdr_scrgb_bridge.cpp` | `BorrowedCopy` + `copy_texture_to_plane` (generalising the existing Feature 18 import), `run_residual_pass` (shared with `gpu_hdr_reconstruct`), master/working/proxy epochs, the 4 new entries, multiscale counters. |
| `hdr_gpu_engine.py` | `GpuHdrMultiscaleCounters` mirror, `_MULTISCALE_BRIDGE` / `MULTISCALE_SYMBOLS` / `probe_multiscale`, `_plane_field` + `_require_scrgb16f_plane`, `_producer_texture` (replaces `_feature18_input`), `multiscale_capability` / `has_multiscale` / `import_master` / `observe` / `set_composite` / `multiscale_counters`. |
| `tests/unit/test_hdr_gpu_engine.py` | `FakeLibrary` multiscale support and 44 new tests (66 -> 110). The 66 pre-existing tests are unchanged. |
| `native/gpu_hdr_scrgb_bridge.dll` | rebuilt from the above sources (see section 6). |

Untouched, deliberately: the fusion solver (`gpu_multiscale_fusion_bridge`), the DLSS
Feature 18 host, NVOF / the flow provider, `gpu_hdr_transfer_export.py`,
`gpu_transfer_export.py`, `pipeline.py`, all remux code, `AGENTS.md`, and
`tests/unit/test_gpu_host_motion_wrapper.py` (its verified fixture repair from the
crashed job is left exactly as it is). Nothing was committed, pushed, stashed or
cleaned.

## 2. New native entries and capability group

```c
#define GPU_HDR_MULTISCALE_ABI_VERSION 1u
GPU_HDR_API uint32_t gpu_hdr_multiscale_abi_version(void);
GPU_HDR_API int32_t gpu_hdr_import_master(GpuHdrBridge* bridge,
    uint64_t source_ptr, uint32_t source_width, uint32_t source_height,
    uint32_t source_pitch, uint64_t* out_master_ptr, uint32_t* out_master_pitch);
GPU_HDR_API int32_t gpu_hdr_observe(GpuHdrBridge* bridge, void* dlss_texture,
    void* dlss_fence, uint64_t dlss_fence_value,
    uint64_t* out_residual_ptr, uint32_t* out_residual_pitch);
GPU_HDR_API int32_t gpu_hdr_set_composite(GpuHdrBridge* bridge,
    void* composite_texture, void* composite_fence,
    uint64_t composite_fence_value, uint64_t* out_composited_ptr,
    uint32_t* out_composited_pitch);
GPU_HDR_API int32_t gpu_hdr_get_multiscale_counters(GpuHdrBridge* bridge,
    GpuHdrMultiscaleCounters* out_counters);
```

The group is **optional and separately versioned**. It is not in `_REQUIRED_EXPORTS`
and not in `_BRIDGE`, so an ABI 1 bridge that predates it still opens and still runs
the whole single-layer chain. Capability is answered by symbol name plus the group's
own version - never inferred from `GPU_HDR_ABI_VERSION == 1` - and a *partial* group
is reported missing rather than bound half-way, so no new entry can be driven against
a build that lacks its siblings.

`GpuHdrMultiscaleCounters` (72 bytes, all `uint32_t`/`uint64_t`, guarded by
`struct_size`): `master_provenance`, `master_import_count`,
`master_import_device_bytes`, `observe_count`, `observe_copy_bytes`,
`observe_residual_bytes`, `composite_import_count`, `composite_import_bytes`,
`pass_dispatches`.

## 3. Python signatures

```python
bridge.multiscale_capability      # {"supported","version","missing","reason"}, probed once
bridge.has_multiscale             # bool
bridge.import_master(source)      # DevicePlane: full-extent rgba16f_cuda, plane 2
bridge.observe(dlss_output, fence=None, fence_value=None)
                                  # DevicePlane: working-extent rgba16f_cuda, plane 6
bridge.set_composite(composite_output, fence=None, fence_value=None)
                                  # DevicePlane: full-extent rgba16f_cuda, plane 7
bridge.multiscale_counters()      # plain JSON-safe dict, struct_size stripped
```

`hdr.probe_multiscale(library)` never raises, so a caller that only wants the single
layer can ask before deciding what to bind.

## 4. Format, units, synchronisation, ownership

Units everywhere are the bridge's existing ones: linear scRGB, `1.0 == 80 cd/m2`,
ST 2084's top at `125.0`. Paper white stays fixed at open, the proxy policy stays as
it is, the reconstruction budget (`kMaxStops`, `kMinSlopePerStop`) is unchanged, and
the support and encode ceilings stay per-call. No exposure control was added.

**Imported master** (`import_master`, plane 2, full extent, RGBA16F).
`source` is a typed full-resolution linear scRGB FP16 plane - a `DevicePlane` as
another bridge's `decode()` returned it, or the equivalent dict. The wrapper refuses
before the bridge is ever reached: absent pointer, `format != "rgba16f_cuda"`, a
`pixel_format` that says `R8G8B8A8` or `P010` while the format claims 16-bit float,
`bytes_per_pixel != 8`, an undeclared or mismatched extent, an undeclared pitch, a
pitch under `width * 8`, a pitch that is not a multiple of 16 (what a pitched
`cuMemcpy2D` requires), and a declared `cuda_context` that is not the context the
bridge was opened in. The native entry copies the plane into the bridge's own master
and does nothing else: no resample, no clamp, no transfer function, no 8-bit step, no
re-decode. The producer keeps ownership and may reuse or free the source the moment
the call returns.
Completion rule, recorded in the header because the bridge cannot check it: the plane
must already be fully written and device-visible. `decode()`'s plane is (the decode
pass waits its own fence before returning), and so is any producer that blocks until
its GPU work retired. There is no fence argument, because a CUDA device pointer has no
fence to name and inventing one would be a claim the bridge cannot keep.

**Signed working residual** (`observe`, plane 6, working extent, RGBA16F).
RGB holds `reconstruct(working master) - working master` in linear scRGB: signed, so a
component is legitimately negative, never normalised into `[0,1]`. Alpha is `0.0` in
every sample - padding, not coverage. The observation is the identical `kPassResidual`
dispatch the single-layer path already runs (same three inputs, same constants, same
owned plane, one `run_residual_pass` helper), so it is the same number, not a second
opinion. No Catmull-Rom, no `residual_strength`, no composite, no encode: weighting
and upscaling are the fused decision made once over all layers.

> Corrected in A2, review item 1. As first written, `gpu_hdr_proxy` stamped its
> provenance with the *master* epoch, so `import master -> proxy -> downscale ->
> observe` passed: that proxy had sampled the working plane of the previous master,
> and the later downscale silently made it look current. A proxy now records the
> *working generation* it actually read and `gpu_hdr_downscale` retires the proxy of
> the plane it rewrites, so the sequence above is refused with a message naming the
> proxy. `gpu_hdr_observe` now requires two things: the working plane belongs to the
> master this bridge holds now, and the proxy read that working plane. No old entry
> gained a refusal and no maths changed.

Staleness is refused natively via the master epoch and the working generation: an
observation whose `downscale` or `proxy` belongs to an older master - or to a working
plane a later downscale replaced - fails instead of returning a plausible number.
Borrowed producer input follows the existing Feature 18 rule unchanged
(descriptor dict, or raw texture plus fence and value).

**Fused composite** (`set_composite`, plane 7, full extent).
A full-resolution `R16G16B16A16_FLOAT` texture from the same D3D12 device, in
`COPY_SOURCE` state, plus the producer's fence and value. It is imported by the same
placed-footprint device-to-device mechanism the Feature 18 output already uses, one
extent up, into the plane `gpu_hdr_encode` already reads - so the existing PQ/P010
encode delivers the fused picture with no new encoder, no new tone map, no new colour
claim and no host pixel traffic. Values are stored exactly as they arrive; the encode
stage remains the only place anything clamps. Unlike the Feature 18 copy, this entry
**requires** the fence (`kCompositeCopy.require_fence = true`): the solver writes
asynchronously on another queue, so a null fence or value 0 is refused by both wrapper
and bridge rather than assumed complete. Only the D3D-texture form was implemented; an
equivalent CUDA-plane variant was deliberately not added, because the downstream
consumer is the fusion solver's own D3D12 output texture and a second path would be a
duplicate API.

## 5. How the old single-layer path is preserved

- Every old export, its argument order, its error strings and its semantics are
  unchanged. `gpu_hdr_reconstruct` still runs copy_neural -> residual pass ->
  composite, in that order (asserted), still returns the full FP16 composite plane,
  and still accepts a Feature 18 output with no producer fence
  (`kFeature18Copy.require_fence = false`), which is the pre-A1 native behaviour.
- `copy_neural` became one call to `copy_texture_to_plane` with a contract object. The
  validation (2D texture, RGBA16F, working extent, same device, optional queue wait,
  placed-footprint copy, completion wait) is the same code path, and all four original
  refusal messages were preserved verbatim through the contract fields.
- The residual constants moved verbatim into `run_residual_pass`; the native source
  now contains exactly one `run_pass(b, kPassResidual,` and one
  `r.put(float(kMinSlopePerStop));`, asserted in the tests, so there is no second copy
  of the reconstruction that could drift.
- `GpuHdrTransferCounters` layout is untouched (field-for-field test against the
  header), and the single-layer per-frame totals are identical: on a successful
  reconstruct the same bytes are attributed to the same fields, with the residual
  write counted immediately after it happens instead of after the composite.
- The epochs are enforced only by the new entries; no old call gained a refusal.
- An old DLL without the group still loads, opens, and runs decode -> downscale ->
  proxy -> reconstruct -> encode (tested end to end against a group-less fake), while
  all four new methods raise a message naming the missing symbols.

## 6. Counters, honestly

Existing fields carry the new traffic where the traffic is the same *kind* of traffic:
`import_master` bumps `process_count` and `source_device_write_bytes` (a frame of
colour entered the bridge, by import rather than by decode) and
`working_device_write_bytes` for the neural and residual writes of `observe`, and
`set_composite` bumps `composited_device_write_bytes`. `source_device_copy_bytes` -
the decoder's borrowed P010 - is **not** re-used for an FP16 import. Per-layer and
per-stage volumes (import/observe/composite counts and bytes, and the master's
provenance: 0 nothing, 1 decode, 2 import) go in the new struct rather than into a
changed layout.

`report_readback_bytes` stays 0 for all three new paths: none of them downloads an
image plane. That is *not* a claim that no host traffic exists. Every executed pass
still copies its counter run (`kCounterSlots * sizeof(uint32_t)` = 256 bytes) from the
device to a CPU-visible buffer, and no field in either struct measures it in bytes.

> Corrected in A2, review item 2. As first written, `pass_dispatches` was incremented
> by `gpu_hdr_observe` alone, while this section told a caller to multiply it by 256
> for *all* of the bridge's scalar readbacks - which silently missed decode, the two
> downscale axes, proxy, reconstruct's composite and encode. The count now happens in
> `run_pass`, the one function every shader pass of this bridge goes through, and
> `gpu_hdr_observe` no longer counts anything itself. `pass_dispatches` is therefore
> the number of passes this bridge *completed*: on a successful frame it times 256
> bytes is every counter readback the bridge took, and after a failed pass it is the
> completed total rather than the submitted one, because a pass that never read its
> counters back does not reach the increment. The copies the new entries run -
> imported master, Feature 18 input, composite import - dispatch no shader and add
> nothing. All counter readbacks are still reported as their own line: no field and no
> sentence here claims total host traffic is zero.

`master_provenance` also tells a consumer that an imported master
has no source report behind it - the decode pass is the only thing that measures one.

## 7. Build and targeted test evidence

Native build (final tree, after all edits):

```
cd native && cmd //c build_gpu_hdr_scrgb_bridge.bat     exit code 0
gpu_hdr_scrgb_bridge.cpp
  creating gpu_hdr_scrgb_bridge.lib and gpu_hdr_scrgb_bridge.exp
-rwxr-xr-x 251392  gpu_hdr_scrgb_bridge.dll   (22:42:48, newer than .cpp 22:21:41 / .h 22:17:28)
```

`/W4 /WX` is the build's own setting, so the compile is warning-free. The rebuilt DLL
was then loaded and asked, no GPU work submitted:

```
abi 1  multiscale_abi 1
missing multiscale exports: []
probe against the real dll: {'supported': True, 'version': 1, 'missing': [], 'reason': ''}
```

Unit tests (no GPU, no DLL loaded by the tests themselves; every refusal is asserted
at the real boundary with an injected fake ABI):

```
python -m pytest tests/unit/test_hdr_gpu_engine.py -q              110 passed in 1.26s
python -m pytest tests/unit/test_hdr_transfer_export_branch.py \
                 tests/unit/test_packaged_hdr_imports.py \
                 tests/unit/test_gpu_host_motion_wrapper.py -q     174 passed in 3.38s
```

No skip, no xfail, no tolerance loosened, and the 66 pre-existing HDR wrapper tests and
the 14 verified motion-wrapper tests were not modified.

Failures encountered and fixed inside this job:

1. `test_import_master_is_a_copy_and_nothing_else` and
  `test_the_new_entries_keep_their_own_counters_apart_from_the_old_struct` failed
  because the source-parsing helper searched the entry's **prose**: the comments in
  `gpu_hdr_import_master` legitimately mention the decoder's borrowed P010 and the
  counter they must not touch. Fixed by making `_function_body` strip `/* */` and `//`
  comments, so these assertions verify code rather than sentences - the tests got
  stricter, not weaker.
2. `test_observe_returns_the_signed_working_residual_plane` and
  `test_set_composite_imports_the_full_extent_texture` compared a `ctypes.c_uint64`
  argument to a Python int, which is never equal. Fixed by reading `.value`.
3. `test_the_old_reconstruction_still_does_not_require_a_fence` asserted a premise that
  was false at HEAD: the wrapper has always required a raw-pointer producer to name
  its fence (`"a raw texture needs its fence and fence_value"`). It was rewritten as
  `test_the_old_producer_fence_rule_survives_and_only_the_new_entry_tightens_it`,
  which pins the old wrapper refusal, the descriptor form, the old native
  `require_fence = false`, and the new composite's `true`.

Also recorded for the audit trail: the crashed job `qbg_20261002_8ca407f3` had left a
rebuilt `gpu_hdr_scrgb_bridge.dll` whose sources contained none of these entries; that
binary was not treated as evidence, and the DLL now checked in (unstaged) corresponds
to the sources above.

## 8. What A1 does not claim

No fusion, no multiscale production wiring, no UI change, no HDR smoke run, no full
unit or benchmark suite, no hardware numeric gate, and nothing about visual quality.
`build_gpu_hdr_scrgb_bridge_test.bat` was not run, so the existing roundtrip gate has
not been re-executed. The new entries have never been exercised against real pixels on
real hardware - that is the first thing A2 and C do. This is an independent
implementation; no GPL source was read or copied.

## 9. How A2 hooks in

1. **Extra layers.** For each scale, open its own `GpuHdrBridge` (same D3D12 device,
   same CUDA context, same fixed paper white) and call
   `import_master(layer0_full_master)` - the plane layer 0's `decode()` returned - so
   every layer starts from one pristine master instead of a second decode. Then run
   that layer's own `downscale()` and `proxy()`, in that order and after the import -
   `observe()` refuses a proxy that read a working plane a later `downscale()`
   replaced - and feed `proxy()`'s descriptor to
   its own `Live(color_format='rgba16f')` session.
2. **Observation.** After each layer's Feature 18 call, `observe(their_dlss_output)`
   returns `(ptr, pitch)` at the working extent, `rgba16f_cuda`, plane 6, signed
   scRGB, alpha 0. That borrowed pair is the `r_i` the existing CG objective consumes:
   `min_r sum_i w_i ||D_i r - r_i||^2 + beta ||r - r0||^2`, in scRGB units, with the
   plan's own float32 taps and their true transpose as `D_i^T`. Read it before the
   next `observe()` or `reconstruct()` on that same bridge rewrites the plane, or give
   each layer its own bridge (the recommended shape: the residual plane is owned
   per-bridge, so one bridge per layer is what makes simultaneous observations safe).
   `multiscale_counters()["pass_dispatches"]` is what to multiply by 256 bytes when
   reporting the scalar boundary - corrected in A2: it counts every shader pass this
   bridge completed (decode, both downscale axes, proxy, residual, composite, encode),
   not the observations alone, so the product is the run's whole counter readback
   rather than a part of it. Report it as its own line; picture readback being zero is
   not host traffic being zero.
3. **Delivery.** The solver writes the fused full-resolution `R16G16B16A16_FLOAT`
   texture on its own queue; hand that texture plus its fence and fence value to one
   bridge's `set_composite(...)` and then call `encode(clip_ceiling_scrgb=...)` on the
   same bridge. Do not apply the SDR `[0,1]`/`[0,255]` composite anywhere here, and do
   not add a clamp before encode.
4. **Refusals are failures.** `has_multiscale` false, a partial group, a version
   mismatch, a stale-proxy `observe`, or a fence-less composite must stop the job with
   a reason. None of them may silently fall back to the single layer.
