# HDR Stage B — backend audit repair (readback semantics, packaging reachability, cookie ownership)

Repairs the three findings Codex raised against the completed background job
`qbg_20261001_4bae1ad6`. It is a diagnostics-and-loading repair only: no export was
driven, no GPU gate was re-run, no benchmark was run, no native source was compiled, and
the accepted hardware evidence was not rewritten.

The HDR task is **not finished**. The UI opt-in is still pending, so nothing in the
product can set `experimental_hdr_scrgb` from a window yet. Performance remains
**NOT_CHARACTERIZED**.

Everything below was read out of this tree; each claim names the file and line it came
from, and the counters it quotes are the ones the code actually raises.

---

## 1. The false claim, and the corrected counter semantics

### What was claimed

`gpu_hdr_transfer_export.hdr_transfer_report` filed the colour bridge's
`report_readback_bytes` under the key `hdr_scalar_report_readback_bytes_measured`, and
`HDR_STAGE_B_PRODUCTION_EXPORT.md` then read that field's `0` as "a report struct is not
fetched off the device". The production gate's checklist line
`report structs cost no readback, and were read anyway` asserted the same thing.

### Why it is wrong

`report_readback_bytes` is the bridge's counter for **image planes** copied to the host:

```
native/gpu_hdr_scrgb_bridge.cpp:2022   (read_plane_to_host)
    b->counters_state.report_readback_bytes += plane.pitch * plane.rows;
```

and `read_plane_to_host` is called only from the three diagnostic entries
`gpu_hdr_readback_source_p010` / `gpu_hdr_readback_encoded_p010` /
`gpu_hdr_readback_scrgb` (`native/gpu_hdr_scrgb_bridge.cpp:2688`, `:2700`, `:2712` in
the current source). The HDR chain binds none of them — `hdr_gpu_engine` deliberately
excludes them from `_REQUIRED_EXPORTS` — so the counter reads `0` and means *no plane
came out*. That zero was never a measurement of the scalar path, and the scalar path
does cross.

### What the bridge actually does per dispatch

```
native/gpu_hdr_scrgb_bridge.cpp:42      constexpr uint32_t kCounterSlots = 64u;
native/gpu_hdr_scrgb_bridge.cpp:1322    create_host_visible(b, kPlaneCounterReadback,
                                kCounterSlots * sizeof(uint32_t), D3D12_HEAP_TYPE_READBACK, false)
native/gpu_hdr_scrgb_bridge.cpp:1948    list->CopyBufferRegion(kPlaneCounterReadback, 0,
                                                    kPlaneCounters, 0,
                                                    kCounterSlots * sizeof(uint32_t));
native/gpu_hdr_scrgb_bridge.cpp:1964    readback->Unmap(0, nullptr); readback->Map(0, nullptr, &host);
native/gpu_hdr_scrgb_bridge.cpp:1967    std::memcpy(b->stage_counters[pass], b->counter_readback,
                                                    sizeof(b->stage_counters[pass]));
```

`kPlaneCounterReadback` lives in a `D3D12_HEAP_TYPE_READBACK` buffer, i.e. CPU-visible
memory, and `run_pass` copies it into the host-side `stage_counters` snapshot before
returning. That snapshot is what every report getter answers from
(`gpu_hdr_get_source_report`, `gpu_hdr_get_proxy_report`,
`gpu_hdr_get_reconstruct_report`, `gpu_hdr_get_encode_report` all read
`counter(b, pass, slot)` / `counter_float(...)`), so the reports the Python layer reads
are the tail of a GPU-to-host transfer.

**Size:** `kCounterSlots * sizeof(uint32_t)` = 64 × 4 = **256 bytes per executed pass**.
The bridge has seven passes (`enum PassId` at `native/gpu_hdr_scrgb_bridge.cpp:308`:
decode, vertical, horizontal, proxy, residual, composite, encode).

**Status:** permitted. The contract this chain is written against forbids image and
pixel transfers across the host boundary, not scalar diagnostics — and the earlier
accepted record said as much: `native/hdr_real_admission_gate.py` closes with
`"the only device-to-host traffic is the scalar reports"`, and
`HDR_STAGE_B_REAL_ADMISSION.md` reports `report_readback_bytes == 0` as evidence that no
**pixel** reached the host, which is what that counter does prove. Only the later label
was wrong.

**Not measured.** No field of `GpuHdrTransferCounters` counts these bytes, so the byte
count of this boundary is **unknown / None**, and the 256 B figure above is read off the
source constants. It is stated as a source fact and is nowhere labelled as a
measurement; no `256 × dispatches` product is reported anywhere.

### The corrected fields

| Report key | Meaning | Value on a production run |
| --- | --- | --- |
| `hdr_diagnostic_image_readback_bytes_measured` | the bridge's own `report_readback_bytes`: image-plane bytes copied out by the `gpu_hdr_readback_*` entries | `0`, and the zero is the bridge's answer |
| `hdr_scalar_report_readback_bytes_measured` | the per-pass counter readback | **`None`** — no counter answers it. Not `0`, and not the field above |
| `hdr_scalar_report_readback` | structural note: `crosses_host: true`, `bytes_measured: None`, `per_dispatch_bytes_from_source: 256`, `measured_by_this_report: false`, the `evidence` citation, and a `why_not_a_total` clause | present in every report the repaired code writes |
| `hdr_pixel_readback_bytes_measured` | the complete HDR picture-byte readback figure | `None`, unchanged: this is the unavailable measurement, distinct from the measured diagnostic counter |
| `hdr_not_claimed["host_traffic"]` | states that nothing claims all host traffic is zero | new |
| `colour_bridge.scalar_counter_readback_per_dispatch` (capability read) | the same fact at capability level, so a refusal records it too | new |

`gtx.transfer_report`'s naming rule is unchanged and is restated in the docstring of
`hdr_transfer_report`: `_measured` is a bridge's own number or `None`, `_logical` is a
buffer the code describes, and a third kind now covers a boundary the source says exists
and no counter measures. Nothing sums them.

### Offline inspector semantics

`tests/unit/hdr_production_export_gate.py` now draws two lines where it drew one, and
read-backs a saved report under both names:

* `diagnostic image-plane readback is the bridge's own zero` — reads
  `hdr_diagnostic_image_readback_bytes_measured`, falling back to the pre-repair key with
  an `OFFLINE RE-READ` provenance string in the line's own detail;
* `scalar counter readback is present and not reported as a measured zero` — accepts
  `unanswered` (the current shape), `absent` and `misfiled` (a pre-repair report, labelled
  as such), and fails the one shape that would repeat the original fault: the same counter
  booked as a measured scalar byte count while the current image field is also present
  (`state = doubled`).

Every image boundary the gate already held at zero is still held at zero, unchanged:
the seven host counters, the decoder scale field, `hdr_pixel_readback_exports_bound == []`
and `decoder_pixel_readback_entry_bound == false`, and `hdr_pixel_readback_bytes_measured`
still `None` rather than zero.

Read-only re-interpretation of the **accepted** report (`--recheck` was **not** run,
because it rewrites `gate_log.json`; the same helpers were called directly on
`bench_out/hdr_production_export/diagnostics.json`, no export, no writes):

```
image : hdr_scalar_report_readback_bytes_measured = 0 | OFFLINE RE-READ ... no new measurement was taken
scalar: misfiled | value 0 | ... the boundary is accepted as present-and-unmeasured
frames_reported = 6  process_count = 6
pixel readback measured = None | exports bound = [] | decoder entry bound = False
```

`total_frames_reported = 6` and `hdr_process_count_measured = 6` are the two counters that
say the passes ran, which is what makes the scalar boundary a fact of that run rather
than an assumption about one.

---

## 2. Packaged reachability

### Measured, not asserted

`PyInstaller 6.22.3`, `Python 3.11.15`,
`C:\Users\18392\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe`, PyInstaller's
own `ModuleGraph` over the spec's own entry script `gui.py` — first-party search path,
0.5 s, no build:

```
first-party collected: ['bundle_paths', 'dlss_gpu_engine', 'gpu_transfer_export',
                        'optical_flow', 'pipeline', 'transfer_profiler']
first-party NOT collected: ['dlss_service', 'gpu_hdr_transfer_export',
                            'hdr_gpu_engine', 'multiscale_fusion']
```

and on the scanner's semantics, isolated two-file fixtures:

```
  uses_literal.py        -> literal_dep      collected: False
  uses_from_literal.py   -> from_literal_dep collected: False
  uses_static.py         -> static_dep       collected: True
```

So a literal `__import__("gpu_hdr_transfer_export")` is **not** an edge PyInstaller
follows, and the two HDR modules were genuinely absent from the bundle. The call sites
are `pipeline.py:3174`, `pipeline.py:3276` and `dlss_service.py:183`;
`gpu_transfer_export` survives only because `gui.py:1933` names it in an import
statement. A frozen exe would have resolved the HDR route and raised as soon as the flag
was reachable from the UI.

### The fix

`DLSS5Tool_standalone.spec`: the two entries, with the measurement above written down
beside them.

```python
HIDDENIMPORTS = ['PIL.Image', 'PIL.ImageTk',
                 'gpu_hdr_transfer_export',
                 'hdr_gpu_engine']
```

Lazy native loading is preserved: being collected makes the module importable at
analysis time, not loaded at runtime — `hdr_gpu_engine.load()` stays the only place
`ctypes.CDLL(native/gpu_hdr_scrgb_bridge.dll)` is called (`hdr_gpu_engine.py:347`), and
`_Cuda` constructs `C.WinDLL("nvcuda.dll")` only when a chain runs
(`gpu_hdr_transfer_export.py:296`). `tests/unit/test_packaged_hdr_imports.py` checks
that in a fresh subprocess with `ctypes`' loaders watched rather than blocked, and also
pins the two seams (`module` / `hdr_module`, `module` / `transfer_module`) and the
positional parameter order of `pipeline.export_gpu_transfer_stream` so the SDR call
signature is provably unchanged.

Two observations, deliberately left alone: `dlss_service.py` (and with it `server.py`) is
not in the packaged entry graph at all — `Analysis` is rooted at `gui.py` — and
`multiscale_fusion.py` is unreferenced by product modules. Neither is this repair's
subject, and the hiddenimports fix covers both routes the HDR chain can be reached by.

No distributable was built and nothing was compiled.

---

## 3. Loader cookies on an HDR refusal

`pipeline.export_gpu_hdr_transfer_stream` loads the bridge set through
`gtx.load_bridges`, which opens the FFmpeg DLL-directory cookies
(`gpu_transfer_export.py:324`) and hands them out inside the set
(`gpu_transfer_export.py:480`). `export_gpu_hdr_transfer` releases them in its own
`finally`. A job that the HDR gate refused never reached that `finally`, and the refusal
raised with the cookies still open — one leaked handle set per refused job.

Ownership is now explicit and single-path:

* `gpu_hdr_transfer_export.release_loader_cookies(bridges)` (new, `:466`) closes the set's
  cookies through `gtx._close_cookie`, **empties the list** so a second call is a no-op
  rather than a double close, returns the count released, and touches nothing else — it
  does not unload a library, close a bridge or free anything another session owns;
* `pipeline.py:3325` releases before re-raising anything that keeps the job from the
  hand-over line (a `negotiate_hdr` that raises), and `pipeline.py:3332` releases on the
  refusal itself, before the diagnostics update and the `RuntimeError`;
* `gpu_hdr_transfer_export.py:1692`, the chain's `finally`, now calls the same helper, so
  there is one definition of "released". An accepted job hands the set over untouched:
  the entry point calls nothing on that path.

CPU stubs prove both directions: `test_a_refused_hdr_job_releases_the_cookies_its_bridge_set_loaded`
(released = 2, each handle closed exactly once, list emptied),
`test_a_negotiation_that_raises_releases_the_cookies_before_it_propagates`,
`test_release_loader_cookies_touches_the_handles_and_nothing_else` (idempotent, no symbol
reaches either bridge, `None`/`{}` are safe), and the accepted-route test asserts the
entry point released nothing and the chain released one.

The SDR entry point has the same shape — `pipeline.export_gpu_transfer_stream` raises on
its own refusal with the cookies open — and is **left exactly as it is**, because the
brief keeps the SDR path unchanged and its module hash is evidence in accepted records.
Recorded here as a known pre-existing gap, not silently fixed and not claimed as fixed.

---

## 4. Changed files

Reporting and semantics:

| File | Change |
| --- | --- |
| `native/gpu_hdr_scrgb_bridge.h` | comment-only: the `GpuHdrTransferCounters` preamble now names the two host classes and states that no field measures the per-pass counter readback; `report_readback_bytes`' own field comment says image planes; the readback-entry block says what its zero does and does not mean |
| `native/gpu_hdr_scrgb_bridge.cpp` | comment-only: file header states the scalar exception; `run_pass` names the 256-byte-per-dispatch boundary at the copy that makes it; `read_plane_to_host` labels its counter as image bytes |
| `hdr_gpu_engine.py` | docstrings only: module docstring, `GpuHdrTransferCounters` mirror, `transfer_counters()`, and the report-getter section note that a report arriving is evidence the scalars crossed |
| `gpu_hdr_transfer_export.py` | report fields corrected (§1) plus `hdr_not_claimed["host_traffic"]` and the capability-level note; docstrings corrected; `release_loader_cookies` added; the chain `finally` routed through it |
| `pipeline.py` | HDR entry only: cookie release on refusal and on a raising negotiation, ownership paragraph in the docstring |

Tests and spec:

| File | Change |
| --- | --- |
| `tests/unit/hdr_production_export_gate.py` | the false checklist line replaced by the two corrected ones, the `image_readback_counter` / `scalar_readback_claim` read-back helpers, inspector docstring states the scalar boundary is present-and-unmeasured. Not run |
| `tests/unit/test_hdr_transfer_export_branch.py` | stub counter commented for what it measures; new report-semantics test; three new cookie-ownership tests; accepted-route test now proves hand-off |
| `tests/unit/test_hdr_gpu_engine.py` | `test_transfer_counters_expose_only_real_fields` now pins that the ABI's only readback field is the image-plane one |
| `tests/unit/test_hdr_production_gate_readback_reread.py` | new: the inspector's read-back over current / pre-repair / doubled / absent report shapes, and a guard that the removed claim does not come back |
| `tests/unit/test_packaged_hdr_imports.py` | new: the packaging and import contract check |
| `DLSS5Tool_standalone.spec` | two hiddenimports plus the measurement that explains them |

No native ABI, no DLL, no maths, no shader, no constant changed. `HDR_SUPPORT_CEILING_SCRGB`
250.0, `HDR_CLIP_CEILING_SCRGB` 125.0 and `HDR_PAPER_WHITE_SCRGB` 8.0 are untouched, and
the SDR chain's modules are byte-identical (`gpu_transfer_export`, `dlss_gpu_engine`,
`dlss_service` — verified against the accepted manifest below).

---

## 5. Focused tests run, and what they said

```
python -m pytest tests/unit/test_hdr_transfer_export_branch.py
                 tests/unit/test_hdr_gpu_engine.py
                 tests/unit/test_hdr_production_gate_readback_reread.py
                 tests/unit/test_packaged_hdr_imports.py
                 tests/unit/test_packaged_binaries.py        -> 173 passed
python -m pytest tests/unit/test_export_cpu_flow_gpu_host.py
                 tests/unit/test_export_backend_ui_contract.py -> 49 passed
python -m py_compile pipeline.py gpu_hdr_transfer_export.py hdr_gpu_engine.py \
                 tests/unit/hdr_production_export_gate.py \
                 tests/unit/test_hdr_transfer_export_branch.py \
                 tests/unit/test_hdr_gpu_engine.py \
                 tests/unit/test_hdr_production_gate_readback_reread.py \
                 tests/unit/test_packaged_hdr_imports.py      -> ok
ast.parse(DLSS5Tool_standalone.spec)                          -> ok
```

Per file: branch 85, engine 66, gate read-back 5, packaging 5, packaged binaries 12,
SDR flow and UI contract 49. **Zero failures, zero skips.** No numeric A/B, no native
gate, no hardware export, no benchmark was run: none of those questions changed.

---

## 6. Accepted hardware evidence, untouched

The single real six-frame production export of `qbg_20261001_4bae1ad6` remains the
evidence it was, and remains the only HDR export in this record:

* `bench_out/hdr_production_export/settings_and_inputs.json`, `diagnostics.json`,
  `gate_log.json`, the deliverable, both elementary streams, and
  `run1_inspector_false_negative/` — **unmodified**, byte-for-byte;
* deliverable sha256 `4727d40adfef6b93b07ca51a15a9f9b70158eafc70229723964bb4b9363ea5d6`;
* the nine native libraries hashed by that gate all still hash to their recorded values:
  `nvdec_bridge 263c4397876e…`, `nvenc_bridge 007871489848…`,
  `nvof_bridge 7185767ec96a…`, `nv12_rgba_bridge 6eae144b5d6f…`,
  `gpu_lanczos_bridge 773037269fbc…`, `gpu_residual_bridge 7b8a438d0997…`,
  `gpu_hdr_scrgb_bridge e016fce6e1b3…`, `dlss_gpu_host 5640d4fe0fcb…`,
  `nvof_d3d12_interop da2db27e49e4…` — **UNCHANGED**, so nothing was rebuilt and no
  binary drift was introduced by comment edits;
* of the six Python modules that gate hashed, three changed and three did not:

  | module | accepted gate | after this repair |
  | --- | --- | --- |
  | `pipeline` | `30b75f4a1bf4f8fead674c2403649e7b043c3301c530f04a4c2bfc5391008e3f` | `e859d1411f157ae9f09b8752949ee38270a0ca70a793f5c60690f149037fdedf` |
  | `gpu_hdr_transfer_export` | `576c86923e9e41451192d1bf5e5af5b136d5cfc58867191856c6bd46c404392d` | `b65c19389085d83968344a905f82fb7be9c0b5e829aa4a535b7b017c60c32dd4` |
  | `hdr_gpu_engine` | `7f8c5734dbfb3978357606e5d7bac87adc32f81174772e5bba1903d992bc7ff5` | `e3524f194e05eaaa79a6ddf112cf78a835eddb3431fee0a6f6f4d68a23f6ea05` |
  | `gpu_transfer_export` | `9b7e8761af42cdfc…` | unchanged |
  | `dlss_gpu_engine` | `afd0f26c94464cde…` | unchanged |
  | `dlss_service` | `baf78d5c6013e7a5…` | unchanged |

The accepted gate is **prior-code evidence**: it validated the FP16 picture transport,
the real NVOF motion, the Main10/PQ/BT2020 declaration, the 137/144 SEI payload
preservation and the seven image-host boundary counters as those modules then stood. This
repair changes reporting and loading only — the three modules above were not re-validated
on hardware by it, and `settings_and_inputs.json` was deliberately **not** updated to
imply that they were. `hdr_gpu_engine`'s edits are docstring-only; `pipeline`'s and
`gpu_hdr_transfer_export`'s are report naming, notes and cookie release. Nothing in the
maths, the borrow order or the encoder request moved, which is why the picture, motion,
metadata and boundary claims of that export still stand — with the one claim about scalar
readback corrected above, in the direction of *more* host traffic than the record asserted,
not less.

How the earlier records read now, without being edited:
`HDR_STAGE_B_REAL_ADMISSION.md` and `HDR_STAGE_B_VALID_FIXTURE.md` quote
`report_readback_bytes == 0` as evidence that no pixel reached the host. That evidence is
sound — it is the image-plane counter, and its zero is real. The one phrase superseded by
this repair is any wording that treats the same field as a count of the scalar report
reads; `HDR_STAGE_B_PRODUCTION_EXPORT.md`, which made that claim outright, carries the
correction inline.

---

## 7. Still open

* **UI opt-in for the HDR chain: not done.** No product surface sets
  `experimental_hdr_scrgb`, so the chain is reachable only through a settings dict. That,
  not this repair, is what keeps the HDR task unfinished.
* **Performance: NOT_CHARACTERIZED.** Nothing here measured throughput, latency or PCIe.
* The scalar counter readback is still unmeasured by the ABI; making it a measured figure
  would need a native counter, which is out of scope and would invalidate the accepted
  binary hashes.
* The SDR entry point's refusal leaks its loader cookies the same way (§3); left untouched
  on purpose.
* This repair adds no hardware re-validation. If the HDR chain is ever run again on
  hardware, the gate's new read-back lines are what that run should be judged with.
