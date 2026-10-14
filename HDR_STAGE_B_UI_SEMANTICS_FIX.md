# HDR Stage B — UI semantics correction (preview, guards, docs)

Corrects the three false statements Codex found in the **product prose** of the completed UI job
(`qbg_20261002_222c0492`), plus three adjacent factual errors, and rewrites the tests that were
guarding them. Nothing in the backend, the native sources, the maths, the DLLs, the UI layout, the
settings path, the payload or the SDR chain changed: every edit in this pass is a string literal,
a test assertion, or a document.

Not re-run, deliberately: A/B, numeric, production export, Playwright and CPU gates, and no new
hardware export was driven. The functional evidence the job already had stands as recorded — build
PASS, node 44 PASS, CPU 125 PASS, mocked UI 3 PASS. Raw gate artifacts under `bench_out/` were not
touched. **Nothing was committed, pushed, stashed, cleaned or reset; the dirty baseline is
preserved.** Performance and HDR conformance remain **NOT_CHARACTERIZED**.

**Signoff belongs to Codex.** This document records what was changed and what was observed locally;
it does not accept the HDR task.

---

## 1. What was wrong, and what the code actually says

### 1.1 The guidance plane

`exportPreview.ts` told the operator that **NVOF produces an RGBA8 motion tensor** that "carries
only motion, not colour" and is "unconnected to the FP16 plane". All three clauses are false.
`gpu_hdr_scrgb_bridge.cpp:765–789` runs one `proxy()` pass that writes both planes off the same
working master; `gpu_hdr_transfer_export.py:1194–1213` then hands the guidance descriptor to
`flow_provider.get_flow_device_rgba(...)` and gives the returned flow to
`live.process_device_color_flow_resource(colour_descriptor, flow, ...)`; the NVOF compute shader
declares its output as `RWTexture2D<float2> Motion` (`nvof_d3d12_interop.cpp:165`).

So the true order is: colour bridge `proxy()` → RGBA8 guidance image **and** bounded FP16 display
proxy, both derived from the HDR working master in the same dispatch → NVOF consumes the guidance
image → float2 device flow → Feature 18 consumes that flow together with the FP16 proxy. The
guidance image is a motion-estimation input derived from the master's own samples (the shader takes
luma from `master.rgb`), so it is not colour-free, and it is not unrelated to the FP16 plane. What
it is never allowed to be is the HDR main-colour transport — that is the signed, unclamped working
master. NVOF's output is motion, not RGBA8.

### 1.2 Host traffic

The preview said the image-plane staging/readback counters were **missing in this build** and that
a scalar readback counter was **present but unmeasured**. Both are false, in opposite directions.

The accepted production gate did measure the image-boundary counters. `gate_log.json` in
`bench_out/hdr_production_export/` carries seven checks named `host boundary zero: *`, each reading
0 — `decode_host_copy_bytes_measured`, `decode_host_copy_count_measured`,
`dlss_host_color_stage_bytes_measured`, `dlss_host_color_stage_count_measured`,
`dlss_output_readback_bytes_measured`, `encoder_host_pixel_bytes_measured`,
`encoder_rgb_conversion_bytes_measured` — and a check
`report structs cost no readback, and were read anyway` reading
`report_readback_bytes=0 frames_reported=6 process_count=6`. That counter is the HDR native
**diagnostic-image** counter, and it was measured.

There is, on the other hand, no scalar-byte counter anywhere in the native ABI
(`gpu_hdr_transfer_export.py:452–460`, `:1449–1470`), so "present/unmeasured" described a thing
that does not exist. Scalar diagnostics do cross GPU→host on every executed dispatch — 256 bytes
read off source constants, `kCounterSlots * sizeof(uint32_t)` — and
`hdr_scalar_report_readback_bytes_measured` is `None` because nothing weighed them. `None` is not a
zero, and `report_readback_bytes` is not a measurement of them.

The corrected text keeps the three-way distinction the audit asked for: **observed** zeros from
that prior run (explicitly not this job's measurement, and not a zero-traffic proof), an
**unanswered** aggregate (`hdr_pixel_readback_bytes_measured = None`, no image-readback export bound
in the product path), and **real** scalar traffic whose byte total is unmeasured. It promises
nothing about what a future job will measure, and it never claims a scalar counter exists.

### 1.3 The three adjacent factual errors

| Statement | Correction | Evidence |
| --- | --- | --- |
| `HDR_LIMITATIONS`: "HLG、双三层、**缩放**、色调映射一律明确拒绝" | What is refused is **output** resizing. Working scale is served at whole 25–100 % integers and resolved by the same `resolve_working_dims` the CPU plan and SDR backend use. | `gpu_hdr_transfer_export.py:786–800`; `gpu_transfer_export.py:178` (`WORKING_PERCENT_MIN, _MAX = 25, 100`) |
| "不做封装 / 本链路不执行封装" | The NVENC bridge writes **and muxes** the deliverable — one video stream into Matroska. What does not exist is a **second** copy/mux pass carrying audio, subtitles, chapters, attachments or container-level FORMAT metadata. | `gpu_hdr_transfer_export.py:51–53`, `:23–25` |
| "固定策略（…也不是 UI 控件）：纸白…" and "由此推导的合法峰值" | Paper white **is** a UI choice, fixed per job (`DLSSPage.tsx:1268`, saved through `saveHdrPatch`); the 250 support and 125 encode ceilings are **not** controls. `207.53` is 125 scRGB — ST 2084's top at this bridge's reference white, i.e. the BT.2020 linear box bound — times the largest positive row sum of the bridge's Rec.2020→Rec.709 matrix, ~1.6602. The 250 ceiling sits above the result with margin and is not an input to it. | `gpu_hdr_transfer_export.py:29–34` |

Two further wording fixes in the same family, both in `hdrExport.ts`: the `flow_engine` refusal
clause said "引导张量必须留在显存里" and the preset summary said "运动张量需要下载回内存". The
plane that would have to be downloaded is the guidance **image**; the thing produced on the device
is the flow. Now named that way.

---

## 2. Changed files

| File | Change |
| --- | --- |
| `frontend/src/services/exportPreview.ts` | `guidanceLine` rewritten (§1.1). Host traffic: one false line replaced by two lines carrying the prior-evidence / unanswered-aggregate / unmeasured-scalar split. `固定策略` block retitled `策略取值`, paper white named as a UI choice, the ceilings named as non-controls, the peak derivation stated from 125 and the matrix row sums. `输出幅面` line now refuses output resizing while naming working scaling as served. `非视频流` line now credits the bridge with writing and muxing the `.mkv`. String literals only. |
| `frontend/src/services/hdrExport.ts` | `HDR_LIMITATIONS` entries 1 and 2 rewritten (output-vs-working scaling; no *additional* copy/mux pass). Container refusal and `copy_attachments` refusal rephrased to "不追加 copy/mux pass". Preset summary container and output-extent lines corrected. `flow_engine` blocker and preset lines renamed 引导张量/运动张量 → 引导图像/光流. String literals only; no predicate, vocabulary or patch key changed. |
| `tests/node/export_preview.test.ts` | Three tests rewritten to guard the semantics instead of the old wording: `the HDR chain states the guidance plane it actually runs`, `the HDR chain states host traffic as prior evidence, an unanswered total, and unmeasured scalars` (renamed from `…as unmeasured, not as absent`), and `the HDR chain names the fixed policy the bridge is opened with`. Each now asserts the corrected flow **and** `doesNotMatch` the superseded sentences, the zero-total claims and any promise of future measurement. |
| `tests/node/hdr_export_ui.test.ts` | `the limits are stated with the switch…` extended: output resizing refused, working scaling served through `resolve_working_dims`, the bridge muxes, and `doesNotMatch` on "HLG、双三层、缩放" and "不做封装\|不执行封装". Still 4 limit entries, so the count assertion is unchanged. |
| `tests/e2e/17_hdr_ui_optin_mock.spec.ts` | One preview text assertion updated: `present/unmeasured` → `unavailable/None` + `未测量不是零`. **Not re-run** (see §4). |
| `HDR_STAGE_B_UI_OPTIN.md` | §3 `Not offered anywhere` and `The preview is honest about itself` rewritten; §7 `No muxing`, `fixed policy` and `Host traffic is not fully counted` bullets rewritten; new **§7.1 Codex audit correction** quoting each superseded sentence with its replacement and source. |
| `HDR_STAGE_B_UI_SEMANTICS_FIX.md` | This file. New. |

Untouched, as scoped: everything under `native/`, `gpu_hdr_transfer_export.py`,
`gpu_transfer_export.py`, `hdr_gpu_engine.py`, `pipeline.py`, `dlss_service.py`, the DLSS host, all
UI layout, the settings round-trip, the job payload, the selectors, the SDR preview and
`bench_out/`.

---

## 3. Verification (exact commands, exact results)

**Node tests**, existing explicit file list — directory form still fails with
`Cannot find module '…\tests\node'`, which is a known tooling limit, not a regression:

```
cd frontend
node --test ../tests/node/export_preview.test.ts ../tests/node/working_scale.test.ts \
            ../tests/node/hdr_export_ui.test.ts
→ # tests 44  # pass 44  # fail 0
```

A baseline was taken **before** editing: 44 pass / 0 fail. That green baseline is the point of this
pass — the suite was green while guarding the false statements, so passing was not evidence of
truth. Count is unchanged at 44: the three preview tests and the one limits test were rewritten in
place, not added to.

**Build**, `npm run build` (`tsc -b && vite build`) → **PASS**, no type errors, no warnings,
1877 modules transformed, `dist/assets/index-55SM7xS1.js` 372.54 kB (gzip 111.54 kB),
`dist-electron/main.js` 6.80 kB, `dist-electron/preload.cjs` 0.44 kB emitted.

Reported honestly: the build was invoked **twice**, not once. The first run emitted
`index-BDzI93UY.js` (372.51 kB); two wording corrections then followed — a ungrammatical
"上一条…闸门运行中" clause in the host-traffic line and the BT.2020 box phrasing — and the second
run emitted the delivered `index-55SM7xS1.js`. No test failure occurred in either pass; there were
no ordinary build or test errors to fix.

Bundle content checked rather than assumed, against the emitted file: `float2 设备光流`,
`unavailable/None`, `不追加 copy/mux pass`, `工作缩放 25%–100%`, `这是 UI 选择` are each present
once; `present/unmeasured`, `运动张量`, `不做封装` are each absent (0 matches).

**Not run**, per scope: the mocked Playwright smoke (its one assertion was edited to the corrected
copy and is therefore **unverified**), the CPU unit suite, and every hardware-driven gate. No export
was executed. The rebuild moved only `frontend/dist/` (gitignored): no `bench_out/` artifact or
hardware manifest hash changed. The one textual reference to a previous bundle name is this stage's
own build record at `HDR_STAGE_B_UI_OPTIN.md:153`; it was left as history and annotated rather than
rewritten. Searching `bench_out/`, `tests/`, `models/`, `dist/` and `build/` found no gate artifact
that keys on the bundle hash.

---

## 4. UI propagation vs prior backend GPU evidence — where each boundary sits

**What this correction actually reaches in the product.** All seven corrected strings are
render-only, and each reaches the operator through an existing path that was not modified:

* `exportPreview.ts` preview text → `buildExportPreview` at `DeliverPage.tsx:638` → the delivery
  preview panel. Only the `lines` array content changed; `kind`, `command: ''` and
  `copyable: false` are untouched, so the copy button stays disabled.
* `HDR_LIMITATIONS` → `DeliverPage.tsx:646` (forwarded as the preview's `limitations`),
  `DeliverPage.tsx:1747` (prepare panel list) and `DLSSPage.tsx:1301` (settings limits list). Three
  renderers, one array — which is why the entries stay 4 and the drift test still binds them.
* `hdrBlockers` → `DeliverPage.tsx:479` (refusal badges and the queue guard) and into the preview's
  `blockers`. `hdrSettingsBlockers` → `DLSSPage.tsx:408`. Rephrased clauses only; the clause set,
  their order and the `flow_engine` / container / `copy_attachments` triggers are unchanged.
* `buildHdrExportPreset` summary → `DeliverPage.tsx:481` → `ul[aria-label="实验性 HDR10 准备摘要"]`.
  The written `settingsPatch` / `transcodePatch` keys are unchanged, so the disclosure still names
  exactly what the click writes.
* Nothing crossed into the dispatch path: `composeJobOutputPath`, the payload builder,
  `experimental_hdr_scrgb`, `hdr_paper_white_scrgb`, `guidance_mode`, `working_scale`,
  `flow_engine` and every selector keep their prior values and wiring. That is the whole reason the
  mocked UI was not re-run — the interaction the smoke drives did not change.

**What the corrected preview is quoting, and how far that reaches.** The seven zeros and
`report_readback_bytes=0` in the host-traffic line are **read out of a past run** —
`bench_out/hdr_production_export/gate_log.json`, an accepted production export of a 6-frame HDR10
clip at 1920x1080. They are not measurements of the job an operator is previewing now, they say
nothing about a different clip, extent or working scale, and the preview text now says so in the
same sentence. The preview also inherits the boundary the report itself documents
(`gpu_hdr_transfer_export.py:1390–1400`): these are counters each bridge reports for the calls it
makes, **not** physical PCIe measurements, and the two `_logical` fields are buffer sizes, never
summed with measured bytes. The doc records that boundary explicitly; the UI line stays brief and
does not restate it, which is a deliberate choice under "keep product prose brief" — flagging it
here so it is not read as a claim about bus traffic.

**Still unknown after this correction.** The aggregate HDR pixel readback byte total has no
counter and stays `None`. The 256 B/dispatch scalar figure is read off source constants, not
counted, and no product path weighs it — no arithmetic estimate of it is reported as a measurement
anywhere. Whether any given job keeps its pixels on the device remains a backend decision the page
cannot pre-judge, which is what `hdrBackendFinalChecks` and the 只有后端能判定的项 list are for.
The formal Feature 18 HDR contract, image quality and performance remain unproven; the earlier
A/B, numeric and 6-frame records, including the FP16 error they recorded, stand as recorded and
were not re-taken or re-claimed here.

**Scalar diagnostics are not image readback, and nothing here claims all host traffic is absent.**
The corrected line says the opposite: scalars do cross, images did not in the measured run, and the
total nobody weighed is not a zero.

---

## 5. Signoff

Delivered as scoped: the three false preview statements and three adjacent factual errors are
corrected, the tests that guarded the old wording now guard the semantics, the doc's contradictory
host-traffic limitation is replaced with a marked Codex audit correction, and the corrected bundle
is emitted. Verification observed: node 44/44 (same count before and after), build PASS, bundle
content confirmed by direct string search.

Not claimed: that the mocked smoke passes after its assertion edit (not re-run), that any device
behaviour was re-measured (none was), or that this stage closes the HDR task. Final acceptance is
Codex's to judge.
