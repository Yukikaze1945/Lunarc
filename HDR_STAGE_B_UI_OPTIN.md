# HDR Stage B — UI opt-in (the switch, the settings path, the honest preview)

Connects the **existing** experimental HDR10 backend (`gpu_hdr_transfer_export.py`) to the
React product surface: a settings opt-in, the job payload that carries it, and an export
preview that states what the chain does and what it refuses. No native source, no DLL, no
Feature change, no math change. No benchmark. **No new hardware export was driven and no
previously passed gate was re-run** — see §6 for what the existing hardware evidence does
and does not cover.

This is the last product surface for the HDR task. **The HDR task is not finished by this
document, and this stage is not signed off here**: final acceptance is Codex's.
Performance and HDR conformance remain **NOT_CHARACTERIZED**.

Everything below was read out of this tree; each claim names the file it came from.

---

## 1. Files

Changed in this stage (the working tree is dirty from prior stages and stays dirty —
nothing was committed, stashed, reset or pushed):

| File | What changed |
| --- | --- |
| `frontend/src/types/index.ts:58,61` | `experimental_hdr_scrgb: boolean` and `hdr_paper_white_scrgb: number` added to `DLSSSettings` |
| `frontend/src/App.tsx:47,91,135` | Defaults `false` / `8.0`; the settings reply is merged **over** the defaults (`{ ...DEFAULT_SETTINGS, ...s }`) so a pre-HDR reply cannot leave the fields `undefined`; `DeliverPage` now receives `onSettingsChange` |
| `frontend/src/services/hdrExport.ts` | **New.** The client-side mirror of every gate clause a form can answer for itself, plus the one-click preset and its disclosure |
| `frontend/src/services/exportPreview.ts:24,319–405,408–416` | A third preview kind, `hdr10-transfer-chain`, with no command and no copy button |
| `frontend/src/services/api.ts:115–122` | `saveSettings` surfaces the backend's `detail` instead of `Failed to update settings` |
| `frontend/src/pages/DLSSPage.tsx:382–408,425,1003,1222,1229,1245–1305` | The switch, the paper-white field, the refusal list, the limits, and the three controls HDR pins |
| `frontend/src/pages/DeliverPage.tsx:469–492,553–598,638–649,1184,1590,1704–1752` | Branch-aware validation, the HDR preview wiring, the gated encoder options, and the prepare/apply panel |
| `dlss_service.py:445–456` | `update_settings` validates the HDR pair **before** merging, through the chain's own predicates |
| `tests/node/export_preview.test.ts` | +9 HDR preview tests (16 pre-existing kept) |
| `tests/node/hdr_export_ui.test.ts` | **New.** 16 tests on the mirror and the preset |
| `tests/unit/test_hdr_ui_optin_settings.py` | **New.** 14 CPU-only tests on the settings/job-snapshot path, plus the mirror-vs-gate drift checks |
| `tests/e2e/17_hdr_ui_optin_mock.spec.ts` | **New.** 3 Playwright tests driving the real React tree against a mocked backend |

Untouched on purpose: `gpu_hdr_transfer_export.py`, `gpu_transfer_export.py`,
`pipeline.py`, everything under `native/`, and the DLSS host.

---

## 2. Where the control is, and what it writes

**Location.** DLSS page → the panel between 导出后端 (Export Backend) and DLSS 宿主
(DLSS Host). A checkbox labelled `实验性 HDR/scRGB` with a `仅导出` badge next to it, a
free-text paper-white field, a refusal list, and the limit lines. The delivery page repeats
the state above its validation badges and owns the one-click preset.

**Default.** Off, everywhere: `DEFAULT_SETTINGS` (`App.tsx:47`), the service defaults
(`dlss_service.py:238–242`), `hdrExport.ts:21` (`HDR_PAPER_WHITE_SCRGB = 8.0`), and
`negotiate_hdr`'s own default. Nothing turns the chain on by itself: no clip load, no
preview, no render-queue refresh writes the key.

**Saved.** The checkbox posts the patch `{ experimental_hdr_scrgb: <bool> }` and nothing
else (`DLSSPage.tsx:1252` → `saveSettings`). The paper-white field posts only a value the
mirror already accepts (`DLSSPage.tsx:402–406`), so an unreadable draft never reaches
`/api/settings`. `saveHdrPatch` writes locally first and, if the backend answers 400, rolls
the local copy back and puts the backend's own sentence in the status bar
(`DLSSPage.tsx:382–395`, `api.ts:115–122`) — an operator reading the status sees which
clause refused, not that something failed.

**Snapshot.** `add_job` copies the settings blob (`job["settings"]`), so a job queued as HDR
stays an HDR job when the panel is moved afterwards, and a job queued without the flag does
not acquire one. `tests/unit/test_hdr_ui_optin_settings.py` asserts both directions through
the backend's own `hdr_requested`, including a per-job override that contradicts the live
settings.

**Pinned while on.** Three controls refuse a change that would put the job outside the
chain, naming the reason in the status line rather than writing it and letting the queue
fail: the export backend away from GPU Transfer (`DLSSPage.tsx:1229`), the reconstruction
mode away from Single (`:1003`), and the flow engine away from `nvof` while guidance is on
(`:425`). The 2- and 3-pass options are already disabled for every chain
(`DLSSPage.tsx:974–975`), so `multi_pass` can only reach a job as 1 from this UI; the mirror
still lists the `multi_pass ≠ 1` clause for a hand-written settings file, and the preset
writes `multi_pass=1`. The encoder's `p010le` and `main10` options stay disabled until the
branch is chosen (`DeliverPage.tsx:1184,1590`) — with the label saying what unlocks them.

**The one-click preset.** `准备实验性 HDR10 导出` on the delivery page opens a disclosure
that lists every choice, then `应用：…` writes them. It is explicit-click only: opening the
disclosure changes nothing, and `DeliverPage.tsx:488–492` sends the settings half through
`saveSettings` (so the backend stays the authority and can refuse) before touching the
local copy and the encoder form. The summary names the keys it writes — including the four
`custom_color_*` clears, `extra_args`, and everything it **drops**: `audio_mode=remove`,
`subtitle_mode=drop`, `copy_attachments=false`, `copy_chapters=false`, `copy_metadata=false`
— and states that ST 2086 mastering display and CTA-861.3 CLL/FALL survive in-bitstream
regardless of the metadata switch. `tests/node/hdr_export_ui.test.ts` fails if any key in
either patch is missing from that text, and fails if applying the patch leaves a form
refusal standing.

---

## 3. What the page refuses, and what only the backend decides

The split is deliberate: **the page can only refuse, never approve.** A clean list here is
not a promise; the job still goes through `negotiate_hdr`.

**Refused client-side** (`hdrExport.ts`, each one a clause of `negotiate_hdr` /
`plan_hdr_encoder`, read with the same defaults and the same trims): a flag that is neither
on nor off; a paper white that is not finite and at least 1.0 scRGB; `export_backend` not
`gpu_transfer`; `dlss_backend` not `gpu_flow_d3d12`; `reconstruction_mode` not `single`;
`multi_pass ≠ 1`; `guidance_mode` outside {0, 1}; a guided job whose `flow_engine` is not
`nvof`; a working scale outside whole 25–100 %; an odd source extent; `pix_fmt` other than
`yuv420p10le`; `bit_depth < 10`; HLG or any non-`smpte2084` transfer; `codec` other than
`hevc_nvenc`; a `profile` outside the entry's accepted spellings; a `pixel_format` outside
`auto|p010le|yuv420p10le`; a `color_preset` outside `preserve|none|bt2020_pq` (tonemap and
`hdr_to_sdr` are refusals, not operations); any `custom_color_*` that would write a second
colour clause; a destination that is not `.mkv` (checked on the composed output path, not on
the container select); non-empty `extra_args`; `rate_mode`/`aq_mode` the encoder plan does
not handle; `copy_attachments=true`; and an output extent other than the source's.

**Left to the backend, and listed as such** (`hdrBackendFinalChecks`, rendered under
只有后端能判定的项（页面无法预知，这里不作保证）): the additive decode/colour-bridge/encoder
entry points, the Feature 18 RGBA16F session, the D3D12 device and capability report; the
source's actual `color_primaries` / `color_range` / `color_space` and its mastering-display
and CLL/FALL values (`VideoInfo` carries none of these); whether the clip really contains
audio, subtitles, chapters or format metadata, which is what makes a job need the copy pass
this chain does not run; and the working-extent match against `resolve_working_dims` that
only the resolver can confirm. With no clip loaded, the page states the unknowns and raises
no source clause at all.

**Not offered anywhere.** The 250 / 125 scRGB ceilings and the derived legal peak are quoted
as fixed policy — `hdrExport.ts:23–25` and the `策略取值` block in `exportPreview.ts` say
`上限来自固定策略，绝不从 MaxCLL 推导`, and no control sets them. Paper white is **not** in
that group: `hdr_paper_white_scrgb` is an operator control (`DLSSPage.tsx:1268`, saved through
`saveHdrPatch`), fixed for the whole job rather than per frame, and `8.0` is only its default.
Tonemap, HLG, dual/triple, output resizing, and a second copy/mux pass have no HDR affordance,
because the backend has no path for them; the shared audio, subtitle, chapter, attachment and
metadata selects are kept visible and are answered with a refusal, not silently ignored.

**The preview is honest about itself.** The HDR chain is rendered as device stages with
`copyable: false` and no command string: pasting a Legacy FFmpeg command into a terminal
would start a *different* pipeline. It names the guidance path the code actually runs — the
colour bridge's `proxy()` derives the RGBA8 guidance image and the bounded FP16 proxy from the
same working master in one dispatch, NVOF reads the guidance image and answers with float2 flow
left on the device, and Feature 18 consumes that flow beside the FP16 proxy
(`gpu_hdr_scrgb_bridge.cpp:765–789`, `nvof_d3d12_interop.cpp:165`,
`gpu_hdr_transfer_export.py:1194–1213`). It separates the gate's *observed* prior image-boundary
zeros from an aggregate nobody answers and from scalar traffic that crosses but is not weighed,
says `不声明任何总量`, says the preview player is not an audited HDR display path, and closes
with `NOT_CHARACTERIZED` plus the FP16 numeric-error admission. The node test
`the HDR chain states host traffic as prior evidence, an unanswered total, and unmeasured
scalars` fails if any of that turns into a claim of zero, into a promise about a future job, or
back into the two superseded sentences recorded in §7.1.

---

## 4. Verification (exact commands, exact results)

All run on this machine, this tree, 2026-10-02.

**Build.** `cd frontend && npm run build` (`tsc -b && vite build`) → **passed**, 1877
modules transformed, `dist/assets/index-UuXe6xNa.js` 370.69 kB, `dist-electron/main.js` and
`dist-electron/preload.cjs` emitted. No type errors, no warnings.

Superseded by §7.1: the semantics correction rebuilt the bundle, so the emitted asset is now
`dist/assets/index-55SM7xS1.js` (372.54 kB, 1877 modules, build passed). The hash above is kept as
this stage's own record, not rewritten. Nothing under `bench_out/` and no hardware manifest hash
moved — the rebuild touched only `frontend/dist/`, which is gitignored.

**Node tests.**

```
cd frontend
node --test ../tests/node/export_preview.test.ts ../tests/node/working_scale.test.ts \
            ../tests/node/hdr_export_ui.test.ts
→ # tests 44  # pass 44  # fail 0
```

25 in `export_preview.test.ts` (16 pre-existing, 9 HDR added here), 3 in
`working_scale.test.ts`, 16 in the new `hdr_export_ui.test.ts`.

Failure encountered and fixed, kept as evidence: the first run of this suite after adding
`hdrExport.ts` aborted with `ERR_MODULE_NOT_FOUND: Cannot find module
'F:\dlss5standaloneV2\frontend\src\services\hdrExport'`. Node's ESM resolver needs the real
file name where the bundler takes either, so the value-level specifiers became
`./hdrExport.ts` (`exportPreview.ts:8`) and `./workingScale.ts` (`hdrExport.ts:4`); type-only
imports stay extensionless. Legal under `allowImportingTsExtensions` in
`frontend/tsconfig.app.json`, and `npm run build` still passes with them.

Ordinary tooling limit, not fixed: `node --test ../tests/node/` (directory form) fails here
with `Error: Cannot find module 'F:\dlss5standaloneV2\tests\node'`. The explicit file list
above is the working invocation.

**Python, CPU-only.**

```
python -m pytest tests/unit/test_hdr_ui_optin_settings.py \
                 tests/unit/test_export_backend_ui_contract.py \
                 tests/unit/test_hdr_transfer_export_branch.py -q
→ 125 passed in 10.06s
```

14 new + 111 pre-existing. The new file asserts behaviour, not strings: the opt-in reaches
the snapshot as read back through `htx.hdr_requested`; a patch carrying one good key and one
unusable paper white stores **neither**; the refusal an operator sees is verbatim
`htx.paper_white_reason(...)`; the HTTP boundary answers 400 with that detail; an HDR job on
`legacy` and an HDR job on `dual` are both refused at `add_job` with the switch named in the
message and never enter the queue. It also fails if the client mirror drifts:
`test_the_front_end_mirror_carries_the_backends_vocabulary` parses the literals out of
`hdrExport.ts` and compares them to `htx.HDR_ALLOWED_*`, `HLG_TRANSFERS`, `NITS_PER_SCRGB`,
`DERIVED_LEGAL_PEAK_SCRGB`, and the codec/profile/container names; the rate/AQ modes are
checked against the actual `plan_hdr_encoder` ladder rather than a restated list; and the
working-scale range is imported rather than copied.

**Mocked UI smoke.** `npx playwright test 17_hdr_ui_optin_mock --reporter=list` →
**3 passed (3.3 s)**, headless chromium loading `frontend/dist` from a throwaway static
server with `**/api/**` intercepted. No Electron, no Python backend, no GPU, no FFmpeg, no
deliverable written. What it proves is propagation through the rendered tree:

1. `the checkbox, the preset and the queue payload are one chain` — the checkbox's POST is
   exactly `{ experimental_hdr_scrgb: true }`; the panel then lists `export_backend=auto` as
   a refusal; moving the chain clears it; the delivery preview is the HDR chain and its copy
   button is disabled; with the shipped av1 form the queue button **alerts and posts
   nothing** (`jobs == []`); after reviewing and applying the preset, `/api/queue/add`
   carries `experimental_hdr_scrgb=true`, `export_backend=gpu_transfer`,
   `reconstruction_mode=single`, `hdr_paper_white_scrgb=8`, `codec=hevc_nvenc`,
   `profile=main10`, `pixel_format=p010le`, `container=.mkv`, `audio_mode=remove`,
   `subtitle_mode=drop`, `copy_attachments=false`, cleared `custom_color_*` and
   `extra_args`, `resolution_mode=original`, and `output_path` `D:\Media\clip_hdr10_out.mkv`.
2. `turning the switch off leaves the SDR job unpoliced…` — after a reload the checkbox
   reads back the stored value; unchecking posts `false`; the preview falls back to the SDR
   device chain (its `-c:v copy` line present, no `scRGB`), the HDR refusal list empties,
   and the next job carries `experimental_hdr_scrgb=false` with the operator's av1 encoder
   and destination untouched.
3. `a paper white the gate would refuse never reaches the service` — typing `abc` shows the
   refusal and posts **nothing** (the stored value stays 8.0); typing `20` posts 20 and both
   pages quote `20 scRGB = 1600 cd/m²`.

Two real failures were hit and fixed on the way, preserved here rather than hidden: the spec
first died on `ReferenceError: __dirname is not defined in ES module scope` (these specs are
ESM; it now imports `ROOT_DIR` from `tests/helpers/paths.js`), then timed out waiting for
`input[placeholder="例如 D:\Output"]` because a single backslash inside a CSS attribute
string is parsed as an escape — the page's own derived output directory is used instead,
which is the more honest assertion anyway. Third environment fix: the root Playwright 1.63.0
wanted `chromium_headless_shell-1243` while the machine had only `-1228`;
`npx playwright install chromium` fetched it (114.6 MiB, into the user's ms-playwright
cache — not a repository change).

**Not run, on purpose.** The full unit suite, the old GPU gates, the A/B, numeric and
6-frame hardware checks, any benchmark, and any packaging/export validation on device. The
mocked smoke is the only UI validation this stage performed, and it deliberately cannot open
a session.

---

## 5. What changed in behaviour, stated narrowly

* A user can now request the HDR chain from the window. Before this stage the only way to
  set `experimental_hdr_scrgb` was to edit a settings file.
* `update_settings` gained one guard. A write that touches neither HDR key behaves exactly
  as before, so no existing caller, preset or script changes route.
* The delivery page's pixel-format and profile selects gained a disabled state that only
  this switch lifts; everything else in the encoder form is unchanged for an SDR job.
* Nothing moved on the preview path: the switch never changes which host draws a frame, and
  `_preview_live_kind()` is untouched. An HDR job still renders nothing until the queue runs
  it, and the queue's own admission is the pre-existing `negotiate_hdr`.

---

## 6. The existing hardware evidence is prior-code evidence

`HDR_STAGE_B_PRODUCTION_EXPORT.md`, `HDR_STAGE_B_REAL_ADMISSION.md`,
`HDR_FP16_FEATURE18_AUDIT.md`, `HDR_PROXY_AB_STAGE_A.md` and the rest of the stage-B series
describe runs made **before** this UI wiring, driven by hand-built job payloads. They are
evidence about the backend as it then was. They are **not** a validation of this stage: no
export driven from the React tree has ever been run on a device, and none was run here.
Nothing in that series was re-run, rewritten or re-claimed by this document, and the losses
they recorded (A/B, numeric, 6-frame) stand as recorded.

---

## 7. Limitations

* **HDR10 single-layer only.** ST 2084 / PQ, 10-bit 4:2:0 BT.2020 NCL sources, one working
  extent, one pass, original full output extent, HEVC Main 10 / P010 into Matroska.
* **HLG is not served**, and no other transfer, no 8-bit, no high-bit-depth-scalar, no
  4:2:2 / 4:4:4 source. Refused by name on both sides, never converted.
* **Multi-layer is not served here.** Dual and Triple remain GPU-Transfer-only SDR features;
  an HDR job naming them is refused at `add_job`, not run as Single.
* **No second copy/mux pass, so no audio, subtitles, chapters, attachments or container-level
  FORMAT metadata** on an HDR job, and no `extra_args`. The NVENC bridge does author and mux the
  deliverable itself — one video stream into Matroska (`gpu_hdr_transfer_export.py:51–53`) — it
  simply runs no additional pass that would carry the other streams across. Whether the source
  actually holds them is a backend judgement; the page states it as pending rather than as clean.
  A non-`.mkv` deliverable cannot be produced at all.
* **The preview and player are not an audited HDR display path.** The switch is labelled
  仅导出, and the limits say so in the panel, the preview and the docs.
* **The formal Feature 18 HDR contract, quality and performance are unproven.** The ceilings
  (250 / 125 scRGB) and the derived legal peak are fixed policy with no control behind them;
  paper white is a per-job UI choice that defaults to 8.0 scRGB, not a policy constant. There is
  no HDR conformance certification, no claim of losslessness, no claim of any speed or
  quality gain, and **an FP16 numeric error exists in the proxy/composite path** — the
  earlier A/B and numeric records show where it loses.
* **Host traffic is not fully counted, and its total is not answerable.** See §7.1. The accepted
  production gate measured seven image-class host-boundary counters at 0 and measured the colour
  bridge's diagnostic-image counter `report_readback_bytes` at 0; that is a past run's evidence,
  not a measurement of any later job. The complete HDR pixel readback byte total has no counter
  and stays unanswered, and the scalar diagnostics that do cross on every dispatch are weighed by
  no ABI counter. No statement in this UI claims a zero total, and none promises what a future
  job will measure.
* The client mirror is a *refusal* mirror, not a gate. If `negotiate_hdr` widens, the page
  can lag behind it; the drift tests bind the vocabularies but cannot bind capability,
  session or source-probe facts.
* The mocked smoke intercepts the API and cannot exercise WebSocket paths (`/ws/player`,
  `/ws/queue`); those connections fail in the smoke and the app's reconnect loop is out of
  scope for it.

### 7.1 Codex audit correction (2026-10-02)

Codex's audit of the finished UI job found three statements in **this stage's own prose** — not
in the backend, not in the gate artifacts — that misdescribed the evidence. They are quoted here
and marked superseded. The raw gate artifacts under `bench_out/` were not touched, re-run or
rewritten.

| Superseded statement (this document, and the preview text it described) | What the evidence says | Source |
| --- | --- | --- |
| "image-plane staging/readback counters are missing in this build" | The accepted production gate **did** measure them. Seven image-class host-boundary counters read 0: `decode_host_copy_bytes_measured`, `decode_host_copy_count_measured`, `dlss_host_color_stage_bytes_measured`, `dlss_host_color_stage_count_measured`, `dlss_output_readback_bytes_measured`, `encoder_host_pixel_bytes_measured`, `encoder_rgb_conversion_bytes_measured`. The colour bridge's diagnostic-image counter `report_readback_bytes` was measured 0 as well (`frames_reported=6`, `process_count=6`). | `bench_out/hdr_production_export/gate_log.json` — checks `host boundary zero: *` and `report structs cost no readback, and were read anyway` |
| "the scalar diagnostic readback counter exists but was never measured (`present/unmeasured`)" | There is **no scalar-byte counter in the native ABI** for that to describe. Scalar diagnostics do cross GPU→host on every executed dispatch — 256 bytes, read off the source constants (`kCounterSlots * sizeof(uint32_t)`) — and `hdr_scalar_report_readback_bytes_measured` is therefore `None`: unweighed, which is not zero. `report_readback_bytes` counts image planes and is not a measurement of them. | `gpu_hdr_transfer_export.py:452–460`, `:1449–1470`; `native/gpu_hdr_scrgb_bridge.cpp` `run_pass` counter copy |
| NVOF "produces an RGBA8 motion tensor in VRAM that carries only motion and is unconnected to the FP16 plane" | `bridge.proxy()` derives **both** the bounded FP16 display proxy and the RGBA8 guidance image from the same working master in one dispatch. NVOF reads the guidance image and produces float2 device flow, which Feature 18 consumes together with the FP16 proxy. The guidance image is a motion-estimation input — never the HDR main-colour transport — and NVOF's output is motion, not RGBA8. | `native/gpu_hdr_scrgb_bridge.cpp:765–789`, `native/nvof_d3d12_interop.cpp:165`, `gpu_hdr_transfer_export.py:1194–1213` |

Two claims that used to sit beside these stay: the aggregate HDR pixel readback byte total is
unanswered, not zero (`hdr_pixel_readback_bytes_measured = None`, and
`pixel_readback_exports_bound = []`), and no image-readback entry point is bound in the product
path. What was wrong was calling measured counters missing, and calling an absent counter present.

The same audit corrected three further factual statements in the shipped prose. `HDR_LIMITATIONS`
refused "缩放" wholesale: what the gate refuses is **output** resizing, while working scale is
served at whole 25–100 % through the same `resolve_working_dims` the CPU path uses
(`gpu_hdr_transfer_export.py:788–800`). The limits said the chain "does no muxing": the NVENC
bridge writes and muxes the single-video Matroska deliverable (`gpu_hdr_transfer_export.py:51–53`);
the pass that does not exist is a **second** copy/mux pass for the other streams. And the preview
grouped paper white with the ceilings as "not a UI control" while deriving `207.53` "from these":
paper white is a per-job UI choice (`DLSSPage.tsx:1268`), 250 and 125 are not controls, and
207.53 scRGB is 125 scRGB — a valid BT.2020 linear box — times the largest positive row sum of
this bridge's Rec.2020→Rec.709 matrix, ~1.6602. The support ceiling sits above that result with
margin and takes no part in deriving it (`gpu_hdr_transfer_export.py:29–34`).

The node tests were rewritten to guard the corrected semantics instead of the old wording. The
mocked Playwright smoke asserts the corrected preview text and **was not re-run** for this
correction (controls, selectors and payload are untouched); it is unverified as of this section.

---

## 8. Signoff

Work complete as scoped: the existing HDR10 backend is reachable from the UI, its settings
round-trip and job snapshot are tested on CPU, its preview states the chain honestly, and
one mocked UI smoke proves the click-to-queue path carries the request. Not declared
complete as a task: the HDR task's acceptance, including any device validation driven from
this UI, remains **Codex's to judge**.
