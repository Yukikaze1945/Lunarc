# Stage A: bounded-proxy A/B feasibility gate (2026-09-29, revision A2)

Scope: one isolated numeric measurement against NGX Feature 18 in a diagnostic
probe build. This is **not** an HDR pipeline, **not** a production colour path,
and it enables nothing. Production GPU Transfer still rejects P010/HDR, and
`native/gpu_hdr_scrgb_bridge.cpp` is unfinished and unbuilt; nothing in this
document should be read as evidence that it works. Performance is
`NOT_CHARACTERIZED` — no benchmark was run.

Revision **`A2_corrected_slope_and_saturation`** supersedes the first stage-A
record. Codex audited that record and found a correctness defect in the
reconstruction math; fixing it exposed two more. This document reports the
corrected run. Every number below comes from
`build/fp16_probe/hdr_proxy_ab_artifact.json` (revision A2) unless it is
explicitly labelled as the stale value it is being contrasted against.

Reference shape is the architectural hypothesis audited from OpenDLSS-NR at
`9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1` (`nr_common.glsl`,
`nr_preprocess.comp`, `nr_composite.comp`). The proxy, the reconstruction, the
oracle and the probe here are independent implementations written for this
repository; no Magpie GPL code was read into or copied into these files.

## What the audit found, and what the fix changed

Three defects, all in `native/hdr_proxy_math.py`, all on the derived-math side.
None of them touched a GPU measurement, and none of them could be caught by the
identity gate that was supposed to be the guard.

1. **`code_slope_per_stop` divided by `paperWhite` twice.** It computed
   `base = _relative(scene, paper_white)` and then called
   `proxy_code(base, paper_white)`, which normalizes again. The measured slope
   was therefore the slope of a scene `paperWhite` times *darker* than the one
   asked for — at `paperWhite` 4.0 the slope of the shadow region, not the
   highlight. The perturbation is now applied to the original scene value:
   `proxy_code(scene * 2**step, paper_white)` and
   `proxy_code(scene / 2**step, paper_white)`. This corrupted every slope,
   signed stop, `cap`/`rejected` disposition and reconstructed HDR value at
   `paperWhite` 4.0 and 8.0. `paperWhite = 1.0` was bug-inert (dividing by 1
   twice is still dividing by 1), which is exactly why the previous run looked
   self-consistent: the session whose numbers were right was the one that was
   failing the gate anyway.
2. **`saturation_relative` solved the wrong equation.** It found where the
   shoulder's *linear* value sits half a grid step from 1.0, but rounding
   happens after the sRGB encode, where the grid is far coarser near black and
   finer near white. The correct boundary is the half value below 1.0,
   `1 − 2⁻¹¹`, and its RNE midpoint `1 − 2⁻¹² = 0.999755859375`, decoded back
   through sRGB: the linear ceiling is `1 − ((0.999755859375 + 0.055)/1.055)^2.4`,
   giving **1.808731372940256** relative whites instead of the stale 1.9511325688380865.
   The stale figure was ~7.9% too generous — it would have called a real and
   already unrecoverable part of the scene usable.
3. **A channel pinned at the ceiling reported a nonzero slope.** Discovered
   while writing the oracle test, not in the audit. The central difference at
   relative 1.81 read `0.015625`: the upper arm is pinned at code 1.0 and the
   lower arm falls off the cliff, so the difference measures the cliff face,
   not a signal — precisely the reading the plateau guard exists to refuse.
   `code_slope_per_stop` now reports exactly 0.0 wherever the base code is 1.0,
   and the independent oracle derives the same rule on its own. No recorded row
   in this run lands in `[saturation, saturation · 2^step]`, so this changes no
   number here; it closes a latent hole.

**The claimed viability at `paperWhite` 4.0 and 8.0 survives the fix.** Most
applied corrections get smaller, one row gets larger, and the gate still passes
at both constants while still failing at 1.0 — from the corrected run, not from
the stale record. Details in *Reconstruction* and *Decision*.

## Provenance

Exact commands, run from `F:\dlss5standaloneV2` in Git Bash (forward slashes —
`python native\hdr_proxy_math.py` does not survive this shell's escaping):

```
python -m pytest tests/unit/test_hdr_proxy_slope_oracle.py -q
python native/hdr_proxy_math.py > scratch/hdr_proxy_math_selfcheck_v2.json
python native/dlss_gpu_hdr_proxy_ab.py > scratch/hdr_proxy_ab_stdout_v2.json 2> scratch/hdr_proxy_ab_v2.err
python native/hdr_proxy_ab_report.py > scratch/hdr_proxy_ab_report_v2.txt
python scratch/hdr_proxy_determinism_check.py
python scratch/hdr_proxy_leaf_audit.py
python scratch/hdr_proxy_tables.py
```

`cmd /c native\build_dlss_gpu_fp16_probe.bat` was **not** re-run. The condition
for rebuilding was any change to native probe source since the recorded build,
and the check says there was none: `native/dlss_gpu_host.cpp` hashes
`b0862a9a…`, `native/dlss_gpu_host.h` `5f216607…`,
`native/nvof_d3d12_interop.h` `62570275…`, the build script `6f9295be…`, and the
DLL on disk hashes `d934dad0…` — all byte-identical to what the artifact records
for the build that produced the first stage-A measurement. The artifact states
this rather than asserting it: `probe_rebuilt_for_this_revision: false`,
`probe_unchanged_since_recorded_build: true`, plus the whole
`probe_source_set` map. Zero native bytes changed; only Python math changed.

| Item | SHA-256 |
|---|---|
| `build/fp16_probe/hdr_proxy_ab_artifact.json` (680,557 B, this record) | `9b644920c8cc8637f14d4318732b3eac9aae08c73206f72116139b5b703e7563` |
| `native/hdr_proxy_math.py` (corrected) | `fd15871d9efa79cb530b91914829d264d2557b70509dc36d13f516460ccfb46c` |
| `native/hdr_proxy_slope_oracle.py` (new) | `751a22dfa5268aefe8e251304675126fadf0c29afcd390285bc8b2ad14aeaa89` |
| `tests/unit/test_hdr_proxy_slope_oracle.py` (new, 13 tests) | `c63c0cc3bc030cbeb0fbeb9ca2dffe06403dda88a5f80ff6fc08b010d0a801e8` |
| `native/dlss_gpu_hdr_proxy_ab.py` (gate driver) | `0a40549102fb6532d8f5b4f9ad1e569fd2b0533d19d44132a560897ad6faa9c5` |
| `native/hdr_proxy_ab_report.py` (unchanged) | `29c296362e59c46b793afbddcecb384950880bf1ea8292d88b14d27fbf89a871` |
| `scratch/hdr_proxy_ab_stdout_v2.json` (7,484 B) | `4700f13abf6baeacd9a79c7a90fa9569735d613f752a2eb89ef736b2363de9f6` |
| `scratch/hdr_proxy_ab_v2.err` (0 B — stderr empty) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `scratch/hdr_proxy_math_selfcheck_v2.json` (4,849 B) | `05d42c3815a03c4e2112588b9b6ab140c76b25232582d848d38037b21ea31e93` |
| `scratch/hdr_proxy_ab_report_v2.txt` (11,423 B) | `15cf83ad9ac59e341771e712f75f50084810372c19df27e3e786c7044237fb2e` |
| `build/fp16_probe/dlss_gpu_fp16_probe.dll` (unchanged) | `d934dad00f6425d5bf7485cff86f667b7383cc85157a404b260973eca8fe92b2` |
| `native/dlss_gpu_host.cpp` (unchanged) | `b0862a9a9828361d479b1b2acee48f918fec32cda4fca87cd2b9eac7f09f6fa6` |

Recorded run time `recorded_utc = 2026-09-29T13:14:29+00:00` (21:14 local). One
GPU gate run for this revision, exit code 0, stderr empty.

### Preserved history

The superseded first stage-A record is kept byte-identical, and the new artifact
carries the pointer so the lineage travels with the data rather than with this
file:

| Path | SHA-256 |
|---|---|
| `build/fp16_probe/hdr_proxy_ab_artifact_run4_stale_slope.json` (575,876 B) | `c93bf1d661c290fe977727e9b3b40fa2596a66eab500f754e67ed0f7be0ba40c` |
| `scratch/hdr_proxy_ab_stdout_run4_stale_slope.json` | `f8b6cae945311daf907e17697fd97150d3ed2f5b231834d6453af4fb55bde362` |
| `scratch/hdr_proxy_math_selfcheck_run4_stale.json` | `5f03c69f36763bcb6b373fe1f181bc925558efb2b2d2e7622a5be2299d990805` |

Those three are copies of the 21:05 artifacts, verified with `cmp`. The older
superseded runs (`…_run1_metric_bug.json`, `…_run2.json`,
`…_run3_pre_guard_fix.json`, and the 20:00/20:16 stdout logs) remain in place
untouched. Nothing was overwritten without a marker.

### Determinism, measured against the stale record

Comparing the two artifacts leaf by leaf: 12,117 leaves in the new record, 10,299
in the old, 10,296 shared paths, 715 of those differing. Categorised:

- **694** under `/sessions/*/arms/B_bounded_proxy/…/reconstruction/…`
- **18** saturation/grid leaves (`relative_saturation_point`,
  `scene_saturation_point`, `scene_levels_above_saturation`,
  `highlight_bits_lost`, `proxy_saturation_scene`)
- **2** gate aggregates: `reconstruction_max_hdr` at 4.0 (4.2227 → 4.3203) and
  at 8.0 (4.2070 → 4.0859)
- **1** `proxy_module_sha256` (the fix itself)
- 1,821 paths exist only in the new record — the fields the audit added:
  `deviation_from_master`, `delta_code`, `slope_per_stop`, `ratio`,
  `reconstruction_stats`, `slope_oracle_max_abs_deviation`, `corrected_defects`,
  `numeric_self_checks`, the dead-zone grid keys and the provenance block.

Every GPU-measurement leaf is identical: **1,782 measured values compared, 0
differing**, including each `session_id`, every `pre_ngx_half` readback, every
`output_center` statistic, every frame and every arm. `probe_dll_sha256` and
`host_source_sha256` match too. So the difference between the two records is
entirely arithmetic performed after the GPU stopped, which is what a math fix
should produce and is the check that says this run did not silently move the
measurement to make the story work.

Export signatures were re-verified against the ctypes declarations written into
the artifact (`api_signatures`): `dlss_gpu_host_abi_version` (→ `c_ulong`),
`dlss_gpu_host_open` (`c_ulong, c_ulong, c_wchar_p, LP_Options, LP_c_void_p` →
`c_long`), `dlss_gpu_host_session_id`, `dlss_gpu_host_close`,
`dlss_gpu_host_last_error`, `dlss_gpu_host_process_zero_motion`, and the
probe-only `dlss_gpu_host_probe_readback_input` (`c_void_p, c_void_p,
c_ulonglong` → `c_long`).

## Rig

640x360, `DXGI_FORMAT_R16G16B16A16_FLOAT` color and output in the probe build
only. Zero-motion guidance (resident texture zeroed at open, untouched per
frame). Settings fixed for every arm and sample (`settings` in the artifact):
preset 0, style 0, intensity 1.0, local_tone 1.0, local_structure 1.0,
skin_structure 0.0, auto_mask 0, ui_correction 0, motion scale (1,1), generic
DLSS exposure keys (1.0, 1.0). Per arm per sample: one `Reset=1` call plus three
successive frames, so `frame_0` is the reset frame.

Arm A submits the raw linear scRGB FP16 value. Arm B submits the bounded proxy
code of that value and keeps the pristine linear master outside Feature 18,
carried alongside for reconstruction. `paperWhite` is a session constant set at
open; three separately opened sessions carry 1.0, 4.0 and 8.0. Nothing adapts
per frame (`per_frame_adaptation: false`). Reported `session_id` is 1 for each
session because the counter lives on the shared NGX core, which is torn down
between sessions — the `paperWhite` key, not the id, distinguishes them.

In every recorded frame `pre_ngx_exact` is true: the FP16 values read back from
the color texture before NGX evaluation equal the submitted half values
bit-for-bit, for both arms, at every level including 4.0. The offline self-check
reproduces those readback codes exactly — at `paperWhite` 1.0 the scalars
0.1/0.5/1.0/1.5/2.0/4.0 predict 0.34912/0.73535/0.97363/0.99854/1.00000/1.00000
and the texture held those same numbers — so the proxy is computed once, in one
place, and staging is not silently changing it.

## The proxy, and what it costs

`x = v / paperWhite`; identity below `KNEE = 0.75`; above it an exponential
shoulder of span 0.25 and knee steepness 5.77078 (`4·log₂e`) that never reaches
1.0 in exact arithmetic; then sRGB encode and round to the half grid.
Constants as recorded in the artifact's `proxy` block: `slope_stop_step`
1/64 = 0.015625, `min_slope_per_stop` = `half_ulp_at_one` = 2⁻¹¹ = 4.8828125e-4,
`max_stops_reconstruction` 1.0, `paper_white_floor` 0.05,
`encoded_rounding_boundary` 0.999755859375.

Rounding, not the shoulder, is where magnitude dies. Two half codes adjacent to
1.0 differ by 2⁻¹¹, so once the *encoded* value reaches the midpoint
`1 − 2⁻¹²` — RNE ties to even, and 1.0 is the even neighbour — every further
increase in `x` lands on the identical code 1.0.
Exhaustive enumeration of the positive finite half grid (31,743 levels) through
the proxy, corrected:

| paperWhite | scene saturation point | scene levels above it | distinct codes, whole domain | codes knee→saturation | codes used above saturation | highlight bits lost |
|---|---|---|---|---|---|---|
| 1.0 | 1.8087 | 15,555 | 10,666 | 244 | 1 | 13.9251 |
| 4.0 | 7.2349 | 13,507 | 12,414 | 244 | 1 | 13.7214 |
| 8.0 | 14.4699 | 12,483 | 13,288 | 244 | 1 | 13.6077 |

The stale row read 1.9511 / 7.8045 / 15.6091, 15,410 / 13,362 / 12,338 levels,
13.9116 / 13.7058 / 13.5908 bits, and "245 distinct codes between knee and
saturation". The 245 was not a different count of useful codes: the old key
swept everything above the knee and so included the pinned code 1.0 itself. The
surviving useful budget is **244**, identical at all three constants because the
proxy is relative. Above the saturation point the proxy is a hard constant, not
a soft one: ~13.6 to ~13.9 bits of highlight magnitude destroyed per channel,
and no algorithm downstream can get them back.

Verified in both directions against enumeration, per session, in the artifact's
`grid`: `levels_at_or_above_saturation_not_pinned = 0`,
`levels_below_saturation_pinned_to_one = 0`,
`codes_used_by_scene_above_saturation = 1.0`, and
`first_grid_level_pinned_to_one` = 1.8095703125 / 7.23828125 / 14.4765625 — the
smallest half level at or above the analytic threshold, i.e. the enumeration and
the closed form agree at the exact level where the collapse starts.
`relative_saturation_point` from the module and from the independent oracle are
the same float, 1.808731372940256.

### The quantized dead zone (new finding)

Saturation is not the only place the code stops moving. Measured on the grid
with the same 1/64-stop probe the reconstruction uses,
`levels_knee_to_saturation_dead = 255` of `levels_knee_to_saturation_tested =
1341` half levels between the knee and the saturation point report **zero** slope
— the ±1/64-stop perturbation lands on the same code at both ends. The dead zone
starts at `first_dead_level_above_knee_relative = 1.4091796875` and runs to
`last_dead_level_below_saturation_relative = 1.7890625`, and it is invariant
across `paperWhite`, exactly like the saturation point.

So a fixed `paperWhite` does not buy a smooth usable ramp up to saturation. It
buys a measurable ramp up to 1.4092 relative whites, dead ground from 1.4092 to
1.7891, and a 0.0196-wide sliver between there and the saturation point that
still carries a slope. In scene units:

| paperWhite | identity below knee | measurable correction up to | code present but dead | code advances again until | pinned at 1.0 from |
|---|---|---|---|---|---|
| 1.0 | 0.75 | 1.4092 | 1.4092 – 1.7891 | 1.8087 | 1.8087 |
| 4.0 | 3.0 | 5.6367 | 5.6367 – 7.1563 | 7.2349 | 7.2349 |
| 8.0 | 6.0 | 11.2734 | 11.2734 – 14.3125 | 14.4699 | 14.4699 |

The plateau guard already refuses to divide by a zero slope, so this costs
nothing in correctness — it is a real limit on what the reconstruction can act
on, and the previous record did not know it was there. It bites two recorded
samples directly: `scalar_1.5` at `paperWhite = 1.0` has its own code, 0.99854,
is not pinned at 1.0, and measures slope 0.00000 on all three channels in all
four frames, because relative 1.5 sits inside the dead zone; and the blue channel
of `color_sub_and_super` at the same constant is the same relative 1.5 in a
colour patch.

`reconstruction_stats.rows_with_a_pinned_channel` is named loosely: it counts
REC_LIN rows with at least one *zero-slope* channel, which is pinned-or-dead-zone.
At `paperWhite = 1.0` it reads 32 of the session's 44 REC_LIN rows. Read the key
as "rows where at least one channel cannot carry a correction".

## Verification of the math

`native/hdr_proxy_slope_oracle.py` is a second implementation of the proxy and
of the finite difference, written against the specification rather than against
the module: float64 throughout, its own half-rounding (`np.frombuffer`/`tobytes`
via a scalar helper), its own knee, its own shoulder, its own sRGB transfer, and
its own independently derived "code at ceiling ⇒ slope 0" rule. It imports
numpy and math and nothing from `hdr_proxy_math`. The comparison lives in the
gate driver, so the artifact carries the result rather than a claim:

| check | recorded value |
|---|---|
| `slope_max_abs_deviation_vs_oracle`, 31,743 half levels × 3 sessions | `0.0` |
| `slope_levels_disagreeing_with_oracle` | `0` |
| `code_max_abs_deviation_vs_oracle` | `6.103515625e-05` |
| `code_levels_disagreeing_with_oracle` | `1` per session |
| `slope_oracle_max_abs_deviation` on the recorded GPU rows | `0.0` |
| `code_slope_matches_independent_oracle`, all three sessions | `true` |
| `saturation_relative` module vs oracle | identical float |

That single disagreeing code level is honest reporting, not a rounding fudge: it
is one relative point reached at each session — scene 0.0061836243 at pw 1.0,
0.024734497 at pw 4.0, 0.049468994 at pw 8.0, i.e. relative 0.0061836 every
time — where the float32 module lands on 0.0717163086 and the float64 oracle on
the immediately adjacent half code 0.0717773438, one grid step apart at a code
value whose step is 2⁻¹⁴. It is a tie case: the scene value sits within float32
rounding of the exact midpoint of two codes, and the two precisions resolve it
differently. It sits at relative 0.00618, some 290× below the highlight values
this gate is about, and it does not perturb any slope: `slope_levels_disagreeing_with_oracle = 0` at all three
`paperWhite` values.

The gate driver also enforces the property the audited bug violated — slope is a
function of the *relative* value, so `slope(paperWhite · x, paperWhite)` must
equal `slope(x, 1.0)` across the three session constants; `pre_ngx` readbacks of
the same relative scene at different `paperWhite` are not otherwise comparable.
The 13 unit tests in `tests/unit/test_hdr_proxy_slope_oracle.py` pass
(`13 passed in 0.55s`) and are non-vacuous: run against the pre-audit code, 49
of the 80 tested (paperWhite, scene) combinations disagreed with the oracle, and
the normalization-twice regression test
(`test_slope_is_invariant_under_paper_white_scaling_of_the_scene`) fails.
`test_identity_gate_cannot_catch_a_wrong_slope` pins the structural blindness
that let this ship in the first place: with a channel pinned at the ceiling and
a nonzero delta, the identity gate stays green while the correction is refused —
the identity gate validates the operator's fixed point, not its slope.

## Arm A: raw linear input

Confirmed against the earlier probe, with the measurements unchanged in this
run: raw >1 clamps inside Feature 18. Levels 1.5, 2.0 and 4.0 each produce
`output_full_min = output_full_max = 1.0` — a frame pinned to a single value,
across-frame band exactly 0.0. All three collapse into one cluster
(`raw_highlight_clusters = 1`, `raw_highlight_levels_clipped_at_ceiling = true`)
at all three session constants. Every colour patch in arm A also has
`output_full_max = 1.0`. Arm A has no highlight information to reconstruct.

## Arm B: bounded proxy

Steady central median luma per sample (`A` = raw arm, `B` = proxy arm), with the
proxy code actually submitted. These are GPU measurements and are byte-identical
to the first stage-A run.

paperWhite 1.0 — highlight levels collide, and one of them was colliding for a
second, unreported reason:

| sample | master | A median | B code | B median | note |
|---|---|---|---|---|---|
| scalar_1.5 | 1.5 | 1.00000 | 0.99854 | 0.99791 | own code, **zero measured slope: dead zone** |
| scalar_2.0 | 2.0 | 1.00000 | 1.00000 | 0.99854 | **identical code triple to 4.0** |
| scalar_4.0 | 4.0 | 1.00000 | 1.00000 | 0.99854 | **identical code triple to 2.0** |
| color_blue_above | 0.2 0.8 2 | 0.66919 | 0.48462 0.91260 1.00000 | 0.83851 | B at ceiling |
| color_green_extreme | 0.5 4 1.5 | 0.79650 | 0.73535 1.00000 0.99854 | 0.94476 | G pinned, B dead zone |
| color_mixed_above | 1.5 2 0.5 | 0.96156 | 0.99854 1.00000 0.73535 | 0.98086 | R dead zone, G pinned |
| color_red_highlight | 4 0.5 0.25 | 0.26534 | 1.00000 0.73535 0.53711 | 0.78056 | R pinned |

Six of the eleven samples drive at least one channel onto code 1.0, and only 10
distinct code triples exist across 11 samples: `scalar_2.0` and `scalar_4.0` are
byte-identical going in, so Feature 18 cannot be told which is which.
`proxy_highlight_distinct_input_codes = 2` while `proxy_highlight_clusters = 1` —
1.5 enters with its own code, yet its output median (0.99791) sits inside the
8.4e-4 across-frame band of 2.0's and 4.0's (0.99854), so the three highlight
levels are not separable in the output once temporal noise is allowed for.
Clustering is measured, not asserted: two levels merge unless their gap exceeds
the larger of their two bands. `proxy_highlight_input_codes_collide = true`, and
`proxy_highlight_not_single_clipped_value = false`. Eight of the eleven samples
now show at least one channel that cannot carry a correction at all — the six
pinned ones plus `scalar_1.5` and the blue channel of `color_sub_and_super`,
which are dead-zone cases rather than ceiling cases.

paperWhite 4.0 and 8.0 — highlights separated:

| sample | master | A median | B code (pw 4) | B median (pw 4) | B code (pw 8) | B median (pw 8) |
|---|---|---|---|---|---|---|
| scalar_0.1 | 0.1 | 0.10107 | 0.17188 | 0.17319 | 0.11493 | 0.11614 |
| scalar_0.5 | 0.5 | 0.50412 | 0.38867 | 0.39223 | 0.27734 | 0.27902 |
| scalar_1.0 | 1.0 | 0.99854 | 0.53711 | 0.54070 | 0.38867 | 0.39223 |
| scalar_1.5 | 1.5 | 1.00000 | 0.64600 | 0.64865 | 0.47021 | 0.47409 |
| scalar_2.0 | 2.0 | 1.00000 | 0.73535 | 0.73699 | 0.53711 | 0.54070 |
| scalar_4.0 | 4.0 | 1.00000 | 0.97363 | 0.97256 | 0.73535 | 0.73699 |
| color_red_highlight | 4 0.5 0.25 | 0.26534 | 0.97363 0.38867 0.27734 | 0.51041 | 0.73535 0.27734 0.19397 | 0.36759 |
| color_green_extreme | 0.5 4 1.5 | 0.79650 | 0.38867 0.97363 0.64600 | 0.83294 | 0.27734 0.73535 0.47021 | 0.62335 |
| color_blue_above | 0.2 0.8 2 | 0.66919 | 0.24780 0.48462 0.73535 | 0.45558 | 0.17188 0.34912 0.53711 | 0.32661 |
| color_mixed_above | 1.5 2 0.5 | 0.96156 | 0.64600 0.73535 0.38867 | 0.69420 | 0.47021 0.53711 0.27734 | 0.50651 |
| color_sub_and_super | 0.1 1 1.5 | 0.80853 | 0.17188 0.53711 0.64600 | 0.46980 | 0.11493 0.38867 0.47021 | 0.33709 |

Both sessions: `proxy_input_bounded_le_one = true`, all Feature 18 output
finite, 11 of 11 samples with distinct input codes, 3 highlight clusters from 3
distinct codes with no collision, all 5 colour patches separated
(`proxy_colored_clusters = 5`), and no proxy code reaches 1.0 at all. The
separation clears the noise by a wide margin: at pw 4.0 the three highlight
medians are 0.64865 / 0.73699 / 0.97256 with bands 3.5e-5 / 3.8e-4 / 9.8e-4; at
pw 8.0, 0.47409 / 0.54070 / 0.73699 with bands 1.2e-3 / 1.0e-4 / 3.8e-4.
Arm-B across-frame bands stay ≤ 1.92e-3 everywhere (`temporal_state_moves_output
= true`); the largest is `color_sub_and_super` at pw 1.0, and the pw 4.0/8.0
sessions stay under 9.8e-4 / 1.2e-3. Arm A's largest band is 4.29e-1, on a
clipped colour patch — the raw arm is both pinned and less stable. At these two
constants the ceiling is never touched, whereas at `paperWhite = 1.0` six of the
eleven samples pin a channel to it.

## Reconstruction

Neither reconstructor inverts the proxy; inverting a saturated code would invent
magnitude the proxy never held. The master supplies the magnitude and the network
supplies only a signed change measured in the proxy domain, converted to stops by
the *quantized* local slope and applied multiplicatively
(`hdr = master · 2^stops`, `|stops| ≤ 1.0`), or by a single luminance ratio with
the master's chroma ratios kept. Three explicit dispositions per channel:
`applied`, `cap`, and `rejected` — a change measured where the code advances by
less than one half-step per stop, or where it is pinned or dead and advances by
nothing, is dropped rather than divided by a near-zero slope.

Identity gate, with neural proxy set equal to input proxy over all six scalars
and all five colour patches: **bit-exact** for both methods at every
`paperWhite`, max relative error 0.0 against a tolerance of 2⁻¹¹ — and measured
on GPU data as well as numerically
(`measured_identity_max_absolute_error = 0.0`,
`measured_identity_bit_exact = true`). This is the check that passed while the
slope was wrong, and the artifact now records why:
`numeric_self_checks.identity_gate_is_blind_to_slope = true`.

Against actual Feature 18 output, all four frames × 11 samples × 2 methods per
session (88 method-rows; the 44 REC_LIN rows carry the per-channel slope):

| paperWhite | all finite | rows hitting the 1-stop cap | rows refusing a correction | REC_LIN rows with an uncorrectable channel | max &#124;stops&#124; | largest relative correction, any frame | reconstructed HDR max | >1 headroom survives |
|---|---|---|---|---|---|---|---|---|
| 1.0 | true | 0 | 24 | 32 | 0.1518 | 11.1% | 4.0352 | true |
| 4.0 | true | 0 | 0 | 0 | 0.1801 | 11.7% | 4.3203 | true |
| 8.0 | true | 0 | 0 | 0 | 0.1475 | 9.7% | 4.0859 | true |

All 24 refusals are in the local-linear arm; the luminance arm records none, and
that is by construction rather than by virtue of a guard — it has no per-channel
slope to guard, only a `input_y > 1e-6` floor on the ratio. Read the first row
carefully: at `paperWhite = 1.0` headroom "survives" because the plateau and
dead-zone corrections were rejected and the master passed through unchanged —
that is the identity path, not a recovery. No proxy-domain signal contributed
anything for 1.5, 2.0 or 4.0 there.

### What the slope fix did to the applied corrections

`frame_3`, largest moves, stale versus corrected. The artifact now records the
two numbers the bug corrupted (`delta_code`, `slope_per_stop`) so this
comparison is re-derivable from the record instead of from a re-run:

| session | sample / channel | master | stale stops → HDR | corrected stops → HDR | delta / slope (corrected) |
|---|---|---|---|---|---|
| 8.0 | `color_sub_and_super` ch0 | 0.1 | −0.45982 → 0.07269 (−27.3%) | −0.12875 → 0.09149 (−8.5%) | −0.00629 / 0.04883 |
| 4.0 | `color_sub_and_super` ch0 | 0.1 | −0.28618 → 0.08203 (−18.0%) | −0.15993 → 0.08948 (−10.5%) | −0.01062 / 0.06641 |
| 4.0 | `color_red_highlight` ch1 | 0.5 | +0.11458 → 0.54150 (+8.3%) | +0.06066 → 0.52148 (+4.3%) | +0.00806 / 0.13281 |
| 8.0 | `scalar_0.1` ch2 | 0.1 | +0.10268 → 0.10736 (+7.4%) | +0.02875 → 0.10199 (+2.0%) | +0.00140 / 0.04883 |
| 4.0 | `color_green_extreme` ch1 | 4 | +0.07187 → 4.20312 (+5.1%) | +0.10268 → 4.29688 (+7.4%) | +0.01123 / 0.10938 |
| 1.0 | `color_sub_and_super` ch1 | 1 | +0.13839 → 1.10059 (+10.1%) | +0.13839 → 1.10059 (+10.1%) | +0.01514 / 0.10938 |

Which way the error goes depends on which side of the knee the true value sits,
and both directions are visible above. Measured slope against relative value at
`paperWhite = 4.0`: 0.0125 → 0.04883, 0.0625 → 0.09375, 0.125 → 0.13281,
0.25 → 0.15625, 0.375 → 0.20313, 0.5 → 0.21875, 0.75 (the knee) → 0.32813, then
1.0 → 0.10938, 1.25 → 0.03125, 1.5 and beyond → 0.00000. The slope rises through
the toe, peaks at the knee and falls off a cliff, so the bug — which read the
slope at `relative / paperWhite`, always further down that curve — shrank
corrections whose true value was above the knee (the fifth row: stale slope
0.15625 at relative 0.25 against the true 0.10938 at relative 1.0, so the stop
count grows from +5.1% to +7.4%) and inflated corrections deep in the toe (the
first row: stale slope 0.013672 at relative 0.00156 against the true 0.04883 at
relative 0.0125, overstating that correction by 3.6×). The fix is not a
limiter; it is a corrector, and it moves numbers in both directions. The last
row is identical in both records, which is the signature of `paperWhite = 1.0`
being bug-inert.

Nothing in the corrected run comes close to the stated budget. The largest
relative move over every frame and method is 11.7% (`color_sub_and_super` ch0,
pw 4.0, `frame_0`/`frame_1`, master 0.1 → 0.08826) against a 1-stop budget that
permits 100% by construction, and zero rows hit the cap. The reset frame,
`frame_0`, is the most moved frame or tied for it at every session constant.

Note which way the outliers go and why, and note that the corrected numbers
change the size of this finding but not its existence. For `color_sub_and_super`
channel 0 at `frame_3`, Feature 18 moved the code by −0.00629 at `paperWhite =
8.0` — 0.63% of the code range — but there the true quantized slope is
only 0.04883 code per stop, because the relative value is 0.1/8 = 0.0125 and the
advance per stop scales with the relative value itself. Dividing one by the
other is −0.12875 stops, −8.5%. The stale math read a slope of
0.013672 — the slope at relative 0.0015625, exactly `paperWhite` times darker
than the scene submitted — and called it −0.45982 stops, −27.3%. The sensitivity
is real; the previous report overstated it by about 3.6× at pw 8.0 and 1.8× at
pw 4.0. The slope range that drives it: `min_nonzero_slope_per_stop` 0.10938 /
0.06641 / 0.04883 and `max_slope_per_stop` 0.29688 / 0.21875 / 0.21875 at
`paperWhite` 1.0 / 4.0 / 8.0. Raising `paperWhite` buys highlight separation and
costs shadow stability; a stage-B design has to decide whether that region wants
a tighter budget than the highlights.

The luminance-ratio arm moves less than the per-channel arm everywhere: largest
`|ratio − 1|` is 0.0305 (`color_sub_and_super` at pw 1.0, stops +0.04340), then
0.0297, 0.0241, 0.0210, 0.0194. There is no independent oracle for this operator
yet — the oracle covers `code_slope_per_stop`, which the luminance arm does not
use. Its dispositions are checked only by the identity gate and by the recorded
`ratio` values, and that asymmetry is a gap, not a pass.

## Decision

**PASS at `paperWhite` 4.0 and 8.0, FAIL at 1.0 — from the corrected run.**
`pass_any_session = true`. The slope fix did not overturn the verdict; it moved
the correction magnitudes and the saturation boundary, and found a dead zone and
a third defect on the way.

Criteria, per the corrected recorded gate:

- proxy input bounded ≤ 1: met at all three sessions, and met as a *measurement* —
  `proxy_input_bounded_le_one` is computed from the maximum `pre_ngx_max` read
  back off the FP16 color texture, with `pre_ngx_matches_submitted = true`
  alongside it, so the bound is what Feature 18 actually received.
- Feature 18 output finite: met.
- output demonstrably not a single clipped value for every highlight/colour
  patch: **met at `paperWhite` 4.0 and 8.0, failed at 1.0**
  (`proxy_highlight_not_single_clipped_value` true/true/false).
- raw >1 clamps as the previous probe saw: confirmed, with zero across-frame
  band, i.e. the whole frame pinned.
- reset and three continuous frames execute: met.
- identity reconstruction retains >1 HDR master headroom: met, bit-exact,
  measured max absolute error 0.0.
- **new in A2, and required for `pass`**: `saturation_verified_by_enumeration`
  (met, all three sessions) and `slope_oracle_max_abs_deviation == 0.0` with
  `code_slope_matches_independent_oracle` (met, 0.0 exactly, all three
  sessions). A session cannot pass stage A on numbers its own math module
  mis-derives.

A `paperWhite` of one scRGB unit is not a viable proxy constant for content
whose highlights reach past 1.8087 units — now 1.4092 units if the correction is
supposed to do anything. The usable form of the result: **a fixed session-wide
proxy with headroom chosen to cover the intended highlight range keeps
highlights distinguishable through Feature 18, at a measurable cost of
everything beyond the saturation point** — and a further cost, previously
unrecorded, of the 1.4092–1.7891 relative band where the code survives but
carries no measurable slope.

This says nothing about image quality. Flat 640x360 uniform patches through a
host-uploaded test array demonstrate number behaviour only; they do not
establish spatial, temporal or colourimetric fidelity, and no fidelity claim is
made here.

## Next smallest feasible step

A per-texel GPU proxy applied to a real device-resident FP16 frame, with the
master kept beside it on the device — not a host colour path — plus the
`paperWhite` constant resolved once from the clip's stated highlight range
against the corrected boundary (1.8087 relative for any signal at all, 1.4092 for
a signal the reconstruction can act on) rather than from a guessed 1.0. An
oracle for `reconstruct_luminance_ratio` before its numbers are trusted at the
same level as the local-linear arm's. Until those exist, nothing here is wired
into export, UI or metadata, and HDR stays disabled:
`production_hdr_enabled: false` in the artifact, and `not_production_color_path:
true`.
