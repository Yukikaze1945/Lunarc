# HDR Stage B — P010 → scRGB FP16 → P010 roundtrip repair

Date: 2026-09-30. Owner: Qoder (implementation), Codex (audit).
Evidence: `bench_out/hdr_gates_roundtripfix/` (fresh directory; earlier artefacts untouched).

## Verdict

The five colour roundtrip gate failures were **an oracle defect in the test harness, not a
defect in the GPU bridge**. The harness's model of the FP16 plane used the reciprocal of the
relative→scRGB unit factor (`1/125` where the store holds `×125`), so it quantised a plane
15625 times finer than the one the device writes, and it compared the device against a number
the encoder never read.

`native/gpu_hdr_scrgb_bridge.cpp` was **not modified**. Its SHA-256 is unchanged from the
counter-fix job (`295643fa…`). The repair is confined to the oracle in
`native/gpu_hdr_scrgb_bridge_test.cpp`.

Harness now: **exit 0, `== 0 failure(s) ==`, 79 `[ ok ]`, 0 `[FAIL]`**. No tolerance was
loosened; the four original failing checks pass on their original bounds, and two new checks
were added on top.

This is **not** a production HDR10 export and **no** HDR export has passed. Performance remains
**NOT_CHARACTERIZED**.

## The defect, proven

`Path::Path()` in the harness:

```cpp
relative_to_scrgb = contract.scrgb_per_nit * 10000.0;   // 0.008 * 10000 = 125   (was: 1.0 / (...))
```

`scrgb_per_nit = 1/80`, so the store's factor is `10000/80 = 125` (`kScrgbPerRelative`,
`gpu_hdr_scrgb_bridge.cpp:152`). The harness held `1/125 = 0.008`.

Consequences, each measured:

1. **The model plane was not the store's plane.** A fixture sample the GPU holds as
   `R = 12.4375` scRGB the harness modelled as `0.000796`. Its binary16 ulp is `4.97e-7`
   instead of `0.0078` — a grid **15625× (125²) finer**.
2. **The quantum box was mis-derived.** `half_step()`/`bound_box()` asked "how far can the codes
   move if the stored half is ±½ ulp?" using an ulp of a plane that does not exist. For a
   gamut-negative sample at the PQ toe that answer came out **0 codes**, so the harness asserted
   a lossless roundtrip where the store physically cannot be lossless. Coded black got the
   opposite error: a box of 35/18/17 codes it did not need.
3. **The reference encoded a different number than the encoder did.** `mirror[]` (the exact
   doubles path) was built from the harness's own rounding, not from the value the device
   actually read. Decode and encode self-cancel in double precision, so the model returned the
   *unrounded* answer — which differs from the device by exactly the real FP16 loss, up to 16
   codes. That is `roundtrip_reference_max_codes = 16`.
4. **The reported luminance error was 15625× under-stated.** The gate printed
   `0.000522 cd/m²` for a sample whose true error is `8.28 cd/m²`.

The failing rows are precisely the rows whose model magnitude was largest relative to their own
grid: the gamut-negative primaries held at coded black, where three cancelling terms reconstruct
a near-zero component.

## Independent evidence, before touching the oracle

Host-only diagnostic `scratch/hdr_roundtrip_diagnostic.py` (numpy, float64 arithmetic with an
explicit float16 store step) replayed the raw plane dumps the harness already writes
(`bench_out/hdr_gates_counterfix/master`, `sweep_master`) — readback inside the isolated test
only; production reads no plane back:

- Doubles decode × 125 rounded to binary16 equals the **GPU's stored half bit-for-bit** for all
  12 fixture samples and 11 of 12 sweep samples (the exception is one binary16 step on
  `violet past 709`, channel B).
- Doubles encode of the **actual stored halves** reproduces **every** GPU output code exactly
  (max gap 0, Y and UV, both jobs). So the first divergence between device and reference is
  *nowhere in the device path*; it is introduced by the harness's own quantiser.
- The transcendental path is not implicated: the same `hdr_pow` result appears at both scales,
  and the code differences the harness blamed on it vanish when the model uses the store's grid.

## Hypotheses tested and rejected

| Hypothesis | Disposition |
| --- | --- |
| Negative scRGB clamped to zero before encode | **Rejected** — `source_negative = 49140` survive the plane, `encode_out_of_gamut = 0`, and the negative primaries' stored halves (`G = -6.1406`, `B = -0.1360`) are held and read back exactly. |
| Limited-range offset mishandled for negative linear | **Rejected** — coded black and near-black rows roundtrip bit-exactly; the errors are at primaries at black, whose offsets are identical in both paths. |
| `hdr_pow`/`hdr_exp2` approximation error moves codes | **Rejected** — doubles vs device at the correct grid differ by 0 codes on every sample that the grid resolves. |
| P010 pack/store race (two luma pixels share a word) | **Rejected** — `master vs composited: 0 of 262144 differing halves`, and the sweep's write pattern is stable across runs (identical metrics in two runs). |
| Reference missing the high clamp the device applies | **Rejected** — no sample reaches `relative > 1.0` before `quantise()` in either job, so the clamp is not exercised; `encode_clamped_high = 0`. |
| Device decode is wrong | **Rejected** — bit-for-bit against the doubles path (above). |
| The prior counter-fix's `box 0/0/0` proves losslessness | **Withdrawn** — see cause 2. The box derivation itself was the suspect, and it failed. |

## Files changed

Only `native/gpu_hdr_scrgb_bridge_test.cpp`:

1. `Path::Path()` unit factor corrected (the root cause, with a comment recording why the
   roundtrip had been agreeing with itself regardless).
2. `PlaneDump` gained `pattern(row, column, channel)` returning the raw binary16 **bit pattern**,
   and `at()` now decodes through it. New free helper `half_pattern_index()` maps a pattern to a
   sign-magnitude total order so two halves can be compared in grid steps.
3. `RowError` gained `stored[3]` (the value the plane actually holds), `model[3]` (the value the
   doubles path rounds to) and `pattern_gap` (steps between them). `Roundtrip` aggregates
   `decode_pattern_gap` / `decode_pattern_samples`.
4. `run_roundtrip` conditions the oracle on the real plane: the reference encodes
   `master_dump.pattern(...)`, i.e. the number the encoder read, instead of its own rounding.
   When the plane is unreadable it falls back to the model rounding and the gap is measured
   against that.
5. `show_row()` prints, per sample, `stored / model / steps / reference codes` under the existing
   row line; both duplicated print loops now call it.
6. Two new checks: `the master plane holds the binary16 the doubles decode rounds to, within one
   grid step` (fixture and sweep). Four new summary keys
   (`roundtrip_decode_pattern_gap_max/_samples`, `sweep_roundtrip_decode_pattern_gap_max/_samples`).
7. `gates()` writes the summary again at the end — the job-2 keys were added after the single
   earlier write and were previously absent from `gate_summary.txt`.

No check was removed and no bound widened; tolerances elsewhere are byte-identical.

## Commands and exits

| Command (from `native/`) | Exit |
| --- | --- |
| `build_gpu_hdr_scrgb_bridge.bat` (`/O2 /EHsc /std:c++17 /W4 /WX /utf-8`) | 0 |
| `compile_hdr_gates_only.bat` | 0 |
| `gpu_hdr_scrgb_bridge_test.exe ..\bench_out\hdr_gates_roundtripfix` | **0** (`scratch/rtfix_run.log`) |
| `compile_hdr_gates_only.bat` (after the summary-write fix) | 0 |
| `gpu_hdr_scrgb_bridge_test.exe ..\bench_out\hdr_gates_roundtripfix` | **0** (`scratch/rtfix_run2.log`, final) |

MSVC x64, VS2022 Developer Command Prompt v17.14.40, `/W4 /WX` clean.

### SHA-256

| File | SHA-256 |
| --- | --- |
| `native/gpu_hdr_scrgb_bridge.cpp` | `295643fa1ea10bc3b2d198a99931374b050da0ed64abc289cc0a0c1a290a6739` (unchanged) |
| `native/gpu_hdr_scrgb_bridge.h` | `4773df5dffae8d45d386031ce2f9e2eb92ed43b03a0734f75a9b734d26384909` (unchanged) |
| `native/gpu_hdr_scrgb_bridge.dll` | `41a38ce99b022ff43676d5c4169babe477ca42d3894e0ec47ef0ea28dab2fbd5` |
| `native/gpu_hdr_scrgb_bridge_test.cpp` | `b1db34cd8f8e1fc8cbe5b776374a1da0b3aa6ae4b7fc4f66870c432069bba4ee` |
| `native/gpu_hdr_scrgb_bridge_test.exe` | `79109740715284233d3b514d9a86d948396ff699e2482263f5c324557a39372d` |
| `bench_out/hdr_gates/gate_summary.txt` | `3647483dc7cbcc4b50bc169569cdf9ea33055fb475d3a9d4de6e10cdd2762f14` (prior evidence, unchanged) |
| `bench_out/hdr_gates_counterfix/gate_summary.txt` | `b1b05b3b049313451f0c6ea30d3018ecfe1dd812d4351ce0cc75594d2a0c081e` (prior evidence, unchanged) |
| `bench_out/hdr_gates_roundtripfix/gate_summary.txt` | `07eaf43aef45b20d5bca82e9b7eea3c5553dbaf400ae3061aaa63a73c9ee43d2` (new evidence, this run) |
| `bench_out/hdr_gates_roundtripfix/gate_run_raw.log` | `fdf9a4b8d79817d80fa65f0aaff13e58b7c733da745eb9b9da13fa4e6ad47d9b` (raw console log of the final run, preserved for review) |
| `bench_out/hdr_gates_roundtripfix/stageb_roundtrip_fix.json` | `111fd12b40dfc9aa10124eede884c9ad0819e459c0ecd166b119820e3dae0623` |

The DLL hash differs from the counter-fix job's `5f1bb773…` while the source hash is identical:
this is the PE link timestamp of a rebuilt-from-same-source binary, not a code change.

## Gate results

### Job 1, fixture (paperWhite from the contract, ceiling 12.5 scRGB)

| Sample | Y want→got | CB | CR | quantum box (nits/step, corrected) | worst light | stored scRGB R,G,B | steps vs model | reference codes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coded black | 64→64 | 512→512 | 512→512 | 0/0/0 (was 35/18/17) | 0.000 | 0.0000 0.0000 0.0000 | 0 | 64 512 512 |
| inside the ceiling | 717→717 | 512→512 | 512→512 | 0/0/0 | 0.099 | 11.7891 ×3 | 0 | 717 512 512 |
| at the ceiling | 722→722 | 512→512 | 512→512 | 0/0/0 | 0.019 | 12.4219 ×3 | 0 | 722 512 512 |
| reference white 80 | 490→490 | 512→512 | 512→512 | 0/0/0 | 0.030 | 1.0049 ×3 | 0 | 490 512 512 |
| SGOT 203 | 573→573 | 512→512 | 512→512 | 0/0/0 | 0.047 | 2.5469 ×3 | 0 | 573 512 512 |
| near black 5 | 281→281 | 512→512 | 512→512 | 0/0/0 | 0.001 | 0.0624 ×3 | 0 | 281 512 512 |
| **gamut-negative red** | 224→243 | 425→415 | 824→811 | **31/12/22** (was 0/0/0) | 2.545 | 12.4375 −0.9326 −0.1360 | 0 | 243 415 811 |
| **gamut-negative green** | 499→511 | 276→269 | 210→233 | **24/12/41** (was 22/20/37) | 2.714 | −6.1406 11.8438 −1.0322 | 0 | 511 269 233 |
| **gamut-negative blue** | 102→112 | 843→838 | 485→489 | **21/12/11** (was 0/0/0) | 8.280 | −0.8042 −0.0922 12.3516 | 0 | 112 838 489 |
| warm mid | 485→485 | 444→444 | 628→628 | 0/0/0 | 0.155 | 9.5938 −0.1788 0.0654 | 0 | 485 444 628 |
| cool mid | 517→517 | 551→551 | 440→440 | 0/0/0 | 0.049 | −0.7305 2.1895 3.1699 | 0 | 517 551 440 |
| yellow low | 576→576 | 411→411 | 520→520 | 0/0/0 | 0.059 | 3.1738 3.0137 −0.0163 | 0 | 576 411 520 |

Every row's GPU codes equal the reference codes computed from the value the plane actually holds:
`roundtrip_reference_max_codes = 0` (was 16), `roundtrip_outside_box_components = 0` (was 6),
`roundtrip_decode_pattern_gap_max = 0`.

### Job 2, full-code sweep at paperWhite 4.0

The three primary-at-black rows move from a box of 0 to **81/28/56**, **58/48/96**, **75/41/44**
codes; their actual losses (17/19/47 luma) sit inside them. `violet past 709` is the one sample
where the device's float32 curve and the doubles curve land on opposite sides of a round-to-
nearest-even midpoint: the plane holds `B = 54.7812`, the model rounds to `54.8125` — **one
binary16 grid step**, and the sample still roundtrips bit-exactly (432/741/713, box 1/0/1).
`sweep_roundtrip_reference_max_codes = 0`, `sweep_roundtrip_outside_box_components = 0`,
`sweep_roundtrip_decode_pattern_gap_max = 1` over 1 of 12 samples.

### Aggregate, before → after

| Key | counter-fix run | this run |
| --- | --- | --- |
| `roundtrip_y_max_codes` / `roundtrip_uv_max_codes` | 19 / 23 | **19 / 23** (device unchanged) |
| `roundtrip_reference_max_codes` | 16 | **0** |
| `roundtrip_outside_box_components` | 6 | **0** |
| `roundtrip_amplified_samples` | 2 | 3 |
| `roundtrip_worst_light_error_nits` | 0.000522 | **8.2798** (now honest) |
| `roundtrip_y_mean` / `uv_mean` codes | 3.4158 / 2.5827 | 3.4158 / 2.5827 |
| `sweep_roundtrip_reference_max_codes` | (not written) | **0** |
| `sweep_roundtrip_outside_box_components` | (not written) | **0** |
| `sweep_roundtrip_worst_light_error_nits` | (not written) | 105.293 |
| harness exit / failures | 1 / 5 | **0 / 0** |

## Representational loss vs implementation deviation, kept separate

- **Representational (irreducible at this store format):** 3 of 12 fixture samples and 3 of 12
  sweep samples lose codes because a 2020 primary held at coded black is reconstructed from three
  cancelling terms, and the PQ toe amplifies one binary16 step into tens of codes. Worst true
  loss: 47 luma codes / 105.29 cd/m² on `BT.2020 primary blue` in the sweep, all inside the
  store's own derived quantum. This is a property of holding linear scRGB in binary16, and it is
  reported as such rather than folded into a pass/fail.
- **Implementation deviation (device vs reference):** zero codes once the reference is asked about
  the number the encoder read. One residual, bounded and separately reported: 1 of 12 sweep
  samples differs from the doubles model by exactly **1 binary16 grid step** in the held plane
  (`violet past 709`, B), from float32-vs-double rounding at a midpoint; it does not change any
  output code.

## Failures remaining

- Colour roundtrip gates: **none** (exit 0).
- The stage-B gates that were already passing (counter map proofs, source/proxy/reconstruction/
  encode reports, identity reconstruction, preservation above the ceiling, HLG/full-range/709/CL
  refusals) all still pass — 79 `[ ok ]`.
- Not addressed and out of scope here: host-side integration of the bridge, production HDR export,
  the SDR gates, and any A/B Feature 18 or 4K work. **No production HDR export has passed.**
- `Performance: NOT_CHARACTERIZED` — no benchmark was run in this job.

## Next smallest action

Take the now-trusting numerical gate as the Stage B baseline and move one step outward: run the
**existing** Stage B chain end-to-end against a real short P010 clip through the host path (no new
kernel work), in a fresh evidence directory, and report the per-frame counter reports against
these same quantum boxes. That is where the FP16 toe loss first becomes a viewing question rather
than a numerical one, and whether a wider store for the near-black primaries is worth proposing.
Nothing in this job should be read as the experimental HDR10 implementation being complete.

## Preservation

No commit, no push, no clean, no reset. `bench_out/hdr_gates/` and
`bench_out/hdr_gates_counterfix/` are byte-unchanged (hashes above). All pre-existing dirty and
untracked files were left as found; the only files this job added are
`native/gpu_hdr_scrgb_bridge_test.cpp` (edited), `scratch/hdr_roundtrip_diagnostic.py`, the
`scratch/rtfix_*` logs, `bench_out/hdr_gates_roundtripfix/`, and this report.
