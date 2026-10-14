# HDR Stage B — signed atomic-min counter map fix (bounded recovery)

Recovery for crashed job `qbg_20260930_d91892fe` (process gone, no new native
changes saved). This is a scoped counter/telemetry fix plus one native
verification run. It is **not** a production export, not a stage-A rerun, not a
performance benchmark, and it does **not** enable HDR. All prior dirty files and
the prior HDR gate artifact are preserved (see "Preservation" below).

## The bug (exact)

The HDR bridge finds the extremes of a plane with the only atomics a typed UAV
buffer offers — `InterlockedMin` / `InterlockedMax` on an *order-preserving*
float→uint key. The HLSL forward map was:

```hlsl
uint sortable(float value) {
    uint bits = asuint(value);
    return bits ^ ((bits >> 31) | 0x80000000u);   // BUG
}
```

For a non-negative float `(bits >> 31)` is `0`, the mask is `0x80000000`, and
this sets only the sign bit — correct. For a **negative** float `(bits >> 31)`
is `1`, the mask becomes `0x80000001`, so it XORs only the sign bit and bit 0
instead of complementing **every** bit. Two consequences:

1. **Ordering inverted among negatives.** The key tracked float *magnitude*, so
   `InterlockedMin` selected the near-zero negative, not the most-negative one.
2. **No roundtrip with the host inverse.** `unsortable_to_bits()` decodes keys
   below `0x80000000` with `~key`, which is the exact inverse of the *correct*
   map, not of this broken one — so the decoded minimum was garbage.

That is the origin of `source_min = -250.68086` in the prior
`bench_out/hdr_gates/gate_summary.txt`: a phantom key decode, **not a pixel**.
The positive path was never touched, which is why `source_max = 12.4355` was
identical before and after the fix.

## The fix

`native/gpu_hdr_scrgb_bridge.cpp` — the HLSL `sortable()` now implements the
canonical form the host inverse already expects (flip the sign bit for
non-negative, complement every bit for negative):

```hlsl
uint sortable(float value) {
    uint bits = asuint(value);
    return bits ^ ((bits >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}
```

This is an exact bijection on all 2^32 bit patterns and a strict order-embedding
of the finite floats. `unsortable_to_bits()` on the host was already correct for
this form; a comment was added tying the two together. No other native file was
modified by the fix itself.

### Defined handling of zeros, extrema, infinity, NaN (the total order)

* `-0.0 -> 0x7FFFFFFF`, `+0.0 -> 0x80000000` — adjacent keys at the join
  (numerically equal, deliberately ordered -0 before +0).
* Strictly increasing over the finite floats: `x < y  =>  sortable(x) < sortable(y)`.
* `-inf` sits below every finite value; `+inf` above every finite value.
* A positive NaN sorts just **above** `+inf`; a negative NaN just **below** `-inf`.
* NaN never reaches the counters on a frame that passes: the decode kernel
  counts non-finite samples and the caller refuses the job before the reported
  extremes are trusted.

## Proof 1 — pure map (host, deterministic; added to the harness)

`native/gpu_hdr_scrgb_bridge_test.cpp` gained `run_sortable_gates()`, a CPU
mirror of both maps, run before any GPU work. From `scratch/stageb_run.log`:

```
-- counter map: sortable() / unsortable_to_bits() --
[ ok ]  unsortable(sortable(x)) == x bit-for-bit for finite floats, both signed zeros, the infinities and NaN
  roundtrip failures: 0
[ ok ]  sortable is strictly increasing over the finite floats (x < y  =>  sortable(x) < sortable(y))
  monotonicity failures: 0
[ ok ]  the signed zeros map to the adjacent keys below and above the join
[ ok ]  a larger-magnitude negative sorts below a near-zero negative (InterlockedMin finds the true minimum)
[ ok ]  the order extends: -inf below every finite, +inf above every finite
[ ok ]  a positive NaN sits at the top of the total order, above +inf
```

Roundtrip is bit-pattern equality over the whole exponent range in both signs
plus the named extrema/`±0`/`±inf`/`±NaN`; monotonicity over a sorted dense set.

## Proof 2 — native GPU vs independently-measured FP16 (the run)

One short run of the existing harness into a **new** directory
(`bench_out/hdr_gates_counterfix/`), comparing the GPU's reported extrema
(decoded from the fixed counter keys) against the actual FP16 diagnostic plane
read back and measured independently on the CPU. GPU extrema are taken from the
pre-store `float32` value, the plane holds the RNE `half` store, so they may
differ by at most one half-grid quantum at that magnitude.

| Quantity | GPU-reported (fixed) | Measured FP16 plane | abs gap | Prior (broken) |
|---|---|---|---|---|
| source min | `-6.1415004730` | `-6.140625` | `0.000875473` | `-250.6808624268` |
| source max | ` 12.4355144501` | ` 12.4375` | `0.00198555` | ` 12.4355144501` (unchanged) |
| composite min | `-6.0776367188` | `-6.078125` | `0.000488281` | `-250.6249694824` |
| composite max | ` 13.2114257812` | ` 13.2109375` | `0.000488281` | ` 13.2114257812` (unchanged) |

Every gap is inside the half quantum (≤ 0.00199 at |6|, ≤ 0.0020 at |12|). The
`-250` reports are gone; the corrected minimum matches the true plane minimum.

Harness log lines:

```
  source: pixels 65536 nonfinite 0 negative 49140 above ceiling 0 range [-6.1415, 12.4355]
  counter check: source GPU [-6.1415, 12.4355] vs measured FP16 [-6.14062, 12.4375] (gaps 0.000875473 / 0.00198555)
[ ok ]  the GPU-reported source min and max agree with the independently measured FP16 master within the half grid
[ ok ]  the reported source minimum is a real in-range negative, not the -250.68 phantom (telemetry defect resolved)
  counter check: composite GPU [-6.07764, 13.2114] vs measured FP16 [-6.07812, 13.2109] (gaps 0.000488281 / 0.000488281)
[ ok ]  the GPU-reported composite min and max agree with the measured FP16 composite within the half grid
[ ok ]  the composite minimum is a real in-range value, not the -250.62 phantom
```

**Verdict: the `-250.68` was a telemetry (counter-map) defect, not a real pixel
and not the HDR colour gate.** `negative_components = 49140` and `source_max`
were identical before/after, confirming the store and the positive path were
never affected — only the negative-minimum key was.

## Not conflated: the colour gate still fails (kept explicit, untouched)

A successful counter fix is **not** a passing HDR colour gate. The independent
P010→scRGB→P010 roundtrip gate fails exactly as it did in the prior artifact
(same numbers, because `sortable()` is not in the roundtrip code path):

```
[FAIL]  the GPU's roundtrip equals the exact-arithmetic reference
[FAIL]  every roundtrip error is inside the store's own quantum for that sample
[FAIL]  a sample the half grid resolves roundtrips to the same code it started on
[FAIL]  the sweep's roundtrip equals the exact-arithmetic reference over the whole legal range
[FAIL]  every sweep error is inside the store's own quantum for that sample
== 5 failure(s) ==
```

`roundtrip_y_max_codes = 19`, `roundtrip_uv_max_codes = 23`,
`roundtrip_reference_max_codes = 16`, `roundtrip_outside_box_components = 6`,
`roundtrip_amplified_samples = 2` — byte-for-byte the same as the prior run. The
overall harness exit code is `1` **from these 5 colour-gate failures, not from
the counter checks** (all counter checks are `[ ok ]`). Nothing was loosened.

## Next smallest concrete numeric defect (from observed evidence)

Not fixed here (out of scope; the whole pipeline must not be built in one job).
The roundtrip failure is isolated to **gamut-negative samples — a BT.2020 primary
at/below black whose 709 conversion goes negative — where the store's own
quantum box is computed as 0 yet the GPU still loses codes.** From the per-row
table:

```
  gamut-negative red     y 224->243 (+19)  cb 425->415 (+10)  cr 824->811 (+13)  box 0/0/0   light 0.000
  gamut-negative green   y 499->511 (+12)  cb 276->269 (+7)   cr 210->233 (+23)  box 22/20/37 light 0.000
  gamut-negative blue    y 102->112 (+10)  cb 843->838 (+5)   cr 485->489 (+4)   box 0/0/0   light 0.001
```

All neutral / in-gamut rows roundtrip exactly (want == got). Green's error falls
*inside* its wide toe box (quantization, expected). Red and blue have
**box 0/0/0** — the harness predicts a lossless roundtrip there — yet lose 10–19
luma codes. That is `outside_box_components = 6` (red+blue RGB). So the smallest
next target is precise: *the device encode/decode path mishandles a negative
linear 2020 primary for samples the half grid resolves, diverging from the
doubles reference by up to 16 codes.* Likely the limited-range offset /
negative-clamp is applied inconsistently between the GPU kernel and the exact
reference for primary-at-black; that is where the next bounded job should look.

## Build hashes and commands

Compiler: MSVC via `vcvars64.bat`, "Visual Studio 2022 Developer Command Prompt
v17.14.40", x64. Flags `/O2 /EHsc /std:c++17 /W4 /WX /utf-8` (warnings-as-errors;
both targets compiled clean).

Commands and exit codes (run from `native/`):

| # | Command | Exit |
|---|---|---|
| 1 | `build_gpu_hdr_scrgb_bridge.bat` (rebuilds `gpu_hdr_scrgb_bridge.dll`) | `0` |
| 2 | `compile_hdr_gates_only.bat` (compiles `gpu_hdr_scrgb_bridge_test.exe`, does not run) | `0` |
| 3 | `gpu_hdr_scrgb_bridge_test.exe ..\bench_out\hdr_gates_counterfix` | `1` (the 5 colour-gate FAILs) |

SHA-256:

| Artifact | Before | After |
|---|---|---|
| `gpu_hdr_scrgb_bridge.dll` | `50f9129294d0fd84ca8a614b7ed94ce32c4d000992c28ac2d97ce298be13e5a0` | `5f1bb773b9253c432c45edc3f3ace993219e2fa89b9c7f54b97b0ffce185133c` |
| `gpu_hdr_scrgb_bridge.obj` | — | `938c45497f618cba9bb6b37c03d0164cca7b94f360ad4d3df10c7116fb475ebc` |
| `gpu_hdr_scrgb_bridge_test.exe` | `dd796bc6709e30bf2dba360921224af0f352c63851c3c74a7ae04bab1901af70` | `105ac8b0cbba18fdf769879be9bb576a474c14c948b7471687ff618aada43403` |
| `gpu_hdr_scrgb_bridge.cpp` (source, post-edit) | — | `295643fa1ea10bc3b2d198a99931374b050da0ed64abc289cc0a0c1a290a6739` |
| `gpu_hdr_scrgb_bridge_test.cpp` (source, post-edit) | — | `b7d5a2b21613f0fe3629b8eda191ad9c7e299bd661cd58720063e213f52ac259` |

## Preservation

* The prior artifact directory `bench_out/hdr_gates/` was **not** written to.
  Its `gate_summary.txt` still hashes to `3647483dc7cbcc4b50bc169569cdf9ea33055fb475d3a9d4de6e10cdd2762f14`
  and still shows `source_min = -250.68086242675781` (the phantom). The harness
  was run into `bench_out/hdr_gates_counterfix/` instead.
* No `git clean` / `reset` / `commit` / `push` was run; no HDR enabled; no
  stage-A rerun; no production export; no performance benchmark.
* Compact machine-readable result with unique keys:
  `bench_out/hdr_gates_counterfix/stageb_counter_fix.json`.
* Full run/build transcripts:
  `scratch/stageb_build_dll.log`, `scratch/stageb_build_test.log`,
  `scratch/stageb_run.log`.
