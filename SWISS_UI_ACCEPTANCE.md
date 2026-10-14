# Swiss UI acceptance — 2026-10-10

## Current UI build — follow-up correction

Current executable: `bench_out/swiss_current_20261010/win-unpacked/Lunarc.exe`.
The earlier executable paths below are historical evidence, not launch targets.
Both `run_gui.bat` and `run_davinci_studio.bat` now build the current frontend
before opening Lunarc; the optional classic Tkinter interface is still available
through `python gui.py`.

The new comparison page now follows the workbench's Swiss design: shared colour
tokens, aligned grid, flat sections and fine divider lines, a maximum 2px control
radius, and one primary action. Removed its English eyebrow, redundant subtitle,
rounded container stack, and persistent technical captions. Stream information
and actual PTS remain available behind a closed Media Information disclosure;
HDR retains a short visible uncalibrated-display notice.

Verified against the actual packaged app, with source/package entry and asset
SHA256 equality: three styles, four primary sliders, automatic mask, closed
Advanced section, eight navigation items, and 1440×900 dark/light comparison
screenshots. SDR8, SDR10 and HDR file comparison passed wipe, A/B, zoom, actual
PTS stepping, real media-time playback at 0.5×, one audio source, and missing-file
handling. Renderer errors: zero. The 50ms SDR10 fixture prompted a real playback
clock fix: decoded frames no longer rewind the clock, and short audio clips
record their actual end time between ticks.

- `bench_out/native_backends_20261010/electron_swiss_current_run3/report.json`
- `bench_out/native_backends_20261010/electron_swiss_current_run3/compare_dark_1440x900.png`
- `bench_out/native_backends_20261010/electron_swiss_current_run3/compare_light_1440x900.png`
- `bench_out/native_backends_20261010/workspace_default_run1/workspace_default_1440x900.png`

Latest CPU validation: 1505 Python unit tests and 57 Node tests passed; current
TypeScript/Vite build and Electron packaging passed. This UI result is not a
claim of general picture quality or performance. See `NATIVE_BACKENDS_ACCEPTANCE.md`
for the completed short native-backend gates and remaining restrictions.

The final comparison follow-up gate is
`bench_out/native_backends_20261010/electron_swiss_final_run4/report.json`.
The SDR display adapter also passed actual 8/10-bit midgray preservation tests;
it no longer mixes a display-gamma inverse with the processing OETF.

## Result

Implemented the approved minimal workbench: slim navigation rail, large preview,
charcoal/warm-yellow dark theme, off-white/blue light theme, and persisted theme.
The basic editor contains Default / Movie / Natural (style 0 / 2 / 1), NR strength,
local tone strength, local structure strength, skin structure strength, and automatic
mask. Remaining controls are behind a closed Advanced disclosure. Existing five
advanced tabs and export routes remain available.

The implementation preserves existing backend defaults and algorithms. The
reconstruction-mode UI handler now sends the intended complete settings patch.
Display density and reduced-motion preferences restore before the workbench mounts.

## Verification

- TypeScript / Vite build and final Windows Electron packaging: PASS.
- Node regression tests: 52 passed.
- Export/backend and render-queue UI contract tests: 41 passed.
- Real Electron interactions with actual backend: PASS; three styles, four sliders,
  automatic mask, advanced tabs, dual-mode settings, theme persistence, and navigation.
- Real six-frame GPU Transfer export through Deliver and the queue: PASS. ffprobe
  confirmed HEVC, 1280×720, six frames. Invalid NV12 selection produced a preflight
  error without creating a job. No renderer errors.
- Final packaged executable: both themes at 1440×900 and 1000×680, decoded preview,
  no document overflow, persisted display preferences, and owned backend cleanup: PASS.
- `git diff --check`: PASS.

Evidence (repository-relative, ignored runtime artifacts):

- `bench_out/swiss_ui_20261010/final/result.json` — interaction gate.
- `bench_out/swiss_ui_20261010/queue_final/result.json` — six-frame export gate.
- `bench_out/swiss_ui_20261010/package_final_check/result.json` — final packaged gate
  and reviewed screenshots. Earlier screenshots are superseded by this directory.
- `bench_out/swiss_ui_20261010/package_final/win-unpacked/Lunarc.exe` — runnable build.

## Scope and limits

This revision changes the GUI and its parameter wiring; it does not change HDR
mathematics, native processing, remux, or codecs. HDR hardware gates and the full
Python suite were not repeated for this GUI-only revision. The six-frame hardware
export above is SDR. Performance=NOT_CHARACTERIZED.

The standard lint command is not green: it scans generated release files and
reports existing React warnings. Scoped source lint exits successfully with eight
baseline warnings; no clean lint result is claimed.

The Windows output is the existing portable application folder and uses the
project's local runtime environment. It is not a newly self-contained installer.
Changes remain uncommitted; unrelated user files were retained.

## Follow-up correction — window chrome and preview overlays

Removed the unrequested registered-mark symbol and the two central black preview
labels. Split comparison retains only its thin divider; the video display path is
unchanged.

The native title bar is now hidden and Windows caption buttons share the workbench
header, using Electron's [title bar overlay](https://www.electronjs.org/docs/latest/tutorial/custom-title-bar).
The header is draggable, interactive buttons are excluded from dragging, and the
actual caption-button safe area reserves space at both tested window sizes. Window
controls follow the current light/dark palette. The build's CommonJS preload
fallback now compiles the actual preload source rather than substituting an old,
duplicated API definition.

Build, packaging, and real packaged Electron verification: PASS. Verified both
themes at 1440×900 and 1000×680, the caption safe area, header drag styles, clean
preview divider, theme/display preference restoration, and backend cleanup.
Native window captures were visually reviewed and include the caption buttons.
No additional hardware export or algorithm gates were repeated for this correction.

- Current executable: `bench_out/swiss_ui_20261010/package_chrome_fix/win-unpacked/Lunarc.exe`.
- Current evidence: `bench_out/swiss_ui_20261010/chrome_fix_window/result.json`.
- Native captures: `workbench_dark_1440x900_window.png` and
  `workbench_light_1440x900_window.png` in the current evidence directory.

These supersede the earlier packaged version and screenshots above.

## Follow-up correction — SDR incorrectly labelled HDR

`pipeline.probe_pixel_info` previously treated every source above eight bits (and
the SDR BT.2020 transfer identifiers) as HDR. The preview badge uses this source
classification independently from the experimental HDR export checkbox.
Classification now identifies PQ / HLG transfers; bit depth alone does not set
`is_hdr`. Existing high-bit-depth preview diagnostics remain separate.

Nine classification regression cases and the existing export/backend UI contract
tests passed (35 total). Actual file probes confirmed eight-bit SDR and a newly
derived two-frame ten-bit SDR sample are not HDR, while the existing PQ HDR sample
remains HDR. The packaged app imported the ten-bit SDR sample, reported `is_hdr=false`,
displayed SDR in the material row, and showed zero HDR preview badges.

- Current executable: `bench_out/swiss_ui_20261010/package_hdr_label_fix/win-unpacked/Lunarc.exe`.
- Evidence: `bench_out/swiss_ui_20261010/hdr_label_fix/probe_result.json` and
  `bench_out/swiss_ui_20261010/hdr_label_fix/electron_result.json`.
- The user's specific video was not accessible through a running backend, so its
  exact source metadata was not inspected. The confirmed classification bug is
  fixed and covered independently of that file.

This executable supersedes the chrome-fix package above and retains its changes.
