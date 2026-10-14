# HDR10 multiscale UI — stage B2

Date: 2026-10-08. Follows `HDR_MULTISCALE_STAGE_B1.md` and the plan at
`docs/plans/2026-10-02-hdr10-multiscale-fusion.md`.

## Changes

- Removed the stale HDR Dual/Triple rejection at `DLSSService.add_job`. Queue
  snapshots continue to copy `layer_scales` and `layer_weights`; the actual HDR
  capability and configuration checks remain in the backend negotiation/run gate.
- Removed the client mirror's single-only clause and the DLSS page's selection
  guard. The HDR preset now keeps the chosen reconstruction mode, working/layer
  scales, weights, fusion mode, and solver settings. It still sets `multi_pass=1`.
- The HDR preview now names the requested layer count, each scale and weight,
  fusion choice and solver parameters. It describes independent FP16 sessions
  and typed CG fusion when selected, and states that real support is decided by
  backend capability checks. No hardware capability or preferred scale is
  inferred by the UI.
- Existing audio, subtitle, attachment, metadata, chapter, container, remux,
  experimental HDR, and SDR behavior remains as before.

The existing `DeliverPage.tsx` already sends `reconstructionMode`, `layerScales`,
`layerWeights`, and fusion settings to both preview and queued settings; no edit was
needed there. Files changed for B2: `dlss_service.py`,
`frontend/src/services/hdrExport.ts`, `frontend/src/services/exportPreview.ts`,
`frontend/src/pages/DLSSPage.tsx`, `tests/unit/test_hdr_ui_optin_settings.py`,
`tests/node/hdr_export_ui.test.ts`, `tests/node/export_preview.test.ts`, and
`tests/e2e/17_hdr_ui_optin_mock.spec.ts`.

## Verification

| Command | Result |
|---|---|
| `python -m pytest tests/unit/test_hdr_ui_optin_settings.py -q` | 17 passed |
| `node --test tests/node/hdr_export_ui.test.ts tests/node/export_preview.test.ts` | 45 passed |
| `npm run build` (`frontend`) | TypeScript and Vite production builds passed |
| `npx playwright test tests/e2e/17_hdr_ui_optin_mock.spec.ts` (repository root) | 3 passed |
| `git diff --check -- <B2 files>` | Passed; Git emitted only existing CRLF normalization warnings |

The initial Playwright invocation from `frontend` hit duplicate Playwright module
loading because both root and frontend dependencies were resolved. Running from the
repository root with its configuration passed all three mock tests.

## Remaining validation

No GPU export, real HDR file validation, benchmark, or full test suite was run in B2.
Stage C still needs its planned single HDR10/remux hardware acceptance, dual 50/75
fusion smoke, then triple 40/67/85 smoke, plus remaining specified final gates.
Hardware acceptance remains outstanding; performance is `NOT_CHARACTERIZED`.
