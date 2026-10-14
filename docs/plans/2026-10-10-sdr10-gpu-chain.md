# SDR 10-bit GPU processing

## Goal

Automatically route a 10-bit limited-range BT.709 SDR source on GPU Transfer through P010 decode, relative linear Rec.709 FP16 master, fixed bounded FP16 Feature 18 proxy, signed reconstruction/fusion and P010 Main10 encode. Keep BT.709 and no HDR metadata/SEI. Preserve existing HDR10 and 8-bit SDR routes.

## Changes

- Add typed SDR10 color bridge entry points; reuse GPU resources, filters and reconstruction. SDR reference white is fixed at relative linear 1; it does not use PQ or absolute HDR luminance.
- Add neutral P010 decoder open and SDR10 encoder open with strict BT.709 signaling and no mastering/content-light metadata.
- Select FP16 before session creation and reuse the existing high precision export lifecycle, multiscale and FFmpeg mapping/verification.
- Show concise precision information in export UI; the 8-bit preview is separate from export precision.

## Verification

- Transfer/reference math, admission, native wrapper and routing tests; existing HDR and SDR regressions.
- A 4–6 frame cut of `D:\录像\闪耀色彩\水团正式版\33.DREAMY COLOR.mkv` (AV1 Main, 3840×2160, yuv420p10le, limited BT.709, PCM audio).
- Confirm actual GPU FP16/P010 chain, full dimensions/count/PTS, Main10 BT.709, copied audio, no HDR mastering/CLL SEI and measured image host boundaries.
- Short affected HDR/SDR8 gates, no benchmark. Performance=NOT_CHARACTERIZED. FP16 and lossy video encoding are not lossless.

No unrelated cleanup, commit or push as part of this revision.
