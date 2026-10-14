# Untagged SDR10 preview routing

## Cause and change

The supplied 1920×1080 HEVC source reports `yuv420p10le` and limited range,
but does not declare matrix, transfer or primaries. Preview previously used the
fully tagged BT.709 admission predicate to choose precision. A missing colour
tag therefore sent a P010 picture to the NV12-only SDR8 gate.

Precision now selects the preview decoder path separately from colour admission.
The shared source interpretation resolves missing colour tags for HD 10-bit
4:2:0 sources with explicit limited range to BT.709 SDR, recording the original
declarations and each assumed field. This is an interpretation, not proof of the
source's intended colour. Explicit conflicting colour tags, HDR side data, full
or unknown range, SD sources and other pixel formats are not covered by this
default. Decoded frame enums are resolved only when UNSPECIFIED; explicit frame
declarations still reach the native colour gate unchanged.

Service input probing, preview and GPU Transfer export share the interpretation.
Preview diagnostics and the export SDR10 policy retain `source_colour_resolution`.
Input files are unchanged; encoded output states the effective BT.709 colour.

## Evidence

- 254 related unit tests passed, including the missing-tag policy, explicit
  declaration protection, HDR classification, SDR10 gate, preview lifecycle and
  preview/export contracts.
- Supplied Kurumi MP4: two actual GPU preview frames, source 1920×1080,
  `colour_mode=sdr10_bt709`, `processing_format=rgba16f`, HDR disabled;
  defaulted fields and original null declarations recorded.
- Supplied Kurumi MP4: two-frame GPU Transfer export passed, HEVC Main10,
  `yuv420p10le`, limited-range BT.709, two decoded output frames with display PTS
  0/42 ms and four copied source audio packets. Measured export image host
  boundaries stayed zero; scalar diagnostic traffic is separate.
- Previously supplied DREAMY COLOR MKV with complete BT.709 tags: two actual
  GPU preview frames passed before the change, identifying missing tags as the
  distinguishing condition.

Raw evidence is in ignored `bench_out/preview_precision_20261011/`: `dreamy.json`,
`kurumi.json`, and `kurumi_export/{diagnostics,result}.json` / `output.mkv`.
The smoke tests do not measure sustained preview performance or establish
colour accuracy for arbitrarily untagged material. Performance=NOT_CHARACTERIZED.
