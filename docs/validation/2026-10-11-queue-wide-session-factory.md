# Wide session creation through the export service

## Failure

The export service replaced the NR constructor with a three-argument lambda for
DLAA/VSR or a mixed render order. Its SDR10/HDR branch still passed the NR-only
keyword `color_format='rgba16f'`. The lambda rejected that keyword before opening
the GPU host. Merely adding the argument without forwarding precision would have
opened an incorrect narrow session.

`DLSSService._create_export_live` now translates the precision request to the
factory actually used: `color_format` for pure NR, `wide` for native effects.
The production `_execute_job` invokes it within the existing timeout and lock.
Plain SDR NR retains its existing default constructor call.

## Verification

- 121 related unit tests passed. Factory regression cases cover all 15 unique
  nonempty NR/DLAA/VSR orders and all five legacy modes, in narrow and wide modes.
- Extracted four compressed video frames and audio from the user's untagged
  Kurumi HEVC MP4 into an ignored test fixture, leaving the source untouched.
- Ran `tests/helpers/sdr10_service_gate.py` through `DLSSService.add_job` and the
  production `_execute_job`, with `DLAA → DLSSNR → VSR`, VSR 2×, HDR disabled.
- Four frames completed; primary session RGBA16F; DLAA and VSR each reported four
  successful calls. VSR used the VFX 1.3 RGB10A2 boundary. Output was 3840×2160,
  HEVC Main10, yuv420p10le, limited-range BT.709, with four decoded output frames
  and five source audio packet payloads copied unchanged.
- Measured export image host boundaries were zero. Scalar diagnostic readback is
  accounted for separately; this is not a claim of zero total host traffic.

Evidence: ignored `bench_out/queue_factory_20261011/dlaa_nr_vsr/` contains job,
diagnostics, run, accepted summary, output stream/packet probe and output MKV.
This smoke gate does not characterize sustained performance or all hardware
rendering permutations. Performance=NOT_CHARACTERIZED.
