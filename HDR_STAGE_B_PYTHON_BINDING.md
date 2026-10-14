# HDR Stage B — production Python binding (reusable wrapper)

Scope of this substage: a reusable production Python wrapper around the
already-implemented, already-verified `native/gpu_hdr_scrgb_bridge.dll` (ABI 1).
It writes and verifies that wrapper only. No production export wiring, no
experimental UI, no NVOF/NVENC orchestration, no real HDR export, and no
final gate ran here.

## What was produced

- `hdr_gpu_engine.py` — the wrapper:
  - ctypes mirrors of the current header: `GpuHdrTapTable`, `GpuHdrContract`,
    the source/proxy/reconstruct/encode reports and `GpuHdrTransferCounters`,
    with explicit prototypes for ABI, fill_contract, open, decode, downscale,
    proxy, reconstruct, encode, the four report getters, transfer counters,
    paper_white, close and last_error.
  - Lazy bundled-DLL load through `bundle_paths.native_dll` / `native_dir` and
    the existing `add_dll_directory` mechanism. Importing the module loads no
    native code and touches no hardware. Required exports and ABI 1 are
    validated before use; a mismatch or a missing symbol is a clear error, not a
    partial chain. Pixel-readback and plane-sizing entries are **not** bound, so
    the wrapper exposes no CPU image path.
  - `GpuHdrBridge` — one bridge owner. Borrows the D3D12 device and the decoder
    CUDA context and never destroys them. Owns the handle: idempotent `close`,
    context-manager cleanup, close-on-open-failure, refuses use after close, and
    refuses cross-thread stage calls (synchronous single-thread convention, no
    `__del__` native teardown). It keeps the CPU tap arrays alive for the whole
    open life. Plans come from the existing CPU reference (`Lanczos-2` down,
    `Catmull-Rom` up) sized from the supplied full/working extents; invalid plans
    and a tap window wider than `GPU_HDR_MAX_TAPS` are refused, never truncated.
  - Fixed `paper_white_scrgb` at open (finite, ≥ 1.0), read-only configured and
    effective values and the contract. Explicit `support_ceiling_scrgb` at decode
    and `clip_ceiling_scrgb` at encode, validated finite and positive, never
    derived from MaxCLL and never mutating paper white per frame. Units are the
    bridge's: scRGB 1.0 = 80 cd/m², PQ top = 125.0 scRGB.
  - Methods return typed device descriptors (dims, pitch, borrowed context, bytes
    per pixel, actual format) for native-produced pitched planes. `proxy()`
    returns TWO separate descriptors — an RGBA16F main proxy and an RGBA8
    motion-only guidance surface — using the field names
    `dlss_gpu_engine.Live.process_device_*` and NVOF accept; the RGBA8 guidance is
    never presented as FP16. `reconstruct()` takes the Feature 18 RGBA16F output
    (descriptor or raw texture + fence), validates format/extent when the
    descriptor declares them, and returns the full-size FP16 composite. `encode()`
    returns full-size CUDA P010 planes. No fallback, no reimplemented shader math.
  - Reports return only the small structs as JSON-safe dicts; counters expose
    exactly the fields this ABI carries, with no fabricated zero for something it
    does not measure.
- `bundle_paths.py` — `gpu_hdr_scrgb_bridge.dll` added to `NATIVE_BINARIES` once,
  existing entries and order preserved.
- `tests/unit/test_hdr_gpu_engine.py` — focused CPU/contract tests against an
  injected fake ABI.

## Verification (CPU/contract only)

- `python -m py_compile hdr_gpu_engine.py bundle_paths.py tests/unit/test_hdr_gpu_engine.py` → exit 0.
- Module import is side-effect free; mirror struct sizes match the x64 layout the
  compiled `struct_size` guards in `native/gpu_hdr_scrgb_bridge.cpp` require
  (TapTable 56, Contract 288, Source 64, Proxy 48, Reconstruct 80, Encode 72,
  Counters 72).
- `python -m pytest tests/unit/test_hdr_gpu_engine.py -q` → **56 passed**, exit 0.
- `python -m pytest tests/unit/test_packaged_binaries.py -q` → **12 passed**, exit 0
  (confirms the manifest addition is consistent with the packaging spec).

No HDR export, no Feature 18 probe, no GPU image readback and no previously-passing
native gate was rerun.

## Still absent (not done in this substage)

- **Production export wiring** — decoder P010 handoff, NVOF orchestration over
  the RGBA8 guidance, Feature 18 RGBA16F session driving, and NVENC P010 encode
  are not connected to this wrapper yet.
- **Experimental UI** — no scale admission, controls or delivery path.
- **Final 4–6 frame gate** — the real end-to-end correctness/perf gate has not
  been run or produced.

This substage does **not** claim HDR is complete or lossless. Performance through
this layer is NOT_CHARACTERIZED.
