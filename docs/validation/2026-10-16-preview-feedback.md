# Preview transitions and GPU feedback — 2026-10-16

## Changes

- Starting rendering retains the already decoded source underneath the native display. New native windows stay hidden until their first complete picture has been drawn.
- Rebuilding a processing chain freezes the last presented swapchain. Its full-resolution input/output buffers and their CUDA imports are released before their borrowed model/decoder context closes. Only one prior frozen display is retained; it is closed after a replacement presenter finishes its first frame. Explicit preview-memory release and shutdown close the frozen display too.
- Frozen displays do not redraw or resize their swapchain. A layout change can move/stretch the retained window while the replacement is prepared. Interactive comparison resumes with the replacement picture.
- Sent first-frame messages can overtake posted layout messages. The frame handler applies outstanding geometry and synchronizes swapchain dimensions before drawing and revealing a picture.
- Following user feedback, loading is represented by the existing sidebar aperture icon; there is no extra progress strip, helper copy or elapsed-time label. White rotation alternates speed for loading, stationary red indicates an error, red rotation indicates backend recovery, and slow green rotation indicates an active export queue. Existing range prerendering continues reporting its actual percentage.
- Explicit allocation errors share a short GPU-memory message; device-removal/reset errors use a separate message. Preview/image/segment errors expose original details in a disclosure, export diagnostics retain their original traceback and error, and the export list/status bar use the same summaries.
- The footer shows adapter VRAM used/total and a meter. At 80% it becomes amber, at 92% red; missing data is shown as unavailable. Process-budget usage has a separate meter in diagnostics. These are sampled usage values, not predictions of model peak memory.

## Evidence

- Frontend TypeScript/Vite/Electron builds passed; existing large-bundle warning remains.
- Processing-feedback and localization Node checks passed (10 checks); later focused feedback check passed (3 checks).
- Cleanup, preview-memory, native-preview and image-cache Python regression checks passed (25 checks); the added freeze-before-context-close check passed in the native-preview file (6 checks including existing cases).
- Presenter shader guard regression passed (2 checks).
- Browser preview smoke passed with a delayed first native frame: original canvas remains mounted, the aperture enters loading state without a strip, 15/16 GB produces a red meter, an active queue uses exporting state, and an injected CUDA OOM shows the common summary with original text under technical details. Recovery and blocked events drive the corresponding red aperture states. The browser smoke opens no GPU pipeline.
- Rebuilt `gpu_native_presenter.dll`. A 257×145 RGBA8 / 514×290 FP16 display check passed: upload, freeze, release old display/context resources, reconfigure frozen panel, create and present a replacement, retire the old presenter. No model inference or export stress run was performed.

This verifies the bounded display lifecycle and UI contracts. It does not establish a throughput improvement or promise continuity across an actual GPU device reset.
