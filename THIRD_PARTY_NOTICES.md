# Third-party notices

Lunarc's project license is **GPL-3.0-or-later**. This designation does not replace the licenses of the following components. Preserve their notices when distributing source or authorized builds.

## Source included in this repository

| Component | Location | License / source |
| --- | --- | --- |
| Legacy project-origin code | `LICENSES/Lunarc-legacy-MIT.txt` | Original MIT notice retained; the current project as a whole is GPL-3.0-or-later. |
| AMD FidelityFX SDK 1.1.4 | `native/vendor/fidelityfx` | AMD MIT; full notice in `LICENSE.txt`. Pinned revision and local build adaptations in `docs/AMD_FLOW_DEPENDENCIES.md`. [Upstream](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK). |
| NVIDIA VFX ABI declarations | `native/vendor/nvidia-vfx-abi` | NVIDIA MIT; full notice in `LICENSE`. These declarations do not contain the proprietary VFX implementation. |
| NVIDIA Optical Flow API headers | `native/include/nvOpticalFlowCommon.h`, `nvOpticalFlowCuda.h` | NVIDIA's embedded three-clause BSD-style source notice, retained in each header. |
| Minimal CUDA driver declarations | `native/include/cuda.h` | Project-written compatibility declarations; no CUDA SDK implementation included. |

The residual and HDR/scRGB implementation's reference/provenance notes are retained in `MAGPIE_RESIDUAL_PROVENANCE.md` and `MAGPIE_HDR_SCRGB_REFERENCE.md`. Referencing another project's behavior does not grant permission to copy its code or relicense its work.

## Dependencies installed separately

- **NVIDIA DLSS / NGX / Maxine VFX / CUDA**: implementations and runtime binaries are not in this source release. Their respective NVIDIA agreements apply. The native DLSS host build uses an external NGX static library; this project does not grant a linking exception or authorize distribution of a combined proprietary build. See [Video Codec SDK license](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/license/index.html), [CUDA EULA](https://docs.nvidia.com/cuda/eula/) and [VFX SDK documentation](https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-system-guide/index.html), plus the agreements supplied with your SDKs.
- **FFmpeg**: license depends on the configured build (LGPL / GPL and optional nonfree components). The project invokes FFmpeg and links external FFmpeg libraries for native decode. The small installer contains shared FFmpeg libraries needed by native bridges; CLI FFmpeg is downloaded separately. Preserve the exact build's notices and corresponding-source obligations when distributing an authorized package. [FFmpeg legal information](https://ffmpeg.org/legal.html).
- **PyTorch / torchvision / RAFT-Large weights**: installed or downloaded separately. See [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE) and [torchvision license](https://github.com/pytorch/vision/blob/main/LICENSE); inspect the terms applicable to the particular weights. Model files are not included in this public snapshot.
- **react-range 1.10.0**: MIT, copyright 2019-present Vojtech Miksu. Used for frame-range controls. Full notice retained in `LICENSES/react-range-MIT.txt`. [Upstream](https://github.com/tajo/react-range).
- **Electron, React, Vite, TypeScript and other Node/Python packages**: recorded in `frontend/package-lock.json` and `requirements.txt`; preserve the license notices provided by their exact installed versions. Source publication does not include `node_modules`, a Python environment or these packages' compiled runtimes.
- **FidelityFX_SC and build tools**: obtained separately from their official distributions; the vendor source tree is not a compiler-binary bundle.

The Windows 1.0.0-beta component installer includes Electron, pinned uv 0.13.0 (MIT/Apache notices in LICENSES), project bridges and their shared native dependencies. Shared FFmpeg/Conda libraries retain exact package notices, recipes and source URLs under resources/native-deps/licenses; their inventory records versions and official package checksums. CLI FFmpeg, managed CPython and Python packages are downloaded separately. PyTorch/CUDA and RAFT weights are optional downloads. Proprietary model runtimes are not bundled. No proprietary linking exception is added.

The Git repository contains source and tests; installer assets are published separately in Releases. `packaging/build_small_resources.py` resolves dependencies and retains notices; it does not grant additional redistribution rights. Exact upstream licensing and corresponding-source requirements continue to apply.
