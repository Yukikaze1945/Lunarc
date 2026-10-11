"""Published integration contracts, distinct from measured provider capability."""


def provider_contracts():
    return {
        "nvof": {
            "implementation": "available_native",
            "direction": "current_to_previous",
            "units": "input_pixels",
            "native_vector": "signed S10.5",
            "requested_grids": [1, 2, 4],
            "consumer_vector": "dense float32 x,y in input pixel units",
            "confidence": None,
            "scene_change": "explicit reset from caller; no native scene-change output",
            "hdr_input": "independent RGBA8 motion-only guidance, not HDR master",
            "history": "independent persistent surfaces per layer",
            "completion": "execute and device densification synchronize before return",
            "quality_comparison": "NOT_CHARACTERIZED",
        },
        "fidelityfx": {
            "implementation": "available_native",
            "scope": "GPU Transfer and native preview; device/session admission checked per job",
            "sdk_release": "1.1.4",
            "effect_version": "1.1.2",
            "native_grid": [8, 8],
            "native_vector": "R16G16_SINT, current_to_previous, input_pixels",
            "direction": "current_to_previous",
            "units": "input_pixels",
            "consumer_vector": "dense CUDA float32 x,y, expanded from 8x8 blocks",
            "scene_change": "3x1 R32_UINT; history mask and cumulative change count",
            "history": "independent SDK context per layer",
            "startup_suppression_frames": 6,
            "scene_history_suppression_frames": 4,
            "confidence": None,
            "hdr_input": "independent RGBA8 motion-only guidance, not FP16 HDR master",
            "license": "MIT; pinned SDK sources and notices in native/vendor/fidelityfx",
            "hardware_test": "2026-10-10 short shifted-frame and independent-layer gates; not a device capability guarantee",
            "quality_comparison": "NOT_CHARACTERIZED",
            "reference": "https://gpuopen.com/manuals/fidelityfx_sdk/techniques/optical-flow/",
        },
    }
