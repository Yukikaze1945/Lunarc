"""Test-only stubs and helpers for the guidance contracts.

StubLive stands in for the D3D12 DLSS session, which needs a GPU and a vendor DLL.
StubFlowService stands in for the optical flow engine, same reason.
"""

import os

import numpy as np

import pipeline

SRC_W, SRC_H = 64, 48
LIVE_W, LIVE_H = 32, 24


class StubLive:
    """Stand-in for the project GPU host: its three entries, recorded.

    The pipeline drives a session through exactly ``process_zero_motion``,
    ``process_device_flow`` or ``process_host_motion`` and picks between them from
    what the session reports, so a stub with only one of them would be a stub of a
    host this build cannot open. ``host_dll``/``host_abi`` are the provenance the
    run record carries.
    """

    def __init__(self, w, h):
        self._w, self._h = w, h
        self.seen_flow = []
        self.seen_rgba = []
        self.process_count = 0
        self.entries = []
        self.host_dll = "dlss_gpu_host.dll"
        self.host_abi = 3

    def supports_zero_motion(self):
        return True

    def supports_host_motion(self):
        return True

    def _draw(self, rgba, reset, out):
        self.process_count += 1
        self.seen_rgba.append(np.array(rgba))
        dst = out if out is not None else np.empty(rgba.shape, dtype=np.uint8)
        dst[:, :, :3] = 130
        dst[:, :, 3] = 255
        return dst

    def process_zero_motion(self, rgba, reset=False, out=None):
        self.entries.append("process_zero_motion")
        return self._draw(rgba, reset, out)

    def process_device_flow(self, rgba, flow, reset=False, out=None):
        self.entries.append("process_device_flow")
        # Recorded as handed: a device flow is a CUDA descriptor, and flattening it to
        # an array here would hide the very thing the entry is chosen for.
        self.seen_flow.append(flow)
        return self._draw(rgba, reset, out)

    def process_host_motion(self, rgba, flow, reset=False, out=None):
        self.entries.append("process_host_motion")
        self.seen_flow.append(np.array(flow, dtype=np.float32))
        return self._draw(rgba, reset, out)


class StubFlowService:
    """Only replaces the flow engine.

    `nvof_rgba_input_only` mirrors the real service's answer to "can your bridge take the
    BGR plane directly". The default is the shipped bridge's answer, and a test that wants
    the old-bridge case says so instead of getting it by omission.

    `supports_device_bgr_flow` is the other half of that answer: whether the vectors can
    be left on the card at all. The default is a CPU engine (DIS/RAFT), because that is
    what every contract here was written against; a test for the NVOF device descriptor
    says so explicitly.
    """

    def __init__(self, value=3.0, nvof_rgba_input_only=False, device_flow=False):
        self.value = value
        self.calls = 0
        self.input_formats = []
        self.nvof_rgba_input_only = bool(nvof_rgba_input_only)
        self.device_flow = bool(device_flow)

    def supports_device_bgr_flow(self):
        return self.device_flow

    def get_guidance(self, frame_id, cur_bgr, prev_bgr, target_w, target_h,
                     guidance_mode, is_discontinuous, out_flow=None,
                     input_format="rgba8_host"):
        self.calls += 1
        # Recorded rather than dropped: which input the production path asked for is what
        # separates the two transfer modes, and a stub that ignored it would let a caller
        # describe itself as reduced without ever having requested it.
        self.input_formats.append(input_format)
        flow = out_flow if out_flow is not None else np.zeros((target_h, target_w, 2), np.float32)
        flow[...] = self.value
        return flow, {"guidance_backend": "stub_flow"}


def make_core(live_w=LIVE_W, live_h=LIVE_H, src_w=SRC_W, src_h=SRC_H, guidance_mode=0,
              flow_service=None, passes=None, job_settings_extra=None,
              flow_provider=None):
    settings = {"guidance_mode": guidance_mode, "working_scale": live_w / float(src_w),
                "residual_pyramid": True, "residual_strength": 1.0}
    settings.update(job_settings_extra or {})
    if flow_provider is None and flow_service is not None:
        flow_provider = pipeline.GuidanceServiceFlowProvider(flow_service,
                                                             guidance_mode=guidance_mode)
    return pipeline.ProductionExportCore(
        src_w=src_w, src_h=src_h, dst_w=src_w, dst_h=src_h,
        passes=passes if passes is not None else [StubLive(live_w, live_h)],
        job_settings=settings,
        flow_provider=flow_provider,
    )


def source_frame(src_w=SRC_W, src_h=SRC_H):
    return np.full((src_h, src_w, 3), 90, dtype=np.uint8)
