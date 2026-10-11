# -*- coding: utf-8 -*-
"""Default GPU-host routing gate: one short real run through the DLSSService on the
legacy FFmpeg pipe, guidance off and on, with the old host blocked, the entry the
frame actually took observed, and the deliverable checked with ffprobe.

Not a pytest test: it needs the GPU, NVOF and an encoder. The default DLSS host is
now the project GPU host, so a service job that names no ``dlss_backend`` at all
must be answered by native/dlss_gpu_host.dll and must take the entry that matches
what its flow provider can actually deliver:

    guidance off (0)        -> process_zero_motion   (resident zero-motion texture)
    guidance on, DIS         -> process_host_motion   (CPU float32x2 plane uploaded)
    guidance on, NVOF        -> process_device_flow   (descriptor that never left the card)

The old host is made to raise if anything reaches for it, so a wrong default is a
loud failure, not a silent run on dlssnr_host.dll.

    python tests/unit/default_gpu_routing_gate.py
"""
import json
import os
import subprocess
import sys
import types

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import dlss_service  # noqa: E402
import pipeline  # noqa: E402

FIXTURE = os.path.join(_ROOT, "tests", "fixtures", "fixture_120fps.mkv")
OUT_DIR = os.path.join(_ROOT, "bench_out")
SHORT = os.path.join(OUT_DIR, "default_gate_src.mp4")


class OldHostLoaded(AssertionError):
    """Raised instead of constructing dlssnr_host.dll, to prove the default is not it."""


def _block_old_host():
    """Guard the retired wrapper by name, not by importing it.

    dlss_engine.py was deleted with the host it loaded, so the tripwire installs a
    module that raises on construction: if any code path still reaches for the old
    wrapper, the gate says so instead of dying on an ImportError.
    """
    blocked = types.ModuleType("dlss_engine")

    def _boom(*args, **kwargs):
        raise OldHostLoaded(
            "a default job (no dlss_backend named) reached for dlssnr_host.dll; the GPU "
            "host is supposed to be the only one")

    blocked.Live = _boom
    sys.modules["dlss_engine"] = blocked


def _make_short_source():
    """The service exports the whole clip, so cut 8 frames to keep the gate short."""
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(SHORT):
        subprocess.run([pipeline._bundle_ffmpeg(), "-y", "-v", "error", "-i", FIXTURE,
                        "-frames:v", "8", "-c:v", "libx264", "-pix_fmt", "yuv420p", SHORT],
                       check=True)
    return SHORT


def _ffprobe_facts(path):
    dump = json.loads(subprocess.check_output([
        pipeline._bundle_ffprobe(), "-v", "error", "-print_format", "json",
        "-show_streams", "-count_packets", "-select_streams", "v:0",
        path]).decode("utf-8", "replace"))
    stream = dump["streams"][0]
    return {"codec": stream.get("codec_name"),
            "size": f"{stream.get('width')}x{stream.get('height')}",
            "pix_fmt": stream.get("pix_fmt"),
            "packets": int(stream.get("nb_read_packets") or 0)}


def _run_service_export(name, flow_settings, out):
    """One default-host legacy-pipe export through the real service. No dlss_backend.

    The settings a pre-migration client sends: the service's own defaults with the host
    key removed, so nothing in the job names a host. Spreading service.settings in whole
    would have written the key into the job and proved only the obvious.
    """
    service = dlss_service.DLSSService()
    service.load_video(SHORT)
    settings = {"guidance_mode": 0, "multi_pass": 1, "working_scale": 1.0,
                "residual_pyramid": False, "residual_backend": "python",
                **flow_settings}
    submitted = {k: v for k, v in service.settings.items() if k != "dlss_backend"}
    submitted.update(settings)
    if os.path.exists(out):
        os.unlink(out)
    job = service.add_job({
        "name": name, "input_path": SHORT, "output_path": out,
        "cq": 24, "with_audio": False, "codec": "libx264", "container": "mp4",
        "transcode_settings": {"codec": "libx264", "preset": "fast", "container": "mp4",
                               "audio_mode": "remove", "decode_mode": "cpu",
                               "b_frames": "0", "lookahead": "0"},
        "settings": submitted,
    })
    result = {"name": name, "host_key_absent": "dlss_backend" not in submitted,
              "resolved_host": job["settings"].get("dlss_backend")}
    try:
        service._execute_job(job, None)
        result["status"] = "completed"
    except Exception as e:  # a wrong routing surfaces here, not as a silent fallback
        result["status"] = "failed"
        result["error"] = f"{type(e).__name__}: {e}"
    diag = job.get("residual_backend_diagnostics", {}) or {}
    result.update({
        "dlss_backend_actual": diag.get("dlss_backend_actual"),
        "dlss_entry": diag.get("dlss_entry"),
        "flow_transport": diag.get("flow_transport"),
        "export_backend_actual": diag.get("export_backend_actual"),
        "file_exists": os.path.exists(out),
        "file_size": os.path.getsize(out) if os.path.exists(out) else 0,
    })
    if os.path.exists(out):
        try:
            result["ffprobe"] = _ffprobe_facts(out)
        except Exception as e:
            result["ffprobe_error"] = str(e)
    return result


def _run_service_preview():
    """guidance off and on through the service preview, old host still blocked."""
    import numpy as np
    import cv2
    service = dlss_service.DLSSService()
    service.load_video(SHORT)
    service.update_settings({"guidance_mode": 0})
    kind_off = service._preview_live_kind()
    jpeg_off, status_off = service.render_preview(0, view="dlss", preview_mode=True)
    arr_off = cv2.imdecode(np.frombuffer(jpeg_off, np.uint8), cv2.IMREAD_COLOR) if jpeg_off else None
    service.update_settings({"guidance_mode": 1, "flow_engine": "dis"})
    kind_on = service._preview_live_kind()
    jpeg_on, status_on = service.render_preview(1, view="dlss", preview_mode=True)
    arr_on = cv2.imdecode(np.frombuffer(jpeg_on, np.uint8), cv2.IMREAD_COLOR) if jpeg_on else None
    return {
        "kind_off": kind_off, "off_status": status_off,
        "off_jpeg_bytes": len(jpeg_off or b""),
        "off_shape": None if arr_off is None else list(arr_off.shape[:2]),
        "kind_on": kind_on, "on_status": status_on,
        "on_jpeg_bytes": len(jpeg_on or b""),
        "on_shape": None if arr_on is None else list(arr_on.shape[:2]),
        "preview_error": getattr(service, "_preview_live_error", None),
    }


def _check(cond, msg, report):
    if not cond:
        report.setdefault("failures", []).append(msg)
        print(f"  FAIL  {msg}")
    else:
        print(f"  ok    {msg}")


def main():
    _block_old_host()
    _make_short_source()
    n, fps, w, h = pipeline.video_info(SHORT)
    print(f"[gate] short source {SHORT}: {n} frames @ {fps} {w}x{h}")
    overall = {"failures": []}

    print("\n[gate] service legacy-pipe export, DEFAULT host (no dlss_backend named)")
    cases = [
        ("guidance_off_zero_motion", {"guidance_mode": 0},
         os.path.join(OUT_DIR, "default_gate_off.mkv"), "process_zero_motion",
         "resident_zero_texture"),
        ("guidance_on_dis_host_motion", {"guidance_mode": 1, "flow_engine": "dis",
                                         "flow_quality": "1F"},
         os.path.join(OUT_DIR, "default_gate_dis.mkv"), "process_host_motion",
         "host_plane_upload"),
        ("guidance_on_nvof_device_flow", {"guidance_mode": 1, "flow_engine": "nvof",
                                          "flow_quality": "1F"},
         os.path.join(OUT_DIR, "default_gate_nvof.mkv"), "process_device_flow",
         "nvof_device_descriptor"),
    ]
    for name, flow_settings, out, want_entry, want_transport in cases:
        res = _run_service_export(name, flow_settings, out)
        print(f"\n  {name}: {json.dumps(res, ensure_ascii=False)}")
        _check(res["host_key_absent"], f"{name}: the job settings named no DLSS host", overall)
        _check(res["resolved_host"] == "gpu_flow_d3d12",
               f"{name}: an unnamed host resolved to gpu_flow_d3d12 "
               f"({res['resolved_host']})", overall)
        _check(res["status"] == "completed", f"{name}: job completed ({res.get('error')})",
               overall)
        _check(res.get("dlss_backend_actual") == "gpu_flow_d3d12",
               f"{name}: ran on the GPU host (not dlssnr_host.dll)", overall)
        _check(res.get("dlss_entry") == want_entry,
               f"{name}: entry {res.get('dlss_entry')} == {want_entry}", overall)
        _check(res.get("flow_transport") == want_transport,
               f"{name}: transport {res.get('flow_transport')} == {want_transport}", overall)
        _check(res.get("export_backend_actual") == "legacy_pipe",
               f"{name}: legacy pipe chain ({res.get('export_backend_actual')})", overall)
        _check(res.get("file_exists") and res.get("file_size", 0) > 1024,
               f"{name}: non-trivial deliverable ({res.get('file_size')} bytes)", overall)
        _check((res.get("ffprobe") or {}).get("packets", 0) >= 1,
               f"{name}: ffprobe sees video packets", overall)

    print("\n[gate] service preview on the DEFAULT host (old host blocked)")
    prev = _run_service_preview()
    print(f"  {json.dumps(prev, ensure_ascii=False)}")
    _check(prev["kind_off"] == "gpu_zero_motion", "preview guidance off -> gpu_zero_motion",
           overall)
    _check(prev["kind_on"] == "gpu_host_motion", "preview guidance on (dis) -> gpu_host_motion",
           overall)
    _check(prev["off_jpeg_bytes"] > 1024 and prev["off_shape"], "preview off decoded a JPEG frame",
           overall)
    _check(prev["on_jpeg_bytes"] > 1024 and prev["on_shape"], "preview on decoded a JPEG frame",
           overall)
    _check(not prev["preview_error"], f"preview produced no host error ({prev['preview_error']})",
           overall)

    print("\n[gate] RESULT:",
          "PASS" if not overall["failures"] else f"FAIL ({len(overall['failures'])})")
    return 0 if not overall["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
