"""Twelve-frame device-flow export gate for the opt-in GPU-flow pipe.

The peer it used to be measured against -- the retired dlssnr_host.dll session --
is gone: the export core now refuses any session but native/dlss_gpu_host.dll, so
this gate asserts the device chain on its own (flow descriptor stays on the card,
zero device-to-host copies, 12 packets out).
"""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dlss_gpu_engine  # noqa: E402
import pipeline  # noqa: E402
import transfer_profiler  # noqa: E402
from optical_flow import FrameGuidanceService  # noqa: E402


def stream_hash(path):
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
               "-c", "copy", "-f", "hash", "-hash", "SHA256", "-"]
    return subprocess.check_output(command, text=True, timeout=30).strip()


def packet_count(path):
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-count_packets", "-show_entries", "stream=nb_read_packets",
               "-of", "csv=p=0", str(path)]
    return int(subprocess.check_output(command, text=True, timeout=30).strip())


def run():
    source = ROOT / "bench_out" / "fixture_4k_300.mp4"
    suffix = os.environ.get("GPU_FLOW_SMOKE_SUFFIX", "")
    output = ROOT / "bench_out" / f"gpu_flow_device_12{suffix}.mkv"
    settings = {"preset": 1, "style": 0, "intensity": 1.0,
                "local_tone": 1.0, "local_struct": 1.0, "skin_struct": 1.0,
                "use_auto_mask": 1, "ui_correction": 0, "guidance_mode": 1,
                "working_scale": 0.5, "residual_pyramid": True,
                "flow_engine": "nvof", "flow_quality": "4M",
                "transfer_mode": "reduced"}
    live = dlss_gpu_engine.Live(1920, 1080, settings)
    service = FrameGuidanceService("nvof", "4M")
    diagnostics = {}
    collector = transfer_profiler.TransferProfiler()
    try:
        pipeline.export_dlss_stream(
            video=str(source), live=live, frames=12, fps=60.0,
            cq=24, with_audio=False,
            flow_provider=pipeline.GuidanceServiceFlowProvider(service, guidance_mode=1),
            out_path=str(output), job_settings=settings,
            diagnostics_out=diagnostics, transfer_collector=collector)
    finally:
        live.close()
        service.close()
    count = packet_count(output)
    if count != 12:
        raise AssertionError(f"device exported {count} packets, expected 12")
    if diagnostics.get("dlss_backend_actual") != "gpu_flow_d3d12":
        raise AssertionError("GPU-flow backend was not selected")

    # The absence of a download only means something about a frame that ran guidance,
    # so each record has to say which entry it took and that NVOF executed. Without
    # this the gate would also pass on a run that silently served zero motion, where
    # no flow was ever produced to download.
    if len(collector.records) != 12:
        raise AssertionError(f"{len(collector.records)} transfer records, expected 12")
    for index, record in enumerate(collector.records):
        if record.get("flow_transport") != "nvof_device_descriptor":
            raise AssertionError(
                f"frame {index}: flow transport {record.get('flow_transport')!r}, "
                f"expected the descriptor that never left the card")
        if record.get("dlss_entry") != "process_device_flow":
            raise AssertionError(
                f"frame {index}: DLSS entry {record.get('dlss_entry')!r}, expected "
                f"process_device_flow")
        if record.get("nvof_execute_count_delta") != 1:
            raise AssertionError(
                f"frame {index}: NVOF executed {record.get('nvof_execute_count_delta')!r} "
                f"times, expected exactly one flow compute per frame")
        # None here is the honest answer from a session that read no bridge counters;
        # only a measured zero is evidence that nothing came down.
        if record.get("nvof_d2h_copy_count_delta") != 0 or \
                record.get("nvof_d2h_bytes_delta") != 0:
            raise AssertionError(
                f"frame {index}: NVOF flow was downloaded to host "
                f"(copies={record.get('nvof_d2h_copy_count_delta')!r} "
                f"bytes={record.get('nvof_d2h_bytes_delta')!r})")
    return stream_hash(output), diagnostics


def main():
    device_hash, device_diag = run()
    print("device_stream", device_hash)
    print("device_mode", device_diag.get("transfer_mode_actual"))
    print("ab_against_retired_host=removed "
          "(pipeline refuses any session but native/dlss_gpu_host.dll)")
    print("12_frame_device_export=PASS")


if __name__ == "__main__":
    main()
