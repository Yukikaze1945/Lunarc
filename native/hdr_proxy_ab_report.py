"""Derive the stage-A evidence tables from the recorded A/B artifact.

Read-only: it re-derives nothing from the GPU, only from the numbers the gate
already wrote, and recomputes the per-channel reconstruction dispositions
(``applied``, ``rejected``, ``cap``) that the artifact stores collapsed to one
flag per sample-frame row. Run it after the gate to print the same tables the
evidence summary quotes.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlss_gpu_hdr_proxy_ab as rig
import hdr_proxy_math as proxy

ARTIFACT = Path(__file__).resolve().parent.parent / \
    "build" / "fp16_probe" / "hdr_proxy_ab_artifact.json"


def dispositions(sample, frame, anchor):
    """Per-channel disposition of the reconstruction for one recorded frame."""
    master = np.asarray(sample["master"], dtype=np.float32)
    neural = np.asarray(frame["output_center"]["median"], dtype=np.float32)
    input_proxy = np.asarray(anchor, dtype=np.float32)
    row = sample["reconstruction"]["frame_3"]
    counts = Counter()
    worst = (0.0, None)
    for method, record in row.items():
        hdr = np.asarray(record["hdr"], dtype=np.float32)
        stops = record["stops"]
        for channel in range(hdr.size):
            delta = float(neural[min(channel, neural.size - 1)]
                          - input_proxy[min(channel, input_proxy.size - 1)]) \
                if method == proxy.REC_LIN else 0.0
            stop = float(stops[0] if len(stops) == 1 else stops[channel])
            if stop == 0.0 and delta != 0.0:
                counts["rejected"] += 1
            elif abs(stop) >= proxy.MAX_STOPS:
                counts["cap"] += 1
            elif delta != 0.0:
                counts["applied"] += 1
            else:
                counts["exact_pass_through"] += 1
            deviation = abs(float(hdr[channel]) / max(float(master[channel]),
                                                      1e-9) - 1.0)
            if deviation > worst[0]:
                worst = (deviation, (method, channel,
                                     float(master[channel]),
                                     float(hdr[channel]), stop))
    return counts, worst


def main() -> int:
    if not ARTIFACT.exists():
        print(f"missing artifact: {ARTIFACT}", file=sys.stderr)
        return 2
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    print(f"probe dll {artifact['probe_dll_sha256']}")
    print(f"host source {artifact['host_source_sha256']}")
    print(f"proxy module {artifact['proxy_module_sha256']}")
    for paper, session in sorted(artifact["sessions"].items(),
                                 key=lambda item: float(item[0])):
        arms = session["arms"]
        raw, bounded = arms["A_raw_linear"]["samples"], \
            arms["B_bounded_proxy"]["samples"]
        print(f"\n=== paperWhite {paper}, session {session['session_id']}, "
              f"proxy saturates at scene {session['proxy_saturation_scene']:.4f} ===")
        print("level        master   raw_out  proxy_code   proxy_out  "
              "band      recon_stops   frame_3_dispositions   worst_dev")
        for name, sample in sorted(bounded.items()):
            steady = sample["summary"]["steady_median_luma"]
            code = sample["proxy_code"]
            frame = sample["frames"][-1]
            counts, worst = dispositions(sample, frame, frame["pre_ngx_half"])
            stops = sample["reconstruction"]["frame_3"][proxy.REC_LIN]["stops"]
            print("%-12s %-8s %-8s %-12s %-10s %-8s %-13s %-22s %.3f%% %s" % (
                name,
                "%.4g" % sample["master"][0],
                "%.5f" % raw[name]["summary"]["steady_median_luma"],
                " ".join("%.5f" % c for c in code[:3]),
                "%.5f" % steady,
                "%.1e" % sample["summary"]["across_frame_band"],
                " ".join("%+.4f" % s for s in stops),
                dict(counts),
                100.0 * worst[0], worst[1]))
        gate = artifact["gate"][paper]
        print("highlight clusters: raw %d, proxy %d (distinct input codes %d)"
              % (gate["raw_highlight_clusters"], gate["proxy_highlight_clusters"],
                 gate["proxy_highlight_distinct_input_codes"]))
        print("all-sample distinct input codes: %d of %d"
              % (gate["proxy_distinct_input_codes_all_samples"],
                 gate["proxy_all_samples_total"]))
        print("grid: %s" % json.dumps(session["grid"], sort_keys=True))
        print("identity: %s" % json.dumps(session["identity"], sort_keys=True))
        print("pass: %s" % gate["pass"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
