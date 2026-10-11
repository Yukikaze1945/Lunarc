"""`gui.py --gpu-transfer-selftest` checked without a card.

The frozen bundle's only way onto the device chain is this probe, and it is worth
exactly what its verdict is worth: a probe that reported OK after a frame crossed
back to host memory, or after the chain had quietly fallen back to the FFmpeg pipe,
would be worse than the gap it closed. So these tests are about the verdict. The
fake exporter answers with the fields the production chain writes into
`diagnostics_out` and `transfer_collector`, named and shaped the way
`gpu_transfer_export.export_gpu_transfer` writes them, and each test asks what the
probe did with an answer it should not have accepted.

What is pinned: the probe asks the production entry for `export_backend=gpu_transfer`
on this build's one DLSS host and closes the session it opened; the frame count stays
inside the bounded window the probe exists to keep; a host boundary, a fallback, a
short deliverable or a returned path that is not its own output is not a pass; a
refusal is written down with the chain's reason; the fixture is generated with the
FFmpeg this layout resolved, into a directory outside it; and the flag is dispatched
before any Tk root exists.

The working scale is the one setting this probe can move, so it carries its own
half: an out-of-set ratio and a non-number are refusions that open no device
session, and each of the three chain shapes the gate serves is held to the branch
fields it answers with — the extents, the resampling name, the residual backend,
and on the 1.0 identity chain no number at all for the two stages it never opened.
Those expectations are written out below per scale rather than recomputed from the
probe's own helper, which is what makes them a check on it.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile

import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gpu_transfer_export
import gui
import optical_flow
import pipeline

SRC_W, SRC_H, FRAMES = 1280, 720, 6
DELIVERABLE_BYTES = 4096
# The chain shape the production gate serves each of these ratios at, at 1280x720,
# copied from the source-tree gates' own assertions rather than from gui.py:
# gpu_transfer_export.negotiate classifies it, the report names it, and this probe
# is only allowed to agree. 0.25 is here because on these extents it resolves to
# 320x176 and needs a 17-tap Lanczos window - wider than the kernel walked before
# it was widened, which is why the tap capacity is asserted here as well.
BRANCHES = {
    0.5: {"dims": [640, 360], "resampling": "exact_two_x", "downscaled": True,
          "residual": "gpu_residual_bridge_d3d12"},
    0.25: {"dims": [320, 176], "resampling": "plan_driven", "downscaled": True,
           "residual": "gpu_residual_bridge_d3d12"},
    0.26: {"dims": [336, 184], "resampling": "plan_driven", "downscaled": True,
           "residual": "gpu_residual_bridge_d3d12"},
    0.30: {"dims": [384, 216], "resampling": "plan_driven", "downscaled": True,
           "residual": "gpu_residual_bridge_d3d12"},
    0.40: {"dims": [512, 288], "resampling": "plan_driven", "downscaled": True,
           "residual": "gpu_residual_bridge_d3d12"},
    0.63: {"dims": [808, 456], "resampling": "plan_driven", "downscaled": True,
           "residual": "gpu_residual_bridge_d3d12"},
    0.75: {"dims": [960, 544], "resampling": "plan_driven", "downscaled": True,
           "residual": "gpu_residual_bridge_d3d12"},
    0.99: {"dims": [1264, 712], "resampling": "plan_driven", "downscaled": True,
           "residual": "gpu_residual_bridge_d3d12"},
    1.0: {"dims": [1280, 720], "resampling": "identity", "downscaled": False,
          "residual": "none"},
}


def _sha1(blob):
    """The digest form the probe's own report uses: twelve hex of the bytes read."""
    return hashlib.sha1(blob).hexdigest()[:12]


def _diagnostics(frames=FRAMES, scale=0.5, copy_pass=False, overrides=None):
    """What a clean device run reports at `scale`: every boundary measured, none
    of them host, and no number at all for a stage this chain does not open."""
    branch = BRANCHES[scale]
    downscaled = branch["downscaled"]
    report = {
        "gpu_transfer_frames_requested": frames,
        "decode_host_copy_bytes_measured": 0,
        "decode_host_copy_count_measured": 0,
        "decode_frames_decoded_measured": frames,
        "decode_device_frame_count_measured": frames,
        "nv12_to_rgba_convert_count_measured": frames,
        "dlss_host_color_stage_bytes_measured": 0,
        "dlss_output_readback_bytes_measured": 0,
        "encoder_host_pixel_bytes_measured": 0,
        "encoder_frames_submitted_measured": frames,
        "encoder_packets_written_measured": frames,
        "export_backend_requested": "gpu_transfer",
        "export_backend_actual": "gpu_transfer",
        "export_backend_fallback_reason": None,
        "gpu_transfer_working_scale": scale,
        "gpu_transfer_working_dims": list(branch["dims"]),
        "gpu_transfer_resampling": branch["resampling"],
        "gpu_transfer_residual_backend": branch["residual"],
        "gpu_transfer_copy_pass": copy_pass,
        "gpu_transfer_nvof_bridge_loaded": True,
        "gpu_transfer_dlss_flow_calls": frames,
        "gpu_transfer_dlss_zero_motion_calls": 0,
        "gpu_transfer_bridge_dll_directory": [gpu_transfer_export.NATIVE_DIR],
        # The flow engine's own device-input counters move on a guided run; the
        # probe records them and does not claim they are zero.
        "nvof_device_input_copy_count_total": frames,
        "nvof_device_input_bytes_total": frames * SRC_W * SRC_H,
    }
    if downscaled:
        # The working-colour and residual bridges are open on this chain, so they
        # answer for themselves: a frame count and a measured zero.
        report["working_color_process_count_measured"] = frames
        report["residual_output_readback_bytes_measured"] = 0
    else:
        report["working_color_process_count_measured"] = None
        report["residual_output_readback_bytes_measured"] = None
    report.update(overrides or {})
    return report


class _FakeChain:
    """The production entry point, answering the way the chain answers.

    A stub the probe could not tell from the real thing on a clean run, which is the
    point: everything this file asserts is a decision the probe makes about the
    report it is handed, not a decision about how to produce one.

    The encoder fields are the one part of that report which is computed rather than
    written out, and computed by the production planner: ``plan_encoder`` on the
    payload this chain was handed, and the caller's own answer about the tune. A fake
    that invented those values would test the probe against an invention; this one
    tests it against what the same dictionary does on its way into a bridge.
    """

    def __init__(self, diagnostics=None, raise_error=None, deliver=None, leak=False,
                 uhq=False):
        self.diagnostics = diagnostics
        self.raise_error = raise_error
        self.deliver = deliver
        self.leak = leak
        # Whether the encoder this process loaded advertises the panel's default tune.
        # False is what the FFmpeg a bundle carries answers for av1_nvenc on this box.
        self.uhq = uhq
        self.calls = []

    def encoder_report(self, kwargs):
        plan = gpu_transfer_export.plan_encoder(
            kwargs.get("transcode_settings") or {},
            int(kwargs.get("cq", 24)),
            gpu_transfer_export.probe_source(kwargs["video"]),
            tune_available=lambda codec: self.uhq)
        return {
            "gpu_transfer_encoder_codec": plan["codec"],
            "gpu_transfer_encoder_av_options": plan["av_options"],
            "gpu_transfer_encoder_options": {
                name: (value.decode("utf-8") if isinstance(value, bytes) else value)
                for name, value in plan["struct"].items()
                if name in ("preset", "tune", "multipass", "rate_control",
                            "constant_quality", "bitrate", "gop", "color_matrix")},
        }

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.diagnostics is None:
            # The report follows the request: the scale the probe asked to run at
            # decides which chain answers, exactly as it does in production. The
            # rewrap question is answered by the production rule as well, because a
            # payload can ask for one on a container that would not otherwise need it.
            report = _diagnostics(frames=kwargs["frames"],
                                  scale=kwargs["job_settings"]["working_scale"],
                                  copy_pass=gpu_transfer_export.needs_copy_pass(
                                      gpu_transfer_export.probe_source(kwargs["video"]),
                                      kwargs.get("transcode_settings") or {},
                                      kwargs.get("with_audio", False),
                                      kwargs["out_path"]))
        else:
            report = dict(self.diagnostics)
        for name, value in self.encoder_report(kwargs).items():
            report.setdefault(name, value)
        # `or {}` would be wrong here in the fake exactly as it would in production:
        # the probe hands the chain an empty dict, and the report has to land in that
        # object rather than in a throwaway.
        sink = kwargs.get("diagnostics_out")
        if self.leak:
            # The copy pass writes this, muxes over it, and unlinks it. Leaving one
            # behind is a device-produced file in a directory nobody named.
            handle, name = tempfile.mkstemp(
                prefix=gui.GPU_TRANSFER_SELFTEST_INTERMEDIATE_PREFIX)
            os.close(handle)
            self.leaked = name
        if self.raise_error is not None:
            # The real chain reports what it got through before it raises, and the
            # session the caller opened is not yet closed.
            if sink is not None:
                sink.update(report)
            raise self.raise_error
        if sink is not None:
            sink.update(report)
        collector = kwargs.get("transfer_collector")
        for index in range(int(report["gpu_transfer_frames_requested"])):
            collector.record({
                "frame_idx": index,
                "export_backend_requested": report["export_backend_requested"],
                "export_backend_actual": report["export_backend_actual"],
                "transfer_mode_actual": "gpu_transfer",
                "dlss_backend_actual": report.get(
                    "gpu_transfer_dlss_host", "gpu_flow_d3d12"),
                "residual_backend": report["gpu_transfer_residual_backend"],
                "nvof_device_input_copy_count": 1,
                "nvof_device_input_bytes": SRC_W * SRC_H})
        out_path = kwargs["out_path"]
        # A run that reports frames leaves a file behind, and the probe reads its
        # size off the deliverable rather than trusting what it was told.
        with open(out_path, "wb") as handle:
            handle.write(b"x" * DELIVERABLE_BYTES)
        return out_path if self.deliver is None else self.deliver


class _Lives:
    def __init__(self):
        self.built = []

    def __call__(self, w, h, settings=None):
        class Live:
            def __init__(self):
                self.dims, self.settings, self.closed = (w, h), dict(settings or {}), False

            def close(self):
                self.closed = True

        live = Live()
        self.built.append(live)
        return live


@pytest.fixture
def probe(monkeypatch, tmp_path):
    """Run the real probe with the tools, the host and the chain stood in for."""
    video = tmp_path / "known.mkv"
    video.write_bytes(b"a known short clip")
    out = tmp_path / "run" / "out.mkv"
    out.parent.mkdir()

    monkeypatch.setattr(pipeline, "video_info", lambda path: (FRAMES * 2, 24.0, SRC_W, SRC_H))
    monkeypatch.setattr(gpu_transfer_export, "probe_source",
                        lambda path, ffprobe=None: {"video_stream_count": 1,
                                                    "pix_fmt": "yuv420p",
                                                    "color_range": "tv",
                                                    "color_space": None,
                                                    "r_frame_rate": "24/1",
                                                    "audio_streams": 0})
    monkeypatch.setattr(optical_flow, "get_frame_guidance_service",
                        lambda *a, **k: object())

    def run(chain=None, frames=FRAMES, packets=None, stream=None, ffmpeg=None,
            use_video=video, out_path=out, scale=None, transcode_settings_path=None,
            uhq=False):
        chain = chain if chain is not None else _FakeChain(uhq=uhq)
        monkeypatch.setattr(pipeline, "export_dlss_stream", chain)
        monkeypatch.setattr(gui.dlss_gpu_engine, "Live", lives)
        # The tune question is the encoder's own, and on this box the answer for the
        # build a bundle carries is no. Stood in for so both halves of it -- a plan
        # that downgraded and a plan that did not -- are reachable without a card.
        monkeypatch.setattr(pipeline, "_supports_tune_uhq",
                            lambda ffmpeg, codec="av1_nvenc": uhq)

        def check_output(command, *args, **kwargs):
            row = {"codec_name": "av1", "width": SRC_W, "height": SRC_H,
                   "pix_fmt": "yuv420p", "color_range": "tv",
                   "color_space": "smpte170m",
                   "nb_read_packets": frames if packets is None else packets}
            row.update(stream or {})
            return json.dumps({"streams": [row], "packets": []}).encode()

        def spawn(command, *args, **kwargs):
            # What the probe asked this layout's FFmpeg to make, made: the last
            # argument is the file it was told to write.
            with open(command[-1], "wb") as handle:
                handle.write(b"generated by the layout's own ffmpeg")
            return subprocess.CompletedProcess(command, 0)

        monkeypatch.setattr(subprocess, "check_output", check_output)
        monkeypatch.setattr(subprocess, "run", spawn)
        if ffmpeg:
            monkeypatch.setattr(pipeline, "_bundle_ffmpeg", lambda: ffmpeg)
        report = tmp_path / "report.json"
        code = gui._gpu_transfer_selftest(
            video=str(use_video) if use_video is not None else None,
            out_path=str(out_path) if out_path is not None else None,
            frames=frames, working_scale=scale, report_path=str(report),
            transcode_settings_path=transcode_settings_path)
        facts = json.loads(report.read_text(encoding="utf-8"))
        facts["_chain"] = chain
        return code, facts, chain, lives

    lives = _Lives()
    return run


def test_the_probe_asks_the_production_entry_for_the_device_chain(probe):
    code, facts, chain, lives = probe()
    assert code == 0, facts.get("failed_checks")
    assert facts["result"] == "GPU_TRANSFER_OK"
    assert chain.calls, "the probe ran no export at all"
    call = chain.calls[0]
    for key, value in (("export_backend", "gpu_transfer"),
                       ("dlss_backend", "gpu_flow_d3d12"),
                       ("guidance_mode", 1), ("flow_engine", "nvof"),
                       ("working_scale", 0.5), ("residual_pyramid", True)):
        assert call["job_settings"][key] == value
    assert call["with_audio"] is False
    assert call["transcode_settings"] == {}, "an invented encoder plan is not the default"
    assert call["frames"] == facts["frames"] == FRAMES
    assert call["out_path"] == facts["output"]
    assert call["flow_provider"] is not None
    assert lives.built[0].closed, "the probe leaked the session it opened"
    assert lives.built[0].dims == (SRC_W // 2, SRC_H // 2), \
        "the host session is opened at the working extent, not the source one"


def test_the_fixture_is_generated_by_the_layout_it_was_started_in(probe, tmp_path):
    code, facts, _, _ = probe(ffmpeg=os.path.join("C:", "bundle", "ffmpeg.exe"),
                              use_video=None)
    assert code == 0, facts.get("failed_checks")
    command = facts["fixture_command"]
    assert command[0] == os.path.join("C:", "bundle", "ffmpeg.exe"), \
        "a generated fixture made by somebody else's FFmpeg is not this layout's"
    assert facts["fixture"]["generated"] is True
    assert facts["fixture"]["path"].startswith(facts["workdir"]), \
        "the clip the probe made for itself stays in its own temporary directory"
    assert facts["output"] == str(tmp_path / "run" / "out.mkv"), \
        "an output the caller named is the output that gets written"
    assert "-an" in command and "yuv420p" in command

    _, unnamed, _, _ = probe(ffmpeg=os.path.join("C:", "bundle", "ffmpeg.exe"),
                             use_video=None, out_path=None)
    assert unnamed["output"].startswith(unnamed["workdir"]), \
        "with no output named, the deliverable is written beside the fixture it came from"


def test_the_frame_count_stays_inside_the_window_it_claims(probe):
    low, high = gui.GPU_TRANSFER_SELFTEST_FRAMES
    for requested, expected in ((1, low), (999, high), (4, 4)):
        code, facts, chain, _ = probe(frames=expected)
        assert facts["frames"] == expected
        assert chain.calls[0]["frames"] == expected
        assert expected >= low and expected <= high


def test_the_default_ratio_is_the_exact_half_and_is_reported_as_such(probe):
    code, facts, chain, lives = probe()
    assert code == 0, facts.get("failed_checks")
    assert facts["working_scale"] == 0.5
    assert chain.calls[0]["job_settings"]["working_scale"] == 0.5
    assert facts["branch"] == {"resampling": "exact_two_x", "working_dims": [640, 360],
                               "downscaled": True}
    assert lives.built[0].dims == (640, 360)
    # The branch fields are checks in their own right, not decorations on the report.
    for name in ("reported_working_scale", "reported_resampling", "reported_working_dims",
                 "reported_residual_backend", "reported_copy_pass"):
        assert facts["checks"][name] is True, name


@pytest.mark.parametrize("scale,dims,resampling", [
    (0.63, [808, 456], "plan_driven"),
    (0.75, [960, 544], "plan_driven"),
])
def test_a_plan_driven_ratio_runs_the_plan_chain_and_says_so(probe, scale, dims,
                                                             resampling):
    code, facts, chain, lives = probe(scale=scale)
    assert code == 0, facts.get("failed_checks")
    assert chain.calls[0]["job_settings"]["working_scale"] == scale
    assert facts["diagnostics"]["gpu_transfer_resampling"] == resampling
    assert facts["diagnostics"]["gpu_transfer_working_dims"] == dims
    assert lives.built[0].dims == tuple(dims), \
        "the DLSS session is opened at the extents the plan was built for"
    # Both resampling stages are on this chain, so both answer with their own numbers.
    assert facts["checks"]["counter:working_color_process_count_measured"] is True
    assert facts["checks"]["zero:residual_output_readback_bytes_measured"] is True


def test_the_identity_chain_answers_no_number_for_the_stages_it_has_none_of(probe):
    code, facts, chain, lives = probe(scale=1.0)
    assert code == 0, facts.get("failed_checks")
    assert facts["branch"] == {"resampling": "identity",
                               "working_dims": [1280, 720], "downscaled": False}
    assert lives.built[0].dims == (1280, 720), "1.0 is the source extent, not a rounding"
    assert facts["diagnostics"]["gpu_transfer_resampling"] == "identity"
    assert facts["diagnostics"]["gpu_transfer_residual_backend"] == "none"
    assert facts["per_frame_residual_backends"] == ["none"], \
        "a per-frame row still claims the residual stage ran"
    for name in ("working_color_process_count_measured",
                 "residual_output_readback_bytes_measured"):
        assert facts["diagnostics"][name] is None
        assert facts["checks"]["unmeasured:" + name] is True
    # The boundaries that do exist on this chain are still measured zeros.
    assert facts["checks"]["zero:encoder_host_pixel_bytes_measured"] is True
    assert facts["checks"]["counter:nv12_to_rgba_convert_count_measured"] is True


def test_an_identity_run_that_reported_a_residual_zero_is_not_a_pass(probe):
    # "0" and "nothing ran" are different claims, and only one of them is true.
    # Accepting the first is how a probe starts certifying a stage it never opened.
    fake = _FakeChain(diagnostics=_diagnostics(
        scale=1.0, overrides={"residual_output_readback_bytes_measured": 0,
                              "working_color_process_count_measured": 0}))
    code, facts, _, _ = probe(scale=1.0, chain=fake)
    assert code == 1
    assert "unmeasured:residual_output_readback_bytes_measured" in facts["failed_checks"]
    assert "unmeasured:working_color_process_count_measured" in facts["failed_checks"]


def test_a_branch_that_disagrees_with_the_requested_ratio_is_not_a_pass(probe):
    # The run claims the half-size constant-tap chain while being asked for 0.63:
    # that is the shape drift this probe exists to catch, because every counter
    # would otherwise still be satisfied.
    wrong = _diagnostics(frames=FRAMES, scale=0.63,
                         overrides={"gpu_transfer_resampling": "exact_two_x",
                                    "gpu_transfer_working_dims": [640, 360]})
    code, facts, _, _ = probe(scale=0.63, chain=_FakeChain(diagnostics=wrong))
    assert code == 1
    assert "reported_resampling" in facts["failed_checks"]
    assert "reported_working_dims" in facts["failed_checks"]


@pytest.mark.parametrize("requested,reason", [
    # 0.3 is a served percentage now; a third is not one, and asking for it means
    # the caller did not say which size it wanted.
    ("0.333", "not a whole percentage between 25 and 100"),
    ("0.24", "not a whole percentage between 25 and 100"),
    ("banana", "not a number"),
])
def test_an_unservable_ratio_is_refused_before_a_session_opens(probe, requested,
                                                               reason):
    code, facts, chain, lives = probe(scale=requested)
    assert code == 1
    assert facts["result"] == "GPU_TRANSFER_ERR"
    assert reason in facts["error"], facts["error"]
    assert not chain.calls, "a refused ratio still ran an export"
    assert not lives.built, "a refused ratio still opened a DLSS session"


def test_the_refusal_names_the_range_it_will_accept(probe):
    code, facts, _, _ = probe(scale=0.333)
    assert code == 1
    assert "25" in facts["error"] and "100" in facts["error"], facts["error"]
    assert "resolve_working_dims" in facts["error"], facts["error"]


@pytest.mark.parametrize("scale", [0.25, 0.26, 0.30, 0.40, 0.50, 0.63, 0.75, 0.99])
def test_every_whole_percentage_passes_the_probe_on_the_branch_it_resolves_to(
        probe, scale):
    """The probe classifies a percentage the way the gate does, for the new ones too.

    Each of these ran a finished six-frame export against the branch the probe
    predicted for it, so 26% and 40% and 99% are checked for the same thing the
    whitelist used to be checked for - and the whitelist is not what decides it.
    """
    code, facts, chain, _ = probe(scale=scale)
    assert code == 0, (facts.get("failed_checks"), facts.get("error"))
    assert chain.calls, "the probe skipped the export it was asked to run"
    assert facts["branch"]["resampling"] == (
        "exact_two_x" if scale == 0.5 else "plan_driven")
    assert facts["checks"]["reported_resampling"] is True
    assert facts["checks"]["reported_working_dims"] is True


def test_the_copy_pass_expectation_follows_the_container_not_the_scale(probe, tmp_path):
    # 1.0 does not imply a rewrap: with this probe's empty encoder plan and a silent
    # MKV source the chain writes the deliverable directly. Naming an MP4 is what
    # asks for the copy pass, and the probe reads the decision off that request.
    code, mkv, chain, _ = probe(scale=1.0)
    assert code == 0, mkv.get("failed_checks")
    assert mkv["diagnostics"]["gpu_transfer_copy_pass"] is False
    assert mkv["checks"]["reported_copy_pass"] is True

    mp4 = tmp_path / "run" / "out.mp4"
    code, facts, _, _ = probe(scale=1.0, out_path=mp4)
    assert code == 0, facts.get("failed_checks")
    assert facts["diagnostics"]["gpu_transfer_copy_pass"] is True
    assert facts["checks"]["reported_copy_pass"] is True


def test_a_run_that_leaked_its_copy_pass_intermediate_is_not_a_pass(probe, tmp_path):
    # The copy pass writes a temp MKV, muxes it into the deliverable and unlinks it.
    # A leftover is a device-produced file in a directory this probe never named.
    mp4 = tmp_path / "run" / "leaked.mp4"
    chain = _FakeChain(leak=True)
    try:
        code, facts, _, _ = probe(out_path=mp4, chain=chain)
        assert code == 1
        assert "no_leaked_intermediate" in facts["failed_checks"]
        assert facts["leaked_intermediates"]
    finally:
        if getattr(chain, "leaked", None):
            os.unlink(chain.leaked)


def test_a_frame_that_crossed_to_host_memory_is_not_a_pass(probe):
    dirty = _diagnostics(overrides={"decode_host_copy_count_measured": 3,
                                    "encoder_host_pixel_bytes_measured": 4096})
    code, facts, _, _ = probe(chain=_FakeChain(diagnostics=dirty))
    assert code == 1
    assert facts["result"] == "GPU_TRANSFER_INCOMPLETE"
    assert "zero:decode_host_copy_count_measured" in facts["failed_checks"]
    assert "zero:encoder_host_pixel_bytes_measured" in facts["failed_checks"]
    assert facts["checks"]["ran_gpu_transfer"] is True, \
        "it is still a device run; the transfer claim is what failed"


def test_a_silent_fallback_to_the_pipe_is_not_a_pass(probe):
    fell_back = _diagnostics(overrides={"export_backend_actual": "legacy_pipe"})
    code, facts, _, _ = probe(chain=_FakeChain(diagnostics=fell_back))
    assert code == 1
    assert "ran_gpu_transfer" in facts["failed_checks"]
    assert facts["per_frame_backends"] == ["legacy_pipe"]


def test_an_unguided_frame_count_is_not_a_pass(probe):
    # Motion never reached NVOF: the flow entry answered zero frames.
    no_flow = _diagnostics(overrides={"gpu_transfer_dlss_flow_calls": 0,
                                      "gpu_transfer_dlss_zero_motion_calls": FRAMES})
    code, facts, _, _ = probe(chain=_FakeChain(diagnostics=no_flow))
    assert code == 1
    assert "flow_drove_every_frame" in facts["failed_checks"]
    assert "counter:gpu_transfer_dlss_flow_calls" in facts["failed_checks"]


def test_a_short_deliverable_is_not_a_pass(probe):
    # The chain's own counters say six; the container holds five packets. The
    # ffprobe of the deliverable, not the exporter's word, decides.
    code, facts, _, _ = probe(packets=FRAMES - 1)
    assert code == 1
    assert "packets_equal_frames" in facts["failed_checks"]
    assert facts["ffprobe"]["packets"] == FRAMES - 1


def test_a_recoded_or_resized_deliverable_is_not_a_pass(probe):
    code, facts, _, _ = probe(stream={"codec_name": "av1", "width": SRC_W,
                                      "height": SRC_H // 2})
    assert code == 1
    assert "encoder_output_shape" in facts["failed_checks"]


def test_a_returned_path_that_is_not_the_output_is_not_a_pass(probe):
    code, facts, _, _ = probe(chain=_FakeChain(deliver="D:/elsewhere/out.mkv"))
    assert code == 1
    assert "return_is_output" in facts["failed_checks"]


def test_a_refusal_is_written_down_with_the_chains_own_reason(probe):
    refusal = RuntimeError("GPU 显存直传导出后端 (export_backend=gpu_transfer) 拒绝执行本任务："
                           "working_scale=0.333 is not a whole percentage between 25 and 100")
    code, facts, _, lives = probe(chain=_FakeChain(raise_error=refusal))
    assert code == 1
    assert facts["result"] == "GPU_TRANSFER_ERR"
    assert "export_backend=gpu_transfer" in facts["traceback"]
    assert "working_scale=0.333" in facts["traceback"]
    assert lives.built[0].closed, "a session opened before the refusal still has to close"


# What the delivery panel puts on every job: the encoder plan its page shows, plus the
# three keys the queue reads and no chain ever is handed. This is a slice a unit test
# can state without pretending to be a capture -- the whole 51-key dictionary is what
# the frozen run is given, read off the wire.
PANEL_PAYLOAD = {
    "codec": "av1_nvenc", "container": ".mkv", "rate_mode": "cq", "cq": 16,
    "preset": "p7", "tune": "uhq", "multipass": "fullres",
    "aq_mode": "spatial", "aq_strength": 8, "weighted_prediction": False,
    "resolution_mode": "original", "custom_factor": 1.0, "copy_attachments": True,
    "suffix": "_AV1_P7_UHQ_Q16", "finish_action": "none",
}


def payload_file(tmp_path, settings, name="payload.json"):
    """A payload on disk, in the form the queue carries it: one JSON object."""
    path = tmp_path / name
    path.write_text(json.dumps(settings), encoding="utf-8")
    return str(path)


def test_a_payload_file_reaches_the_chain_it_was_asked_for(probe, tmp_path):
    path = payload_file(tmp_path, PANEL_PAYLOAD)
    code, facts, chain, lives = probe(transcode_settings_path=path)
    assert code == 0, facts.get("failed_checks")
    # Verbatim. The probe is not allowed to filter, rename or default-fill what the
    # panel sent: a payload this entry edited would no longer be the payload the
    # interface emits, and the run would prove something about the probe instead.
    assert chain.calls[0]["transcode_settings"] == PANEL_PAYLOAD
    assert facts["transcode_settings_accepted"] == PANEL_PAYLOAD
    assert chain.calls[0]["frames"] == FRAMES, "the bounded window still bounds it"
    # The file it read, named and hashed, so a report can be tied to one artifact.
    read = open(path, "rb").read()
    assert facts["transcode_settings_file"]["path"] == os.path.abspath(path)
    assert facts["transcode_settings_file"]["bytes"] == len(read)
    assert facts["transcode_settings_file"]["sha1"] == _sha1(read)


def test_an_empty_payload_is_the_plan_this_probe_always_used(probe, tmp_path):
    code, facts, chain, _ = probe(transcode_settings_path=payload_file(tmp_path, {}))
    assert code == 0, facts.get("failed_checks")
    assert chain.calls[0]["transcode_settings"] == {}
    assert facts["transcode_settings_accepted"] == {}
    # Nothing was requested, so nothing may be claimed about a request: these checks
    # exist only for a run that was handed a payload.
    for name in ("requested_encoder_codec", "requested_encoder_cq",
                 "requested_encoder_tune_reported", "encoder_tune_matches_encoder"):
        assert name not in facts["checks"]


def test_the_encoder_answer_is_reported_beside_the_request(probe, tmp_path):
    code, facts, _, _ = probe(transcode_settings_path=payload_file(tmp_path, PANEL_PAYLOAD))
    assert code == 0, facts.get("failed_checks")
    resolved = facts["encoder_options"]["struct"]
    # The panel asked for UHQ; the encoder this process loaded does not have it, so the
    # plan handed the bridge its own default. Both halves are in the report, and the
    # probe only calls the run clean because the encoder was asked and answered.
    assert resolved["tune"] == "hq"
    assert resolved["constant_quality"] == 16
    assert resolved["preset"] == "p7"
    assert facts["encoder_options"]["codec"] == "av1_nvenc"
    assert ["weighted_pred", "0"] in facts["encoder_options"]["av_options"]
    for name in ("requested_encoder_codec", "requested_encoder_cq",
                 "requested_encoder_tune_reported", "encoder_tune_matches_encoder"):
        assert facts["checks"][name] is True, name


def test_a_run_on_an_encoder_that_has_the_tune_keeps_it(probe, tmp_path):
    code, facts, _, _ = probe(transcode_settings_path=payload_file(tmp_path, PANEL_PAYLOAD),
                              uhq=True)
    assert code == 0, facts.get("failed_checks")
    assert facts["encoder_options"]["struct"]["tune"] == "uhq"
    assert facts["checks"]["encoder_tune_matches_encoder"] is True


def test_a_tune_the_encoder_does_not_have_is_not_reported_as_accepted(probe, tmp_path):
    # The chain claims it handed UHQ to an encoder that says it has no such tune. That
    # is the fault this run exists to catch: the gate approved a value, the plan said it
    # downgraded it, and the container cannot tell which the bridge actually received.
    answered = _diagnostics(scale=0.5, copy_pass=True, overrides={
        "gpu_transfer_encoder_codec": "av1_nvenc",
        "gpu_transfer_encoder_av_options": [["spatial-aq", "1"]],
        "gpu_transfer_encoder_options": {"preset": "p7", "tune": "uhq",
                                         "multipass": None, "rate_control": "vbr",
                                         "constant_quality": 16, "bitrate": 0,
                                         "gop": 48, "color_matrix": 1}})
    code, facts, _, _ = probe(chain=_FakeChain(diagnostics=answered),
                              transcode_settings_path=payload_file(tmp_path, PANEL_PAYLOAD))
    assert code == 1
    assert "encoder_tune_matches_encoder" in facts["failed_checks"]


def test_a_silently_dropped_encoder_setting_is_not_a_pass(probe, tmp_path):
    """A payload that arrives changed, or does not arrive at all."""
    for name, options in (
            # The constant quality the panel set never reached the bridge: this is the
            # probe's own default answering for the request.
            ("requested_encoder_cq", {"preset": "p7", "tune": "hq", "multipass": None,
                                      "rate_control": "vbr", "constant_quality": 24,
                                      "bitrate": 0, "gop": 48, "color_matrix": 1}),
            # No tune reported for a run that was asked for one: the setting went
            # nowhere and nothing says so but this report.
            ("requested_encoder_tune_reported", {"preset": "p7", "tune": None,
                                                 "multipass": None, "rate_control": "vbr",
                                                 "constant_quality": 16, "bitrate": 0,
                                                 "gop": 48, "color_matrix": 1})):
        answered = _diagnostics(scale=0.5, copy_pass=True, overrides={
            "gpu_transfer_encoder_codec": "av1_nvenc",
            "gpu_transfer_encoder_av_options": [],
            "gpu_transfer_encoder_options": options})
        code, facts, _, _ = probe(chain=_FakeChain(diagnostics=answered),
                                  transcode_settings_path=payload_file(
                                      tmp_path, PANEL_PAYLOAD, name + ".json"))
        assert code == 1, name
        assert name in facts["failed_checks"], name


@pytest.mark.parametrize("label, blob", [
    ("not JSON", b"{codec: av1_nvenc,"),
    ("not an object", b'["av1_nvenc", "uhq"]'),
    ("not UTF-8", b"\xff\xfe\x00codec"),
    ("empty", b"")])
def test_a_payload_that_is_not_a_json_object_refuses_without_opening_anything(probe,
                                                                             tmp_path,
                                                                             label, blob):
    path = tmp_path / (label.replace(" ", "_") + ".json")
    path.write_bytes(blob)
    code, facts, chain, lives = probe(transcode_settings_path=str(path))
    assert code == 1, label
    assert facts["result"] == "GPU_TRANSFER_ERR"
    assert "transcode_settings" in facts["error"], label
    assert chain.calls == [], f"{label}: a bad payload still ran an export"
    assert lives.built == [], f"{label}: a bad payload still opened a DLSS session"


def test_a_missing_payload_file_refuses_without_opening_anything(probe, tmp_path):
    code, facts, chain, lives = probe(
        transcode_settings_path=str(tmp_path / "never_written.json"))
    assert code == 1
    assert facts["result"] == "GPU_TRANSFER_ERR"
    assert chain.calls == []
    assert lives.built == []


def test_main_routes_the_flag_before_any_window_exists(monkeypatch):
    calls = []
    monkeypatch.setattr(gui, "_gpu_transfer_selftest", lambda **kwargs: calls.append(kwargs) or 0)
    monkeypatch.setattr(gui.tk, "Tk",
                        lambda *a, **k: pytest.fail("the device probe opened a Tk root"))
    monkeypatch.setattr(sys, "argv", ["gui.py", gui.GPU_TRANSFER_SELFTEST_FLAG,
                                      "frames=4", "scale=0.63", "out=D:/x/out.mkv"])
    # The flag is a CLI entry: whatever the probe says is the process exit code,
    # and the window is never built to say it.
    with pytest.raises(SystemExit) as stopped:
        gui.main()
    assert stopped.value.code == 0
    assert len(calls) == 1
    assert calls[0]["frames"] == "4"
    assert calls[0]["working_scale"] == "0.63"
    assert calls[0]["out_path"] == "D:/x/out.mkv"


def test_the_scale_token_takes_its_long_name_too(monkeypatch):
    calls = []
    monkeypatch.setattr(gui, "_gpu_transfer_selftest", lambda **kwargs: calls.append(kwargs) or 0)
    monkeypatch.setattr(gui.tk, "Tk", lambda *a, **k: pytest.fail("the device probe opened a Tk root"))
    monkeypatch.setattr(sys, "argv", ["gui.py", gui.GPU_TRANSFER_SELFTEST_FLAG,
                                      "working_scale=1"])
    with pytest.raises(SystemExit) as stopped:
        gui.main()
    assert stopped.value.code == 0
    assert calls[0]["working_scale"] == "1"
    # Nothing named it: the entry has to keep running the half-size chain it ran
    # before this argument existed.
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["gui.py", gui.GPU_TRANSFER_SELFTEST_FLAG])
    with pytest.raises(SystemExit):
        gui.main()
    assert calls[0]["working_scale"] is None


def test_the_payload_token_reaches_the_probe_and_defaulting_stays_default(monkeypatch):
    calls = []
    monkeypatch.setattr(gui, "_gpu_transfer_selftest", lambda **kwargs: calls.append(kwargs) or 0)
    monkeypatch.setattr(gui.tk, "Tk", lambda *a, **k: pytest.fail("the device probe opened a Tk root"))
    monkeypatch.setattr(sys, "argv", ["gui.py", gui.GPU_TRANSFER_SELFTEST_FLAG,
                                      "transcode_settings=D:/panel/payload.json"])
    with pytest.raises(SystemExit) as stopped:
        gui.main()
    assert stopped.value.code == 0
    assert calls[0]["transcode_settings_path"] == "D:/panel/payload.json"
    # No token: the entry negotiates against an empty dictionary, which is the encoder
    # plan every frozen run before this one used.
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["gui.py", gui.GPU_TRANSFER_SELFTEST_FLAG, "frames=4"])
    with pytest.raises(SystemExit):
        gui.main()
    assert calls[0]["transcode_settings_path"] is None
    assert calls[0]["frames"] == "4"


def test_the_argv_reader_takes_only_what_follows_the_flag():
    argv = ["gui.py", "--selftest", "--gpu-transfer-selftest", "frames=5", "report=/tmp/r.json"]
    assert gui._selftest_argv_value(argv, "--gpu-transfer-selftest") == {
        "frames": "5", "report": "/tmp/r.json"}
    assert gui._selftest_argv_value(["gui.py", "--selftest"],
                                    gui.GPU_TRANSFER_SELFTEST_FLAG) == {}
