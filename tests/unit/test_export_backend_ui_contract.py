import pathlib
import re
import sys
import types

import pytest

import gpu_transfer_export as gtx
import pipeline
from dlss_service import DLSSService

ROOT = pathlib.Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend" / "src"


def _queued_failure(monkeypatch, settings):
    """Run a job the way the queue thread does, and hand back the queue's own record.

    Only the input probe is faked: it stands between the refusal under test and a
    session that would need a GPU, and the refusal has to arrive before either.
    """
    service = DLSSService()
    job = service.add_job({
        "input_path": str(__file__),
        "output_path": str(ROOT / "build" / "unused_refusal_output.mkv"),
        "settings": {**service.settings, **settings},
    })
    monkeypatch.setattr(pipeline, "video_info", lambda path: (10, 60.0, 1920, 1080))

    service._run_queue(None)

    return service.get_jobs()[0]


class _Abi3GpuHost:
    """The smallest session that can answer the queue's zero-motion probe."""

    def __init__(self, w, h, settings=None):
        self._w, self._h, self.settings = w, h, dict(settings or {})
        self.closed = False
        self.last_error = None

    def is_alive(self):
        return not self.closed

    def supports_zero_motion(self):
        return True

    def process_device_flow(self, *args, **kwargs):
        raise AssertionError("an unguided job reached the guided GPU entry")

    def close(self):
        self.closed = True


def _accepted_pipe_job(monkeypatch, tmp_path, settings):
    """Run a job the gate is expected to ACCEPT, with no GPU and no FFmpeg.

    _queued_failure can only ever see a refusal, because a job that passes the
    gate goes on to open a real session. Here the host and the export seam are
    replaced, so the accepted pairing is observable all the way to the session the
    queue hands the export.
    """
    import dlss_service

    service = DLSSService()
    handed = {}
    hosts = []
    job = service.add_job({
        "input_path": str(__file__),
        "output_path": str(tmp_path / "accepted.mkv"),
        "settings": {**service.settings, **settings},
    })

    def _open_host(w, h, settings=None):
        live = _Abi3GpuHost(w, h, settings)
        hosts.append(live)
        return live

    def _export(**kwargs):
        pathlib.Path(kwargs["out_path"]).write_bytes(b"\x00" * 4096)
        handed["live"] = kwargs["live"]
        handed["flow_provider"] = kwargs["flow_provider"]
        return kwargs["out_path"]

    monkeypatch.setattr(pipeline, "video_info", lambda path: (10, 60.0, 1920, 1080))
    monkeypatch.setattr(pipeline, "export_dlss_stream", _export)
    monkeypatch.setitem(sys.modules, "dlss_gpu_engine",
                        types.SimpleNamespace(Live=_open_host))
    monkeypatch.setattr(dlss_service, "get_frame_guidance_service",
                        lambda *a, **k: pytest.fail(
                            "an unguided job opened an optical flow engine"))

    service._run_queue(None)

    return service.get_jobs()[0], handed, hosts


def test_service_normalizes_and_snapshots_export_backend_for_jobs():
    service = DLSSService()

    assert service.settings["export_backend"] == "auto"
    updated = service.update_settings({"export_backend": "GPU_TRANSFER"})
    job = service.add_job({"settings": updated})
    updated["export_backend"] = "legacy"

    assert job["settings"]["export_backend"] == "gpu_transfer"


@pytest.mark.parametrize("value", ["unknown", "gpu-transfer"])
def test_service_rejects_unknown_export_backend(value):
    service = DLSSService()

    with pytest.raises(ValueError, match="export_backend"):
        service.update_settings({"export_backend": value})
    with pytest.raises(ValueError, match="export_backend"):
        service.add_job({"settings": {"export_backend": value}})


def test_add_job_keeps_the_backend_key_the_chain_negotiates_on():
    # The Deliver page sends dlss_backend next to export_backend; if the job
    # snapshot ever normalized settings down to known keys, GPU Transfer could
    # no longer reach the device path and would refuse every job.
    service = DLSSService()

    job = service.add_job({"settings": {
        **service.settings,
        "export_backend": "gpu_transfer",
        "dlss_backend": "gpu_flow_d3d12",
    }})

    assert job["settings"]["dlss_backend"] == "gpu_flow_d3d12"


def test_export_backend_is_a_typed_choice_that_defaults_to_auto():
    types = (FRONTEND / "types" / "index.ts").read_text(encoding="utf-8")
    workspace = (FRONTEND / "services" / "workspace.ts").read_text(encoding="utf-8")

    assert "export type ExportBackend = 'auto' | 'legacy' | 'gpu_transfer'" in types
    assert "export_backend: ExportBackend" in types
    assert "export_backend: 'auto'," in workspace


def test_gpu_transfer_selection_reaches_the_queue_with_its_flow_backend():
    dlss_page = (FRONTEND / "pages" / "DLSSPage.tsx").read_text(encoding="utf-8")
    deliver_page = (FRONTEND / "pages" / "DeliverPage.tsx").read_text(encoding="utf-8")
    queued = " ".join(deliver_page.split())

    assert 'aria-label="导出后端"' in dlss_page
    assert '<option value="gpu_transfer">' in dlss_page
    # The job carries the whole shared DLSS settings blob, and the panel writes the
    # GPU host into it unconditionally, so every chain - auto, legacy and GPU
    # Transfer - reaches the queue naming the one host this build ships.
    assert "settings: { ...dlssSettings }," in queued
    # The chain used to choose the host: GPU Transfer forced the GPU one and the
    # other two wrote nothing. There is no second host left for that write to select,
    # so the helper and its call are gone rather than inverted.
    assert "dlssHostRequiredByExport" not in queued
    assert "dlssHostRequiredByExport" not in dlss_page


def test_the_auto_and_legacy_chains_carry_the_default_gpu_host_to_the_queue():
    # The whole point of this stage: naming no chain-specific host still routes an
    # auto job onto the GPU host, because the panel default and the service default
    # are now that host rather than the old one.
    service = DLSSService()

    job = service.add_job({"settings": {
        **service.settings,
        "export_backend": "auto",
    }})

    assert job["settings"]["dlss_backend"] == "gpu_flow_d3d12"


def test_the_settings_endpoint_answers_a_retired_host_with_the_reason():
    """The API boundary is where a pre-migration caller learns what to change.

    update_settings refuses the retired name, and FastAPI turns an uncaught ValueError
    into a 500 whose body says "Internal Server Error": the removed binary, the retired
    key and the migration were all inside the exception and none of it reached the
    caller. So the endpoint converts it, the same route the queue endpoint already takes.
    """
    import server
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as raised:
        server.update_settings({"dlss_backend": "host_pointer"})

    assert raised.value.status_code == 400
    detail = raised.value.detail
    assert "host_pointer" in detail and "dlssnr_host.dll" in detail
    assert "gpu_flow_d3d12" in detail
    assert server.dlss_service.settings["dlss_backend"] == "gpu_flow_d3d12", \
        "a refused write must not have been merged"


def test_a_failed_gpu_job_reports_the_gpu_host_and_not_a_log_from_the_retired_one(
        tmp_path, monkeypatch):
    """dlssnr_host.dll wrote dlssnr_run.log next to itself; the GPU host writes no log.

    Every failed GPU job used to carry that file's tail as its own diagnostic, so an
    operator reading a job saw the retired host's old lines blamed on a run that never
    touched it. The job record now says which host ran and quotes that host's own error.
    """
    import server
    from fastapi import HTTPException

    service = DLSSService()
    service._preview_live_error = "GPU 宿主上报: zero-motion DLSS frame failed (5)"
    monkeypatch.setattr(server, "dlss_service", service)

    with pytest.raises(HTTPException):
        server.get_job_diagnostic("nope")

    job = service.add_job({"name": "ledger", "input_path": str(__file__),
                           "output_path": str(tmp_path / "x.mkv"),
                           "settings": {**service.settings}})
    body = server.get_job_diagnostic(job["id"])
    assert "log_tail" not in body and "dlss_log" not in body
    assert body["dlss_host"] is None, "no host has run this job yet"
    assert "dlssnr_run.log" not in str(body)
    assert not hasattr(server, "get_log_tail"), \
        "the diagnostics route must not read the retired host's log at all"


def test_the_retired_host_is_not_a_choice_the_front_end_can_make():
    """What the operator can no longer pick, and what replaces the picking.

    The preview page used to write a host when the export chain changed, and offered a
    select for it. Both went: a type with one member cannot carry the retired name, so
    no client can produce it, and the panel states which binary draws the picture and
    that naming the old one is refused rather than quietly served by the new host.
    """
    api = " ".join((FRONTEND / "services" / "api.ts").read_text(encoding="utf-8").split())
    dlss_page = " ".join((FRONTEND / "pages" / "DLSSPage.tsx").read_text(encoding="utf-8").split())
    types = (FRONTEND / "types" / "index.ts").read_text(encoding="utf-8")
    workspace = (FRONTEND / "services" / "workspace.ts").read_text(encoding="utf-8")

    assert "export type DlssBackend = 'gpu_flow_d3d12'" in types
    assert "dlss_backend?: DlssBackend" in types
    assert "dlss_backend: 'gpu_flow_d3d12'," in workspace
    assert "export function dlssHostRequiredByExport" not in api

    # A retired name is not reachable from the UI as a value, only as prose explaining
    # why a settings file that still carries it is refused.
    assert "<option value=\"host_pointer\"" not in dlss_page
    assert "dlss_backend: 'host_pointer'" not in dlss_page
    assert "if (key === 'export_backend')" not in dlss_page

    # This workbench no longer presents a DLSS-host selector or host-specific panel.
    # The single supported type/default and service refusal tests above/nearby own
    # the migration contract; the redesigned page only exposes export-chain choice.
    assert "dlss_backend" not in dlss_page
    assert 'aria-label="导出后端"' in dlss_page


def test_the_workbench_keeps_preview_and_export_chain_controls_distinct():
    """The redesigned inspector says export routing does not change the preview."""
    page = " ".join((FRONTEND / "pages" / "DLSSPage.tsx").read_text(encoding="utf-8").split())
    start = page.index('<h3>导出后端</h3>')
    panel = page[start:page.index("</p>", start)]
    monitor = page[page.index('aria-label="视频预览画布"'):page.index("</section>", page.index('aria-label="视频预览画布"'))]

    assert "仅导出" in panel
    assert "原生预览复用 GPU Transfer 处理核心" in panel
    assert "实际层数和处理帧率显示在画面中" in panel
    assert "实际处理层数和模式显示在画面中" in monitor
    # Service-level route tests in this module assert that preview continues to use
    # the same GPU host across Auto/GPU Transfer and zero-motion/guided modes.
    for claim in ("更快", "加速", "性能提升", "零拷贝"):
        assert claim not in panel


def test_the_settings_pair_the_page_sends_moves_the_preview_route():
    """Route the preview off the payloads the UI actually posts, GPU-less.

    Nothing about this depends on a device: the question is which entry point the
    next preview frame is driven through, and the answer has to change when the
    operator changes the one control that decides it.
    """
    service = DLSSService()

    # Untouched defaults name the GPU host, and the default guidance mode (0)
    # routes the preview onto its resident zero-motion entry.
    assert service.settings["dlss_backend"] == "gpu_flow_d3d12"
    assert service._preview_live_kind() == "gpu_zero_motion"

    service.update_settings({"guidance_mode": 0, "export_backend": "gpu_transfer",
                             "dlss_backend": "gpu_flow_d3d12"})
    assert service._preview_live_kind() == "gpu_zero_motion"

    # The export chain is not a host selector, in either direction.
    service.update_settings({"export_backend": "auto",
                             "dlss_backend": "gpu_flow_d3d12"})
    assert service._preview_live_kind() == "gpu_zero_motion"


def test_multiscale_settings_survive_save_and_queue_snapshot():
    service = DLSSService()
    scales = [0.5, 0.75]
    weights = [1.0, 1.5]
    saved = service.update_settings({
        "export_backend": "gpu_transfer", "working_scale": 0.5,
        "reconstruction_mode": "dual", "layer_scales": scales,
        "layer_weights": weights,
        "fusion_mode": "data_consistency_experimental",
        "fusion_beta": 0.1, "fusion_max_iterations": 128,
        "fusion_relative_tolerance": 0.001,
    })
    assert saved["layer_scales"] == [0.5, 0.75]
    job = service.add_job({
        "input_path": str(__file__),
        "output_path": str(ROOT / "build" / "unused_multiscale_output.mkv"),
    })
    assert job["settings"]["reconstruction_mode"] == "dual"
    assert job["settings"]["fusion_mode"] == "data_consistency_experimental"
    assert job["settings"]["layer_scales"] == [0.5, 0.75]
    assert job["settings"]["layer_weights"] == [1.0, 1.5]
    scales.append(0.9)
    weights[0] = 3.0
    service.settings["layer_scales"].append(0.9)
    assert job["settings"]["layer_scales"] == [0.5, 0.75]
    assert job["settings"]["layer_weights"] == [1.0, 1.5]
    with pytest.raises(ValueError, match="gpu_transfer"):
        service.add_job({
            "input_path": str(__file__),
            "output_path": str(ROOT / "build" / "unused_multiscale_legacy.mkv"),
            "settings": {**job["settings"], "export_backend": "legacy"},
        })

def test_a_guided_preview_routes_to_the_gpu_host_motion_entry_under_gpu_transfer():
    # The pairing the panel now describes: GPU Transfer moves both preview modes
    # onto the GPU host, the guided one onto its host-memory motion entry.
    service = DLSSService()

    service.update_settings({"guidance_mode": 1, "export_backend": "gpu_transfer",
                             "dlss_backend": "gpu_flow_d3d12"})

    assert service._preview_live_kind() == "gpu_host_motion"

    service.update_settings({"export_backend": "auto",
                             "dlss_backend": "gpu_flow_d3d12"})
    assert service._preview_live_kind() == "gpu_host_motion"


def test_a_settings_write_that_names_the_retired_host_is_refused_not_reinterpreted():
    """The preview page is the writer, so the refusal has to land before the merge.

    A pre-migration settings file or script still says dlss_backend=host_pointer.
    Answering that with the GPU host would report a route nobody asked for, and
    answering it by keeping the old value would leave a setting no host serves.
    """
    service = DLSSService()

    with pytest.raises(ValueError, match="host_pointer") as refused:
        service.update_settings({"export_backend": "auto",
                                 "dlss_backend": "host_pointer"})

    assert "dlssnr_host.dll" in str(refused.value)
    assert service.settings["dlss_backend"] == "gpu_flow_d3d12"
    assert service._preview_live_kind() == "gpu_zero_motion"

    with pytest.raises(ValueError, match="host_pointer"):
        service.add_job({"settings": {**service.settings,
                                      "dlss_backend": "host_pointer"}})


def test_omitting_the_host_key_would_strand_the_gpu_preview():
    """Why the page writes the host explicitly instead of dropping the key.

    update_settings merges, so a patch that only ever added dlss_backend would leave
    a stale value in place after the operator switched chains. This pins the merge
    behaviour the panel's unconditional write depends on.
    """
    service = DLSSService()
    service.update_settings({"guidance_mode": 0, "export_backend": "gpu_transfer",
                             "dlss_backend": "gpu_flow_d3d12"})

    service.update_settings({"export_backend": "auto"})

    assert service.settings["dlss_backend"] == "gpu_flow_d3d12"
    assert service._preview_live_kind() == "gpu_zero_motion"


@pytest.mark.parametrize("value", ["gpu-flow-d3d12", "GPU", ""])
def test_service_rejects_unknown_dlss_backend(value):
    # The preview page is the writer now, so a value that is neither host has to
    # be refused by name instead of silently meaning "the other one".
    service = DLSSService()

    with pytest.raises(ValueError, match="dlss_backend"):
        service.update_settings({"dlss_backend": value})

    # Refused before the merge, so the setting still reads the GPU default rather
    # than the unknown value or a silent switch to the other host.
    assert service.settings["dlss_backend"] == "gpu_flow_d3d12"


def test_gpu_transfer_panel_states_conditions_and_denies_fallback():
    dlss_page = (FRONTEND / "pages" / "DLSSPage.tsx").read_text(encoding="utf-8")
    preview = (FRONTEND / "services" / "exportPreview.ts").read_text(encoding="utf-8")
    start = dlss_page.index('aria-label="导出后端"')
    panel = dlss_page[start:dlss_page.index("</p>", start)]

    assert '<option value="gpu_transfer">GPU Transfer · 条件不符则拒绝</option>' in panel
    assert "拒绝" in panel
    assert "任一条件不满足时后端拒绝并写明原因，不会自动回退到 Legacy 链路。" in preview
    # An opt-in that has never been benchmarked must not read as a win.
    for claim in ("更快", "加速", "性能提升", "零拷贝"):
        assert claim not in panel and claim not in preview


def test_gpu_transfer_panel_offers_the_range_the_gate_serves_and_no_ratio_list():
    """The slider, its presets and the panel copy are one claim about the range.

    Which percentages GPU Transfer accepts is a fact about ``negotiate``: every
    whole one from 25 to 100. So the control has to step over exactly that range,
    every preset on it has to be inside it, and the copy has to name the range
    rather than enumerating ratios - an enumerated panel is what made an
    unmeasured percentage look refused even after the gate stopped refusing it.
    """
    page = (FRONTEND / "pages" / "DLSSPage.tsx").read_text(encoding="utf-8")
    slider = page.index('aria-label="工作分辨率缩放"')
    control = page[slider:page.index("/>", slider)]
    assert f'min={{{gtx.WORKING_PERCENT_MIN}}}' in control, control
    assert f'max={{{gtx.WORKING_PERCENT_MAX}}}' in control, control
    assert 'step={1}' in control, control
    # The control carries whole percentages, not a float that lands between them.
    assert 'value={Math.round(settings.working_scale*100)}' in control, control

    scale_buttons = page[page.index('<div className="scale-presets">'):page.index("</div>", page.index('<div className="scale-presets">'))]
    scale_list = re.search(r"\[([0-9.,]+)\]\.map\(s=>", scale_buttons)
    assert scale_list, scale_buttons
    values = [float(value) for value in scale_list.group(1).split(",")]
    assert values == [1, .75, .67, .63, .5, .25]
    assert all(gtx.WORKING_PERCENT_MIN <= round(value*100) <= gtx.WORKING_PERCENT_MAX for value in values)


def test_the_working_size_the_page_prints_is_the_size_the_resolver_produces():
    """The page shows a working size, and the gate jobs ``resolve_working_dims``.

    A second rounding rule in the page is how an operator is shown 2880x1624 for a
    job the backend resolves to 2880x1616, so the arithmetic lives in one frontend
    module that mirrors the Python clause for clause - and the page is not allowed
    to re-derive it inline.
    """
    page = (FRONTEND / "pages" / "DLSSPage.tsx").read_text(encoding="utf-8")
    mirror = (FRONTEND / "services" / "workingScale.ts").read_text(encoding="utf-8")

    assert "import { resolveWorkingDims } from '../services/workingScale'" in page
    assert "resolveWorkingDims(video?.width||3840,video?.height||2160,settings.working_scale)" in page
    assert "resolveWorkingDims(video?.width||3840,video?.height||2160,scale)" in page
    assert "/ 8) * 8" not in page, "the page re-derives the 8-alignment it imports"

    assert f"WORKING_PERCENT_MIN = {gtx.WORKING_PERCENT_MIN}" in mirror, mirror
    assert f"WORKING_PERCENT_MAX = {gtx.WORKING_PERCENT_MAX}" in mirror, mirror
    # The three clauses of the Python rule, in the one place that spells them out.
    for clause in ("Math.min(1.0, Math.max(0.25, scale))",
                   "Math.max(16, roundToEven((src * s) / 8) * 8)",
                   "Math.abs(s - 1.0) < 1e-4"):
        assert clause in mirror, clause


def test_unguided_on_the_gpu_host_is_routed_on_the_pipe_instead_of_refused(tmp_path, monkeypatch):
    # guidance_mode=0 is the service default, and the GPU host serves it from a
    # resident zero motion texture. The pipe chain used to be refused by name for
    # that pairing, which is how the entry point stayed out of production; both
    # chains drive it now, each through its own boundary. What still has to be
    # true is that the frame is answered by the GPU host and not by the old one,
    # with no flow engine opened for a frame that asks for no motion.
    job, handed, hosts = _accepted_pipe_job(monkeypatch, tmp_path, {
        "export_backend": "legacy",
        "dlss_backend": "gpu_flow_d3d12",
        "guidance_mode": 0,
    })

    assert job["status"] == "completed", job["error"]
    assert len(hosts) == 1
    assert handed["live"] is hosts[0]
    assert handed["flow_provider"] is None
    assert hosts[0].closed, "the job left its GPU session behind for the next one"


def test_auto_export_backend_routes_the_explicit_gpu_host_onto_the_pipe(tmp_path, monkeypatch):
    # "auto" is the legacy chain, so an explicit gpu_flow_d3d12 host has to be
    # honoured there too: refusing it would make auto disagree with legacy for a
    # pairing that asks nothing of the export chain at all.
    job, handed, hosts = _accepted_pipe_job(monkeypatch, tmp_path, {
        "export_backend": "auto",
        "dlss_backend": "gpu_flow_d3d12",
        "guidance_mode": 0,
    })

    assert job["status"] == "completed", job["error"]
    assert handed["flow_provider"] is None and len(hosts) == 1


def test_the_service_no_longer_refuses_guidance_off_on_the_gpu_transfer_chain(monkeypatch):
    """A gate that refuses what the chain can serve makes the feature unreachable.

    This pair used to be refused here by name, which is how the host's zero motion
    texture stayed dead in production. Whatever the job now fails on, it is not
    the queue gate deciding the chain has no unguided entry point.
    """
    job = _queued_failure(monkeypatch, {
        "export_backend": "gpu_transfer",
        "dlss_backend": "gpu_flow_d3d12",
        "guidance_mode": 0,
    })

    error = job["error"] or ""
    assert "零运动推理只由" not in error
    assert "需要光流引导" not in error


def test_cpu_flow_refusal_names_the_flow_engine_and_reaches_the_queue(monkeypatch):
    # DIS remains a host-only flow; RAFT now has a CUDA device provider. The guidance setting in
    # this job is already correct: reusing the guidance message would send the
    # operator to the wrong control.
    job = _queued_failure(monkeypatch, {
        "export_backend": "gpu_transfer",
        "dlss_backend": "gpu_flow_d3d12",
        "guidance_mode": 1,
        "flow_engine": "dis",
    })

    assert job["status"] == "failed"
    assert all(name in job["error"] for name in ("NVIDIA", "AMD FidelityFX", "RAFT CUDA", "flow_engine=dis"))
    assert "guidance_mode" not in job["error"]
