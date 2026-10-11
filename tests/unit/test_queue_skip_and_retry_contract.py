"""Findings 2 and 3: a skipped job is not an export, and a finished job re-runs on word only.

The queue used to answer "the target already exists" by marking the job completed, which
put it in the counters, in the finish-action list and in the operator's confidence. It also
re-ran every failed job each time the queue started, which is a decision nobody made. Both
are fixed here, so both are pinned here.
"""
import asyncio
import json
import pathlib

import pytest

import pipeline
from dlss_service import JOB_SKIPPED, DLSSService


class _Boom:
    def __call__(self, *args, **kwargs):
        raise AssertionError("the queue opened an encoder after the skip")


class _FakeSocket:
    def __init__(self):
        self.sent = []

    async def send_text(self, text):
        self.sent.append(text)


HERE = pathlib.Path(__file__)


@pytest.fixture
def service(monkeypatch):
    """A service whose source probe answers and whose export stage can only be watched."""
    svc = DLSSService()
    monkeypatch.setattr(pipeline, "video_info", lambda path: (2, 60.0, 1920, 1080))
    monkeypatch.setattr(pipeline, "export_dlss_stream", _Boom())
    monkeypatch.setattr(pipeline, "export_gpu_transfer_stream", _Boom())
    return svc


def _existing_output(tmp_path, marker=b"somebody else's finished export"):
    out = tmp_path / "already_there.mkv"
    out.write_bytes(marker)
    return out, marker


def _job(service, output_path, transcode_settings=None):
    return service.add_job({
        "input_path": str(HERE),
        "output_path": str(output_path),
        "settings": {**service.settings},
        "transcode_settings": {"codec": "av1_nvenc", **(transcode_settings or {})},
    })


def _queued(service):
    """Canned stand-in for a real render, so a round can be driven without a GPU."""
    return lambda job, progress_callback: "rendered"


def _capture_finish_actions(service, calls):
    service._handle_finish_action = lambda completed, had_failure: calls.append(
        (list(completed or []), had_failure))


# --- finding 2: the skipped outcome -------------------------------------------

def test_skip_existing_reports_skipped_not_completed(service, tmp_path):
    out, marker = _existing_output(tmp_path)
    job = _job(service, out, {"skip_existing": True})

    service._run_queue(None)

    assert job["status"] == JOB_SKIPPED
    assert job["skipped_reason"] == "output_exists"
    assert job["progress"] == 0.0, "a job that rendered nothing does not read as 100%"
    assert job["current_frame"] == 0
    assert out.read_bytes() == marker, "the file that was there is still there"


def test_overwrite_output_false_skips_the_same_way(service, tmp_path):
    out, marker = _existing_output(tmp_path)
    job = _job(service, out, {"skip_existing": False, "overwrite_output": False})

    service._run_queue(None)

    assert job["status"] == JOB_SKIPPED
    assert job["skipped_reason"] == "output_exists"
    assert out.read_bytes() == marker


def test_a_target_that_may_be_overwritten_still_renders(service, tmp_path):
    """The veto is the file rule, not the new plumbing around it."""
    out, _marker = _existing_output(tmp_path)
    job = _job(service, out, {"skip_existing": False, "overwrite_output": True})
    service._execute_job = _queued(service)

    service._run_queue(None)

    assert job["status"] == "completed"
    assert job["progress"] == 100.0


def test_a_skipped_job_is_not_finished_work(service, tmp_path):
    """A skipped-only round must not put the machine to sleep.

    The finish action is read off the jobs that completed in this run. This one asked for
    shutdown in its own settings and has to stay out of that list - and a skip is neither
    a failure nor a cancel, so it must not veto the round either.
    """
    out, _marker = _existing_output(tmp_path)
    _job(service, out, {"skip_existing": True, "finish_action": "shutdown"})
    calls = []
    _capture_finish_actions(service, calls)

    service._run_queue(None)

    assert calls == [([], False)], "skipped work is neither completed nor a failure"


def test_a_completed_job_still_carries_the_finish_action(service):
    job = _job(service, "unused_output_name.mkv", {"finish_action": "sleep"})
    service._execute_job = _queued(service)
    calls = []
    _capture_finish_actions(service, calls)

    service._run_queue(None)

    assert calls == [([job], False)]


def test_a_mixed_round_counts_only_the_real_export(service, tmp_path):
    out, _marker = _existing_output(tmp_path)
    skipped = _job(service, out, {"skip_existing": True})
    rendered = _job(service, tmp_path / "fresh.mkv", {"finish_action": "sleep"})
    service._execute_job = lambda job, cb: (
        JOB_SKIPPED if job["id"] == skipped["id"] else "rendered")
    calls = []
    _capture_finish_actions(service, calls)

    service._run_queue(None)

    assert skipped["status"] == "skipped" and rendered["status"] == "completed"
    assert calls == [([rendered], False)], "only the export is offered to the finish action"


def test_failure_and_cancellation_still_report_as_they_did(service, tmp_path):
    job = _job(service, tmp_path / "gone" / "nowhere.mkv", {"codec": "not_a_codec"})

    service._run_queue(None)

    assert job["status"] == "failed"
    assert job["skipped_reason"] is None
    assert "not_a_codec" in job["error"]


# --- finding 3: pending only, and retry by name --------------------------------

def test_the_queue_renders_pending_jobs_and_nothing_else(service):
    ran = []
    pending = _job(service, "a.mkv")
    for status in ("failed", "completed", "cancelled", "skipped"):
        other = _job(service, f"{status}.mkv")
        other["status"] = status
    service._execute_job = lambda job, cb: (ran.append(job["id"]), "rendered")[1]

    service._run_queue(None)

    assert ran == [pending["id"]]


def test_starting_a_round_does_not_resurrect_a_failed_job(service, monkeypatch):
    """The regression this replaces: pressing render for a new job re-ran an old failure.

    ``start_queue`` must not normalize or reset anything on its way to spawning the
    worker - the worker takes pending rows and leaves the rest as the operator left them.
    """
    failed = _job(service, "old.mkv")
    failed["status"] = "failed"
    failed["error"] = "a reason from the earlier attempt"
    monkeypatch.setattr(service, "_run_queue", lambda cb: None)

    assert service.start_queue(progress_callback=None) is True

    assert failed["status"] == "failed"
    assert failed["error"] == "a reason from the earlier attempt"


def test_retry_resets_a_failed_job_and_its_last_attempt(service):
    job = _job(service, "will_fail.mkv")
    job.update({"status": "failed", "progress": 37.5, "current_frame": 40,
                "fps": 12.0, "eta": 9, "error": "boom", "error_traceback": "TB",
                "dlss_host": {"backend": "gpu_flow_d3d12"},
                "dlss_host_error": "host said no"})

    retried = service.retry_job(job["id"])

    assert retried["status"] == "pending"
    assert (retried["progress"], retried["current_frame"], retried["fps"],
            retried["eta"]) == (0.0, 0, 0.0, 0)
    assert retried["error"] is None and retried["error_traceback"] is None
    assert retried["dlss_host"] is None and retried["dlss_host_error"] is None


def test_retry_accepts_the_three_states_a_round_leaves(service):
    for status in ("failed", "cancelled", "skipped"):
        job = _job(service, f"{status}.mkv")
        job["status"] = status
        job["skipped_reason"] = "output_exists"

        assert service.retry_job(job["id"])["status"] == "pending"
        assert job["skipped_reason"] is None, "a reset row describes no past attempt"


def test_retrying_a_pending_job_twice_queues_it_once(service):
    job = _job(service, "twice.mkv")

    first = service.retry_job(job["id"])
    again = service.retry_job(job["id"])

    assert first["id"] == again["id"] == job["id"]
    assert again["status"] == "pending"
    assert len(service.get_jobs()) == 1, "one row, not a second run of the same job"


def test_retry_refuses_a_running_job_a_completed_one_and_an_absent_one(service):
    running = _job(service, "running.mkv")
    running["status"] = "running"
    with pytest.raises(ValueError, match="正在渲染中"):
        service.retry_job(running["id"])

    done = _job(service, "done.mkv")
    done["status"] = "completed"
    with pytest.raises(ValueError, match="completed"):
        service.retry_job(done["id"])

    with pytest.raises(KeyError):
        service.retry_job("no-such-job")


def test_retry_is_refused_while_a_round_is_in_flight(service, monkeypatch):
    class _AliveThread:
        def is_alive(self):
            return True

    job = _job(service, "midround.mkv")
    job["status"] = "failed"
    monkeypatch.setattr(service, "_queue_thread", _AliveThread())

    assert service.queue_is_running() is True
    with pytest.raises(ValueError, match="队列仍在运行"):
        service.retry_job(job["id"])
    assert job["status"] == "failed", "a refused retry changed nothing"


def test_a_retried_job_renders_on_the_next_round(service):
    job = _job(service, "retried.mkv")
    job["status"] = "failed"
    ran = []
    service._execute_job = lambda j, cb: (ran.append(j["id"]), "rendered")[1]

    service.retry_job(job["id"])
    service._run_queue(None)

    assert ran == [job["id"]]
    assert job["status"] == "completed"


# --- the API surface -----------------------------------------------------------

def test_the_queue_endpoint_says_whether_a_round_is_running(service, monkeypatch):
    import server
    job = _job(service, "state.mkv")
    monkeypatch.setattr(server, "dlss_service", service)

    state = asyncio.run(server.get_queue())

    assert state["queue_running"] is False
    assert state["active_id"] is None
    assert [j["id"] for j in state["jobs"]] == [job["id"]]


def test_the_retry_endpoint_answers_for_the_row_it_reset(service, monkeypatch):
    import server
    from fastapi import HTTPException

    monkeypatch.setattr(server, "dlss_service", service)
    job = _job(service, "endpoint.mkv")
    job["status"] = "failed"

    answer = asyncio.run(server.retry_queue_job({"id": job["id"]}))

    assert answer["status"] == "ok"
    assert answer["job"]["status"] == "pending"
    assert answer["queue_running"] is False
    assert job["status"] == "pending"

    with pytest.raises(HTTPException) as missing:
        asyncio.run(server.retry_queue_job({"id": "absent"}))
    assert missing.value.status_code == 404

    job["status"] = "running"
    with pytest.raises(HTTPException) as busy:
        asyncio.run(server.retry_queue_job({"id": job["id"]}))
    assert busy.value.status_code == 409
    assert "正在渲染中" in busy.value.detail

    with pytest.raises(HTTPException) as unnamed:
        asyncio.run(server.retry_queue_job({}))
    assert unnamed.value.status_code == 400


def test_a_row_update_carries_the_gate_the_retry_button_reads(service, monkeypatch):
    """A terminal row cannot answer "may I reset this?" on its own.

    Between the last row turning terminal and the worker exiting, nothing is running
    while the queue still owns the GPU session - precisely when a retry must stay
    disabled - so every update states the gate as well as the row.
    """
    import server
    ws = _FakeSocket()
    monkeypatch.setattr(server, "dlss_service", service)
    monkeypatch.setattr(server, "queue_ws_clients", {ws})
    job = _job(service, "broadcast.mkv")

    asyncio.run(server.broadcast_queue_update(job))

    message = json.loads(ws.sent[0])
    assert message["type"] == "job_update"
    assert message["job"]["id"] == job["id"]
    assert message["queue_running"] is False


def test_a_round_announces_its_own_end(service):
    seen = []
    _job(service, "closing.mkv")
    service._execute_job = _queued(service)

    service._run_queue(seen.append)

    assert seen[-1]["type"] == "queue_finished"
    assert seen[-1]["queue_running"] is False
    assert seen[-1]["completed"] == 1


def test_a_skipped_round_announces_it_before_the_end(service, tmp_path):
    out, _marker = _existing_output(tmp_path)
    job = _job(service, out, {"skip_existing": True})
    seen = []

    service._run_queue(seen.append)

    assert job in seen, "the row is reported as it ended"
    assert seen[-1]["type"] == "queue_finished"
    assert seen[-1]["completed"] == 0
    assert [j for j in seen[-1]["jobs"] if j["id"] == job["id"]][0]["status"] == "skipped"
