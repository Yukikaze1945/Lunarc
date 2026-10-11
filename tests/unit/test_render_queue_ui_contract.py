"""What the delivery panel owes the operator about a run it did not render.

These read the shipped front end and the queue endpoints as text, the way the rest of
this suite does: the four findings below are each a promise made in a row, a button or a
preview box, and a source-level assertion is what keeps a later edit from quietly taking
the promise back.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend" / "src"
SERVER = (ROOT / "server.py").read_text(encoding="utf-8")
DELIVER = (FRONTEND / "pages" / "DeliverPage.tsx").read_text(encoding="utf-8")
QUEUE_FEED = (FRONTEND / "services" / "queueFeed.ts").read_text(encoding="utf-8")
TYPES = (FRONTEND / "types" / "index.ts").read_text(encoding="utf-8")
API = (FRONTEND / "services" / "api.ts").read_text(encoding="utf-8")
PREVIEW = (FRONTEND / "services" / "exportPreview.ts").read_text(encoding="utf-8")


# --- finding 2: a skipped job reads as skipped ---------------------------------

def test_skipped_is_a_status_of_its_own_in_the_job_type():
    assert "'pending' | 'running' | 'completed' | 'skipped' | 'cancelled' | 'failed'" in TYPES
    assert "skipped_reason?: string | null" in TYPES


def test_a_skipped_row_is_not_called_a_successful_export():
    assert "已跳过：目标文件已存在，未渲染也未覆盖任何内容" in DELIVER
    assert "已成功导出完成" in DELIVER, "the completed wording stays for completed jobs"
    # The order is the assertion: '已成功导出完成' is the completed branch, and a skip
    # has to be answered before that branch degrades into the pending fallback.
    progress = DELIVER[DELIVER.index("正在渲染: 帧"):]
    progress = progress[:progress.index("job.progress.toFixed")]
    assert progress.index("job.status === 'completed'") < progress.index("'skipped'")
    assert progress.count("'已成功导出完成'") == 1
    assert "'等待渲染'" in progress, "and a skip never falls through to 等待渲染 either"


def test_a_skipped_row_says_so_in_the_icon_the_count_and_the_telemetry():
    assert "job.status === 'skipped' ? (\n                        <MinusCircle" in DELIVER
    assert "已跳过（未渲染）" in DELIVER
    assert "已跳过 · 未写入" in DELIVER
    assert 'title="目标文件保持原样，本任务未编码"' in DELIVER


def test_a_skipped_row_clears_the_active_job_like_any_other_ending():
    terminal = QUEUE_FEED[QUEUE_FEED.index("if (updated.status === 'running')"):]
    for status in ("completed", "failed", "cancelled", "skipped"):
        assert f"'{status}'" in terminal
    assert "active === updated.id" in terminal


# --- finding 3: retry is an action the operator takes --------------------------

def test_retry_is_a_route_the_page_calls_and_the_server_answers():
    assert "/api/queue/retry" in API
    assert "export async function retryQueueJob" in API
    # A refusal has to arrive as a reason: 409 says a round is still running, and the
    # body says what to do about it.
    assert "err.detail || 'Failed to retry job'" in API
    assert '@app.post("/api/queue/retry")' in SERVER
    assert "status_code=404" in SERVER and "status_code=409" in SERVER


def test_the_rows_that_offer_retry_are_the_ones_the_service_accepts():
    assert "const RETRYABLE_STATUSES: RenderJob['status'][] = ['failed', 'cancelled', 'skipped']" in DELIVER
    assert "RETRYABLE_STATUSES.includes(job.status)" in DELIVER
    assert 'aria-label="重试任务 (Retry Job)"' in DELIVER
    assert "await retryQueueJob(id)" in DELIVER
    assert "disabled={queueRunning || isRunning || retryingJobId === job.id}" in DELIVER


def test_render_all_acts_on_pending_rows_and_locks_while_a_round_runs():
    assert "const pendingCount = jobs.filter((j) => j.status === 'pending').length" in DELIVER
    assert "disabled={pendingCount === 0 || queueRunning}" in DELIVER
    # The reason a locked button has to say out loud: a terminal row goes back through
    # the row's own action, not through pressing render again.
    assert "失败/已取消/已跳过的任务需用行内【重试】重置" in DELIVER
    assert "本轮渲染尚未结束" in DELIVER


def test_the_page_learned_the_running_gate_from_the_backend_not_from_itself():
    assert "queue_running: boolean" in TYPES
    assert "queue_running: Boolean(data.queue_running)" in API
    assert '"queue_running": dlss_service.queue_is_running()' in SERVER
    assert "typeof data.queue_running === 'boolean'" in QUEUE_FEED
    assert "setQueueRunning(state.queue_running)" in DELIVER
    for message in ("queue_started", "queue_finished", "job_retried"):
        assert f"'{message}'" in QUEUE_FEED
    assert '"type": "queue_finished"' in (ROOT / "dlss_service.py").read_text(encoding="utf-8")
    assert '{"type": "init", **_queue_state()}' in SERVER


def test_a_refused_retry_is_shown_and_the_rows_are_re_read():
    block = DELIVER[DELIVER.index("const handleRetryJob"):]
    body = block[:block.index("\n  const pendingCount")]
    assert "alert(e.message || '重试任务失败')" in body
    assert "const q = await refreshQueue()" in body, "a refusal re-syncs rather than guesses"


# --- finding 1: the panel refuses to queue a self-overwrite --------------------

def test_the_page_warns_before_the_request_and_the_backend_still_guards_it():
    assert "sameOutputAsInput" in DELIVER
    assert "导出目标与输入素材是同一个文件" in DELIVER
    assert "type: 'error'" in DELIVER[DELIVER.index("sameOutputAsInput"):]
    # Prevention is the courtesy; the guard is the guarantee.
    assert "pipeline.assert_output_not_input(input_path, out_path)" in (
        ROOT / "dlss_service.py").read_text(encoding="utf-8")
    assert "assert_output_not_input(video, out_path)" in (ROOT / "pipeline.py").read_text(
        encoding="utf-8")


def test_the_destination_the_page_sends_is_the_one_it_warned_about():
    assert "const fullOutputPath = composeJobOutputPath(" in DELIVER
    assert "${trimmedDir}\\\\${trimmedName}" not in DELIVER, \
        "the path is composed in one place, shared with the check"


# --- finding 4: the preview describes the chain that will run ------------------

def test_delivery_has_no_device_chain_preview_or_command_copy():
    assert 'Device Chain Preview' not in DELIVER
    assert '设备链路预览' not in DELIVER
    assert 'handleCopyCommand' not in DELIVER
    assert 'deliver-submit' in DELIVER
    assert 'normalizeExportOptions(txDraft, dlssSettings, video)' in DELIVER


def test_the_capability_dependent_tune_is_described_as_one():
    builder = PREVIEW[PREVIEW.index("function tuneLine"):]
    rule = builder[:builder.index("export function buildGpuTransferPreview")]
    assert "实际必定下发 hq（h264_nvenc 没有 UHQ）" in rule
    assert "取决于后端对 ${ts.codec} 编码器能力的探测" in rule
    assert "实际值以任务诊断日志为准" in rule
    # The claim this replaces: one tune that always applies.
    assert "始终使用 uhq" not in rule


def test_the_legacy_preview_is_called_a_mirror_not_the_source_of_truth():
    assert "single source of truth" not in DELIVER.lower()
    assert "e2e 7.1" in DELIVER or "e2e 7.1" in PREVIEW
    # The comment is wrapped, and every line of it starts with //, so the prose is read
    # with the markers and the line breaks folded away.
    joined = " ".join(PREVIEW.replace("//", " ").split())
    assert "Kept as a client-side mirror of pipeline.build_transcode_command, " \
        "not as its source of truth" in joined
