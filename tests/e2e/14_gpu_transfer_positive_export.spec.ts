import { test, expect } from '@playwright/test';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { launchApp } from '../helpers/electron_runner.js';
import {
  FIXTURE_GPU_TRANSFER_SHORT,
  generateFixtureGpuTransferShort,
} from '../fixtures/generate_fixtures.js';
import { clearQueue, getQueue, requestJson, waitForJob } from '../helpers/backend_client.js';
import { importFixture } from '../helpers/ui.js';
import { probeMedia } from '../helpers/media_probe.js';
import { ROOT_DIR } from '../helpers/paths.js';

/**
 * 第十五组测试：界面点选 GPU Transfer 后，一次真实作业跑完并交出可读的文件。
 *
 * 第十四组验的是 opt-in 的接线与被拒原因的可达性；它跑完之后，这个文件补上它故意没跑
 * 的那一半：从界面点下去的那一次选择，确实能让一条显存直传链路在服务里完成一个作业。
 * 这里没有任何 mock：真实 Electron 外壳、真实渲染器、真实 Python 后端、真实 RTX
 * 硬件光流、真实 NVENC。
 *
 * 作业被限制在 6 帧：渲染队列导出的是它拿到的整段素材，所以界定长度的是素材本身
 * （tests/fixtures/fixture_gpu_transfer_short.mkv，1280x720/24fps/6 帧/无声），不是
 * 一个帧数参数。600 帧 120fps 那个 fixture 不属于这里。
 *
 * 断言分四层，缺一不可：
 *   1. 界面上点选的三项（工作分辨率 50%、时空引导=仅光流、导出后端=GPU Transfer）
 *      落到服务设置里，再原样出现在交付页发出的 /api/queue/add 请求体上；
 *   2. 服务对这个作业自己的记录（residual_backend_diagnostics）说：请求与实际都是
 *      gpu_transfer，没有回退原因，跑的是 0.5 的 exact_two_x 链，工作尺寸 640x360，
 *      每一帧的运动都到了 DLSS（flow_calls=帧数、zero_motion_calls=0），且所有宿主
 *      像素边界计数器都是被测量出来的 0；
 *   3. 交付文件本身：存在、非空，ffprobe 数得到的包数等于作业的帧数，1280x720 的
 *      av1/yuv420p；
 *   4. 这一版只有一座 DLSS 宿主：作业记录点名 dlss_gpu_host.dll，而 retirement
 *      掉的 dlssnr_host.dll 在工作树的 native/ 里不存在。
 */

const BACKEND_SELECT = 'select[aria-label="导出后端 (Export Backend)"]';
const GPU_FLOW_BACKEND = 'gpu_flow_d3d12';
const FIXTURE_BASENAME = path.basename(FIXTURE_GPU_TRANSFER_SHORT);

let appInstance: any = null;

async function serviceSettings() {
  return requestJson('/api/settings');
}

async function clickAddToQueue(page: any) {
  const [request] = await Promise.all([
    page.waitForRequest(
      (r: any) => r.url().includes('/api/queue/add') && r.method() === 'POST',
      { timeout: 20000 }
    ),
    page.getByRole('button', { name: /添加到渲染队列/ }).first().click(),
  ]);
  const sent = request.postDataJSON();

  // The panel's payload, verbatim, for a caller that has to run the same job without
  // a browser. Written only when a file is named, so the test's own behavior is
  // unchanged; the hash is of the body the wire carried, so the artifact below can be
  // shown to be a slice of this request rather than a dictionary someone retyped.
  const artifact = process.env.DLSS_E2E_TRANSCODE_ARTIFACT;
  if (artifact) {
    const body = request.postData() || '';
    fs.writeFileSync(artifact, JSON.stringify({
      captured_at: new Date().toISOString(),
      source_spec: 'tests/e2e/14_gpu_transfer_positive_export.spec.ts',
      source_test: '15.1 界面点选 GPU Transfer 后队列完成一个真实作业并交出可探测的文件',
      source_endpoint: 'POST /api/queue/add',
      request_body_bytes: Buffer.byteLength(body),
      request_body_sha256: crypto.createHash('sha256').update(body).digest('hex'),
      job_input_path: sent.input_path,
      job_output_path: sent.output_path,
      job_dock_settings: sent.settings,
      transcode_settings: sent.transcode_settings,
    }, null, 2), 'utf8');
  }
  return sent;
}

async function lastQueuedJob() {
  const queue = await getQueue();
  const jobs = queue.jobs || [];
  expect(jobs.length).toBeGreaterThan(0);
  return jobs[jobs.length - 1];
}

test.describe('第十五组测试：GPU Transfer 正向界面导出（真实硬件，6 帧）', () => {
  test.beforeAll(() => {
    generateFixtureGpuTransferShort();
  });

  test.afterEach(async () => {
    if (appInstance?.electronApp) {
      await appInstance.electronApp.close().catch(() => {});
      appInstance = null;
    }
  });

  test('15.1 界面点选 GPU Transfer 后队列完成一个真实作业并交出可探测的文件', async () => {
    test.setTimeout(300000);
    appInstance = await launchApp({ mockFilePath: FIXTURE_GPU_TRANSFER_SHORT });
    const { page } = appInstance;
    await clearQueue();
    await importFixture(page, FIXTURE_BASENAME);

    // --- 1. the three settings are clicked, not written through the API ---
    // The preset buttons are named by their setting, not their caption: the preview
    // toolbar has its own “50%”, and that one only zooms the canvas.
    await page.getByRole('button', { name: 'DLSS 5', exact: true }).click();
    await page
      .getByRole('button', { name: '工作分辨率缩放 50%', exact: true })
      .click();
    await page
      .locator('select')
      .filter({ hasText: '仅光流' })
      .selectOption('1');
    await page.locator(BACKEND_SELECT).selectOption('gpu_transfer');

    // A click is a request until the service says it kept it. All three have to be
    // stored together: the device chain refuses on any one of them being off.
    await expect
      .poll(async () => {
        const s = await serviceSettings();
        return `${s.export_backend}|${s.guidance_mode}|${s.working_scale}|${s.flow_engine}`;
      }, { timeout: 20000 })
      .toBe(`gpu_transfer|1|0.5|nvof`);

    await page.getByRole('button', { name: '交付' }).click();
    const sent = await clickAddToQueue(page);
    expect(sent.settings.export_backend).toBe('gpu_transfer');
    expect(sent.settings.dlss_backend, 'the opt-in carries this build’s one DLSS host')
      .toBe(GPU_FLOW_BACKEND);
    expect(sent.settings.guidance_mode).toBe(1);
    expect(sent.settings.working_scale).toBe(0.5);
    expect(sent.settings.flow_engine).toBe('nvof');

    const queued = await lastQueuedJob();
    const jobId = queued.id;
    expect(queued.status).toBe('pending');

    // The panel sends skip_existing, so a deliverable left by an earlier run makes
    // this one complete without the chain ever running. Removing it first is what
    // keeps “the job finished” a statement about this run.
    if (queued.output_path && fs.existsSync(queued.output_path)) {
      fs.unlinkSync(queued.output_path);
    }

    // --- 2. render it for real, and let the service's own record answer ---
    await page.getByRole('button', { name: /全部渲染/ }).click();
    const settled = await waitForJob(jobId, 240000);
    expect(settled.error || '', `the job failed: ${settled.error}`).toBe('');
    expect(settled.status).toBe('completed');

    const frames = queued.total_frames;
    expect(frames, 'this test exists to bound a real export').toBeGreaterThanOrEqual(3);
    expect(frames).toBeLessThanOrEqual(8);

    const diag = settled.residual_backend_diagnostics || {};
    expect(diag.export_backend_requested, 'a skipped job carries no backend record')
      .toBe('gpu_transfer');
    expect(diag.export_backend_actual, 'a silent fallback would still say completed')
      .toBe('gpu_transfer');
    expect(diag.export_backend_fallback_reason).toBeNull();
    expect(diag.gpu_transfer_working_scale).toBe(0.5);
    expect(diag.gpu_transfer_resampling).toBe('exact_two_x');
    expect(diag.gpu_transfer_working_dims).toEqual([640, 360]);
    expect(diag.gpu_transfer_residual_backend).toBe('gpu_residual_bridge_d3d12');
    expect(diag.gpu_transfer_dlss_flow_calls, 'motion never reached DLSS')
      .toBe(frames);
    expect(diag.gpu_transfer_dlss_zero_motion_calls, 'a guided job answered with zero motion')
      .toBe(0);
    for (const field of ['decode_host_copy_bytes_measured',
                         'decode_host_copy_count_measured',
                         'dlss_host_color_stage_bytes_measured',
                         'dlss_output_readback_bytes_measured',
                         'residual_output_readback_bytes_measured',
                         'encoder_host_pixel_bytes_measured']) {
      expect(diag[field], `${field} is ${diag[field]}: a frame crossed into or out of ` +
              'the process, which is the claim this backend exists to make').toBe(0);
    }

    // --- 3. the deliverable, read back out of the container ---
    const outPath = settled.output_path;
    expect(outPath, 'a completed job that names no output is not a deliverable').toBeTruthy();
    expect(fs.existsSync(outPath), `no file at ${outPath}`).toBe(true);
    expect(fs.statSync(outPath).size).toBeGreaterThan(1024);
    const probe = probeMedia(outPath);
    const videoStream = probe.streams?.find((s: any) => s.codec_type === 'video');
    expect(videoStream, `${outPath} has no video stream`).toBeTruthy();
    expect(videoStream.codec_name).toBe('av1');
    expect(videoStream.width).toBe(1280);
    expect(videoStream.height).toBe(720);
    expect(videoStream.pix_fmt).toBe('yuv420p');
    expect(parseInt(videoStream.nb_read_packets, 10),
      'the packet count, not the exporter’s word, says how many frames left the encoder')
      .toBe(frames);

    // --- 4. one DLSS host, and the retired one is nowhere near it ---
    expect(settled.dlss_host?.dll, 'the job did not name the host that drew it')
      .toBe('dlss_gpu_host.dll');
    expect(settled.dlss_host?.backend).toBe(GPU_FLOW_BACKEND);
    expect(fs.existsSync(path.join(ROOT_DIR, 'native', 'dlssnr_host.dll')),
      'the retired host is back in native/').toBe(false);
    expect(diag.gpu_transfer_nvof_bridge_loaded, 'no optical-flow bridge was in the process')
      .toBe(true);
  });
});
