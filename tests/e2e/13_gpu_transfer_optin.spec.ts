import { test, expect } from '@playwright/test';
import path from 'node:path';
import { launchApp } from '../helpers/electron_runner.js';
import { FIXTURE_120FPS_MKV, generateFixture120fps } from '../fixtures/generate_fixtures.js';
import {
  BACKEND_URL,
  LOCAL_TOKEN,
  addJob,
  clearQueue,
  getQueue,
  requestJson,
  waitForJob,
} from '../helpers/backend_client.js';
import { importFixture } from '../helpers/ui.js';

/**
 * GPU Transfer 前端 opt-in 与默认 GPU 宿主的行为级验收。
 *
 * 这里守的是五条真实路径，不是源码里的字符串（DLSS 宿主默认已从旧宿主改为本项目 GPU 宿主）：
 *   1. 默认状态：交付页发出的 /api/queue/add 请求体带 export_backend=auto，且 DLSS 宿主
 *      默认就是 GPU 宿主（dlss_backend=gpu_flow_d3d12）——auto/legacy 管道作业跑在它上面；
 *   2. opt-in：在 DLSS 页选一次 GPU Transfer，选择必须原样落到同一个请求体的
 *      export_backend 上，并带上链路唯一能接受的 dlss_backend；切回 Auto 只换导出链路，
 *      不会把 DLSS 宿主交还给旧宿主；
 *   3. 服务端拒绝：条件不满足的作业失败后，拒绝原因要能从既有队列 API 读到，
 *      并且交付页的作业行里看得见；
 *   4. 加入队列那一刻就被拒的作业，原因必须传到用户面前，而不是一个状态码；
 *   5. 预览主机：这一版只有一座原生 DLSS 宿主，guidance_mode=0 预览开在它的零运动入口上
 *      （服务日志里读得到 kind=gpu_zero_motion），页面重载后仍是；界面上不再有选宿主的
 *      控件，而任何仍显式点名旧宿主 dlssnr_host.dll 的设置写入会在 /api/settings 边界
 *      被 400 拒掉并写明迁移方式，既不改动服务设置，也不把预览挪到别处。
 *
 * 预览帧走 mock：这一组验的是作业参数接线与拒绝原因的可达性，不需要真的跑一次
 * 神经推理（那是硬件导出组的活）。设置、队列、校验全部走真实后端。14.5 例外：它验
 * 的就是预览主机本身，所以它走真实后端、真实解码帧与真实推理。
 */

// 1x1 JPEG: the preview socket stays real, the pixels do not need to be.
const STUB_JPEG = Buffer.from(
  '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD+f+iiigD/2Q==',
  'base64'
);

const BACKEND_SELECT = 'select[aria-label="导出后端 (Export Backend)"]';
const ADD_BUTTON = /添加到渲染队列/;
const GPU_FLOW_BACKEND = 'gpu_flow_d3d12';
// The settings blob an opt-in client sends: every precondition the chain asks
// for, so a refusal here can only be about the field under test.
const GPU_READY_SETTINGS = {
  guidance_mode: 1,
  flow_engine: 'nvof',
  working_scale: 0.5,
  residual_pyramid: true,
};

let appInstance: any = null;

async function launchWithStubbedPreview() {
  appInstance = await launchApp({ mockFilePath: FIXTURE_120FPS_MKV });
  const { page } = appInstance;
  await page.route('**/api/frame**', (route: any) =>
    route.fulfill({ contentType: 'image/jpeg', body: STUB_JPEG })
  );
  // Record every dialog; the runner's handler is the one that dismisses.
  const dialogs: string[] = [];
  page.on('dialog', (d: any) => dialogs.push(d.message()));
  return { page, dialogs };
}

async function deliverTab(page: any) {
  await page.getByRole('button', { name: '交付' }).click();
}

async function dlssTab(page: any) {
  // exact: the DLSS page itself carries a "DLSS 5 增强" view button, which
  // matches the same name as soon as that page is the one on screen.
  await page.getByRole('button', { name: 'DLSS 5', exact: true }).click();
}

// The request body is the contract: click the real button, capture what the
// renderer actually put on the wire, and let the server keep its own copy so
// the same assertion can be re-checked from the queue API.
async function clickAddToQueue(page: any) {
  const [request] = await Promise.all([
    page.waitForRequest(
      (r: any) => r.url().includes('/api/queue/add') && r.method() === 'POST',
      { timeout: 20000 }
    ),
    page.getByRole('button', { name: ADD_BUTTON }).first().click(),
  ]);
  return request.postDataJSON();
}

async function lastQueuedJob() {
  const queue = await getQueue();
  const jobs = queue.jobs || [];
  expect(jobs.length).toBeGreaterThan(0);
  return jobs[jobs.length - 1];
}

async function serviceSettings() {
  return requestJson('/api/settings');
}

// The service names the host it opened in its own log; which entry point a frame
// went through is not something the picture can show.
async function previewSessionLines() {
  const logs = await requestJson('/api/diagnostics/logs?limit=500');
  const preview = (logs.logs || []).filter((l: any) => l.tag === 'Preview');
  return {
    kinds: preview.filter((l: any) => String(l.message).includes('kind='))
      .map((l: any) => String(l.message)),
    errors: preview.filter((l: any) => l.level === 'ERROR'),
  };
}

// A real preview frame: the route under test is the one that renders it.
async function requestPreviewFrame(frame: number) {
  const res = await fetch(`${BACKEND_URL}/api/frame?frame=${frame}&view=dlss`, {
    headers: { 'X-DLSS-Token': LOCAL_TOKEN },
  });
  expect(res.status).toBe(200);
  expect(res.headers.get('content-type') || '').toContain('image/jpeg');
  const bytes = await res.arrayBuffer();
  expect(bytes.byteLength).toBeGreaterThan(1024);
}

test.describe('第十四组测试：GPU Transfer 前端 opt-in（行为级）', () => {
  test.beforeAll(() => {
    generateFixture120fps();
  });

  test.afterEach(async () => {
    if (appInstance?.electronApp) {
      await appInstance.electronApp.close().catch(() => {});
      appInstance = null;
    }
  });

  test('14.1 默认 auto 作业跑在默认 GPU 宿主上', async () => {
    const { page } = await launchWithStubbedPreview();
    await importFixture(page);

    const backendSelect = page.locator(BACKEND_SELECT);
    await expect(backendSelect).toBeVisible();
    await expect(backendSelect).toHaveValue('auto');

    await deliverTab(page);
    const sent = await clickAddToQueue(page);

    // The export chain stays the legacy pipe, but the DLSS host defaults to the
    // project GPU host: an auto/legacy job runs its frames there now.
    expect(sent.settings.export_backend).toBe('auto');
    expect(sent.settings.dlss_backend).toBe(GPU_FLOW_BACKEND);

    // The same job, as the service stored it: the panel's host carried through.
    const job = await lastQueuedJob();
    expect(job.settings.export_backend).toBe('auto');
    expect(job.settings.dlss_backend).toBe(GPU_FLOW_BACKEND);
    expect(job.status).toBe('pending');
    await expect(page.getByText('共 1 个任务')).toBeVisible();
  });

  test('14.2 选择 GPU Transfer 后 export_backend 与 dlss_backend 一起到达队列，切回 Auto 仅收回导出链路', async () => {
    const { page } = await launchWithStubbedPreview();
    await importFixture(page);

    const backendSelect = page.locator(BACKEND_SELECT);
    await expect(backendSelect).toBeVisible();

    // The panel has to state the preconditions it will refuse on, and must not
    // dress an unbenchmarked opt-in up as a speed win.
    const panel = backendSelect.locator('xpath=..');
    const optionTexts = await panel.locator('option').allInnerTexts();
    const panelText = `${await panel.innerText()}\n${optionTexts.join('\n')}`;
    for (const condition of ['NVOF', '50%', '100%', '8-bit SDR', 'NVENC']) {
      expect(panelText, `opt-in panel omits the ${condition} precondition`).toContain(condition);
    }
    expect(panelText).toContain('拒绝');
    expect(panelText).toContain('不会自动回退');
    for (const claim of ['更快', '加速', '性能提升', '零拷贝']) {
      expect(panelText, `opt-in panel oversells with ${claim}`).not.toContain(claim);
    }

    await backendSelect.selectOption('gpu_transfer');
    // Selection is a setting, not a local decoration: it has to survive the
    // round trip to the service the Deliver page reads its settings from.
    await expect
      .poll(async () => (await requestJson('/api/settings')).export_backend, { timeout: 15000 })
      .toBe('gpu_transfer');

    await deliverTab(page);
    const sent = await clickAddToQueue(page);
    expect(sent.settings.export_backend).toBe('gpu_transfer');
    expect(sent.settings.dlss_backend).toBe(GPU_FLOW_BACKEND);

    const job = await lastQueuedJob();
    expect(job.settings.export_backend).toBe('gpu_transfer');
    expect(job.settings.dlss_backend).toBe(GPU_FLOW_BACKEND);

    // Switching the chain back is not switching hosts: the export-backend control
    // writes only export_backend, so the DLSS host stays the default GPU one and
    // an auto job still carries it. Returning to the old host takes its own control.
    await dlssTab(page);
    await page.locator(BACKEND_SELECT).selectOption('auto');
    await deliverTab(page);
    const reverted = await clickAddToQueue(page);
    expect(reverted.settings.export_backend).toBe('auto');
    expect(reverted.settings.dlss_backend).toBe(GPU_FLOW_BACKEND);
  });

  test('14.3 服务端拒绝的作业，拒绝原因能从队列 API 与交付页看到', async () => {
    const { page } = await launchWithStubbedPreview();
    await importFixture(page);
    await clearQueue();

    // A job the queue can accept and then fails on: the device chain wants the flow
    // descriptor on the card, which only NVIDIA's engine produces. The host is no
    // longer a thing a job can get wrong — there is one host, and naming the retired
    // one is refused where the settings are read (see 14.5).
    // The precondition bundle goes first: spread last, it also re-wrote
    // flow_engine back to nvof and queued a job the device chain accepts, which
    // then ran to a deliverable instead of being refused.
    const added = await addJob({
      name: 'GpuTransferCpuFlow',
      input_path: FIXTURE_120FPS_MKV,
      output_path: path.join(path.dirname(FIXTURE_120FPS_MKV), 'gpu_transfer_cpu_flow.mkv'),
      settings: { ...GPU_READY_SETTINGS, export_backend: 'gpu_transfer', flow_engine: 'dis' },
    });
    const jobId = added.job.id;

    await deliverTab(page);
    await expect(page.getByText('GpuTransferCpuFlow')).toBeVisible({ timeout: 20000 });
    await page.getByRole('button', { name: /全部渲染/ }).click();

    const settled = await waitForJob(jobId, 60000);
    expect(settled.status).toBe('failed');
    expect(settled.error).toContain('export_backend=gpu_transfer');
    expect(settled.error).toContain('flow_engine=nvof');

    // The reason is not just in the JSON: it is the text the operator reads.
    const row = page.getByText(/失败:/);
    await expect(row).toBeVisible({ timeout: 20000 });
    await expect(row).toContainText('flow_engine=nvof');

    await page.getByRole('button', { name: /查看诊断调用栈/ }).first().click();
    await expect(page.getByText('flow_engine=nvof').first()).toBeVisible();
  });

  test('14.4 加入队列那一刻的拒绝把原因传到界面，而不是留下一个状态码', async () => {
    const { page, dialogs } = await launchWithStubbedPreview();
    await importFixture(page);
    await clearQueue();

    // 1) Against the real service: a backend this build does not offer is a 400
    //    that names the reason and the choices, not an empty 500.
    const rejected = await fetch(`${BACKEND_URL}/api/queue/add`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-DLSS-Token': LOCAL_TOKEN },
      body: JSON.stringify({ name: 'UnknownBackend', settings: { export_backend: 'gpu-transfer' } }),
    });
    const rejectionBody = await rejected.json().catch(() => ({}));
    expect(rejected.status).toBe(400);
    expect(rejectionBody.detail).toContain('export_backend');
    expect(rejectionBody.detail).toContain('gpu_transfer');
    expect(await getQueue()).toEqual({ jobs: [], active_id: null });

    // 2) And the path a refusal takes back into the page: whatever the service
    //    says in `detail` is what the operator is told, verbatim.
    const reason = 'export_backend=gpu_transfer 需要 dlss_backend=gpu_flow_d3d12';
    await page.route('**/api/queue/add', (route: any) =>
      route.fulfill({
        status: 400,
        contentType: 'application/json',
        body: JSON.stringify({ detail: reason }),
      })
    );
    await deliverTab(page);
    const [request] = await Promise.all([
      page.waitForRequest(
        (r: any) => r.url().includes('/api/queue/add') && r.method() === 'POST',
        { timeout: 20000 }
      ),
      page.getByRole('button', { name: ADD_BUTTON }).first().click(),
    ]);
    await expect.poll(() => dialogs.length, { timeout: 15000 }).toBeGreaterThan(0);
    expect(dialogs.join('\n')).toContain(reason);
    expect(request.url()).toContain('/api/queue/add');
  });

  test('14.5 预览只有一座 GPU 宿主，点名旧宿主的设置写入在边界被拒', async () => {
    test.setTimeout(300000);
    // No frame stub on this one: the question is which native host answers the
    // preview, and a mocked /api/frame would answer nothing.
    appInstance = await launchApp({ mockFilePath: FIXTURE_120FPS_MKV });
    const { page } = appInstance;
    await importFixture(page);
    await dlssTab(page);

    // guidance_mode=0 is what the GPU host's resident zero-motion texture serves,
    // and there is no other host to serve it.
    await requestJson('/api/settings', { method: 'POST', body: { guidance_mode: 0 } });
    const before = await serviceSettings();
    expect(before.guidance_mode).toBe(0);
    expect(before.dlss_backend, 'the default DLSS host is no longer the old one')
      .toBe(GPU_FLOW_BACKEND);

    await requestPreviewFrame(0);
    const gpu = await previewSessionLines();
    expect(gpu.errors, 'a preview frame failed to enhance').toEqual([]);
    expect(gpu.kinds.some((m: string) => m.includes('kind=gpu_zero_motion')),
      'the preview never opened the GPU zero-motion session').toBe(true);
    for (const m of gpu.kinds) {
      expect(m, `the preview fell back to a second host: ${m}`).not.toContain('kind=legacy');
    }

    // The host is a stated fact of the build, not a choice the page can make: the
    // select is gone and the panel names the binary that draws the picture.
    await expect(page.locator('select[aria-label="DLSS 宿主 (DLSS Host)"]')).toHaveCount(0);
    await expect(page.getByText('唯一 GPU 宿主（native/dlss_gpu_host.dll）')).toBeVisible();

    // A pre-migration caller that still names dlssnr_host.dll is refused at the API
    // boundary with the reason, not silently served by the GPU host and not a 500.
    const refused = await fetch(`${BACKEND_URL}/api/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-DLSS-Token': LOCAL_TOKEN },
      body: JSON.stringify({ dlss_backend: 'host_pointer' }),
    });
    const refusal = await refused.json().catch(() => ({}));
    expect(refused.status).toBe(400);
    for (const part of ['host_pointer', 'dlssnr_host.dll', GPU_FLOW_BACKEND]) {
      expect(String(refusal.detail), 'the refusal must be a migration, not a status code')
        .toContain(part);
    }
    expect((await serviceSettings()).dlss_backend, 'a refused write must not have been merged')
      .toBe(GPU_FLOW_BACKEND);

    // The default host is a service setting, not a page-local decoration: it survives
    // a reload, and the refused write left the preview route alone.
    await page.reload({ waitUntil: 'load' });
    await expect(page.getByText('唯一 GPU 宿主（native/dlss_gpu_host.dll）')).toBeVisible({
      timeout: 30000,
    });
    expect((await serviceSettings()).dlss_backend).toBe(GPU_FLOW_BACKEND);
    await requestPreviewFrame(1);
    const afterReload = await previewSessionLines();
    expect(afterReload.errors).toEqual([]);
    for (const m of afterReload.kinds) {
      expect(m, 'reloaded preview went to a second host').not.toContain('kind=legacy');
    }

    // The job the page queues carries that one host, never a dropped or downgraded key.
    await deliverTab(page);
    const sent = await clickAddToQueue(page);
    expect(sent.settings.export_backend).toBe('auto');
    expect(sent.settings.dlss_backend).toBe(GPU_FLOW_BACKEND);
  });
});
