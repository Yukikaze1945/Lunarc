import { test, expect, Page, Route } from '@playwright/test';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { ROOT_DIR } from '../helpers/paths.js';

/**
 * The experimental HDR10 opt-in driven through the real React tree, against a mocked
 * backend, with no GPU, no FFmpeg and no native host.
 *
 * What this can prove and the node tests cannot: the checkbox's patch, the delivery
 * page's one-click preset and the queue payload are the SAME settings by the time they
 * leave the page. A unit test of hdrExport.ts says the mirror and the preset agree; only
 * a rendered app says that the control writes the key the job carries, that the page's
 * own gate blocks the queue button while a refusal stands, and that an applied preset
 * unblocks it. Nothing here touches a device, so nothing here is evidence about HDR
 * output quality or about the backend gate, which stays authoritative.
 *
 * Preview frames are stubbed and WebSockets simply fail to connect (the app reconnects);
 * neither is under test.
 */

const DIST = path.join(ROOT_DIR, 'frontend', 'dist');
const UI_PORT = 5199;
const API = 'http://127.0.0.1:18923';

// 1x1 JPEG: the preview <img> stays real, the pixels do not need to be.
const STUB_JPEG = Buffer.from(
  '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVHV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD+f+iiigD/2Q==',
  'base64'
);

// The service's own defaults, as GET /api/settings answers them. export_backend starts
// on auto and the HDR pair starts off: the state the operator has to move to reach the
// chain, not a fixture that pre-opts the job in.
const SERVICE_SETTINGS: Record<string, unknown> = {
  preset: 1,
  style: 0,
  intensity: 1.0,
  local_tone: 1.0,
  local_struct: 1.0,
  skin_struct: 1.0,
  use_auto_mask: 1,
  ui_correction: 0,
  guidance_mode: 1,
  motion_scale_x: 1.0,
  motion_scale_y: 1.0,
  working_scale: 1.0,
  residual_pyramid: true,
  residual_strength: 1.0,
  flow_engine: 'nvof',
  flow_quality: '1F',
  export_backend: 'auto',
  reconstruction_mode: 'single',
  layer_scales: [1.0],
  layer_weights: [1.0],
  fusion_mode: 'catmull_baseline',
  fusion_beta: 0.1,
  fusion_max_iterations: 128,
  fusion_relative_tolerance: 0.001,
  dlss_backend: 'gpu_flow_d3d12',
  multi_pass: 1,
  experimental_hdr_scrgb: false,
  hdr_paper_white_scrgb: 8.0,
};

// An HDR10 clip, as the probe reports it: the source this chain is built for, so a
// refusal seen in the UI below can only be about the form, never about the file.
const VIDEO = {
  path: 'D:\\Media\\clip_hdr10.mkv',
  filename: 'clip_hdr10.mkv',
  frames: 600,
  fps: 59.94,
  width: 3840,
  height: 2160,
  duration_seconds: 10.01,
  pix_fmt: 'yuv420p10le',
  bit_depth: 10,
  is_hdr: true,
  color_transfer: 'smpte2084',
};

const CHECKBOX = 'input[aria-label="实验性 HDR/scRGB"]';
const PAPER_WHITE = 'input[aria-label="HDR 纸白 (scRGB)"]';
const HDR_REFUSALS = 'ul[aria-label="实验性 HDR 拒绝项"]';
const BACKEND_SELECT = 'select[aria-label="导出后端 (Export Backend)"]';
const HDR_PANEL = '[aria-label="实验性 HDR10 导出 (Experimental HDR10 Export)"]';
const PREVIEW = '[aria-label="设备链路预览 (Device Chain Preview)"]';
const OUTPUT_NAME = 'input[placeholder="输出视频文件名"]';

let staticServer: http.Server | null = null;

test.beforeAll(async () => {
  if (!fs.existsSync(path.join(DIST, 'index.html'))) {
    throw new Error(`frontend/dist/index.html is missing; run npm run build in frontend/ first`);
  }
  staticServer = http.createServer((req, res) => {
    const rel = decodeURIComponent((req.url || '/').split('?')[0]);
    const file = path.join(DIST, rel === '/' ? 'index.html' : rel);
    if (!file.startsWith(DIST) || !fs.existsSync(file)) {
      res.writeHead(404);
      res.end('not found');
      return;
    }
    const type = file.endsWith('.html') ? 'text/html'
      : file.endsWith('.js') ? 'text/javascript'
        : file.endsWith('.css') ? 'text/css' : 'application/octet-stream';
    res.writeHead(200, { 'Content-Type': `${type}; charset=utf-8` });
    res.end(fs.readFileSync(file));
  });
  await new Promise<void>((resolve) => staticServer!.listen(UI_PORT, '127.0.0.1', resolve));
});

test.afterAll(async () => {
  await new Promise<void>((resolve) => staticServer?.close(() => resolve()));
});

interface Mocked {
  settings: Record<string, unknown>;
  patches: Record<string, unknown>[];
  jobs: any[];
  alerts: string[];
}

/** The backend half of the app: the settings merge, the queue boundary, nothing else. */
async function openMockedUi(page: Page): Promise<Mocked> {
  const mock: Mocked = { settings: { ...SERVICE_SETTINGS }, patches: [], jobs: [], alerts: [] };

  page.on('dialog', (dialog) => {
    mock.alerts.push(dialog.message());
    return dialog.dismiss();
  });

  await page.route(`${API}/api/**`, async (route: Route) => {
    const url = new URL(route.request().url());
    const at = url.pathname;
    const method = route.request().method();
    const json = (body: unknown, status = 200) => route.fulfill({
      status,
      contentType: 'application/json',
      body: JSON.stringify(body),
    });

    if (at === '/api/system') {
      return json({
        gpu: 'Mock GPU', vram_total_gb: 24, vram_used_gb: 1, os: 'Windows',
        python_version: '3.11', backend_ready: true,
      });
    }
    if (at === '/api/flow/providers') return json({ providers: [], nvof_probe: { available: true } });
    if (at === '/api/frame') {
      return route.fulfill({ status: 200, contentType: 'image/jpeg', body: STUB_JPEG });
    }
    if (at === '/api/video/current') return json({ video: VIDEO });
    if (at === '/api/settings' && method === 'GET') return json({ ...mock.settings });
    if (at === '/api/settings' && method === 'POST') {
      const patch = JSON.parse(route.request().postData() || '{}');
      mock.patches.push(patch);
      // The service refuses a paper white it cannot read, and the refusal has to reach
      // the caller as the reason rather than as a 500.
      const white = patch['hdr_paper_white_scrgb'];
      if (white !== undefined && !Number.isFinite(Number(white))) {
        return json({
          detail: `hdr_paper_white_scrgb=${JSON.stringify(white)} is not a number; it is the `
            + 'diffuse white the HDR bridge is opened with (8.0 scRGB = 640 cd/m2)',
        }, 400);
      }
      Object.assign(mock.settings, patch);
      return json({ settings: { ...mock.settings } });
    }
    if (at === '/api/queue') return json({ jobs: [], active_id: null, queue_running: false });
    if (at === '/api/queue/add' && method === 'POST') {
      const body = JSON.parse(route.request().postData() || '{}');
      mock.jobs.push(body);
      return json({
        job: {
          id: `job-${mock.jobs.length}`, status: 'pending', progress: 0, current_frame: 0,
          name: body.name, input_path: body.input_path, output_path: body.output_path,
          settings: body.settings, transcode_settings: body.transcode_settings,
        },
      });
    }
    return json({ detail: `unmocked ${method} ${at}` }, 404);
  });

  await page.goto(`http://127.0.0.1:${UI_PORT}/index.html`);
  await expect(page.locator(CHECKBOX)).toBeVisible();
  return mock;
}

async function gotoDeliver(page: Page) {
  await page.getByRole('button', { name: '交付' }).click();
  await expect(page.locator(HDR_PANEL)).toBeVisible();
}

test('the checkbox, the preset and the queue payload are one chain', async ({ page }) => {
  const mock = await openMockedUi(page);

  // Off is the shipped state, and nothing about the HDR branch is asserted then.
  await expect(page.locator(CHECKBOX)).not.toBeChecked();
  await expect(page.locator(HDR_REFUSALS)).toHaveCount(0);

  // Opting in names the chain it needs instead of moving the job onto it.
  await page.locator(CHECKBOX).check();
  expect(mock.patches.at(-1)).toEqual({ experimental_hdr_scrgb: true });
  const refusal = page.locator(HDR_REFUSALS).first();
  await expect(refusal).toBeVisible();
  await expect(refusal).toContainText('export_backend=auto');

  // Moving the chain clears that clause: the patch carried through to the settings blob
  // the job is built from, not just to the control.
  await page.locator(BACKEND_SELECT).selectOption('gpu_transfer');
  await expect(page.locator(HDR_REFUSALS)).toHaveCount(0);
  expect(mock.settings.export_backend).toBe('gpu_transfer');

  await gotoDeliver(page);
  await expect(page.locator(PREVIEW)).toContainText('实验性 HDR10 GPU Transfer');
  await expect(page.locator(PREVIEW)).toContainText('NOT_CHARACTERIZED');
  await expect(page.locator(PREVIEW)).toContainText('unavailable/None');
  await expect(page.locator(PREVIEW)).toContainText('未测量不是零');
  // The rewrap is named as the helper it reuses, with the clause that keeps it a copy.
  await expect(page.locator(PREVIEW)).toContainText('gtx.build_mux_command');
  await expect(page.locator(PREVIEW)).toContainText('-c:v copy');
  await expect(page.locator('button[aria-label="复制导出命令 (Copy Export Command)"]'))
    .toBeDisabled();

  // The encoder request as shipped is av1 with the streams being copied: refused by this
  // chain, listed as a refusal, and the queue button is blocked rather than re-routed.
  const badge = page.locator('div', { hasText: '实验性 HDR10：codec=av1_nvenc' });
  await expect(badge.first()).toBeVisible();
  // The output directory is the one the page derived from the loaded clip; only the
  // name is typed, because a job that would overwrite the source is refused earlier.
  await page.locator(OUTPUT_NAME).fill('clip_hdr10_out');
  await page.getByRole('button', { name: /添加到渲染队列/ }).click();
  await expect.poll(() => mock.alerts.length).toBe(1);
  expect(mock.jobs).toEqual([]);
  expect(String(mock.alerts[0])).toContain('codec=av1_nvenc');

  // The disclosure lists the choices before any of them is written - and the mapping the
  // form already selects is listed as kept, not as something the click trades away.
  await page.getByRole('button', { name: '准备实验性 HDR10 导出' }).click();
  const summary = page.locator('ul[aria-label="实验性 HDR10 准备摘要"]');
  await expect(summary).toContainText('codec=hevc_nvenc');
  await expect(summary).toContainText('映射：audio_mode、subtitle_mode、copy_attachments、'
    + 'copy_metadata、copy_chapters 保持当前选择');
  await expect(summary).not.toContainText('audio_mode=remove');
  await expect(summary).not.toContainText('copy_metadata=false');
  expect(mock.jobs).toEqual([]);

  await page.getByRole('button', { name: /^应用：准备实验性 HDR10 导出/ }).click();
  await expect(page.locator(BACKEND_SELECT)).toHaveValue('gpu_transfer');
  await expect(page.locator('div', { hasText: '实验性 HDR10：codec=av1_nvenc' })).toHaveCount(0);
  const applied = mock.patches.at(-1) as any;
  expect(applied.experimental_hdr_scrgb).toBe(true);
  expect(applied.export_backend).toBe('gpu_transfer');
  expect(applied.reconstruction_mode).toBeUndefined();
  // The two selects the branch gates: an option that is still disabled cannot be the
  // value the control shows, so this reads the encoder the click actually wrote.
  await expect(page.locator('label:text-is("编码器 (Codec)") + select')).toHaveValue('hevc_nvenc');
  await expect(page.locator('label:text-is("像素格式 (Pixel Format)") + select')).toHaveValue('p010le');

  // The same click now reaches the queue, and the job carries what the page was showing.
  await page.getByRole('button', { name: /添加到渲染队列/ }).click();
  await expect.poll(() => mock.jobs.length).toBe(1);
  const job = mock.jobs[0];
  expect(job.settings.experimental_hdr_scrgb).toBe(true);
  expect(job.settings.export_backend).toBe('gpu_transfer');
  expect(job.settings.dlss_backend).toBe('gpu_flow_d3d12');
  expect(job.settings.reconstruction_mode).toBe('single');
  expect(job.settings.layer_scales).toEqual([1.0]);
  expect(job.settings.hdr_paper_white_scrgb).toBe(8);
  expect(job.transcode_settings.codec).toBe('hevc_nvenc');
  expect(job.transcode_settings.profile).toBe('main10');
  expect(job.transcode_settings.pixel_format).toBe('p010le');
  expect(job.transcode_settings.container).toBe('.mkv');
  // The mapping is carried to the queue as the form had it: the reused copy pass is what
  // takes it across, so the readiness patch has no reason to switch the streams off.
  expect(job.transcode_settings.audio_mode).toBe('copy');
  expect(job.transcode_settings.subtitle_mode).toBe('copy');
  expect(job.transcode_settings.copy_attachments).toBe(true);
  expect(job.transcode_settings.copy_metadata).toBe(true);
  expect(job.transcode_settings.copy_chapters).toBe(true);
  expect(job.transcode_settings.custom_color_space).toBe('');
  expect(job.transcode_settings.extra_args).toBe('');
  expect(job.output_path).toBe('D:\\Media\\clip_hdr10_out.mkv');
  // Original extent, because the encoder is fed the full-size composite.
  expect(job.transcode_settings.resolution_mode).toBe('original');
});

test('turning the switch off leaves the SDR job unpoliced, and a reload reads back the store',
  async ({ page }) => {
    const mock = await openMockedUi(page);

    await page.locator(CHECKBOX).check();
    await page.locator(BACKEND_SELECT).selectOption('gpu_transfer');
    await gotoDeliver(page);
    await expect(page.locator(PREVIEW)).toContainText('HDR10');

    // A reload answers from the stored settings, so the checkbox reflects the service
    // rather than the type default.
    await page.reload();
    await expect(page.locator(CHECKBOX)).toBeChecked();
    await expect(page.locator(BACKEND_SELECT)).toHaveValue('gpu_transfer');

    await page.locator(CHECKBOX).uncheck();
    expect(mock.patches.at(-1)).toEqual({ experimental_hdr_scrgb: false });
    await gotoDeliver(page);
    // Off means the SDR device chain: same reused copy pass, 8-bit NV12 pictures.
    await expect(page.locator(PREVIEW)).toContainText('-c:v copy');
    await expect(page.locator(PREVIEW)).not.toContainText('scRGB');
    await expect(page.locator(HDR_REFUSALS)).toHaveCount(0);

    await page.locator(OUTPUT_NAME).fill('clip_out');
    await page.getByRole('button', { name: /添加到渲染队列/ }).click();
    await expect.poll(() => mock.jobs.length).toBe(1);
    expect(mock.jobs[0].settings.experimental_hdr_scrgb).toBe(false);
    expect(mock.jobs[0].transcode_settings.codec).toBe('av1_nvenc');
    expect(mock.jobs[0].transcode_settings.audio_mode).toBe('copy');
    // Turning the switch off did not quietly rewrite the encoder or the destination the
    // operator had.
    expect(mock.jobs[0].output_path).toBe('D:\\Media\\clip_out.mkv');
  });

test('a paper white the gate would refuse never reaches the service', async ({ page }) => {
  const mock = await openMockedUi(page);

  await page.locator(CHECKBOX).check();
  await page.locator(PAPER_WHITE).fill('abc');
  await expect(page.locator('p', { hasText: '不是数字' }).first()).toBeVisible();
  expect(mock.patches.filter((p) => 'hdr_paper_white_scrgb' in p)).toEqual([]);

  // Refused at the control, not corrected on the way out: the stored value is untouched.
  expect(mock.settings.hdr_paper_white_scrgb).toBe(8.0);

  await page.locator(PAPER_WHITE).fill('20');
  await expect.poll(() => mock.settings.hdr_paper_white_scrgb).toBe(20);
  await expect(page.locator('p', { hasText: '20 scRGB = 1600 cd/m²' }).first()).toBeVisible();
  await gotoDeliver(page);
  await expect(page.locator(HDR_PANEL)).toContainText('纸白 20 scRGB = 1600 cd/m²');
});
