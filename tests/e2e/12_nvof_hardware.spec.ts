import { test, expect } from '@playwright/test';
import path from 'node:path';
import fs from 'node:fs';
import { launchApp } from '../helpers/electron_runner.js';
import { FIXTURE_120FPS_MKV, generateFixture120fps } from '../fixtures/generate_fixtures.js';
import { BACKEND_URL, LOCAL_TOKEN, getQueue, clearQueue } from '../helpers/backend_client.js';
import { importFixture } from '../helpers/ui.js';
import { ROOT_DIR } from '../helpers/paths.js';

test.describe('第十二组测试：真实 NVIDIA 硬件光流 (NVOF Hardware) 端到端集成', () => {
  let appInstance: any = null;

  test.beforeAll(async () => {
    generateFixture120fps();
  });

  test.beforeEach(async () => {
    await clearQueue();
  });

  test.afterEach(async () => {
    if (appInstance?.electronApp) {
      await appInstance.electronApp.close().catch(() => {});
      appInstance = null;
    }
  });

  test('12.1 验证 /api/flow/providers 硬件状态与前端 DLSS 页面 NVOF 真实展示', async () => {
    appInstance = await launchApp({ mockFilePath: FIXTURE_120FPS_MKV });
    const { page } = appInstance;

    // 1. Query /api/flow/providers directly from backend
    const res = await fetch(`${BACKEND_URL}/api/flow/providers`, {
      headers: { 'X-DLSS-Token': LOCAL_TOKEN },
    });
    expect(res.status).toBe(200);
    const data = await res.json();
    console.log('Backend flow providers:', JSON.stringify(data, null, 2));

    expect(data.providers).toBeDefined();
    const nvofProvider = data.providers.find((p: any) => p.engine === 'nvof');
    expect(nvofProvider).toBeDefined();
    expect(nvofProvider.available).toBe(true);
    expect(nvofProvider.hardware).toBe(true);
    expect(nvofProvider.device).toContain('RTX 5060 Ti');

    const disProvider = data.providers.find((p: any) => p.engine === 'dis');
    expect(disProvider).toBeDefined();
    expect(disProvider.hardware).toBe(false);
    expect(disProvider.device).toBe('cpu');

    // 2. Inspect UI on DLSS tab
    await importFixture(page);

    const engineSelect = page.locator('select').filter({ hasText: /NVIDIA Optical Flow/ });
    await expect(engineSelect).toBeVisible();

    // Verify UI shows real device name, separated grid and preset
    await expect(page.getByText('硬件光流加速 (RTX 5060 Ti)', { exact: false })).toBeVisible();
    const gridOption = page.locator('option', { hasText: '1x1 full-resolution output grid' });
    await expect(gridOption).toHaveCount(1);

    const presetOption = page.locator('option', { hasText: 'FAST - 极速 (推荐)' });
    await expect(presetOption).toHaveCount(1);

    // Verify accurate memory description
    await expect(page.getByText('内存流式传递模式 (0 字节磁盘写入)', { exact: false })).toBeVisible();
  });

  test('12.2 验证 flow 视图硬件光流实时计算与导出任务真实集成 (零 .flo 临时文件落盘)', async () => {
    test.setTimeout(240000);
    appInstance = await launchApp({ mockFilePath: FIXTURE_120FPS_MKV });
    const { page } = appInstance;

    await importFixture(page);

    // 1. Set settings to nvof + flow guidance
    await fetch(`${BACKEND_URL}/api/settings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-DLSS-Token': LOCAL_TOKEN,
      },
      body: JSON.stringify({
        flow_engine: 'nvof',
        flow_quality: '1F',
        guidance_mode: 1, // Flow guidance
      }),
    });

    // 2. Request a flow preview frame from backend
    const frameRes = await fetch(`${BACKEND_URL}/api/frame?frame=1&view=flow`, {
      headers: { 'X-DLSS-Token': LOCAL_TOKEN },
    });
    expect(frameRes.status).toBe(200);
    const contentType = frameRes.headers.get('content-type');
    expect(contentType).toContain('image/jpeg');
    const imgBuffer = await frameRes.arrayBuffer();
    expect(imgBuffer.byteLength).toBeGreaterThan(1024);

    // 3. Switch to Deliver tab and add to queue
    const deliverTabBtn = page.getByRole('button', { name: '交付' });
    await deliverTabBtn.click();
    await expect(page.getByText('渲染队列 (Render Queue)')).toBeVisible();

    const addToQueueBtn = page.getByRole('button', { name: /添加到渲染队列/i });
    await expect(addToQueueBtn).toBeVisible();
    await addToQueueBtn.click();

    // 4. Click Render All
    const renderAllBtn = page.getByRole('button', { name: /全部渲染|Render All/i });
    await expect(renderAllBtn).toBeEnabled();
    await renderAllBtn.click();

    // Wait for queue job to complete
    await expect(async () => {
      const q = await getQueue();
      const currentJob = q.jobs[0];
      expect(currentJob).toBeDefined();
      expect(currentJob.status).toBe('completed');
    }).toPass({ timeout: 180000, intervals: [2000] });

    // 5. Check no .flo files exist anywhere in project directory
    function searchFloFiles(dir: string): string[] {
      let results: string[] = [];
      try {
        const list = fs.readdirSync(dir);
        for (const file of list) {
          if (file === 'node_modules' || file === '.git' || file === 'build') continue;
          const fullPath = path.join(dir, file);
          const stat = fs.statSync(fullPath);
          if (stat.isDirectory()) {
            results = results.concat(searchFloFiles(fullPath));
          } else if (file.endsWith('.flo')) {
            results.push(fullPath);
          }
        }
      } catch {}
      return results;
    }

    const floFiles = searchFloFiles(ROOT_DIR);
    console.log('Found .flo files count:', floFiles.length);
    expect(floFiles.length).toBe(0);
  });
});
