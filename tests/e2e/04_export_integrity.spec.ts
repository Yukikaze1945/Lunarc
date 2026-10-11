import { test, expect } from '@playwright/test';
import path from 'node:path';
import fs from 'node:fs';
import { launchApp } from '../helpers/electron_runner.js';
import { FIXTURE_120FPS_MKV, generateFixture120fps } from '../fixtures/generate_fixtures.js';
import { verifyVideoIntegrity } from '../helpers/media_probe.js';
import { getQueue, clearQueue } from '../helpers/backend_client.js';
import { importFixture } from '../helpers/ui.js';

test.describe('第四组测试：导出完整性 (Export Integrity & Physical ffprobe Verification)', () => {
  let appInstance: any = null;
  const expectedOutputDir = path.dirname(FIXTURE_120FPS_MKV);
  const expectedOutputFile = path.join(expectedOutputDir, 'fixture_120fps_dlss.mkv');

  test.beforeAll(async () => {
    generateFixture120fps();
    if (fs.existsSync(expectedOutputFile)) {
      try {
        fs.unlinkSync(expectedOutputFile);
      } catch {}
    }
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

  test('4.1 导入 5s/120fps/600帧 素材，通过 UI 加入渲染队列并导出，经 ffprobe 物理强校验通过', async () => {
    test.setTimeout(240000);

    // 1. Launch app with fixture
    appInstance = await launchApp({ mockFilePath: FIXTURE_120FPS_MKV });
    const { page } = appInstance;

    // 2. Import material (shared helper retries once on transient races)
    await importFixture(page);

    // 3. Switch to Deliver tab
    const deliverTabBtn = page.getByRole('button', { name: '交付' });
    await deliverTabBtn.click();
    await expect(page.getByText('渲染队列 (Render Queue)')).toBeVisible();

    // 4. Click "添加到渲染队列"
    const addToQueueBtn = page.getByRole('button', { name: /添加到渲染队列/i });
    await expect(addToQueueBtn).toBeVisible();
    await addToQueueBtn.click();

    // Wait for job to appear in queue table
    await expect(page.locator('text=fixture_120fps_dlss').first()).toBeVisible({ timeout: 10000 });

    // 5. Click "全部渲染 (Render All)"
    const renderAllBtn = page.getByRole('button', { name: /全部渲染|Render All/i });
    await expect(renderAllBtn).toBeEnabled();
    await renderAllBtn.click();

    console.log('[Test 4.1] Export job started, waiting for completion...');

    // 6. Poll queue status until job is completed, failed, or timed out (up to 200s)
    let completedJob: any = null;
    const maxWaitMs = 200000;
    const startWait = Date.now();
    let lastLogT = 0;

    while (Date.now() - startWait < maxWaitMs) {
      const q = await getQueue();
      const job = q.jobs?.slice().reverse().find((j: any) => j.name?.includes('fixture_120fps_dlss'));
      if (job) {
        if (Date.now() - lastLogT > 5000) {
          console.log(`[Test 4.1] Progress: ${job.progress}% (${job.current_frame}/${job.total_frames}) | FPS: ${job.fps} | Status: ${job.status}`);
          lastLogT = Date.now();
        }
        if (job.status === 'completed') {
          completedJob = job;
          break;
        }
        if (job.status === 'failed') {
          throw new Error(`Export job FAILED in backend: ${job.error || 'Unknown error'}\n${job.dlss_log_tail || ''}`);
        }
      }
      await page.waitForTimeout(1000);
    }

    expect(completedJob, 'Job did not reach completed state within timeout').not.toBeNull();
    console.log('[Test 4.1] UI reports job completed. Now executing STRICT ffprobe physical inspection...');

    // 7. STRICT FFPROBE PHYSICAL VERIFICATION (DO NOT TRUST UI ALONE!)
    const physicalOutPath = completedJob.output_path || expectedOutputFile;
    expect(fs.existsSync(physicalOutPath), `Physical output file does not exist at: ${physicalOutPath}`).toBe(true);

    const verification = verifyVideoIntegrity(physicalOutPath, {
      minDuration: 4.9,
      maxDuration: 5.1,
      minFrames: 595,
      maxFrames: 605,
      width: 1280,
      height: 720,
      codec: 'av1',
      requireAudio: true,
    });

    console.log('[Test 4.1] Physical ffprobe verification result:', JSON.stringify(verification.details, null, 2));

    if (!verification.ok) {
      console.error('[Test 4.1] ffprobe Verification ERRORS:', verification.errors);
    }

    expect(verification.ok, `Physical output verification FAILED:\n${verification.errors.join('\n')}`).toBe(true);
    expect(verification.details?.hasAudio, 'Physical output must contain audio stream').toBe(true);
    expect(verification.details?.frames, 'Physical frames must be ~600').toBeGreaterThanOrEqual(595);
    expect(verification.details?.frames).toBeLessThanOrEqual(605);
  });
});
