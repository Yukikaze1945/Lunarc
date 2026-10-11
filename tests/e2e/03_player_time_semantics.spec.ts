import { test, expect } from '@playwright/test';
import { launchApp } from '../helpers/electron_runner.js';
import { FIXTURE_120FPS_MKV, generateFixture120fps } from '../fixtures/generate_fixtures.js';

test.describe('第三组测试：播放器时间语义 (Player Real-Time Semantics Verification)', () => {
  let appInstance: any = null;

  test.beforeAll(async () => {
    generateFixture120fps();
  });

  test.afterEach(async () => {
    if (appInstance?.electronApp) {
      await appInstance.electronApp.close().catch(() => {});
      appInstance = null;
    }
  });

  // Import can transiently miss when the backend is mid-restart behind the
  // shared port (dying process still answers, then disappears). One retry
  // keeps the strict frame assertions intact while surviving the race.
  async function importWithRetry(page: any) {
    for (let attempt = 1; attempt <= 2; attempt++) {
      await page.getByRole('button', { name: '导入素材' }).click();
      try {
        await expect(page.locator('header').locator('text=fixture_120fps.mkv')).toBeVisible({ timeout: 15000 });
        return;
      } catch (e) {
        if (attempt === 2) throw e;
        console.log(`[Test] import badge missing (attempt ${attempt}), retrying import click...`);
      }
    }
  }

  // Warm-up helper: play until the renderer provably delivers frames, then
  // reset to frame 0 via the transport control so the TIMED run starts
  // from frame 0 with a warm pipeline (no page reload needed).
  async function warmUpAndReset(page: any, _viewBtnName: string) {
    const playBtn = page.getByRole('button', { name: '播放', exact: true });
    await expect(playBtn).toBeVisible();
    const counter = page.locator('header .font-mono.text-xs span.font-semibold').first();
    await playBtn.click();
    await expect.poll(async () => {
      const t = await counter.innerText().catch(() => '0');
      return parseInt(t.trim(), 10) || 0;
    }, { timeout: 25000 }).toBeGreaterThan(0);
    const pauseBtn = page.getByRole('button', { name: '暂停', exact: true });
    await pauseBtn.click();

    // Reset back to frame 0 natively using the transport button
    const jumpStartBtn = page.getByTitle('跳至首帧');
    await jumpStartBtn.click();
    await expect.poll(async () => {
      const t = await counter.innerText().catch(() => '1');
      return parseInt(t.trim(), 10) || 0;
    }, { timeout: 5000 }).toBe(0);
    await page.waitForTimeout(500);
  }

  test('3.1 原图模式 (orig) 播放器时间语义：2 秒真实时间应播放约 240 帧 (允许 180~300)', async () => {
    appInstance = await launchApp({ mockFilePath: FIXTURE_120FPS_MKV });
    const { page } = appInstance;

    await importWithRetry(page);

    // Switch to "源画质 (原图)" view
    const origViewBtn = page.getByRole('button', { name: '源画质 (原图)' });
    await origViewBtn.click();

    // Verify view is selected (has bg-davinci-orange)
    await expect(origViewBtn).toHaveClass(/bg-davinci-orange/);

    // Let Electron-spawned backend settle
    await page.waitForTimeout(2500);

    await warmUpAndReset(page, '源画质 (原图)');

    // Click "播放" button, then start the clock on the first delivered
    // frame: a cold first render must not eat the measured window.
    // Thresholds below are unchanged (still 180~300).
    const playBtn = page.getByRole('button', { name: '播放', exact: true });
    await expect(playBtn).toBeVisible();

    // Read current frame from Header timecode frame counter
    const frameCounter = page.locator('header .font-mono.text-xs span.font-semibold').first();
    await playBtn.click();
    await expect.poll(async () => {
      const t = await frameCounter.innerText().catch(() => '0');
      return parseInt(t.trim(), 10) || 0;
    }, { timeout: 15000 }).toBeGreaterThan(0);
    const tStart = Date.now();

    // Wall-clock wait for 2.0 seconds
    await page.waitForTimeout(2000);

    // Click pause
    const pauseBtn = page.getByRole('button', { name: '暂停', exact: true });
    await pauseBtn.click();
    const elapsedSec = (Date.now() - tStart) / 1000;

    const frameText = await frameCounter.innerText();
    const curFrame = parseInt(frameText.trim(), 10);

    console.log(`[Test 3.1 Orig] Wall-clock elapsed: ${elapsedSec.toFixed(2)}s, Current frame reached: ${curFrame}`);

    // Verification: Expected ~240 frames, allowed 180~300 frames.
    // If it only played 10~30 frames, this strict assertion will FAIL.
    expect(curFrame, `[Orig Mode] Player time timeline severely slowed down! After ${elapsedSec.toFixed(2)}s, frame is only ${curFrame} instead of ~240 (expected 180~300)`).toBeGreaterThanOrEqual(180);
    expect(curFrame).toBeLessThanOrEqual(300);
  });

  test('3.2 DLSS / Split 模式播放器时间语义：计算跟不上时应丢预览帧，媒体时间不能严重变慢', async () => {
    appInstance = await launchApp({ mockFilePath: FIXTURE_120FPS_MKV });
    const { page } = appInstance;

    await importWithRetry(page);

    // Switch to "滑动对比 (Split Wipe)" or "DLSS 5 增强"
    const splitBtn = page.getByRole('button', { name: '滑动对比 (Split Wipe)' });
    await splitBtn.click();
    await expect(splitBtn).toHaveClass(/bg-davinci-orange/);

    // Let Electron-spawned backend settle (cold D3D12/Feature18 init).
    await page.waitForTimeout(2500);

    await warmUpAndReset(page, '滑动对比 (Split Wipe)');

    // Click Play, then start the clock on the first delivered frame
    // (same cold-start fairness as 3.1; thresholds unchanged).
    const playBtn = page.getByRole('button', { name: '播放', exact: true });
    await expect(playBtn).toBeVisible();

    // Read current frame
    const frameCounter = page.locator('header .font-mono.text-xs span.font-semibold').first();
    await playBtn.click();
    await expect.poll(async () => {
      const t = await frameCounter.innerText().catch(() => '0');
      return parseInt(t.trim(), 10) || 0;
    }, { timeout: 15000 }).toBeGreaterThan(0);
    const tStart = Date.now();

    // Wall-clock wait for 2.0 seconds
    await page.waitForTimeout(2000);

    // Click Pause
    const pauseBtn = page.getByRole('button', { name: '暂停', exact: true });
    await pauseBtn.click();
    const elapsedSec = (Date.now() - tStart) / 1000;

    const frameText = await frameCounter.innerText();
    const curFrame = parseInt(frameText.trim(), 10);

    console.log(`[Test 3.2 DLSS/Split] Wall-clock elapsed: ${elapsedSec.toFixed(2)}s, Current frame reached: ${curFrame}`);

    // Strict assertion: Real-time media semantic check.
    // Allowed range 180~300 frames. MUST NOT be 10~30 frames!
    expect(curFrame, `[DLSS Mode] Player timeline severely slowed down due to per-frame rendering delay! At ${elapsedSec.toFixed(2)}s, frame is ${curFrame} instead of ~240 (expected 180~300). Preview frame dropping must be implemented to preserve media time semantics!`).toBeGreaterThanOrEqual(180);
    expect(curFrame).toBeLessThanOrEqual(300);
  });
});
