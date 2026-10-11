import { test, expect } from '@playwright/test';
import { launchApp } from '../helpers/electron_runner.js';
import { FIXTURE_120FPS_MKV, generateFixture120fps } from '../fixtures/generate_fixtures.js';
import { getCurrentVideo } from '../helpers/backend_client.js';
import { importFixture } from '../helpers/ui.js';

test.describe('第二组测试：素材导入 (Material Import Verification)', () => {
  let appInstance: any = null;

  test.beforeAll(async () => {
    // Ensure fixture exists
    generateFixture120fps();
  });

  test.afterEach(async () => {
    if (appInstance?.electronApp) {
      await appInstance.electronApp.close().catch(() => {});
      appInstance = null;
    }
  });

  test('2.1 自动导入 fixture_120fps.mkv 并验证 UI 与后端均正确识别宽高、FPS、总帧数及时长', async () => {
    // Launch with E2E_MOCK_FILE_PATH set to fixture_120fps.mkv
    appInstance = await launchApp({ mockFilePath: FIXTURE_120FPS_MKV });
    const { page } = appInstance;

    // Click "导入素材" button in Header (shared helper retries once on
    // transient backend-restart races; assertions below unchanged)
    await importFixture(page);

    // Verify UI updates with resolution 1280×720
    const resBadge = page.locator('header').locator('text=1280×720');
    await expect(resBadge).toBeVisible();

    // Verify UI frame count and FPS: ~600 F and (120 FPS)
    const framesBadge = page.locator('header').locator('text=/\\b(59[5-9]|60[0-9]) F\\b/');
    await expect(framesBadge).toBeVisible();

    const fpsBadge = page.locator('header').locator('text=(120 FPS)');
    await expect(fpsBadge).toBeVisible();

    // Verify Backend state via /api/video/current
    const currentVideo = await getCurrentVideo();
    expect(currentVideo.loaded).toBe(true);
    expect(currentVideo.video).toBeDefined();

    const v = currentVideo.video;
    expect(v.filename).toBe('fixture_120fps.mkv');
    expect(v.width).toBe(1280);
    expect(v.height).toBe(720);
    expect(v.fps).toBe(120);
    expect(v.frames).toBeGreaterThanOrEqual(595);
    expect(v.frames).toBeLessThanOrEqual(605);
    expect(v.duration_seconds).toBeGreaterThanOrEqual(4.9);
    expect(v.duration_seconds).toBeLessThanOrEqual(5.1);
  });
});
