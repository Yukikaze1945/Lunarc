import { test, expect } from '@playwright/test';
import path from 'node:path';
import { launchApp } from '../helpers/electron_runner.js';
import { FIXTURE_120FPS_MKV } from '../fixtures/generate_fixtures.js';
import { clearQueue, getQueue, requestJson, restartBackend } from '../helpers/backend_client.js';
import { importFixture } from '../helpers/ui.js';

/**
 * 第十六组：交付面板的链路预览说的是它自己那条链路，同一个文件不会被自己覆盖。
 *
 * 这一组不渲染任何一帧：它验的是页面上那段文字与按钮状态，以及加入队列这条路在
 * 目标与素材实为同一文件时走不走得通。真正的设备链路与后端拒收分别由
 * tests/unit/test_gpu_transfer_export_backend.py 与 tests/unit/test_output_overwrite_guard.py
 * 在无需显示器的情况下钉住。
 */

const DEVICE_PREVIEW = '[aria-label="设备链路预览 (Device Chain Preview)"]';
const COPY_BUTTON = 'button[aria-label="复制导出命令 (Copy Export Command)"]';
const BACKEND_SELECT = 'select[aria-label="导出后端 (Export Backend)"]';
const NAME_INPUT = 'input[placeholder="输出视频文件名"]';
const ADD_BUTTON = /添加到渲染队列/;
const STEM = path.basename(FIXTURE_120FPS_MKV, path.extname(FIXTURE_120FPS_MKV));

let appInstance: any = null;

async function launch() {
  appInstance = await launchApp({ mockFilePath: FIXTURE_120FPS_MKV });
  const dialogs: string[] = [];
  // The runner dismisses dialogs already; this only keeps a copy of what was said.
  appInstance.page.on('dialog', (d: any) => dialogs.push(d.message()));
  return { page: appInstance.page, dialogs };
}

async function deliverTab(page: any) {
  await page.getByRole('button', { name: '交付' }).click();
}

async function dlssTab(page: any) {
  await page.getByRole('button', { name: 'DLSS 5', exact: true }).click();
}

async function chooseExportBackend(page: any, value: string) {
  await dlssTab(page);
  await page.locator(BACKEND_SELECT).selectOption(value);
  // The panel reads the service's settings, not its own local state.
  await expect
    .poll(async () => (await requestJson('/api/settings')).export_backend, { timeout: 15000 })
    .toBe(value);
  await deliverTab(page);
}

test.describe('第十六组：链路预览条件化与自身覆盖拦截 (Chain Preview & Self-Overwrite Prevention)', () => {
  test.beforeAll(async () => {
    await restartBackend();
  });

  test.afterEach(async () => {
    if (appInstance?.electronApp) {
      await appInstance.electronApp.close().catch(() => {});
      appInstance = null;
    }
  });

  test('16.1 GPU Transfer 选中时预览换成设备链路，没有可复制的命令；切回 Auto 立刻换回命令行', async () => {
    const { page } = await launch();
    await importFixture(page);
    await deliverTab(page);

    // Default chain: the legacy pipe really is one FFmpeg command line.
    const legacyBox = page.locator('div.font-mono.select-text').first();
    await expect(legacyBox).toBeVisible();
    expect(await legacyBox.innerText()).toContain('rawvideo');
    await expect(page.locator(COPY_BUTTON)).toBeEnabled();

    await chooseExportBackend(page, 'gpu_transfer');

    // The device chain is not a command line, so it must not be shown as one.
    await expect(page.locator(DEVICE_PREVIEW)).toBeVisible();
    await expect(page.locator('div.font-mono.select-text')).toHaveCount(0);
    const preview = await page.locator(DEVICE_PREVIEW).innerText();
    for (const stage of ['NVDEC', 'DLSS', 'NVENC']) {
      expect(preview, `device-chain preview omits the ${stage} stage`).toContain(stage);
    }
    expect(preview, 'device-chain preview oversells an unbenchmarked chain').toContain('NOT_CHARACTERIZED');
    expect(preview, 'device-chain preview hides the no-fallback rule').toContain('不会自动回退');
    for (const commandToken of ['-pix_fmt bgr24', '-f rawvideo', '-loglevel']) {
      expect(preview, `device-chain preview still shows the legacy token ${commandToken}`)
        .not.toContain(commandToken);
    }
    // The chain is described, not pasted: the panel says so in its own words.
    expect(preview, 'device-chain preview hides that it has no runnable command')
      .toContain('不给出可运行命令');

    // Copying a description would paste a different pipeline than the one selected.
    await expect(page.locator(COPY_BUTTON)).toBeDisabled();

    await chooseExportBackend(page, 'auto');
    await expect(page.locator('div.font-mono.select-text').first()).toBeVisible();
    await expect(page.locator(DEVICE_PREVIEW)).toHaveCount(0);
    await expect(page.locator(COPY_BUTTON)).toBeEnabled();
  });

  test('16.2 目标与素材同一文件时面板报错且加入不了队列；仅大小写不同同样拦截；改名后放行', async () => {
    const { page, dialogs } = await launch();
    await importFixture(page);
    await deliverTab(page);
    await clearQueue();

    const before = (await getQueue()).jobs?.length || 0;
    const nameInput = page.locator(NAME_INPUT);

    // The panel's own default directory is the source's directory, so naming the
    // source's own stem makes the composed output the source itself.
    await nameInput.fill(STEM);
    const error = page.getByText('导出目标与输入素材是同一个文件');
    await expect(error).toBeVisible();

    await page.getByRole('button', { name: ADD_BUTTON }).first().click();
    await expect
      .poll(() => (dialogs.join('\n')), { timeout: 5000, message: 'the panel did not refuse the add' })
      .toContain('同一个文件');
    expect((await getQueue()).jobs?.length || 0, 'a self-overwriting job reached the queue').toBe(before);

    // Windows path identity is case-insensitive; an alias that differs only in
    // letter case is the same file, and a browser can see that much.
    await nameInput.fill(STEM.toUpperCase());
    await expect(error).toBeVisible();

    // A different name is a different file, and refusing that too would be a defect.
    await nameInput.fill(`${STEM}_copy`);
    await expect(error).toHaveCount(0);
    await page.getByRole('button', { name: ADD_BUTTON }).first().click();
    await expect
      .poll(async () => (await getQueue()).jobs?.length || 0, { timeout: 15000 })
      .toBeGreaterThan(before);
  });
});
