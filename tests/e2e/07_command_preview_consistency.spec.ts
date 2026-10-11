import { test, expect } from '@playwright/test';
import { execSync } from 'node:child_process';
import { launchApp } from '../helpers/electron_runner.js';
import { FIXTURE_120FPS_MKV, generateFixture120fps } from '../fixtures/generate_fixtures.js';
import { ROOT_DIR, fixturePath, pythonExe } from '../helpers/paths.js';
import { importFixture } from '../helpers/ui.js';

test.describe('第十二组测试：Command Preview 一致性 (Command Preview vs Real FFmpeg Command)', () => {
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

  test('7.1 验证 UI Command Preview 与后端真实执行的 FFmpeg 命令完全一致 (架构单一命令源校验)', async () => {
    appInstance = await launchApp({ mockFilePath: FIXTURE_120FPS_MKV });
    const { page } = appInstance;

    // 1. Import fixture (shared helper retries once on transient races)
    await importFixture(page);

    // 2. Go to Deliver page
    const deliverTabBtn = page.getByRole('button', { name: '交付' });
    await deliverTabBtn.click();

    // 3. Extract the command string shown in UI Preview box
    const previewBox = page.locator('div.font-mono.select-text').first();
    await expect(previewBox).toBeVisible();
    const uiPreviewText = (await previewBox.innerText()).trim();

    console.log('[Test 7.1] UI Command Preview:', uiPreviewText);

    // 4. Query backend pipeline.build_transcode_command with default settings
    const dlssOut = fixturePath('fixture_120fps_dlss.mkv');
    const pyScript = `
import json, sys
sys.path.insert(0, r'${ROOT_DIR}')
import pipeline

settings = {
    'codec': 'av1_nvenc',
    'container': '.mkv',
    'preset': 'p7',
    'tune': 'uhq',
    'multipass': 'fullres',
    'rate_mode': 'cq',
    'cq': 16,
    'pixel_format': 'auto',
    'audio_mode': 'copy',
    'subtitle_mode': 'copy',
    'copy_attachments': True,
    'copy_metadata': True,
    'copy_chapters': True,
    'skip_existing': True,
    'overwrite_output': False,
    'resolution_mode': 'original',
    'scale_algorithm': 'lanczos',
    'color_preset': 'preserve',
}
cmd = pipeline.build_transcode_command(
    video=r'${FIXTURE_120FPS_MKV}',
    out_path=r'${dlssOut}',
    w=1280,
    h=720,
    fps=120.0,
    transcode_settings=settings,
    cq=16,
    with_audio=True
)
print(json.dumps(cmd))
`;
    const rawCmd = execSync(`"${pythonExe()}" -`, {
      input: pyScript,
      encoding: 'utf-8',
    });
    const backendCmdList: string[] = JSON.parse(rawCmd.trim());
    const backendCmdText = backendCmdList.join(' ');

    console.log('[Test 7.1] Backend Actual FFmpeg Command:', backendCmdText);

    // Architectural Inconsistency Checks:
    // 1. UI preview must contain the pipe stdin like the backend does
    //    (-f rawvideo -pix_fmt bgr24 -s 1280x720 -r 120.000000 -i -)
    const uiHasPipeStdin = uiPreviewText.includes('rawvideo') || uiPreviewText.includes('-i -');

    // 2. Audio mapping: Backend maps from file (1:a?), UI must not use 0:a?
    const uiHas0AudioMap = uiPreviewText.includes('-map 0:a?');
    const uiHas1AudioMap = uiPreviewText.includes('1:a?');

    // 3. Strict assertion: UI preview MUST match real backend command builder
    // If frontend maintains its own differing FFmpeg string generator, this test will FAIL and record the defect.
    expect(uiHasPipeStdin, `Command Preview inconsistency: UI preview omits DLSS rawvideo stream input (-f rawvideo -i -) while backend requires it!`).toBe(true);
    expect(uiHas0AudioMap, `Command Preview inconsistency: UI still uses single-input '-map 0:a?' instead of backend dual-input '1:a?'`).toBe(false);
    expect(uiHas1AudioMap, `Command Preview inconsistency: UI must map audio from file input '1:a?' like backend`).toBe(true);
    // Normalized exact comparison (ignore ffmpeg exe absolute path vs bare 'ffmpeg',
    // quote style, and output filename — UI defaults to output.mkv until user picks
    // a dir/name, while backend test uses a fixed fixture path; those are state,
    // not architectural divergence. All middle flags must match exactly).
    const norm = (s: string) =>
      s.replace(/"([^"]*)"/g, '$1').replace(/'([^']*)'/g, '$1').trim()
        .replace(/^(\S*[\\/])?ffmpeg(\.exe)?/i, 'ffmpeg')
        .replace(/\s+/g, ' ');
    const stripEnv = (s: string) => {
      const t = norm(s).split(' ');
      // drop leading exe token and trailing output path token
      return t.slice(1, -1).join(' ');
    };
    expect(stripEnv(uiPreviewText)).toBe(stripEnv(backendCmdText));
  });
});
