import { test, expect } from '@playwright/test';
import path from 'node:path';
import fs from 'node:fs';
import { FIXTURE_120FPS_MKV, generateFixture120fps } from '../fixtures/generate_fixtures.js';
import { addJob, startQueue, waitForJob, restartBackend } from '../helpers/backend_client.js';

test.describe('第十四组测试：文件覆盖测试 (File Overwrite & skip_existing Verification)', () => {
  const targetDir = path.dirname(FIXTURE_120FPS_MKV);
  const targetExistingFile = path.join(targetDir, 'fixture_overwrite_test.mkv');
  const PRESET_MARKER = 'PRE_EXISTING_FILE_CONTENT_DO_NOT_OVERWRITE';

  test.beforeAll(async () => {
    // Isolate from leftover jobs of previous specs (fresh in-memory queue).
    await restartBackend();
    generateFixture120fps();
  });

  test.afterEach(async () => {
    if (fs.existsSync(targetExistingFile)) {
      try { fs.unlinkSync(targetExistingFile); } catch {}
    }
  });

  test('9.1 skip_existing = true 且存在同名文件时，程序绝不能无条件使用 -y 强行覆盖已有文件', async () => {
    // 1. Create pre-existing file with known marker
    fs.writeFileSync(targetExistingFile, PRESET_MARKER, 'utf-8');
    expect(fs.existsSync(targetExistingFile)).toBe(true);

    // 2. Submit job configured with skip_existing: true, overwrite_output: false
    const jobRes = await addJob({
      name: 'SkipExistingTestJob',
      input_path: FIXTURE_120FPS_MKV,
      output_path: targetExistingFile,
      codec: 'av1_nvenc',
      container: 'mkv',
      cq: 24,
      total_frames: 600,
      with_audio: true,
      transcode_settings: {
        codec: 'av1_nvenc',
        container: '.mkv',
        skip_existing: true,
        overwrite_output: false,
      },
    });

    const jobId = jobRes.job?.id;
    expect(jobId).toBeTruthy();

    // 3. Start render queue
    await startQueue();

    // 4. Wait for job to finish
    await waitForJob(jobId, 120000);

    // 5. Inspect file content: Did the program overwrite it?
    const currentContent = fs.readFileSync(targetExistingFile, 'utf-8').slice(0, 100);

    console.log(`[Test 9.1 File Overwrite] Pre-existing file content was preserved: ${currentContent === PRESET_MARKER}`);

    // Strict assertion:
    // If backend unconditionally injected -y and destroyed the pre-existing file despite skip_existing = true,
    // this test MUST FAIL!
    expect(
      currentContent,
      `Defect detected: Program unconditionally used -y and overwrote existing file even though skip_existing = true!`
    ).toBe(PRESET_MARKER);
  });
});
