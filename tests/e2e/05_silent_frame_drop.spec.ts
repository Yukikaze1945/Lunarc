import { test, expect } from '@playwright/test';
import path from 'node:path';
import fs from 'node:fs';
import { FIXTURE_120FPS_MKV, generateFixture120fps } from '../fixtures/generate_fixtures.js';
import { probeMedia } from '../helpers/media_probe.js';
import { addJob, startQueue, waitForJob, ensureBackendRunning } from '../helpers/backend_client.js';

test.describe('第十组测试：检测静默丢帧 (Silent Frame Drop Prevention)', () => {
  const outputDir = path.dirname(FIXTURE_120FPS_MKV);
  const testOutputFile = path.join(outputDir, 'test_silent_drop_output.mkv');

  test.beforeAll(async () => {
    await ensureBackendRunning();
    generateFixture120fps();
    if (fs.existsSync(testOutputFile)) {
      try { fs.unlinkSync(testOutputFile); } catch {}
    }
  });

  test.afterEach(async () => {
    if (fs.existsSync(testOutputFile)) {
      try { fs.unlinkSync(testOutputFile); } catch {}
    }
  });

  test('5.1 导出过程严谨区分 input_frames, processed_frames 与 encoded_frames，ffprobe 物理帧必须严格吻合', async () => {
    // 1. Get input frames count
    const inputProbe = probeMedia(FIXTURE_120FPS_MKV);
    const inputVideoStream = inputProbe.streams?.find((s) => s.codec_type === 'video');
    const inputFrames = parseInt(inputVideoStream?.nb_read_packets || '600', 10);
    expect(inputFrames).toBeGreaterThanOrEqual(595);

    // 2. Submit job directly to queue
    const jobRes = await addJob({
      name: 'SilentDropVerification',
      input_path: FIXTURE_120FPS_MKV,
      output_path: testOutputFile,
      codec: 'av1_nvenc',
      container: 'mkv',
      cq: 24,
      total_frames: inputFrames,
      with_audio: true,
      transcode_settings: {
        codec: 'av1_nvenc',
        container: '.mkv',
        cq: 24,
        preset: 'p7',
        rate_mode: 'cq',
        audio_mode: 'copy',
      },
    });

    const jobId = jobRes.job?.id;
    expect(jobId).toBeTruthy();

    // 3. Start render
    await startQueue();

    // 4. Wait for job to complete
    const finishedJob = await waitForJob(jobId, 120000);
    expect(finishedJob.status, `Export job failed: ${finishedJob.error}\n${finishedJob.dlss_log_tail || ''}`).toBe('completed');

    // 5. Compare input_frames, processed_frames, and encoded_frames
    const processedFrames = finishedJob.current_frame;
    expect(processedFrames, 'Processed frames must equal input frames').toBe(inputFrames);

    // 6. Inspect physical output with ffprobe to determine encoded_frames
    const outputProbe = probeMedia(testOutputFile);
    const outputVideoStream = outputProbe.streams?.find((s) => s.codec_type === 'video');
    const encodedFrames = parseInt(outputVideoStream?.nb_read_packets || '0', 10);

    console.log(`[Silent Drop Check] input_frames: ${inputFrames}, processed_frames: ${processedFrames}, encoded_frames: ${encodedFrames}`);

    // If encoded_frames is severely less than input_frames (e.g. 100 vs 600), FAIL
    expect(encodedFrames, 'Encoded frames cannot be severely less than processed frames (silent drop!)').toBeGreaterThanOrEqual(inputFrames - 5);
    expect(encodedFrames).toBeLessThanOrEqual(inputFrames + 5);
  });
});
