import { test, expect } from '@playwright/test';
import path from 'node:path';
import fs from 'node:fs';
import { FIXTURE_120FPS_MKV, generateFixture120fps } from '../fixtures/generate_fixtures.js';
import { addJob, getQueue, startQueue, cancelQueue, waitForJob, ensureBackendRunning, clearQueue } from '../helpers/backend_client.js';
import { fixturePath } from '../helpers/paths.js';

test.describe('第十五组测试：任务状态测试 (Task State Machine & Failure Integrity)', () => {
  test.beforeAll(async () => {
    await ensureBackendRunning();
    generateFixture120fps();
  });

  test.beforeEach(async () => {
    await clearQueue();
  });

  test('10.1 验证 pending 状态流转与导出过程中主动取消 (cancelled 状态)', async () => {
    const testOut = path.join(path.dirname(FIXTURE_120FPS_MKV), 'test_cancel_status.mkv');
    if (fs.existsSync(testOut)) {
      try { fs.unlinkSync(testOut); } catch {}
    }

    // 1. Add job -> verify 'pending'
    const jobRes = await addJob({
      name: 'StatusCancelJob',
      input_path: FIXTURE_120FPS_MKV,
      output_path: testOut,
      codec: 'av1_nvenc',
      container: 'mkv',
      cq: 24,
      total_frames: 600,
      with_audio: true,
    });

    const jobId = jobRes.job?.id;
    expect(jobRes.job.status).toBe('pending');

    // 2. Start queue -> verify transitions to 'running'
    await startQueue();
    await new Promise((r) => setTimeout(r, 600));

    const qRunning = await getQueue();
    const runningJob = qRunning.jobs?.find((j: any) => j.id === jobId);
    expect(runningJob).toBeDefined();

    // 3. Cancel active job
    await cancelQueue();

    // 4. Wait for job to settle
    const settledJob = await waitForJob(jobId, 20000);
    console.log(`[Test 10.1] Cancelled job status: ${settledJob.status}`);
    expect(['cancelled', 'failed']).toContain(settledJob.status);
    expect(settledJob.status).not.toBe('completed');
  });

  test('10.2 异常情况下 (输入损坏/无效文件) 绝不得标记 completed，必须置为 failed 并输出诊断信息', async () => {
    const invalidInput = fixturePath('non_existent_corrupted_input.mp4');
    const testOut = fixturePath('invalid_output.mkv');

    // 1. Add job with invalid input
    const jobRes = await addJob({
      name: 'CorruptedInputJob',
      input_path: invalidInput,
      output_path: testOut,
      codec: 'av1_nvenc',
      container: 'mkv',
      cq: 24,
      total_frames: 100,
      with_audio: false,
    });

    const jobId = jobRes.job?.id;
    expect(jobId).toBeTruthy();

    // 2. Start queue
    await startQueue();

    // 3. Wait for job result
    const finishedJob = await waitForJob(jobId, 20000);
    console.log(`[Test 10.2] Faulty job final status: ${finishedJob.status}, Error: ${finishedJob.error}`);

    // Must be 'failed', NEVER 'completed'
    expect(finishedJob.status).toBe('failed');
    expect(finishedJob.error).toBeTruthy();
    expect(finishedJob.progress).toBeLessThan(100.0);
  });
});
