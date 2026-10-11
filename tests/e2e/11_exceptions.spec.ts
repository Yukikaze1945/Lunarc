import { test, expect } from '@playwright/test';
import path from 'node:path';
import { addJob, startQueue, waitForJob, requestJson, restartBackend, stopBackend, clearQueue, wsUrl } from '../helpers/backend_client.js';
import { FIXTURE_120FPS_MKV } from '../fixtures/generate_fixtures.js';
import { fixturePath } from '../helpers/paths.js';
import WebSocket from 'ws';

test.describe('第十六组测试：异常测试 (Exception & Failure Observability)', () => {
  test.beforeAll(async () => {
    await restartBackend();
  });

  test.afterAll(async () => {
    await stopBackend();
  });

  test.beforeEach(async () => {
    await clearQueue();
  });

  test('11.1 输入文件不存在时，任务必须明确报错且可观测，绝不可静默完成', async () => {
    const jobRes = await addJob({
      name: 'MissingInputException',
      input_path: path.join('F:', 'non_existent_dir_12345', 'ghost_file.mp4'),
      output_path: fixturePath('out_ghost.mkv'),
      codec: 'av1_nvenc',
      container: 'mkv',
      cq: 24,
      total_frames: 100,
    });

    const jobId = jobRes.job.id;
    await startQueue();
    const job = await waitForJob(jobId, 15000);

    expect(job.status).toBe('failed');
    expect(job.error).toContain('not found');
  });

  test('11.2 输出目录不可写或路径非法时，必须抛出可观测错误', async () => {
    const jobRes = await addJob({
      name: 'UnwritableOutputException',
      input_path: FIXTURE_120FPS_MKV,
      output_path: path.join('Z:', 'invalid_unmounted_drive_987', 'illegal_path.mkv'),
      codec: 'av1_nvenc',
      container: 'mkv',
      cq: 24,
      total_frames: 600,
    });

    const jobId = jobRes.job.id;
    await startQueue();
    const job = await waitForJob(jobId, 20000);

    expect(job.status).toBe('failed');
    expect(job.error).toBeTruthy();
  });

  test('11.3 指定不可用的编码器参数时，FFmpeg 异常退出且错误被捕获', async () => {
    const jobRes = await addJob({
      name: 'InvalidEncoderException',
      input_path: FIXTURE_120FPS_MKV,
      output_path: fixturePath('out_invalid_codec.mkv'),
      codec: 'non_existent_magic_codec',
      container: 'mkv',
      cq: 24,
      total_frames: 600,
      transcode_settings: {
        codec: 'non_existent_magic_codec',
        container: '.mkv',
      },
    });

    const jobId = jobRes.job.id;
    await startQueue();
    const job = await waitForJob(jobId, 45000);

    expect(job.status).toBe('failed');
    expect(job.error).toBeTruthy();
    expect(job.error).toMatch(/FFmpeg|codec|Unknown encoder/i);
  });

  test('11.4 WebSocket 客户端断开连接后，后端保持稳定并在重连时恢复', async () => {
    const ws = new WebSocket(wsUrl('player'));
    await new Promise<void>((resolve, reject) => {
      ws.on('open', () => resolve());
      ws.on('error', reject);
    });

    // Abruptly terminate connection
    ws.terminate();

    // Verify backend remains healthy
    await new Promise((r) => setTimeout(r, 500));
    const sys = await requestJson('/api/system');
    expect(sys.backend_ready).toBe(true);

    // Verify a new WebSocket can connect smoothly
    const ws2 = new WebSocket(wsUrl('player'));
    const reconnected = await new Promise<boolean>((resolve) => {
      ws2.on('open', () => {
        ws2.close();
        resolve(true);
      });
      ws2.on('error', () => resolve(false));
    });
    expect(reconnected).toBe(true);
  });
});
