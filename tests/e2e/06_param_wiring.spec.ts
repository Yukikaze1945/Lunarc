import { test, expect } from '@playwright/test';
import { execSync } from 'node:child_process';
import { FIXTURE_120FPS_MKV, generateFixture120fps } from '../fixtures/generate_fixtures.js';
import { addJob, ensureBackendRunning } from '../helpers/backend_client.js';
import { ROOT_DIR, fixturePath, pythonExe } from '../helpers/paths.js';

test.describe('第十一组测试：参数真实接线测试 (Parameter Wiring Verification)', () => {
  test.beforeAll(async () => {
    await ensureBackendRunning();
    generateFixture120fps();
  });

  test('6.1 验证参数全链路接线：UI payload -> Backend Job -> 实际 FFmpeg 命令参数', async () => {
    // Define a custom set of transcode parameters
    const customSettings = {
      codec: 'hevc_nvenc',
      container: '.mp4',
      rate_mode: 'cq',
      cq: 19,
      preset: 'p6',
      tune: 'hq',
      pixel_format: 'p010le',
      audio_mode: 'remove',
      subtitle_mode: 'drop',
      gpu: '0',
      gop_mode: 'manual',
      gop: 120,
    };

    // 1. Submit to backend API /api/queue/add
    const paramOut = fixturePath('test_param_out.mp4');
    const jobRes = await addJob({
      name: 'ParamWiringJob',
      input_path: FIXTURE_120FPS_MKV,
      output_path: paramOut,
      codec: customSettings.codec,
      container: 'mp4',
      cq: customSettings.cq,
      preset: customSettings.preset,
      with_audio: false,
      transcode_settings: customSettings,
    });

    const job = jobRes.job;
    expect(job).toBeDefined();
    expect(job.transcode_settings).toBeDefined();

    // 2. Query backend Python pipeline to build the actual FFmpeg command for this job
    const script = `
import json, sys
sys.path.insert(0, r'${ROOT_DIR}')
import pipeline

settings = ${JSON.stringify(customSettings)}
cmd = pipeline.build_transcode_command(
    video=r'${FIXTURE_120FPS_MKV}',
    out_path=r'${paramOut}',
    w=1280,
    h=720,
    fps=120.0,
    transcode_settings=settings,
    cq=19,
    with_audio=False
)
print(json.dumps(cmd))
`;

    const rawCmd = execSync(`"${pythonExe()}" -`, {
      input: script,
      encoding: 'utf-8',
    });
    const cmdList: string[] = JSON.parse(rawCmd.trim());

    console.log('[Test 6.1] Generated actual FFmpeg command:', cmdList.join(' '));

    // 3. Verify each parameter is properly wired into the FFmpeg command line:
    // Codec
    expect(cmdList).toContain('-c:v');
    expect(cmdList[cmdList.indexOf('-c:v') + 1]).toBe('hevc_nvenc');

    // Preset
    expect(cmdList).toContain('-preset');
    expect(cmdList[cmdList.indexOf('-preset') + 1]).toBe('p6');

    // Tune
    expect(cmdList).toContain('-tune');
    expect(cmdList[cmdList.indexOf('-tune') + 1]).toBe('hq');

    // CQ / Rate mode
    expect(cmdList).toContain('-rc:v');
    expect(cmdList[cmdList.indexOf('-rc:v') + 1]).toBe('vbr');
    expect(cmdList).toContain('-cq:v');
    expect(cmdList[cmdList.indexOf('-cq:v') + 1]).toBe('19');

    // Pixel format
    expect(cmdList).toContain('-pix_fmt');
    expect(cmdList[cmdList.lastIndexOf('-pix_fmt') + 1]).toBe('p010le');

    // Audio mode (remove -> -an)
    expect(cmdList).toContain('-an');

    // Subtitle mode (drop -> -sn)
    expect(cmdList).toContain('-sn');

    // GPU device
    expect(cmdList).toContain('-gpu');
    expect(cmdList[cmdList.indexOf('-gpu') + 1]).toBe('0');

    // GOP
    expect(cmdList).toContain('-g');
    expect(cmdList[cmdList.indexOf('-g') + 1]).toBe('120');
  });
});
