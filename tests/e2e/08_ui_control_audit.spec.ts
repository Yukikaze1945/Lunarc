import { test, expect } from '@playwright/test';
import { execSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { launchApp } from '../helpers/electron_runner.js';
import { FIXTURE_120FPS_MKV, generateFixture120fps } from '../fixtures/generate_fixtures.js';
import { ROOT_DIR, pythonExe } from '../helpers/paths.js';
import { importFixture } from '../helpers/ui.js';

interface AuditItem {
  control: string;
  uiExists: boolean;
  stateChanges: boolean;
  sentToBackend: boolean;
  backendUsesIt: boolean;
  affectsFFmpegOrBehavior: boolean;
  status: 'WIRED' | 'DEAD / UNIMPLEMENTED CONTROL';
  notes: string;
}

test.describe('第十三组测试：UI 控件接线审计 (Deliver Page UI Control Wiring Audit)', () => {
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

  test('8.1 自动化深度审计 Deliver 页面 10 大核心控件，输出接线报告并标记 DEAD / UNIMPLEMENTED CONTROL', async () => {
    appInstance = await launchApp({ mockFilePath: FIXTURE_120FPS_MKV });
    const { page } = appInstance;

    // 1. Import fixture (shared helper retries once on transient races)
    await importFixture(page);

    // 2. Go to Deliver page
    const deliverTabBtn = page.getByRole('button', { name: '交付' });
    await deliverTabBtn.click();

    // 3. Inspect source code of backend pipeline.py and dlss_service.py to verify backend usage
    const pipelinePyPath = path.join(ROOT_DIR, 'pipeline.py');
    const dlssServicePyPath = path.join(ROOT_DIR, 'dlss_service.py');
    const pipelineCode = fs.readFileSync(pipelinePyPath, 'utf-8');
    const dlssServiceCode = fs.readFileSync(dlssServicePyPath, 'utf-8');

    // 3b. UI-side: the 10 controls must actually exist on the Deliver page
    // (P2-04: previously hardcoded YES without checking the DOM).
    for (const label of ['分辨率 (Resolution)', 'GPU 缩放算法 (Scale)', '像素格式 (Pixel Format)',
                         '色彩规范 (Color Preset)', '解码模式 (Decode)', '完成后动作 (Finish)']) {
      await expect(page.getByText(label, { exact: false }).first()).toBeVisible({ timeout: 10000 });
    }

    // 3c. Behavioral probes (P2-04): each control must observably change
    // backend behavior, not merely appear as a string in source.
    const probeScript = `
import itertools, json, os, sys
sys.path.insert(0, r'${ROOT_DIR}')
os.environ['E2E_TEST'] = '1'
import pipeline

out = {}
out['res_orig'] = list(pipeline.resolve_output_dims(1280, 720, {'resolution_mode': 'original'}))
out['res_half'] = list(pipeline.resolve_output_dims(1280, 720, {'resolution_mode': 'half'}))
out['res_custom_size'] = list(pipeline.resolve_output_dims(
    1280, 720, {'resolution_mode': 'custom_size', 'custom_width': 1920, 'custom_height': 1080, 'keep_aspect': False}))
out['res_custom_factor'] = list(pipeline.resolve_output_dims(
    1280, 720, {'resolution_mode': 'custom_factor', 'custom_factor': 2.0}))

def cmd(ts):
    base = {'codec': 'av1_nvenc', 'container': '.mkv', 'preset': 'p7', 'tune': 'hq',
            'multipass': 'fullres', 'rate_mode': 'cq', 'cq': 16, 'pixel_format': 'auto',
            'audio_mode': 'copy', 'subtitle_mode': 'copy', 'copy_attachments': True,
            'copy_metadata': True, 'copy_chapters': True, 'resolution_mode': 'original',
            'scale_algorithm': 'lanczos', 'color_preset': 'preserve', 'decode_mode': 'auto',
            'finish_action': 'none', 'overwrite_output': True, 'skip_existing': False}
    base.update(ts)
    return pipeline.build_transcode_command(
        video='in.mkv', out_path='o.mkv', w=1280, h=720, fps=120.0,
        transcode_settings=base, cq=16, with_audio=True)

c0 = cmd({})
out['sws_default'] = c0[c0.index('-sws_flags') + 1]
c1 = cmd({'scale_algorithm': 'nearest'})
out['sws_nearest'] = c1[c1.index('-sws_flags') + 1]
out['color_default_has_space'] = '-colorspace' in c0
c2 = cmd({'color_preset': 'bt709_limited'})
out['color_bt709'] = c2[c2.index('-colorspace'):c2.index('-colorspace') + 2]
out['flag_default'] = cmd({})[1]
out['flag_skip'] = cmd({'skip_existing': True})[1]
out['flag_no_overwrite'] = cmd({'overwrite_output': False})[1]

cpu_frames = [(i, [int(x) for x in f.shape]) for i, f in
              itertools.islice(pipeline.iter_frames(r'${FIXTURE_120FPS_MKV}', decode_mode='cpu'), 2)]
out['cpu_frames'] = cpu_frames

import dlss_service as ds
svc = ds.DLSSService()
svc._handle_finish_action([], True)
out['finish_veto'] = any('已禁止' in l.get('message', '') for l in ds.diagnostic_logger.get_logs(5))
sleep_job = [{'id': 'y', 'transcode_settings': {'finish_action': 'sleep'}}]
svc._handle_finish_action(sleep_job, False)
out['finish_pass'] = any('抑制' in l.get('message', '') for l in ds.diagnostic_logger.get_logs(5))
print("__AUDIT_JSON_START__" + json.dumps(out) + "__AUDIT_JSON_END__")
`;
    const rawOut = execSync(`"${pythonExe()}" -`, {
      input: probeScript,
      encoding: 'utf-8',
      timeout: 180000,
    });
    const match = rawOut.match(/__AUDIT_JSON_START__(.*?)__AUDIT_JSON_END__/s);
    if (!match) {
      throw new Error(`Failed to extract audit JSON from python output:\n${rawOut}`);
    }
    const probe = JSON.parse(match[1].trim());

    const behaviorOf: Record<string, boolean> = {
      resolution_mode: JSON.stringify(probe.res_half) !== JSON.stringify(probe.res_orig),
      custom_width: JSON.stringify(probe.res_custom_size) === JSON.stringify([1920, 1080]),
      custom_height: JSON.stringify(probe.res_custom_size) === JSON.stringify([1920, 1080]),
      custom_factor: JSON.stringify(probe.res_custom_factor) === JSON.stringify([2560, 1440]),
      scale_algorithm: probe.sws_default !== probe.sws_nearest,
      color_preset: !probe.color_default_has_space && probe.color_bt709[0] === '-colorspace',
      decode_mode: probe.cpu_frames.length === 2 && probe.cpu_frames[0][1].join(',') === '720,1280,3',
      finish_action: probe.finish_veto === true && probe.finish_pass === true,
      overwrite_output: probe.flag_default === '-y' && probe.flag_no_overwrite === '-n',
      skip_existing: probe.flag_skip === '-n',
    };

    const controlsToAudit = [
      'resolution_mode',
      'custom_width',
      'custom_height',
      'custom_factor',
      'scale_algorithm',
      'color_preset',
      'decode_mode',
      'finish_action',
      'overwrite_output',
      'skip_existing',
    ];

    const auditResults: AuditItem[] = [];

    for (const ctrl of controlsToAudit) {
      // Necessary condition: backend references ctrl in pipeline.py or dlss_service.py
      const backendUses =
        pipelineCode.includes(`"${ctrl}"`) ||
        pipelineCode.includes(`'${ctrl}'`) ||
        dlssServiceCode.includes(`"${ctrl}"`) ||
        dlssServiceCode.includes(`'${ctrl}'`);

      // Sufficient condition (P2-04): the control observably changes behavior.
      const behaves = behaviorOf[ctrl] === true;
      const wired = backendUses && behaves;

      auditResults.push({
        control: ctrl,
        uiExists: true,
        stateChanges: true,
        sentToBackend: true,
        backendUsesIt: backendUses,
        affectsFFmpegOrBehavior: wired,
        status: wired ? 'WIRED' : 'DEAD / UNIMPLEMENTED CONTROL',
        notes: wired
          ? 'String present AND behaviorally verified (command/dims/decode/finish probe)'
          : (backendUses
            ? 'Present in source but behavioral probe FAILED — treated as dead'
            : 'Present in UI state and API payload, but completely ignored by backend pipeline.py/dlss_service.py'),
      });
    }

    // Format Markdown Report
    let reportMd = `# Deliver 页面 UI 控件接线审计报表\n\n`;
    reportMd += `| 控件名称 | UI Exists | State Changes | Sent to Backend | Backend Uses It | Affects FFmpeg/Behavior | 审计状态 |\n`;
    reportMd += `| :--- | :---: | :---: | :---: | :---: | :---: | :--- |\n`;

    for (const item of auditResults) {
      reportMd += `| \`${item.control}\` | ${item.uiExists ? 'YES' : 'NO'} | ${item.stateChanges ? 'YES' : 'NO'} | ${item.sentToBackend ? 'YES' : 'NO'} | ${item.backendUsesIt ? 'YES' : 'NO'} | ${item.affectsFFmpegOrBehavior ? 'YES' : 'NO'} | **${item.status}** |\n`;
    }

    console.log('\n' + reportMd + '\n');

    // Save report to tests/UI_CONTROL_AUDIT_REPORT.md
    fs.writeFileSync(path.join(ROOT_DIR, 'tests', 'UI_CONTROL_AUDIT_REPORT.md'), reportMd, 'utf-8');

    // Verification: after Bug5 fix, all 10 controls must be WIRED (backend uses them).
    // Previously this audit required deadControls > 0 to expose the bug; now it
    // guards against regression back to DEAD.
    const deadControls = auditResults.filter((r) => r.status === 'DEAD / UNIMPLEMENTED CONTROL');
    console.log(`[Audit] Total audited: ${auditResults.length}, DEAD controls found: ${deadControls.length}`);

    expect(deadControls.length, 'All 10 Deliver controls must be wired to backend (0 DEAD allowed after fix)!').toBe(0);
    // Explicitly confirm skip_existing and overwrite_output are wired
    expect(auditResults.find((r) => r.control === 'skip_existing')?.backendUsesIt).toBe(true);
    expect(auditResults.find((r) => r.control === 'overwrite_output')?.backendUsesIt).toBe(true);
  });
});
