// Actual installed Electron + isolated CPython acceptance. Requires user's DLL.
// Usage: node installer_acceptance.cjs <Lunarc.exe> <evidence-dir> <synthetic-video>
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '../..');
const {_electron} = require(path.join(root, 'frontend/node_modules/playwright'));
const [executable, output, fixture] = process.argv.slice(2).map(p => path.resolve(p));
assert(executable && output && fixture, 'Specify executable, evidence directory and fixture');
const resources = path.join(path.dirname(executable), 'resources');
const result = {passed: false, installerVersion: '2.1.1', executable, checks: []};
fs.mkdirSync(output, {recursive: true});
async function api(page, route, method = 'GET', body) {
  return page.evaluate(async ({route, method, body}) => {
    const response = await fetch('http://127.0.0.1:18923' + route, {
      method, headers: {'X-DLSS-Token': window.electronAPI.localToken, 'Content-Type': 'application/json'},
      ...(body === undefined ? {} : {body: JSON.stringify(body)}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(JSON.stringify(data));
    return data;
  }, {route, method, body});
}
(async () => {
  if (await fetch('http://127.0.0.1:18923/api/system').then(() => true).catch(() => false))
    throw new Error('An existing backend owns 18923; acceptance refuses to attach or stop it');
  assert(!fs.existsSync(path.join(resources, 'nvngx_dlssnr.dll')));
  assert(fs.readFileSync(path.join(resources, 'licenses/Lunarc-LICENSE.txt'), 'utf8').includes('GNU GENERAL PUBLIC LICENSE'));
  for (const file of ['NOTICE.txt', 'THIRD_PARTY_NOTICES.md', 'project-notices/Lunarc-legacy-MIT.txt'])
    assert(fs.existsSync(path.join(resources, 'licenses', file)));
  assert.equal(JSON.parse(fs.readFileSync(path.join(resources, 'app/package.json'))).version, '2.1.1');
  result.checks.push('2.1.1 metadata', 'GPL and component notices', 'DLSSNR absent from resources');
  const env = {...process.env, E2E_TEST: '1', PYTHON_PATH: 'Z:/unavailable/python.exe',
    PYTHONHOME: 'Z:/unavailable', PYTHONPATH: 'Z:/unavailable',
    DLSS5_FFMPEG_DIR: 'Z:/unavailable', E2E_MOCK_FILE_PATH: fixture,
    PATH: [process.env.SystemRoot + '/System32', process.env.SystemRoot].join(';')};
  delete env.ELECTRON_RUN_AS_NODE;
  const app = await _electron.launch({executablePath: executable,
    args: ['--user-data-dir=' + path.join(output, 'profile-' + Date.now())], env, timeout: 60000});
  try {
    const page = await app.firstWindow();
    await page.setViewportSize({width: 1600, height: 1000});
    await page.waitForFunction(async () => {
      const s = await window.electronAPI.getBackendStartup();
      return ['ready', 'starting', 'error'].includes(s.state);
    }, {}, {timeout: 45000});
    const startup = await page.evaluate(() => window.electronAPI.getBackendStartup());
    assert.notEqual(startup.state, 'error', startup.message);
    await page.waitForFunction(async () => {
      try {return (await (await fetch('http://127.0.0.1:18923/api/system', {
        headers: {'X-DLSS-Token': window.electronAPI.localToken}})).json()).backend_ready === true}
      catch {return false}
    }, {}, {timeout: 60000});
    const report = await api(page, '/api/environment');
    assert(report.standalone && report.isolated);
    assert.equal(path.resolve(report.python).toLowerCase(), path.join(resources, 'runtime/python/python.exe').toLowerCase());
    assert.equal(report.checks.find(c => c.id === 'cuda').status, 'ok');
    const missing = await api(page, '/api/runtime/dlssnr');
    assert.equal(missing.error.code, 'runtime_missing');
    const queueEvent = await page.evaluate(() => new Promise((resolve, reject) => {
      const ws = new WebSocket('ws://127.0.0.1:18923/ws/queue?token=' + encodeURIComponent(window.electronAPI.localToken));
      const timeout = setTimeout(() => {ws.close(); reject(new Error('Queue WebSocket timeout'))}, 10000);
      ws.onmessage = e => {clearTimeout(timeout); ws.close(); resolve(JSON.parse(e.data))};
      ws.onerror = () => {clearTimeout(timeout); ws.close(); reject(new Error('Queue WebSocket handshake failed'))};
    }));
    assert.equal(queueEvent.type, 'init');
    result.environment = report;
    result.checks.push('isolated bundled Python', 'real CUDA operation', 'missing NR correctly reported', 'real queue WebSocket handshake');
    const imported = await api(page, '/api/runtime/dlssnr/import', 'POST', {path: path.join(root, 'nvngx_dlssnr.dll')});
    await api(page, '/api/runtime/dlssnr/activate', 'POST', {id: imported.id});
    const settings = await api(page, '/api/settings');
    await page.evaluate(settings => {
      localStorage.setItem('lunarc.workspace.v1', JSON.stringify({version: 1,
        settings: {...(settings.settings || settings), guidance_mode: 0, render_pipeline: ['nr', 'vfg'],
          frame_generation: 'vfg', frame_multiplier: 2, working_scale: 0.75,
          export_backend: 'gpu_transfer'}, recent: []}));
    }, settings);
    await page.reload();
    await page.getByRole('button', {name: '导入', exact: true}).click();
    await page.waitForFunction(() => {
      const image = document.querySelector('.dlss-preview-image');
      return image instanceof HTMLImageElement && image.complete && image.naturalWidth > 0
        && !document.querySelector('.dlss-player-error');
    }, {}, {timeout: 60000});
    const frame = await page.evaluate(() => {
      const img = document.querySelector('.dlss-preview-image');
      return {width: img.naturalWidth, height: img.naturalHeight};
    });
    assert(frame.width > 0 && frame.height > 0);
    result.frame = frame;
    result.preview = await api(page, '/api/preview/diagnostics');
    assert(result.preview.render_pipeline.includes('nr'));
    assert(result.preview.layers.layer_session_ids.some(id => id > 0));
    const initialFrame = result.preview.frame;
    await page.getByRole('button', {name: '开始播放', exact: true}).click();
    await page.waitForFunction(initial =>
      Number(document.querySelector('[aria-label="时间轴帧位置"]').value) >= initial + 3,
      initialFrame, {timeout: 30000});
    const displayed = Number(await page.getByRole('slider', {name: '时间轴帧位置'}).inputValue());
    assert(displayed >= initialFrame + 3);
    result.playback = {...await api(page, '/api/preview/diagnostics'), displayed_frame: displayed};
    const pause = page.getByRole('button', {name: '暂停播放', exact: true});
    if (await pause.count()) await pause.click();
    result.checks.push('external signed NR import', 'actual synthetic video GPU preview');
    result.checks.push('real NR session and continuous playback');
    await page.screenshot({path: path.join(output, 'workbench.png')});
    await page.getByRole('button', {name: '运行与诊断', exact: true}).click();
    await page.locator('.runtime-current strong').getByText('版本 310.8.0.0', {exact: true}).waitFor({timeout: 20000});
    await page.screenshot({path: path.join(output, 'runtime.png')});
    result.checks.push('installed runtime manager UI');
    result.passed = true;
  } catch (error) {
    result.error = error.message;
    const page = await app.firstWindow().catch(() => null);
    if (page) {
      result.previewError = await page.locator('.dlss-player-error').allTextContents().catch(() => []);
      await page.screenshot({path: path.join(output, 'failure.png')}).catch(() => {});
    }
    throw error;
  } finally {
    fs.writeFileSync(path.join(output, 'installer-acceptance.json'), JSON.stringify(result, null, 2));
    await app.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1});
