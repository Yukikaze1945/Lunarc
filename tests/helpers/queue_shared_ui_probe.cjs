// Actual Electron + backend queue; no render or GPU processing is needed.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '../..');
const {_electron} = require(path.join(root, 'frontend/node_modules/playwright'));
const out = path.resolve(process.argv[2]);
const installed = process.argv[3] && path.resolve(process.argv[3]);
const measureOnly = process.argv.includes('--measure-only');
const result = {passed: false, actualElectron: true, actualBackend: false};
const token = crypto.randomUUID();
const headers = {'X-DLSS-Token': token, 'Content-Type': 'application/json'};
async function api(route, body) {
  const response = await fetch('http://127.0.0.1:18923' + route, {
    headers, ...(body ? {method: 'POST', body: JSON.stringify(body)} : {}),
  });
  const data = await response.json();
  assert(response.ok, JSON.stringify(data));
  return data;
}
(async () => {
  if (await fetch('http://127.0.0.1:18923/api/system').then(() => true).catch(() => false))
    throw new Error('Backend port already in use; refusing to attach to a user session');
  assert(!fs.existsSync(out), 'Use a fresh evidence directory');
  fs.mkdirSync(out, {recursive: true});
  let app;
  try {
    app = await _electron.launch({
      executablePath: installed || path.join(root, 'frontend/node_modules/electron/dist/electron.exe'),
      cwd: path.join(root, 'frontend'), args: [...(installed ? [] : ['.']), `--user-data-dir=${path.join(out, 'profile')}`],
      env: {...process.env, E2E_TEST: '1', DLSS_LOCAL_TOKEN: token},
    });
    const page = await app.firstWindow();
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    let connections = 0, reads = 0, logReads = 0, initial = false;
    page.on('websocket', socket => {
      if (!socket.url().includes('/ws/queue')) return;
      connections++;
      socket.on('framereceived', ({payload}) => {
        try {if (JSON.parse(String(payload)).type === 'init') initial = true} catch {}
      });
    });
    page.on('request', request => {
      if (request.method() === 'GET' && new URL(request.url()).pathname === '/api/queue') reads++;
      if (new URL(request.url()).pathname === '/api/diagnostics/logs') logReads++;
    });
    await page.reload();
    await page.getByRole('button', {name: '队列空闲', exact: true}).waitFor({timeout: 60000});
    const deadline = Date.now() + 60000;
    while (!initial && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 100));
    assert(initial, 'Queue WebSocket did not initialize');
    result.actualBackend = true;
    const baselineReads = reads;
    // Includes the old preview page's third, eight-second queue poll.
    await page.waitForTimeout(10500);
    Object.assign(result, {queueConnections: connections, steadyStateHttpReads: reads - baselineReads,
      observationSeconds: 10.5, mode: measureOnly ? 'baseline-measurement' : 'acceptance'});
    if (measureOnly) { result.passed = true; return; }
    assert.equal(reads - baselineReads, 0);
    assert.equal(connections, 1);
    const settings = await api('/api/settings');
    const add = name => api('/api/queue/add', {
      name, input_path: path.join(root, 'tests/fixtures/fixture_gpu_transfer_short.mkv'),
      output_path: path.join(out, `${name}.mkv`), settings,
      codec: 'hevc_nvenc', container: '.mkv', cq: 24, preset: 'p5', with_audio: false,
      transcode_settings: {finish_action: 'none'},
    });
    await add('shared-feed-first');
    await page.getByRole('button', {name: '1 个任务等待导出', exact: true}).waitFor();
    await page.getByRole('navigation', {name: '主导航'}).getByRole('button', {name: '导出与队列', exact: true}).click();
    await page.getByRole('button', {name: '删除 shared-feed-first', exact: true}).click();
    await page.getByRole('button', {name: '队列空闲', exact: true}).waitFor();
    await add('shared-feed-second');
    await page.getByRole('button', {name: '删除 shared-feed-second', exact: true}).waitFor();
    await page.getByRole('button', {name: '1 个任务等待导出', exact: true}).waitFor();
    await page.getByRole('button', {name: '删除 shared-feed-second', exact: true}).click();
    await page.getByRole('button', {name: '队列空闲', exact: true}).waitFor();
    const firstDiagnostic = (await add('diagnostic-first')).job;
    const secondDiagnostic = (await add('diagnostic-second')).job;
    let releaseFirst, firstStarted, firstFinished;
    const delayFirst = new Promise(resolve => {releaseFirst = resolve});
    const started = new Promise(resolve => {firstStarted = resolve});
    const finished = new Promise(resolve => {firstFinished = resolve});
    await page.route('**/api/diagnostics/job/*', async route => {
      const first = new URL(route.request().url()).pathname.endsWith(firstDiagnostic.id);
      if (first) {firstStarted(); await delayFirst;}
      await route.fulfill({json: {job: {residual_backend_diagnostics: {
        marker: first ? 'obsolete-first-response' : 'current-second-response',
      }}}});
      if (first) firstFinished();
    });
    await page.getByRole('button', {name: '查看 diagnostic-first 的实际运行诊断', exact: true}).click();
    await started;
    await page.getByRole('button', {name: '关闭运行诊断', exact: true}).click();
    await page.getByRole('button', {name: '查看 diagnostic-second 的实际运行诊断', exact: true}).click();
    await page.getByRole('dialog').locator('pre').filter({hasText: 'current-second-response'}).waitFor();
    releaseFirst(); await finished;
    await page.waitForTimeout(300);
    const diagnosticText = await page.getByRole('dialog').locator('pre').innerText();
    assert(diagnosticText.includes('current-second-response'));
    assert(!diagnosticText.includes('obsolete-first-response'));
    await page.getByRole('button', {name: '关闭运行诊断', exact: true}).click();
    for (const name of ['diagnostic-first', 'diagnostic-second'])
      await page.getByRole('button', {name: `删除 ${name}`, exact: true}).click();
    await page.getByRole('button', {name: '队列空闲', exact: true}).waitFor();
    assert.deepEqual((await api('/api/queue')).jobs, []);
    const logToggle = page.getByRole('button', {name: '切换渲染日志显示', exact: true});
    await Promise.all([
      page.waitForResponse(response => new URL(response.url()).pathname === '/api/diagnostics/logs'),
      logToggle.click(),
    ]);
    await page.getByRole('navigation', {name: '主导航'}).getByRole('button', {name: '重建工作台', exact: true}).click();
    const hiddenLogBaseline = logReads;
    await page.waitForTimeout(4500);
    assert.equal(logReads, hiddenLogBaseline);
    await Promise.all([
      page.waitForResponse(response => new URL(response.url()).pathname === '/api/diagnostics/logs'),
      page.getByRole('navigation', {name: '主导航'}).getByRole('button', {name: '导出与队列', exact: true}).click(),
    ]);
    assert.equal(await logToggle.getAttribute('aria-expanded'), 'true');
    await logToggle.click();
    assert.equal(connections, 1);
    assert.deepEqual(errors, []);
    await page.screenshot({path: path.join(out, 'queue.png')});
    Object.assign(result, {passed: true, queueConnections: connections,
      steadyStateHttpReads: reads - baselineReads, observationSeconds: 10.5,
      addDeleteThenAddDelete: true, diagnosticRacePassed: true, diagnosticResponsesMocked: true,
      hiddenPageLogReads: logReads - hiddenLogBaseline - 1, hiddenLogObservationSeconds: 4.5,
      logPanelStatePreserved: true, errors});
  } finally {
    fs.writeFileSync(path.join(out, 'result.json'), JSON.stringify(result, null, 2));
    if (app) await app.close();
  }
  console.log(JSON.stringify(result));
})().catch(error => {console.error(error); process.exitCode = 1});
