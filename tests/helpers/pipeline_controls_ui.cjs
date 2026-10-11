// Run from the repository root after building frontend. Uses an isolated Electron profile.
const path = require('node:path'), fs = require('node:fs'), assert = require('node:assert/strict');
const root = process.cwd(), out = path.join(root, 'bench_out/pipeline_sort_20261011');
const { _electron } = require(path.join(root, 'frontend/node_modules/playwright'));
fs.mkdirSync(out, { recursive: true });
(async () => {
  const env = { ...process.env, E2E_TEST: '1', PYTHON_PATH: process.env.PYTHON_PATH || path.join(root, '.venv/Scripts/python.exe') };
  delete env.ELECTRON_RUN_AS_NODE;
  const app = await _electron.launch({ executablePath: require(path.join(root, 'frontend/node_modules/electron')),
    args: [path.join(root, 'frontend'), '--user-data-dir=' + path.join(out, 'profile')], env, timeout: 60000 });
  let api, original;
  try {
    const page = await app.firstWindow();
    await page.setViewportSize({ width: 1600, height: 1300 });
    await page.waitForFunction(() => /工作台就绪|已恢复参数/.test(document.body.textContent), {}, { timeout: 60000 });
    api = (endpoint, body) => page.evaluate(async ({ endpoint, body }) => {
      const response = await fetch('http://127.0.0.1:18923/api/' + endpoint, { method: body ? 'POST' : 'GET',
        headers: { 'X-DLSS-Token': window.electronAPI.localToken, 'Content-Type': 'application/json' },
        body: body ? JSON.stringify(body) : undefined });
      const result = await response.json();
      if (!response.ok) throw Error(JSON.stringify(result));
      return result;
    }, { endpoint, body });
    original = await api('settings');
    await api('settings', { guidance_mode: 0, render_pipeline: ['nr', 'dlaa', 'vsr'], working_scale: 1,
      reconstruction_mode: 'single', export_backend: 'gpu_transfer', frame_generation: 'off',
      vsr_output_mode: 'scale', vsr_scale: 1.86, style: 3 });
    await page.evaluate(() => localStorage.removeItem('lunarc.workspace.v1'));
    await page.reload();
    await page.waitForFunction(() => document.body.textContent.includes('工作台就绪'), {}, { timeout: 60000 });
    await page.getByRole('button', { name: '重建工作台', exact: true }).click();
    const pipeline = page.getByRole('region', { name: '渲染顺序', exact: true });
    const preset = page.getByLabel('RTX VSR 倍率预设', { exact: true });
    const input = page.getByLabel('RTX VSR 输出倍率', { exact: true });
    const order = () => page.locator('.render-pipeline ol > li').evaluateAll(nodes => nodes.map(node => node.dataset.stage));
    const persisted = expected => page.waitForFunction(async expected => {
      const settings = await (await fetch('http://127.0.0.1:18923/api/settings', {
        headers: { 'X-DLSS-Token': window.electronAPI.localToken },
      })).json();
      return Object.entries(expected).every(([key, value]) => JSON.stringify(settings[key]) === JSON.stringify(value));
    }, expected, { timeout: 20000 });
    await page.locator('.qi-advanced > summary').click();
    const style = page.locator('select[aria-label="画面风格"]');
    assert.deepEqual(await style.locator('option').evaluateAll(nodes => nodes.map(node => ({ value: node.value, label: node.textContent }))),
      [{ value: '0', label: '默认' }, { value: '1', label: '自然' }, { value: '2', label: '电影' }]);
    assert.equal(await page.getByText('画质预设', { exact: true }).count(), 0);
    assert.equal(await style.inputValue(), '0');
    await persisted({ style: 0 });
    await style.selectOption('1'); await persisted({ style: 1 });
    await style.selectOption('2'); await persisted({ style: 2 });
    // Restored workspace drafts must retire style 3 before submitting to the backend.
    await page.evaluate(() => {
      const draft = JSON.parse(localStorage.getItem('lunarc.workspace.v1'));
      draft.settings.style = 3;
      localStorage.setItem('lunarc.workspace.v1', JSON.stringify(draft));
    });
    await page.reload();
    await page.waitForFunction(() => document.body.textContent.includes('已恢复参数'), {}, { timeout: 60000 });
    assert.equal(await style.inputValue(), '0');
    await persisted({ style: 0 });
    assert.equal(await preset.inputValue(), 'custom');
    assert.equal(await input.inputValue(), '1.86');
    await page.getByRole('button', { name: 'VSR 倍率增加 1', exact: true }).click();
    assert.equal(await input.inputValue(), '2.86');
    await input.press('ArrowDown');
    assert.equal(await input.inputValue(), '1.86');
    await input.fill('2.5');
    await persisted({ vsr_scale: 2.5 });
    // Selecting/editing a number cannot start a card drag.
    await input.selectText();
    const box = await input.boundingBox();
    await page.mouse.move(box.x + 15, box.y + box.height / 2);
    await page.mouse.down(); await page.mouse.move(box.x + 90, box.y + box.height / 2, { steps: 6 }); await page.mouse.up();
    assert.deepEqual(await order(), ['nr', 'dlaa', 'vsr']);
    assert.equal(await page.locator('.render-dragging, .render-pipeline [draggable="true"]').count(), 0);
    await preset.selectOption('2'); await persisted({ vsr_scale: 2 });
    assert.equal(await input.count(), 0);
    await preset.selectOption('4'); await persisted({ vsr_scale: 4 });
    await preset.selectOption('custom'); await input.fill('1.86');
    await input.fill(''); await input.blur(); assert.equal(await input.inputValue(), '1.86');
    await persisted({ vsr_scale: 1.86 });

    async function dragTo(stage, target, cancel = false) {
      const handle = page.getByRole('button', { name: '拖动排序 ' + ({ nr: 'DLSSNR', dlaa: 'DLAA', vsr: 'RTX VSR', vfg: 'NVIDIA VFG' })[stage], exact: true });
      await handle.scrollIntoViewIfNeeded();
      const handleBox = await handle.boundingBox();
      const rects = await page.locator('.render-pipeline ol > li').evaluateAll(nodes => nodes.map(node => ({ stage: node.dataset.stage, top: node.getBoundingClientRect().top, height: node.getBoundingClientRect().height })));
      const from = rects.find(rect => rect.stage === stage), to = rects.find(rect => rect.stage === target);
      const delta = (to.top + to.height / 2) - (from.top + from.height / 2) + (to.top > from.top ? 12 : -12);
      await page.mouse.move(handleBox.x + handleBox.width / 2, handleBox.y + handleBox.height / 2);
      await page.mouse.down();
      await page.mouse.move(handleBox.x + handleBox.width / 2, handleBox.y + handleBox.height / 2 + delta, { steps: 12 });
      await page.waitForFunction(() => document.querySelector('.render-dragging') && [...document.querySelectorAll('.render-sorting li:not(.render-dragging)')].some(node => Math.abs(new DOMMatrix(getComputedStyle(node).transform).m42) > 1));
      assert.match(await page.locator('.render-sorting li:not(.render-dragging)').first().evaluate(node => getComputedStyle(node).transition), /transform/);
      if (cancel) await page.keyboard.press('Escape');
      await page.mouse.up();
    }
    await dragTo('nr', 'vsr');
    await persisted({ render_pipeline: ['dlaa', 'vsr', 'nr'] });
    await dragTo('nr', 'dlaa');
    await persisted({ render_pipeline: ['nr', 'dlaa', 'vsr'] });
    await dragTo('nr', 'vsr', true);
    assert.deepEqual(await order(), ['nr', 'dlaa', 'vsr']);
    await page.getByRole('button', { name: 'RTX VSR 上移', exact: true }).click();
    await persisted({ render_pipeline: ['nr', 'vsr', 'dlaa'] });
    await page.getByRole('button', { name: '拖动排序 RTX VSR', exact: true }).press('ArrowUp');
    await persisted({ render_pipeline: ['vsr', 'nr', 'dlaa'] });
    assert.deepEqual(await page.getByLabel('RTX VSR 档位', { exact: true }).locator('option').allTextContents(), ['低','中','高','极高']);
    await page.getByRole('button', { name: 'NVIDIA VFG', exact: true }).click();
    await persisted({ render_pipeline: ['vsr','nr','dlaa','vfg'], frame_generation:'vfg' });
    await dragTo('vfg','vsr');
    await persisted({ render_pipeline: ['vfg','vsr','nr','dlaa'] });
    const vfg = page.getByLabel('NVIDIA VFG 卡片', { exact: true });
    assert.equal(await vfg.count(), 1);
    await vfg.getByLabel('导出插帧', { exact: true }).selectOption('vfg');
    await vfg.getByLabel('插帧倍率', { exact: true }).selectOption('8');
    await vfg.getByLabel('VFG 自动镜头切换检测', { exact: true }).uncheck();
    await persisted({ frame_generation: 'vfg', frame_multiplier: 8, framegen_scene_detection: false });
    await vfg.getByLabel('导出插帧', { exact: true }).selectOption('vfg_dynamic');
    await vfg.getByLabel('插帧目标帧率', { exact: true }).fill('120');
    await persisted({ frame_generation: 'vfg_dynamic', framegen_target_fps: '120' });
    await page.locator('.render-pipeline').screenshot({ path: path.join(out, 'controls.png') });
    for (const name of ['RTX VSR','DLSSNR','DLAA']) await page.getByRole('button', {name:'移除 '+name,exact:true}).click();
    await persisted({render_pipeline:['vfg'],frame_generation:'vfg_dynamic',guidance_mode:0});
    assert.deepEqual(await order(), ['vfg']);
    await page.locator('.render-pipeline').screenshot({ path:path.join(out,'vfg_only.png') });
    await page.getByRole('button',{name:'DLSSNR',exact:true}).click();
    await page.getByRole('button', { name: '移除 NVIDIA VFG', exact: true }).click();
    await persisted({ frame_generation: 'off' });
    await api('settings', {render_pipeline:['vsr'],frame_generation:'off',vsr_quality:4,vsr_output_mode:'scale',vsr_scale:2,guidance_mode:0});
    await api('video/load', {path:path.join(root,'bench_out/queue_factory_20261011/kurumi_four.mp4')});
    await page.evaluate(()=>localStorage.removeItem('lunarc.workspace.v1'));
    await page.reload();
    await page.getByLabel('预览实际执行',{exact:true}).filter({hasText:'极高'}).waitFor({timeout:60000});
    await page.getByRole('button',{name:'按输出原始像素查看',exact:true}).click();
    await page.waitForFunction(()=>document.querySelector('select[aria-label="预览缩放"]')?.selectedOptions[0]?.textContent==='原像素');
    await page.getByLabel('预览实际执行',{exact:true}).filter({hasText:'3840×2160'}).waitFor({timeout:60000});
    await page.waitForFunction(()=>{const img=document.querySelector('.dlss-preview-image');if(!img?.naturalWidth)return false;const box=img.getBoundingClientRect();return Math.abs(box.width*devicePixelRatio/img.naturalWidth-1)<.01&&Math.abs(box.height*devicePixelRatio/img.naturalHeight-1)<.01},{},{timeout:60000});
    const pixelView=await page.locator('.dlss-preview-image').evaluate(img=>({natural:[img.naturalWidth,img.naturalHeight],display:[img.getBoundingClientRect().width,img.getBoundingClientRect().height],dpr:devicePixelRatio}));
    fs.writeFileSync(path.join(out,'pixel_view.json'),JSON.stringify(pixelView,null,2));
    await page.locator('.dlss-monitor').screenshot({path:path.join(out,'vsr_ultra_preview.png')});
    fs.writeFileSync(path.join(out, 'ui.json'), JSON.stringify({ status: 'PASS', checks: ['quality preset removed', 'three supported styles', 'legacy backend/workspace style migration', 'presets', 'fractional input', 'unit steps', 'input selection does not drag', 'animated down/up sorting', 'Escape cancellation', 'keyboard sorting', 'VFG animated sorting and standalone persistence', 'VSR low/medium/high/ultra labels and live ultra counter', 'native pixel preview dimensions'] }, null, 2));
    console.log('PASS: three styles and legacy migration, VSR controls, animated sorting, VFG card');
  } finally {
    if (api && original) await api('settings', original);
    await app.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1 });
