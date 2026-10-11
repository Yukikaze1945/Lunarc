// Real Electron/backend smoke in an isolated user-data profile. No DLL distribution.
const path = require('node:path');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '../..');
const {_electron} = require(path.join(root, 'frontend/node_modules/playwright'));
const out = path.join(root, 'build/dlssnr-runtime-ui-20261011');
fs.mkdirSync(out, {recursive: true});
(async () => {
  const env = {...process.env, E2E_TEST: '1'};
  delete env.ELECTRON_RUN_AS_NODE;
  const app = await _electron.launch({
    executablePath: require(path.join(root, 'frontend/node_modules/electron')),
    args: [path.join(root, 'frontend'), '--user-data-dir=' + path.join(out, 'profile-' + Date.now())],
    env, timeout: 60000,
  });
  try {
    const page = await app.firstWindow();
    await page.setViewportSize({width: 1440, height: 1100});
    await page.waitForFunction(() => /工作台就绪|已恢复参数/.test(document.body.textContent), {}, {timeout: 60000});
    await page.getByRole('button', {name: '运行与诊断', exact: true}).click();
    const panel = page.locator('.dlssnr-runtime');
    await panel.getByLabel('DLSSNR DLL 或 ZIP 路径').fill(path.join(root, 'nvngx_dlssnr.dll'));
    await panel.getByRole('button', {name: '导入', exact: true}).click();
    await panel.getByText('已校验并缓存，请选择使用版本。', {exact: true}).waitFor({timeout: 30000});
    await panel.getByRole('button', {name: '使用', exact: true}).click();
    await panel.getByText('已选择，新会话使用该版本。', {exact: true}).waitFor({timeout: 30000});
    assert.equal(await panel.locator('.runtime-current strong').textContent(), '版本 310.8.0.0');
    await panel.getByRole('button', {name: '重新校验', exact: true}).click();
    await panel.getByText('文件哈希与数字签名检查通过。', {exact: true}).waitFor({timeout: 30000});
    await panel.getByText('版本目录与下载', {exact: true}).click();
    assert.equal(await panel.getByRole('button', {name: '已缓存', exact: true}).isDisabled(), true);
    await panel.scrollIntoViewIfNeeded();
    await page.screenshot({path: path.join(out, 'runtime-manager.png')});
    await panel.getByRole('button', {name: '取消选择', exact: true}).click();
    await panel.getByText('已取消选择。', {exact: true}).waitFor({timeout: 30000});
    await panel.getByRole('button', {name: '删除版本 310.8.0.0', exact: true}).click();
    await panel.getByText('缓存版本已删除。', {exact: true}).waitFor({timeout: 30000});
    assert.equal(await panel.locator('.runtime-versions > div').count(), 0);
    assert.equal(fs.existsSync(path.join(root, 'nvngx_dlssnr.dll')), true);
    fs.writeFileSync(path.join(out, 'result.json'), JSON.stringify({passed: true,
      checks: ['real import', 'signature and version UI', 'selection', 'forced recheck',
        'catalog cache state', 'unselect', 'delete', 'original preserved']}, null, 2));
    console.log('Runtime manager Electron/backend smoke passed');
  } finally {await app.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
