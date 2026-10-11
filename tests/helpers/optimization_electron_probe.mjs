// Real Electron acceptance; closes only the app/backend it launches.
import { _electron as electron } from '@playwright/test';
import path from 'node:path';
import os from 'node:os';
import fs from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import { ROOT_DIR, pythonExe } from './paths.js';

const out = path.resolve(process.argv[2] || path.join(ROOT_DIR, 'bench_out/optimization_20261009/electron_core'));
try {
  if ((await fs.readdir(out)).length) throw new Error(`Evidence directory is not fresh: ${out}`);
} catch (error) { if (error.code !== 'ENOENT') throw error; }
await fs.mkdir(out, { recursive: true });
const errors = [];
const ui = process.argv.includes('--ui');
const packaged = process.argv.includes('--packaged');
const scale = process.argv.find(a=>a.startsWith('--scale='))?.split('=')[1];
const executable = packaged ? path.join(ROOT_DIR,'bench_out/optimization_20261009/packaged_release/win-unpacked/Lunarc.exe') : path.join(ROOT_DIR,'frontend/node_modules/electron/dist/electron.exe');
const userData = path.join(os.tmpdir(), `lunarc_probe_${Date.now()}`);
if(await fetch('http://127.0.0.1:18923/api/system').then(()=>true).catch(()=>false))throw new Error('18923 belongs to an existing backend; probe refuses to attach');
const app = await electron.launch({
  executablePath: executable,
  cwd: path.join(ROOT_DIR, 'frontend'),
  args: [...(packaged ? [] : ['.']), `--user-data-dir=${userData}`, '--no-sandbox', ...(scale ? [`--force-device-scale-factor=${scale}`] : [])],
  env: { ...process.env, E2E_TEST: '1', PYTHON_PATH: pythonExe(),
    E2E_MOCK_FILE_PATH: path.join(ROOT_DIR, 'bench_out/optimization_20261009/sdr_source.mkv') },
});
let backendPid = null;
app.process().stdout?.on('data',chunk=>{const match=String(chunk).match(/Python backend spawned with PID:\s*(\d+)/);if(match)backendPid=Number(match[1]);});
try {
  const page = await app.firstWindow({ timeout: 90000 });
  page.on('pageerror', e => errors.push(String(e)));
  let acceptReset = false;
  page.on('dialog', d => acceptReset ? d.accept() : d.dismiss());
  await page.waitForLoadState('domcontentloaded');
  await page.waitForFunction(async () => {
    try {
      const response = await fetch('http://127.0.0.1:18923/api/system');
      const system = await response.json();
      return response.ok && system.backend_ready === true && Boolean(system.gpu);
    } catch { return false; }
  }, null, { timeout: 45000 });
  if(process.platform==='win32'){
    // netstat observes the live listener directly; Win32 CIM's TCP inventory can lag.
    const value=execFileSync('netstat.exe',['-ano','-p','tcp'],{encoding:'utf8',windowsHide:true}).split(/\r?\n/).find(line=>/127\.0\.0\.1:18923\s+.*LISTENING/.test(line))?.trim().split(/\s+/).at(-1);
    backendPid=Number(value);if(!Number.isInteger(backendPid)||backendPid<=0)throw new Error('Invalid listening backend PID');
    const command=execFileSync('powershell.exe',['-NoProfile','-Command',`(Get-CimInstance Win32_Process -Filter 'ProcessId=${backendPid}').CommandLine`],{encoding:'utf8',windowsHide:true}).trim();
    const expected=packaged?path.join(path.dirname(executable),'resources/server.py'):path.join(ROOT_DIR,'server.py');
    if(!command.replaceAll('\\','/').toLowerCase().includes(expected.replaceAll('\\','/').toLowerCase()))throw new Error('Listening backend does not belong to this app layout');
  }
  await page.screenshot({ path: path.join(out, 'startup.png') });
  const interactions = [];
  if (ui) {
    if (!(await page.title()).includes('Lunarc')) throw new Error('New brand window title missing');
    const png = await page.evaluate(async () => {
      const img = new Image(); img.src = './lunarc.svg'; await img.decode();
      const canvas = document.createElement('canvas'); canvas.width=256;canvas.height=256;
      canvas.getContext('2d').drawImage(img,0,0,256,256);return canvas.toDataURL('image/png').split(',')[1];
    });
    await fs.writeFile(path.join(out,'brand-preview.png'),Buffer.from(png,'base64'));
    await page.getByRole('button',{name:'导入',exact:true}).click();
    await page.waitForFunction(()=>document.querySelector('.dlss-material strong')?.textContent.includes('sdr_source.mkv'));
    await page.locator('.dlss-view-column img').first().waitFor({state:'visible',timeout:45000});
    await page.screenshot({path:path.join(out,'workbench.png')});interactions.push('real fixture import and preview');
    await page.locator('.qi-advanced > summary').click();
    for(const tab of ['图层','画质','时域','HDR','基础']) {
      await page.getByRole('tab',{name:new RegExp(tab)}).click();
      await page.screenshot({path:path.join(out,`inspector_${tab}.png`)});
    }
    await page.getByRole('tab',{name:/画质/}).click();
    const slider=page.getByRole('slider',{name:'总增益'}).or(page.getByRole('slider',{name:'总残差强度'}));
    await slider.evaluate(el=>{Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(el,'0.83');el.dispatchEvent(new Event('input',{bubbles:true}));el.dispatchEvent(new Event('change',{bubbles:true}));});
    await page.waitForFunction(async()=>{const t=window.electronAPI.localToken;const d=await (await fetch('http://127.0.0.1:18923/api/settings',{headers:{'X-DLSS-Token':t}})).json();return d.settings?.post_controls?.total_gain===.83});
    interactions.push('post-control slider verified against real backend');
    await page.locator('.qi-reset').click();
    await page.waitForFunction(async()=>{const d=await (await fetch('http://127.0.0.1:18923/api/settings',{headers:{'X-DLSS-Token':window.electronAPI.localToken}})).json();return d.settings?.post_controls?.total_gain===1});
    interactions.push('neutral reset verified against real backend');
    for(const [id,name] of [['deliver','导出与队列'],['diagnostics','运行与诊断'],['presets','预设管理'],['settings','软件设置'],['about','关于 Lunarc'],['home','首页']]) {
      await page.getByRole('navigation',{name:'主导航'}).getByRole('button',{name:new RegExp(name)}).click();
      await page.screenshot({path:path.join(out,`${id}.png`)});
    }
    await page.getByRole('navigation',{name:'主导航'}).getByRole('button',{name:/预设管理/}).click();
    await page.getByRole('textbox',{name:'预设名称'}).fill('Electron 验收预设');
    await page.getByRole('button',{name:'保存预设',exact:true}).click();
    await page.getByRole('heading',{name:'Electron 验收预设'}).waitFor();
    await page.getByRole('button',{name:'删除预设 Electron 验收预设',exact:true}).click();
    if(await page.getByRole('heading',{name:'Electron 验收预设'}).count())throw new Error('Preset delete failed');
    interactions.push('preset save and delete');
    for(const [width,height] of [[1000,680],[1920,1080],[3840,2160]]) {
      await app.evaluate(({BrowserWindow},size)=>BrowserWindow.getAllWindows()[0].setSize(size[0],size[1]),[width,height]);
      await page.getByRole('navigation',{name:'主导航'}).getByRole('button',{name:/重建工作台/}).click();
      await page.screenshot({path:path.join(out,`workbench_${width}.png`)});
      const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
      if(overflow)throw new Error(`Window overflow at ${width}`);
      await page.getByRole('navigation',{name:'主导航'}).getByRole('button',{name:/导出与队列/}).click();
      await page.screenshot({path:path.join(out,`export_${width}.png`)});
    }
    interactions.push('1000/1920/3840 window layout screenshots');
    await page.getByRole('navigation',{name:'主导航'}).getByRole('button',{name:/预设管理/}).click();
    await page.getByRole('textbox',{name:'预设名称'}).fill('恢复测试');
    await page.getByRole('button',{name:'保存预设',exact:true}).click();
    await page.getByRole('navigation',{name:'主导航'}).getByRole('button',{name:/软件设置/}).click();
    await page.locator('select').filter({has:page.locator('option[value=compact]')}).selectOption('compact');
    await page.getByRole('checkbox',{name:'减少界面动效'}).check();
    // Electron downloads are observed through its native session, not a browser-only Playwright event.
    await app.evaluate(({session},dir)=>{globalThis.__probeDownloadDone=new Promise(resolve=>{session.defaultSession.once('will-download',(_event,item)=>{item.setSavePath(`${dir}/settings-backup.json`);item.once('done',(_ev,state)=>resolve({state,path:item.getSavePath()}));});});},out);
    await page.getByRole('button',{name:'备份参数',exact:true}).click();
    const download=await app.evaluate(async()=>await globalThis.__probeDownloadDone);if(download.state!=='completed')throw new Error(`Native Electron backup download failed: ${download.state}`);
    const backup=JSON.parse(await fs.readFile(path.join(out,'settings-backup.json'),'utf8'));
    if(backup.version!==1||backup.settings.post_controls.total_gain!==1)throw new Error('Settings backup invalid');
    await page.getByRole('navigation',{name:'主导航'}).getByRole('button',{name:/预设管理/}).click();
    await page.getByLabel('导入参数文件').setInputFiles({name:'partial-v1.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify({version:1,settings:{...backup.settings,post_controls:{total_gain:.91}}}))});
    await page.waitForFunction(async()=>{const d=await (await fetch('http://127.0.0.1:18923/api/settings',{headers:{'X-DLSS-Token':window.electronAPI.localToken}})).json();return d.settings?.post_controls?.total_gain===.91});
    await page.waitForFunction(()=>JSON.parse(localStorage.getItem('lunarc.workspace.v1')).settings.post_controls.total_gain===.91);
    let injectedFailure=false;
    await page.route('**/api/settings',async route=>{if(!injectedFailure&&route.request().method()==='GET'){injectedFailure=true;await route.fulfill({status:503,contentType:'application/json',body:'{}'});}else await route.continue();});
    await page.reload();
    await page.getByRole('navigation',{name:'主导航'}).waitFor();
    await page.waitForFunction(()=>document.body.innerText.includes('已恢复参数'),{timeout:20000});
    await page.unroute('**/api/settings');
    if(!injectedFailure)throw new Error('Startup recovery scenario did not execute');
    if(await page.evaluate(()=>document.documentElement.dataset.density!=='compact'||document.documentElement.dataset.reducedMotion!=='true'))throw new Error('Display preferences did not restore');
    await page.getByRole('navigation',{name:'主导航'}).getByRole('button',{name:/预设管理/}).click();
    await page.getByRole('heading',{name:'恢复测试'}).waitFor();
    await page.getByRole('navigation',{name:'主导航'}).getByRole('button',{name:/软件设置/}).click();
    acceptReset=true;await page.getByRole('button',{name:'恢复增强默认值',exact:true}).click();
    await page.waitForFunction(async()=>{const d=await (await fetch('http://127.0.0.1:18923/api/settings',{headers:{'X-DLSS-Token':window.electronAPI.localToken}})).json();return d.settings?.post_controls?.total_gain===1});acceptReset=false;
    interactions.push('backup, partial-v1 import, reload persistence, preset persistence, confirmed reset, transient startup failure recovery');
    const legacy=await page.evaluate(()=>/DaVinci|Resolve|Studio Pro/.test(document.body.innerText));
    if(legacy)throw new Error('Old brand still visible');
  }
  const title = await page.title();
  const body = await page.locator('body').innerText();
  if (errors.length) throw new Error(errors.join('\n'));
  await fs.writeFile(path.join(out, 'result.json'), JSON.stringify({ title, errors,
    actualElectron: true, packaged, executable, scale:scale||'default', devicePixelRatio:await page.evaluate(()=>devicePixelRatio), backendReady: true, userData, interactions, screenshot: 'startup.png', body }, null, 2));
  console.log('Real Electron launch + backend ready + screenshot: PASS');
} catch (error) {
  await fs.writeFile(path.join(out, 'failure.txt'), String(error.stack || error));
  throw error;
} finally {
  await app.close();
  if(!backendPid)await fs.writeFile(path.join(out,'cleanup.json'),JSON.stringify({backendPid:null,backendExited:null,reason:'No owned backend PID recorded'},null,2));
  if(backendPid){let alive=true;try{process.kill(backendPid,0)}catch{alive=false}
    await fs.writeFile(path.join(out,'cleanup.json'),JSON.stringify({backendPid,backendExited:!alive},null,2));
    if(alive)throw new Error('Owned Python backend survived Electron close');}
}
