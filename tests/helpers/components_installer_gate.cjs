// Real Electron UI, managed runtime, WebSocket and native preview acceptance.
// Download caches were populated by the separately recorded official-source gate.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..');
const {_electron}=require(path.join(root,'frontend/node_modules/playwright'));
const [executable,out,cache,fixture,dll]=process.argv.slice(2).map(p=>path.resolve(p));
const result={passed:false,actualElectron:true,componentDownloads:'reuse previously verified official-source cache; no developer Python',checks:[]};
const api=async(page,route,method='GET',body)=>page.evaluate(async({route,method,body})=>{
  const r=await fetch('http://127.0.0.1:18923'+route,{method,headers:{'X-DLSS-Token':window.electronAPI.localToken,'Content-Type':'application/json'},...(body===undefined?{}:{body:JSON.stringify(body)})});
  const data=await r.json();if(!r.ok)throw Error(JSON.stringify(data));return data;
},{route,method,body});
async function eventually(fn,timeout=60000){const deadline=Date.now()+timeout;while(Date.now()<deadline){if(await fn())return;await new Promise(r=>setTimeout(r,200))}throw Error('Timed out')}
(async()=>{
  assert(!fs.existsSync(out),'Fresh evidence folder required');fs.mkdirSync(out,{recursive:true});
  if(await fetch('http://127.0.0.1:18923/api/system').then(()=>true).catch(()=>false))throw Error('User backend running; refusing to attach');
  const resources=path.join(path.dirname(executable),'resources');
  assert(!fs.existsSync(path.join(resources,'runtime/python/python.exe')));
  assert(!fs.existsSync(path.join(resources,'runtime/torch_home')));
  assert(!fs.existsSync(path.join(resources,'nvngx_dlssnr.dll')));
  const profile=path.join(out,'profile'),components=path.join(profile,'components');fs.mkdirSync(components,{recursive:true});
  for(const name of ['python','cache','downloads'])fs.symlinkSync(path.join(cache,name),path.join(components,name),'junction');
  let app;
  try{
    const env={...process.env,E2E_TEST:'1',PYTHON_PATH:'Z:/missing/python.exe',PYTHONHOME:'Z:/missing',PYTHONPATH:'Z:/missing',PATH:process.env.SystemRoot+'/System32;'+process.env.SystemRoot,E2E_MOCK_FILE_PATH:fixture};delete env.ELECTRON_RUN_AS_NODE;
    app=await _electron.launch({executablePath:executable,args:['--user-data-dir='+profile],env,timeout:30000});
    const page=await app.firstWindow(),errors=[];page.on('pageerror',e=>errors.push(String(e)));
    await page.setViewportSize({width:1480,height:940});
    await page.getByRole('heading',{name:'开始使用',exact:true}).waitFor();
    assert.equal((await page.evaluate(()=>window.electronAPI.getBackendStartup())).state,'setup');
    assert.equal(await page.getByRole('button',{name:'队列空闲',exact:true}).count(),0);
    await page.screenshot({path:path.join(out,'first-start.png')});
    await page.locator('.component-row').filter({hasText:'基础运行环境'}).getByRole('button',{name:'安装',exact:true}).click();
    await eventually(async()=>{const s=await page.evaluate(()=>window.electronAPI.getComponents());if(s.error)throw Error(s.error);return s.components.find(c=>c.id==='base').installed},180000);
    await page.locator('.component-row').filter({hasText:'视频处理组件'}).getByRole('button',{name:'安装',exact:true}).click();
    await eventually(async()=>{const s=await page.evaluate(()=>window.electronAPI.getComponents());if(s.error)throw Error(s.error);return !s.requiresSetup},180000);
    result.checks.push('required components installed through real IPC', 'unavailable developer paths ignored');
    await page.getByRole('button',{name:'下一步',exact:true}).click();await page.getByRole('heading',{name:'导入素材，按顺序处理',exact:true}).waitFor();
    await page.screenshot({path:path.join(out,'guide-workflow.png')});
    await page.getByRole('button',{name:'下一步',exact:true}).click();await page.getByRole('button',{name:'进入工作台',exact:true}).click();
    await page.getByRole('button',{name:'队列空闲',exact:true}).waitFor({timeout:60000});
    const report=await api(page,'/api/environment');fs.writeFileSync(path.join(out,'environment.json'),JSON.stringify(report,null,2));
    assert(report.standalone&&report.isolated);assert.equal(report.checks.find(c=>c.id==='cuda').status,'ok');
    assert.equal(report.checks.find(c=>c.id==='torch_cuda').status,'missing');
    assert.equal(report.features.find(c=>c.id==='gpu_transfer').status,'pending');
    const system=await api(page,'/api/system');assert(system.gpu.includes('5060'));result.gpu=system;
    result.checks.push('native driver GPU detection without Torch','native bridge dependency closure loaded');
    const event=await page.evaluate(()=>new Promise((resolve,reject)=>{const ws=new WebSocket('ws://127.0.0.1:18923/ws/queue?token='+encodeURIComponent(window.electronAPI.localToken));const timer=setTimeout(()=>{ws.close();reject(Error('queue timeout'))},6000);ws.onmessage=e=>{clearTimeout(timer);ws.close();resolve(JSON.parse(e.data))};ws.onerror=()=>reject(Error('queue failure'))}));
    assert.equal(event.type,'init');result.checks.push('real queue WebSocket');
    const imported=await api(page,'/api/runtime/dlssnr/import','POST',{path:dll});assert(imported.id);await api(page,'/api/runtime/dlssnr/activate','POST',{id:imported.id});
    await api(page,'/api/settings','POST',{guidance_mode:0,render_pipeline:['nr'],frame_generation:'off'});
    await page.getByRole('button',{name:'导入',exact:true}).click();
    await page.waitForFunction(()=>{const img=document.querySelector('.dlss-preview-image');return img instanceof HTMLImageElement&&img.complete&&img.naturalWidth>0&&!document.querySelector('.dlss-player-error')},{},{timeout:60000});
    const frame=await page.evaluate(()=>{const img=document.querySelector('.dlss-preview-image');return {width:img.naturalWidth,height:img.naturalHeight}});
    assert(frame.width>0&&frame.height>0);result.frame=frame;
    const preview=await api(page,'/api/preview/diagnostics');assert(preview.layers.layer_session_ids.some(id=>id>0));result.preview=preview;
    await page.getByRole('button',{name:'开始播放',exact:true}).click();
    await eventually(async()=>{const d=await api(page,'/api/preview/diagnostics');return d.frame>preview.frame},30000);
    await page.getByRole('button',{name:'暂停播放',exact:true}).click();
    result.checks.push('native preview playback frame advancement');
    result.checks.push('real native NR session and JPEG preview without Torch');
    await page.getByRole('navigation',{name:'主导航'}).getByRole('button',{name:'软件设置',exact:true}).click();
    await page.getByRole('heading',{name:'组件管理',exact:true}).waitFor();assert.match(await page.locator('.component-panel').innerText(),/RAFT 光流/);
    await page.getByRole('button',{name:'打开使用教程',exact:true}).click();await page.getByRole('heading',{name:'使用教程',exact:true}).waitFor();
    await page.screenshot({path:path.join(out,'guide-reopen.png')});
    assert.deepEqual(errors,[]);result.checks.push('tutorial navigation and reopening','no renderer errors');result.passed=true;
  }catch(e){result.error=String(e);throw e}
  finally{fs.writeFileSync(path.join(out,'result.json'),JSON.stringify(result,null,2));if(app)await app.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
