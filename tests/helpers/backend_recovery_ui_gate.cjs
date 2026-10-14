// Isolated profile. Tests an unexpected backend exit, never induces a GPU reset.
const path=require('node:path'),fs=require('node:fs'),assert=require('node:assert/strict');
const videoGate=process.argv.includes('--video');
const root=path.resolve(__dirname,'../..'),out=path.join(root,'build/backend-recovery',videoGate?'video-ui':'ui');
const {_electron}=require(path.join(root,'frontend/node_modules/playwright'));
fs.mkdirSync(out,{recursive:true});
const result={passed:false};
(async()=>{
  if(await fetch('http://127.0.0.1:18923/api/system').then(()=>true).catch(()=>false))throw Error('User backend already running; isolated check cannot use its port');
  const env={...process.env,E2E_TEST:'1',PYTHON_PATH:path.join(root,'build/standalone-runtime/python/python.exe'),
    E2E_MOCK_FILE_PATH:path.join(root,'build/p010-codec-check/sdr709_hd_source.mkv')};delete env.ELECTRON_RUN_AS_NODE;
  const app=await _electron.launch({executablePath:require(path.join(root,'frontend/node_modules/electron')),
    args:[path.join(root,'frontend'),'--user-data-dir='+path.join(out,'profile')],env,timeout:60000});
  try{
    const page=await app.firstWindow();
    const api=(route,method='GET',body)=>page.evaluate(async({route,method,body})=>{
      const response=await fetch('http://127.0.0.1:18923'+route,{method,headers:{'X-DLSS-Token':window.electronAPI.localToken,'Content-Type':'application/json'},...(body===undefined?{}:{body:JSON.stringify(body)})});
      const data=await response.json();if(!response.ok)throw Error(JSON.stringify(data));return data;
    },{route,method,body});
    await page.waitForFunction(()=>/工作台就绪|已恢复参数/.test(document.body.textContent),{},{timeout:60000});
    if(videoGate){
      const dll=require('node:child_process').execFileSync('F:/WhisperJAV/python.exe',['-c','import dlssnr_runtime; print(dlssnr_runtime.resolve_runtime())'],{cwd:root,encoding:'utf8',windowsHide:true}).trim();
      const imported=await api('/api/runtime/dlssnr/import','POST',{path:dll});await api('/api/runtime/dlssnr/activate','POST',{id:imported.id});
      await page.evaluate(()=>localStorage.setItem('lunarc.workspace.v1',JSON.stringify({version:1,settings:{guidance_mode:0,render_pipeline:['nr'],export_backend:'gpu_transfer'},recent:[]})));
      await page.reload();await page.getByRole('button',{name:'导入',exact:true}).click();
      const slider=page.getByRole('slider',{name:'时间轴帧位置',exact:true});await slider.waitFor({timeout:30000});
      await page.waitForFunction(()=>Number(document.querySelector('[aria-label="时间轴帧位置"]')?.getAttribute('aria-valuemax'))>1,{},{timeout:15000});
      await page.getByRole('button',{name:'前进一帧',exact:true}).click();
      await page.waitForFunction(()=>document.querySelector('[aria-label="时间轴帧位置"]')?.getAttribute('aria-valuenow')==='1',{},{timeout:10000});
      result.frameBefore=await slider.getAttribute('aria-valuenow');
    }
    const health=await api('/api/backend/health');result.before=health;
    const first=await api('/api/queue/add','POST',{name:'interrupted',input_path:path.join(root,'build/p010-codec-check/sdr709_hd_source.mkv'),output_path:path.join(out,'first.mkv'),settings:{guidance_mode:0,render_pipeline:['nr'],export_backend:'gpu_transfer'}});
    const second=await api('/api/queue/add','POST',{name:'pending',input_path:path.join(root,'build/p010-codec-check/sdr709_hd_source.mkv'),output_path:path.join(out,'second.mkv'),settings:{guidance_mode:0,render_pipeline:['nr'],export_backend:'gpu_transfer'}});
    // Emulate the checkpoint immediately before native work; no export is started.
    const checkpoint=path.join(out,'profile/render-queue.json'),archive=JSON.parse(fs.readFileSync(checkpoint,'utf8'));
    archive.jobs.find(job=>job.id===first.job.id).status='running';fs.writeFileSync(checkpoint,JSON.stringify(archive));
    if(!videoGate)await page.evaluate(()=>window.dispatchEvent(new Event('lunarc:open-deliver')));
    require('node:child_process').execFileSync('taskkill',['/F','/PID',String(health.pid)],{windowsHide:true,stdio:'ignore'});
    await page.waitForFunction(()=>document.body.textContent.includes('后端已恢复'),{},{timeout:60000});
    result.after=await api('/api/backend/health');assert.notEqual(result.after.pid,health.pid);
    if(videoGate){
      result.video=(await api('/api/video/current')).video;assert(result.video.path.endsWith('sdr709_hd_source.mkv'));
      result.frameAfter=await page.getByRole('slider',{name:'时间轴帧位置',exact:true}).getAttribute('aria-valuenow');
      assert.equal(result.frameAfter,result.frameBefore);
      result.settings=await api('/api/settings');assert.equal(result.settings.guidance_mode,0);
      const {expect}=require(path.join(root,'frontend/node_modules/@playwright/test'));
      await expect.poll(async()=>{
        const data=await api('/api/preview/diagnostics');return data.frame===1&&data.gpu_memory?.process?.local_usage>0;
      },{timeout:30000}).toBe(true);
      result.preview=await api('/api/preview/diagnostics');assert(result.preview.gpu_memory?.process?.local_usage>0);
    }
    result.queue=await api('/api/queue');
    assert(result.queue.jobs.find(job=>job.id===first.job.id).interrupted);
    assert.equal(result.queue.jobs.find(job=>job.id===first.job.id).status,'failed');
    assert.equal(result.queue.jobs.find(job=>job.id===second.job.id).status,'pending');
    assert.equal(result.queue.queue_running,false);
    await api('/api/queue/remove','POST',{id:first.job.id});
    const third=await api('/api/queue/add','POST',{name:'new',input_path:path.join(root,'build/p010-codec-check/sdr709_hd_source.mkv'),output_path:path.join(out,'third.mkv')});assert(third.job.id);
    await page.screenshot({path:path.join(out,'recovered.png')});
    if(videoGate){
      require('node:child_process').execFileSync('taskkill',['/F','/PID',String(result.after.pid)],{windowsHide:true,stdio:'ignore'});
      await page.waitForFunction(()=>document.body.textContent.includes('停止自动重启'),{},{timeout:10000});
      result.repeatedFailureBlocked=true;
    }
    result.passed=true;
  }catch(error){result.error=String(error);throw error}
  finally{fs.writeFileSync(path.join(out,'result.json'),JSON.stringify(result,null,2));await app.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
