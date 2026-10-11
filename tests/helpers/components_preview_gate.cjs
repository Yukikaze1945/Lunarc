const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..');const {_electron}=require(path.join(root,'frontend/node_modules/playwright'));
const [executable,profile,out,fixture,dll]=process.argv.slice(2).map(x=>path.resolve(x));
const result={passed:false,reusedAcceptedBaseComponents:true,torchInstalledInBase:false};fs.mkdirSync(out,{recursive:true});
async function api(page,route,method='GET',body){return page.evaluate(async({route,method,body})=>{const r=await fetch('http://127.0.0.1:18923'+route,{method,headers:{'X-DLSS-Token':window.electronAPI.localToken,'Content-Type':'application/json'},...(body===undefined?{}:{body:JSON.stringify(body)})});const data=await r.json();if(!r.ok)throw Error(JSON.stringify(data));return data},{route,method,body})}
(async()=>{
  if(await fetch('http://127.0.0.1:18923/api/system').then(()=>true).catch(()=>false))throw Error('User backend running');
  let app,page;try{
    const env={...process.env,E2E_TEST:'1',PYTHON_PATH:'Z:/missing',PYTHONPATH:'Z:/missing',PYTHONHOME:'Z:/missing',E2E_MOCK_FILE_PATH:fixture};delete env.ELECTRON_RUN_AS_NODE;
    app=await _electron.launch({executablePath:executable,args:['--user-data-dir='+profile],env});page=await app.firstWindow();
    await page.getByRole('button',{name:'队列空闲',exact:true}).waitFor({timeout:60000});
    const imported=await api(page,'/api/runtime/dlssnr/import','POST',{path:dll});await api(page,'/api/runtime/dlssnr/activate','POST',{id:imported.id});
    const settings=await api(page,'/api/settings');
    await page.evaluate(settings=>localStorage.setItem('lunarc.workspace.v1',JSON.stringify({version:1,settings:{...(settings.settings||settings),guidance_mode:0,render_pipeline:['nr'],frame_generation:'off',working_scale:0.75,export_backend:'gpu_transfer'},recent:[]})),settings);
    await page.reload();await page.getByRole('button',{name:'导入',exact:true}).click();
    await page.waitForFunction(()=>{const img=document.querySelector('.dlss-preview-image');return img instanceof HTMLImageElement&&img.complete&&img.naturalWidth>0&&!document.querySelector('.dlss-player-error')},{},{timeout:60000});
    result.preview=await api(page,'/api/preview/diagnostics');assert(result.preview.layers.layer_session_ids.some(id=>id>0));
    const report=await api(page,'/api/environment');assert(report.isolated&&report.standalone);assert.equal(report.checks.find(c=>c.id==='torch_cuda').status,'missing');
    await page.getByRole('navigation',{name:'主导航'}).getByRole('button',{name:'软件设置',exact:true}).click();await page.getByRole('heading',{name:'组件管理',exact:true}).waitFor();
    await page.getByRole('button',{name:'打开使用教程',exact:true}).click();await page.getByRole('heading',{name:'使用教程',exact:true}).waitFor();
    await page.screenshot({path:path.join(out,'guide-reopen.png')});result.passed=true;
  }catch(e){result.error=String(e);if(page){result.logs=await api(page,'/api/diagnostics/logs').catch(()=>null);result.preview=await api(page,'/api/preview/diagnostics').catch(()=>null);await page.screenshot({path:path.join(out,'failure.png')}).catch(()=>{})}throw e}
  finally{fs.writeFileSync(path.join(out,'result.json'),JSON.stringify(result,null,2));if(app)await app.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
