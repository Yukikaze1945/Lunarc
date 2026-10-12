// Real installed build, reused managed components, isolated test preferences.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..');
const {_electron}=require(path.join(root,'frontend/node_modules/playwright'));
const [exe,out,components,fixture,dll,version]=process.argv.slice(2);
assert(exe&&out&&components&&fixture&&dll&&version);
const resources=path.join(path.dirname(exe),'resources');
const result={passed:false,version,executable:exe,checks:[]};
async function api(page,route,method='GET',body){return page.evaluate(async({route,method,body})=>{
  const response=await fetch('http://127.0.0.1:18923'+route,{method,headers:{'X-DLSS-Token':window.electronAPI.localToken,'Content-Type':'application/json'},...(body===undefined?{}:{body:JSON.stringify(body)})});
  const data=await response.json();if(!response.ok)throw Error(JSON.stringify(data));return data;
},{route,method,body});}
(async()=>{
  assert(!fs.existsSync(out),'Fresh evidence required');fs.mkdirSync(out,{recursive:true});
  if(await fetch('http://127.0.0.1:18923/api/system').then(()=>true).catch(()=>false))throw Error('Existing backend owns the port');
  assert.equal(JSON.parse(fs.readFileSync(path.join(resources,'app/package.json'))).version,version);
  assert(fs.existsSync(path.join(resources,'fusion_diagnostics.py')));
  assert(!fs.existsSync(path.join(resources,'nvngx_dlssnr.dll')));
  const profile=path.join(out,'profile'),linked=path.join(profile,'components');fs.mkdirSync(linked,{recursive:true});
  for(const dir of ['base-v1','python','runtime',...(fs.existsSync(path.join(components,'raft-v1'))?['raft-v1']:[])])
    fs.symlinkSync(path.join(components,dir),path.join(linked,dir),'junction');
  for(const file of ['base-ready.json','ffmpeg-ready.json',...(fs.existsSync(path.join(components,'raft-ready.json'))?['raft-ready.json']:[])])
    fs.copyFileSync(path.join(components,file),path.join(linked,file));
  let app,page;
  try{
    const env={...process.env,E2E_TEST:'1',E2E_MOCK_FILE_PATH:fixture,PYTHON_PATH:'Z:/missing',PYTHONHOME:'Z:/missing',PYTHONPATH:'Z:/missing'};delete env.ELECTRON_RUN_AS_NODE;
    app=await _electron.launch({executablePath:exe,args:['--user-data-dir='+profile],env,timeout:30000});
    page=await app.firstWindow();await page.setViewportSize({width:1480,height:940});
    await page.getByRole('button',{name:'跳过教学',exact:true}).click({timeout:30000});
    await page.getByRole('button',{name:'队列空闲',exact:true}).waitFor({timeout:60000});
    result.environment=await api(page,'/api/environment');
    assert(result.environment.standalone&&result.environment.isolated);
    assert.equal(result.environment.checks.find(c=>c.id==='cuda').status,'ok');
    const state=await page.evaluate(()=>window.electronAPI.getComponents());
    assert(!state.requiresSetup&&state.components.filter(c=>['base','ffmpeg'].includes(c.id)).every(c=>c.installed));
    result.checks.push('installed version and fusion diagnostics','managed components after relocation','native CUDA operation');
    const imported=await api(page,'/api/runtime/dlssnr/import','POST',{path:dll});
    await api(page,'/api/runtime/dlssnr/activate','POST',{id:imported.id});
    const current=await api(page,'/api/settings');
    const settings={...(current.settings||current),guidance_mode:0,render_pipeline:['nr'],frame_generation:'off',reconstruction_mode:'dual',working_scale:.5,layer_scales:[.5,.75],layer_weights:[1,1],fusion_mode:'data_consistency_experimental',fusion_beta:.1,fusion_relative_tolerance:.001,fusion_max_iterations:128,export_backend:'gpu_transfer'};
    await page.evaluate(settings=>localStorage.setItem('lunarc.workspace.v1',JSON.stringify({version:1,settings,recent:[]})),settings);
    await page.reload();await page.getByRole('button',{name:'导入',exact:true}).click();
    await page.waitForFunction(()=>{const image=document.querySelector('.dlss-preview-image');return image instanceof HTMLImageElement&&image.complete&&image.naturalWidth>0&&!document.querySelector('.dlss-player-error')},{},{timeout:60000});
    result.preview=await api(page,'/api/preview/diagnostics');
    assert.equal(result.preview.layers.layer_count,2);
    assert.equal(new Set(result.preview.layers.layer_session_ids).size,2);
    assert(result.preview.solver.converged);
    assert.equal(result.preview.solver.diagnostics.solver_path,'primal_gpu_cg');
    assert.equal(result.preview.solver.diagnostics.batch_size,4);
    result.checks.push('two distinct NR sessions','new GPU CG solver in installed preview');
    const initial=result.preview.frame;
    await page.getByRole('button',{name:'开始播放',exact:true}).click();
    await page.waitForFunction(async initial=>{const r=await fetch('http://127.0.0.1:18923/api/preview/diagnostics',{headers:{'X-DLSS-Token':window.electronAPI.localToken}});return (await r.json()).frame>initial},initial,{timeout:30000});
    await page.getByRole('button',{name:'暂停播放',exact:true}).click();
    result.playback=await api(page,'/api/preview/diagnostics');
    result.checks.push('preview playback advances');
    await page.screenshot({path:path.join(out,'installed-workbench.png')});result.passed=true;
  }catch(error){result.error=String(error);if(page){result.logs=await api(page,'/api/diagnostics/logs').catch(()=>null);await page.screenshot({path:path.join(out,'failure.png')}).catch(()=>{})}throw error;}
  finally{fs.writeFileSync(path.join(out,'result.json'),JSON.stringify(result,null,2));if(app)await app.close();}
})().catch(error=>{console.error(error);process.exitCode=1});
