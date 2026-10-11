// Verify activation of accepted optional components in the actual packaged app.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..'),{_electron}=require(path.join(root,'frontend/node_modules/playwright'));
const [executable,components,out]=process.argv.slice(2).map(p=>path.resolve(p));
(async()=>{
  assert(!fs.existsSync(out),'Fresh evidence directory required');fs.mkdirSync(out,{recursive:true});
  if(await fetch('http://127.0.0.1:18923/api/system').then(()=>true).catch(()=>false))throw Error('User backend running');
  const profile=path.join(out,'profile');fs.mkdirSync(profile);fs.symlinkSync(components,path.join(profile,'components'),'junction');
  let app;const result={passed:false,scope:'verified component reuse and packaged restart activation'};
  try{
    const env={...process.env,E2E_TEST:'1',PYTHON_PATH:'Z:/missing',PYTHONHOME:'Z:/missing',PYTHONPATH:'Z:/missing'};delete env.ELECTRON_RUN_AS_NODE;
    app=await _electron.launch({executablePath:executable,args:['--user-data-dir='+profile],env});
    const page=await app.firstWindow();await page.getByRole('heading',{name:'开始使用',exact:true}).waitFor();
    const state=await page.evaluate(()=>window.electronAPI.getComponents());assert(state.components.every(c=>c.installed));
    await page.getByRole('button',{name:'跳过教学',exact:true}).click();
    await page.getByRole('button',{name:'队列空闲',exact:true}).waitFor({timeout:60000});
    const report=await page.evaluate(async()=>{
      const r=await fetch('http://127.0.0.1:18923/api/environment',{headers:{'X-DLSS-Token':window.electronAPI.localToken}});if(!r.ok)throw Error(await r.text());return r.json();
    });
    result.report=report;
    assert(report.isolated&&report.standalone);
    for(const id of ['cuda','torch_cuda','raft'])assert.equal(report.checks.find(c=>c.id===id)?.status,'ok',id);
    assert(fs.realpathSync(report.python).toLowerCase().startsWith(fs.realpathSync(components).toLowerCase()+path.sep));
    result.report=report;result.components=state.components;result.passed=true;
  }catch(e){result.error=String(e);throw e}
  finally{fs.writeFileSync(path.join(out,'result.json'),JSON.stringify(result,null,2));if(app)await app.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
