// Real Electron/preload/IPC/file writes; only GitHub responses use tiny test assets.
const fs=require('node:fs');
const path=require('node:path');
const assert=require('node:assert/strict');
const crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..');
const {_electron}=require(path.join(root,'frontend/node_modules/playwright'));
const out=path.resolve(process.argv[2]);
(async()=>{
  if(await fetch('http://127.0.0.1:18923/api/system').then(()=>true).catch(()=>false)) throw new Error('User backend running; refusing to attach');
  assert(!fs.existsSync(out),'Fresh evidence directory required');fs.mkdirSync(out,{recursive:true});
  let app;
  const result={passed:false,actualElectron:true,networkMocked:true,installerLaunched:false};
  try{
    app=await _electron.launch({executablePath:path.join(root,'frontend/node_modules/electron/dist/electron.exe'),cwd:path.join(root,'frontend'),args:['.',`--user-data-dir=${path.join(out,'profile')}`],env:{...process.env,E2E_TEST:'1',DLSS_LOCAL_TOKEN:crypto.randomUUID()}});
    const page=await app.firstWindow();const errors=[];page.on('pageerror',e=>errors.push(String(e)));
    await page.getByRole('button',{name:'队列空闲',exact:true}).waitFor({timeout:60000});
    const version='9.0.0';
    const assets=[{name:`Lunarc-${version}-x64-Installer.zip`,data:'test-zip'},{name:`lunarc-workbench-${version}-x64.nsis.7z.001`,data:'test-first-part'},{name:`lunarc-workbench-${version}-x64.nsis.7z.002`,data:'test-second-part'}].map(a=>({...a,size:Buffer.byteLength(a.data),state:'uploaded',digest:'sha256:'+crypto.createHash('sha256').update(a.data).digest('hex'),browser_download_url:`https://github.com/Yukikaze1945/Lunarc/releases/download/v${version}/${a.name}`}));
    const data=assets.map(a=>a.digest.slice(7)+'  '+a.name).join('\n')+'\n';
    assets.push({name:'SHA256SUMS.txt',data,size:Buffer.byteLength(data),state:'uploaded',digest:'sha256:'+crypto.createHash('sha256').update(data).digest('hex'),browser_download_url:`https://github.com/Yukikaze1945/Lunarc/releases/download/v${version}/SHA256SUMS.txt`});
    const release={tag_name:`v${version}`,prerelease:true,draft:false,published_at:'2026-10-12',body:'UI fixture, not an actual release',assets};
    await app.evaluate(({net},release)=>{
      const original=net.fetch;
      globalThis.updateTestRestore=()=>net.fetch=original;
      net.fetch=async url=>{
        if(String(url).startsWith('https://api.github.com/repos/Yukikaze1945/Lunarc/releases'))return new Response(JSON.stringify([release]),{headers:{'Content-Type':'application/json'}});
        const asset=release.assets.find(a=>a.browser_download_url===String(url));
        if(asset)return new Response(asset.data);
        return original(url);
      };
    },release);
    await page.getByRole('navigation',{name:'主导航'}).getByRole('button',{name:'关于 Lunarc',exact:true}).click();
    await page.getByRole('heading',{name:'软件更新',exact:true}).waitFor();
    assert.match(await page.locator('.app-update-panel').innerText(),/2\.1\.0 · 源码运行/);
    await page.getByRole('button',{name:'检查更新',exact:true}).click();
    await page.getByText('可用版本 9.0.0 · 预发布',{exact:true}).waitFor();
    await page.getByRole('button',{name:/下载更新/}).click();
    await page.getByRole('button',{name:'打开安装目录',exact:true}).waitFor();
    const state=await page.evaluate(()=>window.electronAPI.getAppUpdate());
    assert.equal(state.phase,'ready');assert.equal(state.received,state.total);
    for(const a of assets)assert.equal(fs.readFileSync(path.join(state.directory,a.name),'utf8'),a.data);
    await page.getByRole('checkbox',{name:'包含预发布版本',exact:true}).uncheck();
    await page.getByRole('button',{name:'检查更新',exact:true}).click();
    await page.getByText('未发现更新的完整安装包',{exact:true}).waitFor();
    await page.screenshot({path:path.join(out,'updates.png')});
    assert.deepEqual(errors,[]);
    Object.assign(result,{passed:true,downloadedAssets:assets.length,sha256Verified:true,previewChannelPassed:true,errors});
  }catch(e){result.error=String(e);throw e}
  finally{fs.writeFileSync(path.join(out,'result.json'),JSON.stringify(result,null,2));if(app)await app.close();}
})().catch(e=>{console.error(e);process.exitCode=1});

