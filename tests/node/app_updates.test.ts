import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import {createHash} from 'node:crypto'
import {AppUpdates, compareVersions, parseRelease} from '../../frontend/electron/appUpdates.ts'

const sha=(data:string)=>createHash('sha256').update(data).digest('hex')
function smallFixture(bad=false){
  const v='2.1.2',exe={name:`Lunarc-${v}-x64-Setup.exe`,data:'small-installer'}
  const info={name:'INSTALLER.json',data:JSON.stringify({schema:1,mode:'components',version:v,installer:{name:exe.name,bytes:exe.data.length,sha256:bad?'0'.repeat(64):sha(exe.data)}})}
  const files=[exe,info];const ledger={name:'SHA256SUMS.txt',data:files.map(a=>`${sha(a.data)}  ${a.name}`).join('\n')+'\n'}
  return {tag_name:`v${v}`,draft:false,prerelease:true,published_at:'2026-10-12',assets:[...files,ledger].map(a=>({...a,size:Buffer.byteLength(a.data),state:'uploaded',digest:'sha256:'+sha(a.data),browser_download_url:`https://github.com/Yukikaze1945/Lunarc/releases/download/v${v}/${a.name}`}))}
}

test('small standalone setup requires a verified component format declaration',async()=>{
  for(const bad of [false,true]){
    const r=smallFixture(bad),root=await fs.mkdtemp(path.join(os.tmpdir(),'lunarc-small-update-'))
    try{
      const updater=new AppUpdates('2.1.0',true,root,()=>{},async url=>String(url).startsWith('https://api.')?Response.json([r]):new Response(r.assets.find(a=>a.browser_download_url===String(url))!.data))
      await updater.check(true)
      assert.equal(updater.state.phase,bad?'error':'available')
      if(!bad){await updater.download();assert.equal(updater.state.phase,'ready');assert.equal(updater.state.release!.assets.length,3)}
      const missing={...r,assets:r.assets.filter(a=>a.name!=='INSTALLER.json')};assert.equal(parseRelease(missing),null)
    }finally{await fs.rm(root,{recursive:true,force:true})}
  }
})
const fixture=(v='2.1.1')=>{const r=({tag_name:`v${v}`,draft:false,prerelease:true,published_at:'2026-10-11',body:'notes',assets:[
  {name:`Lunarc-${v}-x64-Installer.zip`,data:'zip'},
  {name:`lunarc-workbench-${v}-x64.nsis.7z.001`,data:'abcdefgh'},
  {name:`lunarc-workbench-${v}-x64.nsis.7z.002`,data:'ijkl'},
].map(a=>({...a,size:a.data.length,state:'uploaded',digest:`sha256:${sha(a.data)}`,browser_download_url:`https://github.com/Yukikaze1945/Lunarc/releases/download/v${v}/${a.name}`}))});const data=r.assets.map(a=>`${a.digest.slice(7)}  ${a.name}`).join('\n')+'\n';r.assets.push({name:'SHA256SUMS.txt',data,size:data.length,state:'uploaded',digest:'sha256:'+sha(data),browser_download_url:`https://github.com/Yukikaze1945/Lunarc/releases/download/v${v}/SHA256SUMS.txt`});return r}

test('version comparison handles numeric components and prerelease order',()=>{
  assert.equal(compareVersions('2.10.0','2.9.0'),1)
  assert.equal(compareVersions('2.1.1','2.1.1-beta.2'),1)
  assert.equal(compareVersions('2.1.1-beta.10','2.1.1-beta.2'),1)
  assert.equal(compareVersions('2.1.1-beta.2','2.1.1-beta.a'),-1)
  assert.equal(compareVersions('2.1.1+build','v2.1.1'),0)
  assert.throws(()=>compareVersions('../../evil','2.1.0'))
  assert.throws(()=>compareVersions('2.1.0-beta..1','2.1.0'))
  assert.throws(()=>compareVersions('2.1.0-beta.01','2.1.0'))
})
test('only complete, uploaded, hashed installer sets from the exact official repo are accepted',()=>{
  assert.equal(parseRelease(fixture())?.assets.length,4)
  for(const mutate of [
    (r:any)=>r.draft=true,
    (r:any)=>r.assets[0].browser_download_url='https://evil.invalid/installer',
    (r:any)=>r.assets[0].digest=null,
    (r:any)=>r.assets[0].size=-1,
    (r:any)=>r.assets[0].state='new',
    (r:any)=>r.assets[1].name=r.assets[1].name.replace('.001','.003'),
    (r:any)=>r.assets=r.assets.slice(0,1),
    (r:any)=>r.assets.push(r.assets[0]),
  ]) {const r=fixture();mutate(r);assert.equal(parseRelease(r),null)}
})
test('checks choose newer complete releases by version, respect preview channel, and report rate limit',async()=>{
  const r=fixture(), newer=fixture('2.2.0'); newer.prerelease=false
  const updater=new AppUpdates('2.1.0',true,'unused',()=>{},async url=>String(url).startsWith('https://api.')?Response.json([r,newer,fixture('2.0.0')]):new Response(newer.assets.at(-1)!.data))
  await updater.check(false);assert.equal(updater.state.release?.version,'2.2.0')
  const preview=new AppUpdates('2.1.0',true,'unused',()=>{},async url=>String(url).startsWith('https://api.')?Response.json([r]):new Response(r.assets.at(-1)!.data))
  await preview.check(false);assert.equal(preview.state.phase,'current')
  await preview.check(true);assert.equal(preview.state.phase,'available')
  const rate=new AppUpdates('2.1.0',true,'unused',()=>{},async()=>new Response('',{status:403}))
  await rate.check(true);assert.equal(rate.state.phase,'error');assert.match(rate.state.error!,/限制/)
})
test('the checksum ledger catches a missing final volume and altered ledger bytes',async()=>{
  for(const badLedger of [false,true]) {
    const r=fixture();const ledger=r.assets.at(-1)!.data
    if(!badLedger)r.assets.splice(2,1) // .001 alone looks contiguous but .002 is declared.
    const updater=new AppUpdates('2.1.0',true,'unused',()=>{},async url=>String(url).startsWith('https://api.')?Response.json([r]):new Response(badLedger?'wrong':ledger))
    await updater.check(true);assert.equal(updater.state.phase,'error')
    assert.match(updater.state.error!,badLedger?/清单无效/:/尚未上传完整/)
  }
})
test('downloads every volume, verifies SHA, reuses only verified files and resumes precise byte ranges',async()=>{
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'lunarc-update-'))
  const r=fixture(), ranges:string[]=[]
  let requests=0
  const request:typeof fetch=async(url,init)=>{
    if(String(url).startsWith('https://api.')) return Response.json([r])
    requests++
    const asset=r.assets.find(a=>String(url)===a.browser_download_url)!
    const range=new Headers(init?.headers).get('Range')
    if(range) {ranges.push(range);const offset=Number(/bytes=(\d+)-/.exec(range)![1]);return new Response(asset.data.slice(offset),{status:206,headers:{'Content-Range':`bytes ${offset}-${asset.size-1}/${asset.size}`}})}
    return new Response(asset.data)
  }
  try {
    const updater=new AppUpdates('2.1.0',true,root,()=>{},request)
    await updater.check(true)
    const dir=path.join(root,'2.1.1');await fs.mkdir(dir)
    await fs.writeFile(path.join(dir,r.assets[0].name),'bad') // Same length is not sufficient.
    await fs.writeFile(path.join(dir,r.assets[1].name+'.part'),'abc')
    await updater.download();assert.equal(updater.state.phase,'ready');assert.deepEqual(ranges,['bytes=3-'])
    assert.equal(updater.state.received,r.assets.reduce((n,a)=>n+a.size,0))
    for(const a of r.assets) assert.equal(await fs.readFile(path.join(dir,a.name),'utf8'),a.data)
    const count=requests;await updater.download();assert.equal(requests,count)
  } finally {await fs.rm(root,{recursive:true,force:true})}
})
test('corrupt data cannot become ready; a bad partial is discarded then retried',async()=>{
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'lunarc-update-'))
  const r=fixture();let corrupt=true
  try {
    const updater=new AppUpdates('2.1.0',true,root,()=>{},async(url)=>{
      if(String(url).startsWith('https://api.'))return Response.json([r])
      const a=r.assets.find(x=>x.browser_download_url===String(url))!
      return new Response(corrupt&&a.name!=='SHA256SUMS.txt'?'x'.repeat(a.size):a.data)
    })
    await updater.check(true);await updater.download();assert.equal(updater.state.phase,'error')
    assert.match(updater.state.error!,/校验失败/);assert.equal(updater.state.directory,null)
    await assert.rejects(fs.stat(path.join(root,'2.1.1',r.assets[0].name+'.part')))
    corrupt=false;await updater.download();assert.equal(updater.state.phase,'ready')
  } finally {await fs.rm(root,{recursive:true,force:true})}
})
test('pause aborts a pending response, preserves partial and rejects concurrent check/download',async()=>{
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'lunarc-update-'))
  const r=fixture();let pending!:()=>void, entered=new Promise<void>(resolve=>pending=resolve)
  try {
    const updater=new AppUpdates('2.1.0',true,root,()=>{},async(url,init)=>{
      if(String(url).startsWith('https://api.'))return Response.json([r])
      if(String(url).endsWith('/SHA256SUMS.txt')) return new Response(r.assets.at(-1)!.data)
      pending()
      return new Promise((_resolve,reject)=>init!.signal!.addEventListener('abort',()=>reject(new Error('abort')),{once:true}))
    })
    await updater.check(true)
    const download=updater.download();await entered
    await updater.check(false);assert.equal(updater.state.phase,'downloading')
    await updater.download();assert.equal(updater.state.phase,'downloading')
    updater.cancel();await download;assert.equal(updater.state.phase,'paused')
  } finally {await fs.rm(root,{recursive:true,force:true})}
})
test('resume rejects a mismatched server range and truncated response',async()=>{
  for(const mode of ['range','truncated']) {
    const root=await fs.mkdtemp(path.join(os.tmpdir(),'lunarc-update-'));const r=fixture()
    try {
      const updater=new AppUpdates('2.1.0',true,root,()=>{},async(url)=>String(url).startsWith('https://api.')?Response.json([r]):String(url).endsWith('/SHA256SUMS.txt')?new Response(r.assets.at(-1)!.data):mode==='range'?new Response('ip',{status:206,headers:{'Content-Range':'bytes 0-1/3'}}):new Response('zi'))
      await updater.check(true);await fs.mkdir(path.join(root,'2.1.1'));await fs.writeFile(path.join(root,'2.1.1',r.assets[0].name+'.part'),'z')
      await updater.download();assert.equal(updater.state.phase,'error');assert.equal(updater.state.directory,null)
      assert.match(updater.state.error!,mode==='range'?/范围/:/中断/)
    } finally {await fs.rm(root,{recursive:true,force:true})}
  }
})
