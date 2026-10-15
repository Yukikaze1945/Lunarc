import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import {createHash} from 'node:crypto'
import {ComponentManager,raftWheelLock} from '../../frontend/electron/componentManager.ts'

function fixture(){
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'lunarc-components-'))
  const resources=path.join(root,'resources');fs.mkdirSync(path.join(resources,'components'),{recursive:true})
  for(const name of ['base.lock','image-models.lock','raft.lock','raft-model.json','raft-wheel.json'])fs.copyFileSync(path.resolve('packaging/components',name),path.join(resources,'components',name))
  const m=new ComponentManager(resources,path.join(root,'data'),()=>{})
  return {m,root,resources,close:()=>fs.rmSync(root,{recursive:true,force:true})}
}

test('verified local Torch requirement pins only its exact platform digest and preserves other requirements',()=>{
  const locked=fs.readFileSync('packaging/components/raft.lock','utf8')
  const wheel=JSON.parse(fs.readFileSync('packaging/components/raft-wheel.json','utf8'))
  const result=raftWheelLock(locked,path.resolve(wheel.name),wheel.sha256)
  const torch=result.match(/^torch @[^\n]+\n[^\n]+/m)![0]
  assert(torch.includes('%2Bcu128'))
  assert.equal((torch.match(/--hash=/g)||[]).length,1)
  assert(torch.includes(wheel.sha256))
  assert(result.includes(locked.slice(locked.indexOf('torchvision=='))))
  assert.throws(()=>raftWheelLock(locked,path.resolve(wheel.name),'0'.repeat(64)),/不匹配/)
})
test('missing base is independent of optional RAFT and unknown IDs are refused',async()=>{
  const f=fixture();try{
    assert(f.m.state.requiresSetup)
    assert(f.m.state.components.every(c=>!c.installed))
    await assert.rejects(f.m.install('../evil'),/未知/)
    await f.m.install('raft');assert.match(f.m.state.error!,/先安装基础/)
    assert(!fs.existsSync(path.join(f.root,'data/raft-ready.json')))
  }finally{f.close()}
})

test('image inference extension requires the shared AI environment',async()=>{
  const f=fixture();try{
    assert(f.m.state.components.some(c=>c.id==='image-models'&&!c.installed))
    await f.m.install('image-models')
    assert.match(f.m.state.error!,/AI 模型运行环境/)
    assert(!fs.existsSync(path.join(f.root,'data/image-models-ready.json')))
  }finally{f.close()}
})

test('CLI-only FFmpeg cannot satisfy native video processing component',()=>{
  const f=fixture();try{
    const data=path.join(f.root,'data')
    const tools=path.join(f.m.runtime(),'tools');fs.mkdirSync(tools,{recursive:true})
    const identity='cba748035c21ce1431d0823c7a3a711f38616f89f87a265dceddf9b7f6749d2d'
    fs.writeFileSync(path.join(data,'ffmpeg-ready.json'),JSON.stringify({identity}))
    for(const n of ['ffmpeg.exe','ffprobe.exe'])fs.writeFileSync(path.join(tools,n),'fixture')
    f.m.refresh();assert(!f.m.state.components.find(c=>c.id==='ffmpeg')!.installed)
    for(const n of ['avcodec-62.dll','avformat-62.dll','avfilter-11.dll','avutil-60.dll','swresample-6.dll','swscale-9.dll'])fs.writeFileSync(path.join(tools,n),'fixture')
    f.m.refresh();assert(f.m.state.components.find(c=>c.id==='ffmpeg')!.installed)
  }finally{f.close()}
})
test('corrupt install tool never executes and failed install remains retryable',async()=>{
  const f=fixture();try{
    fs.mkdirSync(path.join(f.resources,'component-tools'))
    fs.writeFileSync(path.join(f.resources,'component-tools/uv.exe'),'bad')
    fs.writeFileSync(path.join(f.resources,'component-tools/manifest.json'),JSON.stringify({files:{'uv.exe':{bytes:3,sha256:'0'.repeat(64)}}}))
    let executions=0;(f.m as any).command=async()=>{executions++}
    await f.m.install('base');assert.match(f.m.state.error!,/工具损坏/);assert.equal(executions,0)
    assert.equal(f.m.state.busy,null);assert(!fs.existsSync(path.join(f.root,'data/base-ready.json')))
    await f.m.install('base');assert.match(f.m.state.error!,/工具损坏/)
  }finally{f.close()}
})
test('one install at a time, cancellation cannot create a ready marker',async()=>{
  const f=fixture();try{
    let release:()=>void=()=>{};let calls=0
    ;(f.m as any).uv=async()=> 'not-executed'
    ;(f.m as any).command=async()=>{calls++;if(calls===1)await new Promise<void>(r=>release=r);if((f.m as any).cancelled)throw Error('paused')}
    const running=f.m.install('base');await new Promise(r=>setImmediate(r))
    const other=await f.m.install('raft');assert.equal(other.busy,'base');assert.equal(calls,1)
    f.m.cancel();release();await running
    assert.equal(f.m.state.busy,null);assert.match(f.m.state.error!,/暂停/)
    assert(!fs.existsSync(path.join(f.root,'data/base-ready.json')))
  }finally{f.close()}
})
test('changed locks invalidate completion while unrelated RAFT absence does not invalidate base',()=>{
  const f=fixture();try{
    const python=f.m.python();fs.mkdirSync(path.dirname(python),{recursive:true});fs.writeFileSync(python,'fixture')
    const identity=createHash('sha256').update('3.10.18').update(fs.readFileSync(path.join(f.resources,'components/base.lock'))).digest('hex')
    fs.writeFileSync(path.join(f.root,'data/base-ready.json'),JSON.stringify({identity}))
    f.m.refresh();assert(f.m.state.components.find(c=>c.id==='base')!.installed)
    const lock=path.join(f.resources,'components/base.lock')
    fs.writeFileSync(lock,fs.readFileSync(lock,'utf8').replace(/\r?\n/g,'\r\n'))
    f.m.refresh();assert(f.m.state.components.find(c=>c.id==='base')!.installed)
    fs.appendFileSync(path.join(f.resources,'components/base.lock'),'\nchanged')
    f.m.refresh();assert(!f.m.state.components.find(c=>c.id==='base')!.installed)
  }finally{f.close()}
})
