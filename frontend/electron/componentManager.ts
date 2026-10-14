import fs from 'node:fs'
import path from 'node:path'
import {createHash} from 'node:crypto'
import {spawn, type ChildProcess} from 'node:child_process'
import {pathToFileURL} from 'node:url'
import {AppUpdates} from './appUpdates.ts'
import type {ComponentId, ComponentState} from '../src/types/components.ts'

const PYTHON='3.10.18'
const ffmpeg={name:'ffmpeg-8.1.2-full_build-shared.7z',size:59459100,sha256:'cba748035c21ce1431d0823c7a3a711f38616f89f87a265dceddf9b7f6749d2d',url:'https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-8.1.2-full_build-shared.7z'}
const raftModel={name:'raft_large_C_T_SKHT_V2-ff5fadd5.pth',size:21100755,sha256:'',url:'https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth'}
export function raftWheelLock(locked:string,file:string,sha256:string) {
  const block=/^torch==2\.10\.0\+cu128[ \t]+\\\r?\n(?:[ \t]+--hash=sha256:[a-f0-9]{64}(?:[ \t]+\\)?\r?\n)+/m
  const original=locked.match(block)?.[0]
  if(!original?.includes(`--hash=sha256:${sha256}`))throw new Error('RAFT 主包与依赖清单不匹配')
  // A local platform wheel has exactly one allowed digest. Keep all other
  // requirements intact; do not feed uv hashes for unrelated platform wheels.
  const url=pathToFileURL(file).href.replace(/\+/g,'%2B')
  return locked.replace(block,`torch @ ${url} \\\n    --hash=sha256:${sha256}\n`)
}
export class ComponentManager {
  private root:string
  private resources:string
  private notify:(state:ComponentState)=>void
  private child:ChildProcess|null=null
  private cancelled=false
  private downloader:AppUpdates|null=null
  state:ComponentState
  constructor(resources:string, directory:string, notify:(state:ComponentState)=>void, privateRequest:typeof fetch=fetch) {
    this.resources=resources;this.root=directory;this.notify=notify;this.request=privateRequest
    this.state={components:[],busy:null,phase:'',received:0,total:0,error:null,requiresSetup:true,directory}
    this.refresh()
  }
  private request:typeof fetch
  python() {return path.join(this.root,'base-v1','Scripts','python.exe')}
  runtime() {return path.join(this.root,'runtime')}
  raftSite() {return path.join(this.root,'raft-v1','site')}
  private lock(id:'base'|'raft') {return path.join(this.resources,'components',`${id}.lock`)}
  private identity(id:ComponentId) {
    if(id==='ffmpeg')return ffmpeg.sha256
    const hash=createHash('sha256').update(PYTHON).update(fs.readFileSync(this.lock(id),'utf8').replace(/\r\n/g,'\n'))
    if(id==='raft')hash.update(fs.readFileSync(path.join(this.resources,'components','raft-model.json')))
    return hash.digest('hex')
  }
  private marker(id:ComponentId) {return path.join(this.root,`${id}-ready.json`)}
  private ready(id:ComponentId) {
    try {
      const value=JSON.parse(fs.readFileSync(this.marker(id),'utf8'))
      if(value.identity!==this.identity(id)) return false
      if(id==='base') return fs.existsSync(this.python())
      if(id==='ffmpeg') return ['ffmpeg.exe','ffprobe.exe','avcodec-62.dll','avformat-62.dll','avfilter-11.dll','avutil-60.dll','swresample-6.dll','swscale-9.dll'].every(n=>fs.existsSync(path.join(this.runtime(),'tools',n)))
      return fs.existsSync(path.join(this.raftSite(),'torch','__init__.py')) && fs.existsSync(path.join(this.runtime(),'torch_home','hub','checkpoints',raftModel.name))
    } catch {return false}
  }
  refresh() {
    const base=this.ready('base'), ff=this.ready('ffmpeg'), raft=this.ready('raft')
    this.state={...this.state,requiresSetup:!base||!ff,components:[
      {id:'base',label:'基础运行环境',installed:base,version:PYTHON,detail:'Python 与后端依赖，用于素材读取、预览和导出管理。'},
      {id:'ffmpeg',label:'视频处理组件',installed:ff,version:'8.1.2',detail:'FFmpeg 与编码运行库，用于读取素材、视频对比和导出。'},
      {id:'raft',label:'RAFT 光流',installed:raft,version:'Torch 2.10.0 / CUDA 12.8',detail:'可选。使用 RAFT 光流时需要，下载量较大；原生 NVIDIA / AMD 光流不需要。'},
    ]};this.notify(this.state);return this.state
  }
  private set(patch:Partial<ComponentState>) {this.state={...this.state,...patch};this.notify(this.state)}
  private environment() {
    const env:NodeJS.ProcessEnv={}
    for(const key of ['SystemRoot','WINDIR','TEMP','TMP','USERPROFILE','LOCALAPPDATA','APPDATA','HTTP_PROXY','HTTPS_PROXY','NO_PROXY']) if(process.env[key])env[key]=process.env[key]
    env.PATH=path.join(process.env.SystemRoot||'C:\\Windows','System32')
    return {...env,UV_CACHE_DIR:path.join(this.root,'cache'),UV_PYTHON_INSTALL_DIR:path.join(this.root,'python'),UV_PYTHON_BIN_DIR:path.join(this.root,'python-bin'),UV_HTTP_RETRIES:'3',UV_HTTP_TIMEOUT:'60',UV_PYTHON_PREFERENCE:'only-managed',UV_NO_PROGRESS:'1',PYTHONNOUSERSITE:'1',PYTHONUTF8:'1'}
  }
  private async command(executable:string,args:string[],timeout=1200000) {
    if(this.cancelled)throw new Error('安装已暂停')
    await fs.promises.mkdir(this.root,{recursive:true})
    return new Promise<string>((resolve,reject)=>{
      const child=spawn(executable,args,{cwd:this.root,env:this.environment(),windowsHide:true,stdio:['ignore','pipe','pipe']});this.child=child
      let output='',settled=false
      const timer=setTimeout(()=>{child.kill();finish(new Error('安装超时，可重试'))},timeout)
      const finish=(error?:Error)=>{if(settled)return;settled=true;clearTimeout(timer);if(this.child===child)this.child=null;error?reject(error):resolve(output)}
      const collect=(chunk:Buffer)=>{output=(output+chunk.toString()).slice(-6000)}
      child.stdout?.on('data',collect);child.stderr?.on('data',collect)
      child.on('error',finish);child.on('close',code=>finish(this.cancelled?new Error('安装已暂停'):code?new Error(`组件安装失败，可重试。\n${output}`):undefined))
    })
  }
  private async uv() {
    const directory=path.join(this.resources,'component-tools')
    const manifest=JSON.parse(await fs.promises.readFile(path.join(directory,'manifest.json'),'utf8'))
    const file=path.join(directory,'uv.exe'),entry=manifest.files?.['uv.exe']
    if(!entry || (await fs.promises.stat(file)).size!==entry.bytes || createHash('sha256').update(await fs.promises.readFile(file)).digest('hex')!==entry.sha256) throw new Error('组件安装工具损坏，请重新安装软件')
    return file
  }
  private async download(asset:typeof ffmpeg, version:string) {
    const downloader=new AppUpdates('0.0.0',true,path.join(this.root,'downloads'),s=>this.set({received:s.received,total:s.total}),this.request)
    this.downloader=downloader
    downloader.state.release={version,preview:false,url:'',notes:'',assets:[asset]};downloader.state.total=asset.size
    await downloader.download();this.downloader=null
    if(downloader.state.phase!=='ready')throw new Error(downloader.state.error||'下载未完成')
    return path.join(downloader.state.directory!,asset.name)
  }
  async install(id:unknown) {
    if(!['base','ffmpeg','raft'].includes(String(id)))throw new Error('未知组件')
    if(this.state.busy)return this.state
    const key=id as ComponentId
    if(this.ready(key))return this.refresh()
    this.cancelled=false;this.set({busy:key,error:null,received:0,total:0,phase:'正在准备安装'})
    try {
      if(key==='base') {
        const uv=await this.uv();this.set({phase:'正在下载 Python'})
        await this.command(uv,['--no-config','python','install',PYTHON])
        this.set({phase:'正在创建独立运行环境'})
        if(!fs.existsSync(this.python()))await this.command(uv,['--no-config','venv',path.dirname(path.dirname(this.python())),'--python',PYTHON])
        this.set({phase:'正在安装后端依赖'})
        await this.command(uv,['--no-config','pip','sync','--python',this.python(),'--require-hashes','--only-binary',':all:','--index-url','https://pypi.org/simple',this.lock('base')])
        this.set({phase:'正在检查运行环境'})
        await this.command(this.python(),['-I','-c','import numpy,cv2,fastapi,uvicorn,websockets,pydantic,av,PIL,ssl,sqlite3; assert __import__("importlib.util",fromlist=["find_spec"]).find_spec("torch") is None; print("base-ready")'],60000)
      } else if(key==='ffmpeg') {
        if(!this.ready('base'))throw new Error('请先安装基础运行环境')
        this.set({phase:'正在下载视频处理组件'})
        const archive=await this.download(ffmpeg,'ffmpeg-8.1.2')
        this.set({phase:'正在解压并检查视频处理组件'})
        await this.command(this.python(),['-I',path.join(this.resources,'component_install.py'),'ffmpeg',archive,this.runtime()],120000)
      } else {
        if(!this.ready('base'))throw new Error('请先安装基础运行环境')
        const uv=await this.uv();this.set({phase:'正在下载 PyTorch / CUDA，支持暂停后继续下载'})
        const wheel=JSON.parse(await fs.promises.readFile(path.join(this.resources,'components','raft-wheel.json'),'utf8'))
        const locked=await fs.promises.readFile(this.lock('raft'),'utf8')
        if(wheel.name!=='torch-2.10.0+cu128-cp310-cp310-win_amd64.whl' ||
          wheel.url!=='https://download-r2.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp310-cp310-win_amd64.whl' ||
          !/^[a-f0-9]{64}$/.test(wheel.sha256) || !locked.includes(`--hash=sha256:${wheel.sha256}`) || !Number.isSafeInteger(wheel.size))throw new Error('RAFT 下载清单无效，请重新安装软件')
        const wheelFile=await this.download(wheel,'raft-wheel-v1')
        const localLock=path.join(this.root,'raft-install.lock')
        await fs.promises.writeFile(localLock,raftWheelLock(locked,wheelFile,wheel.sha256))
        this.set({phase:'正在安装 RAFT 光流依赖',received:0,total:0})
        await this.command(uv,['--no-config','pip','install','--python',this.python(),'--target',this.raftSite(),'--require-hashes','--only-binary',':all:','--index-url','https://pypi.org/simple','--index','https://download.pytorch.org/whl/cu128','--index-strategy','unsafe-best-match','-r',localLock])
        this.set({phase:'正在下载 RAFT 模型'})
        const config=JSON.parse(await fs.promises.readFile(path.join(this.resources,'components','raft-model.json'),'utf8'))
        const archive=await this.download({...raftModel,sha256:config.sha256,size:config.bytes},'raft-model-v1')
        const target=path.join(this.runtime(),'torch_home','hub','checkpoints',raftModel.name)
        await fs.promises.mkdir(path.dirname(target),{recursive:true});await fs.promises.copyFile(archive,target)
        this.set({phase:'正在检查 RAFT 与 CUDA'})
        await this.command(this.python(),['-I',path.join(this.resources,'component_install.py'),'raft',this.raftSite(),target],120000)
      }
      if(this.cancelled)throw new Error('安装已暂停')
      const marker=this.marker(key)
      await fs.promises.writeFile(marker+'.tmp',JSON.stringify({schema:1,identity:this.identity(key),installedAt:new Date().toISOString()}))
      await fs.promises.rename(marker+'.tmp',marker)
      if(this.readyForBackend()) {
        await fs.promises.mkdir(this.runtime(),{recursive:true})
        await fs.promises.writeFile(path.join(this.runtime(),'manifest.json'),JSON.stringify({schema:1,mode:'components',python:PYTHON,base:this.identity('base'),ffmpeg:this.identity('ffmpeg')}))
      }
      this.set({phase:key==='raft'?'RAFT 已安装，重启软件后生效':'组件已安装'})
    } catch(e) {this.set({error:this.cancelled?'安装已暂停，可继续安装。已完成的下载会复用。':(e as Error).message,phase:this.cancelled?'已暂停':'安装失败'})}
    finally {this.child=null;this.downloader=null;this.set({busy:null});this.refresh()}
    return this.state
  }
  cancel() {this.cancelled=true;this.downloader?.cancel();this.child?.kill()}
  readyForBackend() {return this.ready('base')&&this.ready('ffmpeg')}
  aiReady() {return this.ready('raft')}
}
