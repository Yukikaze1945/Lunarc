import fs from 'node:fs'
import path from 'node:path'
import {createHash} from 'node:crypto'

const repository = 'https://github.com/Yukikaze1945/Lunarc'
import type {UpdateAsset, UpdateRelease, UpdateState} from '../src/types/appUpdates.ts'

function version(value: string) {
  const m = /^v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$/.exec(value)
  if (!m || m.slice(1,4).some(x => !Number.isSafeInteger(Number(x)))) throw new Error('版本号无效')
  if (m[4]?.split('.').some(x=>!/^[0-9A-Za-z-]+$/.test(x)||/^0\d+$/.test(x))) throw new Error('预发布版本号无效')
  return {numbers: m.slice(1,4).map(Number), pre: m[4]?.split('.') || []}
}
export function compareVersions(a: string, b: string): number {
  const x = version(a), y = version(b)
  for (let i=0; i<3; i++) if (x.numbers[i] !== y.numbers[i]) return Math.sign(x.numbers[i]-y.numbers[i])
  if (!x.pre.length || !y.pre.length) return x.pre.length === y.pre.length ? 0 : x.pre.length ? -1 : 1
  for (let i=0; i<Math.max(x.pre.length,y.pre.length); i++) {
    const l=x.pre[i], r=y.pre[i]
    if (l === r) continue
    if (l === undefined || r === undefined) return l === undefined ? -1 : 1
    const ln=/^\d+$/.test(l), rn=/^\d+$/.test(r)
    if (ln && rn) return BigInt(l)<BigInt(r) ? -1 : 1
    if (ln !== rn) return ln ? -1 : 1
    return l<r ? -1 : 1
  }
  return 0
}

export function parseRelease(raw: any): UpdateRelease|null {
  try {
    if (raw.draft || !raw.published_at || !Array.isArray(raw.assets)) return null
    const v=String(raw.tag_name).replace(/^v/, ''); version(v)
    const tag=encodeURIComponent(raw.tag_name)
    const assets: UpdateAsset[]=[]
    for (const a of raw.assets) {
      if (!/^(?:Lunarc-[0-9A-Za-z.-]+-x64-(?:Installer\.zip|Setup\.exe)|lunarc-workbench-[0-9A-Za-z.-]+-x64\.nsis\.7z(?:\.\d{3})?|SHA256SUMS\.txt|INSTALL\.md)$/.test(a.name)) continue
      if (assets.some(x=>x.name===a.name) || a.state !== 'uploaded' || !Number.isSafeInteger(a.size) || a.size<=0 || !/^sha256:[a-f0-9]{64}$/.test(a.digest)) return null
      const url=`${repository}/releases/download/${tag}/${encodeURIComponent(a.name)}`
      if (a.browser_download_url !== url) return null
      assets.push({name:a.name, size:a.size, url, sha256:a.digest.slice(7)})
    }
    const zip=assets.find(a=>a.name===`Lunarc-${v}-x64-Installer.zip`)
    const parts=assets.filter(a=>a.name.startsWith(`lunarc-workbench-${v}-x64.nsis.7z.`)).sort((a,b)=>a.name.localeCompare(b.name))
    const exe=assets.find(a=>a.name===`Lunarc-${v}-x64-Setup.exe`)
    const payload=assets.find(a=>a.name===`lunarc-workbench-${v}-x64.nsis.7z`)
    let selected:UpdateAsset[]
    if (zip && parts.length && parts.every((a,i)=>a.name.endsWith(`.${String(i+1).padStart(3,'0')}`))) selected=[zip,...parts]
    else if (exe && payload) selected=[exe,payload]
    else return null // A bootstrap exe alone is not a complete installer.
    if (!assets.some(a=>a.name==='SHA256SUMS.txt' && a.size<65536)) return null
    selected.push(...assets.filter(a=>a.name==='SHA256SUMS.txt'||a.name==='INSTALL.md'))
    return {version:v,preview:!!raw.prerelease,url:`${repository}/releases/tag/${tag}`,notes:typeof raw.body==='string'?raw.body.slice(0,20000):'',assets:selected}
  } catch {return null}
}

async function digest(file:string) {
  const hash=createHash('sha256')
  for await (const chunk of fs.createReadStream(file)) hash.update(chunk)
  return hash.digest('hex')
}

export class AppUpdates {
  state:UpdateState
  private request:typeof fetch
  private cache:string
  private notify:(state:UpdateState)=>void
  private controller:AbortController|null=null
  constructor(currentVersion:string, packaged:boolean, cache:string, notify:(state:UpdateState)=>void, request:typeof fetch=fetch) {
    this.cache=cache; this.notify=notify; this.request=request
    this.state={currentVersion,packaged,phase:'idle',release:null,received:0,total:0,directory:null,error:null}
  }
  private set(patch:Partial<UpdateState>) {this.state={...this.state,...patch}; this.notify(this.state)}
  async check(includePreview:boolean) {
    if (this.controller) return this.state
    const controller=new AbortController(); this.controller=controller
    this.set({phase:'checking',release:null,directory:null,error:null,received:0,total:0})
    const timeout=setTimeout(()=>controller.abort(),20000)
    try {
      const response=await this.request('https://api.github.com/repos/Yukikaze1945/Lunarc/releases?per_page=100', {signal:controller.signal,headers:{Accept:'application/vnd.github+json','User-Agent':'Lunarc-updater'}})
      if (!response.ok) throw new Error(response.status===403?'GitHub 暂时限制请求，请稍后重试':`检查更新失败（HTTP ${response.status}）`)
      const data=await response.json()
      if (!Array.isArray(data)) throw new Error('发布信息无效')
      const releases=data.map(parseRelease).filter((r):r is UpdateRelease=>!!r && (includePreview || !r.preview) && compareVersions(r.version,this.state.currentVersion)>0)
      releases.sort((a,b)=>compareVersions(b.version,a.version))
      const release=releases[0]||null
      if (release) {
        const ledger=release.assets.find(a=>a.name==='SHA256SUMS.txt')!
        const response=await this.request(ledger.url,{signal:controller.signal})
        if (!response.ok) throw new Error('无法获取安装包校验清单，请重试')
        const content=Buffer.from(await response.arrayBuffer())
        if(content.length!==ledger.size || createHash('sha256').update(content).digest('hex')!==ledger.sha256) throw new Error('安装包校验清单无效')
        const rows=new Map<string,string>()
        for(const line of content.toString('utf8').split(/\r?\n/)) {
          if (!line.trim()) continue
          const m=/^([a-f0-9]{64})\s+\*?([A-Za-z0-9._-]+)$/.exec(line)
          if (!m || rows.has(m[2])) throw new Error('安装包校验清单格式无效')
          rows.set(m[2],m[1])
        }
        const names=release.assets.filter(a=>a.name!=='SHA256SUMS.txt')
        if(names.some(a=>rows.get(a.name)!==a.sha256) || [...rows.keys()].some(name=>
          (name===`Lunarc-${release.version}-x64-Installer.zip` || name.startsWith(`lunarc-workbench-${release.version}-x64.nsis.7z`)) && !names.some(a=>a.name===name)))
          throw new Error('安装包尚未上传完整或校验信息不一致，请稍后重试')
      }
      this.set({phase:release?'available':'current',release,total:release?.assets.reduce((n,a)=>n+a.size,0)||0})
    } catch(e) {this.set({phase:'error',error:controller.signal.aborted?'检查更新超时或已取消，请重试':(e as Error).message})}
    finally {clearTimeout(timeout); this.controller=null}
    return this.state
  }
  cancel() {this.controller?.abort(new Error('已暂停，下载进度已保存'))}
  async download() {
    if (this.controller || !this.state.release) return this.state
    const release=this.state.release
    const directory=path.join(this.cache,release.version)
    const controller=new AbortController(); this.controller=controller
    this.set({phase:'downloading',received:0,directory:null,error:null})
    let completed=0
    let lastNotification=0
    try {
      await fs.promises.mkdir(directory,{recursive:true})
      for (const asset of release.assets) {
        const file=path.join(directory,asset.name), partial=file+'.part'
        if (fs.existsSync(file) && (await fs.promises.stat(file)).size===asset.size && await digest(file)===asset.sha256) {
          completed+=asset.size; this.set({received:completed}); continue
        }
        if (controller.signal.aborted) throw new Error('已暂停')
        let offset=fs.existsSync(partial)?(await fs.promises.stat(partial)).size:0
        if (offset>asset.size) {await fs.promises.truncate(partial,0); offset=0}
        let timer:ReturnType<typeof setTimeout>|undefined
        const touch=()=>{clearTimeout(timer); timer=setTimeout(()=>controller.abort(new Error('网络超时，请继续下载')),60000)}
        touch()
        try {
          if (offset<asset.size) {
            const response=await this.request(asset.url,{signal:controller.signal,headers:offset?{Range:`bytes=${offset}-`}:{}})
            if (offset && response.status===206) {
              if (response.headers.get('content-range')!==`bytes ${offset}-${asset.size-1}/${asset.size}`) throw new Error('服务器返回的续传范围不匹配')
            } else if (response.status===200) offset=0
            else throw new Error(`下载失败（HTTP ${response.status}），可以稍后继续`)
            if (!response.body) throw new Error('下载内容为空')
            const handle=await fs.promises.open(partial,offset?'a':'w')
            const reader=response.body.getReader()
            try {
              while (true) {
                const {done,value}=await reader.read(); if(done) break
                touch()
                if (offset+value.byteLength>asset.size) throw new Error('下载大小超过发布声明')
                // FileHandle.write may write fewer bytes than requested.
                let written=0
                while(written<value.byteLength) {const r=await handle.write(value,written,value.byteLength-written); if (!r.bytesWritten) throw new Error('无法写入更新文件'); written+=r.bytesWritten}
                offset+=value.byteLength
                if (Date.now()-lastNotification>150) {this.set({received:completed+offset}); lastNotification=Date.now()}
              }
            } finally {await reader.cancel().catch(()=>{}); await handle.close()}
          }
        } finally {clearTimeout(timer)}
        if (offset!==asset.size) throw new Error('下载中断，已保存进度，可继续下载')
        if (await digest(partial)!==asset.sha256) {
          await fs.promises.unlink(partial)
          throw new Error(`${asset.name} 校验失败，请重新下载`)
        }
        if(controller.signal.aborted) throw new Error('已暂停')
        await fs.promises.rename(partial,file)
        completed+=asset.size; this.set({received:completed})
      }
      if(controller.signal.aborted) throw new Error('已暂停')
      this.set({phase:'ready',directory})
    } catch(e) {this.set({phase:controller.signal.aborted?'paused':'error',error:controller.signal.aborted?String(controller.signal.reason?.message||'已暂停，下载进度已保存'):(e as Error).message})}
    finally {this.controller=null}
    return this.state
  }
}
