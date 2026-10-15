import {useEffect,useState} from 'react'
import {ArrowUpRight,Folder,Check,Image as ImageIcon} from 'lucide-react'
import type {VideoInfo,DLSSSettings} from '../types'
import {exportImage,imageThumbnail,imageExportFormats,openNativeDirDialog} from '../services/api'
import {ImageViewport} from '../components/ImageViewport'
import {renderStages,hasScaleStages,stageName,vsrOutputSize} from '../services/renderPipeline'
import './image-deliver.css'

export function ImageDeliverPage({video,settings,onStatusChange}:{video:VideoInfo;settings:DLSSSettings;onStatusChange?:(message:string)=>void}) {
  const [format,setFormat]=useState('png')
  const [directory,setDirectory]=useState('')
  const [name,setName]=useState('')
  const [quality,setQuality]=useState(95)
  const [alpha,setAlpha]=useState(true)
  const [overwrite,setOverwrite]=useState(false)
  const [busy,setBusy]=useState(false)
  const [error,setError]=useState('')
  const [result,setResult]=useState<any>(null)
  const [thumbnail,setThumbnail]=useState('')
  const [formats,setFormats]=useState<Record<string,boolean>>({png:true,jpeg:true,webp:true,avif:false,heic:false})
  const [previewZoom,setPreviewZoom]=useState(1)
  const [previewPan,setPreviewPan]=useState({x:.5,y:.5})
  useEffect(()=>{let active=true;imageExportFormats().then(value=>{if(active)setFormats(value)}).catch(()=>{});return()=>{active=false}},[])
  useEffect(()=>{
    const slash=Math.max(video.path.lastIndexOf('\\'),video.path.lastIndexOf('/'))
    setDirectory(video.path.slice(0,slash));setName(video.filename.replace(/\.[^.]+$/,'')+'_lunarc')
    setResult(null);setError('')
    setPreviewZoom(1);setPreviewPan({x:.5,y:.5})
  },[video.path,video.filename])
  useEffect(()=>{
    let cancelled=false,url=''
    setThumbnail('');setPreviewZoom(1);setPreviewPan({x:.5,y:.5})
    imageThumbnail(result?.path||video.path,0).then(blob=>{
      if(cancelled)return
      url=URL.createObjectURL(blob);setThumbnail(url)
    }).catch(()=>{})
    return()=>{cancelled=true;if(url)URL.revokeObjectURL(url)}
  },[video.path,result?.path])
  const chain=renderStages(settings).filter(stage=>stage!=='vfg')
  const dimensions=hasScaleStages(settings)?vsrOutputSize(settings,video):video
  const extension=format==='jpeg'?'jpg':format
  const output=`${directory.replace(/[\\/]+$/,'')}\\${name}.${extension}`
  const valid=Boolean(directory&&name.trim()&&!/[<>:"/\\|?*\x00-\x1f]/.test(name))
  const run=async()=>{
    setBusy(true);setError('');setResult(null);onStatusChange?.('正在渲染并保存图片…')
    try {
      const saved=await exportImage({path:video.path,output_path:output,format,quality,preserve_alpha:alpha,overwrite},settings)
      setResult(saved);onStatusChange?.(`图片已保存：${saved.path}`)
    }catch(e:any){setError(e.message);onStatusChange?.(`图片导出失败：${e.message}`)}
    finally{setBusy(false)}
  }
  return <div className="image-deliver workspace-page">
    <div className="image-deliver-title"><h1>图片导出</h1><span>{video.filename}</span></div>
    <div className="image-deliver-grid">
      <section className="image-deliver-preview" aria-label="图片预览">
        <div className="image-deliver-canvas">{thumbnail?<ImageViewport src={thumbnail} zoom={previewZoom} onZoom={setPreviewZoom} pan={previewPan} onPan={setPreviewPan}/>:<ImageIcon size={40}/>}</div>
        <div className="image-deliver-caption"><span>{result?'已导出':'原图片'}</span><button className="image-fit" title="恢复适配" onClick={()=>{setPreviewZoom(1);setPreviewPan({x:.5,y:.5})}}>{Math.round(previewZoom*100)}%</button><strong>{result?`${result.width} × ${result.height}`:`${video.width} × ${video.height}`}</strong></div>
        <div className="image-deliver-chain">{chain.length?chain.map((stage,index)=><span key={stage}>{index>0&&<i>→</i>}{stageName(stage)}</span>):<span>原图输出</span>}</div>
      </section>
      <section className="image-deliver-form" aria-label="图片保存设置">
        <fieldset disabled={busy}>
          <div className="image-field">格式<div className="image-format" role="group" aria-label="图片格式">{['png','jpeg','webp','avif','heic'].map(value=><button type="button" disabled={!formats[value]} title={formats[value]?value.toUpperCase():'当前运行环境缺少此格式的编码组件'} aria-pressed={format===value} key={value} onClick={()=>setFormat(value)}>{value.toUpperCase()}</button>)}</div></div>
          {format!=='png'&&<label className="image-field">质量 <output>{quality===100&&['webp','heic'].includes(format)?'100 · 无损':quality}</output><input type="range" aria-label="图片质量" min={1} max={100} step={1} value={quality} onChange={e=>setQuality(Number(e.target.value))}/></label>}
          <label className="image-field">保存位置<div className="image-directory"><input aria-label="图片保存目录" value={directory} onChange={e=>setDirectory(e.target.value)}/><button className="icon-button" type="button" aria-label="选择图片保存目录" onClick={async()=>{try{const path=await openNativeDirDialog();if(path)setDirectory(path)}catch(e:any){setError(e.message)}}}><Folder size={18}/></button></div></label>
          <label className="image-field">文件名<div className="image-filename"><input aria-label="图片文件名" value={name} onChange={e=>setName(e.target.value)}/><span>.{extension}</span></div></label>
          {format!=='jpeg'&&<label className="image-check"><input type="checkbox" checked={alpha} onChange={e=>setAlpha(e.target.checked)}/>保留透明度</label>}
          <label className="image-check"><input type="checkbox" checked={overwrite} onChange={e=>setOverwrite(e.target.checked)}/>覆盖同名文件</label>
        </fieldset>
        <div className="image-deliver-output"><span>输出</span><strong>{dimensions.width} × {dimensions.height}</strong><span>sRGB · 8-bit</span></div>
        {chain.includes('truehdr')&&<p className="image-deliver-note">HDR 结果映射为 SDR 图片后保存。</p>}
        {renderStages(settings).includes('vfg')&&<p className="image-deliver-note">静态图片不执行插帧。</p>}
        {format==='jpeg'&&video.has_alpha&&<p className="image-deliver-note">透明区域填充白色。</p>}
        <button className="button primary image-save" disabled={!valid||busy} onClick={run}>{busy?'正在保存…':'保存图片'}<ArrowUpRight size={18}/></button>
        {error&&<p className="image-deliver-error" role="alert">{error}</p>}
        {result&&<div className="image-deliver-saved" role="status"><Check size={18}/><div><strong>已保存</strong><p>{result.path}</p></div></div>}
      </section>
    </div>
  </div>
}
