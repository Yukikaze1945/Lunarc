import {useEffect,useState} from 'react'
import type {DLSSSettings} from '../types'
import {imageModelRequest} from '../services/api'

export function ImageModelTile({kind,settings,onChange}:{kind:'illustration'|'animesharp';settings:DLSSSettings;onChange:(patch:Partial<DLSSSettings>)=>void}) {
  const key=kind==='illustration'?`illustration-${settings.image_model_variant??'denoise'}`:'animesharp'
  const [state,setState]=useState<any>(null),[error,setError]=useState(''),[path,setPath]=useState('')
  useEffect(()=>{
    let active=true,timer:ReturnType<typeof setTimeout>,wasBusy=false
    const refresh=async()=>{
      try {const value=await imageModelRequest();if(active){setState(value);if(wasBusy&&!value.busy&&!value.error)window.dispatchEvent(new Event('lunarc:settings-settled'));wasBusy=Boolean(value.busy)}}catch(e:any){if(active)setError(e.message)}
      if(active)timer=setTimeout(refresh,1500)
    }
    void refresh();return()=>{active=false;clearTimeout(timer)}
  },[key])
  const action=async(command:string)=>{
    setError('')
    try {setState(await imageModelRequest(command,{key,...(command==='/import'?{path}: {})}));if(command==='/import')window.dispatchEvent(new Event('lunarc:settings-settled'))}
    catch(e:any){setError(e.message)}
  }
  return <div className="render-stage-options">
    {kind==='illustration'&&<label className="render-full">版本<select aria-label="IllustrationJaNai V3 版本" value={settings.image_model_variant??'denoise'} onChange={e=>onChange({image_model_variant:e.target.value as 'denoise'|'detail'})}><option value="denoise">去噪 · 4× DAT2</option><option value="detail">细节 · 4× DAT2</option></select></label>}
    {kind==='animesharp'&&<label className="render-full">模型<span>2× RCAN · 标准版</span></label>}
    <label className="render-full">处理方式<select aria-label="图片模型处理方式" value={settings.image_model_tile??0} onChange={e=>onChange({image_model_tile:Number(e.target.value) as DLSSSettings['image_model_tile']})}><option value={0}>整图</option><option value={128}>分块 128</option><option value={256}>分块 256</option><option value={512}>分块 512</option></select></label>
    {!!settings.image_model_tile&&<p className="dlss-note">分块降低显存用量；结果可能与整图不同。</p>}
    <div className="render-full image-model-actions"><span>{state?.models?.[key]?.installed?'模型已安装':'模型未安装'}</span><button type="button" disabled={!!state?.busy} onClick={()=>void action('/download')}>{state?.busy===key?'下载中…':'下载模型'}</button></div>
    <label className="render-full">本地导入<input aria-label={`${kind} 模型文件`} value={path} placeholder=".safetensors / .pth 文件路径" onChange={e=>setPath(e.target.value)}/><button type="button" disabled={!path.trim()||!!state?.busy} onClick={()=>void action('/import')}>导入</button></label>
    {state?.busy===key&&<progress className="render-full" max={state.total||1} value={state.received||0}/>}
    {(error||state?.error)&&<p className="dlss-note" role="alert">{error||state.error}</p>}
  </div>
}
