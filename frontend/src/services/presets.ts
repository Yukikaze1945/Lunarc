import {completeSettings,DEFAULT_SETTINGS} from './workspace.ts'
import {renderStages} from './renderPipeline.ts'
import type {DLSSSettings} from '../types'

export type Preset = {name:string; settings:DLSSSettings}
const OPTIONAL_DEFAULTS:Partial<DLSSSettings>={dlaa_preset:0,truehdr_enabled:false,
  truehdr_contrast:100,truehdr_saturation:100,truehdr_middle_gray:50,truehdr_peak_nits:650,truehdr_debanding:true,
  layer_settings:[],layer_execution:'sequential'}
export function presetSettings(raw:Partial<DLSSSettings>):DLSSSettings {
  const quality=raw.flow_quality||DEFAULT_SETTINGS.flow_quality
  const settings=completeSettings({...OPTIONAL_DEFAULTS,
    raft_analysis_edge:quality==='Fast'?512:quality==='Quality'?1080:720,
    raft_iterations:quality==='Fast'?4:quality==='Quality'?12:6,...raw})
  return {...settings,render_pipeline:renderStages(settings)}
}
export const DEFAULT_PRESET=presetSettings(DEFAULT_SETTINGS)
export function samePreset(a:Partial<DLSSSettings>,b:Partial<DLSSSettings>):boolean {
  const sorted=(value:any):any=>Array.isArray(value)?value.map(sorted):value&&typeof value==='object'
    ?Object.fromEntries(Object.keys(value).sort().map(key=>[key,sorted(value[key])])):value
  return JSON.stringify(sorted(presetSettings(a)))===JSON.stringify(sorted(presetSettings(b)))
}
const KEY='lunarc.presets.v1'
let raw='', snapshot:Preset[]=[]
export function presetSnapshot():Preset[] {
  const value=localStorage.getItem(KEY)||'[]'
  if(value!==raw){raw=value;try{const list=JSON.parse(value);snapshot=Array.isArray(list)?list.filter(p=>
    typeof p?.name==='string'&&p.name.trim()&&p.settings&&typeof p.settings==='object'&&!Array.isArray(p.settings))
    .map(p=>({name:p.name,settings:presetSettings(p.settings)})):[]}catch{snapshot=[]}}
  return snapshot
}
export function subscribePresets(listener:()=>void) {
  window.addEventListener('lunarc:presets',listener);window.addEventListener('storage',listener)
  return()=>{window.removeEventListener('lunarc:presets',listener);window.removeEventListener('storage',listener)}
}
export function writePresets(value:Preset[]) {
  localStorage.setItem(KEY,JSON.stringify(value));window.dispatchEvent(new Event('lunarc:presets'))
}
export function savePreset(name:string,settings:DLSSSettings) {
  name=name.trim();if(!name||name.length>80)throw new Error('请输入 1–80 字的预设名称')
  writePresets([...presetSnapshot().filter(p=>p.name!==name),{name,settings:structuredClone(presetSettings(settings))}])
}

const labels:Record<string,string>={render_pipeline:'渲染顺序',style:'画面风格',intensity:'NR 强度',local_tone:'局部色调强度',
  local_struct:'局部结构强度',skin_struct:'皮肤结构强度',use_auto_mask:'自动遮罩',ui_correction:'NR 界面修正',
  guidance_mode:'光流引导',flow_engine:'光流引擎',flow_quality:'光流质量',working_scale:'工作缩放',
  reconstruction_mode:'重建层数',layer_execution:'图层执行方式',layer_scales:'图层缩放',layer_weights:'图层权重',
  fusion_mode:'融合方式',fusion_beta:'融合 β',fusion_max_iterations:'求解迭代上限',fusion_relative_tolerance:'求解误差阈值',
  residual_strength:'残差倍率',residual_pyramid:'保留源细节',motion_scale_x:'运动 X',motion_scale_y:'运动 Y',
  dlaa_preset:'DLAA 模型',vsr_quality:'VSR 强度',vsr_output_mode:'VSR 输出方式',vsr_target:'VSR 输出分辨率',
  vsr_scale:'VSR 放大倍率',vsr_width:'VSR 输出宽度',vsr_height:'VSR 输出高度',frame_generation:'插帧模式',
  frame_multiplier:'插帧倍率',framegen_target_fps:'插帧目标帧率',framegen_quality:'插帧质量',framegen_scene_detection:'镜头切换检测',
  truehdr_enabled:'TrueHDR',truehdr_contrast:'TrueHDR 对比度',truehdr_saturation:'TrueHDR 饱和度',
  truehdr_middle_gray:'TrueHDR 中灰亮度',truehdr_peak_nits:'TrueHDR 目标峰值',truehdr_debanding:'AI 去色带',
  total_gain:'总强度',lightness_gain:'明度总强度',chroma_gain:'色度强度',low_frequency_gain:'低频强度',
  high_frequency_gain:'高频强度',hue_protection:'色相保护',dark_protection:'暗部保护',highlight_protection:'亮部保护',
  soft_compression:'过度修正抑制',roi_kind:'区域保护',roi_bounds:'保护区域边界',roi_strength:'区域保护强度',
  mode:'时域稳定方式',strength:'时域稳定强度',scene_detection:'镜头切换检测',scene_threshold:'镜头切换阈值',
  raft_analysis_edge:'RAFT 分析长边',raft_iterations:'RAFT 迭代次数',export_backend:'导出后端',
  experimental_hdr_scrgb:'HDR 导出',hdr_paper_white_scrgb:'HDR 纸白',dlss_dll_path:'DLAA 运行库',vsr_runtime_path:'VFX 运行库'}
const words:Record<string,string>={nr:'DLSSNR',dlaa:'DLAA',vsr:'RTX VSR',vfg:'NVIDIA VFG（固定）',truehdr:'TrueHDR',
  vfg_dynamic:'NVIDIA VFG（动态）',off:'关闭',single:'单层',dual:'双层',triple:'三层',sequential:'顺序堆叠',
  catmull_baseline:'仅第一层',data_consistency_experimental:'多层融合',scale:'按倍率',resolution:'按分辨率',
  nvof:'NVIDIA 光流',fidelityfx:'AMD 光流',raft:'RAFT',raft_large:'RAFT-Large',gpu_transfer:'GPU Transfer',auto:'自动',legacy:'Legacy'}
export function presetValue(key:string,value:unknown):string {
  if(value===undefined||value===null)return '默认'
  if(Array.isArray(value))return value.map(v=>presetValue(key,v)).join(' / ')
  if(typeof value==='boolean')return value?'开启':'关闭'
  if(key==='style')return ['默认','自然','电影'][Number(value)]??String(value)
  if(key==='vsr_quality')return ['低','中','高','极高'][Number(value)-1]??String(value)
  if(key==='framegen_quality')return ['低','中','高'][Number(value)]??String(value)
  if(['use_auto_mask','ui_correction','guidance_mode'].includes(key))return value?'开启':'关闭'
  if(typeof value==='number')return Number(value.toFixed(4)).toString()
  return words[String(value)]??String(value)
}
export type PresetChange={key:string;label:string;before:string;after:string;delta?:number;percent?:number}
export function presetChanges(settings:DLSSSettings):PresetChange[] {
  const baseline=DEFAULT_PRESET
  const current=presetSettings(settings)
  const changes:PresetChange[]=[]
  const visit=(value:any,base:any,path:string)=>{
    if(Array.isArray(value)&&['layer_scales','layer_weights'].includes(path)){
      value.forEach((entry,index)=>visit(entry,base?.[index]??base?.[0],`第 ${index+1} 层.${path}`))
      return
    }
    if(Array.isArray(value)&&path==='post_controls.roi_bounds'){
      value.forEach((entry,index)=>visit(entry,base?.[index],`${path}.${['左边界','上边界','右边界','下边界'][index]}`))
      return
    }
    if(value&&typeof value==='object'&&!Array.isArray(value)){
      for(const key of Object.keys(value))visit(value[key],base?.[key],path?`${path}.${key}`:key)
      return
    }
    if(JSON.stringify(value)===JSON.stringify(base))return
    const key=path.split('.').pop()!
    if(key==='preset'||key==='neural_mode'||key==='dlss_backend'||key.startsWith('_'))return
    const delta=typeof value==='number'&&typeof base==='number'&&
      !['style','use_auto_mask','ui_correction','guidance_mode','dlaa_preset','vsr_quality','framegen_quality','roi_kind','debug_mode'].includes(key)?value-base:undefined
    changes.push({key:path,label:labels[key]??key,before:presetValue(key,base),after:presetValue(key,value),delta,
      percent:delta!==undefined&&base!==0?delta/Math.abs(base)*100:undefined})
  }
  for(const [key,value] of Object.entries(current)){
    if(key==='layer_settings'){
      const nr={style:0,intensity:1,local_tone:1,local_struct:1,skin_struct:1,use_auto_mask:1,ui_correction:0}
      ;(value as Record<string,number>[]|undefined)?.forEach((layer,index)=>{if(index)visit(layer,nr,`第 ${index+1} 层`)})
    }else visit(value,(baseline as any)[key],key)
  }
  return changes.map(item=>({...item,label:item.key.startsWith('第 ')?`${item.key.split('.')[0]} · ${item.label}`:item.label}))
}
