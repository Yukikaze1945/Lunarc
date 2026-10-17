// localized-render-sinks
import {ui} from '../i18n/locale'
import {useState} from 'react'
import {ComponentPanel} from './ComponentPanel'
import './guide.css'

export function Guide({setup=false,onFinish}:{setup?:boolean;onFinish?:()=>void}) {
  const [step,setStep]=useState(0)
  return <div className="guide workspace-page">
    <div className="page-title"><span className="eyebrow">LUNARC</span><h1>{ui(setup?'开始使用':'使用教程')}</h1><p>{ui("准备组件、调整画面，再导出视频。")}</p></div>
    <nav className="guide-steps" aria-label={ui("教程步骤")}>{ui(['准备组件','调整画面','导出视频'].map((s,i)=><button key={s} aria-current={step===i?'step':undefined} onClick={()=>setStep(i)}><span>{ui(String(i+1).padStart(2,'0'))}</span>{ui(s)}</button>))}</nav>
    {ui(step===0?<><ComponentPanel basicOnly={setup}/><section className="panel"><h2>{ui("需要哪些组件")}</h2><p>{ui("基础运行环境和视频处理组件用于启动工作台。RAFT 光流是可选项，可以稍后在软件设置中安装。")}</p><p>{ui("DLSSNR 运行库单独管理。在“运行与诊断”中下载或导入，并完成签名与版本校验。DLAA、RTX VSR 和 NVIDIA VFG 也需要对应的 NVIDIA 运行库。")}</p></section></>:step===1?<section className="panel guide-instructions"><h2>{ui("导入素材，按顺序处理")}</h2><ol><li>{ui("点击“导入”，或将视频、图片拖入工作台。")}</li><li>{ui("导入后先设置渲染链，再点击“开启渲染”。视频可选择实时渲染，或从当前帧预渲染一小段后回看。")}</li><li>{ui("添加需要的渲染卡片。拖动卡片左上角的手柄调整顺序，处理从上到下执行。")}</li><li>{ui("先用默认参数预览，再调整 NR 强度、局部色调、局部结构和皮肤结构。更多参数在高级设置中。")}</li></ol><p>{ui("预览用于检查画面。HDR 预览经过显示映射，不能用它判断显示器上的 HDR 亮度。VFG 用于导出插帧，实时预览保持原帧率。")}</p></section>:<section className="panel guide-instructions"><h2>{ui("加入队列，导出后对比")}</h2><ol><li>{ui("进入“导出”，选择保存位置、编码格式和画质。")}</li><li>{ui("点击加入队列，再开始导出。任务保存加入时的参数，修改工作台不会影响已加入的任务。")}</li><li>{ui("导出完成后，在“视频对比”中打开原视频和导出视频，检查画面与播放效果。")}</li></ol><p>{ui("任务失败时，打开“运行诊断”查看原因。失败或完成的任务都可以从队列删除。")}</p></section>)}
    <div className="guide-actions">{ui(onFinish&&<button className="button quiet" onClick={onFinish}>{ui("跳过教学")}</button>)}<button className="button" disabled={step===0} onClick={()=>setStep(s=>s-1)}>{ui("上一步")}</button>{ui(step<2?<button className="button primary" onClick={()=>setStep(s=>s+1)}>{ui("下一步")}</button>:onFinish&&<button className="button primary" onClick={onFinish}>{ui("进入工作台")}</button>)}</div>
  </div>
}
