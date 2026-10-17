import {AlertTriangle} from 'lucide-react'
import {ui} from '../i18n/locale'
import {gpuFailure} from '../services/processingFeedback'
export function ProcessingError({message,onDismiss}:{message:string;onDismiss?:()=>void}){
  const failure=gpuFailure(message)
  return <div className="processing-error" role="alert"><AlertTriangle size={16}/><div><p>{ui(failure?.message||message)}</p>
    {failure&&<details><summary>{ui('技术详情')}</summary><pre>{failure.detail}</pre></details>}
  </div>{onDismiss&&<button type="button" aria-label={ui('关闭错误')} onClick={onDismiss}>×</button>}</div>
}
