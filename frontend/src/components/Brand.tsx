// localized-render-sinks
import {ui} from '../i18n/locale'
export function Brand({compact = false}: {compact?: boolean}) {
  return <div className="brand"><img src="./lunarc.svg" alt={ui("Lunarc 光弧标志")} width={36} height={36}/>{ui(!compact && <div><strong>Lunarc</strong><small>LIGHT IN MOTION</small></div>)}</div>
}
