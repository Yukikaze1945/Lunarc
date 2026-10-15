import {useLocale,setLocale,ui} from '../i18n/locale'
import {type Locale} from '../i18n/translate'
export function LanguagePicker({compact=false}:{compact?:boolean}){
  const locale=useLocale()
  return <label className={`language-picker${compact?' language-picker-compact':''}`}>
    {!compact&&<span>{ui('界面语言')}</span>}
    <select aria-label={ui('界面语言')} value={locale} onChange={event=>setLocale(event.target.value as Locale)}>
      <option value="zh-CN">中文</option><option value="en">English</option><option value="ja">日本語</option><option value="ru">Русский</option>
    </select>
  </label>
}
