import {useLanguagePreference,setLocale,ui} from '../i18n/locale'
import {type LanguagePreference} from '../i18n/translate'
export function LanguagePicker(){
  const preference=useLanguagePreference()
  return <label className="language-picker">
    <span>{ui('界面语言')}</span>
    <select aria-label={ui('界面语言')} value={preference} onChange={event=>setLocale(event.target.value as LanguagePreference)}>
      <option value="auto">{ui('跟随系统')}</option>
      <option value="zh-CN">中文</option><option value="en">English</option><option value="ja">日本語</option><option value="ru">Русский</option>
    </select>
  </label>
}
