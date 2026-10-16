import en from './en.json'
import ja from './ja.json'
import ru from './ru.json'

export const LOCALES = ['zh-CN', 'en', 'ja', 'ru'] as const
export type Locale = typeof LOCALES[number]
export function isLocale(value: unknown): value is Locale {
  return typeof value === 'string' && (LOCALES as readonly string[]).includes(value)
}
export type LanguagePreference = Locale | 'auto'
export function isLanguagePreference(value: unknown): value is LanguagePreference {
  return value === 'auto' || isLocale(value)
}
export function resolveLocale(preference: LanguagePreference, languages: readonly string[]): Locale {
  if (isLocale(preference)) return preference
  for (const language of languages) {
    const base = language.toLowerCase().replaceAll('_', '-').split('-')[0]
    if (base === 'zh') return 'zh-CN'
    if (base === 'en' || base === 'ja' || base === 'ru') return base
  }
  return 'en'
}
const catalogs: Record<string, Record<string,string>> = {en,ja,ru}
const escape = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
type Pattern = {regex:RegExp; slots:string[]; target:string; weight:number}
const patterns = new Map<Locale, Pattern[]>()
function compiled(locale:Locale):Pattern[] {
  const cached=patterns.get(locale)
  if(cached)return cached
  const result=Object.entries(catalogs[locale]||{}).filter(([source])=>/\{\d+\}/.test(source)).map(([source,target])=>{
    const slots:string[]=[]
    let pattern='',end=0
    for(const match of source.matchAll(/\{(\d+)\}/g)){
      pattern+=escape(source.slice(end,match.index))+'([\\s\\S]*?)';slots.push(match[1]);end=match.index!+match[0].length
    }
    pattern+=escape(source.slice(end))
    return {regex:new RegExp('^'+pattern+'$'),slots,target,weight:source.replace(/\{\d+\}/g,'').length}
  }).sort((a,b)=>b.weight-a.weight)
  patterns.set(locale,result);return result
}
export function translateText(value:string,locale:Locale):string {
  if(locale==='zh-CN'||!value||/^(?:[a-z]:[\\/]|\\\\|https?:\/\/)/i.test(value))return value
  const catalog=catalogs[locale]
  if(Object.hasOwn(catalog,value))return catalog[value]
  const trimmed=value.trim()
  if(trimmed!==value&&Object.hasOwn(catalog,trimmed))return value.slice(0,value.indexOf(trimmed))+catalog[trimmed]+value.slice(value.indexOf(trimmed)+trimmed.length)
  if(!/[\u3400-\u9fff]/.test(value))return value
  for(const item of compiled(locale)){
    const match=item.regex.exec(value)
    if(!match)continue
    const values:Record<string,string>={}
    // Captured text can be a user filename, preset name or raw diagnostic.
    item.slots.forEach((slot,i)=>{values[slot]=match[i+1]})
    return item.target.replace(/\{(\d+)\}/g,(token,slot)=>values[slot]??token)
  }
  return value
}

/** Display-only translation: never call this for settings, payloads or raw logs. */
export function translateVisible<T>(value:T,locale:Locale):T {
  if(typeof value==='string')return translateText(value,locale) as T
  if(Array.isArray(value))return value.map(item=>translateVisible(item,locale)) as T
  return value
}
