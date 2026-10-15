import {useSyncExternalStore} from 'react'
import {isLocale,translateVisible,type Locale} from './translate'
const STORAGE_KEY='lunarc.language.v1'
function read():Locale {
  try{const value=localStorage.getItem(STORAGE_KEY);if(isLocale(value))return value}catch{}
  return 'zh-CN'
}
let current=read()
const listeners=new Set<()=>void>()
export function getLocale(){return current}
export function subscribeLocale(listener:()=>void){listeners.add(listener);return()=>{listeners.delete(listener)}}
export function useLocale(){return useSyncExternalStore(subscribeLocale,getLocale,()=> 'zh-CN' as Locale)}
function apply(){
  document.documentElement.lang=current
  document.documentElement.dataset.locale=current
}
export function setLocale(locale:Locale){
  if(!isLocale(locale))return
  const changed=current!==locale;current=locale;apply()
  try{localStorage.setItem(STORAGE_KEY,current)}catch{}
  void window.electronAPI?.setLanguage?.(current).catch(()=>{})
  if(changed)listeners.forEach(listener=>listener())
}
export async function initializeLocale(){
  let stored:unknown
  try{stored=localStorage.getItem(STORAGE_KEY)}catch{}
  if(!isLocale(stored)){
    try{const native=await window.electronAPI?.getLanguage?.();if(isLocale(native))current=native}catch{}
  }
  apply()
  void window.electronAPI?.setLanguage?.(current).catch(()=>{})
}
export function ui<T>(value:T):T{return translateVisible(value,current)}
