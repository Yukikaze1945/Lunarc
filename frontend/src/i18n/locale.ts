import {useSyncExternalStore} from 'react'
import {isLanguagePreference,resolveLocale,translateVisible,type LanguagePreference} from './translate'
const STORAGE_KEY='lunarc.language.v1'
function read():LanguagePreference {
  try{const value=localStorage.getItem(STORAGE_KEY);if(isLanguagePreference(value))return value}catch{}
  return 'auto'
}
let systemLanguages=Array.from(navigator.languages?.length?navigator.languages:[navigator.language])
let preference=read()
let current=resolveLocale(preference,systemLanguages)
const listeners=new Set<()=>void>()
export function getLocale(){return current}
export function getLanguagePreference(){return preference}
export function subscribeLocale(listener:()=>void){listeners.add(listener);return()=>{listeners.delete(listener)}}
export function useLocale(){return useSyncExternalStore(subscribeLocale,getLocale,()=> 'en' as const)}
export function useLanguagePreference(){return useSyncExternalStore(subscribeLocale,getLanguagePreference,()=> 'auto' as const)}
function apply(){
  document.documentElement.lang=current
  document.documentElement.dataset.locale=current
}
export function setLocale(value:LanguagePreference){
  if(!isLanguagePreference(value))return
  const next=resolveLocale(value,systemLanguages)
  const changed=current!==next||preference!==value
  preference=value;current=next;apply()
  try{localStorage.setItem(STORAGE_KEY,preference)}catch{}
  void window.electronAPI?.setLanguage?.(preference).catch(()=>{})
  if(changed)listeners.forEach(listener=>listener())
}
export async function initializeLocale(){
  let stored:unknown
  try{stored=localStorage.getItem(STORAGE_KEY)}catch{}
  try{
    const native=await window.electronAPI?.getLanguage?.()
    if(native&&typeof native==='object'&&'preference' in native&&'systemLanguages' in native){
      if(Array.isArray(native.systemLanguages))systemLanguages=native.systemLanguages.filter((value):value is string=>typeof value==='string')
      if(!isLanguagePreference(stored)&&isLanguagePreference(native.preference))preference=native.preference
    }else if(!isLanguagePreference(stored)&&isLanguagePreference(native))preference=native
  }catch{}
  current=resolveLocale(preference,systemLanguages)
  apply()
  void window.electronAPI?.setLanguage?.(preference).catch(()=>{})
}
window.addEventListener('languagechange',()=>{
  systemLanguages=Array.from(navigator.languages?.length?navigator.languages:[navigator.language])
  if(preference==='auto'){
    const next=resolveLocale(preference,systemLanguages)
    if(next!==current){current=next;apply();listeners.forEach(listener=>listener())}
  }
})
export function ui<T>(value:T):T{return translateVisible(value,current)}
