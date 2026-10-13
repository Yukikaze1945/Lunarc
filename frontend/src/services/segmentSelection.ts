import {useSyncExternalStore} from 'react'
import type {TranscodeSettings} from '../types'

export type FrameSelection={enabled:boolean;start_frame:number;end_frame:number}
const ranges=new Map<string,FrameSelection>(),listeners=new Set<()=>void>()
let tx:TranscodeSettings|null=null
const subscribe=(listener:()=>void)=>{listeners.add(listener);return()=>{listeners.delete(listener)}}
const notify=()=>listeners.forEach(listener=>listener())
export function setFrameSelection(path:string,value:FrameSelection){ranges.set(path,value);notify()}
export function useFrameSelection(path?:string){return useSyncExternalStore(subscribe,()=>path?ranges.get(path):undefined)}
export function publishExportSettings(value:TranscodeSettings){if(JSON.stringify(value)!==JSON.stringify(tx)){tx=value;notify()}}
export function useExportSettings(){return useSyncExternalStore(subscribe,()=>tx)}
