export type ComponentId = 'base'|'ffmpeg'|'raft'
export type ComponentInfo = {id:ComponentId; label:string; installed:boolean; version:string; detail:string}
export type ComponentState = {components:ComponentInfo[]; busy:ComponentId|null; phase:string; received:number; total:number; error:string|null; requiresSetup:boolean; directory:string}
