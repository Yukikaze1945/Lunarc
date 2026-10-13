let metrics:any=null,pending:any=null,timer:ReturnType<typeof setTimeout>|null=null
const listeners=new Set<()=>void>()
export const previewMetricsSnapshot=()=>metrics
export const subscribePreviewMetrics=(listener:()=>void)=>{listeners.add(listener);return()=>{listeners.delete(listener)}}
export function publishPreviewMetrics(value:any){
  pending=value
  if(!timer)timer=setTimeout(()=>{metrics=pending;timer=null;listeners.forEach(listener=>listener())},300)
}
