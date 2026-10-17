export type ActivityState='idle'|'loading'|'error'|'recovering'|'exporting'
export function activityState({recovering,error,exporting,loading}:{recovering:boolean;error:boolean;exporting:boolean;loading:boolean}):ActivityState{
  if(recovering)return 'recovering'
  if(error)return 'error'
  if(exporting)return 'exporting'
  return loading?'loading':'idle'
}
