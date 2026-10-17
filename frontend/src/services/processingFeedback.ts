export function gpuFailure(message:unknown):{kind:'memory'|'reset';message:string;detail:string}|null {
  const detail=typeof message==='string'?message:message instanceof Error?message.message:''
  if(!detail)return null
  if(/cuda(?:_error_| error: | )out.of.memory|e_outofmemory|0x8007000e|nv_enc_err_out_of_memory|显存不足|GPU 内存分配失败|GPU memory allocation failed|可用显存约[^\n]*低于启动检查|(?:cuMemAlloc|cudaMalloc)[^\n]*(?:out.of.memory|failed[^\n]*\b2\b)/i.test(detail))
    return {kind:'memory',message:'显存不足，本次处理已停止。请降低输出分辨率、减少处理项，或释放其他程序占用后重试。',detail}
  if(/0x887a000[567]|DXGI_ERROR_DEVICE_(?:REMOVED|HUNG|RESET)|GPU 设备已被驱动重置|GPU (?:device )?(?:has been )?(?:removed|reset)/i.test(detail))
    return {kind:'reset',message:'GPU 已重置，本次处理已停止。请等待后端恢复后重试；反复出现时可降低分辨率或减少处理项。',detail}
  return null
}
export function memoryLevel(used:number|undefined,total:number|undefined){
  if(used==null||total==null||!Number.isFinite(used)||!Number.isFinite(total)||used<0||total<=0)return null
  const percent=used/total*100
  return {percent,fill:Math.min(100,percent),level:percent>=92?'critical':percent>=80?'warning':'normal'} as const
}
