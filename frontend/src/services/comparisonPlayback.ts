export type PlaybackFrame = {blob:Blob;actualTime:number;actualPts:number}
export type PlaybackDecoderInfo = {decoder:string;gpu_scale:boolean;fallback_reason?:string|null;display_size:number[];display_fps:number;jpeg_mean_ms:number}
export type PlaybackInfo = {original:PlaybackDecoderInfo;exported:PlaybackDecoderInfo;display_fps:number}
export type PlaybackPair = {time:number;original:PlaybackFrame;exported:PlaybackFrame;playback?:PlaybackInfo}

export function decodeComparisonPacket(buffer:ArrayBuffer):PlaybackPair {
  if(buffer.byteLength<4)throw new Error('对比播放数据不完整')
  const headerSize=new DataView(buffer).getUint32(0)
  if(headerSize>buffer.byteLength-4)throw new Error('对比播放头部不完整')
  const header=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,4,headerSize)))
  const start=4+headerSize, aSize=header.original?.size, bSize=header.exported?.size
  if(!Number.isFinite(header.time)||!Number.isInteger(aSize)||!Number.isInteger(bSize)||
    aSize<=0||bSize<=0||start+aSize+bSize!==buffer.byteLength)throw new Error('对比播放帧大小无效')
  const frame=(side:'original'|'exported',offset:number,size:number):PlaybackFrame=>{
    if(!Number.isFinite(header[side].actualTime)||!Number.isFinite(header[side].actualPts))
      throw new Error('对比播放帧时间无效')
    return {blob:new Blob([buffer.slice(offset,offset+size)],{type:'image/jpeg'}),
      actualTime:header[side].actualTime,actualPts:header[side].actualPts}
  }
  return {time:header.time,original:frame('original',start,aSize),exported:frame('exported',start+aSize,bSize),playback:header.playback}
}
