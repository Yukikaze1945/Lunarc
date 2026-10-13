import {useCallback, useEffect, useMemo, useRef, useState} from 'react'
import type {PointerEvent as ReactPointerEvent} from 'react'
import {ChevronLeft, ChevronRight, FolderOpen, Pause, Play, Scan, Volume2} from 'lucide-react'
import {localToken, openNativeFileDialog} from '../services/api'
import {decodeComparisonPacket} from '../services/comparisonPlayback'
import type {PlaybackInfo} from '../services/comparisonPlayback'
import './compare.css'

const API = 'http://127.0.0.1:18923'
type Side = 'original'|'exported'
type PairRequest = {original:string;exported:string;label?:string;original_offset_seconds?:number;duration_seconds?:number}
type MediaInfo = {path:string;name:string;width:number;height:number;codec?:string;profile?:string;
  pixel_format?:string;bit_depth?:number;frame_rate?:number;duration_seconds:number;
  color_range?:string;color_matrix?:string;color_transfer?:string;color_primaries?:string;
  is_hdr?:boolean;display_assumption?:string;audio?:{codec?:string;channels?:number}|null}
type PairInfo = {session:string;original:MediaInfo;exported:MediaInfo;duration_seconds:number;
  hdr_display_note?:string;alignment?:string}
type FrameInfo = {url:string;actualTime:number;actualPts:number|null;requestedTime:number}

export interface ComparePageProps {
  initialPair?: PairRequest|null
  originalPath?: string
  exportedPath?: string
  onPairOpened?: (pair:PairInfo)=>void
}

function withToken(url:string) {
  const token=localToken()
  if(!token)return url
  return `${url}${url.includes('?')?'&':'?'}token=${encodeURIComponent(token)}`
}
function seconds(value:number) {
  if(!Number.isFinite(value)||value<0)return '--:--.---'
  const m=Math.floor(value/60),s=value%60
  return `${String(m).padStart(2,'0')}:${s.toFixed(3).padStart(6,'0')}`
}
function mediaLabel(info?:MediaInfo) {
  if(!info)return '尚未载入'
  const fps=info.frame_rate?`${info.frame_rate.toFixed(3)} fps`:'帧率未知'
  const depth=info.bit_depth?`${info.bit_depth}-bit`:'位深未知'
  return `${info.width} × ${info.height} · ${info.codec||'未知编码'} · ${fps} · ${depth} · ${info.is_hdr?'HDR':'SDR'}`
}

export function ComparePage({initialPair,originalPath,exportedPath,onPairOpened}:ComparePageProps) {
  const [original,setOriginal]=useState(originalPath||initialPair?.original||'')
  const [exported,setExported]=useState(exportedPath||initialPair?.exported||'')
  const [pair,setPair]=useState<PairInfo|null>(null)
  const [mode,setMode]=useState<'side'|'wipe'|'ab'>('side')
  const [ab,setAb]=useState<Side>('original')
  const [split,setSplit]=useState(50)
  const [zoom,setZoom]=useState<'fit'|'100'|'200'>('fit')
  const [currentTime,setCurrentTime]=useState(0)
  const [playing,setPlaying]=useState(false)
  const [playbackRevision,setPlaybackRevision]=useState(0)
  const [playbackRate,setPlaybackRate]=useState(1)
  const [playbackInfo,setPlaybackInfo]=useState<PlaybackInfo|null>(null)
  const [audioSide,setAudioSide]=useState<Side>('original')
  const [error,setError]=useState('')
  const [busy,setBusy]=useState(false)
  const [frame,setFrame]=useState<Record<Side,FrameInfo|null>>({original:null,exported:null})
  const [status,setStatus]=useState('等待载入')
  const imageUrls=useRef<string[]>([])
  const frameInFlight=useRef(false)
  const pendingFrame=useRef<{time:number;pair:PairInfo}|null>(null)
  const loadSerial=useRef(0)
  const frameAbort=useRef<AbortController|null>(null)
  const openAbort=useRef<AbortController|null>(null)
  const openSerial=useRef(0)
  const audioRef=useRef<HTMLAudioElement>(null)
  const splitStage=useRef<HTMLDivElement>(null)
  const dragSplit=useRef(false)
  const timeRef=useRef(0)
  const playingRef=useRef(false)
  const playbackSocket=useRef<WebSocket|null>(null)
  const playbackGeneration=useRef(0)
  const pairRef=useRef<PairInfo|null>(null)
  const duration=pair?.duration_seconds||0

  useEffect(()=>{setOriginal(originalPath||initialPair?.original||'');setExported(exportedPath||initialPair?.exported||'')},
    [originalPath,exportedPath,initialPair?.original,initialPair?.exported])

  const loadFrame=useCallback(async (at:number, pairData=pair) => {
    if(!pairData)return
    if(frameInFlight.current){pendingFrame.current={time:at,pair:pairData};loadSerial.current++;return}
    frameInFlight.current=true
    const serial=++loadSerial.current
    const controller=new AbortController()
    frameAbort.current=controller
    setBusy(true)
    try {
      const fetchSide=async (side:Side):Promise<FrameInfo>=>{
        const query=new URLSearchParams({session:pairData.session,time:String(Math.max(0,at)),side})
        const res=await fetch(`${API}/api/compare/frame?${query}`,{headers:localToken()?{'X-DLSS-Token':localToken()}:undefined,signal:controller.signal})
        if(!res.ok){const body=await res.json().catch(()=>null);throw new Error(body?.detail||`${side==='original'?'原片':'导出'}帧读取失败 (${res.status})`)}
        const blob=await res.blob(), url=URL.createObjectURL(blob)
        const actual=Number(res.headers.get('X-Comparison-Actual-Time'))
        const pts=Number(res.headers.get('X-Comparison-Actual-PTS'))
        return {url,actualTime:Number.isFinite(actual)?actual:at,
          actualPts:Number.isFinite(pts)?pts:null,requestedTime:at}
      }
      const [a,b]=await Promise.allSettled([fetchSide('original'),fetchSide('exported')])
      if(serial!==loadSerial.current){if(a.status==='fulfilled')URL.revokeObjectURL(a.value.url);if(b.status==='fulfilled')URL.revokeObjectURL(b.value.url);return}
      const next={original:a.status==='fulfilled'?a.value:null,
        exported:b.status==='fulfilled'?b.value:null}
      if(a.status==='fulfilled')imageUrls.current.push(a.value.url)
      if(b.status==='fulfilled')imageUrls.current.push(b.value.url)
      setFrame(next)
      if(a.status==='rejected'&&b.status==='rejected')throw a.reason
      setStatus(a.status==='rejected'||b.status==='rejected'
        ? '一侧已到达末帧'
        : '已载入')
      // Revoke old object URLs after React has switched to the replacements.
      const retained=new Set([next.original?.url,next.exported?.url])
      for(const url of imageUrls.current)if(!retained.has(url))URL.revokeObjectURL(url)
      imageUrls.current=Array.from(retained).filter((url):url is string=>Boolean(url))
    } catch(e:any) {if(serial===loadSerial.current)setError(e?.message||'读取比较帧失败')}
    finally {
      frameInFlight.current=false
      if(frameAbort.current===controller)frameAbort.current=null
      if(serial===loadSerial.current)setBusy(false)
      const queued=pendingFrame.current
      if(queued){pendingFrame.current=null;window.setTimeout(()=>void loadFrame(queued.time,queued.pair),0)}
    }
  },[pair])

  const closeSession=useCallback((session:string)=>{
    void fetch(`${API}/api/compare/${encodeURIComponent(session)}`,{
      method:'DELETE',headers:localToken()?{'X-DLSS-Token':localToken()}:undefined
    }).catch(()=>{})
  },[])

  const openPair=useCallback(async (source=original,result=exported)=>{
    if(!source||!result){setError('请先选择原片和导出结果。');return}
    const serial=++openSerial.current
    openAbort.current?.abort()
    frameAbort.current?.abort()
    pendingFrame.current=null
    loadSerial.current++
    setPlaying(false);playingRef.current=false;playbackGeneration.current++;playbackSocket.current?.close();audioRef.current?.pause()
    const controller=new AbortController();openAbort.current=controller
    lastOpened.current=`${source}\n${result}`
    setBusy(true);setError('');setStatus('正在载入…')
    try {
      const res=await fetch(`${API}/api/compare/open`,{method:'POST',
        headers:localToken()?{'Content-Type':'application/json','X-DLSS-Token':localToken()}:{'Content-Type':'application/json'},
        body:JSON.stringify({original:source,exported:result,original_offset_seconds:initialPair?.original_offset_seconds,duration_seconds:initialPair?.duration_seconds}),signal:controller.signal})
      const body=await res.json().catch(()=>null)
      if(serial!==openSerial.current)return
      if(!res.ok)throw new Error(body?.detail||'无法打开这组媒体文件')
      const opened=(body.pair||body) as PairInfo
      if(!opened.session||!opened.original||!opened.exported)throw new Error('对比服务未返回完整媒体信息')
      const previous=pairRef.current
      pairRef.current=opened
      setPair(opened);setPlaybackInfo(null);setFrame({original:null,exported:null});setCurrentTime(0);timeRef.current=0
      if(previous&&previous.session!==opened.session)closeSession(previous.session)
      setAudioSide(opened.original.audio?'original':opened.exported.audio?'exported':'original')
      setStatus('正在读取画面…')
      onPairOpened?.(opened)
      await loadFrame(0,opened)
    }catch(e:any){if(serial===openSerial.current&&!controller.signal.aborted){
      const previous=pairRef.current;pairRef.current=null;setPair(null)
      if(previous)closeSession(previous.session)
      setError(e?.message||'打开对比失败');setStatus('无法载入对比素材。')
    }}
    finally{if(serial===openSerial.current){if(openAbort.current===controller)openAbort.current=null;setBusy(false)}}
  },[original,exported,initialPair?.original_offset_seconds,initialPair?.duration_seconds,loadFrame,onPairOpened,closeSession])

  const lastOpened=useRef('')
  useEffect(()=>{
    if(original&&exported){const key=`${original}\n${exported}`;if(key!==lastOpened.current){lastOpened.current=key;void openPair(original,exported)}}
  },[original,exported,openPair])

  useEffect(()=>()=>{
    frameAbort.current?.abort();openAbort.current?.abort()
    if(pairRef.current)closeSession(pairRef.current.session)
    for(const url of imageUrls.current)URL.revokeObjectURL(url)
  },[closeSession])

  const audioUrl=useMemo(()=>pair&&pair[audioSide].audio?withToken(`${API}/api/compare/audio?${new URLSearchParams({session:pair.session,side:audioSide})}`):'',
    [pair,audioSide])
  useEffect(()=>{const audio=audioRef.current;if(!audio||!audioUrl)return;audio.pause();audio.src=audioUrl;audio.load();return()=>{audio.pause()}},[audioUrl])
  useEffect(()=>{if(audioRef.current)audioRef.current.playbackRate=playbackRate},[playbackRate])

  const play=async()=>{
    if(!pair)return
    if(timeRef.current>=duration){timeRef.current=0;setCurrentTime(0)}
    frameAbort.current?.abort();pendingFrame.current=null;loadSerial.current++;setBusy(false);setError('')
    setPlaying(true);playingRef.current=true
    const audio=audioRef.current
    if(audio&&audio.src){audio.pause();audio.currentTime=timeRef.current}
  }
  const pause=()=>{setPlaying(false);playingRef.current=false;playbackGeneration.current++;playbackSocket.current?.close();audioRef.current?.pause();void loadFrame(timeRef.current)}
  const audioEnded=()=>{
    // Short clips can finish between clock ticks. Record the real media time,
    // and keep the picture clock running if the other video is longer.
    const endedAt=Math.min(duration,audioRef.current?.currentTime||0)
    timeRef.current=endedAt;setCurrentTime(endedAt)
    if(endedAt>=duration)pause()
  }
  useEffect(()=>{
    if(!playing||!pair)return
    const generation=++playbackGeneration.current
    let disposed=false, first=true, drawing=false
    let lastDecoderInfo=''
    let pending:ArrayBuffer|null=null
    const query=new URLSearchParams({session:pair.session,time:String(timeRef.current),rate:String(playbackRate)})
    const socket=new WebSocket(withToken(`ws://127.0.0.1:18923/ws/compare?${query}`))
    socket.binaryType='arraybuffer';playbackSocket.current=socket
    const draw=async()=>{
      if(drawing)return
      drawing=true
      try{
        while(pending&&!disposed&&generation===playbackGeneration.current){
          const buffer=pending;pending=null
          const urls:string[]=[]
          try{
            const packet=decodeComparisonPacket(buffer)
            if(packet.playback){
              const key=JSON.stringify([packet.playback.original.decoder,packet.playback.exported.decoder,packet.playback.display_fps])
              if(key!==lastDecoderInfo){lastDecoderInfo=key;setPlaybackInfo(packet.playback)}
            }
            const make=async (side:Side):Promise<FrameInfo>=>{
              const source=packet[side],url=URL.createObjectURL(source.blob);urls.push(url)
              const image=new Image();image.src=url;await image.decode()
              return {url,actualTime:source.actualTime,actualPts:source.actualPts,requestedTime:packet.time}
            }
            const images=await Promise.allSettled([make('original'),make('exported')])
            if(images[0].status==='rejected')throw images[0].reason
            if(images[1].status==='rejected')throw images[1].reason
            if(disposed||!playingRef.current||generation!==playbackGeneration.current){urls.forEach(url=>URL.revokeObjectURL(url));break}
            const old=imageUrls.current;imageUrls.current=urls
            setFrame({original:images[0].value,exported:images[1].value})
            timeRef.current=packet.time;setCurrentTime(packet.time);setStatus('播放中')
            const audio=audioRef.current
            if(audio&&audioUrl){
              if(first||Math.abs(audio.currentTime-packet.time)>.18)audio.currentTime=packet.time
              if(audio.paused&&playingRef.current)void audio.play().catch(()=>{})
            }
            first=false
            await new Promise<void>(resolve=>window.requestAnimationFrame(()=>{old.forEach(url=>URL.revokeObjectURL(url));resolve()}))
          }catch(error:any){
            urls.forEach(url=>URL.revokeObjectURL(url))
            if(!disposed&&generation===playbackGeneration.current){setError(error.message||'对比播放帧读取失败');setPlaying(false);playingRef.current=false;audioRef.current?.pause()}
            break
          }
        }
      }finally{drawing=false}
    }
    socket.onmessage=event=>{
      if(disposed||generation!==playbackGeneration.current)return
      if(typeof event.data==='string'){
        const message=JSON.parse(event.data)
        if(message.type==='error'){setError(message.detail||'对比播放失败');setPlaying(false);playingRef.current=false;audioRef.current?.pause()}
        if(message.type==='ended'){timeRef.current=message.time;setCurrentTime(message.time);setPlaying(false);playingRef.current=false;audioRef.current?.pause();setStatus('播放结束')}
      }else{pending=event.data;void draw()}
    }
    socket.onerror=()=>{if(!disposed&&generation===playbackGeneration.current)setError('对比播放连接失败，请检查后端是否已重启')}
    socket.onclose=()=>{if(!disposed&&generation===playbackGeneration.current){setPlaying(false);playingRef.current=false;audioRef.current?.pause()}}
    return()=>{disposed=true;pending=null;socket.close();if(playbackSocket.current===socket)playbackSocket.current=null}
  },[playing,pair,playbackRate,playbackRevision,audioUrl])

  const seek=useCallback((value:number)=>{const next=Math.max(0,Math.min(duration,value));setCurrentTime(next);timeRef.current=next;if(playingRef.current){playbackGeneration.current++;playbackSocket.current?.close();setPlaybackRevision(value=>value+1)}else void loadFrame(next);if(audioRef.current&&!audioRef.current.paused)audioRef.current.currentTime=next},[duration,loadFrame])
  const step=async(direction:-1|1)=>{
    if(!pair)return
    try{
      const q=new URLSearchParams({session:pair.session,time:String(currentTime)})
      const res=await fetch(`${API}/api/compare/timeline?${q}`,{headers:localToken()?{'X-DLSS-Token':localToken()}:undefined})
      if(!res.ok)throw new Error('无法读取源帧时间戳')
      const data=await res.json()
      const candidates=(['original','exported'] as Side[]).map(side=>{
        const times=data?.sides?.[side]
        return direction<0?times?.previous_time:times?.next_time
      }).filter((time):time is number=>Number.isFinite(time))
      const actual=candidates.length
        ?(direction<0?Math.max(...candidates):Math.min(...candidates)):NaN
      if(Number.isFinite(actual))seek(actual)
      else setStatus(direction<0?'已到达可用帧的起点。':'已到达可用帧的终点。')
    }catch(e:any){setError(e?.message||'无法读取相邻帧的实际 PTS')}
  }
  const choose=async(side:Side)=>{
    try{const path=await openNativeFileDialog();if(!path)return
      if(side==='original')setOriginal(path);else setExported(path)
      const other=side==='original'?exported:original
      if(!other)setError('已选择文件；还需要选择配对的另一份文件。')
    }catch(e:any){setError(e.message||'文件选择失败')}
  }
  const pointerMove=(event:ReactPointerEvent<HTMLDivElement>)=>{
    if(!dragSplit.current||!splitStage.current)return
    const rect=splitStage.current.getBoundingClientRect()
    setSplit(Math.max(0,Math.min(100,(event.clientX-rect.left)/rect.width*100)))
  }
  const zoomClass=`compare-zoom-${zoom}`
  const image=(which:Side,alt:string)=><img className={`compare-image ${zoomClass}`} src={frame[which]?.url||''} alt={alt} draggable={false}/>
  const activeFrame=mode==='ab'?frame[ab]:frame.original

  return <main className="compare-page">
    <header className="compare-header"><h1>视频对比</h1>
      <div className="compare-pickers"><button className="compare-button" onClick={()=>void choose('original')}><FolderOpen size={15}/>选择原片</button><button className="compare-button primary" onClick={()=>void choose('exported')}><FolderOpen size={15}/>选择导出结果</button></div>
    </header>
    <div className="compare-filebar"><div><span>原片</span><strong title={original}>{pair?.original.name||original.split(/[\\/]/).pop()||'未选择'}</strong></div><div><span>导出结果</span><strong title={exported}>{pair?.exported.name||exported.split(/[\\/]/).pop()||'未选择'}</strong></div></div>
    <section className="compare-workspace">
      <div className={`compare-stage ${zoomClass}`} ref={splitStage} onPointerMove={pointerMove} onPointerUp={()=>{dragSplit.current=false}} onPointerCancel={()=>{dragSplit.current=false}}>
        {!pair?<div className="compare-empty"><Scan size={28}/><b>选择原片与导出文件</b></div>:<>
          {mode==='side'&&<div className="compare-side-grid"><figure><div className="compare-image-well">{image('original','原片实际帧')}</div></figure><figure><div className="compare-image-well">{image('exported','导出结果实际帧')}</div></figure></div>}
          {mode==='wipe'&&<div className="compare-wipe"><div className="compare-wipe-image">{image('exported','导出结果实际帧')}</div><div className="compare-wipe-image compare-wipe-original" style={{clipPath:`inset(0 ${100-split}% 0 0)`}}>{image('original','原片实际帧')}</div><span className="compare-side-tag left">原片</span><span className="compare-side-tag right">导出</span><button className="compare-divider" style={{left:`${split}%`}} aria-label="拖动对比滑块" onPointerDown={e=>{dragSplit.current=true;e.currentTarget.setPointerCapture(e.pointerId)}}><i/></button></div>}
          {mode==='ab'&&<div className="compare-ab"><div className="compare-ab-image">{activeFrame?.url?<img className={`compare-image ${zoomClass}`} src={activeFrame.url} alt={ab==='original'?'原片实际帧':'导出结果实际帧'} draggable={false}/>:null}</div><span className="compare-side-tag left">{ab==='original'?'原片':'导出结果'}</span><div className="compare-ab-switch"><button className={ab==='original'?'active':''} onClick={()=>setAb('original')}>A · 原片</button><button className={ab==='exported'?'active':''} onClick={()=>setAb('exported')}>B · 导出</button></div></div>}
          {pair.original.is_hdr||pair.exported.is_hdr?<div className="compare-hdr-note" title="双方使用一致的固定显示映射；当前显示链路未校准，不代表色彩精确 HDR 输出。">HDR 显示未校准</div>:null}
        </>}
      </div>
      <div className="compare-toolbar"><div className="compare-mode-tabs" role="group" aria-label="对比方式"><button className={mode==='side'?'active':''} onClick={()=>setMode('side')}>左右并排</button><button className={mode==='wipe'?'active':''} onClick={()=>setMode('wipe')}>中央滑块</button><button className={mode==='ab'?'active':''} onClick={()=>setMode('ab')}>A / B</button></div><div className="compare-zoom" role="group" aria-label="查看缩放"><button className={zoom==='fit'?'active':''} onClick={()=>setZoom('fit')}>适配</button><button className={zoom==='100'?'active':''} onClick={()=>setZoom('100')}>100%</button><button className={zoom==='200'?'active':''} onClick={()=>setZoom('200')}>200%</button></div></div>
    </section>
    <section className="compare-transport"><div className="compare-transport-buttons"><button className="compare-icon-button" aria-label="后退一帧" disabled={!pair||busy} onClick={()=>void step(-1)}><ChevronLeft/></button><button className="compare-play" aria-label={playing?'暂停':'播放'} disabled={!pair} onClick={()=>playing?pause():void play()}>{playing?<Pause size={17}/>:<Play size={17}/>}</button><button className="compare-icon-button" aria-label="前进一帧" disabled={!pair||busy} onClick={()=>void step(1)}><ChevronRight/></button></div><div className="compare-timeline"><div className="compare-time-labels"><span>{seconds(currentTime)}</span><span>{seconds(duration)}</span></div><input aria-label="同步时间轴" type="range" min="0" max={Math.max(duration,.001)} step="0.001" value={Math.min(currentTime,duration||0)} disabled={!pair} onChange={e=>{const v=Number(e.target.value);setCurrentTime(v);timeRef.current=v}} onPointerUp={e=>seek(Number((e.currentTarget as HTMLInputElement).value))} onKeyUp={e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key))seek(Number(e.currentTarget.value))}}/></div><select aria-label="播放速度" disabled={!pair} value={playbackRate} onChange={e=>{const rate=Number(e.target.value);setPlaybackRate(rate);if(audioRef.current)audioRef.current.playbackRate=rate}}><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1">1×</option><option value="1.5">1.5×</option><option value="2">2×</option></select><label className="compare-audio-choice"><Volume2 size={15}/><span>单音轨</span><select aria-label="播放音轨" value={audioSide} disabled={!pair} onChange={e=>setAudioSide(e.target.value as Side)}><option value="original" disabled={!pair?.original.audio}>原片</option><option value="exported" disabled={!pair?.exported.audio}>导出结果</option></select></label><audio ref={audioRef} preload="none" src={audioUrl||undefined} onEnded={audioEnded}/></section>
    <footer className="compare-footer"><span role="status" className={error?'error':''}>{error||(busy?'正在读取画面…':status)}</span>{pair&&<details className="compare-media-details"><summary>媒体信息</summary><div className="compare-media-panel"><h2>媒体信息</h2>{(['original','exported'] as Side[]).map(side=><section key={side}><h3>{side==='original'?'原片':'导出结果'}</h3><p>{mediaLabel(pair[side])}</p>{playbackInfo&&<><p>{playbackInfo[side].decoder==='nvdec'?'NVIDIA 硬件解码 · GPU 缩放':'软件解码'} · {playbackInfo[side].display_size.join(' × ')} · {playbackInfo.display_fps.toFixed(0)} fps 显示</p>{playbackInfo[side].fallback_reason&&<p>解码回退原因：{playbackInfo[side].fallback_reason}</p>}</>}<p>画面时间 {seconds(frame[side]?.actualTime??0)} · PTS {frame[side]?.actualPts??'—'}</p></section>)}<p>按相对媒体时间定位，使用实际帧时间戳。对齐差 {seconds(Math.abs((frame.original?.actualTime||0)-(frame.exported?.actualTime||0)))}</p>{pair.original.is_hdr||pair.exported.is_hdr?<p>HDR 使用双方一致的固定显示映射；显示链路未经校准。</p>:null}</div></details>}</footer>
  </main>
}
