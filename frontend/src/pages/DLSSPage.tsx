import React, { useState, useEffect, useRef, useCallback } from 'react'
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  RotateCcw,
  Maximize,
  Move,
  AlertTriangle,
} from 'lucide-react'
import { ParameterHelp } from '../components/ParameterHelp'
import { FLOW_QUALITY_LABELS } from '../services/parameterHelp'
import { QualityInspector } from '../components/QualityInspector'
import { RenderPipeline } from '../components/RenderPipeline'
import { renderStages, vsrOutputSize } from '../services/renderPipeline'
import { VideoInfo, DLSSSettings } from '../types'
import { saveSettings, wsUrl, fetchFlowProviders, fetchQueue, fetchJobDiagnostics } from '../services/api'
import { resolveWorkingDims } from '../services/workingScale'
import {
  HDR_LIMITATIONS, HDR_PAPER_WHITE_SCRGB, HDR_SETTING, HDR_PAPER_WHITE_SETTING,
  hdrPaperWhiteReason, hdrSettingsBlockers, isHdrRequested, paperWhiteLabel,
} from '../services/hdrExport'
import { activeLayerCount, chooseLayerScale, chooseReconstructionMode,
  chooseWorkingScale } from '../services/multiScaleSettings'

interface DLSSPageProps {
  video: VideoInfo | null
  settings: DLSSSettings
  onSettingsChange: (newSettings: DLSSSettings) => void
  currentFrame: number
  setCurrentFrame: (frame: number) => void
  onStatusChange?: (msg: string) => void
  isActive?: boolean
}

export const DLSSPage: React.FC<DLSSPageProps> = ({
  video,
  settings,
  onSettingsChange,
  currentFrame,
  setCurrentFrame,
  onStatusChange,
  isActive = true,
}) => {
  // View states
  const [view, setView] = useState<'dlss' | 'orig' | 'split' | 'dual' | 'flow'>('split')
  const [splitX, setSplitX] = useState<number>(0.5)
  const [zoom, setZoom] = useState<number>(1.0)
  const [pixelPreview, setPixelPreview] = useState(false)
  const [previewImageSize, setPreviewImageSize] = useState<{width:number;height:number}|null>(null)
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0.5, y: 0.5 })
  const [isPlaying, setIsPlaying] = useState<boolean>(false)
  const [loop, setLoop] = useState<boolean>(true)
  const [activeFrameUrl, setActiveFrameUrl] = useState<string>('')
  const [playerError, setPlayerError] = useState<string | null>(null)
  const [previewStats, setPreviewStats] = useState<any>(null)
  const [previewPending, setPreviewPending] = useState(false)
  const [isDraggingSplit, setIsDraggingSplit] = useState<boolean>(false)
  const [isPanning, setIsPanning] = useState<boolean>(false)
  const [panStart, setPanStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 })

  const canvasRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WebSocket | null>(null)
  // P1-10: desired playback state, separate from what the socket confirmed.
  // Lets onopen (re)send play after drops instead of freezing on "playing".
  const wantPlayingRef = useRef<boolean>(false)
  const lastPlayRef = useRef<Record<string, unknown> | null>(null)

  // Optical flow hardware probe state
  const [flowProviders, setFlowProviders] = useState<any[]>([])
  const [nvofProbe, setNvofProbe] = useState<any>(null)
  const [recentFusionJob, setRecentFusionJob] = useState<{name:string;status:string;report:any}|null>(null)
  const inspectedJobId = useRef('')

  useEffect(() => {
    fetchFlowProviders()
      .then((data) => {
        setFlowProviders(data.providers || [])
        setNvofProbe(data.nvof_probe || null)
      })
      .catch((err) => console.warn('Failed to fetch flow providers:', err))
  }, [])

  useEffect(() => {
    if (!isActive) return
    let stopped = false
    const pollLatestJob = async () => {
      try {
        const queue = await fetchQueue()
        if (stopped) return
        const latestTerminal = [...queue.jobs].reverse().find((job) =>
          ['completed', 'failed', 'cancelled', 'skipped'].includes(job.status))
        if (!latestTerminal || latestTerminal.id === inspectedJobId.current) return
        const body = await fetchJobDiagnostics(latestTerminal.id, true)
        if (stopped) return
        inspectedJobId.current = latestTerminal.id
        setRecentFusionJob({name:latestTerminal.name,status:latestTerminal.status,
          report:body?.job?.residual_backend_diagnostics ?? null})
      } catch { /* Diagnostics are optional; the requested-settings diagram remains available. */ }
    }
    void pollLatestJob()
    const timer = setInterval(() => { void pollLatestJob() }, 8000)
    return () => { stopped = true; clearInterval(timer) }
  }, [isActive])

  const requestRevisionRef = useRef(0)
  const lastSeekRef = useRef<Record<string, unknown> | null>(null)
  const displayedBlobRef = useRef('')
  const sendPlayer = useCallback((command: Record<string, unknown>) => {
    setPreviewPending(command.action !== 'pause')
    const bounds = canvasRef.current?.getBoundingClientRect()
    const edge = pixelPreview || Number(command.zoom ?? zoom) > 1 ? 0 : Math.max(512,
      Math.min(1920, Math.ceil(Math.max(bounds?.width || 1280, bounds?.height || 720) * devicePixelRatio)))
    const payload = {...command, request_id: ++requestRevisionRef.current,
      transport: 'binary', ack: true, display_edge: edge}
    if (command.action === 'seek') lastSeekRef.current = payload
    if (command.action === 'play') lastPlayRef.current = payload
    if (command.action === 'set_view' && lastPlayRef.current) lastPlayRef.current = {...lastPlayRef.current, ...payload}
    if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(JSON.stringify(payload))
  }, [zoom, pixelPreview])

  useEffect(() => {
    let ws: WebSocket | null = null, stopped = false, header: any = null
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined
    let latestSequence = 0, lastPresented = 0, raf = 0
    const transportTiming = new Map<number, number>()
    const connect = () => {
      if (stopped) return
      ws = new WebSocket(wsUrl('player')); ws.binaryType = 'blob'; wsRef.current = ws
      const socket = ws
      const present = async (data: any, jpeg?: Blob) => {
        if (data.request_id !== requestRevisionRef.current) return
        const sequence = ++latestSequence, received = performance.now()
        const url = jpeg ? URL.createObjectURL(jpeg) : data.image
        const image = new Image(); image.src = url
        try { await image.decode() } catch {
          if (jpeg) URL.revokeObjectURL(url)
          if (data.request_id === requestRevisionRef.current) setPlayerError('预览画面解码失败')
          if (data.request_id === requestRevisionRef.current) setPreviewPending(false)
          return
        }
        const decoded = performance.now()
        const stale = () => stopped || sequence !== latestSequence || data.request_id !== requestRevisionRef.current
        if (stale()) {if (jpeg) URL.revokeObjectURL(url); return}
        raf = requestAnimationFrame(() => {
          if (stale()) {if (jpeg) URL.revokeObjectURL(url); return}
          const previous = displayedBlobRef.current
          displayedBlobRef.current = jpeg ? url : ''
          setPlayerError(null); setCurrentFrame(data.frame); setActiveFrameUrl(url)
          // Presentation opportunity after React commits, not physical scanout.
          raf = requestAnimationFrame(() => {
            if (previous) URL.revokeObjectURL(previous)
            if (stale()) return
            const painted = performance.now()
            setPreviewPending(false)
            const metrics = {decode_ms: decoded-received, paint_wait_ms: painted-decoded,
              receive_to_present_ms: painted-received,
              presentation_interval_ms: lastPresented ? painted-lastPresented : 0}
            lastPresented = painted
            setPreviewStats({...data.preview, client_presentation: metrics,
              display_sequence: data.sequence, ws_send_ms: transportTiming.get(data.sequence) ?? null,
              presentation_fps: metrics.presentation_interval_ms ? 1000/metrics.presentation_interval_ms : null})
            transportTiming.delete(data.sequence)
            if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({action: 'presented', sequence: data.sequence, ...metrics}))
          })
        })
      }
      ws.onopen = () => {
        if (wantPlayingRef.current && lastPlayRef.current) {
          socket.send(JSON.stringify({action: 'play', ...lastPlayRef.current,
            transport: 'binary', ack: true, request_id: requestRevisionRef.current})); setIsPlaying(true)
        } else if (lastSeekRef.current) socket.send(JSON.stringify(lastSeekRef.current))
      }
      ws.onmessage = event => {
        if (event.data instanceof Blob) {
          const metadata = header; header = null
          if (metadata) void present(metadata, event.data)
          return
        }
        try {
          const data = JSON.parse(event.data)
          if (data.type === 'frame') {
            if (data.encoding === 'jpeg_binary') header = data
            else void present(data)
          } else if (data.request_id === requestRevisionRef.current && data.type === 'frame_timing') {
            transportTiming.set(data.sequence, data.ws_send_ms)
            if (transportTiming.size > 8) transportTiming.delete(transportTiming.keys().next().value!)
            setPreviewStats((previous: any) => previous?.display_sequence === data.sequence
              ? {...previous, ws_send_ms: data.ws_send_ms} : previous)
          } else if (data.request_id === requestRevisionRef.current && data.type === 'error') {
            setPreviewPending(false)
            wantPlayingRef.current = false; setPlayerError(data.message); setIsPlaying(false); onStatusChange?.(`预览失败：${data.message}`)
          } else if (data.request_id === requestRevisionRef.current && data.type === 'ended') {
            wantPlayingRef.current = false; setIsPlaying(false); onStatusChange?.('播放结束')
          }
        } catch {setPlayerError('预览连接返回了无效数据')}
      }
      ws.onclose = () => {
        if (wsRef.current === socket) wsRef.current = null
        if (!stopped) {setIsPlaying(false); reconnectTimer = setTimeout(connect, 1500)}
      }
      ws.onerror = () => socket.close()
    }
    connect()
    return () => {
      stopped = true; ++latestSequence; clearTimeout(reconnectTimer); cancelAnimationFrame(raf); ws?.close()
      if (displayedBlobRef.current) URL.revokeObjectURL(displayedBlobRef.current)
      displayedBlobRef.current = ''
    }
  }, [setCurrentFrame, onStatusChange])

  const currentFrameRef = useRef(currentFrame)
  // NOTE(oxlint baseline react(refs)): intentional latest-value ref so async
  // socket callbacks always see the current frame without re-subscribing.
  currentFrameRef.current = currentFrame

  const requestFrame = useCallback(
    (
      targetFrame?: number,
      targetView?: string,
      targetSplit?: number,
      targetZoom?: number,
      targetPan?: { x: number; y: number }
    ) => {
      if (!video) return
      const f = targetFrame !== undefined ? targetFrame : currentFrameRef.current
      const v = targetView !== undefined ? targetView : view
      const sx = targetSplit !== undefined ? targetSplit : splitX
      const zm = targetZoom !== undefined ? targetZoom : zoom
      const p = targetPan !== undefined ? targetPan : pan

      sendPlayer({action: 'seek', frame: f, view: v, split_x: sx, zoom: zm, pan_x: p.x, pan_y: p.y})

    },
    [video, view, splitX, zoom, pan, sendPlayer]
  )

  useEffect(() => {
    const pending = () => {++requestRevisionRef.current; setPlayerError(null); setPreviewPending(true)}
    const settled = () => {
      if (wantPlayingRef.current) sendPlayer({action: 'set_view'})
      else requestFrame(currentFrameRef.current)
    }
    window.addEventListener('lunarc:settings-pending', pending)
    window.addEventListener('lunarc:settings-settled', settled)
    return () => {
      window.removeEventListener('lunarc:settings-pending', pending)
      window.removeEventListener('lunarc:settings-settled', settled)
    }
  }, [requestFrame, sendPlayer])

  useEffect(() => {
    clearTimeout(seekTimerRef.current)
    pendingSeekFrameRef.current = null
    wantPlayingRef.current = false
    lastPlayRef.current = null
    setIsPlaying(false)
    setActiveFrameUrl('')
    setPreviewStats(null)
    setPreviewPending(false)
  }, [video?.path])

  // Update view/zoom/split dynamically
  useEffect(() => {
    if (!video) return
    if (isPlaying) {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        sendPlayer({
            action: 'set_view',
            view,
            split_x: splitX,
            zoom,
            pan_x: pan.x,
            pan_y: pan.y,
          })
      }
    } else {
      requestFrame(currentFrameRef.current, view, splitX, zoom, pan)
    }
  }, [video, isPlaying, requestFrame, view, splitX, zoom, pan, sendPlayer])

  // Play / Pause handling (P1-08 sends loop; P1-10 never claims playing
  // without an open socket — disconnected play becomes buffered intent)
  const togglePlay = () => {
    if (!video) return
    const nextPlaying = !isPlaying
    wantPlayingRef.current = nextPlaying

    if (nextPlaying) {
      // Record intent first so a later (re)connect can replay it (P1-10).
      lastPlayRef.current = {
        frame: currentFrameRef.current,
        fps: video.fps || 30,
        view,
        split_x: splitX,
        zoom,
        pan_x: pan.x,
        pan_y: pan.y,
        loop,
      }
    } else {
      lastPlayRef.current = null
    }

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      if (nextPlaying) {
        sendPlayer({ action: 'play', ...lastPlayRef.current })
        setIsPlaying(true)
        onStatusChange?.('正在播放')
      } else {
        sendPlayer({ action: 'pause' })
        setIsPlaying(false)
        onStatusChange?.('播放已暂停')
      }
    } else {
      setIsPlaying(false)
      onStatusChange?.(nextPlaying ? '等待连接，后台就绪后自动开始…' : '播放已暂停')
    }
  }

  const lastSeekTimeRef = useRef<number>(0)
  const pendingSeekFrameRef = useRef<number | null>(null)
  const seekTimerRef = useRef<any>(null)

  // NOTE(oxlint baseline react(set-state-in-effect)): intentional sync —
  // leaving the tab must immediately stop playback and its socket stream.
  useEffect(() => {
    if (isActive === false && isPlaying) {
      setIsPlaying(false)
      wantPlayingRef.current = false
      lastPlayRef.current = null
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        try {
          sendPlayer({ action: 'pause' })
        } catch {}
      }
      onStatusChange?.('播放已暂停')
    }
  }, [isActive, isPlaying, onStatusChange])

  useEffect(() => {
    return () => {
      clearTimeout(seekTimerRef.current)
      if (isPlaying) {
        onStatusChange?.('播放已暂停')
      }
    }
  }, [isPlaying, onStatusChange])

  const handleSeek = (frame: number) => {
    const total = video?.frames || 1
    const clamped = Math.max(0, Math.min(total - 1, frame))
    setCurrentFrame(clamped)
    if (isPlaying) {
      setIsPlaying(false)
      wantPlayingRef.current = false
      lastPlayRef.current = null
    }

    const now = Date.now()
    if (now - lastSeekTimeRef.current > 50) {
      lastSeekTimeRef.current = now
      clearTimeout(seekTimerRef.current)
      pendingSeekFrameRef.current = null
      requestFrame(clamped)
    } else {
      pendingSeekFrameRef.current = clamped
      clearTimeout(seekTimerRef.current)
      seekTimerRef.current = setTimeout(() => {
        if (pendingSeekFrameRef.current !== null) {
          requestFrame(pendingSeekFrameRef.current)
          pendingSeekFrameRef.current = null
        }
      }, 50)
    }
  }

  // Parameter update helper
  const handleParamChange = async (key: keyof DLSSSettings, val: any) => {
    const patch: Partial<DLSSSettings> = { [key]: val }
    const updated = { ...settings, ...patch }
    onSettingsChange(updated)
    try {
      await saveSettings(patch)
      onStatusChange?.('参数已保存')

    } catch (e) {
      onSettingsChange(settings)
      onStatusChange?.(`设置未保存：${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const handleParamsBatchChange = async (newParams: Partial<DLSSSettings>) => {
    const updated = { ...settings, ...newParams }
    onSettingsChange(updated)
    try {
      await saveSettings(newParams)
      onStatusChange?.('参数已更新')

    } catch (e) {
      onSettingsChange(settings)
      onStatusChange?.(`设置未保存：${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const handleWorkingScaleChange = (scale: number) => {
    handleParamsBatchChange(chooseWorkingScale(settings, scale))
  }

  // The experimental HDR pair is written through its own helper because a refusal has
  // to be visible: the backend is the authority on both values, and a panel that had
  // already moved its own toggle would be showing a setting that was never stored.
  const saveHdrPatch = async (patch: Partial<DLSSSettings>) => {
    onSettingsChange({ ...settings, ...patch })
    try {
      await saveSettings(patch)
    } catch (e: any) {
      onSettingsChange(settings)
      onStatusChange?.(`实验性 HDR 设置被后端拒绝：${e?.message ?? String(e)}`)
    }
  }

  // Paper white is typed as text so a half-written value ("8.", "0.5", "abc") can be
  // shown and refused as it is being written, instead of being snapped to a number the
  // operator never typed. Only a value that passes the backend's own predicate is sent.
  const [paperWhiteDraft, setPaperWhiteDraft] = useState<string | null>(null)
  const hdrOn = isHdrRequested(settings)
  const savedPaperWhite = settings.hdr_paper_white_scrgb ?? HDR_PAPER_WHITE_SCRGB
  const paperWhiteInput = paperWhiteDraft ?? String(savedPaperWhite)
  const paperWhiteRefusal = hdrPaperWhiteReason(
    paperWhiteDraft === null ? savedPaperWhite : paperWhiteDraft)
  const handlePaperWhiteChange = (raw: string) => {
    setPaperWhiteDraft(raw)
    if (!hdrPaperWhiteReason(raw)) {
      saveHdrPatch({ [HDR_PAPER_WHITE_SETTING]: Number(raw) })
    }
  }
  const hdrRefusals = hdrOn ? hdrSettingsBlockers(settings, video) : []
  const handleLayerScaleChange = (index: number, scale: number) => {
    try {
      handleParamChange('layer_scales', chooseLayerScale(settings, index, scale))
    } catch (error) {
      onStatusChange?.(String(error))
    }
  }

  const handleLayerWeightChange = (index: number, weight: number) => {
    const count = activeLayerCount(settings.reconstruction_mode || 'single')
    const weights = Array.from({ length: count }, (_, at) =>
      at === index ? weight : (settings.layer_weights?.[at] || 1))
    handleParamChange('layer_weights', weights)
  }

  const handleEngineChange = (eng: string) => {
    if (isHdrRequested(settings) && Number(settings.guidance_mode ?? 0) === 1 && !['nvof','fidelityfx','raft'].includes(eng)) {
      // Refused out loud rather than saved and met at the first frame: a CPU flow
      // engine would have to download the plane to compute motion, which is exactly
      // what this chain does not do.
      onStatusChange?.('HDR 光流引导需要可用的 NVIDIA / AMD / RAFT CUDA 设备路径')
      return
    }
    let defaultQ = '1F'
    if (eng === 'dis') defaultQ = 'Medium'
    else if (eng === 'raft') defaultQ = 'Balanced'
    else if (eng === 'fidelityfx') defaultQ = 'Quality'
    else if (eng === 'nvof') defaultQ = '1F'
    handleParamsBatchChange({ flow_engine: eng, flow_quality: defaultQ, ...(eng==='fidelityfx'?{export_backend:'gpu_transfer' as const}:{}) })
  }

  const nvofProvider = flowProviders.find((p) => p.engine === 'nvof')
  const nvofGpuName = nvofProvider?.device || nvofProbe?.device || 'NVIDIA GPU'
  const displayGpu = nvofGpuName.replace(/^NVIDIA\s+(GeForce\s+)?/i, '')
  const isNvofAvailable = nvofProvider?.available ?? nvofProbe?.available ?? false
  const nvofGrids = nvofProvider?.grids && nvofProvider.grids.length > 0
    ? nvofProvider.grids
    : [
        { id: 1, label: '1 × 1 网格' },
        { id: 2, label: '2 × 2 网格' },
        { id: 4, label: '4 × 4 网格' },
      ]
  const nvofPresets = nvofProvider?.presets && nvofProvider.presets.length > 0
    ? nvofProvider.presets
    : [
        { id: 'fast', label: 'FAST - 极速 (推荐)' },
        { id: 'medium', label: 'MEDIUM - 平衡' },
        { id: 'slow', label: 'SLOW - 质量优先' },
      ]

  const nvofGrid = settings.flow_quality.startsWith('2') ? '2' : settings.flow_quality.startsWith('4') ? '4' : '1'
  const nvofPreset = settings.flow_quality.endsWith('S') ? 'SLOW' : settings.flow_quality.endsWith('M') ? 'MEDIUM' : 'FAST'

  const handleNvofGridChange = (newGrid: string) => {
    const pChar = nvofPreset === 'SLOW' ? 'S' : nvofPreset === 'MEDIUM' ? 'M' : 'F'
    handleParamChange('flow_quality', `${newGrid}${pChar}`)
  }

  const handleNvofPresetChange = (newPreset: string) => {
    const pChar = newPreset === 'SLOW' ? 'S' : newPreset === 'MEDIUM' ? 'M' : 'F'
    handleParamChange('flow_quality', `${nvofGrid}${pChar}`)
  }

  // Split Wipe mouse drag
  const handleMouseDown = (e: React.MouseEvent) => {
    if (view === 'split' && e.button === 0) {
      setIsDraggingSplit(true)
      handleSplitMove(e)
    } else if (e.button === 2 || (e.button === 0 && zoom > 1.0)) {
      setIsPanning(true)
      setPanStart({ x: e.clientX, y: e.clientY })
    }
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDraggingSplit && canvasRef.current) {
      handleSplitMove(e)
    } else if (isPanning && zoom > 1.0) {
      const dx = (e.clientX - panStart.x) / (canvasRef.current?.clientWidth || 1)
      const dy = (e.clientY - panStart.y) / (canvasRef.current?.clientHeight || 1)
      setPan((prev) => ({
        x: Math.max(0, Math.min(1, prev.x - dx)),
        y: Math.max(0, Math.min(1, prev.y - dy)),
      }))
      setPanStart({ x: e.clientX, y: e.clientY })
    }
  }

  const handleSplitMove = (e: React.MouseEvent) => {
    if (!canvasRef.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const frac = (e.clientX - rect.left) / rect.width
    setSplitX(Math.max(0.01, Math.min(0.99, frac)))
  }

  const handleMouseUp = () => {
    setIsDraggingSplit(false)
    setIsPanning(false)
  }

  return (
    <div className="dlss-workbench select-none">
      <main className="dlss-view-column">
        <div className="dlss-view-toolbar">
          <div className="dlss-view-modes" role="group" aria-label="预览视图">{([
            {id:'split',label:'对比'},{id:'dual',label:'双画面'},{id:'dlss',label:'增强'},{id:'orig',label:'原图'},{id:'flow',label:'光流'},
          ] as const).map(item=><button key={item.id} type="button" aria-pressed={view===item.id} onClick={()=>setView(item.id)}>{item.label}</button>)}</div>
          <div className="dlss-zoom-tools">
            <select aria-label="预览缩放" value={pixelPreview?'native':zoom} onChange={e=>{if(e.target.value==='native')return;setPixelPreview(false);setZoom(Number(e.target.value));setPan({x:.5,y:.5})}}>
              {pixelPreview&&<option value="native">原像素</option>}<option value={1}>适配</option><option value={.5}>缩小</option><option value={2}>放大 2×</option><option value={3}>放大 3×</option>
            </select>
            <button type="button" className="icon-button" aria-label="按输出原始像素查看" title="按输出原始像素查看 VSR 细节" onClick={()=>{
              const bounds=canvasRef.current?.getBoundingClientRect()
              if (!video || !bounds) return
              const size=view!=='orig'&&renderStages(settings).includes('vsr')?vsrOutputSize(settings,video):video
              setPixelPreview(true)
              setZoom(Math.max(1,size.width*(view==='dual'?2:1)/(bounds.width*devicePixelRatio),size.height/(bounds.height*devicePixelRatio)))
              setPan({x:.5,y:.5})
            }}>1:1</button>
            <button className="icon-button" title="全屏预览" aria-label="全屏预览" onClick={()=>{const action=document.fullscreenElement?document.exitFullscreen():canvasRef.current?.requestFullscreen();action?.catch(e=>onStatusChange?.(`无法全屏：${e.message}`))}}><Maximize size={17}/></button>
          </div>
        </div>
        <section ref={canvasRef} className="dlss-monitor" aria-label="视频预览画布" title="实际处理层数和模式显示在画面中" onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onContextMenu={event=>event.preventDefault()}>

          {playerError&&<div className="dlss-player-error" role="alert"><AlertTriangle size={15}/><span>{playerError}</span><button type="button" aria-label="关闭错误" onClick={()=>setPlayerError(null)}>×</button></div>}
          {activeFrameUrl?<img src={activeFrameUrl} alt="当前预览帧" className="dlss-preview-image" draggable={false}
            onLoad={event=>setPreviewImageSize({width:event.currentTarget.naturalWidth,height:event.currentTarget.naturalHeight})}
            style={pixelPreview&&previewImageSize?{width:previewImageSize.width/devicePixelRatio,height:previewImageSize.height/devicePixelRatio}:undefined}/>:<div className="dlss-empty-preview"><div className="empty-glyph"><span/><span/><span/></div><strong>{video?'正在准备预览…':'导入素材，开始重建。'}</strong></div>}
          {view==='split'&&activeFrameUrl&&<div className="split-marker" aria-hidden="true" style={{left:`${splitX*100}%`}}/>}
          {zoom>1.01&&<div className="pan-hint"><Move size={13}/>右键拖动画布以查看细节</div>}
          {video?.is_hdr&&<div className="preview-warning"><AlertTriangle size={13}/>HDR 显示用映射 · 未校准</div>}
          {previewPending && activeFrameUrl && <div className="preview-execution" role="status">更新预览…</div>}
          {!previewPending && previewStats?.layer_count && <div className="preview-execution" aria-label="预览实际执行" title={`实际尺寸 ${previewStats.preview_resolution?.join(' × ')} · 处理 ${Number(previewStats.processing_fps).toFixed(1)} FPS · 播放 ${Number(previewStats.presentation_fps||0).toFixed(1)} FPS · 跳过 ${previewStats.dropped_frames||0} 帧${previewStats.solver?` · GPU 融合 ${previewStats.solver.converged===false?'未收敛':'已执行'}`:''}`}>
            {previewStats.render_pipeline?.includes('nr') ? `${previewStats.layer_count} 层` : previewStats.render_pipeline?.includes('vsr') ? 'VSR' : previewStats.render_pipeline?.includes('dlaa') ? 'DLAA' : '原帧预览'}
            {previewStats.render_pipeline?.includes('vsr') && <>{(() => {
              const stages=[...(previewStats.native_input_effect?.stages||[]),...(previewStats.native_output_effect?.stages||[])]
              const vsr=stages.find((stage:any)=>stage.stage==='vsr')
              return vsr?.counters?.model_loaded ? ` · ${['低','中','高','极高'][vsr.counters.quality_level-1]} · ${vsr.output_size.join('×')}` : ''
            })()}</>}
            {isPlaying && <> · {Number(previewStats.presentation_fps || previewStats.processing_fps).toFixed(1)} fps</>}
            {previewStats.solver?.converged === false && <> · 未收敛</>}
          </div>}

        </section>
        <div className="dlss-material"><strong title={video?.path}>{video?.filename||'未载入素材'}</strong><span>{video?`${video.width} × ${video.height} · ${video.is_hdr?'HDR10':'SDR'}`:''}</span></div>
        <div className="dlss-transport">
          <div className="transport-scrub"><span>{String(video?currentFrame+1:0).padStart(3,'0')}</span><input aria-label="时间轴帧位置" type="range" min={0} max={Math.max(1,(video?.frames||1)-1)} value={currentFrame} onChange={event=>handleSeek(Number(event.target.value))}/><span>{String(video?.frames||0).padStart(3,'0')}</span></div>
          <div className="transport-buttons"><button type="button" aria-label="跳至首帧" onClick={()=>handleSeek(0)}><SkipBack size={16}/></button><button type="button" aria-label="后退一帧" onClick={()=>handleSeek(currentFrame-1)}>−1</button><button type="button" className="transport-play" onClick={togglePlay} aria-label={isPlaying?'暂停播放':'开始播放'}>{isPlaying?<Pause size={16}/>:<Play size={16} />}</button><button type="button" aria-label="前进一帧" onClick={()=>handleSeek(currentFrame+1)}>+1</button><button type="button" aria-label="跳至末帧" onClick={()=>handleSeek((video?.frames||1)-1)}><SkipForward size={16}/></button><button type="button" aria-pressed={loop} aria-label="循环播放" onClick={()=>setLoop(!loop)}><RotateCcw size={15}/></button></div>
          <div className="transport-meta">{video?`${video.fps||30} fps · ${video.duration_seconds.toFixed(2)} s`:'—'}</div>
        </div>
      </main>
      <QualityInspector settings={settings} onSettingsChange={onSettingsChange} onStatusChange={onStatusChange}
        primary={<>
          <RenderPipeline settings={settings} onChange={handleParamsBatchChange}/>
          {renderStages(settings).includes('nr')&&<>
          <div className="style-switch" role="group" aria-label="画面风格">
            {([{value:0,label:'默认'},{value:2,label:'电影'},{value:1,label:'自然'}] as const).map(item=><button type="button" key={item.value} aria-pressed={settings.style===item.value} onClick={()=>handleParamChange('style',item.value)}>{item.label}</button>)}
          </div>
          {([['intensity','NR 强度'],['local_tone','局部色调强度'],['local_struct','局部结构强度'],['skin_struct','皮肤结构强度']] as const).map(([key,label])=><label className="primary-slider" key={key}><span>{label}<ParameterHelp name={key} label={label} /><output>{Number(settings[key].toFixed(2))}</output></span><input aria-label={label} type="range" min={0} max={2} step={.05} value={settings[key]} style={{'--range-fill':`${settings[key]/2*100}%`} as React.CSSProperties} onChange={e=>handleParamChange(key,Number(e.target.value))}/></label>)}
          <label className="primary-check"><input type="checkbox" aria-label="自动遮罩" checked={settings.use_auto_mask===1} onChange={e=>handleParamChange('use_auto_mask',e.target.checked?1:0)}/>自动遮罩<ParameterHelp name="use_auto_mask" label="自动遮罩" /></label>
          </>}
        </>}
        basic={<div className="dlss-inspector-content">
          <section className="dlss-control-card"><h3>运行库</h3>
            <p className="dlss-note">默认自动查找本机运行库。留空即可。</p>
            {(renderStages(settings).includes('vsr')||(settings.frame_generation && settings.frame_generation!=='off'))&&<label className="dlss-control-label">VFX 目录覆盖<input aria-label="VFX运行库目录" placeholder="自动查找" value={settings.vsr_runtime_path||''} onChange={e=>handleParamChange('vsr_runtime_path',e.target.value)}/></label>}
            {renderStages(settings).includes('dlaa')&&<label className="dlss-control-label">DLSS DLL 覆盖<input aria-label="DLSS DLL路径" placeholder="自动查找" value={settings.dlss_dll_path||''} onChange={e=>handleParamChange('dlss_dll_path',e.target.value)}/></label>}
            {renderStages(settings).includes('dlaa')&&<p className="dlss-note">DLAA 支持 SDR 8/10-bit 与 HDR10；视频输入的深度和抖动为零。</p>}
            {renderStages(settings).includes('vsr')&&<p className="dlss-note">VSR 10-bit / HDR 使用 VFX 1.3 RGB10A2，运行库自动检测。</p>}
          </section>
          <section className="dlss-control-card"><div className="dlss-card-head"><div><h3>神经重建</h3></div></div>
            <label className="dlss-control-label">画面风格<select aria-label="画面风格" value={settings.style} onChange={e=>handleParamChange('style',Number(e.target.value))}><option value={0}>默认</option><option value={1}>自然</option><option value={2}>电影</option></select></label>
            <label className="dlss-check-row"><input type="checkbox" checked={settings.ui_correction===1} onChange={e=>handleParamChange('ui_correction',e.target.checked?1:0)}/>NR 界面修正<ParameterHelp name="ui_correction" label="NR 界面修正" /></label>
          </section>
          <section className="dlss-control-card"><div className="dlss-card-head"><div><h3>引导与光流</h3></div><span className="dlss-chip">{settings.flow_engine.toUpperCase()}</span></div>
            <label className="dlss-control-label">引导模式<select value={settings.guidance_mode} onChange={e=>handleParamChange('guidance_mode',Number(e.target.value))}><option value={0}>关闭 · 零运动入口</option><option value={1}>启用 · 光流引导</option></select></label>
            {settings.guidance_mode===1&&<><label className="dlss-control-label">光流引擎<select value={settings.flow_engine} onChange={e=>handleEngineChange(e.target.value)}><option value="nvof">NVIDIA Optical Flow</option><option value="fidelityfx">AMD FidelityFX Optical Flow</option><option value="dis">OpenCV DIS</option><option value="raft">RAFT-Large</option></select></label>
              {settings.flow_engine==='fidelityfx'?<label className="dlss-control-label"><span>引擎质量<ParameterHelp name="flow_quality_amd" label="光流质量" /></span><select aria-label="AMD 光流质量" value={settings.flow_quality==='Performance'?'Performance':'Quality'} onChange={e=>handleParamChange('flow_quality',e.target.value)}><option value="Performance">性能</option><option value="Quality">质量</option></select></label>:settings.flow_engine==='nvof'?<div className="dlss-control-grid"><label>网格<select value={nvofGrid} onChange={e=>handleNvofGridChange(e.target.value)}>{nvofGrids.map((g:any)=><option key={g.id} value={String(g.id)}>{g.id} × {g.id} 网格</option>)}</select></label><label>预设<select value={nvofPreset} onChange={e=>handleNvofPresetChange(e.target.value)}>{nvofPresets.map((p:any)=><option key={p.id} value={p.id.toUpperCase()}>{FLOW_QUALITY_LABELS[p.id.toUpperCase()] || p.label}</option>)}</select></label></div>:<label className="dlss-control-label"><span>引擎质量<ParameterHelp name={settings.flow_engine==='raft'?'flow_quality_raft':'flow_quality_dis'} label="光流质量" /></span><select value={settings.flow_quality} onChange={e=>handleParamChange('flow_quality',e.target.value)}>{(settings.flow_engine==='dis'?['Fast','Medium','Slow']:['Fast','Balanced','Quality']).map(v=><option key={v} value={v}>{FLOW_QUALITY_LABELS[v] || v}</option>)}</select></label>}
              {settings.flow_engine==='raft' && <div className="dlss-control-grid">
                <label><span>分析长边<ParameterHelp name="raft_analysis_edge" label="分析长边" /></span><select aria-label="RAFT 分析长边" value={settings.raft_analysis_edge ?? (settings.flow_quality==='Fast'?512:settings.flow_quality==='Quality'?1080:720)} onChange={e=>handleParamChange('raft_analysis_edge',Number(e.target.value))}>{[512,720,1080].map(v=><option key={v} value={v}>{v}</option>)}</select></label>
                <label><span>迭代次数<ParameterHelp name="raft_iterations" label="迭代次数" /></span><input aria-label="RAFT 迭代次数" type="number" min={1} max={24} step={1} value={settings.raft_iterations ?? (settings.flow_quality==='Fast'?4:settings.flow_quality==='Quality'?12:6)} onChange={e=>{const v=Number(e.target.value);if(Number.isInteger(v)&&v>=1&&v<=24)handleParamChange('raft_iterations',v)}} /></label>
              </div>}
              <div className="dlss-control-grid">{(['motion_scale_x','motion_scale_y'] as const).map((key,i)=><label key={key}>运动 {i?'Y':'X'} · {settings[key].toFixed(2)}<input type="range" min={0} max={4} step={.05} value={settings[key]} onChange={e=>handleParamChange(key,Number(e.target.value))}/></label>)}</div>
              {settings.flow_engine==='nvof'&&<div className="provider-summary"><span className={isNvofAvailable?'provider-state-ready':'provider-state-off'}/><span>{isNvofAvailable?`${displayGpu} · 设备可用`:nvofProvider?.error||'NVOF 设备不可用'}</span></div>}
            </>}
          </section>
          <section className="dlss-control-card"><div className="dlss-card-head"><div><h3>导出后端</h3></div><span className="dlss-chip dlss-chip-export">仅导出</span></div><label className="dlss-control-label">执行链路<select aria-label="导出后端" value={settings.export_backend} onChange={e=>{if(e.target.value!=='gpu_transfer'&&(settings.reconstruction_mode||'single')!=='single'){onStatusChange?.('Dual / Triple 只支持 GPU Transfer；请先切回 Single');return}if(e.target.value!=='gpu_transfer'&&hdrOn){onStatusChange?.('实验性 HDR/scRGB 只在 GPU Transfer 上运行；请先关闭该开关');return}handleParamChange('export_backend',e.target.value)}}><option value="auto">Auto · Legacy 路径</option><option value="legacy">Legacy FFmpeg</option><option value="gpu_transfer">GPU Transfer · 条件不符则拒绝</option></select></label><p className="dlss-note">原生预览复用 GPU Transfer 处理核心；实际层数和处理帧率显示在画面中。</p><details className="dlss-note"><summary>支持条件与预览范围</summary><p>单层预览与所有导出链使用同一 GPU 宿主。关闭引导使用零运动纹理入口；开启光流需所选 NVIDIA、AMD 或 RAFT CUDA 引导后端可用。Dual / Triple 在原生预览和 GPU Transfer 导出中使用独立会话。缺少必要入口或输入条件不符时明确拒绝，不会换用其他宿主。旧 host-pointer 配置已移除，载入时须迁移到当前 GPU 宿主。</p></details></section>
        </div>}
        layers={<div className="dlss-inspector-content">
          <section className="dlss-control-card"><div className="dlss-card-head"><div><h3>工作分辨率</h3></div><span className="dlss-chip">{Math.round(settings.working_scale*100)}%</span></div><label className="dlss-slider"><span>工作缩放<output>{Math.round(settings.working_scale*100)}%</output></span><input aria-label="工作分辨率缩放" type="range" min={25} max={100} step={1} value={Math.round(settings.working_scale*100)} onChange={e=>handleWorkingScaleChange(Number(e.target.value)/100)}/></label><div className="scale-presets">{[1,.75,.67,.63,.5,.25].map(s=><button type="button" key={s} aria-pressed={Math.abs(settings.working_scale-s)<.005} onClick={()=>handleWorkingScaleChange(s)}>{Math.round(s*100)}%</button>)}</div>{(()=>{const [w,h]=resolveWorkingDims(video?.width||3840,video?.height||2160,settings.working_scale);return <div className="dimension-row"><span>源 {video?.width||3840} × {video?.height||2160}</span><strong>请求 {w} × {h}</strong></div>})()}</section>
          <p className="dlss-note">工作缩放支持 25%–100% 的整数百分比；显示尺寸由共享解析器按 GPU 对齐规则计算。</p><section className="dlss-control-card"><div className="dlss-card-head"><div><h3>残差重构</h3></div><span className="dlss-chip">{settings.residual_pyramid?'已请求':'关闭'}</span></div><label className="dlss-check-row"><input type="checkbox" checked={settings.residual_pyramid} onChange={e=>handleParamChange('residual_pyramid',e.target.checked)}/>保留源细节残差</label><label className="dlss-slider"><span>残差倍率<output>{settings.residual_strength.toFixed(2)}×</output></span><input aria-label="残差倍率" type="range" min={0} max={1.5} step={.05} value={settings.residual_strength} onChange={e=>handleParamChange('residual_strength',Number(e.target.value))}/></label></section>
          <section className="dlss-control-card"><div className="dlss-card-head"><div><h3>尺度与融合</h3></div><span className="dlss-chip">导出请求</span></div><label className="dlss-control-label">重建层数<select value={settings.reconstruction_mode||'single'} onChange={e=>handleParamsBatchChange(chooseReconstructionMode(settings,e.target.value as DLSSSettings['reconstruction_mode']))}><option value="single">单层</option><option value="dual">双层</option><option value="triple">三层</option></select></label>
            <div className="layer-diagram" aria-label="请求的独立层尺寸与融合权重">{Array.from({length:activeLayerCount(settings.reconstruction_mode||'single')},(_,i)=>{const scale=i===0?settings.working_scale:settings.layer_scales?.[i]??settings.working_scale;const [w,h]=resolveWorkingDims(video?.width||3840,video?.height||2160,scale);return <div className="layer-node" key={i}><span className="layer-index">L{i+1}</span><div><strong>{w} × {h}</strong><small>{Math.round(scale*100)}% · weight {(settings.layer_weights?.[i]||1).toFixed(2)}</small></div>{i<activeLayerCount(settings.reconstruction_mode||'single')-1&&<span className="layer-link">→</span>}</div>})}</div><p className="dlss-note">以上为当前请求值；下面的最近任务实际状态来自该任务的后端诊断，不代表实时预览。</p>
            <div className="dlss-control-card"><div className="dlss-card-head"><div><h3>最近任务实际</h3></div><span className="dlss-chip">{recentFusionJob?.status||'暂无报告'}</span></div>{recentFusionJob?<><p className="dlss-note">{recentFusionJob.name}</p>{recentFusionJob.report?.fusion_mode&&<div className="dimension-row"><span>后端报告模式</span><strong>{recentFusionJob.report.fusion_mode}</strong></div>}{Array.isArray(recentFusionJob.report?.layer_working_dims)&&<div className="dimension-row"><span>实际层尺寸</span><strong>{recentFusionJob.report.layer_working_dims.map((d:number[])=>`${d[0]}×${d[1]}`).join(' / ')}</strong></div>}{Array.isArray(recentFusionJob.report?.layer_dlss_calls)&&<div className="dimension-row"><span>各层 DLSS 调用数</span><strong>{recentFusionJob.report.layer_dlss_calls.join(' / ')}</strong></div>}{Array.isArray(recentFusionJob.report?.fusion_solver_frames)&&recentFusionJob.report.fusion_solver_frames.length>0?<div className="dimension-row"><span>CG 求解帧数 / 收敛</span><strong>{recentFusionJob.report.fusion_solver_frames.length} / {recentFusionJob.report.fusion_solver_frames.every((r:{converged?:boolean})=>r.converged)?'全部':'部分或无'}</strong></div>:recentFusionJob.report?.fusion_solver_frames?.details_omitted?<div className="dimension-row"><span>CG 求解帧数 / 收敛</span><strong>{recentFusionJob.report.fusion_solver_frames.item_count} / {recentFusionJob.report.fusion_solver_frames.all_converged?"全部":"部分或无"}</strong></div>:<p className="dlss-note">该任务诊断未包含 CG 求解帧记录；不推断实际求解状态。</p>}</>:<p className="dlss-note">尚无可读取的终态任务诊断。导出任务完成后，此处读取最新任务报告。</p>}</div>
            {(settings.reconstruction_mode||'single')!=='single'&&Array.from({length:activeLayerCount(settings.reconstruction_mode)},(_,i)=>{const scale=i===0?settings.working_scale:settings.layer_scales?.[i]??settings.working_scale;return <div className="layer-inputs" key={i}><strong>第 {i+1} 层 · {Math.round(scale*100)}%</strong>{i>0&&<label>独立输入比例<select value={Math.round(scale*100)} onChange={e=>handleLayerScaleChange(i,Number(e.target.value)/100)}>{Array.from({length:76},(_,at)=>at+25).map(v=><option key={v} value={v} disabled={settings.layer_scales?.some((item,at)=>at!==i&&Math.round(item*100)===v)||Math.round(settings.working_scale*100)===v}>{v}%</option>)}</select></label>}<label className="dlss-slider"><span>观测权重<output>{(settings.layer_weights?.[i]||1).toFixed(2)}</output></span><input aria-label={`第 ${i+1} 层观测权重`} type="range" min={.05} max={3} step={.05} value={settings.layer_weights?.[i]||1} onChange={e=>handleLayerWeightChange(i,Number(e.target.value))}/></label></div>})}
            {(settings.reconstruction_mode||'single')!=='single'&&<><label className="dlss-control-label">融合输出<select value={settings.fusion_mode||'catmull_baseline'} onChange={e=>handleParamChange('fusion_mode',e.target.value)}><option value="catmull_baseline">Catmull 基线</option><option value="data_consistency_experimental">数据一致性 · 实验性 GPU CG</option></select></label>{settings.fusion_mode==='data_consistency_experimental'&&<details className="solver-details"><summary>GPU CG 求解器参数</summary><div className="dlss-control-grid"><label>β<select value={settings.fusion_beta??.1} onChange={e=>handleParamChange('fusion_beta',Number(e.target.value))}>{[.01,.05,.1,.2,.5,1].map(v=><option key={v} value={v}>{v}</option>)}</select></label><label>迭代<select value={settings.fusion_max_iterations??128} onChange={e=>handleParamChange('fusion_max_iterations',Number(e.target.value))}>{[32,64,128,256,512].map(v=><option key={v} value={v}>{v}</option>)}</select></label><label>相对阈值<select value={settings.fusion_relative_tolerance??.001} onChange={e=>handleParamChange('fusion_relative_tolerance',Number(e.target.value))}>{[.01,.001,.0001,.00001].map(v=><option key={v} value={v}>{v}</option>)}</select></label></div></details>}<p className="dlss-note">各层为独立 Feature；Data Consistency 执行 GPU 融合，Baseline 输出第一层。预览与导出运行同一多层处理核心；图片流程仍为单层。</p></>}
          </section>
        </div>}
        hdr={<div className="dlss-inspector-content"><section className="dlss-control-card"><div className="dlss-card-head"><div><h3>HDR / scRGB</h3></div><span className="dlss-chip dlss-chip-export">仅导出</span></div><label className="dlss-check-row"><input type="checkbox" aria-label="实验性 HDR/scRGB" checked={hdrOn} onChange={e=>saveHdrPatch({[HDR_SETTING]:e.target.checked})}/>请求实验性 HDR/scRGB 导出</label><label className="dlss-control-label">纸白 · scRGB<input type="text" inputMode="decimal" aria-label="HDR 纸白 (scRGB)" value={paperWhiteInput} onChange={e=>handlePaperWhiteChange(e.target.value)}/></label>{paperWhiteRefusal?<p className="dlss-warning">{paperWhiteRefusal}</p>:<p className="dlss-note">{paperWhiteLabel(Number(paperWhiteInput))} · 1 scRGB = 80 cd/m²</p>}{hdrOn&&hdrRefusals.length>0&&<ul className="dlss-refusals" aria-label="HDR 拒绝项">{hdrRefusals.map(reason=><li key={reason}><AlertTriangle size={13}/>{reason}</li>)}</ul>}<div className="hdr-preview-warning"><AlertTriangle size={16}/><p>HDR10 以 FP16 处理，再统一映射为 8-bit 显示；未校准，不作 HDR 色彩参考。HDR 开关控制导出。</p></div><p className="dlss-note">兼容性由后端能力闸门决定；不满足条件会拒绝，不会自动回退。HDR10 输出约束由服务端校验。</p><ul className="dlss-limitations">{HDR_LIMITATIONS.map(item=><li key={item}>{item}</li>)}</ul></section></div>}
      />
    </div>
  )
}
