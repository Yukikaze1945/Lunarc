import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import { renderStages, vsrOutputSize, framegenOutputFps, nativeOutputBlockers } from '../services/renderPipeline'
import {
  Film,
  Plus,
  Play,
  Square,
  Trash2,
  Folder,
  Sliders,
  CheckCircle,
  AlertTriangle,
  Clock,
  Activity,
  Check,
  Copy,
  Layers,
  Settings2,
  Volume2,
  Tv,
  Terminal,
  RefreshCw,
  RotateCcw,
  MinusCircle,
  X,
  ChevronRight,
} from 'lucide-react'
import { VideoInfo, DLSSSettings, RenderJob, TranscodeSettings, DiagnosticLogEntry } from '../types'
import { isSdr10Source, sourcePrecisionNotice } from '../services/sourcePrecision'
import {
  addToQueue,
  removeFromQueue,
  retryQueueJob,
  startQueue,
  cancelQueue,
  openNativeDirDialog,
  fetchDiagnosticLogs,
  fetchJobDiagnostics,
} from '../services/api'
import {
  composeJobOutputPath,
  sameOutputAsInput,
} from '../services/exportPreview'
import {
  hdrBlockers, isHdrRequested,
} from '../services/hdrExport'
import { hasExportControls } from '../services/qualityControls'
import {exportOptions, normalizeExportOptions} from '../services/exportOptions'
import { subscribeQueue, refreshQueue } from '../services/queue'
import './deliver.css'

const TRANSCODE_DRAFT_KEY = 'lunarc.transcodeDraft'
const TRANSCODE_DRAFT_VERSION = 1
const DRAFT_ENUMS: Partial<Record<keyof TranscodeSettings, readonly string[]>> = {
  codec: ['av1_nvenc', 'hevc_nvenc', 'h264_nvenc'],
  container: ['.mkv', '.mp4', '.mov', '.webm'],
  rate_mode: ['cq', 'constqp', 'cbr', 'vbr'],
  peak_policy: ['auto', 'custom'],
  preset: ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'],
  tune: ['auto', 'hq', 'uhq', 'll', 'ull', 'lossless'],
  multipass: ['auto', 'disabled', 'qres', 'fullres'],
  decode_mode: ['auto', 'cuda', 'cpu'],
  finish_action: ['none', 'sleep', 'shutdown'],
  gop_mode: ['auto', 'manual'],
  b_frames: ['auto', '0', '1', '2', '3', '4', '5', '6', '7'],
  b_ref_mode: ['auto', 'disabled', 'middle', 'each'],
  lookahead: ['auto', '0', '8', '16', '24', '32'],
  lookahead_level: ['auto', '0', '1', '2', '3'],
  aq_mode: ['spatial', 'temporal', 'disabled'],
  temporal_filter: ['auto', '0', '1', '2', '3', '4'],
  split_encode: ['auto', 'disabled', '2', '3', '4'],
  profile: ['auto', 'main', 'main10', 'high'],
  tier: ['auto', 'main', 'high'],
  resolution_mode: ['original', 'double', 'half', '5120x2880', '3840x2160', '2560x1440', '1920x1080', '1280x720', 'custom_size', 'custom_factor'],
  scale_algorithm: ['lanczos', 'bicubic', 'bilinear', 'nearest'],
  pixel_format: ['auto', 'yuv420p', 'nv12', 'p010le', 'p016le', 'nv16', 'p210le', 'p216le', 'yuv444p', 'yuv444p10msble', 'yuv444p16le'],
  color_preset: ['preserve', 'bt709_limited', 'bt709_full', 'bt2020_sdr', 'bt2020_pq', 'bt2020_hlg', 'hdr_to_sdr', 'custom'],
  custom_color_range: ['tv', 'pc'],
  audio_mode: ['copy', 'aac', 'opus', 'flac', 'pcm', 'remove'],
  audio_bitrate: ['128k', '192k', '256k', '320k', '512k'],
  subtitle_mode: ['copy', 'convert', 'drop'],
}

function readTranscodeDraft(defaults: TranscodeSettings): TranscodeSettings {
  try {
    const raw = localStorage.getItem(TRANSCODE_DRAFT_KEY)
    if (!raw) return defaults
    const parsed = JSON.parse(raw)
    const saved = parsed?.settings
    if (!saved || typeof saved !== 'object' || Array.isArray(saved)) return defaults
    // Merge known, type-compatible fields so a schema upgrade keeps everything it
    // still understands while new settings inherit safe defaults.
    const merged = { ...defaults }
    for (const key of Object.keys(defaults) as (keyof TranscodeSettings)[]) {
      const value = saved[key]
      const fallback = defaults[key]
      const allowed = DRAFT_ENUMS[key]
      const numericBounds: Partial<Record<keyof TranscodeSettings, [number, number]>> = {
        cq: [0, 51], qp: [0, 51], gop: [1, 100000], aq_strength: [1, 15],
        custom_width: [2, 16384], custom_height: [2, 16384], custom_factor: [0.1, 8],
      }
      const bounds = numericBounds[key]
      const invalidNumber = typeof value === 'number' && (!Number.isFinite(value) ||
        (bounds !== undefined && (value < bounds[0] || value > bounds[1])) ||
        ((key === 'custom_width' || key === 'custom_height') && value % 2 !== 0))
      if (typeof value === typeof fallback && !invalidNumber &&
          !(typeof value === 'string' && allowed && !allowed.includes(value))) {
        ;(merged as any)[key] = value
      }
    }
    return merged
  } catch {
    return defaults
  }
}

interface DeliverPageProps {
  isActive?: boolean
  video: VideoInfo | null
  settings: DLSSSettings
  // The reviewed HDR preset writes the shared DLSS settings as well as the transcode
  // form: the opt-in lives on the DLSS page, and a job carries the settings the panel
  // is showing, so both halves have to move together on one explicit click.
  onSettingsChange?: (newSettings: DLSSSettings) => void
  onStatusChange?: (msg: string) => void
}

// 4 Classic RTX Transcoder Presets from RTXVideoTranscoderSlint
const BUILTIN_PRESETS: { name: string; settings: Partial<TranscodeSettings> }[] = [
  {
    name: '默认：AV1 · MKV · P7 · UHQ · Fullres · CQ16',
    settings: {
      codec: 'av1_nvenc',
      container: '.mkv',
      preset: 'p7',
      tune: 'uhq',
      multipass: 'fullres',
      rate_mode: 'cq',
      cq: 16,
      pixel_format: 'auto',
      color_preset: 'preserve',
      audio_mode: 'copy',
    },
  },
  {
    name: '4K 极致画质：HEVC · MKV · P7 · UHQ · CQ14 · P010',
    settings: {
      codec: 'hevc_nvenc',
      container: '.mkv',
      preset: 'p7',
      tune: 'uhq',
      multipass: 'fullres',
      rate_mode: 'cq',
      cq: 14,
      pixel_format: 'p010le',
      color_preset: 'preserve',
      audio_mode: 'copy',
    },
  },
  {
    name: '高压缩分享：AV1 · MP4 · P6 · HQ · CQ22',
    settings: {
      codec: 'av1_nvenc',
      container: '.mp4',
      preset: 'p6',
      tune: 'hq',
      multipass: 'qres',
      rate_mode: 'cq',
      cq: 22,
      pixel_format: 'yuv420p',
      color_preset: 'preserve',
      audio_mode: 'aac',
      audio_bitrate: '256k',
    },
  },
  {
    name: '兼容广播级：H.264 · MP4 · P5 · HQ · CQ18 · NV12',
    settings: {
      codec: 'h264_nvenc',
      container: '.mp4',
      preset: 'p5',
      tune: 'hq',
      multipass: 'qres',
      rate_mode: 'cq',
      cq: 18,
      pixel_format: 'nv12',
      color_preset: 'bt709_limited',
      audio_mode: 'aac',
      audio_bitrate: '320k',
    },
  },
]

// The states dlss_service.retry_job accepts, listed once so a row cannot offer a reset
// the backend would refuse, and cannot refuse one it would take.
const RETRYABLE_STATUSES: RenderJob['status'][] = ['failed', 'cancelled', 'skipped']

const DEFAULT_SETTINGS: TranscodeSettings = {
  codec: 'av1_nvenc',
  container: '.mkv',
  suffix: '_AV1_P7_UHQ_Q16',
  rate_mode: 'cq',
  cq: 16,
  video_bitrate: '80M',
  qp: 20,
  peak_policy: 'auto',
  maxrate: '380M',
  bufsize: '1520M',
  preset: 'p7',
  tune: 'uhq',
  multipass: 'fullres',
  gpu: '-1',
  decode_mode: 'auto',
  finish_action: 'none',
  gop_mode: 'auto',
  gop: 240,
  b_frames: 'auto',
  b_ref_mode: 'auto',
  lookahead: 'auto',
  lookahead_level: 'auto',
  aq_mode: 'spatial',
  aq_strength: 8,
  weighted_prediction: false,
  temporal_filter: 'auto',
  split_encode: 'auto',
  profile: 'auto',
  level: 'auto',
  tier: 'auto',
  resolution_mode: 'original',
  custom_width: 1920,
  custom_height: 1080,
  custom_factor: 1.0,
  keep_aspect: true,
  scale_algorithm: 'lanczos',
  pixel_format: 'auto',
  color_preset: 'preserve',
  custom_color_space: 'bt709',
  custom_color_primaries: 'bt709',
  custom_color_transfer: 'bt709',
  custom_color_range: 'tv',
  audio_mode: 'copy',
  audio_bitrate: '320k',
  subtitle_mode: 'copy',
  copy_attachments: true,
  copy_metadata: true,
  copy_chapters: true,
  overwrite_output: false,
  skip_existing: true,
  extra_args: '',
}

export const DeliverPage: React.FC<DeliverPageProps> = React.memo(({
  isActive = true,
  video,
  settings: dlssSettings,
  onStatusChange,
}) => {
  // Transcode Settings State
  const [txDraft, setTxSettings] = useState<TranscodeSettings>(() => readTranscodeDraft(DEFAULT_SETTINGS))
  const txSettings = useMemo(() => normalizeExportOptions(txDraft, dlssSettings, video), [txDraft, dlssSettings, video])
  const choices = exportOptions(dlssSettings, video, txSettings)
  const [activeSection, setActiveSection] = useState<'basic' | 'video' | 'color' | 'tracks' | 'advanced'>('basic')
  const [fileName, setFileName] = useState<string>('')
  const [outputDir, setOutputDir] = useState<string>('')
  const [advancedOpen, setAdvancedOpen] = useState<boolean>(false)
  const [selectedPresetIdx, setSelectedPresetIdx] = useState<number>(0)
  useEffect(() => {
    if (txDraft !== txSettings) { setTxSettings(txSettings); setSelectedPresetIdx(-1) }
  }, [txDraft, txSettings])
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false)
  const [justAdded, setJustAdded] = useState<boolean>(false)

  // Queue State
  const [jobs, setJobs] = useState<RenderJob[]>([])
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  // Whether the backend's queue worker still owns a round. A row's own status cannot
  // say this: between the last row turning terminal and that thread exiting nothing is
  // running, and that is exactly when a retry has to stay disabled.
  const [queueRunning, setQueueRunning] = useState<boolean>(false)
  const [retryingJobId, setRetryingJobId] = useState<string | null>(null)
  const [selectedFailedJob, setSelectedFailedJob] = useState<RenderJob | null>(null)
  const [diagnosticLogs, setDiagnosticLogs] = useState<DiagnosticLogEntry[]>([])
  const [consoleOpen, setConsoleOpen] = useState<boolean>(false)
  const [copiedDiag, setCopiedDiag] = useState<boolean>(false)
  const [diagnosticJob, setDiagnosticJob] = useState<RenderJob | null>(null)
  const [jobDiagnostics, setJobDiagnostics] = useState<any>(null)
  const [diagnosticsLoading, setDiagnosticsLoading] = useState(false)
  const [diagnosticsError, setDiagnosticsError] = useState('')
  const diagnosticRevision = useRef(0)
  const logsAbort = useRef<AbortController | null>(null)
  const logsPending = useRef(false)

  useEffect(() => {
    try { localStorage.setItem(TRANSCODE_DRAFT_KEY, JSON.stringify({ version: TRANSCODE_DRAFT_VERSION, settings: txSettings })) }
    catch (e) { console.warn('Unable to save transcode draft:', e) }
  }, [txSettings])

  const inspectJob = async (job: RenderJob) => {
    const revision = ++diagnosticRevision.current
    setDiagnosticJob(job)
    setJobDiagnostics(null)
    setDiagnosticsError('')
    setDiagnosticsLoading(true)
    try {
      const diagnostics = await fetchJobDiagnostics(job.id)
      if (revision === diagnosticRevision.current) setJobDiagnostics(diagnostics)
    } catch (e: any) {
      if (revision === diagnosticRevision.current) setDiagnosticsError(e?.message || '无法读取作业诊断；确认本地后端仍在运行后重试。')
    } finally { if (revision === diagnosticRevision.current) setDiagnosticsLoading(false) }
  }

  const closeDiagnostics = () => {
    diagnosticRevision.current++
    setDiagnosticJob(null)
  }

  const refreshLogs = useCallback(async () => {
    if (logsPending.current || !consoleOpen || !isActive) return
    const signal = logsAbort.current?.signal
    if (signal?.aborted) return
    logsPending.current = true
    try {
      const res = await fetchDiagnosticLogs(150, signal)
      if (!signal?.aborted && res?.logs) setDiagnosticLogs(res.logs)
    } catch (error) {
      if (!signal?.aborted) console.warn('Failed to fetch diagnostic logs:', error)
    } finally { logsPending.current = false }
  }, [consoleOpen, isActive])

  // NOTE(oxlint baseline react(set-state-in-effect)): intentional external
  // sync — starts log polling only while the console panel is open.
  useEffect(() => {
    if (consoleOpen && isActive) {
      const controller = new AbortController()
      logsAbort.current = controller
      void refreshLogs()
      const interval = setInterval(refreshLogs, 2000)
      return () => { controller.abort(); clearInterval(interval) }
    }
  }, [consoleOpen, isActive, refreshLogs])

  // Set default filename & directory when video loads
  // NOTE(oxlint baseline): intentional one-way defaulting of form fields from
  // newly loaded video metadata (see above re disable comments).
  useEffect(() => {
    if (video) {
      const stem = video.filename.replace(/\.[^/.]+$/, '')
      setFileName(`${stem}_dlss`)
      const lastSlash = Math.max(video.path.lastIndexOf('/'), video.path.lastIndexOf('\\'))
      if (lastSlash > 0) {
        setOutputDir(video.path.substring(0, lastSlash))
      }
    }
  }, [video])

  // Share the live queue with the bottom bar; HTTP recovery is owned by the feed.
  useEffect(() => {
    if (!isActive) return
    return subscribeQueue(state => {
      setJobs(state.jobs)
      setActiveJobId(state.active_id)
      setQueueRunning(state.queue_running)
    })
  }, [isActive])

  // Helper update
  const updateSetting = <K extends keyof TranscodeSettings>(key: K, value: TranscodeSettings[K]) => {
    setSelectedPresetIdx(-1)
    setTxSettings((prev) => {
      const next = { ...prev, [key]: value }

      // Enforce weighted prediction mutual exclusion
      if (key === 'weighted_prediction') {
        if (value) {
          next.b_frames = '0'
          next.b_ref_mode = 'disabled'
          if (next.codec === 'hevc_nvenc') {
            next.split_encode = 'disabled'
          }
        }
      } else if (key === 'b_frames' && value !== '0' && next.weighted_prediction) {
        next.weighted_prediction = false
      } else if (key === 'codec') {
        if (value === 'h264_nvenc' && next.tune === 'uhq') {
          next.tune = 'hq'
        }
        if (value === 'av1_nvenc') {
          next.weighted_prediction = false
          if (next.b_ref_mode === 'each') {
            next.b_ref_mode = 'auto'
          }
        }
      }
      return next
    })
  }

  // Load Preset
  const handleSelectPreset = (idx: number) => {
    setSelectedPresetIdx(idx)
    if (idx >= 0 && idx < BUILTIN_PRESETS.length) {
      setTxSettings((prev) => ({
        ...prev,
        ...BUILTIN_PRESETS[idx].settings,
      }))
    }
  }

  // Native Browse directory
  const handleBrowseDir = async () => {
    const selected = await openNativeDirDialog()
    if (selected) {
      setOutputDir(selected)
    }
  }

  // Dynamic Output Dimensions Calculation
  const outputDimensions = useMemo(() => {
    if (renderStages(dlssSettings).includes('vsr')) {
      return vsrOutputSize(dlssSettings, video ?? {width:1920,height:1080})
    }
    const origW = video ? video.width : 1920
    const origH = video ? video.height : 1080

    const even = (v: number) => {
      let r = Math.round(v)
      if (r % 2 !== 0) r -= 1
      return Math.max(2, r)
    }

    switch (txSettings.resolution_mode) {
      case 'original':
        return { width: even(origW), height: even(origH) }
      case 'double':
        return { width: even(origW * 2), height: even(origH * 2) }
      case 'half':
        return { width: even(origW / 2), height: even(origH / 2) }
      case '5120x2880':
        return { width: 5120, height: 2880 }
      case '3840x2160':
        return { width: 3840, height: 2160 }
      case '2560x1440':
        return { width: 2560, height: 1440 }
      case '1920x1080':
        return { width: 1920, height: 1080 }
      case '1280x720':
        return { width: 1280, height: 720 }
      case 'custom_factor': {
        const factor = Math.min(8.0, Math.max(0.1, txSettings.custom_factor))
        return { width: even(origW * factor), height: even(origH * factor) }
      }
      case 'custom_size': {
        if (!txSettings.keep_aspect) {
          return {
            width: even(txSettings.custom_width),
            height: even(txSettings.custom_height),
          }
        }
        const ratio = Math.min(
          txSettings.custom_width / origW,
          txSettings.custom_height / origH
        )
        return { width: even(origW * ratio), height: even(origH * ratio) }
      }
      default:
        return { width: origW, height: origH }
    }
  }, [video, dlssSettings, txSettings.resolution_mode, txSettings.custom_width, txSettings.custom_height, txSettings.custom_factor, txSettings.keep_aspect])

  // Dynamic Rate Summary calculation
  const rateSummary = useMemo(() => {
    const fps = framegenOutputFps(dlssSettings, video ? video.fps : 30.0)
    const pixelCeiling = outputDimensions.width * outputDimensions.height * fps * 0.75
    const raw = Math.max(pixelCeiling, 20_000_000)
    const rounded = Math.ceil(raw / 10_000_000) * 10_000_000
    const autoMax = Math.min(1_000_000_000, Math.max(20_000_000, rounded))
    const autoBuf = Math.min(2_000_000_000, autoMax * 4)

    const formatM = (v: number) => `${Math.round(v / 1_000_000)}M`

    if (txSettings.rate_mode === 'cq') {
      return `CQ ${txSettings.cq} · 瞬时峰值 ${formatM(autoMax)} · 缓冲 ${formatM(autoBuf)}`
    } else if (txSettings.rate_mode === 'constqp') {
      return `恒定 QP ${txSettings.qp}`
    } else if (txSettings.rate_mode === 'cbr') {
      return `CBR 恒定码率 ${txSettings.video_bitrate}`
    } else {
      return `目标码率 ${txSettings.video_bitrate} · 峰值 ${formatM(autoMax)} · 缓冲 ${formatM(autoBuf)}`
    }
  }, [video, outputDimensions, dlssSettings.frame_generation, dlssSettings.frame_multiplier, dlssSettings.framegen_target_fps, txSettings.rate_mode, txSettings.cq, txSettings.qp, txSettings.video_bitrate])

  // --- the experimental HDR10 branch -----------------------------------------------
  //
  // The opt-in is a request a job carries, not a page mode: with it off, everything
  // below stays the SDR form it always was. With it on, the clauses the backend gates
  // on are shown as blocking errors here, and what only the backend can see is shown
  // as undecided rather than as a green light.
  const hdrOn = isHdrRequested(dlssSettings)
  const sdr10On = !hdrOn && dlssSettings.export_backend === 'gpu_transfer' && isSdr10Source(video)
  const hdrForm = useMemo(() => ({
    settings: dlssSettings,
    ts: txSettings,
    video,
    outPath: composeJobOutputPath(outputDir, fileName, txSettings.container,
      video ? video.filename.replace(/\.[^/.]+$/, '') : ''),
    outputWidth: outputDimensions.width,
    outputHeight: outputDimensions.height,
  }), [dlssSettings, txSettings, video, outputDir, fileName, outputDimensions])
  const hdrRefusals = useMemo(() => hdrBlockers(hdrForm), [hdrForm])
  // Validation Logic
  const validationIssues = useMemo(() => {
    const issues: { type: 'error' | 'warning'; text: string }[] = []
    issues.push(...nativeOutputBlockers(dlssSettings, video).map(text=>({type:'error' as const,text})))

    if (hasExportControls(dlssSettings) && dlssSettings.export_backend !== 'gpu_transfer') {
      issues.push({ type: 'error', text: '残差后处理或时间控制已启用，但当前导出后端不支持这些控制。请切换到 GPU Transfer，或在 DLSS 页恢复默认控制。' })
    }

    if (sdr10On) {
      if (txSettings.codec !== 'hevc_nvenc' || txSettings.container !== '.mkv')
        issues.push({ type: 'error', text: 'SDR 10-bit 导出请选择 HEVC · MKV。' })
      if (!['auto', 'p010le', 'yuv420p10le'].includes(txSettings.pixel_format))
        issues.push({ type: 'error', text: 'SDR 10-bit 的像素格式请选择 Auto 或 P010。' })
      if (!['auto', 'main10'].includes(txSettings.profile))
        issues.push({ type: 'error', text: 'SDR 10-bit 的 Profile 请选择 Auto 或 Main10。' })
      if (txSettings.color_preset !== 'preserve')
        issues.push({ type: 'error', text: 'SDR 10-bit 请选择保留源色彩。' })
    }
    if (!hdrOn && !sdr10On && dlssSettings.export_backend === 'gpu_transfer' && !['auto', 'yuv420p'].includes(txSettings.pixel_format)) {
      issues.push({ type: 'error', text: 'SDR GPU Transfer 的像素格式请选择 Auto 或 YUV420P；编码器实际提交 NV12 8-bit 4:2:0。其他像素格式只适用于相应的普通导出链。' })
    }

    if (txSettings.codec === 'h264_nvenc' && txSettings.tune === 'uhq') {
      issues.push({ type: 'error', text: 'H.264 NVENC 不支持 UHQ 调优，请改用 HQ 或选择 AV1 / HEVC' })
    }

    if (dlssSettings.reconstruction_mode !== 'single') {
      if (dlssSettings.export_backend !== 'gpu_transfer') {
        issues.push({ type: 'error', text: 'Dual / Triple 独立多尺度层只支持 GPU Transfer 导出' })
      }
      const count = dlssSettings.reconstruction_mode === 'triple' ? 3 : 2
      const scales = dlssSettings.layer_scales || []
      if (scales.length !== count ||
          Math.round((scales[0] || 0) * 100) !== Math.round(dlssSettings.working_scale * 100) ||
          scales.some((scale) => !Number.isFinite(scale) || scale < 0.25 || scale > 1 ||
            Math.abs(scale * 100 - Math.round(scale * 100)) > 1e-6) ||
          new Set(scales.map((scale) => Math.round(scale * 100))).size !== count) {
        issues.push({ type: 'error', text: '每层需使用不同的 25–100% 整数比例，且第 1 层须等于工作分辨率' })
      }
    }

    if (txSettings.weighted_prediction) {
      if (txSettings.codec === 'av1_nvenc') {
        issues.push({ type: 'error', text: 'NVIDIA 加权预测仅支持 H.264 和 HEVC，不支持 AV1' })
      }
      if (txSettings.b_frames !== '0') {
        issues.push({ type: 'error', text: '加权预测不能与 B 帧共用，需将 B 帧设为 0' })
      }
    }

    if (txSettings.codec === 'av1_nvenc' && txSettings.b_ref_mode === 'each') {
      issues.push({ type: 'error', text: 'AV1 不支持“每个 B 帧作为参考”，请改用“中间 B 帧”或关闭' })
    }

    if (txSettings.container === '.webm' && txSettings.codec !== 'av1_nvenc') {
      issues.push({ type: 'error', text: 'WebM 容器仅允许封装 AV1 视频' })
    }

    const precisionNotice = sourcePrecisionNotice(video, hdrOn, dlssSettings.export_backend)
    if (precisionNotice?.type === 'error') issues.push(precisionNotice)
    if (hdrOn) {
      // Each of these is a clause of negotiate_hdr / plan_hdr_encoder: the job is
      // refused as configured, and nothing here switches it to another chain.
      for (const reason of hdrRefusals) {
        issues.push({ type: 'error', text: `实验性 HDR10：${reason}` })
      }

    }

    // The destination and the source resolving to one file is not a configuration
    // preference: the encoder would overwrite the material still being decoded. This
    // check is prevention only - a browser cannot see a link or a hard link, so the
    // backend re-checks against the file table before any decoder or encoder opens,
    // and a job that gets past here still comes back refused with that reason.
    if (video && sameOutputAsInput(
      video.path, outputDir, fileName, txSettings.container,
      video.filename.replace(/\.[^/.]+$/, '')
    )) {
      issues.push({
        type: 'error',
        text: '导出目标与输入素材是同一个文件：请修改【文件名】或【导出目标文件夹】后再加入队列（仅大小写或斜杠写法不同仍算同一文件）',
      })
    }

    return issues
  }, [video, txSettings, outputDir, fileName, dlssSettings, hdrOn, sdr10On, hdrRefusals])

  const isValid = !validationIssues.some((i) => i.type === 'error')

  // Add to render queue
  const handleAddToQueue = async () => {
    if (!video) {
      onStatusChange?.('请先导入素材')
      return
    }
    if (!isValid) {
      onStatusChange?.(validationIssues.find(issue => issue.type === 'error')?.text || '请检查导出设置')
      return
    }

    setIsSubmitting(true)
    const trimmedName = fileName.trim() || video.filename.replace(/\.[^/.]+$/, '')
    const fullOutputPath = composeJobOutputPath(
      outputDir, fileName, txSettings.container, video.filename.replace(/\.[^/.]+$/, ''))

    // A job carries the DLSS host the panel shows rather than deriving one: a job
    // that arrived naming none would inherit whatever the service default happened
    // to be at the moment the queue ran it, which is not the same thing as what the
    // operator was looking at. There is one host to carry and the export chain
    // forces nothing.
    const newJob: Partial<RenderJob> = {
      name: trimmedName,
      input_path: video.path,
      output_path: fullOutputPath,
      codec: txSettings.codec,
      container: txSettings.container.replace('.', ''),
      cq: txSettings.cq,
      preset: txSettings.preset,
      with_audio: txSettings.audio_mode !== 'remove',
      total_frames: video.frames,
      settings: { ...dlssSettings },
      transcode_settings: { ...txSettings },
    }

    try {
      const created = await addToQueue(newJob)
      setJobs((prev) => {
        if (prev.some((j) => j.id === created.id)) return prev
        return [...prev, created]
      })
      setJustAdded(true)
      setTimeout(() => setJustAdded(false), 2500)
      onStatusChange?.(`已添加作业到渲染队列: ${created.name}`)

      // Refresh queue from server to guarantee sync
      try {
        const q = await refreshQueue()
        setJobs(q.jobs)
        setActiveJobId(q.active_id)
        setQueueRunning(q.queue_running)
      } catch {}
    } catch (e: any) {
      onStatusChange?.(e.message || '添加任务失败')
    } finally {
      setIsSubmitting(false)
    }
  }

  // Start Queue (P1-09: destructive finish actions need explicit consent)
  const handleStartQueue = async () => {
    try {
      const queuedRisky = jobs.some(
        (j) => j.status === 'pending' &&
          ['sleep', 'shutdown'].includes(
            String(j.transcode_settings?.finish_action || 'none'))
      )
      const currentRisky = ['sleep', 'shutdown'].includes(String(txSettings?.finish_action || 'none'))
      if (queuedRisky || currentRisky) {
        const ok = window.confirm(
          '检测到“完成后睡眠/关机”设置。\n\n后端仅在本轮成功渲染的任务上执行；任一失败/取消都会禁止该动作，已跳过（未渲染）的任务不计入。\n点击“确定”开始渲染。'
        )
        if (!ok) return
      }
      const state = await startQueue()
      if (state) {
        setJobs(state.jobs)
        setActiveJobId(state.active_id)
        setQueueRunning(state.queue_running)
      } else {
        // The answer carried no state, so ask instead of assuming a round began.
        try {
          const q = await refreshQueue()
          setQueueRunning(q.queue_running)
        } catch {}
      }
      onStatusChange?.('渲染队列已启动')
    } catch (e: any) {
      alert(e.message || '启动渲染失败')
    }
  }

  // Cancel Active Job
  const handleCancelJob = async () => {
    try {
      await cancelQueue()
      onStatusChange?.('已请求终止当前渲染')
    } catch (e: any) {
      alert(e.message || '取消失败')
    }
  }

  // Remove Job
  const handleRemoveJob = async (id: string) => {
    try {
      await removeFromQueue(id)
      setJobs((prev) => prev.filter((j) => j.id !== id))
    } catch (e: any) {
      alert(e.message || '删除任务失败')
    }
  }

  // Put one finished job back in the queue, by name, on this click alone.
  // The backend renders pending jobs only, so nothing else re-runs a past failure; the
  // 409 it returns when a round is still in flight is shown rather than swallowed,
  // because "wait for this round" is the part an operator can act on.
  const handleRetryJob = async (id: string) => {
    setRetryingJobId(id)
    try {
      const state = await retryQueueJob(id)
      setJobs(state.jobs)
      setActiveJobId(state.active_id)
      setQueueRunning(state.queue_running)
      onStatusChange?.(`已重置任务为待渲染：${state.jobs.find((j) => j.id === id)?.name || id}`)
    } catch (e: any) {
      alert(e.message || '重试任务失败')
      try {
        const q = await refreshQueue()
        setJobs(q.jobs)
        setActiveJobId(q.active_id)
        setQueueRunning(q.queue_running)
      } catch {}
    } finally {
      setRetryingJobId(null)
    }
  }

  // Render All acts on pending rows only, so the button is enabled by pending work and
  // locked while a round is in flight; a terminal row goes back to pending through
  // handleRetryJob, one row at a time, on the operator's word.
  const pendingCount = jobs.filter((j) => j.status === 'pending').length

  return (
    <div className="swiss-delivery flex-1 flex flex-col h-full bg-lunarc-dark text-lunarc-text overflow-hidden select-none">
      {/* Top Header / Bar */}
      <div className="h-14 px-6 border-b border-lunarc-border bg-lunarc-surface flex items-center justify-between flex-shrink-0">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Film className="w-5 h-5 text-lunarc-accent" />
            <span className="text-sm font-semibold tracking-wide text-lunarc-text uppercase">
              导出设置
            </span>
          </div>
          <div className="h-4 w-px bg-lunarc-border" />
          {/* Preset Selector */}
          <div className="flex items-center space-x-2">
            <span className="text-xs text-lunarc-muted">预设方案:</span>
            <select
              value={selectedPresetIdx}
              onChange={(e) => handleSelectPreset(Number(e.target.value))}
              className="bg-lunarc-panel border border-lunarc-border text-xs text-lunarc-text rounded px-2.5 py-1.5 focus:outline-none focus:border-lunarc-accent"
            >
              <option value={-1} disabled>自定义</option>
              {BUILTIN_PRESETS.map((p, idx) => (
                <option key={idx} value={idx}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          {/* Validation Pill */}
          <div
            className={`flex items-center space-x-1.5 px-3 py-1 rounded text-xs font-medium ${
              isValid
                ? 'text-lunarc-green border border-lunarc-border'
                : 'text-lunarc-red border border-lunarc-border'
            }`}
          >
            {isValid ? <CheckCircle className="w-3.5 h-3.5" /> : <AlertTriangle className="w-3.5 h-3.5" />}
            <span>{isValid ? '格式匹配' : '需要修正'}</span>
          </div>

          {/* Advanced Drawer Button */}
          <button
            onClick={() => { setActiveSection('advanced'); setAdvancedOpen(!advancedOpen) }}
            className={`flex items-center space-x-1.5 px-3.5 py-1.5 text-xs font-medium rounded transition-all ${
              advancedOpen
                ? 'bg-lunarc-accent text-[var(--accent-ink)] font-bold shadow-sm'
                : 'bg-lunarc-panel hover:bg-lunarc-card text-lunarc-text border border-lunarc-border'
            }`}
          >
            <Settings2 className="w-3.5 h-3.5" />
            <span>{advancedOpen ? '收起高级 NVENC' : '高级 NVENC'}</span>
          </button>
        </div>
      </div>

      <div className="deliver-layout">
      {/* Main Settings Area */}
      <div className="deliver-editor flex flex-col overflow-hidden">
        <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Left / Center Settings Forms (Scrollable) */}
        <div className="flex-1 min-w-0 overflow-y-auto p-6 space-y-5">
          <nav aria-label="导出设置分类" className="sticky top-0 z-10 -mx-2 flex gap-2 overflow-x-auto rounded-sm border border-lunarc-border bg-lunarc-surface p-2 shadow-none">
            {([
              ['basic', '基础'], ['video', '视频'], ['color', '色彩'], ['tracks', '音轨封装'], ['advanced', '高级'],
            ] as const).map(([key, label]) => (
              <button key={key} type="button" aria-current={activeSection === key ? 'page' : undefined}
                onClick={() => { setActiveSection(key); if (key === 'advanced') setAdvancedOpen(true); else setAdvancedOpen(false) }}
                className={`min-h-9 shrink-0 rounded-sm px-4 text-xs font-semibold transition ${activeSection === key ? 'bg-lunarc-accent text-[var(--accent-ink)]' : 'text-lunarc-muted hover:bg-lunarc-panel hover:text-lunarc-text'}`}>
                {label}<ChevronRight className="ml-1 inline h-3 w-3" aria-hidden="true" />
              </button>
            ))}
          </nav>
          {/* File Output Card */}
          <div className="bg-lunarc-surface border border-lunarc-border rounded-sm p-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-lunarc-text uppercase tracking-wider flex items-center space-x-1.5">
                <Folder className="w-4 h-4 text-lunarc-accent" />
                <span>输出文件与目录</span>
              </span>
              <span className="text-[11px] text-lunarc-dim">
                目标格式: {outputDimensions.width}×{outputDimensions.height} · {txSettings.container.toUpperCase()}
              </span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-12 gap-3">
              <div className="md:col-span-4">
                <label className="text-[11px] text-lunarc-muted mb-1 block">文件名</label>
                <input
                  type="text"
                  value={fileName}
                  onChange={(e) => setFileName(e.target.value)}
                  placeholder="输出视频文件名"
                  className="w-full bg-lunarc-dark border border-lunarc-border focus:border-lunarc-accent rounded px-3 py-1.5 text-xs text-lunarc-text focus:outline-none"
                />
              </div>
              <div className="md:col-span-8">
                <label className="text-[11px] text-lunarc-muted mb-1 block">导出目标文件夹</label>
                <div className="flex space-x-2">
                  <input
                    type="text"
                    value={outputDir}
                    onChange={(e) => setOutputDir(e.target.value)}
                    placeholder="例如 D:\Output"
                    className="flex-1 bg-lunarc-dark border border-lunarc-border focus:border-lunarc-accent rounded px-3 py-1.5 text-xs text-lunarc-text focus:outline-none"
                  />
                  <button
                    onClick={handleBrowseDir}
                    className="px-3 py-1.5 bg-lunarc-panel hover:bg-lunarc-card text-lunarc-text border border-lunarc-border rounded text-xs transition"
                  >
                    浏览...
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Group 1: 基础 (Basic) */}
          <div className={`bg-lunarc-surface border border-lunarc-border rounded-sm p-4 space-y-4 ${activeSection !== 'basic' ? 'hidden' : ''}`}>
            <div className="text-xs font-bold text-lunarc-text uppercase tracking-wider flex items-center space-x-1.5">
              <Film className="w-4 h-4 text-lunarc-accent" />
              <span>编码设置</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
              <div>
                <label className="text-[11px] text-lunarc-muted mb-1 block">编码器</label>
                <select
                  value={txSettings.codec}
                  onChange={(e) => updateSetting('codec', e.target.value)}
                  className="w-full bg-lunarc-dark border border-lunarc-border focus:border-lunarc-accent rounded px-3 py-1.5 text-xs text-lunarc-text focus:outline-none"
                >
                  <option value="av1_nvenc" disabled={Boolean(choices.codec && !choices.codec.includes("av1_nvenc"))}>AV1 NVENC (RTX 40/50 系列)</option>
                  <option value="hevc_nvenc" disabled={Boolean(choices.codec && !choices.codec.includes("hevc_nvenc"))}>HEVC NVENC (H.265)</option>
                  <option value="h264_nvenc" disabled={Boolean(choices.codec && !choices.codec.includes("h264_nvenc"))}>H.264 NVENC (AVC)</option>
                </select>
              </div>

              <div>
                <label className="text-[11px] text-lunarc-muted mb-1 block">容器</label>
                <select
                  value={txSettings.container}
                  onChange={(e) => updateSetting('container', e.target.value)}
                  className="w-full bg-lunarc-dark border border-lunarc-border focus:border-lunarc-accent rounded px-3 py-1.5 text-xs text-lunarc-text focus:outline-none"
                >
                  <option value=".mkv" disabled={Boolean(choices.container && !choices.container.includes(".mkv"))}>MKV (Matroska)</option>
                  <option value=".mp4" disabled={Boolean(choices.container && !choices.container.includes(".mp4"))}>MP4 (MPEG-4)</option>
                  <option value=".mov" disabled={Boolean(choices.container && !choices.container.includes(".mov"))}>MOV (QuickTime)</option>
                  <option value=".webm" disabled={Boolean(choices.container && !choices.container.includes(".webm"))}>WebM</option>
                </select>
              </div>

              <div>
                <label className="text-[11px] text-lunarc-muted mb-1 block">预设档位</label>
                <select
                  value={txSettings.preset}
                  onChange={(e) => updateSetting('preset', e.target.value)}
                  className="w-full bg-lunarc-dark border border-lunarc-border focus:border-lunarc-accent rounded px-3 py-1.5 text-xs text-lunarc-text focus:outline-none"
                >
                  <option value="p7">P7 (最高质量 · 最慢)</option>
                  <option value="p6">P6 (极高画质)</option>
                  <option value="p5">P5 (高画质)</option>
                  <option value="p4">P4 (默认均衡)</option>
                  <option value="p3">P3 (较快)</option>
                  <option value="p2">P2 (快速)</option>
                  <option value="p1">P1 (最快 · 极速)</option>
                </select>
              </div>

              <div>
                <label className="text-[11px] text-lunarc-muted mb-1 block">调优策略</label>
                <select
                  value={txSettings.tune}
                  onChange={(e) => updateSetting('tune', e.target.value)}
                  className="w-full bg-lunarc-dark border border-lunarc-border focus:border-lunarc-accent rounded px-3 py-1.5 text-xs text-lunarc-text focus:outline-none"
                >
                  <option value="auto">自动 (Auto)</option>
                  <option value="hq">HQ (高质量)</option>
                  {txSettings.codec !== 'h264_nvenc' && <option value="uhq">UHQ (超高质量)</option>}
                  <option value="ll">低延迟 (LL)</option>
                  <option value="ull">超低延迟 (ULL)</option>
                  <option value="lossless">无损 (Lossless)</option>
                </select>
              </div>

              <div>
                <label className="text-[11px] text-lunarc-muted mb-1 block">多遍编码</label>
                <select
                  value={txSettings.multipass}
                  onChange={(e) => updateSetting('multipass', e.target.value)}
                  className="w-full bg-lunarc-dark border border-lunarc-border focus:border-lunarc-accent rounded px-3 py-1.5 text-xs text-lunarc-text focus:outline-none"
                >
                  <option value="auto">自动 (Auto)</option>
                  <option value="disabled">关闭 (Disabled)</option>
                  <option value="fullres">Fullres (全分辨率 2-Pass)</option>
                  <option value="qres">Qres (1/4 分辨率 2-Pass)</option>
                </select>
              </div>

              <div>
                <label className="text-[11px] text-lunarc-muted mb-1 block">解码模式</label>
                <select
                  value={txSettings.decode_mode}
                  onChange={(e) => updateSetting('decode_mode', e.target.value)}
                  className="w-full bg-lunarc-dark border border-lunarc-border focus:border-lunarc-accent rounded px-3 py-1.5 text-xs text-lunarc-text focus:outline-none"
                >
                  <option value="auto">自动 CUDA / CPU 重试</option>
                  <option value="cuda">强制 CUDA 硬件解码</option>
                  <option value="cpu">CPU 解码 (缩放仍用 GPU)</option>
                </select>
              </div>

              <div>
                <label className="text-[11px] text-lunarc-muted mb-1 block">完成后动作</label>
                <select
                  value={txSettings.finish_action}
                  onChange={(e) => updateSetting('finish_action', e.target.value)}
                  className="w-full bg-lunarc-dark border border-lunarc-border focus:border-lunarc-accent rounded px-3 py-1.5 text-xs text-lunarc-text focus:outline-none"
                >
                  <option value="none">仅系统提醒</option>
                  <option value="sleep">计算机睡眠</option>
                  <option value="shutdown">关闭计算机</option>
                </select>
              </div>
            </div>
          </div>

          {/* Group 2: 码率控制 (Rate Control) */}
          <div className={`bg-lunarc-surface border border-lunarc-border rounded-sm p-4 space-y-4 ${activeSection !== 'basic' ? 'hidden' : ''}`}>
            <div className="flex items-center justify-between">
              <div className="text-xs font-bold text-lunarc-text uppercase tracking-wider flex items-center space-x-1.5">
                <Sliders className="w-4 h-4 text-lunarc-accent" />
                <span>码率控制</span>
              </div>
              <span className="text-xs font-mono text-lunarc-accent bg-lunarc-panel px-2.5 py-0.5 rounded">
                {rateSummary}
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
              <div>
                <label className="text-[11px] text-lunarc-muted mb-1 block">控制模式</label>
                <select
                  value={txSettings.rate_mode}
                  onChange={(e) => updateSetting('rate_mode', e.target.value)}
                  className="w-full bg-lunarc-dark border border-lunarc-border focus:border-lunarc-accent rounded px-3 py-1.5 text-xs text-lunarc-text focus:outline-none"
                >
                  <option value="cq">CQ (恒定质量 · 推荐)</option>
                  <option value="vbr">VBR (动态码率)</option>
                  <option value="cbr">CBR (恒定码率)</option>
                  <option value="constqp">恒定 QP (Constant QP)</option>
                </select>
              </div>

              {txSettings.rate_mode === 'cq' && (
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <label className="text-[11px] text-lunarc-muted">CQ 质量数值 (0~{txSettings.codec === 'av1_nvenc' ? 63 : 51})</label>
                    <span className="text-xs font-mono text-lunarc-accent font-bold">{txSettings.cq}</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max={txSettings.codec === 'av1_nvenc' ? 63 : 51}
                    value={txSettings.cq}
                    onChange={(e) => updateSetting('cq', Number(e.target.value))}
                    className="w-full accent-lunarc-accent"
                  />
                </div>
              )}

              {txSettings.rate_mode === 'constqp' && (
                <div>
                  <label className="text-[11px] text-lunarc-muted mb-1 block">QP 数值</label>
                  <input
                    type="number"
                    min="0"
                    max="51"
                    value={txSettings.qp}
                    onChange={(e) => updateSetting('qp', Number(e.target.value))}
                    className="w-full bg-lunarc-dark border border-lunarc-border focus:border-lunarc-accent rounded px-3 py-1.5 text-xs text-lunarc-text focus:outline-none"
                  />
                </div>
              )}

              {(txSettings.rate_mode === 'vbr' || txSettings.rate_mode === 'cbr') && (
                <div>
                  <label className="text-[11px] text-lunarc-muted mb-1 block">目标码率</label>
                  <input
                    type="text"
                    value={txSettings.video_bitrate}
                    onChange={(e) => updateSetting('video_bitrate', e.target.value)}
                    placeholder="如 80M, 6000k"
                    className="w-full bg-lunarc-dark border border-lunarc-border focus:border-lunarc-accent rounded px-3 py-1.5 text-xs text-lunarc-text focus:outline-none"
                  />
                </div>
              )}

              {(txSettings.rate_mode === 'cq' || txSettings.rate_mode === 'vbr') && (
                <div>
                  <label className="text-[11px] text-lunarc-muted mb-1 block">峰值策略</label>
                  <select
                    value={txSettings.peak_policy}
                    onChange={(e) => updateSetting('peak_policy', e.target.value)}
                    className="w-full bg-lunarc-dark border border-lunarc-border focus:border-lunarc-accent rounded px-3 py-1.5 text-xs text-lunarc-text focus:outline-none"
                  >
                    <option value="auto">自动计算 (推荐)</option>
                    <option value="custom">手动填写</option>
                  </select>
                </div>
              )}
            </div>

            {txSettings.peak_policy === 'custom' && (txSettings.rate_mode === 'cq' || txSettings.rate_mode === 'vbr') && (
              <div className="grid grid-cols-2 gap-4 pt-2 border-t border-lunarc-border">
                <div>
                  <label className="text-[11px] text-lunarc-muted mb-1 block">Maxrate (最大瞬时码率)</label>
                  <input
                    type="text"
                    value={txSettings.maxrate}
                    onChange={(e) => updateSetting('maxrate', e.target.value)}
                    placeholder="如 380M"
                    className="w-full bg-lunarc-dark border border-lunarc-border focus:border-lunarc-accent rounded px-3 py-1.5 text-xs text-lunarc-text focus:outline-none"
                  />
                </div>
                <div>
                  <label className="text-[11px] text-lunarc-muted mb-1 block">Bufsize (VBV 缓冲区，最高 2000M)</label>
                  <input
                    type="text"
                    value={txSettings.bufsize}
                    onChange={(e) => updateSetting('bufsize', e.target.value)}
                    placeholder="如 1520M"
                    className="w-full bg-lunarc-dark border border-lunarc-border focus:border-lunarc-accent rounded px-3 py-1.5 text-xs text-lunarc-text focus:outline-none"
                  />
                </div>
              </div>
            )}
          </div>

          {/* Group 3: 缩放与色彩 (Scaling & Color) */}
          <div className={`bg-lunarc-surface border border-lunarc-border rounded-sm p-4 space-y-4 ${activeSection !== 'video' && activeSection !== 'color' ? 'hidden' : ''}`}>
            <div className="text-xs font-bold text-lunarc-text uppercase tracking-wider flex items-center space-x-1.5">
              <Tv className="w-4 h-4 text-lunarc-accent" />
              <span>3. 缩放与色彩 (Scaling & Color)</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
              <div className={activeSection === 'color' ? 'hidden' : ''}>
                <label className="text-[11px] text-lunarc-muted mb-1 block">分辨率</label>
                <select
                  value={txSettings.resolution_mode}
                  disabled={choices.resolution_mode?.length === 1}
                  onChange={(e) => updateSetting('resolution_mode', e.target.value)}
                  className="w-full bg-lunarc-dark border border-lunarc-border focus:border-lunarc-accent rounded px-3 py-1.5 text-xs text-lunarc-text focus:outline-none"
                >
                  <option value="original" disabled={Boolean(choices.resolution_mode && !choices.resolution_mode.includes("original"))}>{choices.gpu || choices.wide ? `渲染链输出 (${video ? `${outputDimensions.width}×${outputDimensions.height}` : '未加载素材'})` : `保持素材分辨率 (${video ? `${video.width}×${video.height}` : '未加载素材'})`}</option>
                  <option value="double" disabled={Boolean(choices.resolution_mode && !choices.resolution_mode.includes("double"))}>原始 ×2 超分辨率倍乘</option>
                  <option value="half" disabled={Boolean(choices.resolution_mode && !choices.resolution_mode.includes("half"))}>原始 ×0.5 降采倍乘</option>
                  <option value="5120x2880" disabled={Boolean(choices.resolution_mode && !choices.resolution_mode.includes("5120x2880"))}>5120×2880 (5K UHD)</option>
                  <option value="3840x2160" disabled={Boolean(choices.resolution_mode && !choices.resolution_mode.includes("3840x2160"))}>3840×2160 (4K UHD)</option>
                  <option value="2560x1440" disabled={Boolean(choices.resolution_mode && !choices.resolution_mode.includes("2560x1440"))}>2560×1440 (2K QHD)</option>
                  <option value="1920x1080" disabled={Boolean(choices.resolution_mode && !choices.resolution_mode.includes("1920x1080"))}>1920×1080 (1080p FHD)</option>
                  <option value="1280x720" disabled={Boolean(choices.resolution_mode && !choices.resolution_mode.includes("1280x720"))}>1280×720 (720p HD)</option>
                  <option value="custom_size" disabled={Boolean(choices.resolution_mode && !choices.resolution_mode.includes("custom_size"))}>自定义指定宽高</option>
                  <option value="custom_factor" disabled={Boolean(choices.resolution_mode && !choices.resolution_mode.includes("custom_factor"))}>自定义缩放倍数 (0~8x)</option>
                </select>
              </div>

              <div className={activeSection === 'color' ? 'hidden' : ''}>
                <label className="text-[11px] text-lunarc-muted mb-1 block">GPU 缩放算法</label>
                <select
                  value={txSettings.scale_algorithm}
                  onChange={(e) => updateSetting('scale_algorithm', e.target.value)}
                  className="w-full bg-lunarc-dark border border-lunarc-border focus:border-lunarc-accent rounded px-3 py-1.5 text-xs text-lunarc-text focus:outline-none"
                >
                  <option value="lanczos">GPU 兰索斯 (Lanczos · 锐利保真)</option>
                  <option value="bicubic">GPU 双三次 (Bicubic · 平滑柔和)</option>
                  <option value="bilinear">GPU 双线性 (Bilinear · 均衡)</option>
                  <option value="nearest">GPU 最近邻 (Nearest · 像素点对点)</option>
                </select>
              </div>

              <div className={activeSection === 'video' ? 'hidden' : ''}>
                <label className="text-[11px] text-lunarc-muted mb-1 block">像素格式</label>
                <select
                  value={txSettings.pixel_format}
                  onChange={(e) => updateSetting('pixel_format', e.target.value)}
                  className="w-full bg-lunarc-dark border border-lunarc-border focus:border-lunarc-accent rounded px-3 py-1.5 text-xs text-lunarc-text focus:outline-none"
                >
                  <option value="auto" disabled={Boolean(choices.pixel_format && !choices.pixel_format.includes("auto"))}>自动匹配处理精度</option>
                  <option value="yuv420p" disabled={Boolean(choices.pixel_format && !choices.pixel_format.includes("yuv420p"))}>YUV420P · 4:2:0 · 8-bit (最高兼容)</option>
                  <option value="nv12" disabled={Boolean(choices.pixel_format && !choices.pixel_format.includes("nv12"))}>NV12 · 4:2:0 · 8-bit (CUDA 原生)</option>
                  <option value="p010le" disabled={Boolean(choices.pixel_format && !choices.pixel_format.includes("p010le"))}>
                    P010 · 4:2:0 · 10-bit
                  </option>
                  <option value="p016le" disabled={Boolean(choices.pixel_format && !choices.pixel_format.includes("p016le"))}>P016 · 16-bit</option>
                  <option value="nv16" disabled={Boolean(choices.pixel_format && !choices.pixel_format.includes("nv16"))}>NV16 · 4:2:2 · 8-bit · 兼容转换</option>
                  <option value="p210le" disabled={Boolean(choices.pixel_format && !choices.pixel_format.includes("p210le"))}>P210 · 10-bit</option>
                  <option value="p216le" disabled={Boolean(choices.pixel_format && !choices.pixel_format.includes("p216le"))}>P216 · 16-bit</option>
                  <option value="yuv444p" disabled={Boolean(choices.pixel_format && !choices.pixel_format.includes("yuv444p"))}>YUV444P · 4:4:4 · 8-bit · 兼容转换</option>
                  <option value="yuv444p10msble" disabled={Boolean(choices.pixel_format && !choices.pixel_format.includes("yuv444p10msble"))}>YUV444P10MSB · 4:4:4 · 10-bit（已禁用）</option>
                  <option value="yuv444p16le" disabled={Boolean(choices.pixel_format && !choices.pixel_format.includes("yuv444p16le"))}>YUV444P16 · 4:4:4 · 16-bit（已禁用）</option>
                </select>
              </div>

              <div className={activeSection === 'video' ? 'hidden' : ''}>
                <label className="text-[11px] text-lunarc-muted mb-1 block">色彩规范</label>
                <select
                  value={txSettings.color_preset}
                  onChange={(e) => updateSetting('color_preset', e.target.value)}
                  className="w-full bg-lunarc-dark border border-lunarc-border focus:border-lunarc-accent rounded px-3 py-1.5 text-xs text-lunarc-text focus:outline-none"
                >
                  <option value="preserve" disabled={Boolean(choices.color_preset && !choices.color_preset.includes("preserve"))}>保留输入色彩元数据</option>
                  <option value="bt709_limited" disabled={Boolean(choices.color_preset && !choices.color_preset.includes("bt709_limited"))}>BT.709 SDR 有限范围 (标准电视)</option>
                  <option value="bt709_full" disabled={Boolean(choices.color_preset && !choices.color_preset.includes("bt709_full"))}>BT.709 SDR 全范围 (PC 显示器)</option>
                  <option value="bt2020_sdr" disabled={Boolean(choices.color_preset && !choices.color_preset.includes("bt2020_sdr"))}>BT.2020 SDR 广色域（8-bit 管线暂不支持，已禁用）</option>
                  <option value="bt2020_pq" disabled={Boolean(choices.color_preset && !choices.color_preset.includes("bt2020_pq"))}>BT.2020 PQ</option>
                  <option value="bt2020_hlg" disabled={Boolean(choices.color_preset && !choices.color_preset.includes("bt2020_hlg"))}>BT.2020 HLG</option>
                  <option value="hdr_to_sdr" disabled={Boolean(choices.color_preset && !choices.color_preset.includes("hdr_to_sdr"))}>HDR → SDR 色调映射 (libplacebo)</option>
                  <option value="custom" disabled={Boolean(choices.color_preset && !choices.color_preset.includes("custom"))}>自定义色彩矩阵与传递函数</option>
                </select>
              </div>
            </div>

            {/* Custom Dimensions Form */}
            {txSettings.resolution_mode === 'custom_size' && (
              <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4 pt-2 border-t border-lunarc-border">
                <div>
                  <label className="text-[11px] text-lunarc-muted mb-1 block">自定义宽度</label>
                  <input
                    type="number"
                    min="2"
                    step="2"
                    value={txSettings.custom_width}
                    onChange={(e) => updateSetting('custom_width', Number(e.target.value))}
                    className="w-full bg-lunarc-dark border border-lunarc-border focus:border-lunarc-accent rounded px-3 py-1.5 text-xs text-lunarc-text focus:outline-none"
                  />
                </div>
                <div>
                  <label className="text-[11px] text-lunarc-muted mb-1 block">自定义高度</label>
                  <input
                    type="number"
                    min="2"
                    step="2"
                    value={txSettings.custom_height}
                    onChange={(e) => updateSetting('custom_height', Number(e.target.value))}
                    className="w-full bg-lunarc-dark border border-lunarc-border focus:border-lunarc-accent rounded px-3 py-1.5 text-xs text-lunarc-text focus:outline-none"
                  />
                </div>
                <div className="flex items-center pt-5">
                  <label className="flex items-center space-x-2 text-xs text-lunarc-text cursor-pointer">
                    <input
                      type="checkbox"
                      checked={txSettings.keep_aspect}
                      onChange={(e) => updateSetting('keep_aspect', e.target.checked)}
                      className="accent-lunarc-accent rounded"
                    />
                    <span>保持等比例缩放 (防止画面拉伸变形)</span>
                  </label>
                </div>
              </div>
            )}

            {txSettings.resolution_mode === 'custom_factor' && (
              <div className="pt-2 border-t border-lunarc-border">
                <div className="flex justify-between items-center mb-1">
                  <label className="text-[11px] text-lunarc-muted">缩放倍率 (0.1 ~ 8.0x，强制偶数对齐)</label>
                  <span className="text-xs font-mono text-lunarc-accent font-bold">{txSettings.custom_factor.toFixed(2)}x</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="8.0"
                  step="0.05"
                  value={txSettings.custom_factor}
                  onChange={(e) => updateSetting('custom_factor', Number(e.target.value))}
                  className="w-full accent-lunarc-accent"
                />
              </div>
            )}
          </div>

          {/* Group 4: 音频与轨道 (Audio & Tracks) */}
          <div className={`bg-lunarc-surface border border-lunarc-border rounded-sm p-4 space-y-4 ${activeSection !== 'tracks' ? 'hidden' : ''}`}>
            <div className="text-xs font-bold text-lunarc-text uppercase tracking-wider flex items-center space-x-1.5">
              <Volume2 className="w-4 h-4 text-lunarc-accent" />
              <span>音轨与封装</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
              <div>
                <label className="text-[11px] text-lunarc-muted mb-1 block">音频编码模式</label>
                <select
                  value={txSettings.audio_mode}
                  onChange={(e) => updateSetting('audio_mode', e.target.value)}
                  className="w-full bg-lunarc-dark border border-lunarc-border focus:border-lunarc-accent rounded px-3 py-1.5 text-xs text-lunarc-text focus:outline-none"
                >
                  <option value="copy" disabled={Boolean(choices.audio_mode && !choices.audio_mode.includes("copy"))}>复制原音频流 (无损 Direct Stream Copy)</option>
                  <option value="aac" disabled={Boolean(choices.audio_mode && !choices.audio_mode.includes("aac"))}>重编码为 AAC</option>
                  <option value="opus" disabled={Boolean(choices.audio_mode && !choices.audio_mode.includes("opus"))}>重编码为 Opus</option>
                  <option value="flac" disabled={Boolean(choices.audio_mode && !choices.audio_mode.includes("flac"))}>重编码为 FLAC (无损压缩)</option>
                  <option value="pcm" disabled={Boolean(choices.audio_mode && !choices.audio_mode.includes("pcm"))}>重编码为 PCM 24-bit (广播母带)</option>
                  <option value="remove" disabled={Boolean(choices.audio_mode && !choices.audio_mode.includes("remove"))}>移除全部音频流 (静音输出)</option>
                </select>
              </div>

              {(txSettings.audio_mode === 'aac' || txSettings.audio_mode === 'opus') && (
                <div>
                  <label className="text-[11px] text-lunarc-muted mb-1 block">音频码率</label>
                  <select
                    value={txSettings.audio_bitrate}
                    onChange={(e) => updateSetting('audio_bitrate', e.target.value)}
                    className="w-full bg-lunarc-dark border border-lunarc-border focus:border-lunarc-accent rounded px-3 py-1.5 text-xs text-lunarc-text focus:outline-none"
                  >
                    <option value="128k">128 kbps (标准)</option>
                    <option value="192k">192 kbps (优质)</option>
                    <option value="256k">256 kbps (极高画质常用)</option>
                    <option value="320k">320 kbps (最高保真度)</option>
                    <option value="512k">512 kbps (环绕声 5.1/7.1)</option>
                  </select>
                </div>
              )}

              <div>
                <label className="text-[11px] text-lunarc-muted mb-1 block">字幕处理模式</label>
                <select
                  value={txSettings.subtitle_mode}
                  onChange={(e) => updateSetting('subtitle_mode', e.target.value)}
                  className="w-full bg-lunarc-dark border border-lunarc-border focus:border-lunarc-accent rounded px-3 py-1.5 text-xs text-lunarc-text focus:outline-none"
                >
                  <option value="copy">原样复制字幕流 (Copy)</option>
                  <option value="convert">转换为容器兼容格式 (SRT / WebVTT)</option>
                  <option value="drop">丢弃全部字幕轨 (Drop)</option>
                </select>
              </div>
            </div>

            {/* Checkboxes row */}
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4 pt-2 border-t border-lunarc-border">
              <label className="flex items-center space-x-2 text-xs text-lunarc-text cursor-pointer">
                <input
                  type="checkbox"
                  checked={txSettings.copy_attachments}
                  disabled={txSettings.container !== '.mkv'}
                  onChange={(e) => updateSetting('copy_attachments', e.target.checked)}
                  className="accent-lunarc-accent rounded"
                />
                <span>复制附件/字体 (MKV专属)</span>
              </label>

              <label className="flex items-center space-x-2 text-xs text-lunarc-text cursor-pointer">
                <input
                  type="checkbox"
                  checked={txSettings.copy_metadata}
                  onChange={(e) => updateSetting('copy_metadata', e.target.checked)}
                  className="accent-lunarc-accent rounded"
                />
                <span>保留视频元数据标签</span>
              </label>

              <label className="flex items-center space-x-2 text-xs text-lunarc-text cursor-pointer">
                <input
                  type="checkbox"
                  checked={txSettings.copy_chapters}
                  onChange={(e) => updateSetting('copy_chapters', e.target.checked)}
                  className="accent-lunarc-accent rounded"
                />
                <span>保留章节标记 (Chapters)</span>
              </label>

              <label className="flex items-center space-x-2 text-xs text-lunarc-text cursor-pointer">
                <input
                  type="checkbox"
                  checked={txSettings.skip_existing}
                  onChange={(e) => updateSetting('skip_existing', e.target.checked)}
                  className="accent-lunarc-accent rounded"
                />
                <span>跳过同名已存在文件</span>
              </label>
            </div>
          </div>
        </div>

        {/* Right Drawer / Preview Sidebar */}
        <div
          className={`border-l border-lunarc-border bg-lunarc-dark flex flex-col transition-all duration-300 ${
            advancedOpen ? 'w-[420px] shrink-0' : 'hidden'
          }`}
        >
          {/* If Advanced Drawer is OPEN */}
          {advancedOpen ? (
            <div className="flex-1 flex flex-col h-full overflow-hidden">
              <div className="p-4 border-b border-lunarc-border flex items-center justify-between flex-shrink-0 bg-lunarc-surface">
                <div className="flex items-center space-x-2">
                  <Settings2 className="w-4 h-4 text-lunarc-accent" />
                  <span className="text-xs font-bold text-lunarc-text uppercase tracking-wider">
                    高级 NVENC 硬件参数
                  </span>
                </div>
                <button
                  onClick={() => setAdvancedOpen(false)}
                  className="text-xs text-lunarc-muted hover:text-lunarc-text px-2 py-1 bg-lunarc-panel rounded"
                >
                  收起
                </button>
              </div>

              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {txSettings.codec !== 'av1_nvenc' && (
                  <div className="flex items-center justify-between p-2.5 bg-lunarc-surface border border-lunarc-border rounded">
                    <div>
                      <div className="text-xs font-medium text-lunarc-text">加权预测 (Weighted Pred)</div>
                      <div className="text-[10px] text-lunarc-dim">改善淡入淡出与渐变平滑度</div>
                    </div>
                    <input
                      type="checkbox"
                      checked={txSettings.weighted_prediction}
                      onChange={(e) => updateSetting('weighted_prediction', e.target.checked)}
                      className="accent-lunarc-accent w-4 h-4 rounded"
                    />
                  </div>
                )}

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-[11px] text-lunarc-muted mb-1 block">B 帧数量</label>
                    <select
                      value={txSettings.b_frames}
                      disabled={txSettings.weighted_prediction}
                      onChange={(e) => updateSetting('b_frames', e.target.value)}
                      className="w-full bg-lunarc-dark border border-lunarc-border rounded px-2.5 py-1.5 text-xs text-lunarc-text focus:outline-none disabled:opacity-50"
                    >
                      <option value="auto">自动 (Auto)</option>
                      <option value="0">0 帧 (无 B 帧)</option>
                      <option value="1">1 帧</option>
                      <option value="2">2 帧</option>
                      <option value="3">3 帧</option>
                      <option value="4">4 帧 (UHQ 推荐)</option>
                      <option value="5">5 帧</option>
                      <option value="6">6 帧</option>
                      <option value="7">7 帧</option>
                    </select>
                  </div>

                  <div>
                    <label className="text-[11px] text-lunarc-muted mb-1 block">B 帧参考模式</label>
                    <select
                      value={txSettings.b_ref_mode}
                      disabled={txSettings.weighted_prediction}
                      onChange={(e) => updateSetting('b_ref_mode', e.target.value)}
                      className="w-full bg-lunarc-dark border border-lunarc-border rounded px-2.5 py-1.5 text-xs text-lunarc-text focus:outline-none disabled:opacity-50"
                    >
                      <option value="auto" disabled={Boolean(choices.b_ref_mode && !choices.b_ref_mode.includes("auto"))}>自动 (Auto)</option>
                      <option value="disabled" disabled={Boolean(choices.b_ref_mode && !choices.b_ref_mode.includes("disabled"))}>关闭 (Disabled)</option>
                      <option value="middle" disabled={Boolean(choices.b_ref_mode && !choices.b_ref_mode.includes("middle"))}>中间 B 帧 (Middle)</option>
                      {txSettings.codec !== 'av1_nvenc' && <option value="each" disabled={Boolean(choices.b_ref_mode && !choices.b_ref_mode.includes("each"))}>每个 B 帧 (Each)</option>}
                    </select>
                  </div>

                  <div>
                    <label className="text-[11px] text-lunarc-muted mb-1 block">自适应量化</label>
                    <select
                      value={txSettings.aq_mode}
                      onChange={(e) => updateSetting('aq_mode', e.target.value)}
                      className="w-full bg-lunarc-dark border border-lunarc-border rounded px-2.5 py-1.5 text-xs text-lunarc-text focus:outline-none"
                    >
                      <option value="spatial">Spatial AQ (空间自适应)</option>
                      <option value="temporal">Temporal AQ (时域自适应)</option>
                      <option value="disabled">关闭 (Disabled)</option>
                    </select>
                  </div>

                  <div>
                    <label className="text-[11px] text-lunarc-muted mb-1 block">AQ 强度 (1~15)</label>
                    <input
                      type="number"
                      min="1"
                      max="15"
                      disabled={txSettings.aq_mode !== 'spatial'}
                      value={txSettings.aq_strength}
                      onChange={(e) => updateSetting('aq_strength', Number(e.target.value))}
                      className="w-full bg-lunarc-dark border border-lunarc-border rounded px-2.5 py-1.5 text-xs text-lunarc-text focus:outline-none disabled:opacity-50"
                    />
                  </div>

                  <div>
                    <label className="text-[11px] text-lunarc-muted mb-1 block">GOP 模式</label>
                    <select
                      value={txSettings.gop_mode}
                      onChange={(e) => updateSetting('gop_mode', e.target.value)}
                      className="w-full bg-lunarc-dark border border-lunarc-border rounded px-2.5 py-1.5 text-xs text-lunarc-text focus:outline-none"
                    >
                      <option value="auto">自动 (基于 2 秒帧率)</option>
                      <option value="manual">手动指定帧数</option>
                    </select>
                  </div>

                  <div>
                    <label className="text-[11px] text-lunarc-muted mb-1 block">GOP 关键帧间隔</label>
                    <input
                      type="number"
                      min="1"
                      disabled={txSettings.gop_mode !== 'manual'}
                      value={txSettings.gop}
                      onChange={(e) => updateSetting('gop', Number(e.target.value))}
                      className="w-full bg-lunarc-dark border border-lunarc-border rounded px-2.5 py-1.5 text-xs text-lunarc-text focus:outline-none disabled:opacity-50"
                    />
                  </div>

                  <div>
                    <label className="text-[11px] text-lunarc-muted mb-1 block">Lookahead (前瞻)</label>
                    <select
                      value={txSettings.lookahead}
                      onChange={(e) => updateSetting('lookahead', e.target.value)}
                      className="w-full bg-lunarc-dark border border-lunarc-border rounded px-2.5 py-1.5 text-xs text-lunarc-text focus:outline-none"
                    >
                      <option value="auto">自动 (Auto)</option>
                      <option value="0">0 帧</option>
                      <option value="8">8 帧</option>
                      <option value="16">16 帧</option>
                      <option value="24">24 帧</option>
                      <option value="32">32 帧 (最高分析度)</option>
                    </select>
                  </div>

                  <div>
                    <label className="text-[11px] text-lunarc-muted mb-1 block">Lookahead Level</label>
                    <select
                      value={txSettings.lookahead_level}
                      onChange={(e) => updateSetting('lookahead_level', e.target.value)}
                      className="w-full bg-lunarc-dark border border-lunarc-border rounded px-2.5 py-1.5 text-xs text-lunarc-text focus:outline-none"
                    >
                      <option value="auto">自动 (Auto)</option>
                      <option value="0">0 (基础)</option>
                      <option value="1">1 (适度)</option>
                      <option value="2">2 (深度)</option>
                      <option value="3">3 (极致)</option>
                    </select>
                  </div>

                  <div>
                    <label className="text-[11px] text-lunarc-muted mb-1 block">时间滤波</label>
                    <select
                      value={txSettings.temporal_filter}
                      onChange={(e) => updateSetting('temporal_filter', e.target.value)}
                      className="w-full bg-lunarc-dark border border-lunarc-border rounded px-2.5 py-1.5 text-xs text-lunarc-text focus:outline-none"
                    >
                      <option value="auto">自动 (Auto)</option>
                      <option value="0">0 (关闭降噪)</option>
                      <option value="1">1 级</option>
                      <option value="2">2 级</option>
                      <option value="3">3 级</option>
                      <option value="4">4 级</option>
                    </select>
                  </div>

                  <div>
                    <label className="text-[11px] text-lunarc-muted mb-1 block">分路编码</label>
                    <select
                      value={txSettings.split_encode}
                      disabled={txSettings.codec === 'hevc_nvenc' && txSettings.weighted_prediction}
                      onChange={(e) => updateSetting('split_encode', e.target.value)}
                      className="w-full bg-lunarc-dark border border-lunarc-border rounded px-2.5 py-1.5 text-xs text-lunarc-text focus:outline-none disabled:opacity-50"
                    >
                      <option value="auto">自动 (Auto)</option>
                      <option value="disabled">关闭 (Disabled)</option>
                      <option value="2">强制双路 (Dual NVENC)</option>
                      <option value="3">强制三路</option>
                      <option value="4">强制四路</option>
                    </select>
                  </div>

                  <div>
                    <label className="text-[11px] text-lunarc-muted mb-1 block">Profile</label>
                    <select
                      value={txSettings.profile}
                      onChange={(e) => updateSetting('profile', e.target.value)}
                      className="w-full bg-lunarc-dark border border-lunarc-border rounded px-2.5 py-1.5 text-xs text-lunarc-text focus:outline-none"
                    >
                      <option value="auto" disabled={Boolean(choices.profile && !choices.profile.includes("auto"))}>自动 (Auto)</option>
                      <option value="main" disabled={Boolean(choices.profile && !choices.profile.includes("main"))}>Main (8-bit)</option>
                      {/* Main 10 is the profile the experimental HDR10 entry encodes;
                          on the 8-bit SDR pipe it would name a profile the frames
                          cannot fill, so it is offered on the branch that means it. */}
                      <option value="main10" disabled={Boolean(choices.profile && !choices.profile.includes("main10"))}>
                        Main10 · 10-bit
                      </option>
                      <option value="high" disabled={Boolean(choices.profile && !choices.profile.includes("high"))}>High (兼容广播)</option>
                    </select>
                  </div>

                  <div>
                    <label className="text-[11px] text-lunarc-muted mb-1 block">Tier (仅 AV1/HEVC)</label>
                    <select
                      value={txSettings.tier}
                      onChange={(e) => updateSetting('tier', e.target.value)}
                      className="w-full bg-lunarc-dark border border-lunarc-border rounded px-2.5 py-1.5 text-xs text-lunarc-text focus:outline-none"
                    >
                      <option value="auto">自动 (Auto)</option>
                      <option value="main">Main Tier</option>
                      <option value="high">High Tier</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="text-[11px] text-lunarc-muted mb-1 block">额外安全参数 (安全透传)</label>
                  <input
                    type="text"
                    value={txSettings.extra_args}
                    disabled={choices.wide}
                    onChange={(e) => updateSetting('extra_args', e.target.value)}
                    placeholder="如 -surfaces 32"
                    className="w-full bg-lunarc-dark border border-lunarc-border rounded px-2.5 py-1.5 text-xs text-lunarc-text focus:outline-none font-mono"
                  />
                </div>


              </div>
            </div>
          ) : null}

        </div>
        </div>
        <div className="deliver-submit border-t border-lunarc-border px-6 py-4 space-y-2">
          {validationIssues.length > 0 && <p role="alert" className="text-sm text-rose-400">{validationIssues[0].text}</p>}
              {/* Add to Queue Button */}
              <button
                onClick={handleAddToQueue}
                disabled={isSubmitting || !isValid || !video}
                className={`w-full py-3 rounded-sm font-bold text-xs uppercase tracking-wider flex items-center justify-center space-x-2 transition shadow-none ${
                  justAdded
                    ? 'bg-emerald-600 text-white shadow-emerald-950/50'
                    : isValid && video
                    ? 'bg-lunarc-accent text-[var(--accent-ink)] hover:brightness-110 active:scale-[0.99] cursor-pointer'
                    : 'bg-zinc-800 hover:bg-lunarc-border text-lunarc-text border border-lunarc-border cursor-pointer'
                } ${isSubmitting ? 'opacity-70 cursor-wait' : ''}`}
              >
                {isSubmitting ? (
                  <Activity className="w-4 h-4 animate-spin text-lunarc-accent" />
                ) : justAdded ? (
                  <CheckCircle className="w-4 h-4 text-white" />
                ) : (
                  <Plus className="w-4 h-4" />
                )}
                <span>
                  {isSubmitting
                    ? '正在添加到队列...'
                    : justAdded
                    ? '已加入队列'
                    : '加入队列'}
                </span>
              </button>
        </div>
      </div>

      {/* Persistent Render Queue Monitor */}
      <div className="deliver-queue border-l border-lunarc-border bg-lunarc-dark flex flex-col min-h-0 min-w-0">
        {/* Queue Header Controls */}
        <div className="deliver-queue-header h-10 px-4 border-b border-lunarc-border bg-lunarc-surface flex items-center justify-between flex-shrink-0">
          <div className="flex items-center space-x-3">
            <span className="text-xs font-bold text-lunarc-text uppercase tracking-wide flex items-center space-x-1.5">
              <Layers className="w-3.5 h-3.5 text-lunarc-accent" />
              <span>导出队列</span>
            </span>
            <span className="text-[11px] text-lunarc-dim">
              共 {jobs.length} 个任务 · {jobs.filter((j) => j.status === 'completed').length} 已完成
              {jobs.filter((j) => j.status === 'skipped').length > 0 && (
                <> · {jobs.filter((j) => j.status === 'skipped').length} 已跳过（未渲染）</>
              )}
              {jobs.filter((j) => j.status === 'failed').length > 0 && (
                <> · {jobs.filter((j) => j.status === 'failed').length} 失败</>
              )}
              {queueRunning && <span className="text-lunarc-accent"> · 队列运行中</span>}
            </span>
          </div>

          <div className="flex items-center space-x-2">
            {activeJobId ? (
              <button
                onClick={handleCancelJob}
                className="flex items-center space-x-1.5 px-3 py-1 bg-rose-950/80 hover:bg-rose-900 text-rose-200 border border-rose-800/80 rounded text-xs transition"
              >
                <Square className="w-3.5 h-3.5" />
                <span>终止当前任务</span>
              </button>
            ) : (
              <button
                onClick={handleStartQueue}
                aria-label="全部渲染 (Render All)"
                disabled={pendingCount === 0 || queueRunning}
                title={
                  queueRunning
                    ? '本轮渲染尚未结束；按钮只启动【待渲染】任务，已结束的任务请用行内【重试】显式重置'
                    : pendingCount === 0
                    ? '没有待渲染任务：本按钮只渲染【待渲染】，失败/已取消/已跳过的任务需用行内【重试】重置'
                    : undefined
                }
                className={`flex items-center space-x-1.5 px-4 py-1 rounded text-xs font-bold transition ${
                  pendingCount > 0 && !queueRunning
                    ? 'bg-lunarc-accent hover:bg-[#3fb99f] text-[var(--accent-ink)]'
                    : 'bg-zinc-800 text-lunarc-dim cursor-not-allowed'
                }`}
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>开始导出</span>
              </button>
            )}
          </div>
        </div>

        {/* Queue List Table */}
        <div className="flex-1 overflow-y-auto">
          {jobs.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-lunarc-dim space-y-2">
              <Film className="w-8 h-8 opacity-30" />
              <span className="text-xs">队列为空，设置完成后点击「加入队列」</span>
            </div>
          ) : (
            <div className="divide-y divide-[#27272a]">
              {jobs.map((job) => {
                const isRunning = job.status === 'running'
                return (
                  <div
                    key={job.id}
                    className={`deliver-queue-job px-4 py-3 flex items-center space-x-3 hover:bg-lunarc-surface/50 transition ${
                      isRunning ? 'bg-lunarc-surface' : ''
                    }`}
                  >
                    {/* Status Icon */}
                    <div className="deliver-job-status flex-shrink-0">
                      {job.status === 'completed' ? (
                        <CheckCircle className="w-5 h-5 text-emerald-400" />
                      ) : isRunning ? (
                        <Activity className="w-5 h-5 text-lunarc-accent animate-pulse" />
                      ) : job.status === 'failed' ? (
                        <AlertTriangle className="w-5 h-5 text-rose-400" />
                      ) : job.status === 'skipped' ? (
                        <MinusCircle className="w-5 h-5 text-amber-400" />
                      ) : (
                        <Clock className="w-5 h-5 text-lunarc-dim" />
                      )}
                    </div>

                    {/* Job Info */}
                    <div className="deliver-job-info w-64 flex-shrink-0">
                      <div className="text-xs font-bold text-lunarc-text truncate">{job.name}</div>
                      <div className="text-[11px] text-lunarc-dim truncate">{job.output_path || '默认目录'}</div>
                    </div>

                    {/* Transcode Details */}
                    <div className="deliver-job-details w-48 flex-shrink-0 text-[11px] text-lunarc-muted space-y-0.5">
                      <div>
                        {job.codec?.toUpperCase()} · {job.container?.toUpperCase()} · {job.preset?.toUpperCase()}
                      </div>
                      <div className="text-lunarc-dim">
                        CQ {job.cq} · {job.with_audio ? '含音频' : '无音频'}
                      </div>
                    </div>

                    {/* Progress Bar & Telemetry */}
                    <div className="deliver-job-progress flex-1 space-y-1.5">
                      <div className="flex justify-between items-center text-[11px]">
                        {job.status === 'failed' ? (
                          <div className="flex items-center gap-2 max-w-[80%]">
                            <span className="text-rose-400 font-medium truncate">
                              失败: {job.error || '未知异常中断'}
                            </span>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                setSelectedFailedJob(job)
                              }}
                              className="px-2 py-0.5 bg-rose-950/90 hover:bg-rose-900 border border-rose-700/60 text-rose-300 rounded text-[10px] font-medium transition flex items-center gap-1 flex-shrink-0 cursor-pointer shadow-sm"
                            >
                              <AlertTriangle className="w-3 h-3 text-rose-400" />
                              查看诊断调用栈
                            </button>
                          </div>
                        ) : (
                          <span className="text-lunarc-muted">
                            {isRunning
                              ? `正在渲染: 帧 ${job.current_frame} / ${job.total_frames}`
                              : job.status === 'completed'
                              ? '已成功导出完成'
                              : job.status === 'skipped'
                              ? '已跳过：目标文件已存在，未渲染也未覆盖任何内容'
                              : job.status === 'cancelled'
                              ? '已取消：本轮被终止'
                              : '等待渲染'}
                          </span>
                        )}
                        <span className="font-mono text-lunarc-text font-bold">{job.progress.toFixed(1)}%</span>
                      </div>
                      <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                        <div
                          className={`h-full transition-all duration-200 ${
                            job.status === 'completed'
                              ? 'bg-emerald-500'
                              : job.status === 'failed'
                              ? 'bg-rose-500'
                              : job.status === 'skipped'
                              ? 'bg-amber-500'
                              : 'bg-lunarc-accent'
                          }`}
                          style={{ width: `${job.progress}%` }}
                        />
                      </div>
                    </div>

                    {/* Telemetry (FPS & ETA) */}
                    <div className="deliver-job-telemetry w-36 flex-shrink-0 text-right font-mono text-[11px] text-lunarc-muted">
                      {isRunning && (
                        <>
                          <div className="text-lunarc-accent font-bold">{job.fps} FPS</div>
                          <div className="text-lunarc-dim">ETA: {job.eta}s</div>
                        </>
                      )}
                      {job.status === 'completed' && <span className="text-emerald-400">完成</span>}
                      {job.status === 'pending' && <span className="text-lunarc-dim">队列就绪</span>}
                      {job.status === 'skipped' && (
                        <span className="text-amber-400" title="目标文件保持原样，本任务未编码">
                          已跳过 · 未写入
                        </span>
                      )}
                      {job.status === 'cancelled' && <span className="text-lunarc-dim">已取消</span>}
                    </div>

                    {/* Action */}
                    <div className="deliver-job-actions flex-shrink-0 flex items-center gap-1">
                      {job.status === 'completed' && <button type="button" className="button quiet"
                        onClick={() => window.dispatchEvent(new CustomEvent('lunarc:compare', {
                          detail: {original: job.input_path, exported: job.output_path}}))}>查看对比</button>}
                      <button type="button" onClick={() => inspectJob(job)}
                        className="rounded border border-zinc-600 px-2 py-1 text-[10px] text-lunarc-text hover:border-lunarc-accent hover:text-lunarc-accent"
                        aria-label={`查看 ${job.name} 的实际运行诊断`}>
                        运行诊断
                      </button>
                      {RETRYABLE_STATUSES.includes(job.status) && (
                        <button
                          onClick={() => handleRetryJob(job.id)}
                          disabled={queueRunning || isRunning || retryingJobId === job.id}
                          aria-label="重试任务 (Retry Job)"
                          title={
                            queueRunning || isRunning
                              ? '本轮渲染结束前不能重置任务：队列仍占用 GPU 会话'
                              : '把该任务重置为待渲染（仅此任务；不会重跑其它已完成或失败的任务）'
                          }
                          className={`flex items-center space-x-1 px-2 py-1 rounded text-[10px] font-medium transition border ${
                            queueRunning || isRunning
                              ? 'bg-zinc-800 text-lunarc-dim border-lunarc-border cursor-not-allowed'
                              : 'bg-zinc-800/80 hover:bg-lunarc-border text-lunarc-text border-zinc-600 cursor-pointer'
                          }`}
                        >
                          <RotateCcw className="w-3.5 h-3.5" />
                          <span>{retryingJobId === job.id ? '重置中…' : '重试'}</span>
                        </button>
                      )}
                      <button
                        onClick={() => handleRemoveJob(job.id)}
                        disabled={isRunning}
                        aria-label={`删除 ${job.name}`}
                        className="p-1.5 hover:bg-zinc-800 text-lunarc-muted hover:text-rose-400 rounded transition disabled:opacity-30"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                )
              })}
            </div>
          )}

          {/* Live Diagnostic Console Dock */}
          <div className="mt-4 border border-lunarc-border rounded-sm overflow-hidden bg-[#111114]">
            <div
              onClick={() => setConsoleOpen(!consoleOpen)}
              role="button" tabIndex={0} aria-expanded={consoleOpen} aria-label="切换渲染日志显示"
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setConsoleOpen(!consoleOpen) } }}
              className="flex items-center justify-between px-3 py-2 bg-[#18181c] cursor-pointer hover:bg-[#1f1f24] transition select-none"
            >
              <div className="flex items-center gap-2 text-xs font-mono text-lunarc-text">
                <Terminal className="w-4 h-4 text-emerald-400" />
                <span className="font-bold">系统与渲染诊断终端 (Live Diagnostic Console)</span>
                <span className="text-[10px] px-1.5 py-0.2 bg-zinc-800 rounded text-lunarc-muted">
                  {diagnosticLogs.length} 条日志
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    refreshLogs()
                  }}
                  className="p-1 hover:bg-lunarc-border text-lunarc-muted rounded transition"
                  title="刷新日志"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                </button>
                <button type="button" onClick={async (e) => {
                  e.stopPropagation()
                  const body = diagnosticLogs.map((l) => `${l.time} ${l.level} [${l.tag}] ${l.message}${l.detail ? `\n${l.detail}` : ''}`).join('\n')
                  try { await navigator.clipboard.writeText(body); setCopiedDiag(true); setTimeout(() => setCopiedDiag(false), 1800) }
                  catch { onStatusChange?.('无法复制日志：请允许此页面访问剪贴板。') }
                }} className="rounded p-1 text-lunarc-muted hover:bg-lunarc-border hover:text-lunarc-text" aria-label="复制诊断日志">
                  {copiedDiag ? <Check className="h-3.5 w-3.5 text-lunarc-accent" /> : <Copy className="h-3.5 w-3.5" />}
                </button>
                <span className="text-[11px] text-lunarc-dim font-mono">
                  {consoleOpen ? '▲ 收起' : '▼ 展开'}
                </span>
              </div>
            </div>

            {consoleOpen && (
              <div className="p-3 bg-lunarc-surface font-mono text-[11px] max-h-64 overflow-y-auto space-y-1 select-text">
                {diagnosticLogs.length === 0 ? (
                  <div className="text-lunarc-dim italic py-2">暂无系统诊断日志</div>
                ) : (
                  diagnosticLogs.map((l, i) => (
                    <div key={i} className="flex items-start gap-2 leading-tight">
                      <span className="text-lunarc-dim flex-shrink-0">{l.time}</span>
                      <span
                        className={`px-1 py-0.2 rounded text-[10px] font-bold flex-shrink-0 ${
                          l.level === 'ERROR' || l.level === 'FATAL'
                            ? 'bg-rose-950 text-rose-300 border border-rose-800'
                            : l.level === 'WARN'
                            ? 'bg-amber-950 text-amber-300 border border-amber-800'
                            : 'bg-zinc-800 text-emerald-400'
                        }`}
                      >
                        {l.level}
                      </span>
                      <span className="text-lunarc-muted font-semibold flex-shrink-0">[{l.tag}]</span>
                      <span
                        className={
                          l.level === 'ERROR' || l.level === 'FATAL'
                            ? 'text-rose-300 font-semibold'
                            : l.level === 'WARN'
                            ? 'text-amber-200'
                            : 'text-lunarc-text'
                        }
                      >
                        {l.message}
                      </span>
                      {l.detail && (
                        <div className="text-lunarc-dim text-[10px] mt-0.5 pl-4 border-l border-lunarc-border whitespace-pre-wrap">
                          {l.detail}
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        </div>
      </div>
      </div>

      {diagnosticJob && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4" role="dialog" aria-modal="true" aria-labelledby="job-diagnostics-title">
          <section className="flex max-h-[85vh] w-full max-w-3xl flex-col overflow-hidden rounded-xl border border-lunarc-border bg-lunarc-surface shadow-2xl">
            <header className="flex items-center justify-between border-b border-lunarc-border px-5 py-4">
              <div><h2 id="job-diagnostics-title" className="font-semibold text-lunarc-text">运行诊断 · {diagnosticJob.name}</h2><p className="mt-1 text-xs text-lunarc-dim">来自本地后端当前可读取的作业记录；不代表跨后端重启的归档历史。</p></div>
              <button onClick={closeDiagnostics} className="rounded p-2 text-lunarc-muted hover:bg-lunarc-border" aria-label="关闭运行诊断"><X className="h-4 w-4" /></button>
            </header>
            <div className="space-y-3 overflow-y-auto p-5 text-sm">
              {diagnosticsLoading && <p className="text-lunarc-muted">正在读取后端诊断…</p>}
              {diagnosticsError && <p role="alert" className="rounded border border-rose-800 bg-rose-950/40 p-3 text-rose-200">{diagnosticsError}</p>}
              {!diagnosticsLoading && !diagnosticsError && (() => {
                const record = jobDiagnostics?.job || {}
                const run = record.residual_backend_diagnostics || record.diagnostics || {}
                const requested = record.settings?.fusion_mode ?? record.transcode_settings?.fusion_mode ?? diagnosticJob.settings?.fusion_mode ?? '未记录'
                const actual = run.fusion_mode ?? run.actual_fusion_mode ?? '后端未报告'
                const requestedExport = record.settings?.export_backend ?? diagnosticJob.settings?.export_backend ?? '未记录'
                const actualExport = run.export_backend ?? run.actual_export_backend ?? record.export_backend ?? '后端未报告'
                return <>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="rounded border border-lunarc-border bg-lunarc-panel p-3"><div className="text-xs text-lunarc-dim">导出后端（请求）</div><div className="mt-1 font-mono text-lunarc-text">{String(requestedExport)}</div></div>
                    <div className="rounded border border-lunarc-border bg-lunarc-panel p-3"><div className="text-xs text-lunarc-dim">导出后端（实际报告）</div><div className="mt-1 font-mono text-lunarc-accent">{String(actualExport)}</div></div>
                    <div className="rounded border border-lunarc-border bg-lunarc-panel p-3"><div className="text-xs text-lunarc-dim">融合模式（请求）</div><div className="mt-1 font-mono text-lunarc-text">{String(requested)}</div></div>
                    <div className="rounded border border-lunarc-border bg-lunarc-panel p-3"><div className="text-xs text-lunarc-dim">融合模式（实际报告）</div><div className="mt-1 font-mono text-lunarc-accent">{String(actual)}</div></div>
                  </div>
                  <p className="text-xs leading-relaxed text-lunarc-muted">传输诊断只展示后端实际报告的边界计数；标量边界与整帧图像边界是不同指标，不将标量传输描述成图像零拷贝。</p>
                  <pre className="max-h-[45vh] overflow-auto rounded border border-lunarc-border bg-black/60 p-3 text-[11px] text-lunarc-text">{JSON.stringify(run, null, 2)}</pre>
                  {!Object.keys(run).length && <p className="text-xs text-amber-300">该作业当前没有可用的融合/传输诊断字段。</p>}
                </>
              })()}
              <button type="button" onClick={() => inspectJob(diagnosticJob)} disabled={diagnosticsLoading} className="rounded border border-zinc-600 px-3 py-2 text-xs text-lunarc-text hover:border-lunarc-accent disabled:opacity-50">重新读取</button>
            </div>
          </section>
        </div>
      )}

      {/* Failure Diagnostics Detail Modal */}
      {selectedFailedJob && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-lunarc-surface border border-rose-600/40 rounded-xl max-w-2xl w-full max-h-[85vh] flex flex-col shadow-2xl overflow-hidden">
            {/* Modal Header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-lunarc-border bg-rose-950/20">
              <div className="flex items-center gap-2 text-rose-300 font-bold">
                <AlertTriangle className="w-5 h-5 text-rose-400" />
                <span>渲染失败诊断报告 — {selectedFailedJob.name}</span>
              </div>
              <button
                onClick={() => setSelectedFailedJob(null)}
                className="p-1 hover:bg-zinc-800 rounded text-lunarc-muted hover:text-lunarc-text transition"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-5 overflow-y-auto space-y-4 font-mono text-xs flex-1">
              <div>
                <div className="text-lunarc-dim text-[10px] uppercase tracking-wider mb-1">作业信息</div>
                <div className="bg-zinc-900/80 p-2.5 rounded border border-lunarc-border text-lunarc-text space-y-1">
                  <div>ID: {selectedFailedJob.id}</div>
                  <div>源文件: {selectedFailedJob.input_path}</div>
                  <div>目标文件: {selectedFailedJob.output_path}</div>
                  <div>编码配置: {selectedFailedJob.codec} · {selectedFailedJob.container} · CQ{selectedFailedJob.cq}</div>
                </div>
              </div>

              <div>
                <div className="text-rose-400 font-semibold text-[10px] uppercase tracking-wider mb-1">失败原因</div>
                <div className="bg-rose-950/40 p-3 rounded border border-rose-800/60 text-rose-200 font-medium">
                  {selectedFailedJob.error || '未捕获的具体错误'}
                </div>
              </div>

              {selectedFailedJob.error_traceback && (
                <div>
                  <div className="text-lunarc-dim text-[10px] uppercase tracking-wider mb-1">Python 调用栈 (Traceback)</div>
                  <pre className="bg-black/80 p-3 rounded border border-lunarc-border text-lunarc-text text-[11px] overflow-x-auto whitespace-pre leading-relaxed">
                    {selectedFailedJob.error_traceback}
                  </pre>
                </div>
              )}

              {(selectedFailedJob.dlss_host || selectedFailedJob.dlss_host_error) && (
                <div>
                  <div className="text-lunarc-dim text-[10px] uppercase tracking-wider mb-1">DLSS 宿主上报 (dlss_host)</div>
                  <pre className="bg-black/80 p-3 rounded border border-lunarc-border text-amber-300/80 text-[11px] overflow-x-auto whitespace-pre leading-relaxed">
                    {`dll: ${selectedFailedJob.dlss_host?.dll || '未记录'}\nabi: ${selectedFailedJob.dlss_host?.abi ?? '未记录'}\nbackend: ${selectedFailedJob.dlss_host?.backend || '未记录'}\nlast_error: ${selectedFailedJob.dlss_host_error || '宿主未上报'}`}
                  </pre>
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div className="px-5 py-3 border-t border-lunarc-border bg-[#141416] flex justify-between items-center">
              <span className="text-[11px] text-lunarc-dim">
                可将诊断信息复制后用于排查或向工程师反馈
              </span>
              <div className="flex gap-2">
                <button
                  onClick={() => {
                    const text = `=== DLSS 5 渲染失败诊断报告 ===\n作业ID: ${selectedFailedJob.id}\n作业名称: ${selectedFailedJob.name}\n输入: ${selectedFailedJob.input_path}\n输出: ${selectedFailedJob.output_path}\n错误信息: ${selectedFailedJob.error}\n\n[Traceback]\n${selectedFailedJob.error_traceback || '无'}\n\n[DLSS 宿主上报]\n${selectedFailedJob.dlss_host ? `dll=${selectedFailedJob.dlss_host.dll || '未记录'} abi=${selectedFailedJob.dlss_host.abi ?? '未记录'} backend=${selectedFailedJob.dlss_host.backend || '未记录'}` : '无'}\nlast_error=${selectedFailedJob.dlss_host_error || '无'}`
                    navigator.clipboard.writeText(text)
                    setCopiedDiag(true)
                    setTimeout(() => setCopiedDiag(false), 2500)
                  }}
                  className="px-3 py-1.5 bg-zinc-800 hover:bg-lunarc-border text-lunarc-text rounded text-xs font-medium transition flex items-center gap-1.5"
                >
                  {copiedDiag ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                  {copiedDiag ? '已复制诊断报告' : '复制诊断报告'}
                </button>
                <button
                  onClick={() => setSelectedFailedJob(null)}
                  className="px-4 py-1.5 bg-lunarc-border hover:bg-zinc-600 text-lunarc-text rounded text-xs font-medium transition"
                >
                  关闭
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
})
