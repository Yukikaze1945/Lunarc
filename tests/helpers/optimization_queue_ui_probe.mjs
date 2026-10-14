// Real short GPU Transfer delivery through the built Electron app. Only native
// file/directory selection is mocked; the backend queue, encoder and export are real.
import { _electron as electron } from '@playwright/test'
import path from 'node:path'
import crypto from 'node:crypto'
import fs from 'node:fs/promises'
import { spawnSync } from 'node:child_process'
import { ROOT_DIR, pythonExe } from './paths.js'

const out = path.resolve(process.argv[2] || path.join(ROOT_DIR, 'bench_out/optimization_20261009/electron_queue_ui'))
const cancelRetry = process.argv.includes('--cancel-retry')
const fixture = path.join(ROOT_DIR, 'tests/fixtures/fixture_gpu_transfer_short.mkv')
const ffprobe = process.env.FFPROBE_PATH || 'ffprobe'
const portCheck = await fetch('http://127.0.0.1:18923/api/system').then(() => true).catch(() => false)
if (portCheck) throw new Error('Port 18923 already has a backend. Refusing to attach to or stop another user session.')
try {
  const prior = await fs.readdir(out)
  if (prior.length) throw new Error(`Evidence directory is not fresh: ${out}`)
} catch (error) {
  if (error.code !== 'ENOENT') throw error
}
await fs.mkdir(out, { recursive: true })
await fs.mkdir(path.join(out, 'user_data'), { recursive: true })
const userData = path.join(out, 'user_data')
const outputName = 'fixture_gpu_transfer_short_ui_hevc_cq24'
const outputFile = path.join(out, `${outputName}.mkv`)
const token = `queue-ui-${crypto.randomUUID()}`
const errors = []
const interactions = []
let app
let backendReady = false
let result = { mode: cancelRetry ? 'cancel-retry' : 'complete', actualElectron: true, actualBackend: false, actualGpuExport: false, fixture, outputFile, userData, interactions, errors }

const draft = {
  version: 1,
  settings: {
    codec: 'hevc_nvenc', container: '.mkv', suffix: '_ui_hevc_cq24',
    rate_mode: 'cq', cq: 24, tune: 'hq', preset: 'p7', multipass: 'fullres',
    pixel_format: 'auto', color_preset: 'preserve', audio_mode: 'copy',
    subtitle_mode: 'copy', copy_attachments: true, copy_metadata: true, copy_chapters: true,
    overwrite_output: false, skip_existing: true, finish_action: 'none', profile: 'main',
  },
}
const workspaceSettings = {
  preset: 1, style: 0, intensity: 1, local_tone: 1, local_struct: 1, skin_struct: 1,
  use_auto_mask: 1, ui_correction: 0, guidance_mode: 0, motion_scale_x: 1, motion_scale_y: 1,
  working_scale: 1, residual_pyramid: true, residual_strength: 1, flow_engine: 'nvof', flow_quality: '1F',
  export_backend: 'gpu_transfer', dlss_backend: 'gpu_flow_d3d12', reconstruction_mode: 'single',
  layer_scales: [1], layer_weights: [1], fusion_mode: 'catmull_baseline', fusion_beta: .1,
  fusion_max_iterations: 128, fusion_relative_tolerance: .001,
  experimental_hdr_scrgb: false, hdr_paper_white_scrgb: 8,
  post_controls: { total_gain: 1, lightness_gain: 1, chroma_gain: 1, low_frequency_gain: 1,
    high_frequency_gain: 1, hue_protection: 0, dark_protection: 0, highlight_protection: 0,
    soft_compression: 0, roi_kind: 0, roi_bounds: [0, 0, 1, 1], roi_feather: .02, debug_mode: 0 },
  temporal_controls: { scene_detection: false, mode: 'off', history_weight: .6 },
}

function runProbe(binary, args) {
  const proc = spawnSync(binary, args, { encoding: 'utf8', windowsHide: true, timeout: 30000 })
  if (proc.error) throw proc.error
  if (proc.status !== 0) throw new Error(`${path.basename(binary)} exited ${proc.status}: ${proc.stderr}`)
  return JSON.parse(proc.stdout)
}

async function pageApi(page, route, method = 'GET', body) {
  return page.evaluate(async ({ route, method, body }) => {
    const tokenValue = window.electronAPI?.localToken || ''
      const response = await fetch(`http://127.0.0.1:18923${route}`, {
        method,
        cache: 'no-store',
      headers: { 'X-DLSS-Token': tokenValue, ...(body ? { 'Content-Type': 'application/json' } : {}) },
      ...(body ? { body: JSON.stringify(body) } : {}),
    })
    const text = await response.text()
    let parsed
    try { parsed = JSON.parse(text) } catch { parsed = text }
    if (!response.ok) throw new Error(`${method} ${route}: ${response.status} ${text}`)
    return parsed
  }, { route, method, body })
}

async function waitForJobStatus(page, id, acceptedStatuses, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs
  const transitions = []
  let lastStatus = null
  let queue = null
  let job = null
  while (Date.now() < deadline) {
    queue = await pageApi(page, '/api/queue')
    job = queue.jobs?.find((row) => row.id === id)
    if (job && job.status !== lastStatus) {
      transitions.push({ at: new Date().toISOString(), status: job.status, progress: job.progress, current_frame: job.current_frame })
      lastStatus = job.status
    }
    if (job && acceptedStatuses.includes(job.status)) return { queue, job, transitions }
    await new Promise((resolve) => setTimeout(resolve, 250))
  }
  throw new Error(`Timed out waiting for job ${id} status ${acceptedStatuses.join('|')}; last=${JSON.stringify({ job, transitions })}`)
}

try {
  app = await electron.launch({
    executablePath: path.join(ROOT_DIR, 'frontend/node_modules/electron/dist/electron.exe'),
    cwd: path.join(ROOT_DIR, 'frontend'),
    args: ['.', `--user-data-dir=${userData}`, '--no-sandbox'],
    env: {
      ...process.env,
      E2E_TEST: '1',
      PYTHON_PATH: pythonExe(),
      DLSS_LOCAL_TOKEN: token,
      E2E_MOCK_FILE_PATH: fixture,
      E2E_MOCK_DIR_PATH: out,
    },
  })
  const proc = app.process()
  proc.stdout?.on('data', (data) => fs.appendFile(path.join(out, 'electron.log'), data))
  proc.stderr?.on('data', (data) => fs.appendFile(path.join(out, 'electron.log'), data))
  const page = await app.firstWindow({ timeout: 90000 })
  page.on('pageerror', (error) => errors.push(String(error)))
  page.on('console', (message) => { if (message.type() === 'error') errors.push(`console: ${message.text()}`) })
  await page.waitForLoadState('domcontentloaded')
  // The UI draft is installed before React mounts on the next document load.
  await page.addInitScript(({ saved, workspace }) => {
    localStorage.setItem('lunarc.transcodeDraft', JSON.stringify(saved))
    localStorage.setItem('lunarc.workspace.v1', JSON.stringify({ version: 1, settings: workspace, recent: [] }))
  }, { saved: draft, workspace: workspaceSettings })
  await page.reload()
  await page.getByRole('button', { name: '导入', exact: true }).waitFor({ state: 'visible', timeout: 45000 })
  result.storageSeed = await page.evaluate(() => ({
    draft: JSON.parse(localStorage.getItem('lunarc.transcodeDraft') || 'null'),
    workspace: JSON.parse(localStorage.getItem('lunarc.workspace.v1') || 'null'),
  }))
  await page.screenshot({ path: path.join(out, 'startup.png'), fullPage: true })

  const system = await page.waitForFunction(async () => {
    try {
      const tokenValue = window.electronAPI?.localToken || ''
      const response = await fetch('http://127.0.0.1:18923/api/system', { headers: { 'X-DLSS-Token': tokenValue } })
      if (!response.ok) return false
      const value = await response.json()
      return value.backend_ready ? value : false
    } catch { return false }
  }, null, { timeout: 45000 })
  result.system = await system.jsonValue()
  backendReady = true
  result.actualBackend = true

  await page.getByRole('button', { name: '导入', exact: true }).click()
  await page.locator('.dlss-material strong').filter({ hasText: 'fixture_gpu_transfer_short.mkv' }).waitFor({ state: 'visible', timeout: 45000 })
  await page.locator('.qi-advanced > summary').click()
  await page.getByRole('tablist', { name: '参数分区' }).getByRole('tab', { name: '基础', exact: true }).click()
  const backendSelect = page.locator('select[aria-label="导出后端"]')
  await backendSelect.waitFor({ state: 'visible', timeout: 10000 })
  await page.waitForFunction(async () => {
    try {
      const tokenValue = window.electronAPI?.localToken || ''
      const response = await fetch('http://127.0.0.1:18923/api/settings', { headers: { 'X-DLSS-Token': tokenValue } })
      const value = await response.json()
      return response.ok && value.export_backend === 'gpu_transfer'
    } catch { return false }
  }, null, { timeout: 15000 })
  await page.waitForFunction(() => document.querySelector('select[aria-label="导出后端"]')?.value === 'gpu_transfer', null, { timeout: 10000 })
  const settingsResponse = await pageApi(page, '/api/settings')
  const liveSettings = settingsResponse.settings || settingsResponse
  result.settingsAfterWorkspaceRestore = { export_backend: liveSettings.export_backend, dlss_backend: liveSettings.dlss_backend, ui_backend_value: await backendSelect.inputValue() }
  if (liveSettings.export_backend !== 'gpu_transfer' || liveSettings.dlss_backend !== 'gpu_flow_d3d12' || result.settingsAfterWorkspaceRestore.ui_backend_value !== 'gpu_transfer') {
    throw new Error(`Restored workspace did not configure valid GPU Transfer: ${JSON.stringify(result.settingsAfterWorkspaceRestore)}`)
  }
  interactions.push('real six-frame fixture imported; isolated GPU Transfer workspace restored, UI selection and backend settings verified')

  await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '导出与队列' }).click()
  const sectionNav = page.getByRole('navigation', { name: '导出设置分类' })
  await page.locator('input[placeholder="输出视频文件名"]').fill(outputName)
  const targetDirLocator = page.locator('label').filter({ hasText: '导出目标文件夹' }).locator('xpath=..').locator('input')
  await targetDirLocator.waitFor({ state: 'visible' })
  await page.getByRole('button', { name: '浏览...' }).click()
  await page.waitForFunction((expected) => {
    const label = [...document.querySelectorAll('label')].find((node) => node.textContent?.includes('导出目标文件夹'))
    return label?.parentElement?.querySelector('input')?.value === expected
  }, out, { timeout: 10000, polling: 100 })
  const outputDir = await targetDirLocator.inputValue()
  if (path.resolve(outputDir) !== out) throw new Error(`Native directory dialog did not return evidence directory: ${outputDir}`)
  const selectNearLabel = (text) => page.locator('label').filter({ hasText: text }).locator('xpath=..').locator('select')
  await selectNearLabel('编码器 (Codec)').selectOption('hevc_nvenc')
  await selectNearLabel('容器 (Container)').selectOption('.mkv')
  await sectionNav.getByRole('button', { name: /色彩/ }).click()
  if (cancelRetry) {
    await selectNearLabel('像素格式').selectOption('auto')
  } else {
    await selectNearLabel('像素格式').selectOption('nv12')
    const nv12Preflight = 'SDR GPU Transfer 的像素格式请选择 Auto 或 YUV420P'
    await page.getByText(nv12Preflight, { exact: false }).waitFor({ state: 'visible', timeout: 10000 })
    const addButton = page.getByRole('button', { name: /添加到渲染队列/ }).last()
    await page.screenshot({ path: path.join(out, 'nv12_preflight.png'), fullPage: true })
    result.nv12PreflightButtonDisabled = await addButton.isDisabled()
    if (!result.nv12PreflightButtonDisabled) throw new Error('Invalid GPU NV12 preflight did not disable the queue button')
    const queueBeforeRefused = await pageApi(page, '/api/queue')
    const queueAfterRefused = await pageApi(page, '/api/queue')
    if (queueAfterRefused.jobs.length !== queueBeforeRefused.jobs.length) throw new Error('Invalid NV12 request was unexpectedly added to the queue')
    interactions.push('invalid GPU NV12 output selection surfaced preflight guidance and did not create a queue job')
    await fs.writeFile(path.join(out, 'nv12_preflight.json'), JSON.stringify({ queueBefore: queueBeforeRefused, queueAfter: queueAfterRefused, guidance: await page.getByText(nv12Preflight, { exact: false }).innerText(), addButtonDisabled: result.nv12PreflightButtonDisabled }, null, 2))
  }
  await selectNearLabel('像素格式').selectOption('auto')
  await sectionNav.getByRole('button', { name: /基础/ }).click()
  await selectNearLabel('控制模式 (Rate Mode)').selectOption('cq')
  const cqSlider = page.locator('label').filter({ hasText: 'CQ 质量数值' }).locator('xpath=../..').locator('input[type="range"]')
  if (await cqSlider.inputValue() !== '24') throw new Error(`CQ slider did not restore the requested value: ${await cqSlider.inputValue()}`)
  await sectionNav.getByRole('button', { name: /高级/ }).click()
  await selectNearLabel('Profile').selectOption('main')
  await sectionNav.getByRole('button', { name: /基础/ }).click()
  const normalizedSettings = await page.evaluate(() => JSON.parse(localStorage.getItem('lunarc.transcodeDraft') || 'null'))
  await fs.writeFile(path.join(out, 'transcode_draft.json'), JSON.stringify(normalizedSettings, null, 2))
  const exportScreenshot = path.join(out, 'deliver_gpu_ready.png')
  await page.screenshot({ path: exportScreenshot, fullPage: true })
  const previewText = await page.locator('body').innerText()
  if (!previewText.includes('GPU Transfer') || previewText.includes('实时命令预览 (FFmpeg)')) {
    throw new Error('GPU Transfer UI preview is not displaying the selected device chain')
  }

  await page.getByRole('button', { name: /添加到渲染队列/ }).last().click()
  await page.waitForFunction(async (name) => {
    const tokenValue = window.electronAPI?.localToken || ''
    const response = await fetch('http://127.0.0.1:18923/api/queue', { headers: { 'X-DLSS-Token': tokenValue } })
    if (!response.ok) return false
    const value = await response.json()
    return value.jobs?.some((job) => job.name?.includes(name)) || false
  }, outputName, { timeout: 15000 })
  const queued = await pageApi(page, '/api/queue')
  const job = queued.jobs.find((item) => item.name?.includes(outputName))
  if (!job) throw new Error('UI add-to-queue did not create the job in backend queue')
  if (job.settings?.export_backend !== 'gpu_transfer' || job.codec !== 'hevc_nvenc' || job.cq !== 24 || job.transcode_settings?.pixel_format !== 'auto') {
    throw new Error(`Queued snapshot differs from valid GPU export request: ${JSON.stringify({ backend: job.settings?.export_backend, codec: job.codec, pixel_format: job.transcode_settings?.pixel_format, cq: job.cq })}`)
  }
  interactions.push('job added through Deliver UI; frozen snapshot verified GPU Transfer + HEVC + legal auto output format + CQ24')
  await page.screenshot({ path: path.join(out, 'queue_pending.png'), fullPage: true })
  await page.getByRole('button', { name: /全部渲染/ }).click()
  let terminal
  if (cancelRetry) {
    const running = await waitForJobStatus(page, job.id, ['running'], 60000)
    result.cancelCycle = { running: running.transitions }
    await fs.writeFile(path.join(out, 'running_queue.json'), JSON.stringify(running.queue, null, 2))
    await page.screenshot({ path: path.join(out, 'queue_running_before_cancel.png'), fullPage: true })
    const cancelButton = page.getByRole('button', { name: '终止当前任务', exact: true })
    await cancelButton.waitFor({ state: 'visible', timeout: 10000 })
    await cancelButton.click()
    const cancelled = await waitForJobStatus(page, job.id, ['cancelled', 'completed', 'failed', 'skipped'], 60000)
    result.cancelCycle.cancelled = cancelled.transitions
    result.cancelCycle.statusAfterCancel = cancelled.job.status
    await fs.writeFile(path.join(out, 'queue_after_cancel.json'), JSON.stringify(cancelled.queue, null, 2))
    await page.screenshot({ path: path.join(out, 'queue_after_cancel.png'), fullPage: true })
    if (cancelled.job.status !== 'cancelled') throw new Error(`UI cancellation did not reach cancelled state: ${cancelled.job.status}`)
    const retryButton = page.getByRole('button', { name: '重试任务 (Retry Job)', exact: true })
    await retryButton.waitFor({ state: 'visible', timeout: 10000 })
    await retryButton.click()
    const pending = await waitForJobStatus(page, job.id, ['pending'], 15000)
    result.cancelCycle.retryToPending = pending.transitions
    await fs.writeFile(path.join(out, 'queue_after_retry.json'), JSON.stringify(pending.queue, null, 2))
    await page.screenshot({ path: path.join(out, 'queue_after_retry.png'), fullPage: true })
    interactions.push('UI cancelled the running job, verified terminal cancelled, retried that row to pending, then restarted it')
    await page.getByRole('button', { name: /全部渲染/ }).click()
    terminal = await waitForJobStatus(page, job.id, ['completed', 'failed', 'cancelled', 'skipped'], 240000)
    result.cancelCycle.completedRun = terminal.transitions
  } else {
    terminal = await waitForJobStatus(page, job.id, ['completed', 'failed', 'cancelled', 'skipped'], 240000)
  }
  const finalQueue = terminal.queue
  const finalJob = terminal.job
  result.job = finalJob
  await fs.writeFile(path.join(out, 'queue.json'), JSON.stringify(finalQueue, null, 2))
  if (finalJob.status !== 'completed') throw new Error(`Actual GPU export did not complete: ${finalJob.status} ${finalJob.error || ''}`)
  result.actualGpuExport = true
  if (!(await fs.stat(outputFile)).isFile()) throw new Error('Completed job did not produce its output file')
  const probe = runProbe(ffprobe, ['-v', 'error', '-count_frames', '-show_streams', '-show_format', '-of', 'json', outputFile])
  const videoStream = probe.streams.find((stream) => stream.codec_type === 'video')
  if (!videoStream || videoStream.codec_name !== 'hevc' || Number(videoStream.nb_read_frames) !== 6) {
    throw new Error(`Output stream failed HEVC/6-frame verification: ${JSON.stringify(videoStream)}`)
  }
  result.output = { bytes: (await fs.stat(outputFile)).size, sha256: crypto.createHash('sha256').update(await fs.readFile(outputFile)).digest('hex'), probe }
  await fs.writeFile(path.join(out, 'output_probe.json'), JSON.stringify(probe, null, 2))

  const diagnostics = await pageApi(page, `/api/diagnostics/job/${encodeURIComponent(job.id)}`)
  result.diagnostics = diagnostics
  await fs.writeFile(path.join(out, 'job_diagnostics.json'), JSON.stringify(diagnostics, null, 2))
  const diagnosticJob = diagnostics.job || {}
  const report = diagnosticJob.residual_backend_diagnostics || diagnosticJob.diagnostics || {}
  if (report.export_backend_actual !== 'gpu_transfer') {
    throw new Error(`Actual GPU Transfer diagnostic missing or incorrect: ${JSON.stringify({ export_backend_actual: report.export_backend_actual, reportKeys: Object.keys(report) })}`)
  }
  interactions.push('queue reached actual completed state; output verified by ffprobe; backend job diagnostics captured')
  await page.screenshot({ path: path.join(out, 'queue_completed.png'), fullPage: true })
  await page.getByRole('button', { name: new RegExp(`查看 ${outputName} 的实际运行诊断`) }).click()
  await page.screenshot({ path: path.join(out, 'job_diagnostics.png'), fullPage: true })

  result.errors = errors
  if (errors.length) throw new Error(`Renderer page errors: ${errors.join('\n')}`)
  result.status = 'PASS'
} catch (error) {
  result.status = 'FAIL'
  result.error = String(error?.stack || error)
  if (app) {
    try { await app.firstWindow().then((page) => page.screenshot({ path: path.join(out, 'failure.png'), fullPage: true })) } catch {}
  }
  throw error
} finally {
  result.actualBackend = backendReady
  result.errors = errors
  await fs.writeFile(path.join(out, 'result.json'), JSON.stringify(result, null, 2))
  if (app) await app.close().catch(() => {})
}

console.log(`Real Electron GPU queue acceptance: ${result.status}`)
console.log(`Evidence: ${out}`)
