// Packaged Electron SDR10 UI acceptance. Adds a real pending job but never starts
// the queue; the hardware service export is covered by sdr10_service_gate.py.
import { _electron as electron } from '@playwright/test'
import path from 'node:path'
import crypto from 'node:crypto'
import fs from 'node:fs/promises'
import { execFileSync } from 'node:child_process'
import { ROOT_DIR, pythonExe } from './paths.js'

const out = path.resolve(process.argv[2] || path.join(ROOT_DIR, 'bench_out/sdr10_20261010/ui'))
const exeOverride = process.argv.find(arg => arg.startsWith('--exe='))?.slice('--exe='.length)
const executable = path.resolve(exeOverride ||
  path.join(ROOT_DIR, 'bench_out/sdr10_20261010/package/win-unpacked/Lunarc.exe'))
const fixture = path.join(ROOT_DIR, 'bench_out/sdr10_20261010/source6.mkv')
const userData = path.join(out, 'user_data')
const token = `sdr10-ui-${crypto.randomUUID()}`
const errors = []
let app = null
let backendPid = null
let initialSettings = null
let result = { status: 'RUNNING', executable, fixture, userData, screenshots: [], errors }

try {
  const existing = await fs.readdir(out)
  if (existing.length) throw new Error(`Evidence directory is not fresh: ${out}`)
} catch (error) { if (error.code !== 'ENOENT') throw error }
await fs.mkdir(userData, { recursive: true })
await fs.mkdir(out, { recursive: true })

async function portIsBusy() {
  return fetch('http://127.0.0.1:18923/api/system', { signal: AbortSignal.timeout(1000) })
    .then(() => true).catch(() => false)
}
function listenerPid() {
  if (process.platform !== 'win32') return null
  try {
    const row = execFileSync('netstat.exe', ['-ano', '-p', 'tcp'],
      { encoding: 'utf8', windowsHide: true }).split(/\r?\n/)
      .find(line => /127\.0\.0\.1:18923\s+.*LISTENING/.test(line))
    const pid = Number(row?.trim().split(/\s+/).at(-1))
    return Number.isInteger(pid) && pid > 0 ? pid : null
  } catch { return null }
}
function commandLine(pid) {
  if (!pid || process.platform !== 'win32') return ''
  try {
    return execFileSync('powershell.exe', ['-NoProfile', '-Command',
      `(Get-CimInstance Win32_Process -Filter 'ProcessId=${pid}').CommandLine`],
    { encoding: 'utf8', windowsHide: true }).trim()
  } catch { return '' }
}
async function api(page, route, method = 'GET', body) {
  return page.evaluate(async ({ route, method, body }) => {
    const response = await fetch(`http://127.0.0.1:18923${route}`, {
      method, cache: 'no-store',
      headers: { 'X-DLSS-Token': window.electronAPI?.localToken || '',
        ...(body ? { 'Content-Type': 'application/json' } : {}) },
      ...(body ? { body: JSON.stringify(body) } : {}),
    })
    const text = await response.text()
    let value
    try { value = JSON.parse(text) } catch { value = text }
    if (!response.ok) throw new Error(`${method} ${route}: ${response.status} ${text}`)
    return value
  }, { route, method, body })
}
async function settings(page) {
  const value = await api(page, '/api/settings')
  return value.settings || value
}
async function waitForSettledTheme(page) {
  await page.waitForFunction(() => {
    const root = getComputedStyle(document.documentElement)
    const queue = document.querySelector('.deliver-queue')
    const header = queue?.querySelector('.deliver-queue-header')
    const title = header?.querySelector('.text-lunarc-text')
    if (!queue || !header || !title) return false
    const resolver = document.createElement('span')
    resolver.style.backgroundColor = 'var(--color-lunarc-dark)'
    resolver.style.color = 'var(--color-lunarc-text)'
    document.body.appendChild(resolver)
    const dark = getComputedStyle(resolver).backgroundColor
    const text = getComputedStyle(resolver).color
    resolver.remove()
    const animations = [...document.querySelectorAll('.deliver-layout, .deliver-layout *')]
      .flatMap(element => element.getAnimations())
    const finiteTransitionRunning = animations.some(animation => {
      const iterations = animation.effect?.getComputedTiming()?.iterations
      return animation.playState === 'running' && iterations !== Infinity
    })
    return getComputedStyle(queue).backgroundColor === dark &&
      getComputedStyle(title).color === text && !finiteTransitionRunning
  }, null, { timeout: 10000, polling: 50 })
  return page.evaluate(() => {
    const queue = document.querySelector('.deliver-queue')
    const header = queue?.querySelector('.deliver-queue-header')
    const title = header?.querySelector('.text-lunarc-text')
    const resolver = document.createElement('span')
    resolver.style.backgroundColor = 'var(--color-lunarc-dark)'
    resolver.style.color = 'var(--color-lunarc-text)'
    resolver.style.borderColor = 'var(--color-lunarc-surface)'
    document.body.appendChild(resolver)
    const facts = { theme: document.documentElement.dataset.theme,
      expectedBackground: getComputedStyle(resolver).backgroundColor,
      actualBackground: getComputedStyle(queue).backgroundColor,
      expectedText: getComputedStyle(resolver).color,
      actualText: getComputedStyle(title).color }
    resolver.remove()
    return facts
  })
}

try {
  if (!(await fs.stat(executable)).isFile()) throw new Error(`Packaged executable not found: ${executable}`)
  if (!(await fs.stat(fixture)).isFile()) throw new Error(`SDR10 source not found: ${fixture}`)
  if (await portIsBusy()) throw new Error('Port 18923 is already occupied; refusing to attach to or stop another backend')
  const beforePid = listenerPid()
  if (beforePid) throw new Error(`Port 18923 already has listener PID ${beforePid}; refusing to launch`)

  app = await electron.launch({
    executablePath: executable,
    cwd: path.dirname(executable),
    args: [`--user-data-dir=${userData}`, '--no-sandbox'],
    env: { ...process.env, E2E_TEST: '1', PYTHON_PATH: pythonExe(),
      DLSS_LOCAL_TOKEN: token, E2E_MOCK_FILE_PATH: fixture },
  })
  app.process().stdout?.on('data', chunk => {
    const match = String(chunk).match(/Python backend spawned with PID:\s*(\d+)/)
    if (match) backendPid = Number(match[1])
  })
  app.process().stderr?.on('data', chunk => {
    const match = String(chunk).match(/Started server process\s*\[(\d+)\]/)
    if (match) backendPid = Number(match[1])
  })

  const page = await app.firstWindow({ timeout: 90000 })
  page.on('pageerror', error => errors.push(String(error)))
  page.on('console', message => { if (message.type() === 'error') errors.push(`console: ${message.text()}`) })
  await page.waitForLoadState('domcontentloaded')
  await page.waitForFunction(async () => {
    try {
      const r = await fetch('http://127.0.0.1:18923/api/system', {
        headers: { 'X-DLSS-Token': window.electronAPI?.localToken || '' }, cache: 'no-store' })
      const s = await r.json()
      return r.ok && s.backend_ready === true && Boolean(s.gpu)
    } catch { return false }
  }, null, { timeout: 60000 })
  const ownPid = listenerPid()
  backendPid = ownPid || backendPid
  const expectedServer = path.join(path.dirname(executable), 'resources/server.py').replaceAll('\\', '/')
  const ownerCommand = commandLine(backendPid)
  if (process.platform === 'win32' && (!backendPid || backendPid === beforePid ||
      !ownerCommand.replaceAll('\\', '/').toLowerCase().includes(expectedServer.toLowerCase()))) {
    throw new Error(`Could not prove backend belongs to this packaged app: pid=${backendPid}, command=${ownerCommand}`)
  }
  result.backendOwnership = { pid: backendPid, command: ownerCommand, expectedServer }

  initialSettings = await settings(page)
  result.initialSettings = initialSettings
  const nav = page.getByRole('navigation', { name: '主导航' })
  await nav.getByRole('button', { name: /重建工作台/ }).click()
  await page.getByRole('button', { name: '导入', exact: true }).click()
  await page.waitForFunction(name =>
    document.querySelector('.dlss-material strong')?.textContent?.includes(name),
  'source6.mkv', { timeout: 45000 })
  await nav.getByRole('button', { name: /导出与队列/ }).click()
  const prepare = page.getByRole('button', { name: '准备 SDR 10-bit 导出', exact: true })
  await prepare.waitFor({ state: 'visible', timeout: 20000 })
  await prepare.click()
  await page.getByText('已准备 SDR 10-bit 导出 · HEVC Main10 · BT.709', { exact: true })
    .waitFor({ state: 'visible', timeout: 15000 })

  const requested = await settings(page)
  if (requested.export_backend !== 'gpu_transfer' || requested.experimental_hdr_scrgb !== false) {
    throw new Error(`Prepared backend settings mismatch: ${JSON.stringify({
      export_backend: requested.export_backend,
      experimental_hdr_scrgb: requested.experimental_hdr_scrgb })}`)
  }
  const categories = page.getByRole('navigation', { name: '导出设置分类' })
  await categories.getByRole('button', { name: /视频/ }).click()
  await categories.getByRole('button', { name: /色彩/ }).click()
  await categories.getByRole('button', { name: /高级/ }).click()
  const selectFacts = await page.evaluate(() => {
    const facts = {}
    for (const label of document.querySelectorAll('.deliver-layout label')) {
      const select = label.parentElement?.querySelector('select')
      if (select) facts[label.textContent.trim()] = select.value
    }
    return facts
  })
  const mustMatch = [
    ['编码器 (Codec)', 'hevc_nvenc'], ['容器 (Container)', '.mkv'],
    ['Profile', 'main10'], ['像素格式 (Pixel Format)', 'p010le'],
    ['色彩规范 (Color Preset)', 'preserve'],
  ]
  for (const [label, expected] of mustMatch) {
    if (selectFacts[label] !== expected) throw new Error(`${label}=${selectFacts[label]} (wanted ${expected})`)
  }
  const presetLabel = await page.evaluate(() => {
    const label = [...document.querySelectorAll('span')]
      .find(element => element.textContent.trim() === '预设方案:')
    return label?.parentElement?.querySelector('select')?.selectedOptions[0]?.textContent?.trim() || null
  })
  if (presetLabel !== '自定义' || /AV1/i.test(presetLabel || '')) {
    throw new Error(`SDR10 parameters must be labeled 自定义, got ${presetLabel}`)
  }
  result.preparedSettings = requested
  result.transcodeControls = selectFacts
  result.presetLabelAfterPrepare = presetLabel?.trim()

  const previewHeading = await page.locator('.deliver-editor').innerText()
  if (/设备链路预览\s*\(实验性 HDR10/i.test(previewHeading) ||
      /降级|回退到\s*8-bit|已降为\s*8-bit/i.test(previewHeading)) {
    throw new Error('SDR10 UI shows an HDR preview badge or SDR downgrade message')
  }
  result.previewHonest = true

  const darkToggle = page.getByRole('button', { name: '切换浅色模式', exact: true })
  await darkToggle.waitFor({ state: 'visible' })
  result.themeSnapshots = { dark: await waitForSettledTheme(page) }
  await page.screenshot({ path: path.join(out, 'deliver_dark.png'), fullPage: true })
  result.screenshots.push('deliver_dark.png')
  await darkToggle.click()
  await page.getByRole('button', { name: '切换深色模式', exact: true }).waitFor({ state: 'visible' })
  result.themeSnapshots.light = await waitForSettledTheme(page)
  await page.screenshot({ path: path.join(out, 'deliver_light.png'), fullPage: true })
  result.screenshots.push('deliver_light.png')

  const fileName = page.locator('input[placeholder="输出视频文件名"]')
  await fileName.fill('sdr10_ui_pending_acceptance')
  const beforeQueue = await api(page, '/api/queue')
  const beforeIds = new Set((beforeQueue.jobs || []).map(job => job.id))
  const addButton = page.getByRole('button', { name: /添加到渲染队列/ }).last()
  await addButton.waitFor({ state: 'visible' })
  await addButton.click()
  await page.waitForFunction(async before => {
    try {
      const token = window.electronAPI?.localToken || ''
      const response = await fetch('http://127.0.0.1:18923/api/queue', {
        headers: { 'X-DLSS-Token': token }, cache: 'no-store' })
      if (!response.ok) return false
      const queue = await response.json()
      return queue.jobs?.some(job => !before.includes(job.id) && job.name === 'sdr10_ui_pending_acceptance')
    } catch { return false }
  }, [...beforeIds], { timeout: 15000 })
  const queue = await api(page, '/api/queue')
  const job = queue.jobs.find(item => !beforeIds.has(item.id) && item.name === 'sdr10_ui_pending_acceptance')
  if (!job || job.status !== 'pending' || queue.queue_running) {
    throw new Error(`New job was not pending in an idle queue: ${JSON.stringify({ job, queue_running: queue.queue_running })}`)
  }
  const ts = job.transcode_settings || {}
  if (job.settings?.export_backend !== 'gpu_transfer' ||
      job.settings?.experimental_hdr_scrgb !== false ||
      ts.codec !== 'hevc_nvenc' || ts.container !== '.mkv' ||
      ts.pixel_format !== 'p010le' || ts.profile !== 'main10' ||
      ts.color_preset !== 'preserve' || job.input_path !== fixture) {
    throw new Error(`Queued job snapshot is not the prepared SDR10 request: ${JSON.stringify(job)}`)
  }
  result.queuedJob = job
  result.queueRunning = queue.queue_running
  await page.screenshot({ path: path.join(out, 'queued_pending.png'), fullPage: true })
  result.screenshots.push('queued_pending.png')
  if (errors.length) throw new Error(`Renderer errors: ${errors.join('\n')}`)
  result.status = 'PASS'
} catch (error) {
  result.status = 'FAIL'
  result.error = String(error?.stack || error)
  if (app) {
    try { await app.firstWindow().then(page => page.screenshot({ path: path.join(out, 'failure.png'), fullPage: true })) } catch {}
  }
  throw error
} finally {
  if (app && initialSettings) {
    try {
      const page = await app.firstWindow()
      await api(page, '/api/settings', 'POST', initialSettings)
      result.settingsRestored = true
    } catch (error) { result.settingsRestoreError = String(error) }
  }
  if (app) await app.close().catch(() => {})
  if (backendPid && process.platform === 'win32') {
    const expectedServer = path.join(path.dirname(executable), 'resources/server.py').replaceAll('\\', '/')
    const pid = listenerPid()
    if (pid === backendPid) {
      const command = commandLine(pid).replaceAll('\\', '/').toLowerCase()
      if (command.includes(expectedServer.toLowerCase())) {
        try { execFileSync('taskkill.exe', ['/F', '/T', '/PID', String(pid)],
          { encoding: 'utf8', windowsHide: true }) } catch {}
      } else result.cleanupError = `Refusing to terminate unverified listener PID ${pid}: ${command}`
    }
    await new Promise(resolve => setTimeout(resolve, 500))
    result.backendCleanup = { pid: backendPid, listenerPidAfterClose: listenerPid() }
    if (result.backendCleanup.listenerPidAfterClose === backendPid) {
      result.cleanupError ||= `Owned backend PID ${backendPid} survived cleanup`
    }
  } else result.backendCleanup = { pid: backendPid, listenerPidAfterClose: listenerPid() }
  result.errors = errors
  await fs.writeFile(path.join(out, 'result.json'), JSON.stringify(result, null, 2))
}

if (result.status !== 'PASS' || result.cleanupError || result.settingsRestoreError) {
  throw new Error(result.cleanupError || result.settingsRestoreError || `SDR10 UI probe failed: ${result.error || result.status}`)
}
console.log(`Packaged SDR10 UI pending-job acceptance: ${result.status}`)
console.log(`Evidence: ${out}`)
