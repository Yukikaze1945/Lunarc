// Real Electron visual/interaction contract for the compact Swiss-style workbench.
// Uses an isolated Chromium profile and the repository's own short SDR fixture.
// Deliberately not run during implementation: the root agent owns the UI-ready gate.
import { _electron as electron } from '@playwright/test'
import path from 'node:path'
import crypto from 'node:crypto'
import fs from 'node:fs/promises'
import { execFileSync } from 'node:child_process'
import { ROOT_DIR, pythonExe } from './paths.js'

const outputArg = process.argv.slice(2).find((arg) => !arg.startsWith('--'))
const out = path.resolve(outputArg || path.join(ROOT_DIR, 'bench_out/optimization_20261009/swiss_ui'))
const fixture = path.join(ROOT_DIR, 'bench_out/optimization_20261009/sdr_source.mkv')
const packaged = process.argv.includes('--packaged')
const screenshotsOnly = process.argv.includes('--screenshots-only')
const packagedIndex = process.argv.indexOf('--packaged')
const packagedExecutable = packagedIndex >= 0 ? process.argv[packagedIndex + 1] : null
if (packaged && (!packagedExecutable || packagedExecutable.startsWith('--'))) {
  throw new Error('Usage: swiss_ui_probe.mjs <evidence-dir> --packaged <Lunarc.exe> [--screenshots-only]')
}
const executable = packaged ? path.resolve(packagedExecutable)
  : path.join(ROOT_DIR, 'frontend/node_modules/electron/dist/electron.exe')
const userData = path.join(out, 'user_data')
const token = `swiss-ui-${crypto.randomUUID()}`
const errors = []
const interactions = []
let app = null
let initialSettings = null
let backendPid = null
let result = { actualElectron: true, fixture, executable, userData, interactions, errors, screenshots: [] }

try {
  const prior = await fs.readdir(out)
  if (prior.length) throw new Error(`Evidence directory is not fresh: ${out}`)
} catch (error) { if (error.code !== 'ENOENT') throw error }
await fs.mkdir(userData, { recursive: true })
if (await fetch('http://127.0.0.1:18923/api/system').then(() => true).catch(() => false)) {
  throw new Error('Port 18923 already has a backend; refusing to attach to or stop another session')
}
if (!(await fs.stat(fixture)).isFile()) throw new Error(`SDR source fixture is missing: ${fixture}`)

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

async function waitForSettings(page, expected, description) {
  await page.waitForFunction(async ({ expected }) => {
    try {
      const response = await fetch('http://127.0.0.1:18923/api/settings', {
        headers: { 'X-DLSS-Token': window.electronAPI?.localToken || '' }, cache: 'no-store',
      })
      if (!response.ok) return false
      const data = await response.json()
      const settings = data.settings || data
      return Object.entries(expected).every(([key, value]) => settings[key] === value)
    } catch { return false }
  }, { expected }, { timeout: 15000, polling: 100 })
    .catch((error) => { throw new Error(`Timed out waiting for backend settings (${description}): ${error.message}`) })
}

async function setSlider(page, label, value, settingKey) {
  const slider = page.getByRole('slider', { name: label })
  await slider.waitFor({ state: 'visible' })
  const limits = await slider.evaluate((el) => ({ min: Number(el.min), max: Number(el.max), step: Number(el.step || 1) }))
  if (value < limits.min || value > limits.max) throw new Error(`${label} test value ${value} is outside ${JSON.stringify(limits)}`)
  await slider.evaluate((el, next) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(next))
    el.dispatchEvent(new Event('input', { bubbles: true }))
    el.dispatchEvent(new Event('change', { bubbles: true }))
  }, value)
  await waitForSettings(page, { [settingKey]: value }, `${settingKey}=${value}`)
}

async function waitForWorkbenchReload(page, expected, requireSource = true) {
  await page.waitForFunction(async ({ expected, requireSource }) => {
    try {
      const headers = { 'X-DLSS-Token': window.electronAPI?.localToken || '' }
      const [systemResponse, settingsResponse] = await Promise.all([
        fetch('http://127.0.0.1:18923/api/system', { headers, cache: 'no-store' }),
        fetch('http://127.0.0.1:18923/api/settings', { headers, cache: 'no-store' }),
      ])
      if (!systemResponse.ok || !settingsResponse.ok) return false
      const system = await systemResponse.json()
      const data = await settingsResponse.json()
      const settings = data.settings || data
      const status = document.querySelector('.status-bar [role="status"]')?.textContent || ''
      const filename = document.querySelector('.dlss-material strong')?.textContent || ''
      return system.backend_ready === true && Boolean(system.gpu) &&
        Object.entries(expected).every(([key, value]) => settings[key] === value) &&
        !/连接本地工作台|正在重试|连接中|等待连接|正在读取素材/.test(status) &&
        (!requireSource || filename.includes('sdr_source.mkv'))
    } catch { return false }
  }, { expected, requireSource }, { timeout: 60000, polling: 150 })
}

async function setWindowSize(width, height) {
  await app.evaluate(({ BrowserWindow }, size) => BrowserWindow.getAllWindows()[0].setContentSize(size[0], size[1]), [width, height])
  const page = await app.firstWindow()
  await page.waitForFunction(({ width, height }) =>
    innerWidth === width && innerHeight === height, { width, height }, { timeout: 10000 })
}

async function verifyNoOverflow(page, width, height) {
  const dimensions = await page.evaluate(() => ({
    width: innerWidth, height: innerHeight,
    documentWidth: document.documentElement.scrollWidth,
    documentHeight: document.documentElement.scrollHeight,
    bodyWidth: document.body.scrollWidth,
    bodyHeight: document.body.scrollHeight,
  }))
  if (dimensions.width !== width || dimensions.height !== height ||
      dimensions.documentWidth > width + 1 || dimensions.bodyWidth > width + 1 ||
      dimensions.documentHeight > height + 1 || dimensions.bodyHeight > height + 1) {
    throw new Error(`Workbench overflows ${width}x${height}: ${JSON.stringify(dimensions)}`)
  }
  return dimensions
}

async function saveScreenshot(page, name) {
  const deadline = Date.now() + 60000
  let visualState = null
  let attempts = 0
  while (Date.now() < deadline) {
    attempts += 1
    await page.waitForFunction(() => {
      const status = document.querySelector('.status-bar [role="status"]')?.textContent?.trim() || ''
      const image = document.querySelector('.dlss-preview-image')
      const rect = image?.getBoundingClientRect()
      const ready = Boolean(status) && !/连接本地工作台|正在重试|连接中|等待连接|正在读取素材/.test(status) &&
        !document.querySelector('.dlss-empty-preview') && image instanceof HTMLImageElement &&
        image.complete && image.naturalWidth > 0 && image.naturalHeight > 0 && rect.width > 100 && rect.height > 100 &&
        Boolean(document.querySelector('.dlss-material strong')?.textContent?.includes('sdr_source.mkv'))
      const stableKey = ready ? `${status}|${image.currentSrc || image.src}|${rect.width}x${rect.height}` : ''
      if (!stableKey) {
        delete document.documentElement.dataset.captureReadyKey
        delete document.documentElement.dataset.captureReadySince
        return false
      }
      if (document.documentElement.dataset.captureReadyKey !== stableKey) {
        document.documentElement.dataset.captureReadyKey = stableKey
        document.documentElement.dataset.captureReadySince = String(Date.now())
        return false
      }
      return Date.now() - Number(document.documentElement.dataset.captureReadySince) >= 1000
    }, null, { timeout: Math.max(1000, deadline - Date.now()), polling: 150 })
    await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))))
    await page.waitForTimeout(350)
    visualState = await readVisualState(page)
    if (visualState.ready) break
  }
  if (!visualState?.ready) throw new Error(`Preview did not remain ready before screenshot ${name} after ${attempts} attempts: ${JSON.stringify(visualState)}`)
  visualState.attempts = attempts
  result.screenshotStates ||= {}
  result.screenshotStates[name] = visualState
  const target = path.join(out, name)
  const chrome = await page.evaluate(() => {
    const header = document.querySelector('.project-header')
    const actions = document.querySelector('.header-actions')
    const overlay = navigator.windowControlsOverlay
    const safe = overlay?.getTitlebarAreaRect()
    return {
      wordmark: document.querySelector('.wordmark')?.textContent,
      markerChildren: document.querySelector('.split-marker')?.childElementCount,
      headerDrag: getComputedStyle(header).webkitAppRegion,
      buttonDrag: getComputedStyle(header.querySelector('button')).webkitAppRegion,
      overlayVisible: overlay?.visible,
      safeRight: safe ? safe.x + safe.width : null,
      actionsRight: actions.getBoundingClientRect().right,
    }
  })
  if (chrome.wordmark !== 'Lunarc' || chrome.markerChildren !== 0 ||
      chrome.headerDrag !== 'drag' || chrome.buttonDrag !== 'no-drag' ||
      !chrome.overlayVisible || chrome.actionsRight > chrome.safeRight) {
    throw new Error(`Integrated window chrome contract failed: ${JSON.stringify(chrome)}`)
  }
  result.windowChrome ||= {}
  result.windowChrome[name] = chrome
  await page.screenshot({ path: target })
  result.screenshots.push(name)
  if (name.includes('1440x900') && process.platform === 'win32') {
    // Renderer screenshots omit native caption buttons; capture only this HWND.
    const nativeImage = await app.evaluate(async ({ BrowserWindow, desktopCapturer }) => {
      const window = BrowserWindow.getAllWindows()[0]
      const handle = window.getNativeWindowHandle().readBigUInt64LE().toString()
      const sources = await desktopCapturer.getSources({ types: ['window'], thumbnailSize: { width: 1600, height: 1100 }, fetchWindowIcons: false })
      const source = sources.find((item) => item.id.split(':')[1] === handle)
      if (!source || source.thumbnail.isEmpty()) throw new Error('Native application window capture unavailable')
      return source.thumbnail.toPNG().toString('base64')
    })
    const nativeName = name.replace('.png', '_window.png')
    await fs.writeFile(path.join(out, nativeName), Buffer.from(nativeImage, 'base64'))
    result.screenshots.push(nativeName)
  }
}

async function readVisualState(page) {
  return page.evaluate(async () => {
    const status = document.querySelector('.status-bar [role="status"]')?.textContent?.trim() || ''
    const image = document.querySelector('.dlss-preview-image')
    const rect = image?.getBoundingClientRect()
    const emptyPreviewCount = document.querySelectorAll('.dlss-empty-preview').length
    let decoded = false
    if (image instanceof HTMLImageElement && image.complete && image.naturalWidth > 0) {
      try { await image.decode(); decoded = true } catch {}
    }
    return {
      ready: Boolean(status) && !/连接本地工作台|正在重试|连接中|等待连接|正在读取素材/.test(status) &&
        emptyPreviewCount === 0 && image instanceof HTMLImageElement && decoded &&
        rect.width > 100 && rect.height > 100 &&
        Boolean(document.querySelector('.dlss-material strong')?.textContent?.includes('sdr_source.mkv')),
      status, emptyPreviewCount,
      image: image instanceof HTMLImageElement ? {
        src: (image.currentSrc || image.src).startsWith('data:')
          ? `${(image.currentSrc || image.src).split(',')[0]},<embedded image>`
          : (image.currentSrc || image.src), complete: image.complete, decoded,
        naturalWidth: image.naturalWidth, naturalHeight: image.naturalHeight,
        rect: rect ? { x: rect.x, y: rect.y, width: rect.width, height: rect.height } : null,
      } : null,
    }
  })
}

async function assertReadyPreview(page) {
  await page.waitForFunction(async () => {
    const status = document.querySelector('.status-bar [role="status"]')?.textContent?.trim() || ''
    const image = document.querySelector('.dlss-preview-image')
    if (!status || /连接本地工作台|正在重试|连接中|等待连接|正在读取素材/.test(status) ||
        document.querySelector('.dlss-empty-preview') ||
        !(image instanceof HTMLImageElement) || !image.complete || !image.naturalWidth || !image.naturalHeight) return false
    const rect = image.getBoundingClientRect()
    if (rect.width <= 100 || rect.height <= 100 ||
        !document.querySelector('.dlss-material strong')?.textContent?.includes('sdr_source.mkv')) return false
    try { await image.decode(); return true } catch { return false }
  }, null, { timeout: 60000, polling: 150 })
}

async function ensureImportedPreview(page) {
  const hasSource = await page.locator('.dlss-material strong').textContent().catch(() => '')
  if (!hasSource?.includes('sdr_source.mkv') || await page.locator('.dlss-empty-preview').count()) {
    await page.getByRole('button', { name: '导入', exact: true }).click()
    await page.waitForFunction((name) => document.querySelector('.dlss-material strong')?.textContent?.includes(name),
      'sdr_source.mkv', { timeout: 45000 })
  }
  await assertReadyPreview(page)
}

function ownedBackendPid() {
  if (process.platform !== 'win32') return backendPid
  const line = execFileSync('netstat.exe', ['-ano', '-p', 'tcp'], { encoding: 'utf8', windowsHide: true })
    .split(/\r?\n/).find((row) => /(?:127\.0\.0\.1|0\.0\.0\.0|\[::\]|::):18923\s+.*LISTENING/.test(row))
  const pid = Number(line?.trim().split(/\s+/).at(-1))
  return Number.isInteger(pid) && pid > 0 ? pid : backendPid
}

function listeningBackendPid() {
  if (process.platform !== 'win32') return backendPid
  const line = execFileSync('netstat.exe', ['-ano', '-p', 'tcp'], { encoding: 'utf8', windowsHide: true })
    .split(/\r?\n/).find((row) => /127\.0\.0\.1:18923\s+.*LISTENING/.test(row))
  const pid = Number(line?.trim().split(/\s+/).at(-1))
  return Number.isInteger(pid) && pid > 0 ? pid : null
}

function processCommandLine(pid) {
  try {
    return execFileSync('powershell.exe', ['-NoProfile', '-Command',
      `(Get-CimInstance Win32_Process -Filter 'ProcessId=${pid}').CommandLine`],
    { encoding: 'utf8', windowsHide: true }).trim()
  } catch { return '' }
}

try {
  app = await electron.launch({
    executablePath: executable,
    cwd: packaged ? path.dirname(executable) : path.join(ROOT_DIR, 'frontend'),
    args: [...(packaged ? [] : ['.']), `--user-data-dir=${userData}`, '--no-sandbox'],
    env: { ...process.env, E2E_TEST: '1', PYTHON_PATH: pythonExe(), DLSS_LOCAL_TOKEN: token,
      E2E_MOCK_FILE_PATH: fixture },
  })
  app.process().stdout?.on('data', (chunk) => {
    const match = String(chunk).match(/Python backend spawned with PID:\s*(\d+)/)
    if (match) backendPid = Number(match[1])
  })
  app.process().stderr?.on('data', (chunk) => {
    const text = String(chunk)
    result.electronStderr = `${result.electronStderr || ''}${text}`
    const uvicorn = text.match(/Started server process\s*\[(\d+)\]/)
    if (uvicorn) backendPid = Number(uvicorn[1])
  })

  const page = await app.firstWindow({ timeout: 90000 })
  page.on('pageerror', (error) => errors.push(String(error)))
  page.on('console', (message) => { if (message.type() === 'error') errors.push(`console: ${message.text()}`) })
  await page.waitForLoadState('domcontentloaded')
  await page.waitForFunction(async () => {
    try {
      const response = await fetch('http://127.0.0.1:18923/api/system', {
        headers: { 'X-DLSS-Token': window.electronAPI?.localToken || '' }, cache: 'no-store',
      })
      const system = await response.json()
      return response.ok && system.backend_ready === true && Boolean(system.gpu)
    } catch { return false }
  }, null, { timeout: 60000 })
  result.actualBackend = true
  if (process.platform === 'win32') {
    const deadline = Date.now() + 10000
    let listenerPid = null
    while (Date.now() < deadline) {
      listenerPid = listeningBackendPid()
      if (listenerPid || backendPid) break
      await new Promise((resolve) => setTimeout(resolve, 100))
    }
    // The bundled/dev server logs its Uvicorn child PID on stderr. netstat is
    // preferred; retain that PID as a fallback for Windows builds where the
    // local listener row is omitted from the active TCP table.
    backendPid = listenerPid || backendPid
    const commandLine = backendPid ? processCommandLine(backendPid) : ''
    const expected = (packaged ? path.join(path.dirname(executable), 'resources/server.py')
      : path.join(ROOT_DIR, 'server.py')).replaceAll('\\', '/')
    if (!backendPid ||
        !commandLine.replaceAll('\\', '/').toLowerCase().includes(expected.toLowerCase())) {
      throw new Error(`Backend listener ownership mismatch: listener=${listenerPid}, command=${commandLine}`)
    }
    result.backendOwnership = { pid: backendPid, listenerPid, verifiedServerPath: expected }
  }

  const nav = page.getByRole('navigation', { name: '主导航' })
  const workbenchNav = nav.getByRole('button', { name: /重建工作台/ })
  const main = page.locator('.page-host')
  await main.waitFor({ state: 'visible' })
  if (!(await workbenchNav.getAttribute('aria-current') === 'page' ||
        await workbenchNav.evaluate((el) => el.classList.contains('active')))) {
    throw new Error('The app did not open on the reconstruction workbench')
  }
  const basicGroup = page.getByRole('group', { name: '基础控制', exact: true })
  await basicGroup.waitFor({ state: 'visible', timeout: 15000 })
  const railWidth = await nav.evaluate((el) => el.getBoundingClientRect().width)
  if (railWidth > 110) throw new Error(`Navigation rail is not compact: ${railWidth}px`)
  interactions.push('initial reconstruction workbench and compact navigation rail')

  initialSettings = await api(page, '/api/settings')
  initialSettings = initialSettings.settings || initialSettings
  result.initialSettings = initialSettings

  // Import the real, short SDR source through Electron's native-dialog test seam.
  await page.getByRole('button', { name: '导入', exact: true }).click()
  await page.waitForFunction((name) => document.querySelector('.dlss-material strong')?.textContent?.includes(name),
    'sdr_source.mkv', { timeout: 45000 })
  await basicGroup.waitFor({ state: 'visible' })
  await page.waitForFunction(() => {
    const image = document.querySelector('.dlss-preview-image')
    return image instanceof HTMLImageElement && image.complete && image.naturalWidth > 0 && image.naturalHeight > 0
  }, null, { timeout: 45000 })
  result.previewDecoded = true
  await assertReadyPreview(page)
  interactions.push('short SDR fixture imported')

  if (screenshotsOnly) {
    const lightModeButton = page.getByRole('button', { name: '切换浅色模式', exact: true })
    await lightModeButton.waitFor({ state: 'visible' })
    await setWindowSize(1440, 900)
    result.layout1440 = await verifyNoOverflow(page, 1440, 900)
    await saveScreenshot(page, 'workbench_dark_1440x900.png')
    await lightModeButton.click()
    await page.getByRole('button', { name: '切换深色模式', exact: true }).waitFor({ state: 'visible' })
    await saveScreenshot(page, 'workbench_light_1440x900.png')

    await page.evaluate(() => {
      localStorage.setItem('lunarc.density', 'compact')
      localStorage.setItem('lunarc.reduced-motion', 'true')
    })
    await page.reload()
    await page.getByRole('button', { name: '切换深色模式', exact: true }).waitFor({ state: 'visible', timeout: 20000 })
    const restoredPreferences = await page.evaluate(() => ({
      density: document.documentElement.dataset.density,
      reducedMotion: document.documentElement.dataset.reducedMotion,
    }))
    if (restoredPreferences.density !== 'compact' || restoredPreferences.reducedMotion !== 'true') {
      throw new Error(`Display preferences did not restore on the workbench: ${JSON.stringify(restoredPreferences)}`)
    }
    result.restoredDisplayPreferences = restoredPreferences
    await waitForWorkbenchReload(page, {}, false)
    await ensureImportedPreview(page)
    result.reloadReadyLight = true
    await setWindowSize(1000, 680)
    result.layout1000Light = await verifyNoOverflow(page, 1000, 680)
    await saveScreenshot(page, 'workbench_light_1000x680.png')

    await page.getByRole('button', { name: '切换深色模式', exact: true }).click()
    await page.getByRole('button', { name: '切换浅色模式', exact: true }).waitFor({ state: 'visible' })
    result.layout1000Dark = await verifyNoOverflow(page, 1000, 680)
    await saveScreenshot(page, 'workbench_dark_1000x680.png')
    await page.evaluate(() => {
      localStorage.setItem('lunarc.density', 'comfortable')
      localStorage.setItem('lunarc.reduced-motion', 'false')
    })
    await page.reload()
    await page.getByRole('button', { name: '切换浅色模式', exact: true }).waitFor({ state: 'visible', timeout: 20000 })
    await waitForWorkbenchReload(page, {}, false)
    await ensureImportedPreview(page)
    result.reloadReadyDark = true
    result.status = 'PASS'
    interactions.push('packaged visual-only run: both themes and sizes captured after decoded-preview gates')
  } else {

  // The simplified top-level editor deliberately exposes only the common controls.
  // The longer feature/layer/post/temporal/HDR panels remain behind one closed disclosure.
  const advanced = page.locator('.qi-advanced')
  await advanced.waitFor({ state: 'visible' })
  if (await advanced.evaluate((el) => el.open)) throw new Error('高级 details must start closed')
  if (await advanced.getByRole('tab').count() !== 0) throw new Error('Advanced tabs must stay hidden while details are closed')
  if (await basicGroup.getByRole('button').count() !== 3 ||
      await basicGroup.locator('input[type="range"]').count() !== 4 ||
      await basicGroup.getByRole('checkbox').count() !== 1) {
    throw new Error('Basic controls must contain exactly three style buttons, four sliders and one checkbox')
  }
  const basicSliders = [
    { display: 'NR强度', name: /NR\s*强度/, key: 'intensity', value: 1.35 },
    { display: '局部色调强度', name: /局部色调强度/, key: 'local_tone', value: 0.85 },
    { display: '局部结构强度', name: /局部结构强度/, key: 'local_struct', value: 1.15 },
    { display: '皮肤结构强度', name: /皮肤结构强度/, key: 'skin_struct', value: 0.75 },
  ]
  for (const control of basicSliders) {
    const slider = basicGroup.getByRole('slider', { name: control.name })
    await slider.waitFor({ state: 'visible' })
    if (await slider.evaluate((el) => Boolean(el.closest('details')))) {
      throw new Error(`Basic slider ${control.display} is nested inside the closed Advanced details`)
    }
  }
  const autoMask = basicGroup.getByRole('checkbox', { name: /自动.*遮罩/ })
  await autoMask.waitFor({ state: 'visible' })
  if (await autoMask.evaluate((el) => Boolean(el.closest('details')))) {
    throw new Error('Automatic mask must remain in the basic controls')
  }
  for (const [role, advancedName] of [['slider', '工作缩放'], ['slider', '总残差强度'],
    ['checkbox', '场景切换检测'], ['checkbox', '实验性 HDR/scRGB']]) {
    if (await page.getByRole(role, { name: new RegExp(advancedName) }).count()) {
      throw new Error(`Advanced control is exposed before opening details: ${advancedName}`)
    }
  }
  interactions.push('basic editor contract verified with Advanced details closed')

  const styles = [
    { label: /默认|Default/, value: 0 },
    { label: /电影|Movie/, value: 2 },
    { label: /自然|Natural/, value: 1 },
  ]
  for (const style of styles) {
    await basicGroup.getByRole('button', { name: style.label }).click()
    await waitForSettings(page, { style: style.value }, `style=${style.value}`)
  }
  result.styleMapping = { default: 0, movie: 2, natural: 1 }
  interactions.push('default/movie/natural style buttons persist style 0/2/1 to backend')

  for (const control of basicSliders) {
    await setSlider(page, control.name, control.value, control.key)
  }
  const maskWasChecked = await autoMask.isChecked()
  await autoMask.setChecked(!maskWasChecked)
  const expectedMask = maskWasChecked ? 0 : 1
  await waitForSettings(page, { use_auto_mask: expectedMask }, `use_auto_mask=${expectedMask}`)
  result.basicSettingsMutation = { intensity: 1.35, local_tone: 0.85, local_struct: 1.15,
    skin_struct: 0.75, use_auto_mask: expectedMask }
  interactions.push('four basic sliders and automatic mask persisted to backend settings')

  await advanced.locator(':scope > summary').click()
  if (!(await advanced.evaluate((el) => el.open))) throw new Error('Advanced details did not open')
  const tabs = ['基础', '图层', '画质', '时域', 'HDR']
  for (const name of tabs) await advanced.getByRole('tab', { name: new RegExp(name) }).waitFor({ state: 'visible' })
  if (await advanced.getByRole('tab').count() !== tabs.length) throw new Error('Advanced disclosure must retain exactly the five original tabs')
  await advanced.getByRole('tab', { name: /图层/ }).click()
  const reconstructionMode = advanced.getByRole('combobox', { name: /重建层数/ })
  await reconstructionMode.waitFor({ state: 'visible' })
  await reconstructionMode.selectOption('dual')
  await waitForSettings(page, { reconstruction_mode: 'dual', export_backend: 'gpu_transfer' },
    'dual reconstruction selects GPU Transfer')
  const dualSettings = await api(page, '/api/settings')
  const dualLayerScales = (dualSettings.settings || dualSettings).layer_scales
  if (!Array.isArray(dualLayerScales) || new Set(dualLayerScales).size !== dualLayerScales.length) {
    throw new Error(`Dual reconstruction produced duplicate or invalid layer scales: ${JSON.stringify(dualLayerScales)}`)
  }
  result.dualMode = { reconstruction_mode: 'dual', export_backend: 'gpu_transfer', layer_scales: dualLayerScales }
  await reconstructionMode.selectOption('single')
  await waitForSettings(page, { reconstruction_mode: 'single' }, 'restore single reconstruction')
  await advanced.getByRole('tab', { name: /基础/ }).click()
  const exportBackend = advanced.getByRole('combobox', { name: '导出后端', exact: true })
  await exportBackend.waitFor({ state: 'visible' })
  const currentSettings = await api(page, '/api/settings')
  const currentExportBackend = (currentSettings.settings || currentSettings).export_backend
  if (currentExportBackend !== initialSettings.export_backend) {
    await exportBackend.selectOption(initialSettings.export_backend)
  }
  await waitForSettings(page, { export_backend: initialSettings.export_backend }, 'restore original export backend')
  result.singleModeRestored = true
  await advanced.getByRole('tab', { name: /画质/ }).click()
  await advanced.getByRole('slider', { name: /总残差强度/ }).waitFor({ state: 'visible' })
  await advanced.getByRole('tab', { name: /时域/ }).click()
  await advanced.getByRole('checkbox', { name: /场景切换检测/ }).waitFor({ state: 'visible' })
  await advanced.getByRole('tab', { name: /HDR/ }).click()
  await advanced.getByRole('checkbox', { name: /实验性 HDR\/scRGB/ }).waitFor({ state: 'visible' })
  interactions.push('Advanced disclosure retains the original five tabs and their controls')

  // Restore through the UI so both backend state and the app's durable workspace
  // draft return to their original values before theme reloads/screenshots.
  await advanced.locator(':scope > summary').click()
  if (await advanced.evaluate((el) => el.open)) throw new Error('Advanced details did not close')
  const originalStyleLabel = initialSettings.style === 0 ? /默认|Default/
    : initialSettings.style === 2 ? /电影|Movie/ : /自然|Natural/
  await basicGroup.getByRole('button', { name: originalStyleLabel }).click()
  await waitForSettings(page, { style: initialSettings.style }, 'restore original style')
  for (const control of basicSliders) {
    await setSlider(page, control.name, initialSettings[control.key], control.key)
  }
  await autoMask.setChecked(initialSettings.use_auto_mask === 1)
  const restoreFields = ['preset', 'style', 'intensity', 'local_tone', 'local_struct', 'skin_struct', 'use_auto_mask']
  await waitForSettings(page, Object.fromEntries(restoreFields.map((key) => [key, initialSettings[key]])),
    'restore original backend settings')
  await waitForSettings(page, { reconstruction_mode: initialSettings.reconstruction_mode || 'single',
    export_backend: initialSettings.export_backend }, 'restore original layer and export settings')
  result.settingsRestored = true
  interactions.push('original backend settings restored after mutations')

  const lightModeButton = page.getByRole('button', { name: '切换浅色模式', exact: true })
  await lightModeButton.waitFor({ state: 'visible' })
  await setWindowSize(1440, 900)
  result.layout1440 = await verifyNoOverflow(page, 1440, 900)
  await saveScreenshot(page, 'workbench_dark_1440x900.png')
  await lightModeButton.click()
  await page.getByRole('button', { name: '切换深色模式', exact: true }).waitFor({ state: 'visible' })
  result.layout1440Light = await verifyNoOverflow(page, 1440, 900)
  await saveScreenshot(page, 'workbench_light_1440x900.png')
  await page.reload()
  await page.getByRole('button', { name: '切换深色模式', exact: true }).waitFor({ state: 'visible', timeout: 20000 })
  await waitForWorkbenchReload(page, Object.fromEntries(restoreFields.map((key) => [key, initialSettings[key]])), false)
  await ensureImportedPreview(page)
  result.reloadReadyLight = true
  interactions.push('light theme persists across reload')

  await setWindowSize(1000, 680)
  result.layout1000Light = await verifyNoOverflow(page, 1000, 680)
  await saveScreenshot(page, 'workbench_light_1000x680.png')
  await page.getByRole('button', { name: '切换深色模式', exact: true }).click()
  await page.getByRole('button', { name: '切换浅色模式', exact: true }).waitFor({ state: 'visible' })
  result.layout1000Dark = await verifyNoOverflow(page, 1000, 680)
  await saveScreenshot(page, 'workbench_dark_1000x680.png')
  await page.reload()
  await page.getByRole('button', { name: '切换浅色模式', exact: true }).waitFor({ state: 'visible', timeout: 20000 })
  await waitForWorkbenchReload(page, Object.fromEntries(restoreFields.map((key) => [key, initialSettings[key]])), false)
  await ensureImportedPreview(page)
  result.reloadReadyDark = true
  interactions.push('dark theme persists across reload; both themes captured at 1440x900 and 1000x680')

  const routes = [
    ['deliver', /导出与队列/, /导出设置/],
    ['diagnostics', /运行与诊断/, /运行与诊断/], ['presets', /预设管理/, /预设管理/],
    ['settings', /软件设置/, /软件设置/], ['about', /关于 Lunarc/, /Lunarc\s*2\.1/],
    ['home', /首页/, /每一帧/], ['dlss', /重建工作台/, null],
  ]
  for (const [route, accessibleName, heading] of routes) {
    await nav.getByRole('button', { name: accessibleName }).click()
    if (route === 'dlss') await basicGroup.waitFor({ state: 'visible' })
    else if (route === 'home') await main.getByRole('heading', { name: heading }).waitFor({ state: 'visible' })
    else if (route === 'about') await main.getByRole('heading', { name: heading }).waitFor({ state: 'visible' })
    else await main.getByText(heading, { exact: false }).first().waitFor({ state: 'visible' })
  }
  result.routes = routes.map(([name]) => name)
  interactions.push('all original workspace routes remain reachable')

  if (errors.length) throw new Error(`Renderer errors: ${errors.join('\n')}`)
  result.status = 'PASS'
  }
} catch (error) {
  result.status = 'FAIL'
  result.error = String(error?.stack || error)
  if (app) {
    try { await app.firstWindow().then((page) => page.screenshot({ path: path.join(out, 'failure.png') })) } catch {}
  }
  throw error
} finally {
  if (app && initialSettings) {
    try {
      const page = await app.firstWindow()
      await api(page, '/api/settings', 'POST', initialSettings)
      result.settingsRestored = true
    } catch (error) {
      result.settingsRestoreError = String(error)
    }
  }
  if (app) await app.close().catch(() => {})
  if (backendPid) {
    let alive = true
    try { process.kill(backendPid, 0) } catch { alive = false }
    if (alive && process.platform === 'win32') {
      const pid = ownedBackendPid()
      const commandLine = pid ? processCommandLine(pid) : ''
      const expected = (packaged ? path.join(path.dirname(executable), 'resources/server.py')
        : path.join(ROOT_DIR, 'server.py')).replaceAll('\\', '/')
      if (pid === backendPid && commandLine.replaceAll('\\', '/').toLowerCase().includes(expected.toLowerCase())) {
        execFileSync('taskkill.exe', ['/F', '/T', '/PID', String(pid)], { encoding: 'utf8', windowsHide: true })
      } else {
        result.cleanupError = `Refusing to terminate an unverified process: pid=${pid}, command=${commandLine}`
      }
      try { process.kill(backendPid, 0); alive = true } catch { alive = false }
    }
    result.backendCleanup = { backendPid, backendExited: !alive }
    if (alive) result.cleanupError = `Owned backend ${backendPid} survived cleanup`
  } else result.backendCleanup = { backendPid: null, backendExited: null, reason: 'No owned backend PID recorded' }
  result.errors = errors
  await fs.writeFile(path.join(out, 'result.json'), JSON.stringify(result, null, 2))
}

if (result.status !== 'PASS' || result.cleanupError) {
  throw new Error(result.cleanupError || `Swiss UI probe did not pass: ${result.error || result.status}`)
}
console.log(`Real Electron Swiss UI contract: ${result.status}`)
console.log(`Evidence: ${out}`)
