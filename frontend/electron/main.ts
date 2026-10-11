import { app, BrowserWindow, dialog, ipcMain, protocol, net } from 'electron'
import crypto from 'node:crypto'
import path from 'node:path'
import http from 'node:http'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { spawn, execFileSync, ChildProcess } from 'node:child_process'
import fs from 'node:fs'
import {runtimeCandidates, runtimeDirectory, runtimeFingerprint, runtimeEnvironment, probeRuntime, type BackendStartup} from './backendRuntime'

protocol.registerSchemesAsPrivileged([
  {
    scheme: 'app',
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      corsEnabled: true,
      stream: true,
    },
  },
])

const __dirname = path.dirname(fileURLToPath(import.meta.url))

process.env.APP_ROOT = path.join(__dirname, '..')

export const VITE_DEV_SERVER_URL = process.env['VITE_DEV_SERVER_URL']
export const MAIN_DIST = path.join(process.env.APP_ROOT, 'dist-electron')
export const RENDERER_DIST = path.join(process.env.APP_ROOT, 'dist')

process.env.VITE_PUBLIC = VITE_DEV_SERVER_URL ? path.join(process.env.APP_ROOT, 'public') : RENDERER_DIST

try {
  if (process.env.E2E_TEST !== '1') {
    // Keep the existing storage path to preserve Chromium/localStorage and user configuration.
    // New Lunarc drafts use explicit v1 keys and default only newly added fields.
    app.setPath('userData', path.join(app.getPath('appData'), 'DaVinci_DLSS5_Studio'))
  }
} catch {
  // Ignore
}

let win: BrowserWindow | null = null
let pyProc: ChildProcess | null = null
let backendStartup: BackendStartup = {state: 'checking', message: '正在检查运行环境'}
const backendPort = process.env.E2E_TEST === '1' ? Number(process.env.E2E_BACKEND_PORT || 18923) : 18923

function checkServerListening(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const req = http.get(`http://127.0.0.1:${port}/api/system`, (res) => {
      resolve(res.statusCode === 200)
    })
    req.on('error', () => resolve(false))
    req.setTimeout(500, () => {
      req.destroy()
      resolve(false)
    })
  })
}

async function resolveExistingBackendToken(rootDir: string): Promise<string | null> {
  const fs = await import('node:fs')
  const os = await import('node:os')
  let userDataDir = ''
  try {
    userDataDir = app.getPath('userData')
  } catch {}

  const candidateFiles = [
    path.join(rootDir, '.session_token'),
    userDataDir ? path.join(userDataDir, '.session_token') : '',
    path.join(os.homedir(), 'AppData', 'Roaming', 'DaVinci_DLSS5_Studio', '.session_token'),
    path.join(os.homedir(), '.dlss5', 'session_token'),
  ].filter(Boolean)

  for (const f of candidateFiles) {
    try {
      if (fs.existsSync(f)) {
        const t = fs.readFileSync(f, 'utf-8').trim()
        if (t) return t
      }
    } catch {}
  }
  return null
}

async function startPythonBackend() {
  const rootDir = path.resolve(process.env.APP_ROOT, '..')
  const runtimeDir = runtimeDirectory(rootDir, app.isPackaged)
  const standalone = app.isPackaged || fs.existsSync(path.join(runtimeDir, 'manifest.json'))
  const isRunning = await checkServerListening(backendPort)
  if (isRunning) {
    if (standalone) {
      const expected = runtimeFingerprint(rootDir, runtimeDir)
      const matches = await new Promise<boolean>(resolve => {
        const req = http.get(`http://127.0.0.1:${backendPort}/api/system`, res => {
          let data = ''; res.on('data', chunk => {data += chunk; if (data.length > 65536) req.destroy()})
          res.on('end', () => {try {const info = JSON.parse(data); resolve(!!expected && info.service === 'lunarc' && info.runtime_id === expected)} catch {resolve(false)}})
        })
        req.on('error', () => resolve(false)); req.setTimeout(2000, () => {req.destroy(); resolve(false)})
      })
      if (!matches) {
        backendStartup = {state: 'error', message: '另一个版本的工作台正在运行。请先退出旧版，再启动此版本。'}
        return false
      }
    }
    console.log('Python backend server is already running on port 18923.')
    if (!process.env.DLSS_LOCAL_TOKEN) {
      const existingToken = await resolveExistingBackendToken(rootDir)
      if (existingToken) {
        process.env.DLSS_LOCAL_TOKEN = existingToken
        console.log('[Backend] Successfully negotiated token with existing backend.')
      } else {
        console.warn('[Backend] Existing backend detected but no .session_token found!')
      }
    }
    backendStartup = {state: 'ready', message: '已连接本机后端'}
    return true
  }

  const serverScript = path.join(rootDir, 'backend_entry.py')
  // Per-launch session token for the local backend (P0-01): generated here,
  // handed to the backend via env, and exposed to the renderer via preload.
  // An explicit DLSS_LOCAL_TOKEN env (tests, manual runs) always wins.
  const localToken = process.env.DLSS_LOCAL_TOKEN || crypto.randomUUID()
  process.env.DLSS_LOCAL_TOKEN = localToken
  const userData = app.getPath('userData')
  const environment = runtimeEnvironment(rootDir, userData, {...process.env, LUNARC_RUNTIME_DIR: runtimeDir}, standalone)
  const pythonCandidates = runtimeCandidates(rootDir, app.isPackaged, process.env)
  let pythonPath = ''
  let failure = ''
  for (const c of pythonCandidates) {
    try {
      const result = await probeRuntime(c, rootDir, environment)
      backendStartup.checks = result.checks
      if (result.ready) {pythonPath = c; break}
      failure = '运行环境不完整：' + (result.checks as {status: string; label: string; detail: string}[])
        .filter(c => c.status !== 'ok').map(c => `${c.label}：${c.detail}`).join('；')
    } catch (err) {failure = (err as Error).message}
  }
  if (!pythonPath) {
    const msg = standalone ? `随软件携带的运行环境无法启动，请重新安装。\n${failure}` : `未找到可用的开发运行环境。\n${failure}`
    backendStartup = {...backendStartup, state: 'error', message: msg}
    console.error('[Backend]', msg)
    // E2E runs must never block on a modal dialog (it would wedge teardown).
    if (process.env.E2E_TEST !== '1') {
      dialog.showErrorBox('后端启动失败', msg)
    }
    return false
  }

  try {
    const logDir = path.join(userData, 'logs')
    fs.mkdirSync(logDir, {recursive: true})
    const logPath = path.join(logDir, 'backend.log')
    const log = fs.createWriteStream(logPath, {flags: 'a'})
    log.on('error', err => console.error('[Backend] log error:', err.message))
    log.write(`\n[${new Date().toISOString()}] Backend starting\n`)
    backendStartup = {...backendStartup, state: 'starting', message: '正在启动后端', logPath}
    pyProc = spawn(pythonPath, [serverScript, '--port', String(backendPort)], {
      cwd: rootDir,
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
      env: { ...environment, DLSS_LOCAL_TOKEN: localToken },
    })
    pyProc.stdout?.pipe(log, {end: false})
    pyProc.stderr?.pipe(log, {end: false})
    console.log('Python backend spawned with PID:', pyProc.pid, 'using:', pythonPath)
    // Async spawn errors (e.g. ENOENT) do NOT throw synchronously — listen.
    pyProc.on('error', (err) => {
      backendStartup = {...backendStartup, state: 'error', message: `后端启动失败：${err.message}`}
      log.end()
      console.error('[Backend] spawn error:', err)
      if (process.env.E2E_TEST !== '1') {
        dialog.showErrorBox(
          '后端启动失败',
          `无法启动 Python 后端 (${pythonPath})：${(err as Error).message}\n请检查 PYTHON_PATH 后重试。`
        )
      }
      pyProc = null
    })
    pyProc.on('exit', (code, signal) => {
      log.end()
      backendStartup = {...backendStartup, state: 'error', message: `后端已退出（${code ?? signal}），请查看日志`}
      console.log(`[Backend] exited code=${code} signal=${signal}`)
      try {
        win?.webContents.send('backend-exit', { code, signal: String(signal) })
      } catch {}
      if (process.env.E2E_TEST !== '1' && pyProc && (code !== 0 && code !== null)) {
        dialog.showErrorBox(
          '后端已退出',
          `Python 后端异常退出（code=${code}）。请查看后端终端日志后重新启动应用。`
        )
      }
      pyProc = null
    })
    return true
  } catch (err) {
    console.error('Failed to spawn Python backend:', err)
    backendStartup = {...backendStartup, state: 'error', message: String(err)}
    return false
  }
}

app.setName('Lunarc')

function createWindow() {
  win = new BrowserWindow({
    title: 'Lunarc',
    titleBarStyle: 'hidden',
    ...(process.platform !== 'darwin' ? {
      titleBarOverlay: { color: '#202020', symbolColor: '#efeeea', height: 76 },
    } : {}),
    icon: path.join(process.env.VITE_PUBLIC!, 'lunarc.png'),
    width: 1480,
    height: 940,
    minWidth: 1000,
    minHeight: 680,
    backgroundColor: '#191919',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      nodeIntegration: false,
      contextIsolation: true,
    },
  })

  // Hide default window menu
  win.setMenuBarVisibility(false)

  if (VITE_DEV_SERVER_URL) {
    win.loadURL(VITE_DEV_SERVER_URL)
  } else {
    win.loadURL('app://studio/index.html')
  }
}

// Security: remote debugging port is prohibited in production mode.
if (process.env.E2E_TEST !== '1') {
  if (app.commandLine.hasSwitch('remote-debugging-port')) {
    console.warn('[Security] removing prohibited remote-debugging-port switch.')
    app.commandLine.removeSwitch('remote-debugging-port')
  }
}

// Minimal necessary E2E test interfaces (strictly gated by E2E_TEST === '1')
let e2eMockOpenFile: string | null = process.env.E2E_MOCK_FILE_PATH || null
let e2eMockOpenDir: string | null = process.env.E2E_MOCK_DIR_PATH || null

if (process.env.E2E_TEST === '1') {
  ipcMain.handle('e2e:setMockOpenFilePath', (_event, filePath: string | null) => {
    e2eMockOpenFile = filePath
    return true
  })
  ipcMain.handle('e2e:setMockOpenDirPath', (_event, dirPath: string | null) => {
    e2eMockOpenDir = dirPath
    return true
  })
}

// IPC handlers for native file dialogs
ipcMain.handle('dialog:openFile', async () => {
  if (process.env.E2E_TEST === '1' && e2eMockOpenFile) {
    return e2eMockOpenFile
  }
  if (!win) return null
  const result = await dialog.showOpenDialog(win, {
    title: '选择导入素材 (视频 / 图像)',
    properties: ['openFile'],
    filters: [
      { name: 'Media Files', extensions: ['mp4', 'mkv', 'mov', 'avi', 'wmv', 'flv', 'webm', 'm4v', 'mpg', 'mpeg', 'ts', 'mts', 'm2ts', 'png', 'jpg', 'jpeg', 'bmp', 'webp', 'tif', 'tiff'] },
      { name: 'Video Files', extensions: ['mp4', 'mkv', 'mov', 'avi', 'wmv', 'flv', 'webm', 'm4v', 'mpg', 'mpeg', 'ts', 'mts', 'm2ts'] },
      { name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'bmp', 'webp', 'tif', 'tiff'] },
      { name: 'All Files', extensions: ['*'] },
    ],
  })
  if (result.canceled || result.filePaths.length === 0) return null
  return result.filePaths[0]
})

ipcMain.handle('dialog:openDirectory', async () => {
  if (process.env.E2E_TEST === '1' && e2eMockOpenDir) {
    return e2eMockOpenDir
  }
  if (!win) return null
  const result = await dialog.showOpenDialog(win, {
    title: '选择导出保存目录',
    properties: ['openDirectory', 'createDirectory'],
  })
  if (result.canceled || result.filePaths.length === 0) return null
  return result.filePaths[0]
})

// Authentication token IPC handler (P0-01)
ipcMain.handle('auth:getToken', () => process.env.DLSS_LOCAL_TOKEN || '')
ipcMain.handle('backend:getStartup', (event) => event.sender === win?.webContents ? backendStartup : null)

// Only the application renderer can update the two supported chrome palettes.
ipcMain.handle('window:setTheme', (event, theme: unknown) => {
  if (!win || event.sender !== win.webContents || (theme !== 'light' && theme !== 'dark')) return
  if (process.platform !== 'darwin') {
    win.setTitleBarOverlay(theme === 'light'
      ? { color: '#f7f6f2', symbolColor: '#242522', height: 76 }
      : { color: '#202020', symbolColor: '#efeeea', height: 76 })
  }
})

function killPythonBackend() {
  if (pyProc && pyProc.pid) {
    const pid = pyProc.pid
    try {
      if (process.platform === 'win32') {
        execFileSync('taskkill', ['/F', '/T', '/PID', String(pid)], { windowsHide: true, stdio: 'ignore', timeout: 10000 })
      } else {
        pyProc.kill('SIGKILL')
      }
    } catch (error) {
      console.error('[Backend] process-tree cleanup failed:', error)
      pyProc.kill('SIGKILL')
    }
    pyProc = null
  }
}

app.on('window-all-closed', () => {
  killPythonBackend()
  if (process.platform !== 'darwin') {
    app.quit()
    win = null
  }
})

app.on('before-quit', () => {
  killPythonBackend()
})

app.on('will-quit', () => {
  killPythonBackend()
})

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow()
  }
})

app.whenReady().then(async () => {
  protocol.handle('app', (request) => {
    const url = new URL(request.url)
    let relativePath = decodeURIComponent(url.pathname)
    if (relativePath === '/' || !relativePath) {
      relativePath = '/index.html'
    }
    const filePath = path.join(RENDERER_DIST, relativePath)
    return net.fetch(pathToFileURL(filePath).toString())
  })

  const started = await startPythonBackend()
  // P1-05: wait for the API before painting UI (cold torch/DLL import can
  // take tens of seconds; the page used to load against a dead backend).
  for (let i = 0; started && i < 120 && backendStartup.state !== 'error'; i++) {
    if (await checkServerListening(backendPort)) {backendStartup = {...backendStartup, state: 'ready', message: '后端已连接'}; break}
    await new Promise((r) => setTimeout(r, 500))
  }
  if (backendStartup.state === 'starting') backendStartup = {...backendStartup, state: 'error', message: '后端启动超时，请查看日志'}
  createWindow()
})
