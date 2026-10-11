// tests/helpers/electron_runner.js
import { _electron as electron } from '@playwright/test';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { waitForBackendReady, stopBackend } from './backend_client.js';
import { pythonExe } from './paths.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
export const ROOT_DIR = path.resolve(__dirname, '../..');
export const FRONTEND_DIR = path.resolve(ROOT_DIR, 'frontend');
export const ELECTRON_EXECUTABLE = path.resolve(FRONTEND_DIR, 'node_modules/electron/dist/electron.exe');

export async function launchApp({ mockFilePath = null, mockDirPath = null, extraEnv = {}, skipStopBackend = false } = {}) {
  if (process.platform === 'win32') {
    try {
      const { execSync } = await import('node:child_process');
      execSync(`powershell -NoProfile -Command "Get-Process electron -ErrorAction SilentlyContinue | Where-Object Path -like '*dlss5standaloneV2*' | Stop-Process -Force -ErrorAction SilentlyContinue"`);
    } catch {}
  }
  if (!skipStopBackend) {
    await stopBackend();
    await new Promise((r) => setTimeout(r, 1000));
  }
  const tmpUserData = path.join(os.tmpdir(), `dlss_e2e_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`);
  const electronApp = await electron.launch({
    executablePath: ELECTRON_EXECUTABLE,
    args: ['.', `--user-data-dir=${tmpUserData}`, '--disable-gpu', '--disable-software-rasterizer', '--no-sandbox'],
    cwd: FRONTEND_DIR,
    env: {
      ...process.env,
      E2E_TEST: '1',
      E2E_MOCK_FILE_PATH: mockFilePath || '',
      E2E_MOCK_DIR_PATH: mockDirPath || '',
      PYTHON_PATH: pythonExe(),
      DLSS_LOCAL_TOKEN: process.env.DLSS_LOCAL_TOKEN || 'e2e-test-token',
      ...extraEnv,
    },
  });

  const proc = electronApp.process();
  proc.stdout?.on('data', (d) => process.stdout.write(`[Electron stdout] ${d}`));
  proc.stderr?.on('data', (d) => process.stderr.write(`[Electron stderr] ${d}`));

  const page = await electronApp.firstWindow({ timeout: 60000 });
  await page.waitForLoadState('domcontentloaded');
  // Never let a native dialog (alert/confirm/error box) wedge the run and
  // orphan the app: auto-dismiss, tests assert on DOM/backend state instead.
  page.on('dialog', (d) => {
    console.log(`[e2e] auto-dismissed dialog: ${d.type()} ${(d.message() || '').slice(0, 120)}`);
    d.dismiss().catch(() => {});
  });

  // Ensure python backend is ready
  await waitForBackendReady(25000);

  const origClose = electronApp.close.bind(electronApp);
  electronApp.close = async () => {
    try {
      await Promise.race([
        origClose(),
        new Promise((_, reject) => setTimeout(() => reject(new Error('close timeout')), 4000)),
      ]);
    } catch {}
    try {
      const proc = electronApp.process();
      const pid = proc?.pid;
      if (pid && process.platform === 'win32') {
        const { execSync } = await import('node:child_process');
        try {
          execSync(`taskkill /F /T /PID ${pid} 2>nul`);
        } catch {}
      }
    } catch {}
    if (process.platform === 'win32') {
      try {
        const { execSync } = await import('node:child_process');
        execSync(`powershell -NoProfile -Command "Get-Process electron -ErrorAction SilentlyContinue | Where-Object Path -like '*dlss5standaloneV2*' | Stop-Process -Force -ErrorAction SilentlyContinue"`);
      } catch {}
    }
    await stopBackend();
  };

  return { electronApp, page };
}

export async function setMockFilePath(electronApp, filePath) {
  await electronApp.evaluate(({ ipcMain: _ipcMain }, p) => {
    // Overriding the in-memory mock path
    return p;
  }, filePath);
}
