// tests/helpers/backend_client.js
import net from 'node:net';
import path from 'node:path';
import { spawn, execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { pythonExe } from './paths.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, '../..');

export const BACKEND_PORT = 18923;
export const BACKEND_URL = `http://127.0.0.1:${BACKEND_PORT}`;
// Local-backend session token (P0-01). Test-spawned backends use this fixed
// value; Electron-spawned backends use a per-launch random token instead.
export const LOCAL_TOKEN = process.env.DLSS_LOCAL_TOKEN || 'e2e-test-token';

export function wsUrl(path) {
  return `ws://127.0.0.1:${BACKEND_PORT}/ws/${path}?token=${encodeURIComponent(LOCAL_TOKEN)}`;
}

let _backendProc = null;

export function canBindPort(port = BACKEND_PORT) {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once('error', () => resolve(false));
    server.once('listening', () => {
      server.close(() => resolve(true));
    });
    server.listen(port, '127.0.0.1');
  });
}

export async function stopBackend() {
  if (_backendProc) {
    try {
      _backendProc.kill();
    } catch {}
    _backendProc = null;
  }
  try {
    const stdout = execSync(
      `powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort ${BACKEND_PORT} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess"`,
      { encoding: 'utf-8' }
    ).trim();
    for (const line of stdout.split(/\r?\n/)) {
      const pid = parseInt(line.trim(), 10);
      if (pid && !isNaN(pid)) {
        try {
          execSync(`taskkill /F /T /PID ${pid} 2>nul`);
        } catch {}
      }
    }
  } catch {}
  // Verify the port can truly be bound before returning: ensures Windows has
  // fully released the socket from TIME_WAIT / closing state.
  for (let i = 0; i < 20; i++) {
    if (await canBindPort(BACKEND_PORT)) {
      return;
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  console.warn(`[stopBackend] port ${BACKEND_PORT} still busy after 10s; continuing anyway`);
}

export async function restartBackend() {
  await stopBackend();
  return ensureBackendRunning();
}

export async function ensureBackendRunning() {
  if (await checkBackendReady()) {
    return true;
  }
  const pythonPath = pythonExe();
  const serverScript = path.join(ROOT_DIR, 'server.py');
  _backendProc = spawn(pythonPath, [serverScript, '--port', '18923'], {
    cwd: ROOT_DIR,
    stdio: 'ignore',
    windowsHide: true,
    env: { ...process.env, DLSS_LOCAL_TOKEN: LOCAL_TOKEN, PYTHON_PATH: pythonPath },
  });
  await waitForBackendReady(25000);
  return true;
}

export async function requestJson(endpoint, options = {}) {
  const url = `${BACKEND_URL}${endpoint}`;
  const method = options.method || 'GET';
  const headers = { 'Content-Type': 'application/json', 'X-DLSS-Token': LOCAL_TOKEN, ...options.headers };
  const body = options.body ? JSON.stringify(options.body) : undefined;

  const res = await fetch(url, { method, headers, body });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} ${res.statusText}: ${JSON.stringify(data)}`);
  }
  return data;
}

export async function checkBackendReady() {
  try {
    const data = await requestJson('/api/system');
    return data && data.backend_ready === true;
  } catch {
    return false;
  }
}

export async function waitForBackendReady(timeoutMs = 15000, pollIntervalMs = 500) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (await checkBackendReady()) {
      return true;
    }
    await new Promise((r) => setTimeout(r, pollIntervalMs));
  }
  throw new Error(`Python backend did not become ready within ${timeoutMs}ms at ${BACKEND_URL}`);
}

export async function getSystemInfo() {
  return requestJson('/api/system');
}

export async function loadVideo(videoPath) {
  return requestJson('/api/video/load', {
    method: 'POST',
    body: { path: videoPath },
  });
}

export async function getCurrentVideo() {
  return requestJson('/api/video/current');
}

export async function getQueue() {
  return requestJson('/api/queue');
}

export async function addJob(jobData) {
  return requestJson('/api/queue/add', {
    method: 'POST',
    body: jobData,
  });
}

export async function startQueue() {
  let res = await requestJson('/api/queue/start', { method: 'POST' });
  const start = Date.now();
  // If thread was just terminating from previous job, wait up to 3s and retry
  while (res.status === 'already_running' && Date.now() - start < 4000) {
    await new Promise((r) => setTimeout(r, 400));
    res = await requestJson('/api/queue/start', { method: 'POST' });
  }
  return res;
}

export async function cancelQueue() {
  return requestJson('/api/queue/cancel', { method: 'POST' });
}

export async function clearQueue() {
  try {
    const q = await getQueue();
    // If active job, wait briefly
    if (q.active_id) {
      await new Promise((r) => setTimeout(r, 1000));
    }
    const q2 = await getQueue();
    for (const j of q2.jobs || []) {
      if (j.status !== 'running') {
        await requestJson('/api/queue/remove', { method: 'POST', body: { id: j.id } }).catch(() => {});
      }
    }
    await new Promise((r) => setTimeout(r, 500));
  } catch {}
}

// The states a job can no longer leave on its own. 'skipped' belongs here as much as
// 'completed' does: the queue has moved past the row and written nothing, so a caller
// waiting on the outcome is answered — it is just answered with "nothing was rendered".
const TERMINAL_JOB_STATUSES = ['completed', 'skipped', 'failed', 'cancelled'];

export async function waitForJob(jobId, timeoutMs = 90000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const q = await getQueue();
    const job = q.jobs?.find((j) => j.id === jobId);
    if (!job) {
      throw new Error(`Job ${jobId} not found in queue`);
    }
    if (Date.now() - start > 1000 && (Date.now() - start) % 3000 < 600) {
      console.log(`[waitForJob ${jobId}] status: ${job.status}, active_id: ${q.active_id}`);
    }
    if (TERMINAL_JOB_STATUSES.includes(job.status)) {
      return job;
    }
    await new Promise((r) => setTimeout(r, 600));
  }
  throw new Error(`Job ${jobId} timed out after ${timeoutMs}ms`);
}
