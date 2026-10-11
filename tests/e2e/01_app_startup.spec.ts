import { test, expect } from '@playwright/test';
import { launchApp } from '../helpers/electron_runner.js';
import {
  BACKEND_PORT,
  BACKEND_URL,
  LOCAL_TOKEN,
  getSystemInfo,
  wsUrl,
  ensureBackendRunning,
  stopBackend,
  waitForBackendReady,
} from '../helpers/backend_client.js';
import { pythonExe, ROOT_DIR } from '../helpers/paths.js';
import WebSocket from 'ws';
import path from 'node:path';
import fs from 'node:fs';
import http from 'node:http';
import { spawn } from 'node:child_process';

function rawHttpRequest(options: http.RequestOptions): Promise<{ statusCode?: number; headers: http.IncomingHttpHeaders; body: string }> {
  return new Promise((resolve, reject) => {
    const req = http.request(options, (res) => {
      let body = '';
      res.on('data', (c) => { body += c; });
      res.on('end', () => resolve({ statusCode: res.statusCode, headers: res.headers, body }));
    });
    req.on('error', reject);
    req.end();
  });
}

test.describe('第一组测试：应用启动 (App Startup & Backend Verification)', () => {
  let appInstance: any = null;

  test.afterEach(async () => {
    if (appInstance?.electronApp) {
      await appInstance.electronApp.close().catch(() => {});
      appInstance = null;
    }
    await stopBackend();
  });

  test('1.1 启动 Electron，验证窗口创建，确认 Python 后端 /api/system 就绪与 WebSocket 连通', async () => {
    // 1. Launch Electron
    appInstance = await launchApp();
    const { page } = appInstance;

    // 2. Confirm window created, title is correct, and origin is fixed custom scheme app://studio
    const title = await page.title();
    expect(title).toContain('DaVinci Resolve');
    expect(page.url()).toBe('app://studio/index.html');

    const actualWindowOrigin = await page.evaluate(() => window.origin);
    console.log('[Test 1.1] actual window.origin:', actualWindowOrigin);
    expect(actualWindowOrigin).toBe('app://studio');

    const electronApiType = await page.evaluate(() => typeof (window as any).electronAPI);
    expect(electronApiType).toBe('object');
    const localToken = await page.evaluate(() => (window as any).electronAPI?.localToken);
    expect(localToken).toBeTruthy();
    const ipcToken = await page.evaluate(async () => await (window as any).electronAPI?.getToken?.());
    expect(ipcToken).toBe(localToken);

    // 3. Confirm Python backend /api/system response & ready status
    const sysInfo = await getSystemInfo();
    expect(sysInfo).toBeDefined();
    expect(sysInfo.backend_ready).toBe(true);
    expect(sysInfo.gpu).toBeTruthy();
    expect(sysInfo.vram_total_gb).toBeGreaterThan(0);

    // 4. Confirm WebSocket connections (/ws/player and /ws/queue)
    const wsPlayerSuccess = await new Promise<boolean>((resolve) => {
      const ws = new WebSocket(wsUrl('player'), { headers: { Origin: 'app://studio' } });
      ws.on('open', () => {
        ws.close();
        resolve(true);
      });
      ws.on('error', () => resolve(false));
      setTimeout(() => resolve(false), 5000);
    });
    expect(wsPlayerSuccess).toBe(true);

    const wsQueueSuccess = await new Promise<boolean>((resolve) => {
      const ws = new WebSocket(wsUrl('queue'), { headers: { Origin: 'app://studio' } });
      ws.on('open', () => {
        ws.close();
        resolve(true);
      });
      ws.on('error', () => resolve(false));
      setTimeout(() => resolve(false), 5000);
    });
    expect(wsQueueSuccess).toBe(true);

    // 5. Check UI displays hardware status
    const hardwarePill = page.locator('header').locator('text=RTX');
    await expect(hardwarePill).toBeVisible({ timeout: 10000 });
  });

  test('1.2 模拟后端未启动/不可用时测试必须失败 (Strict Startup Guard)', async () => {
    const invalidBackendReachable = await fetch('http://127.0.0.1:54321/api/system')
      .then((r) => r.ok)
      .catch(() => false);
    expect(invalidBackendReachable).toBe(false);
  });

  test('1.3 控制面三层防御与 CORS 预检验证 (P0-01 核心门禁：拒绝 file://, null, devtools, 宽泛 app://)', async () => {
    await ensureBackendRunning();

    // --- 1. REST API Token 门禁 ---
    // 无 Token 访问受控 API -> 401
    const noToken = await fetch(`${BACKEND_URL}/api/queue`);
    expect(noToken.status).toBe(401);

    // 错 Token 访问受控 API -> 401
    const badToken = await fetch(`${BACKEND_URL}/api/queue`, {
      headers: { 'X-DLSS-Token': 'wrong-token-12345' },
    });
    expect(badToken.status).toBe(401);

    // 正确 Token 访问受控 API -> 200
    const validToken = await fetch(`${BACKEND_URL}/api/queue`, {
      headers: { 'X-DLSS-Token': LOCAL_TOKEN },
    });
    expect(validToken.status).toBe(200);

    // 探针接口免鉴权 -> 200
    const probe = await fetch(`${BACKEND_URL}/api/system`);
    expect(probe.status).toBe(200);

    // --- 2. CORS 预检 (OPTIONS) 严格验证 ---
    // (a) 合法开发环境 Origin (http://127.0.0.1:18923) -> 200 + CORS headers
    const preflightLocal = await fetch(`${BACKEND_URL}/api/queue`, {
      method: 'OPTIONS',
      headers: {
        Origin: `http://127.0.0.1:${BACKEND_PORT}`,
        'Access-Control-Request-Method': 'GET',
        'Access-Control-Request-Headers': 'x-dlss-token',
      },
    });
    expect(preflightLocal.status).toBe(200);
    expect(preflightLocal.headers.get('access-control-allow-origin')).toBe(`http://127.0.0.1:${BACKEND_PORT}`);
    expect(preflightLocal.headers.get('access-control-allow-headers')?.toLowerCase()).toContain('x-dlss-token');

    // (b) 生产 Electron 固定自定义 scheme app://studio 预检 -> 200 + CORS headers
    const preflightApp = await fetch(`${BACKEND_URL}/api/queue`, {
      method: 'OPTIONS',
      headers: {
        Origin: 'app://studio',
        'Access-Control-Request-Method': 'GET',
        'Access-Control-Request-Headers': 'x-dlss-token',
      },
    });
    expect(preflightApp.status).toBe(200);
    expect(preflightApp.headers.get('access-control-allow-origin')).toBe('app://studio');

    // (c) 拒绝 file:// Origin 预检 -> 403 (明确禁止 file://.* 作为生产 CORS 修复方案)
    const preflightFile = await fetch(`${BACKEND_URL}/api/queue`, {
      method: 'OPTIONS',
      headers: {
        Origin: 'file://',
        'Access-Control-Request-Method': 'GET',
        'Access-Control-Request-Headers': 'x-dlss-token',
      },
    });
    expect(preflightFile.status).toBe(403);

    // (d) 拒绝 null Origin 预检 -> 403 (严格防御 iframe sandbox bypass)
    const preflightNull = await fetch(`${BACKEND_URL}/api/queue`, {
      method: 'OPTIONS',
      headers: {
        Origin: 'null',
        'Access-Control-Request-Method': 'GET',
        'Access-Control-Request-Headers': 'x-dlss-token',
      },
    });
    expect(preflightNull.status).toBe(403);

    // (e) 拒绝 devtools://.* 预检 -> 403
    const preflightDevTools = await fetch(`${BACKEND_URL}/api/queue`, {
      method: 'OPTIONS',
      headers: {
        Origin: 'devtools://devtools',
        'Access-Control-Request-Method': 'GET',
        'Access-Control-Request-Headers': 'x-dlss-token',
      },
    });
    expect(preflightDevTools.status).toBe(403);

    // (f) 拒绝宽泛 app://.* 预检 -> 403 (仅允许固定 app://studio)
    const preflightBroadApp = await fetch(`${BACKEND_URL}/api/queue`, {
      method: 'OPTIONS',
      headers: {
        Origin: 'app://evil-app',
        'Access-Control-Request-Method': 'GET',
        'Access-Control-Request-Headers': 'x-dlss-token',
      },
    });
    expect(preflightBroadApp.status).toBe(403);

    // (g) 拒绝外部恶意网站预检 -> 403
    const preflightEvil = await fetch(`${BACKEND_URL}/api/queue`, {
      method: 'OPTIONS',
      headers: {
        Origin: 'http://evil.example',
        'Access-Control-Request-Method': 'GET',
        'Access-Control-Request-Headers': 'x-dlss-token',
      },
    });
    expect(preflightEvil.status).toBe(403);

    // (h) 拒绝非回环 Host 预检 -> 403 (DNS-rebinding 防护)
    const preflightBadHost = await rawHttpRequest({
      host: '127.0.0.1',
      port: BACKEND_PORT,
      path: '/api/queue',
      method: 'OPTIONS',
      headers: {
        host: 'evil.example',
        origin: `http://127.0.0.1:${BACKEND_PORT}`,
        'access-control-request-method': 'GET',
        'access-control-request-headers': 'x-dlss-token',
      },
    });
    expect(preflightBadHost.statusCode).toBe(403);

    // --- 3. 来源与主机 (Origin / Host) 业务请求门禁 ---
    // 恶意 Origin 访问开放探针 -> 403
    const evilOrigin = await fetch(`${BACKEND_URL}/api/system`, {
      headers: { Origin: 'http://evil.example' },
    });
    expect(evilOrigin.status).toBe(403);

    // 沙箱 null Origin 访问开放探针 -> 403
    const nullOrigin = await fetch(`${BACKEND_URL}/api/system`, {
      headers: { Origin: 'null' },
    });
    expect(nullOrigin.status).toBe(403);

    // file:// Origin 访问开放探针 -> 403
    const fileOrigin = await fetch(`${BACKEND_URL}/api/system`, {
      headers: { Origin: 'file://' },
    });
    expect(fileOrigin.status).toBe(403);

    // devtools:// Origin 访问开放探针 -> 403
    const devtoolsOrigin = await fetch(`${BACKEND_URL}/api/system`, {
      headers: { Origin: 'devtools://devtools' },
    });
    expect(devtoolsOrigin.status).toBe(403);

    // 宽泛 app:// 访问开放探针 -> 403
    const broadAppOrigin = await fetch(`${BACKEND_URL}/api/system`, {
      headers: { Origin: 'app://other-app' },
    });
    expect(broadAppOrigin.status).toBe(403);

    // 伪造 Host 访问开放探针 -> 403
    const evilHost = await rawHttpRequest({
      host: '127.0.0.1',
      port: BACKEND_PORT,
      path: '/api/system',
      method: 'GET',
      headers: { host: 'evil.example' },
    });
    expect(evilHost.statusCode).toBe(403);

    // 生产 Electron app://studio 携带合法 Token 访问 -> 200 且附带 Allow-Origin
    const appReq = await fetch(`${BACKEND_URL}/api/queue`, {
      headers: {
        Origin: 'app://studio',
        'X-DLSS-Token': LOCAL_TOKEN,
      },
    });
    expect(appReq.status).toBe(200);
    expect(appReq.headers.get('access-control-allow-origin')).toBe('app://studio');

    // --- 4. WebSocket 严格安全拒绝门禁 ---
    const testWS = (url: string, headers: Record<string, string> = {}): Promise<{ opened: boolean; rejected: boolean }> => {
      return new Promise((resolve) => {
        let opened = false;
        const ws = new WebSocket(url, { headers });
        ws.on('open', () => {
          opened = true;
          ws.close();
          resolve({ opened: true, rejected: false });
        });
        ws.on('unexpected-response', (_req, res) => {
          resolve({ opened: false, rejected: res.statusCode === 403 || res.statusCode === 401 });
        });
        ws.on('close', (code) => {
          if (!opened) {
            resolve({ opened: false, rejected: code === 4401 || code === 1006 });
          }
        });
        ws.on('error', () => {});
        setTimeout(() => {
          if (!opened) {
            ws.terminate();
            resolve({ opened: false, rejected: false });
          }
        }, 4000);
      });
    };

    // 无 Token WebSocket 连接 -> 拒绝
    const wsNoToken = await testWS(`ws://127.0.0.1:${BACKEND_PORT}/ws/player`);
    expect(wsNoToken.opened).toBe(false);
    expect(wsNoToken.rejected).toBe(true);

    // 错 Token WebSocket 连接 -> 拒绝
    const wsBadToken = await testWS(`ws://127.0.0.1:${BACKEND_PORT}/ws/player?token=invalid-ws-token`);
    expect(wsBadToken.opened).toBe(false);
    expect(wsBadToken.rejected).toBe(true);

    // file:// Origin WebSocket 连接 -> 拒绝
    const wsFileOrigin = await testWS(wsUrl('player'), { Origin: 'file://' });
    expect(wsFileOrigin.opened).toBe(false);
    expect(wsFileOrigin.rejected).toBe(true);

    // devtools:// Origin WebSocket 连接 -> 拒绝
    const wsDevtoolsOrigin = await testWS(wsUrl('player'), { Origin: 'devtools://devtools' });
    expect(wsDevtoolsOrigin.opened).toBe(false);
    expect(wsDevtoolsOrigin.rejected).toBe(true);

    // 宽泛 app:// Origin WebSocket 连接 -> 拒绝
    const wsBroadAppOrigin = await testWS(wsUrl('player'), { Origin: 'app://other-app' });
    expect(wsBroadAppOrigin.opened).toBe(false);
    expect(wsBroadAppOrigin.rejected).toBe(true);

    // 恶意跨站 Origin WebSocket 连接 -> 拒绝
    const wsEvilOrigin = await testWS(wsUrl('player'), { Origin: 'http://evil.example' });
    expect(wsEvilOrigin.opened).toBe(false);
    expect(wsEvilOrigin.rejected).toBe(true);

    // 非回环 Host WebSocket 连接 -> 拒绝
    const wsEvilHost = await testWS(wsUrl('player'), { Host: 'evil.example' });
    expect(wsEvilHost.opened).toBe(false);
    expect(wsEvilHost.rejected).toBe(true);

    // 正确 Token 与固定生产 Origin (app://studio) -> 正常连通
    const wsValid = await testWS(wsUrl('player'), { Origin: 'app://studio' });
    expect(wsValid.opened).toBe(true);
  });

  test('1.4 真实 Chromium 渲染进程 (固定 app://studio Origin) 级 REST 与 WebSocket 验证', async () => {
    // 启动实际 Electron 实例（加载本地固定自定义 scheme app://studio/index.html）
    appInstance = await launchApp();
    const { page } = appInstance;

    const pageUrl = page.url();
    expect(pageUrl).toBe('app://studio/index.html');

    // 在 Chromium 渲染上下文执行请求，并记录实际 window.origin 与 backend 收到的 Origin
    const auditInfo = await page.evaluate(async () => {
      const token = (window as any).electronAPI?.localToken;
      const res = await fetch('http://127.0.0.1:18923/api/system', {
        headers: { 'X-DLSS-Token': token },
      });
      const data = await res.json();
      return {
        windowOrigin: window.origin,
        backendReceivedOrigin: data.client_origin,
        status: res.status,
      };
    });

    console.log('=== [E2E Security Audit] ===');
    console.log('Actual renderer window.origin:', auditInfo.windowOrigin);
    console.log('Backend received Origin header:', auditInfo.backendReceivedOrigin);
    console.log('API Status Code:', auditInfo.status);
    console.log('============================');

    expect(auditInfo.windowOrigin).toBe('app://studio');
    expect(auditInfo.backendReceivedOrigin).toBe('app://studio');
    expect(auditInfo.status).toBe(200);

    // 在 Chromium 渲染上下文执行携带 Token 的受控 API 请求
    const fetchQueueStatus = await page.evaluate(async () => {
      const token = (window as any).electronAPI?.localToken;
      const res = await fetch('http://127.0.0.1:18923/api/queue', {
        headers: { 'X-DLSS-Token': token },
      });
      return res.status;
    });
    expect(fetchQueueStatus).toBe(200);

    // 在 Chromium 渲染上下文执行未授权 fetch
    const fetchUnauthorizedStatus = await page.evaluate(async () => {
      const res = await fetch('http://127.0.0.1:18923/api/queue');
      return res.status;
    });
    expect(fetchUnauthorizedStatus).toBe(401);

    // 在 Chromium 渲染上下文执行页面级 WebSocket 握手
    const wsConnectSuccess = await page.evaluate(async () => {
      const token = (window as any).electronAPI?.localToken;
      return new Promise<boolean>((resolve) => {
        try {
          const ws = new WebSocket(`ws://127.0.0.1:18923/ws/player?token=${encodeURIComponent(token)}`);
          ws.onopen = () => {
            ws.close();
            resolve(true);
          };
          ws.onerror = () => resolve(false);
          setTimeout(() => resolve(false), 5000);
        } catch {
          resolve(false);
        }
      });
    });
    expect(wsConnectSuccess).toBe(true);
  });

  test('1.5 已有后端运行时 Electron 启动与 Token 协商验证 (Pre-existing Backend Negotiation)', async () => {
    // 1. 确保先前环境停止
    await stopBackend();

    // 2. 独立启动一个后端进程，使用独立生成的 Token
    const testEphemeralToken = `pre-existing-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const pythonPath = pythonExe();
    const serverScript = path.join(ROOT_DIR, 'server.py');

    const standaloneBackend = spawn(pythonPath, [serverScript, '--port', '18923'], {
      cwd: ROOT_DIR,
      stdio: 'ignore',
      windowsHide: true,
      env: { ...process.env, DLSS_LOCAL_TOKEN: testEphemeralToken, PYTHON_PATH: pythonPath },
    });

    try {
      await waitForBackendReady(25000);

      // 验证本地写入了 .session_token 文件且内容与生成的 Token 一致
      const tokenFilePath = path.join(ROOT_DIR, '.session_token');
      expect(fs.existsSync(tokenFilePath)).toBe(true);
      const savedToken = fs.readFileSync(tokenFilePath, 'utf-8').trim();
      expect(savedToken).toBe(testEphemeralToken);

      // 3. 启动 Electron，不传递 DLSS_LOCAL_TOKEN（模拟已有后端检测与协商流程）
      appInstance = await launchApp({
        skipStopBackend: true,
        extraEnv: { DLSS_LOCAL_TOKEN: '' },
      });
      const { page } = appInstance;

      // 4. 验证渲染进程拿到了协商出的真实 Token
      const negotiatedToken = await page.evaluate(() => (window as any).electronAPI?.localToken);
      expect(negotiatedToken).toBe(testEphemeralToken);

      const ipcNegotiatedToken = await page.evaluate(async () => await (window as any).electronAPI?.getToken?.());
      expect(ipcNegotiatedToken).toBe(testEphemeralToken);

      // 5. 验证渲染进程使用该协商 Token 发起业务 API 成功 200，并记录 Origin
      const auditNegotiation = await page.evaluate(async () => {
        const token = (window as any).electronAPI?.localToken;
        const res = await fetch('http://127.0.0.1:18923/api/system', {
          headers: { 'X-DLSS-Token': token },
        });
        const data = await res.json();
        const queueRes = await fetch('http://127.0.0.1:18923/api/queue', {
          headers: { 'X-DLSS-Token': token },
        });
        return {
          windowOrigin: window.origin,
          backendReceivedOrigin: data.client_origin,
          queueStatus: queueRes.status,
        };
      });

      console.log('=== [Negotiation Security Audit] ===');
      console.log('Renderer window.origin:', auditNegotiation.windowOrigin);
      console.log('Backend received Origin:', auditNegotiation.backendReceivedOrigin);
      console.log('Queue Status:', auditNegotiation.queueStatus);
      console.log('====================================');

      expect(auditNegotiation.windowOrigin).toBe('app://studio');
      expect(auditNegotiation.backendReceivedOrigin).toBe('app://studio');
      expect(auditNegotiation.queueStatus).toBe(200);
    } finally {
      if (appInstance?.electronApp) {
        await appInstance.electronApp.close().catch(() => {});
        appInstance = null;
      }
      try {
        standaloneBackend.kill();
      } catch {}
      await stopBackend();
    }
  });
});
