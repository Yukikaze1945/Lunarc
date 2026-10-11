import fs from 'node:fs'
import path from 'node:path'
import {execFile} from 'node:child_process'
import crypto from 'node:crypto'

export type BackendStartup = {state: 'checking'|'starting'|'ready'|'error'; message: string; logPath?: string; checks?: unknown[]}

export function runtimeDirectory(root: string, packaged: boolean): string {
  const installed = path.join(root, 'runtime')
  const staged = path.join(root, 'build/standalone-runtime')
  return !packaged && !fs.existsSync(path.join(installed, 'manifest.json')) && fs.existsSync(path.join(staged, 'manifest.json')) ? staged : installed
}

export function runtimeFingerprint(root: string, runtime: string): string | null {
  const manifest = path.join(runtime, 'manifest.json')
  if (!fs.existsSync(manifest)) return null
  const hash = crypto.createHash('sha256').update(fs.readFileSync(manifest))
  for (const name of fs.readdirSync(root).filter(n => n.endsWith('.py')).sort()) {
    hash.update(name + '\0'); hash.update(fs.readFileSync(path.join(root, name)))
  }
  return hash.digest('hex')
}

export function runtimeCandidates(root: string, packaged: boolean, env: NodeJS.ProcessEnv): string[] {
  const bundled = path.join(runtimeDirectory(root, packaged), 'python', 'python.exe')
  // A released package always uses its own runtime. It must not accidentally
  // start another application's Python due to environment variables or PATH.
  if (packaged || fs.existsSync(bundled)) return [bundled]
  if (env.PYTHON_PATH) return [env.PYTHON_PATH]
  let saved = ''
  try {saved = fs.readFileSync(path.join(root, 'python_path.txt'), 'utf8').trim()} catch {}
  return [...new Set([saved, path.join(root, '.venv/Scripts/python.exe'), path.join(root, 'venv/Scripts/python.exe'), 'python', 'py'].filter(Boolean))]
}

export function runtimeEnvironment(root: string, userData: string, source: NodeJS.ProcessEnv, standalone: boolean): NodeJS.ProcessEnv {
  const env = {...source}
  const runtime = source.LUNARC_RUNTIME_DIR || path.join(root, 'runtime')
  if (standalone) {
    for (const key of ['PYTHONPATH', 'PYTHONHOME', 'PYTHON_PATH', 'VIRTUAL_ENV', 'CONDA_PREFIX', 'FFMPEG_SDK_DIR', 'DLSS5_FFMPEG_DIR']) delete env[key]
    env.PATH = [path.join(runtime, 'python'), path.join(runtime, 'tools'),
      path.join(source.SystemRoot || 'C:\\Windows', 'System32'), source.SystemRoot || 'C:\\Windows'].join(path.delimiter)
    env.LUNARC_STANDALONE = '1'
  }
  env.PYTHONNOUSERSITE = '1'
  env.PYTHONUTF8 = '1'
  env.LUNARC_DATA_DIR = userData
  // Never write model caches into Program Files; packaged weights are read from
  // the private runtime while explicit user-supplied models remain supported.
  env.TORCH_HOME = standalone ? path.join(runtime, 'torch_home') : (source.TORCH_HOME || path.join(root, 'torch_home'))
  return env
}

export async function probeRuntime(python: string, root: string, env: NodeJS.ProcessEnv): Promise<{ready: boolean; checks: unknown[]}> {
  return new Promise((resolve, reject) => {
    execFile(python, [path.join(root, 'runtime_environment.py'), '--bootstrap'],
      {cwd: root, env, timeout: 30000, windowsHide: true, maxBuffer: 1024 * 1024, encoding: 'utf8'},
      (error, stdout, stderr) => {
        const line = stdout.split(/\r?\n/).reverse().find(l => l.startsWith('LUNARC_ENV='))
        if (line) {
          try {resolve(JSON.parse(line.slice('LUNARC_ENV='.length))); return} catch {}
        }
        reject(new Error(`${python}: ${stderr.trim() || error?.message || '运行环境检查没有返回结果'}`))
      })
  })
}
