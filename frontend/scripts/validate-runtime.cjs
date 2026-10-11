// No partial runtime may be labelled a standalone release.
const fs = require('node:fs')
const path = require('node:path')
const crypto = require('node:crypto')
module.exports = async function () {
  const root = path.resolve(__dirname, '../..')
  const runtime = path.join(root, 'build/standalone-runtime')
  if (fs.existsSync(path.join(runtime, 'build-in-progress.json'))) throw new Error('独立运行环境尚未完成构建')
  const manifestPath = path.join(runtime, 'manifest.json')
  if (!fs.existsSync(manifestPath)) throw new Error('先运行 packaging/build_runtime.py 构建独立运行环境，再打包')
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'))
  for (const required of ['python/python.exe', 'tools/ffmpeg.exe', 'tools/ffprobe.exe', 'torch_home/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth']) {
    if (!manifest.files?.[required]) throw new Error(`运行环境清单缺少 ${required}`)
  }
  for (const [name, entry] of Object.entries(manifest.files)) {
    const target = path.resolve(runtime, name)
    if (!target.startsWith(runtime + path.sep)) throw new Error(`清单路径非法：${name}`)
    if (!fs.existsSync(target) || fs.statSync(target).size !== entry.bytes) throw new Error(`运行环境文件缺失或大小错误：${name}`)
    // Stream hashes so even the CUDA runtime's largest DLL needs bounded memory.
    const hash = crypto.createHash('sha256')
    for await (const chunk of fs.createReadStream(target)) hash.update(chunk)
    if (hash.digest('hex') !== entry.sha256) throw new Error(`运行环境校验失败：${name}`)
  }
  console.log('Standalone runtime integrity verified')
}
