import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import electron from 'vite-plugin-electron'
import renderer from 'vite-plugin-electron-renderer'
import path from 'node:path'
import fs from 'node:fs'
import ts from 'typescript'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    electron([
      {
        // Main process entry file
        entry: 'electron/main.ts',
      },
      {
        entry: 'electron/preload.ts',
        onstart(options) {
          options.reload()
        },
        vite: {
          build: {
            rollupOptions: {
              output: {
                format: 'cjs',
                entryFileNames: 'preload.cjs',
              },
            },
          },
        },
      },
    ]),
    renderer(),
    {
      name: 'fix-preload-cjs',
      closeBundle() {
        const p = path.resolve(__dirname, 'dist-electron/preload.cjs')
        if (fs.existsSync(p)) {
          let code = fs.readFileSync(p, 'utf-8')
          if (code.includes('import {') || code.includes('export {};')) {
            // Compile the actual preload source rather than duplicating its API here.
            code = ts.transpileModule(
              fs.readFileSync(path.resolve(__dirname, 'electron/preload.ts'), 'utf-8'),
              { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } },
            ).outputText
            fs.writeFileSync(p, code, 'utf-8')
          }
        }
      },
    },
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
  server: {
    port: 5173,
    strictPort: true,
  },
})
