// tests/helpers/paths.js — portable repo/python resolution (P2-03).
// Derive everything from this file's location; never hardcode drive letters.
// Override with env: PYTHON_PATH for the interpreter.
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const ROOT_DIR = path.resolve(__dirname, '../..');
export const FIXTURES_DIR = path.join(ROOT_DIR, 'tests', 'fixtures');

export function pythonExe() {
  if (process.env.PYTHON_PATH) return process.env.PYTHON_PATH;
  if (process.platform === 'win32') {
    const local = path.join(ROOT_DIR, '.venv/Scripts/python.exe');
    return fs.existsSync(local) ? local : 'python';
  }
  return 'python3';
}

export function fixturePath(name) {
  return path.join(FIXTURES_DIR, name);
}
