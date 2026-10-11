import { defineConfig } from '@playwright/test';

// P2-04: every run writes an unambiguous timestamped JSON report next to the
// rolling tests/test-results.json, so doc claims always map to one artifact.
function stamp(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

const RUN_TAG = process.env.PLAYWRIGHT_RUN_TAG || stamp();

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 180000, // 3 minutes per test for export tests
  expect: {
    timeout: 15000,
  },
  fullyParallel: false,
  workers: 1, // Single worker to prevent port 18923 and GPU D3D12 device collisions
  reporter: [
    ['list'],
    ['json', { outputFile: 'tests/test-results.json' }],
    ['json', { outputFile: `test-results/e2e-${RUN_TAG}.json` }],
  ],
  use: {
    trace: 'on-first-retry',
  },
});
