// tests/helpers/ui.js — shared UI interaction helpers (test-harness only).
import { expect } from '@playwright/test';

// Click 导入素材 and wait for the header badge, retrying once: the click can
// transiently miss when the backend is mid-restart behind the shared port
// (dying process still answers, then disappears). Assertions unchanged.
export async function importFixture(page, filename = 'fixture_120fps.mkv') {
  const badge = page.locator('header').locator(`text=${filename}`);
  for (let attempt = 1; attempt <= 2; attempt++) {
    await page.getByRole('button', { name: '导入素材' }).click();
    try {
      await expect(badge).toBeVisible({ timeout: 15000 });
      return;
    } catch (e) {
      if (attempt === 2) throw e;
      console.log(`[ui] import badge missing (attempt ${attempt}), retrying import click...`);
    }
  }
}

// Play until the header frame counter provably advances past `pastFrame`,
// then pause. Proves the renderer delivers frames before any timed window.
// A permanently broken renderer fails at the poll timeout instead of
// silently measuring a dead window.
export async function warmUpPlayback(page, pastFrame = 0, timeoutMs = 25000) {
  const playBtn = page.getByRole('button', { name: '播放', exact: true });
  await expect(playBtn).toBeVisible();
  const counter = page.locator('header .font-mono.text-xs span.font-semibold').first();
  await playBtn.click();
  await expect.poll(async () => {
    const t = await counter.innerText().catch(() => '0');
    return parseInt(t.trim(), 10) || 0;
  }, { timeout: timeoutMs }).toBeGreaterThan(pastFrame);
  await page.getByRole('button', { name: '暂停', exact: true }).click();
}

// Rewind to frame 0 via the transport control (icon-only button, located by
// title; no page reload: reload churns the preview socket while a native
// inference may be in flight, which the DLL cannot interrupt and stalls the
// next window).
export async function rewindToStart(page) {
  await page.getByTitle('跳至首帧').click();
  const counter = page.locator('header .font-mono.text-xs span.font-semibold').first();
  await expect.poll(async () => {
    const t = await counter.innerText().catch(() => '-1');
    return parseInt(t.trim(), 10);
  }, { timeout: 15000 }).toBe(0);
}
