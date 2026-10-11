export type Theme = 'dark' | 'light'
const THEME_KEY = 'lunarc.theme.v1'
export function initWindowChrome() {
  if (!window.electronAPI?.windowOverlay) return
  const overlay = (navigator as Navigator & {
    windowControlsOverlay?: EventTarget & { visible: boolean; getTitlebarAreaRect: () => DOMRect }
  }).windowControlsOverlay
  const update = () => {
    const area = overlay?.getTitlebarAreaRect()
    const inset = overlay?.visible && area ? Math.max(0, innerWidth - area.x - area.width) : 138
    document.documentElement.style.setProperty('--window-controls-inset', `${inset}px`)
  }
  overlay?.addEventListener('geometrychange', update)
  window.addEventListener('resize', update)
  update()
}
export function readTheme(): Theme {
  try { return localStorage.getItem(THEME_KEY) === 'light' ? 'light' : 'dark' }
  catch { return 'dark' }
}
export function applyTheme(theme: Theme) {
  document.documentElement.dataset.theme = theme
  document.documentElement.dataset.windowOverlay = String(window.electronAPI?.windowOverlay === true)
  void window.electronAPI?.setWindowTheme?.(theme).catch(() => { /* Browser theme remains usable. */ })
  try { localStorage.setItem(THEME_KEY, theme) } catch { /* Theme works without storage. */ }
}
