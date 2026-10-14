import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import {applyTheme, readTheme, initWindowChrome} from './services/appearance'

applyTheme(readTheme())
initWindowChrome()
// The workbench opens directly, so restore display preferences before any page mounts.
try {
  document.documentElement.dataset.density = localStorage.getItem('lunarc.density') === 'compact' ? 'compact' : 'comfortable'
  document.documentElement.dataset.reducedMotion = String(localStorage.getItem('lunarc.reduced-motion') === 'true')
} catch { /* Display defaults remain usable when storage is unavailable. */ }

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
