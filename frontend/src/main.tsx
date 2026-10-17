import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import {applyTheme, readTheme, initWindowChrome} from './services/appearance'
import {initializeLocale} from './i18n/locale'
import {initializePresets} from './services/presets'

applyTheme(readTheme())
initWindowChrome()
// The workbench opens directly, so restore display preferences before any page mounts.
try {
  document.documentElement.dataset.density = localStorage.getItem('lunarc.density') === 'compact' ? 'compact' : 'comfortable'
  document.documentElement.dataset.reducedMotion = String(localStorage.getItem('lunarc.reduced-motion') === 'true')
} catch { /* Display defaults remain usable when storage is unavailable. */ }

void initializePresets().catch(()=>{})
void initializeLocale().then(()=>createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
))
