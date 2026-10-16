// localized-render-sinks
import {ui} from '../i18n/locale'
import { FolderOpen, Moon, Sun, ArrowUpRight } from 'lucide-react'
import { openNativeFileDialog } from '../services/api'
import type { Theme } from '../services/appearance'
import type { TabType } from './BottomBar'
interface Props {
  activeTab: TabType; onNavigate: (tab: TabType) => void
  theme: Theme; onThemeChange: () => void
  onFileLoaded: (path: string) => void; onError: (message: string) => void
  image?: boolean
}
export function Header({ activeTab, onNavigate, theme, onThemeChange, onFileLoaded, onError, image }: Props) {
  const importFile = async () => {
    try { const path = await openNativeFileDialog(); if (path) onFileLoaded(path) }
    catch (error) { onError(`无法打开素材：${error instanceof Error ? error.message : String(error)}`) }
  }
  return <header className="project-header">
    <button className="wordmark" onClick={() => onNavigate('home')} aria-label={ui("Lunarc 首页")}>Lunarc</button>
    <nav className="header-tabs" aria-label={ui("工作区")}>
      <button aria-current={activeTab === 'dlss' ? 'page' : undefined} onClick={() => onNavigate('dlss')}>{ui("工作台")}</button>
      <button aria-current={activeTab === 'deliver'||activeTab==='image-deliver' ? 'page' : undefined} onClick={() => onNavigate(image?'image-deliver':'deliver')}>{ui("导出")}</button>
    </nav>
    <div className="header-actions">
      <button className="icon-button" onClick={onThemeChange} aria-label={ui(theme === 'dark' ? '切换浅色模式' : '切换深色模式')} title={ui(theme === 'dark' ? '浅色模式' : '深色模式')}>
        {ui(theme === 'dark' ? <Sun size={19}/> : <Moon size={19}/>)}
      </button>
      <button className="button quiet" onClick={importFile}><FolderOpen size={17}/>{ui("导入")}</button>
      <button className="button primary" onClick={() => onNavigate(image?'image-deliver':'deliver')}>{ui(image?'导出图片':'导出视频')}<ArrowUpRight size={17}/></button>
    </div>
  </header>
}
