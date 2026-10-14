import { FolderOpen, Moon, Sun, ArrowUpRight } from 'lucide-react'
import { openNativeFileDialog } from '../services/api'
import type { Theme } from '../services/appearance'
import type { TabType } from './BottomBar'
interface Props {
  activeTab: TabType; onNavigate: (tab: TabType) => void
  theme: Theme; onThemeChange: () => void
  onFileLoaded: (path: string) => void; onError: (message: string) => void
}
export function Header({ activeTab, onNavigate, theme, onThemeChange, onFileLoaded, onError }: Props) {
  const importFile = async () => {
    try { const path = await openNativeFileDialog(); if (path) onFileLoaded(path) }
    catch (error) { onError(`无法打开素材：${error instanceof Error ? error.message : String(error)}`) }
  }
  return <header className="project-header">
    <button className="wordmark" onClick={() => onNavigate('home')} aria-label="Lunarc 首页">Lunarc</button>
    <nav className="header-tabs" aria-label="工作区">
      <button aria-current={activeTab === 'dlss' ? 'page' : undefined} onClick={() => onNavigate('dlss')}>工作台</button>
      <button aria-current={activeTab === 'deliver' ? 'page' : undefined} onClick={() => onNavigate('deliver')}>导出</button>
    </nav>
    <div className="header-actions">
      <button className="icon-button" onClick={onThemeChange} aria-label={theme === 'dark' ? '切换浅色模式' : '切换深色模式'} title={theme === 'dark' ? '浅色模式' : '深色模式'}>
        {theme === 'dark' ? <Sun size={19}/> : <Moon size={19}/>}
      </button>
      <button className="button quiet" onClick={importFile}><FolderOpen size={17}/>导入</button>
      <button className="button primary" onClick={() => onNavigate('deliver')}>导出视频<ArrowUpRight size={17}/></button>
    </div>
  </header>
}
