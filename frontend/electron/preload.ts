import { contextBridge, ipcRenderer, webUtils } from 'electron'

contextBridge.exposeInMainWorld('electronAPI', {
  openFileDialog: () => ipcRenderer.invoke('dialog:openFile'),
  openDirDialog: () => ipcRenderer.invoke('dialog:openDirectory'),
  getPathForFile: (file: File) => webUtils.getPathForFile(file),
  windowOverlay: process.platform !== 'darwin',
  setWindowTheme: (theme: 'dark' | 'light') => ipcRenderer.invoke('window:setTheme', theme),
  // Per-launch backend token (P0-01). Main process sets process.env first.
  localToken: process.env.DLSS_LOCAL_TOKEN || '',
  getToken: () => ipcRenderer.invoke('auth:getToken'),
  getBackendStartup: () => ipcRenderer.invoke('backend:getStartup'),
  onBackendExit: (cb: (info: { code: number | null; signal: string }) => void) => {
    const handler = (_event: Electron.IpcRendererEvent, info: { code: number | null; signal: string }) => cb(info)
    ipcRenderer.on('backend-exit', handler)
    return () => ipcRenderer.removeListener('backend-exit', handler)
  },
  isElectron: true
})
