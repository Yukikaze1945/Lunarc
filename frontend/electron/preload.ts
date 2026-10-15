import { contextBridge, ipcRenderer, webUtils } from 'electron'

contextBridge.exposeInMainWorld('electronAPI', {
  getLanguage: () => ipcRenderer.invoke('ui:language:get'),
  setLanguage: (locale: string) => ipcRenderer.invoke('ui:language:set', locale),
  openRuntimeDialog: () => ipcRenderer.invoke('dialog:openRuntime'),
  openRuntimeDirDialog: () => ipcRenderer.invoke('dialog:openRuntimeDirectory'),
  openFileDialog: () => ipcRenderer.invoke('dialog:openFile'),
  openDirDialog: () => ipcRenderer.invoke('dialog:openDirectory'),
  getPathForFile: (file: File) => webUtils.getPathForFile(file),
  windowOverlay: process.platform !== 'darwin',
  setWindowTheme: (theme: 'dark' | 'light') => ipcRenderer.invoke('window:setTheme', theme),
  // Per-launch backend token (P0-01). Main process sets process.env first.
  localToken: process.env.DLSS_LOCAL_TOKEN || '',
  getToken: () => ipcRenderer.invoke('auth:getToken'),
  getPreviewWindow: () => ipcRenderer.invoke('preview:window'),
  getBackendStartup: () => ipcRenderer.invoke('backend:getStartup'),
  getComponents: () => ipcRenderer.invoke('components:get'),
  installComponent: (id: string) => ipcRenderer.invoke('components:install', id),
  cancelComponent: () => ipcRenderer.invoke('components:cancel'),
  startComponentBackend: () => ipcRenderer.invoke('components:startBackend'),
  onComponents: (cb: (state: unknown) => void) => {
    const handler = (_event: Electron.IpcRendererEvent, state: unknown) => cb(state)
    ipcRenderer.on('components:state', handler)
    return () => ipcRenderer.removeListener('components:state', handler)
  },
  getAppUpdate: () => ipcRenderer.invoke('updates:get'),
  checkAppUpdate: (preview: boolean) => ipcRenderer.invoke('updates:check', preview),
  downloadAppUpdate: () => ipcRenderer.invoke('updates:download'),
  cancelAppUpdate: () => ipcRenderer.invoke('updates:cancel'),
  openAppUpdateDirectory: () => ipcRenderer.invoke('updates:openDirectory'),
  onAppUpdate: (cb: (state: unknown) => void) => {
    const handler = (_event: Electron.IpcRendererEvent, state: unknown) => cb(state)
    ipcRenderer.on('updates:state', handler)
    return () => ipcRenderer.removeListener('updates:state', handler)
  },
  onBackendExit: (cb: (info: { code: number | null; signal: string }) => void) => {
    const handler = (_event: Electron.IpcRendererEvent, info: { code: number | null; signal: string }) => cb(info)
    ipcRenderer.on('backend-exit', handler)
    return () => ipcRenderer.removeListener('backend-exit', handler)
  },
  onBackendRecovery: (cb: (state: {phase:string;message:string}) => void) => {
    const handler=(_event:Electron.IpcRendererEvent,state:{phase:string;message:string})=>cb(state)
    ipcRenderer.on('backend-recovery',handler)
    return()=>ipcRenderer.removeListener('backend-recovery',handler)
  },
  isElectron: true
})
