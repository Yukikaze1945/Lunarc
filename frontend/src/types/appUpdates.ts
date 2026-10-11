export type UpdateAsset = {name: string; size: number; url: string; sha256: string}
export type UpdateRelease = {version: string; preview: boolean; url: string; notes: string; assets: UpdateAsset[]}
export type UpdateState = {currentVersion: string; packaged: boolean; phase: 'idle'|'checking'|'available'|'current'|'downloading'|'paused'|'ready'|'error'; release: UpdateRelease|null; received: number; total: number; directory: string|null; error: string|null}
