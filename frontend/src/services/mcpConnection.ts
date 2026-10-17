import {localToken} from './api'

export interface McpConnectionState {
  enabled: boolean
  transport: 'streamable-http'
  url: string
  config: {mcpServers: {lunarc: {url: string}}}
  prompt: string
  config_path: string
  error?: string
}

export async function fetchMcpConnection(): Promise<McpConnectionState> {
  return request('/api/mcp')
}

export async function setMcpConnection(enabled: boolean): Promise<McpConnectionState> {
  return request('/api/mcp', {method: 'POST', body: JSON.stringify({enabled})})
}

async function request(path: string, init: RequestInit = {}): Promise<McpConnectionState> {
  const token = localToken()
  const response = await fetch(`http://127.0.0.1:18923${path}`, {
    ...init,
    headers: {
      ...(init.body ? {'Content-Type': 'application/json'} : {}),
      ...(token ? {'X-DLSS-Token': token} : {}),
      ...init.headers,
    },
  })
  const data = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(data.detail || data.error || `HTTP ${response.status}`)
  return data as McpConnectionState
}
