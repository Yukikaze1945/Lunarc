"""Persist the user's MCP permission and generate local connection instructions."""
import json
import os
import tempfile
import threading
from pathlib import Path


class MCPControl:
    def __init__(self,path):
        self.path=Path(path).resolve()
        self._lock=threading.RLock()
        self.enabled=False
        self.error=None
        self.runtime_error=None
        try:
            data=json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(data,dict) or type(data.get('enabled')) is not bool:
                raise ValueError('enabled must be a boolean')
            self.enabled=data['enabled']
        except FileNotFoundError:pass
        except (OSError,ValueError) as error:
            self.error=f'MCP 设置读取失败：{error}'

    def set_enabled(self,enabled):
        if type(enabled) is not bool:raise ValueError('enabled 必须是布尔值')
        if enabled and self.runtime_error:raise RuntimeError(self.runtime_error)
        with self._lock:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            temporary=None
            try:
                with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=self.path.parent,
                        prefix='.mcp-',suffix='.tmp',delete=False) as stream:
                    temporary=stream.name
                    json.dump(dict(enabled=enabled),stream)
                    stream.flush();os.fsync(stream.fileno())
                os.replace(temporary,self.path)
                self.enabled=enabled
                self.error=None
            finally:
                if temporary and os.path.exists(temporary):os.unlink(temporary)

    def info(self,port=18923):
        url=f'http://127.0.0.1:{port}/mcp'
        config=dict(mcpServers=dict(lunarc=dict(url=url)))
        prompt=(f'Add a Streamable HTTP MCP server named "lunarc" at {url} '
            'to my user-level MCP configuration on this PC. If you cannot configure MCP servers '
            'directly, show me where to add it manually.')
        with self._lock:
            return dict(enabled=self.enabled,transport='streamable-http',url=url,config=config,prompt=prompt,
                config_path=str(self.path),error=self.error or self.runtime_error)
