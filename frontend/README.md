# Lunarc workbench

React / TypeScript / Electron UI. The current workbench uses a compact Swiss-inspired layout, an ordered render pipeline, a comparison page and an export queue.

See the root [README](../README.md) for native components, Python setup and licensing.

```powershell
npm ci
npm run build
npm start
```

The Electron shell starts `backend_entry.py` from the project root. Development may select an environment through `PYTHON_PATH`; a packaged application uses its private runtime and checks it before launch. Run the project's `run_gui.bat` for the usual source workflow.

Packaging requires verified `build/small-resources` and separately built native components. The small installer downloads Python and video tools at first launch; RAFT is optional. See [packaging/README.md](../packaging/README.md).

License: **GPL-3.0-or-later**; third-party packages retain their own notices.
