# Lunarc workbench

React / TypeScript / Electron UI. The current workbench uses a compact Swiss-inspired layout, an ordered render pipeline, a comparison page and an export queue.

See the root [README](../README.md) for native components, Python setup and licensing.

```powershell
npm ci
npm run build
npm start
```

The Electron shell starts `backend_entry.py` from the project root. Development may select an environment through `PYTHON_PATH`; a packaged application uses its private runtime and checks it before launch. Run the project's `run_gui.bat` for the usual source workflow.

Packaging requires a verified `build/standalone-runtime` and separately built native components. See [packaging/README.md](../packaging/README.md). This repository does not publish a binary installer.

License: **GPL-3.0-or-later**; third-party packages retain their own notices.
