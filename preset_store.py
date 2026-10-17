"""Atomic preset storage guarded against concurrent access within one process."""
from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping


def _validate_json(value: Any, path: str = "settings") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite numbers")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or key.startswith("_"):
                raise ValueError(f"{path} contains an invalid key")
            _validate_json(item, f"{path}.{key}")
        return
    raise ValueError(f"{path} must contain JSON values")


def _name(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("preset name must be a string")
    result = value.strip()
    if not result or len(result) > 80:
        raise ValueError("preset name must contain 1–80 characters")
    return result


class PresetStore:
    def __init__(self, path: str | os.PathLike[str] | None = None):
        if path is None:
            root = os.environ.get("LUNARC_DATA_DIR")
            if not root:
                root = os.path.join(os.environ.get("APPDATA") or str(Path.home()), "DaVinci_DLSS5_Studio")
            path = Path(root) / "presets.json"
        self.path = Path(path)
        self._lock = threading.RLock()
        self._presets: dict[str, dict[str, Any]] = {}
        self._revision = 0
        self._load_error: str | None = None
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"),
                              parse_constant=lambda token: (_ for _ in ()).throw(
                                  ValueError(f"non-finite JSON number {token}")))
            if not isinstance(data, dict):
                raise ValueError("root must be an object")
            entries = data.get("presets", [])
            revision = data.get("revision", 0)
            if not isinstance(entries, list):
                raise ValueError("presets must be a list")
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
                raise ValueError("revision must be a non-negative integer")
            loaded: dict[str, dict[str, Any]] = {}
            for index, item in enumerate(entries):
                if not isinstance(item, dict):
                    raise ValueError(f"preset entry {index} must be an object")
                name = _name(item.get("name"))
                if name in loaded:
                    raise ValueError(f"duplicate preset name {name!r}")
                settings = item.get("settings")
                if not isinstance(settings, dict):
                    raise ValueError("preset settings must be an object")
                _validate_json(settings)
                loaded[name] = copy.deepcopy(settings)
            self._presets, self._revision = loaded, revision
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            # Construction remains safe for server startup, but every operation
            # fails closed so a later write cannot replace the recovery source.
            self._load_error = f"{type(exc).__name__}: {exc}"

    def _ensure_loaded(self) -> None:
        if self._load_error is not None:
            raise RuntimeError(
                f"Preset store could not load {self.path}: {self._load_error}. "
                "The file was left unchanged; recover or repair it before retrying."
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_loaded()
            return {"presets": [{"name": n, "settings": copy.deepcopy(s)} for n, s in self._presets.items()],
                    "revision": self._revision}

    def _commit(self, presets: dict[str, dict[str, Any]]) -> dict[str, Any]:
        self._ensure_loaded()
        revision = self._revision + 1
        payload = {"presets": [{"name": n, "settings": s} for n, s in presets.items()], "revision": revision}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent,
                                             prefix=f".{self.path.name}.", suffix=".tmp", delete=False) as stream:
                temp_name = stream.name
                json.dump(payload, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.path)
        except Exception:
            if temp_name:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass
            raise
        self._presets = presets
        self._revision = revision
        return self.snapshot()

    def save(self, name: str, settings: Mapping[str, Any], overwrite: bool = False) -> dict[str, Any]:
        with self._lock:
            self._ensure_loaded()
            name = _name(name)
            if not isinstance(settings, Mapping) or not isinstance(settings, dict):
                raise ValueError("preset settings must be an object")
            _validate_json(settings)
            if name in self._presets and not overwrite:
                raise ValueError("preset already exists; explicit overwrite is required")
            updated = copy.deepcopy(self._presets)
            updated[name] = copy.deepcopy(settings)
            return self._commit(updated)

    def delete(self, name: str) -> dict[str, Any]:
        with self._lock:
            self._ensure_loaded()
            name = _name(name)
            if name not in self._presets:
                raise KeyError(name)
            updated = copy.deepcopy(self._presets)
            del updated[name]
            return self._commit(updated)

    def import_missing(self, presets: list[Mapping[str, Any]]) -> dict[str, Any]:
        with self._lock:
            self._ensure_loaded()
            if not isinstance(presets, list):
                raise ValueError("presets must be a list")
            updated = copy.deepcopy(self._presets)
            for item in presets:
                if not isinstance(item, Mapping):
                    raise ValueError("each preset must be an object")
                name = _name(item.get("name"))
                settings = item.get("settings")
                if not isinstance(settings, dict):
                    raise ValueError("preset settings must be an object")
                _validate_json(settings)
                if name not in updated:
                    updated[name] = copy.deepcopy(settings)
            if updated == self._presets:
                return self.snapshot()
            return self._commit(updated)
