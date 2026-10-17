import json

import pytest

from preset_store import PresetStore


def test_save_snapshot_detached_and_explicit_overwrite(tmp_path):
    store = PresetStore(tmp_path / "presets.json")
    saved = store.save(" Film ", {"quality": 2, "nested": {"values": [1, 2]}})
    assert saved["revision"] == 1
    saved["presets"][0]["settings"]["nested"]["values"].append(3)
    assert store.snapshot()["presets"][0]["settings"]["nested"]["values"] == [1, 2]
    with pytest.raises(ValueError, match="overwrite"):
        store.save("Film", {"quality": 3})
    assert store.save("Film", {"quality": 3}, overwrite=True)["revision"] == 2


def test_import_only_adds_missing_and_delete(tmp_path):
    store = PresetStore(tmp_path / "presets.json")
    store.save("Existing", {"v": 1})
    state = store.import_missing([{"name": "Existing", "settings": {"v": 9}},
                                  {"name": "New", "settings": {"v": 2}}])
    assert [(p["name"], p["settings"]) for p in state["presets"]] == [
        ("Existing", {"v": 1}), ("New", {"v": 2})]
    assert store.delete("New")["revision"] == 3
    with pytest.raises(KeyError):
        store.delete("New")


@pytest.mark.parametrize("name", ["", "   ", "x" * 81])
def test_rejects_invalid_names(tmp_path, name):
    with pytest.raises(ValueError):
        PresetStore(tmp_path / "presets.json").save(name, {})


@pytest.mark.parametrize("settings", [[], {"_secret": 1}, {"nested": {"_secret": 2}},
                                       {"bad": float("nan")}, {"bad": float("inf")},
                                       {"bad": object()}])
def test_rejects_malformed_settings(tmp_path, settings):
    with pytest.raises(ValueError):
        PresetStore(tmp_path / "presets.json").save("valid", settings)


def test_failed_atomic_write_keeps_memory_and_disk_state(tmp_path, monkeypatch):
    path = tmp_path / "presets.json"
    store = PresetStore(path)
    store.save("A", {"v": 1})
    original = path.read_text(encoding="utf-8")
    def fail_replace(*_args):
        raise OSError("disk unavailable")
    monkeypatch.setattr("preset_store.os.replace", fail_replace)
    with pytest.raises(OSError):
        store.save("B", {"v": 2})
    assert store.snapshot()["presets"] == [{"name": "A", "settings": {"v": 1}}]
    assert path.read_text(encoding="utf-8") == original


def test_reload_from_atomic_json(tmp_path):
    path = tmp_path / "presets.json"
    first = PresetStore(path)
    expected = first.save("A", {"nested": [True, None, 4]})
    assert PresetStore(path).snapshot() == json.loads(path.read_text(encoding="utf-8")) == expected


@pytest.mark.parametrize("content", [
    "[]", "null", '{"presets":null}', '{"presets":[null]}',
    '{"presets":[[]]}', '{"presets":[{"name":"A","settings":{}}],"revision":true}',
    '{"presets":[{"name":"A","settings":{}},{"name":"A","settings":{}}]}',
    '{"presets":[{"name":"A","settings":{"bad":NaN}}]}',
])
def test_malformed_persisted_store_fails_closed_and_preserves_file(tmp_path, content):
    path = tmp_path / "presets.json"
    original = content.encode("utf-8")
    path.write_bytes(original)
    store = PresetStore(path)  # Loading errors must not prevent server startup.

    with pytest.raises(RuntimeError, match="recover or repair"):
        store.snapshot()
    with pytest.raises(RuntimeError, match="could not load"):
        store.save("New", {"v": 1})
    with pytest.raises(RuntimeError, match="could not load"):
        store.import_missing([{"name": "New", "settings": {"v": 1}}])
    assert path.read_bytes() == original
