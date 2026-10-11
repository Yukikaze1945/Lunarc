# -*- coding: utf-8 -*-
"""The packaged build has to contain what the loaders go looking for.

A frozen ONEDIR bundle is a second layout that nothing in a source-tree test run
exercises. This covers the three ways it breaks:

* the spec names a file that does not exist, which is how
  ``E:\\games\\dlss5\\DLSS5Tool_build\\ffmpeg.exe`` sat in ``binaries`` after the tree
  moved drives - PyInstaller stops there, so nobody found out until a build was tried;
* the spec names no file at all for something a loader resolves: ``--selftest`` opens
  ``native/dlss_gpu_host.dll``, which imports ``native/nvof_d3d12_interop.dll`` at load
  time, and neither was collected, so the packaged window could not draw a frame;
* a loader resolves a directory the bundle did not put it in, because
  ``os.path.dirname(__file__)``, ``os.path.dirname(sys.executable)`` and ``sys._MEIPASS``
  are one path in the source tree and two in a bundle.

So the spec file is executed here against stubs standing in for PyInstaller's API, the
sources it declares are required to exist, and the names the production loaders ask
``bundle_paths`` for are required to appear among them - at the destination the loader
will search. That last check reads the loader sources rather than the manifest, which is
what lets it notice a bridge that was added to the code and not to the build.

The frozen-layout check runs in a child process, because a module works out its own paths
when it is first imported: to ask what ``dlss_gpu_engine`` resolves to inside a bundle,
something has to set ``sys.frozen`` before it is imported, and that is not a question to
put to the process holding the rest of this suite.
"""

import json
import os
import re
import subprocess
import sys

import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bundle_paths  # noqa: E402

_SPEC = os.path.join(_ROOT, "DLSS5Tool_standalone.spec")
# Everything a packaged GUI can reach, so a path ladder is only ever written once.
_PRODUCTION = ("gui.py", "pipeline.py", "dlss_errors.py", "dlss_gpu_engine.py",
               "optical_flow.py", "residual_native.py", "gpu_transfer_export.py",
               "transfer_profiler.py", "bundle_paths.py", "DLSS5Tool_standalone.spec")
# A machine's own installs, none of which a portable build may name. C:\\Windows is the
# directory the OS owns, and the two FFmpeg ones are where the Windows installers put it
# on any machine, which is why the runtime ladder may ask them.
PORTABLE_PATHS = (r"C:\Windows", r"C:\ffmpeg-8.0-full_build\bin", r"C:\ffmpeg\bin")
# module -> how it names a library that lives in native/. gpu_transfer_export names
# without the extension because it adds it, the others name the file itself.
_NATIVE_LOADERS = (("dlss_gpu_engine.py", r'native_dll\("([\w.]+)"\)'),
                   ("residual_native.py", r'native_dll\(\s*"([\w.]+)"'),
                   ("optical_flow.py", r'native_dll\("([\w.]+)"\)'),
                   ("gpu_transfer_export.py", r'_load\("([\w.]+)"\)'))


def _dll(name):
    return name if name.lower().endswith(".dll") else name + ".dll"


def _destinations(entries):
    """basename -> the directories of the bundle it is collected into."""
    out = {}
    for entry in entries:
        destination = str(entry[0]).replace("\\", "/")
        out.setdefault(os.path.basename(destination).lower(), set()).add(
            os.path.dirname(destination) or ".")
    return out


def _exec_spec(extra_collected=()):
    """Run the spec file with PyInstaller's API stubbed, and report what it built.

    ``extra_collected`` stands in for what the dependency walk adds by itself: the
    spec's filter is meant to sort those, so a test can hand it a flattened copy of a
    bridge or a driver DLL off System32 and see which of them survive into the bundle.
    """
    captured = {}

    class _Analysis:
        def __init__(self, scripts, **kwargs):
            self.scripts = list(scripts)
            self.kwargs = kwargs
            # A spec hands in (source, destination directory) and gets back what lands
            # in the bundle: (destination path inside it, source, kind). The spec's own
            # filter reads the second form, so the stub has to make it.
            self.binaries = [("/".join(([entry[1]] if entry[1] not in (".", "") else [])
                                        + [os.path.basename(entry[0])])
                             .replace("\\", "/"), entry[0], "EXTENSION")
                             for entry in kwargs.get("binaries", ())]
            self.binaries += [tuple(entry) for entry in extra_collected]
            self.datas = list(kwargs.get("datas", ()))
            self.pure = ["<pure>"]

    namespace = {
        "__name__": "dlss5tool_standalone_spec", "SPECPATH": _ROOT,
        "DISTPATH": os.path.join(_ROOT, "dist"), "workpath": os.path.join(_ROOT, "build"),
        "Analysis": _Analysis, "PYZ": lambda pure: ("pyz", pure),
        "EXE": lambda *args, **kwargs: captured.setdefault("exe", (args, kwargs)) or "exe",
        "COLLECT": lambda *args, **kwargs: captured.setdefault("collect", args) or args}
    with open(_SPEC, encoding="utf-8") as handle:
        source = handle.read()
    exec(compile(source, _SPEC, "exec"), namespace)
    return {"declared": list(namespace["BINARIES"]),
            "kept": [tuple(entry) for entry in namespace["a"].binaries],
            "entry": namespace["a"].scripts,
            "excludes": namespace["a"].kwargs["excludes"],
            "pathex": namespace["a"].kwargs["pathex"],
            "captured": captured}


def test_spec_declares_only_files_that_exist():
    """The first thing a spec owes the next person: the files in it are there.

    This is the whole of the E:\\games failure - an entry that was a path on the machine
    that wrote it, with nothing between then and a build attempt to say whether it is
    real.
    """
    declared = _exec_spec()["declared"]
    assert declared, "the spec collects nothing"
    assert [source for source, destination in declared
            if not os.path.isfile(source)] == []


def test_spec_keeps_every_file_it_declared():
    """The filter that drops what the machine owns must not eat the bundle's own files."""
    spec = _exec_spec()
    kept = {os.path.basename(str(entry[0]).replace("\\", "/")).lower()
            for entry in spec["kept"]}
    assert [os.path.basename(source) for source, destination in spec["declared"]
            if os.path.basename(source).lower() not in kept] == []


def test_every_native_library_a_loader_names_is_collected_beside_it():
    """Read the loader sources, not the manifest, and require the build to match.

    gpu_transfer_export opens five bridges by name and dlss_gpu_engine one host; a sixth
    added to the code with no line in the spec would otherwise show up as a bundle that
    cannot export, on whichever machine tried to.
    """
    named = set()
    for module, pattern in _NATIVE_LOADERS:
        with open(os.path.join(_ROOT, module), encoding="utf-8") as handle:
            found = {_dll(name) for name in re.findall(pattern, handle.read())}
        assert found, f"{module} names no native library, so the pattern is stale"
        named |= found
    destinations = _destinations(_exec_spec()["kept"])
    absent = {name for name in named if destinations.get(name) != {"native"}}
    assert absent == set(), f"{sorted(absent)} are loaded from native/ but not there"


def test_the_host_dependency_that_loads_with_it_shares_its_directory():
    """nvof_d3d12_interop.dll is imported by dlss_gpu_host.dll at load time.

    A file no loader names by hand, so the check above cannot see it: the OS loader
    resolves it, and finds it in the directory of the DLL that asked. That makes
    "collected somewhere in the bundle" the wrong property, and "collected beside it"
    the one worth holding.
    """
    with open(os.path.join(_ROOT, "native", "dlss_gpu_host.dll"), "rb") as handle:
        assert b"nvof_d3d12_interop.dll" in handle.read(), "the host imports it no longer"
    destinations = _destinations(_exec_spec()["kept"])
    assert destinations.get("nvof_d3d12_interop.dll") == {"native"}
    assert destinations.get("dlss_gpu_host.dll") == {"native"}


def test_both_tools_the_window_shells_out_to_are_collected():
    """ffmpeg.exe and ffprobe.exe: the GUI opens a video with one and counts frames with
    the other, and the spec shipped half of that pair."""
    with open(os.path.join(_ROOT, "pipeline.py"), encoding="utf-8") as handle:
        tools = set(re.findall(r"_bundle_tool\([\"']([\w.]+)[\"']\)", handle.read()))
    assert tools == {"ffmpeg.exe", "ffprobe.exe"}, \
        "pipeline no longer asks both tools by name, so this check needs its names"
    destinations = _destinations(_exec_spec()["kept"])
    assert {tool: destinations.get(tool) for tool in tools} == \
        {"ffmpeg.exe": {"."}, "ffprobe.exe": {"."}}


def test_a_flattened_copy_of_a_bridge_and_the_machines_own_dlls_are_not_shipped():
    """The dependency walk adds what it resolves; the spec has to refuse three kinds.

    Each entry is the shape PyInstaller actually produces: a bare name at the contents
    root, a source path under the Windows directory, a driver DLL found on the build
    machine. The bridges and the launcher's own runtime are what the bundle owns.
    """
    windows = os.environ.get("SystemRoot", r"C:\Windows")
    native = os.path.join(_ROOT, "native")
    spec = _exec_spec(extra_collected=[
        ("nvof_d3d12_interop.dll", os.path.join(native, "nvof_d3d12_interop.dll"), "EXT"),
        ("nvcuda.dll", os.path.join(windows, "System32", "nvcuda.dll"), "EXT"),
        ("d3dcompiler_47.dll", os.path.join(windows, "System32", "d3dcompiler_47.dll"), "EXT"),
        ("nvofapi64.dll", os.path.join(r"D:\NVIDIA\DisplayDriver", "nvofapi64.dll"), "EXT"),
        ("nvngx_dlssnr.dll", os.path.join(windows, "System32", "nvngx_dlssnr.dll"), "EXT"),
        ("python311.dll", os.path.join(os.path.dirname(sys.executable), "python311.dll"), "EXT"),
    ])
    kept = [str(entry[0]).replace("\\", "/") for entry in spec["kept"]]
    lower = [os.path.basename(name).lower() for name in kept]
    assert "native/dlss_gpu_host.dll" in kept
    assert "python311.dll" in kept
    assert lower.count("nvof_d3d12_interop.dll") == 1, \
        [name for name in kept if "interop" in name.lower()]
    assert lower.count("nvngx_dlssnr.dll") == 1, "the bundle's copy won over System32's"
    for refused in ("nvcuda.dll", "d3dcompiler_47.dll", "nvofapi64.dll"):
        assert refused not in lower, f"{refused} was shipped with the rest"


def test_ffmpeg_dll_directories_follow_the_machine_not_a_developers_path(tmp_path,
                                                                         monkeypatch):
    """Where the bridges' FFmpeg import comes from, now that no install is hardcoded.

    The old answer had a second machine's application directory in the middle of the
    list, so a bundle built elsewhere shipped nothing and a source tree elsewhere failed
    on a path that meant nothing to it. The three real answers are each checked here -
    the knob the bridges were linked with, a bundle's own contents directory, and
    whatever PATH offers - and a directory that merely exists is not one of them,
    because load_bridges registers the first of this list and nothing else.
    """
    import gpu_transfer_export as gtx

    sdk_bin = tmp_path / "sdk" / "bin"
    sdk_bin.mkdir(parents=True)
    (sdk_bin / "avcodec-61.dll").write_bytes(b"x")
    shim = tmp_path / "shim-directory"
    shim.mkdir()
    (shim / "ffmpeg.exe").write_bytes(b"x")
    on_path = tmp_path / "install-on-path"
    on_path.mkdir()
    (on_path / "avcodec-99.dll").write_bytes(b"x")

    monkeypatch.setenv("FFMPEG_SDK_DIR", str(tmp_path / "sdk"))
    monkeypatch.setenv("PATH", str(on_path))
    monkeypatch.setattr(gtx.pipeline, "_bundle_ffmpeg",
                        lambda: str(shim / "ffmpeg.exe"))
    assert gtx.ffmpeg_dll_directories()[0] == str(sdk_bin)

    monkeypatch.delenv("FFMPEG_SDK_DIR")
    # A directory holding the tool but none of its shared libraries is not an answer.
    assert gtx.ffmpeg_dll_directories() == [str(on_path)]

    monkeypatch.setenv("PATH", str(shim))
    assert gtx.ffmpeg_dll_directories() == []

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert gtx.ffmpeg_dll_directories() == []
    (tmp_path / "avcodec-61.dll").write_bytes(b"x")
    assert gtx.ffmpeg_dll_directories() == [str(tmp_path)]


def test_no_other_machines_install_path_is_baked_into_a_production_module():
    """A drive-letter literal in a path ladder is one machine, not a rule.

    Only the Windows directory and the two conventional FFmpeg install folders are
    portable answers; an application's own directory on whoever developed the bridge is
    not, and neither is a game library.
    """
    offenders = []
    portable = tuple(path.lower() for path in PORTABLE_PATHS)
    for module in _PRODUCTION:
        with open(os.path.join(_ROOT, module), encoding="utf-8") as handle:
            source = handle.read()
        # A string literal that opens with a drive letter and a separator: not a
        # relative rule anyone else's machine can satisfy, but one folder on one disk.
        for _, path in re.findall(r"[\"'](r?)([A-Za-z]:[\\/][^\"'\n]*)[\"']", source):
            if not path.lower().startswith(portable):
                offenders.append(f"{module}: {path}")
    assert offenders == []


def test_a_missing_binary_names_the_layout_that_is_short(tmp_path, monkeypatch):
    """The message has to say what to go and do, and that differs by layout.

    A source tree needs the bridge rebuilt; a bundle needs the bundle rebuilt. One
    sentence cannot serve both, and the bare OSError ctypes raises for a missing DLL
    serves neither.
    """
    root = tmp_path / "tree"
    (root / "native").mkdir(parents=True)
    for name in bundle_paths.CONTENTS_BINARIES:
        (root / name).write_bytes(b"x")
    (root / "native" / bundle_paths.NATIVE_BINARIES[0]).write_bytes(b"x")
    tools = tmp_path / "ffmpeg"
    tools.mkdir()
    for name in bundle_paths.TOOL_BINARIES:
        (tools / name).write_bytes(b"x")

    with pytest.raises(FileNotFoundError) as raised:
        bundle_paths.resolve_sources(str(root), ffmpeg_dir=str(tools))
    message = str(raised.value)
    assert _dll(bundle_paths.NATIVE_BINARIES[1]) in message
    assert str(root / "native") in message
    assert "native/" in message and "build_*.bat" in message

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    frozen = str(bundle_paths.missing_binary(
        bundle_paths.native_dll("dlss_gpu_host.dll"), "run the GPU-resident DLSS host"))
    assert "not in this bundle" in frozen and "Rebuild it" in frozen
    assert str(bundle_paths.native_dir()) in frozen


def test_the_source_tree_still_resolves_every_named_binary():
    """Nothing about a bundle may break the tree the bundle is built from."""
    for name in bundle_paths.NATIVE_BINARIES:
        assert os.path.isfile(bundle_paths.native_dll(name)), name
    for name in bundle_paths.CONTENTS_BINARIES:
        assert os.path.isfile(bundle_paths.collected(name)), name


def test_the_retired_dlss_host_is_neither_shipped_nor_reachable():
    """A bundle that still carries dlssnr_host.dll is a bundle with two hosts in it.

    The old binary shipped because dlss_engine.py resolved it at import time and the
    manifest collected whatever a loader named. The service now refuses that backend by
    name, so a build that still held the file would carry a host nothing can select -
    and one reachable by any script that imports the wrapper by hand. Three separate
    facts have to stay false for that: the manifest naming it, the spec keeping it, and
    a production module importing the wrapper that loaded it.
    """
    assert bundle_paths.CONTENTS_BINARIES == ("nvngx_dlssnr.dll",)

    kept = {os.path.basename(str(entry[0]).replace("\\", "/")).lower()
            for entry in _exec_spec()["kept"]}
    assert "dlssnr_host.dll" not in kept
    assert "nvngx_dlssnr.dll" in kept

    for module in _PRODUCTION + ("dlss_service.py", "server.py"):
        with open(os.path.join(_ROOT, module), encoding="utf-8") as handle:
            source = handle.read()
        assert re.search(r"^\s*(import|from)\s+dlss_engine\b", source, re.M) is None, (
            f"{module} imports the retired wrapper again")
    for module in ("bundle_paths.py", "dlss_gpu_engine.py", "dlss_service.py",
                   "gui.py", "server.py", "pipeline.py"):
        with open(os.path.join(_ROOT, module), encoding="utf-8") as handle:
            source = handle.read()
        # Prose may name the retired host to say what it was; nothing may build a path
        # to it. That is the difference between documentation and a second loader.
        assert re.search(r"(collected|native_dll|os\.path\.join|Path)\s*\([^)\n]*dlssnr_host",
                         source) is None, (
                f"{module} resolves a path to the retired host again")


_FROZEN_CHILD = """
import json, os, sys
sys.frozen = True
sys._MEIPASS = %(meipass)r
sys.executable = os.path.join(%(dist)r, "DLSS5Tool.exe")
import bundle_paths, dlss_gpu_engine, optical_flow, pipeline
import residual_native, gpu_transfer_export as gtx
optical_flow._load_native_bridge()
print("BUNDLE_PATHS_JSON:" + json.dumps({
    "contents": bundle_paths.contents_dir(),
    "app": bundle_paths.app_dir(),
    "host_library": dlss_gpu_engine._LIBRARY,
    "native_dir": gtx.NATIVE_DIR,
    "residual": residual_native._library_path(),
    "snippet": dlss_gpu_engine.DLSSNR_DLL,
    "flow_searched": optical_flow._NATIVE_BRIDGE_LOAD_ERROR or "",
    "ffmpeg": pipeline._bundle_ffmpeg() or "",
}))
"""


def test_a_frozen_process_resolves_every_loader_into_the_bundle(tmp_path):
    """What each loader resolves to once the process says it was frozen.

    ``_internal`` is where a spec destination of ``'.'`` puts a file, and the launcher's
    own directory is where a user puts what they were told to supply. A loader answering
    with the second while the bundle held the first is the bug - and in the source tree
    those are one folder, so no other test can see it.
    """
    dist = tmp_path / "DLSS5Tool"
    meipass = dist / "_internal"
    (meipass / "native").mkdir(parents=True)
    (meipass / "ffmpeg.exe").write_bytes(b"x")
    child = subprocess.run(
        [sys.executable, "-c",
         _FROZEN_CHILD % {"dist": str(dist), "meipass": str(meipass)}],
        cwd=_ROOT, capture_output=True, text=True, timeout=600)
    assert child.returncode == 0, child.stderr[-2000:]
    marker = child.stdout.index("BUNDLE_PATHS_JSON:") + len("BUNDLE_PATHS_JSON:")
    resolved = json.loads(child.stdout[marker:])

    contents, app = os.path.normcase(meipass), os.path.normcase(dist)
    assert os.path.normcase(resolved["contents"]) == contents
    assert os.path.normcase(resolved["app"]) == app
    for key in ("host_library", "native_dir", "residual"):
        path = os.path.normcase(resolved[key])
        assert path.startswith(os.path.normcase(os.path.join(str(meipass), "native"))), \
            (key, resolved[key])
    # The NVIDIA snippet the GPU host is handed by path at session open.
    assert os.path.normcase(os.path.dirname(resolved["snippet"])) == contents, \
        "snippet"
    # The window runs the tool it was shipped with, not the build machine's.
    assert os.path.normcase(resolved["ffmpeg"]) == os.path.normcase(
        os.path.join(str(meipass), "ffmpeg.exe"))
    # When the bridge is not there, optical_flow names the directories it searched; a
    # frozen tree has to have looked inside the bundle.
    assert "nvof_bridge.dll" in resolved["flow_searched"]
    assert contents in os.path.normcase(resolved["flow_searched"])
