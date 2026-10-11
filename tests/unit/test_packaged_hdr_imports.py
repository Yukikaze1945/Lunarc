# -*- coding: utf-8 -*-
"""The packaged product can reach the HDR chain, and reaching it loads nothing.

``pipeline.export_gpu_transfer_stream`` delegates an HDR job to
``export_gpu_hdr_transfer_stream``, and both that entry and ``dlss_service``'s opt-in
read resolve their module through a literal ``__import__("gpu_hdr_transfer_export")``.
PyInstaller's module scanner follows import statements, not that call form, so a bundle
built without the name cannot resolve it at runtime - which is a failure that only
appears the first time somebody turns the flag on in a shipped exe.

Three things are checked here, all on CPU and none of them a build:

  * the spec names the chain (read out of the spec's own AST, not by substring);
  * PyInstaller's own graph, rooted at the spec's own entry script, does *not* reach it
    without that naming, and does reach it with the naming applied - the reason the
    entries are load-bearing rather than decorative;
  * the analysis-time import that hiddenimports causes opens no native library, because
    ``bundle_paths``/``ctypes`` loading stays inside the lazy loaders.

``pytest.importorskip`` guards the graph test: without PyInstaller installed there is no
analysis to inspect, and asserting about a scanner that is not present would be an
assertion about nothing.
"""
import ast
import inspect
import json
import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

SPEC = os.path.join(_ROOT, "DLSS5Tool_standalone.spec")
HDR_CHAIN = ("gpu_hdr_transfer_export", "hdr_gpu_engine", "hdr_fusion_engine")
# The modules the packaged product resolves through a literal __import__ call. The
# graph test asserts every name found here is either statically reachable from the
# spec's entry script or named in the spec, which is the contract a future lazy import
# has to keep.
LITERAL_IMPORT_SITES = ("pipeline.py", "dlss_service.py")


def _spec_tree():
    with open(SPEC, encoding="utf-8") as handle:
        return ast.parse(handle.read())


def _hiddenimports():
    """The list the spec hands to Analysis, resolved through its own assignment."""
    tree = _spec_tree()
    assigned = {}
    analysis_call = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and isinstance(node.value, (ast.List, ast.Tuple)):
            assigned[node.targets[0].id] = [
                element.value for element in node.value.elts
                if isinstance(element, ast.Constant)]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "Analysis":
            analysis_call = node
    assert analysis_call is not None, "the spec has no Analysis() call"
    for keyword in analysis_call.keywords:
        if keyword.arg != "hiddenimports":
            continue
        if isinstance(keyword.value, ast.Constant):
            return [keyword.value.value]
        if isinstance(keyword.value, (ast.List, ast.Tuple)):
            return [element.value for element in keyword.value.elts
                    if isinstance(element, ast.Constant)]
        if isinstance(keyword.value, ast.Name):
            return assigned.get(keyword.value.id, [])
    raise AssertionError("the spec's Analysis() carries no hiddenimports")


def _literal_imports(module_file):
    """Every module this file resolves through __import__("literal")."""
    with open(os.path.join(_ROOT, module_file), encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        called = getattr(function, "id", None) or getattr(function, "attr", None)
        if called != "__import__" or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            names.add(first.value)
    return names


def _graph_with(hiddenimports):
    """PyInstaller's own graph over the spec's own entry script.

    The search path is this tree alone, because the question is which of *these* modules
    the bundle reaches; a full-``sys.path`` emulation costs ~30 s and PyInstaller's own
    AST scanner raises RecursionError compiling a third-party module here, so third-
    party names are not asserted by this check and are not this repair's subject.
    """
    modulegraph = pytest.importorskip("PyInstaller.lib.modulegraph.modulegraph")
    graph = modulegraph.ModuleGraph(path=[_ROOT])
    graph.add_script(os.path.join(_ROOT, "gui.py"))
    reached = {node.identifier for node in graph.iter_graph()
               if getattr(node, "identifier", None)}
    for name in hiddenimports:
        try:
            graph.import_hook(name)
        except Exception:                       # a third-party name this path lacks
            continue
    collected = {node.identifier for node in graph.iter_graph()
                 if getattr(node, "identifier", None)}
    return reached, collected


def test_the_spec_names_the_hdr_chain_in_hiddenimports():
    listed = _hiddenimports()
    for name in HDR_CHAIN:
        assert name in listed, f"{name} is not in the spec's hiddenimports: {listed}"


def test_the_literal_import_route_is_the_one_the_spec_has_to_name():
    """The route the brief names, read out of the product's own source.

    ``gpu_transfer_export`` is imported by gui.py in a statement, so it is reached
    without being named; the HDR chain is not, which is the asymmetry the entries fix.
    And every name the lazy route resolves is either statically reachable or named: no
    literal import is allowed to be invisible to the build.
    """
    listed = _hiddenimports()
    _reached, collected = _graph_with(listed)
    for module_file in LITERAL_IMPORT_SITES:
        for name in _literal_imports(module_file):
            if not os.path.exists(os.path.join(_ROOT, name + ".py")):
                continue                    # a stdlib or third-party lazy import
            assert name in collected or name in listed, (
                f"{module_file} resolves {name!r} lazily and the bundle can reach "
                f"neither")
    assert "gpu_hdr_transfer_export" in _literal_imports("pipeline.py")
    assert "gpu_hdr_transfer_export" in _literal_imports("dlss_service.py")
    with open(os.path.join(_ROOT, "gui.py"), encoding="utf-8") as handle:
        gui = ast.parse(handle.read())
    statically = set()
    for node in ast.walk(gui):
        if isinstance(node, ast.Import):
            statically.update(alias.name for alias in node.names)
    assert "gpu_transfer_export" in statically
    assert not any(name in statically for name in HDR_CHAIN)


def test_the_analysis_graph_needs_the_entries_and_gets_the_chain_with_them():
    reached_alone, _collected = _graph_with([])
    for name in HDR_CHAIN:
        assert name not in reached_alone, (
            f"{name} is reachable without being named, so the spec entry is dead "
            f"weight - say so rather than carrying a claim about it")
    listed = _hiddenimports()
    _reached, collected = _graph_with(listed)
    # Every name the spec has to resolve that this tree owns must land in the graph.
    # A third-party entry is the build machine's business, not this check's: this
    # interpreter has no pillow, and the shipping build does.
    for name in listed:
        if not os.path.exists(os.path.join(_ROOT, name + ".py")):
            print(f"[packaging] {name} is third-party: resolved against site-packages, "
                  f"which this check deliberately does not scan")
            continue
        assert name in collected, f"{name} named in hiddenimports but not resolved"
    for name in HDR_CHAIN:
        assert name in collected, name
    # And with the chain collected, its own first-party owners come with it, which is
    # how a frozen bundle answers the call the flag routes to.
    import gpu_hdr_transfer_export as htx
    for name in ("hdr_gpu_engine", "gpu_transfer_export", "bundle_paths"):
        assert name in collected, name
    assert htx.hdr_requested({"experimental_hdr_scrgb": True}) is True


def test_the_bundle_entries_cost_a_module_import_not_a_native_load():
    """Importing the chain the way analysis does must open no DLL and no CUDA context.

    A subprocess, because the question is what an import of these modules does from
    cold; ctypes is watched, not blocked, so a third-party loader cannot fail the build
    for the test's sake and the answer is the list of native names this tree owns.
    """
    wanted = sorted(name for name in os.listdir(os.path.join(_ROOT, "native"))
                    if name.endswith(".dll"))
    script = r"""
import ctypes, json, os, sys
root = sys.argv[1]
sys.path.insert(0, root)
wanted = json.loads(sys.argv[2])
seen = []


def watch(base):
    class Watched(base):
        def __init__(self, *args, **kwargs):
            seen.append(str(args[0] if args else kwargs.get("name", "")))
            base.__init__(self, *args, **kwargs)
    Watched.__name__ = base.__name__
    return Watched


for _name in ("CDLL", "WinDLL", "PyDLL", "OleDLL"):
    setattr(ctypes, _name, watch(getattr(ctypes, _name)))
import gpu_hdr_transfer_export
import hdr_gpu_engine
import hdr_fusion_engine
hits = [entry for entry in seen
        if os.path.basename(entry).lower() in {w.lower() for w in wanted}]
print(json.dumps({"hits": hits, "hdr_library": hdr_gpu_engine._library,
                  "fusion_library": hdr_fusion_engine._library,
                  "imports": sorted(m for m in sys.modules
                                    if m in ("gpu_hdr_transfer_export",
                                             "hdr_gpu_engine",
                                             "hdr_fusion_engine"))}))
"""
    outcome = subprocess.run(
        [sys.executable, "-c", script, _ROOT, json.dumps(wanted)],
        capture_output=True, text=True, timeout=300, cwd=_ROOT)
    assert outcome.returncode == 0, outcome.stderr
    reported = json.loads(outcome.stdout.strip().splitlines()[-1])
    assert reported["hits"] == [], f"native load at CPU import: {reported['hits']}"
    assert reported["hdr_library"] is None
    assert reported["fusion_library"] is None
    assert reported["imports"] == ["gpu_hdr_transfer_export", "hdr_fusion_engine",
                                   "hdr_gpu_engine"]


def test_the_hdr_route_keeps_the_test_seams_and_the_sdr_parameter_order():
    """The repair must not disturb what callers already bind these entries by."""
    import pipeline
    sdr = inspect.signature(pipeline.export_gpu_transfer_stream)
    assert list(sdr.parameters)[:11] == [
        "video", "live", "frames", "fps", "cq", "with_audio", "out_path",
        "transcode_settings", "job_settings", "flow_provider", "lock"], \
        list(sdr.parameters)
    assert {"module", "hdr_module"} <= set(sdr.parameters)
    hdr = inspect.signature(pipeline.export_gpu_hdr_transfer_stream)
    assert {"module", "transfer_module"} <= set(hdr.parameters)
    assert list(hdr.parameters)[:11] == [
        "video", "live", "dims", "frames", "fps", "cq", "with_audio", "out_path",
        "transcode_settings", "job_settings", "flow_provider"], list(hdr.parameters)
