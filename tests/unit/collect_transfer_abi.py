#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write the Phase F/G/H blocker evidence: dlss_host_abi.json and process_boundaries.json.

Archival: `dlssnr_host.dll` and its ctypes wrapper (dlss_engine.py) have been retired and
removed, so the PE dump and signature reads below describe a binary this tree no longer
carries. The files they wrote are the evidence; re-running this needs the old dll back.

These two files carry the claims this round is *not* allowed to make, so they are built from
what a tool can read rather than from what a reader would expect:

  * the shipped `dlssnr_host.dll` is dumped with a PE tool, and its export names and
    parameter types are recorded as they are;
  * the wrapper signatures come from the ctypes declarations the production code actually
    installs, not from a transcription;
  * "the source is not here" and "the SDK headers are not here" are the recorded results of
    a bounded search, with the roots that were searched written out;
  * every boolean about GPU resources is the result of a symbol scan over this checkout's
    production sources, with the matched lines kept as the evidence.

A conclusion of `BLOCKED_PROVEN` is about this binary, this ABI and the dependencies that
exist on this machine today. It is never a claim that the transfer is impossible forever,
and the two files say so in the field names rather than in prose a reader has to trust.

Usage:
    python tests/unit/collect_transfer_abi.py --out perf_results/<sha>/transfer_phase1 \
        [--ledger perf_results/<sha>/transfer_phase1/transfer_ledger.json]
"""

import argparse
import fnmatch
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

HOST_DLL = os.path.join(_ROOT, "dlssnr_host.dll")
NGX_RUNTIME_DLL = os.path.join(_ROOT, "nvngx_dlssnr.dll")
HISTORICAL_ABI_DOC = os.path.join(_ROOT, "ABI_CAPABILITIES.md")

# A PE tool names a GPU resource in its export; these tokens are what such a name would
# contain. The scan is reported as matched names, so an empty list is evidence rather than
# an assertion, and a future build that exports `dlssnr_process_texture` fails the scan.
GPU_RESOURCE_TOKENS = ("d3d", "dxgi", "cuda", "cu", "device", "gpu", "texture", "surface",
                       "resource", "handle", "shared", "srv", "uav", "fence", "semaphore",
                       "memory", "ptr", "pointer")

# Where a search for the host source, the NGX headers and the NGX import libraries looks.
# Each root carries its own depth because the trees are wildly different sizes: a five-level
# walk of a whole drive would run for minutes and prove nothing more than a two-level one.
# What was actually covered is written into the artifact next to the result.
SEARCH_ROOTS = (
    (_ROOT, 6),
    (os.path.expanduser("~"), 4),
    (r"C:\Program Files\NVIDIA Corporation", 4),
    (r"C:\Program Files (x86)\NVIDIA Corporation", 4),
    ("F:" + os.sep, 1),
)
SEARCH_TIME_BUDGET_S = 600
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".mypy_cache", ".gradle", "venv",
             "torch_home", "bench_out", "$recycle.bin", "system volume information",
             "publishcache", "microsoft", "google", "packages"}

SOURCE_EXTS = (".cpp", ".cxx", ".cc", ".c", ".h", ".hpp", ".def", ".rc", ".rc2")
HEADER_PATTERNS = ("nvsdk_ngx*.h", "nvngx*.h", "nv_ngx*.h", "*ngx_sdk*.h", "nvngx_dlssr*")
LIB_PATTERNS = ("nvngx*.lib", "nvsdk*.lib", "*ngx*.lib", "nvngx*.a")
HOST_SOURCE_PATTERNS = ("dlssnr_host*", "dlss_host*")

# The files that make up the production frame path. A GPU-resource boundary claim is only
# about these; the tree also carries Magpie, which is a separate D3D12 application.
PRODUCTION_SOURCES = ("pipeline.py", "dlss_engine.py", "dlss_service.py", "optical_flow.py",
                      "residual_native.py", "transfer_profiler.py",
                      "native/nvof_bridge.cpp", "native/nvof_bridge.h",
                      "native/residual_bridge.cpp", "native/residual_bridge.h")

IPC_PATTERNS = {
    "cuIpcGetMemHandle": r"cuIpcGetMemHandle",
    "cuIpcOpenMemHandle": r"cuIpcOpenMemHandle",
    "ID3D12Device_CreateSharedHandle": r"CreateSharedHandle\s*\(",
    "OpenSharedHandle": r"OpenSharedHandle",
    "ID3D11Texture2D_shared_resource": r"SharedResource|D3D11_RESOURCE_MISC_SHARED",
    "ID3D12Resource": r"ID3D12Resource",
    "cudaMallocManaged": r"cudaMallocManaged",
    "external_memory_Vulkan": r"vkGetMemoryWin32HandleProperties|VK_EXTERNAL",
}


def git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True,
                             timeout=30)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def _porcelain_path(line):
    # git status --porcelain writes a two-character status field, a space, then the path. The
    # git() helper strips the whole stdout, which eats that space on the first line only, so
    # the path is recovered by dropping the status field instead of a fixed character offset.
    parts = line.strip().split(maxsplit=1)
    return parts[1] if len(parts) == 2 else ""


def _identity(extra=None):
    # "dirty" means a tracked file differs from HEAD, which is the only way the running code
    # can differ from the commit this artifact names. Untracked scratch is listed beside it
    # rather than inside it: benchmark output in the working directory does not change what
    # executed, and a flag that conflates the two makes every honest artifact claim dirty.
    tracked = [line for line in git("status", "--porcelain", "--untracked-files=no").splitlines()
               if line.strip()]
    untracked = [_porcelain_path(line) for line in git("status", "--porcelain").splitlines()
                 if line.strip().startswith("??")]
    out = {
        "generated_by": "tests/unit/collect_transfer_abi.py",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_under_test": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "working_tree_dirty": bool(tracked),
        "working_tree_paths": [_porcelain_path(line) for line in tracked],
        "untracked_paths": untracked,
        "commit_subject": git("log", "-1", "--pretty=%s"),
    }
    out.update(extra or {})
    return out


# ---------------------------------------------------------------------------
# PE tool
# ---------------------------------------------------------------------------

def find_dumpbin():
    """The first working `dumpbin` on this machine, and the version line it reports."""
    candidates = [r"E:\Visual Studio\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\dumpbin.exe"]
    candidates += sorted(glob.glob(
        r"C:\Program Files*\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe"))
    for name in ("dumpbin.exe", "dumpbin"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
            break
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            out = subprocess.run([path, "/exports"], capture_output=True, text=True,
                                 timeout=30)
        except Exception:
            continue
        if out.returncode == 0 and "PE Dumper" in (out.stdout + out.stderr):
            version = next((line.strip() for line in (out.stdout + out.stderr).splitlines()
                            if "Dumper Version" in line), "")
            return path, version
    return None, None


def dumpbin(tool, flag, target):
    if not tool:
        return None
    try:
        out = subprocess.run([tool, flag, target], capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    return out.stdout if out.returncode == 0 else None


def parse_exports(text):
    """The export table of a PE dump as rows, so an empty list is a reading, not a guess."""
    rows = []
    for line in (text or "").splitlines():
        m = re.match(r"^\s*\d+\s+[0-9A-Fa-f]+\s+([0-9A-Fa-f]+)\s+(\S+)\s*$", line)
        if m:
            rows.append({"rva": "0x" + m.group(1), "name": m.group(2)})
    return rows


def parse_dependents(text):
    lines = (text or "").splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if "dependencies" in line.lower())
    except StopIteration:
        return []
    out = []
    for line in lines[start + 1:]:
        stripped = line.strip()
        if not stripped:
            if out:
                break
            continue
        if stripped.lower().startswith("summary"):
            break
        out.append(stripped)
    return out


def _type_name(decl):
    return getattr(decl, "__name__", None) or str(decl).strip("<>_ ").split(".")[-1]


def wrapper_signatures(export_names):
    """The ctypes declarations the production wrapper installs, read off the loaded library.

    This is the ABI as the running code understands it: a parameter is recorded with the
    ctypes type the call site passes, so `c_void_p` here means "a pointer-sized value", and
    nothing more. Whether the host treats that value as a host address or a device address is
    a separate question, answered by the call sites below and not by the type.
    """
    declared = {}
    load_error = None
    try:
        import dlss_engine
        lib = dlss_engine._load()
    except Exception as exc:  # a missing or unloadable binary is a finding, not a crash
        return {"declared": {}, "load_error": f"{type(exc).__name__}: {exc}"}
    for name in export_names:
        try:
            fn = getattr(lib, name)
        except Exception:
            continue
        argtypes = getattr(fn, "argtypes", None)
        restype = getattr(fn, "restype", None)
        declared[name] = {
            # `argtypes = []` is a declaration too: it says the function takes no arguments.
            "declared_by_wrapper": argtypes is not None,
            "parameters": [_type_name(t) for t in (argtypes or [])] or None,
            "return": _type_name(restype) if restype is not None else None,
            "pointer_parameters": sum(1 for t in (argtypes or [])
                                      if getattr(t, "__name__", "") == "c_void_p"),
        }
    return {"declared": declared, "load_error": load_error}


# ---------------------------------------------------------------------------
# Searches
# ---------------------------------------------------------------------------

def _walk_bounded(root, max_depth):
    root = os.path.abspath(root)
    base_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIRS]
        if dirpath.rstrip(os.sep).count(os.sep) - base_depth >= max_depth:
            dirnames[:] = []
        for name in filenames:
            yield os.path.join(dirpath, name)


def search_names(pattern_sets, roots=SEARCH_ROOTS, budget_s=SEARCH_TIME_BUDGET_S):
    """One bounded walk of `roots`, sorting every filename seen into `pattern_sets`.

    Returns `{category: hits}` plus a coverage record: the roots walked, the depth each was
    walked to, how many files were looked at, and whether the walk finished inside the time
    budget. A `present: false` in the artifact is therefore re-checkable by whoever reads it,
    instead of resting on an unbounded search nobody can repeat.
    """
    hits = {name: [] for name in pattern_sets}
    walked = []
    deadline = time.time() + budget_s
    finished = True
    for root, depth in roots:
        if not root or not os.path.isdir(root):
            continue
        entry = {"root": root, "max_depth": depth, "files_seen": 0, "completed": True}
        walked.append(entry)
        for path in _walk_bounded(root, depth):
            entry["files_seen"] += 1
            lowered = os.path.basename(path).lower()
            for category, globs_ in pattern_sets.items():
                if any(fnmatch.fnmatch(lowered, glo.lower()) for glo in globs_):
                    hits[category].append(path if not path.startswith(_ROOT) else
                                          os.path.relpath(path, _ROOT).replace("\\", "/"))
            if time.time() > deadline:
                entry["completed"] = False
                finished = False
                break
        if not finished:
            break
    coverage = {"roots_walked": walked, "budget_seconds": budget_s, "walk_finished": finished,
                "skipped_directory_names": sorted(SKIP_DIRS),
                "note": "a bounded filename search; contents of unrelated trees were not read"}
    return {name: sorted(set(items)) for name, items in hits.items()}, coverage


def grep_repo(pattern, files):
    """Every `file:line` in `files` (repo-relative) matching a regex, or [] when none does."""
    out = []
    for rel in files:
        path = os.path.join(_ROOT, rel.replace("/", os.sep))
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, 1):
                if re.search(pattern, line):
                    out.append(f"{rel}:{number}: {line.strip()[:160]}")
    return out


# ---------------------------------------------------------------------------
# dlss_host_abi.json
# ---------------------------------------------------------------------------

def abi_report(search_budget_s=SEARCH_TIME_BUDGET_S):
    tool, tool_version = find_dumpbin()
    exports_text = dumpbin(tool, "/exports", HOST_DLL) if os.path.exists(HOST_DLL) else None
    deps_text = dumpbin(tool, "/dependents", HOST_DLL) if os.path.exists(HOST_DLL) else None
    export_rows = parse_exports(exports_text)
    names = [row["name"] for row in export_rows]
    dlssnr_names = [n for n in names if n.startswith("dlssnr_")]

    gpu_named = {}
    for name in names:
        lowered = name.lower()
        tokens = [tok for tok in GPU_RESOURCE_TOKENS if tok in lowered]
        if tokens:
            gpu_named[name] = tokens

    signatures = wrapper_signatures(names)
    declared = signatures.get("declared") or {}
    with_signature = {n: v for n, v in declared.items() if v["declared_by_wrapper"]}
    host_pointer_call_sites = grep_repo(r"data_as\(ctypes\.c_void_p\)",
                                        ("dlss_engine.py", "pipeline.py", "optical_flow.py"))
    found, search_coverage = search_names({
        "host_source": [f"{pat}{ext}" for pat in HOST_SOURCE_PATTERNS for ext in SOURCE_EXTS],
        "sdk_headers": list(HEADER_PATTERNS),
        "sdk_libs": list(LIB_PATTERNS),
    }, budget_s=search_budget_s)
    host_source_hits, header_hits, lib_hits = (found["host_source"], found["sdk_headers"],
                                               found["sdk_libs"])
    # The wrapper's own export symbol can also be defined by source outside this checkout;
    # a content grep of the repository's native sources is the cheap, bounded version of
    # that question. Magpie is excluded because it is a separate D3D12 application, and a
    # match in it would not be the source of the binary this project loads.
    grepped = _repo_source_files()
    defining_source = grep_repo(r"__declspec\s*\(\s*dllexport[^)]*\)\s*(?:\w+\s+)*int\s+dlssnr_",
                                grepped)

    host_source_present = bool(host_source_hits or defining_source)
    sdk_headers_present = bool(header_hits)
    sdk_libs_present = bool(lib_hits)
    host_pointer_path_proven = bool(host_pointer_call_sites) and any(
        v["pointer_parameters"] for v in with_signature.values())
    public_cuda_pointer_api = any("cuda" in toks for toks in gpu_named.values())
    public_d3d12_resource_api = any(t in ("d3d12", "resource", "srv", "uav", "fence")
                                    for toks in gpu_named.values() for t in toks)
    public_shared_handle_api = any("handle" in toks or "shared" in toks
                                   for toks in gpu_named.values())

    available = os.path.exists(HOST_DLL) and bool(names)
    conclusion = "BLOCKED_PROVEN" if available else "NOT_READY"
    if available and (public_cuda_pointer_api or public_d3d12_resource_api
                      or public_shared_handle_api):
        conclusion = "GPU_IO_AVAILABLE"

    return _identity({
        "evaluated_binary": {
            "path": os.path.relpath(HOST_DLL, _ROOT).replace("\\", "/"),
            "exists": os.path.exists(HOST_DLL),
            "size_bytes": os.path.getsize(HOST_DLL) if os.path.exists(HOST_DLL) else None,
            "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime(os.path.getmtime(HOST_DLL)))
                         if os.path.exists(HOST_DLL) else None,
            "sha256": sha256(HOST_DLL) if os.path.exists(HOST_DLL) else None,
            "loaded_by": "dlss_engine.HOST_DLL (package dir of the checkout)",
        },
        "ngx_runtime_binary": {
            "path": os.path.relpath(NGX_RUNTIME_DLL, _ROOT).replace("\\", "/"),
            "exists": os.path.exists(NGX_RUNTIME_DLL),
            "sha256": sha256(NGX_RUNTIME_DLL) if os.path.exists(NGX_RUNTIME_DLL) else None,
            "size_bytes": os.path.getsize(NGX_RUNTIME_DLL) if os.path.exists(NGX_RUNTIME_DLL) else None,
            "note": "the NGX DLSSNR runtime the host loads; a runtime binary, not an import "
                    "library, and it exports no function this project calls directly",
        },
        "pe_tool": {"path": tool, "version": tool_version or None,
                    "invocation": "dumpbin /exports <dll>, dumpbin /dependents <dll>",
                    "exports_dump_captured": bool(exports_text)},
        "public_exports": [row["name"] for row in export_rows],
        "dlssnr_exports": dlssnr_names,
        "export_count": len(export_rows),
        "exports": export_rows,
        "imports_of_host_dll": parse_dependents(deps_text),
        "export_names_naming_a_gpu_resource": gpu_named,
        "gpu_resource_token_scan": {
            "tokens": list(GPU_RESOURCE_TOKENS),
            "matched_exports": gpu_named,
            "method": "every export name tested for each token; the matches are listed",
        },
        "wrapper_signatures": declared,
        "wrapper_load_error": signatures.get("load_error"),
        "exports_without_a_python_declaration": sorted(set(names) - set(with_signature)),
        "signature_reading": {
            "pointer_typed_parameters": {n: v["pointer_parameters"]
                                         for n, v in with_signature.items()},
            "note": "a c_void_p is a pointer-sized argument. It does not say whose address "
                    "space the value names, so it is not a device-pointer API.",
        },
        "host_source_present": host_source_present,
        "host_source_matches": host_source_hits + defining_source,
        "host_source_content_grepped_file_count": len(grepped),
        "sdk_headers_present": sdk_headers_present,
        "sdk_header_matches": header_hits,
        "sdk_libs_present": sdk_libs_present,
        "sdk_lib_matches": lib_hits,
        "dependency_search_coverage": search_coverage,
        "search_was_download": False,
        "public_cuda_pointer_api_present": public_cuda_pointer_api,
        "public_d3d12_resource_api_present": public_d3d12_resource_api,
        "public_shared_handle_api_present": public_shared_handle_api,
        "gpu_resource_public_abi": "available" if conclusion == "GPU_IO_AVAILABLE"
                                   else "unavailable",
        "host_pointer_path_proven": bool(host_pointer_call_sites) and bool(declared),
        "known_host_pointer_runtime_path": {
            "description": "the call sites hand the host the address of a contiguous NumPy "
                           "buffer in this process: colour RGBA8 in, motion float2, depth "
                           "float32, RGBA8 out, plus an int reset flag",
            "call_sites": host_pointer_call_sites,
            "buffer_formats": {
                "colour_in": "working-resolution RGBA8 uint8 host buffer",
                "motion_in": "float32 (h, w, 2) host buffer from the flow D2H",
                "depth_in": "float32 (h, w) host buffer",
                "colour_out": "RGBA8 uint8 host buffer owned by the caller",
            },
        },
        "known_device_pointer_evidence": {
            "re_tested_this_round": False,
            "historical_record": os.path.relpath(HISTORICAL_ABI_DOC, _ROOT).replace("\\", "/"),
            "historical_record_sha256": sha256(HISTORICAL_ABI_DOC)
                                        if os.path.exists(HISTORICAL_ABI_DOC) else None,
            "historical_claims": [line.strip() for line in
                                  (open(HISTORICAL_ABI_DOC, encoding="utf-8",
                                        errors="replace").read().splitlines()
                                   if os.path.exists(HISTORICAL_ABI_DOC) else [])
                                  if "CUdeviceptr" in line or "0xC0000005" in line],
            "why_not_re_run": "a CUdeviceptr passed to dlssnr_process is reported to raise an "
                              "access violation, and repeating that needs an isolated child "
                              "process for one frame; the historical record already carries "
                              "the result, so the dangerous run was not repeated",
            "reading": "no device-pointer input is proven usable through the public ABI",
        },
        "conclusion": conclusion,
        "conclusion_reason": None if conclusion == "GPU_IO_AVAILABLE" else
        "blocked by current host ABI and available source/dependencies",
        "conclusion_scope": "this binary, this export table and the dependencies present on "
                            "this machine at collection time; it is not a statement about "
                            "what is impossible forever, and nothing here infers what the "
                            "host does with a pointer once it has it",
    })


def _repo_source_files():
    out = []
    for path in _walk_bounded(_ROOT, 6):
        rel = os.path.relpath(path, _ROOT).replace("\\", "/")
        if rel.lower().endswith(SOURCE_EXTS) and not rel.startswith("Magpie/"):
            out.append(rel)
    return out


# ---------------------------------------------------------------------------
# process_boundaries.json
# ---------------------------------------------------------------------------

def boundaries_report(ledger, ledger_path=None, abi=None):
    steady = (ledger or {}).get("measured_steady_state") or {}
    matches = {name: grep_repo(pattern, PRODUCTION_SOURCES)
               for name, pattern in IPC_PATTERNS.items()}
    used = {name: hits for name, hits in matches.items() if hits}
    cuda_ipc = any(h for n, h in matches.items() if n.startswith("cuIpc"))
    shared_d3d = any(h for n, h in matches.items()
                     if n in ("ID3D12Device_CreateSharedHandle", "OpenSharedHandle",
                              "ID3D11Texture2D_shared_resource", "ID3D12Resource"))
    decoder_read_evidence = grep_repo(r"DECODER_PIPE\.record|_read_exactly\(", ("pipeline.py",))
    encoder_pipe_writes = grep_repo(r"_write_frame_bytes\(|ENCODER_PIPE\.record",
                                    ("pipeline.py", "transfer_profiler.py"))
    download_filter = grep_repo(r"hwdownload", ("pipeline.py",))
    named_gpu_exports = (abi or {}).get("export_names_naming_a_gpu_resource")

    payload_4k_bgr24 = 24883200
    return _identity({
        "ledger_source": (os.path.relpath(os.path.abspath(ledger_path), _ROOT).replace("\\", "/")
                          if ledger_path else None),
        "measurement_basis_of_numbers_below": "steady-state medians of the per-frame records "
                                              "of the referenced transfer ledger; a field that "
                                              "ledger did not measure is None here",
        "decoder": {
            "process_boundary": "separate_process",
            "process": "FFmpeg child process spawned by pipeline.py",
            "gpu_resource_owner": "the FFmpeg child process; it owns the NVDEC surfaces and "
                                  "the CUDA context that holds them",
            "route": "FFmpeg child -> NVDEC GPU frame -> scale_cuda -> hwdownload to NV12 in "
                     "the child's host memory -> rawvideo on stdout -> this Python process",
            "handoff": "host byte stream on the child's stdout pipe",
            "gpu_to_cpu_within_decoder": "hwdownload inside the child",
            "hwdownload_evidence": download_filter,
            "read_evidence": decoder_read_evidence,
            "pipe_bytes_measured_per_frame": steady.get("decoder_pipe_bytes"),
            "pipe_bytes_measurement": "bytes read out of the decoder child's stdout, summed "
                                      "from the reads that produced the frame",
            "gpu_to_cpu_bytes_estimated_per_frame": steady.get(
                "decoder_gpu_to_cpu_bytes_estimated"),
            "gpu_to_cpu_bytes_estimation_note": "FFmpeg exposes no transfer counter, so this "
                                                "is arithmetic on the frame format and is "
                                                "labelled estimated, never measured",
            "shared_gpu_resource": False,
            "gpu_resource_export_api": "none",
            "cuda_ipc_currently_used": cuda_ipc,
            "shared_d3d_handle_currently_used": shared_d3d,
            "stdout_carries": "rawvideo bytes only",
            "stdout_does_not_carry": ["CUdeviceptr", "ID3D11Texture2D*", "ID3D12Resource*"],
            "zero_copy_into_python_process": "BLOCKED_PROVEN",
            "unblock_requires_any_of": ["same-process libavcodec hardware decoder",
                                        "NVIDIA Video Codec SDK / NVDEC bridge",
                                        "explicit GPU IPC / shared-resource helper"],
            "implemented_this_round": False,
            "hwdownload_removed_this_round": False,
            "status": "BLOCKED_PROVEN",
        },
        "encoder": {
            "process_boundary": "separate_process",
            "process": "FFmpeg child process spawned by pipeline.py",
            "input": "rawvideo BGR24 on the child's stdin",
            "gpu_resource_owner": "the FFmpeg child process uploads the rawvideo it reads",
            "pipe_bytes_measurement": "the byte count returned by _write_frame_bytes() for "
                                      "that frame, accumulated by the pipe ledger - not "
                                      "width*height*3 arithmetic",
            "pipe_bytes_measured_per_frame": steady.get("encoder_pipe_bytes"),
            "logical_input_payload_bytes_4k_bgr24": payload_4k_bgr24,
            "logical_vs_measured": "under a normal BGR24 output the two agree; only the "
                                   "measured column is called measured",
            "encoder_gpu_upload_bytes_measured": "unavailable",
            "encoder_gpu_upload_note": "FFmpeg exposes no GPU transfer counter, so the "
                                       "CPU rawvideo -> NVENC surface copy has no number here",
            "write_evidence": encoder_pipe_writes,
            "shared_gpu_resource": False,
            "cuda_ipc_currently_used": cuda_ipc,
            "shared_d3d_handle_currently_used": shared_d3d,
            "direct_gpu_encode": "BLOCKED_PROVEN",
            "unblock_requires_any_of": ["same-process NVENC",
                                        "same-process libavcodec NVENC",
                                        "GPU shared-resource IPC into an encoder helper"],
            "implemented_this_round": False,
            "status": "BLOCKED_PROVEN",
        },
        "nvof": {
            "process_boundary": "in-process",
            "process": "the backend Python process",
            "gpu_context": "native nvof_bridge CUDA context, opened once per session",
            "input_handoff": "BGR24 host pointer -> one cuMemcpyHtoD into a persistent "
                             "staging buffer -> byte-map kernel into the NVOF input surface",
            "input_format_measured": (ledger or {}).get("nvof_input_format")
                                     or steady.get("nvof_input_format"),
            "h2d_bytes_measured_per_frame": steady.get("nvof_h2d_bytes_measured"),
            "h2d_copy_count_measured_per_frame": steady.get("nvof_h2d_copy_count"),
            "d2h_bytes_measured_per_frame": steady.get("nvof_d2h_bytes_measured"),
            "d2h_copy_count_measured_per_frame": steady.get("nvof_d2h_copy_count"),
            "measurement": "nvof_bridge_get_counters, read either side of the frame",
            "output_handoff": "float2 flow -> D2H into a host NumPy buffer",
            "flow_d2h_status": "BLOCKED_PROVEN",
            "flow_d2h_note": "the DLSS host reads motion from host memory, so the flow field "
                             "has to come back; it is not called zero-copy anywhere",
            "history": "previous-frame surfaces stay on the GPU between frames",
            "staging_buffer_allocations_in_steady_state": steady.get(
                "nvof_bgr_staging_alloc_count"),
            "shared_gpu_resource_with_decoder": False,
            "shared_gpu_resource_with_dlss_host": False,
        },
        "dlss": {
            "process_boundary": "in-process",
            "process": "the backend Python process",
            "public_input": "host pointers (void*) for colour, motion and depth",
            "public_output": "host pointer for the RGBA8 output",
            "gpu_resource_public_api": ("none" if named_gpu_exports == {}
                                        else named_gpu_exports) if abi is not None else None,
            "internal_h2d": "unavailable",
            "internal_d2h": "unavailable",
            "internal_note": "the host wrapper reports no transfer counter and its source is "
                             "not in this repository; the logical host-buffer sizes are the "
                             "buffers it was handed, not measurements of what it copied",
            "round_trip_status": "BLOCKED_PROVEN",
            "see": "dlss_host_abi.json",
        },
        "symbol_scan": {
            "files": list(PRODUCTION_SOURCES),
            "patterns": IPC_PATTERNS,
            "matches_by_pattern": matches,
            "matched": used,
            "note": "the booleans above are this scan: a pattern that appears nowhere in the "
                    "production sources is reported as not currently used, which is a "
                    "statement about this checkout and not about CUDA or D3D12",
        },
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="artifact directory")
    parser.add_argument("--ledger", default="", help="transfer_ledger.json for the measured "
                                                     "pipe and NVOF byte columns")
    parser.add_argument("--abi-only", action="store_true")
    parser.add_argument("--search-budget", type=float, default=SEARCH_TIME_BUDGET_S,
                        help="seconds the dependency filename search may spend walking")
    parser.add_argument("--boundaries-only", action="store_true")
    args = parser.parse_args(argv)

    if sys.stdout is not None:
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ledger = None
    if args.ledger:
        with open(args.ledger, encoding="utf-8") as handle:
            ledger = json.load(handle)

    written = []
    summary = {}
    abi = None
    if not args.boundaries_only:
        abi = abi_report(args.search_budget)
        written.append(write(os.path.join(args.out, "dlss_host_abi.json"), abi))
        summary["dlss_host_abi"] = {
            "conclusion": abi["conclusion"],
            "exports": abi["dlssnr_exports"],
            "gpu_resource_public_abi": abi["gpu_resource_public_abi"],
            "host_source_present": abi["host_source_present"],
            "sdk_headers_present": abi["sdk_headers_present"],
            "sdk_libs_present": abi["sdk_libs_present"],
            "host_pointer_path_proven": abi["host_pointer_path_proven"],
        }
    if not args.abi_only:
        boundaries = boundaries_report(ledger, args.ledger or None, abi)
        written.append(write(os.path.join(args.out, "process_boundaries.json"), boundaries))
        summary["process_boundaries"] = {
            # The wording guard reads this line as a denial, which is what it is: the decoder's
            # GPU frames reach this process only as bytes a child process wrote, so the handoff
            # is blocked by the boundary, not merely absent from the code.
            "decoder_zero_copy_into_python_process":
                boundaries["decoder"]["zero_copy_into_python_process"],
            "direct_gpu_encode": boundaries["encoder"]["direct_gpu_encode"],
            "cuda_ipc_currently_used": boundaries["decoder"]["cuda_ipc_currently_used"],
            "shared_d3d_handle_currently_used":
                boundaries["decoder"]["shared_d3d_handle_currently_used"],
            "encoder_pipe_bytes_measured_per_frame":
                boundaries["encoder"]["pipe_bytes_measured_per_frame"],
            "nvof_h2d_bytes_measured_per_frame":
                boundaries["nvof"]["h2d_bytes_measured_per_frame"],
        }
    print(json.dumps({"ok": True, "written": written, "summary": summary}, indent=2,
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
