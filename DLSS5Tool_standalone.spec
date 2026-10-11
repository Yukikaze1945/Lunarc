# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller ONEDIR spec for the standalone DLSS5 tool (dlss5standaloneV2).

This ships WITHOUT torch / depth / optical-flow models: the exe runs the no-guidance
DLSS path (single-image, image-folder, and no-guidance video) plus CRF+audio export.
Torch & models stay external; the "生成深度/光流" buttons report that they need a full
build if torch is missing.

What has to be collected, and where each file lands, is not spelled out here: it comes
from bundle_paths.resolve_sources(), the same manifest the production loaders resolve
their own DLLs against and that tests/unit/test_packaged_binaries.py checks. A spec
carrying its own private list of binaries is how this file ended up shipping neither
native/dlss_gpu_host.dll nor ffprobe.exe, while every DLSS call the window makes needs
the first one and every video it opens needs the second.

A frozen ONEDIR bundle has two roots. The launcher goes in <dist>/DLSS5Tool/ and every
entry below goes into <dist>/DLSS5Tool/_internal/ - the contents directory, which is
what sys._MEIPASS names and what bundle_paths.contents_dir() answers. A destination of
'native' therefore means _internal/native, beside the copies of itself the loaders look
for.

Build (from wherever the tree is, no working directory assumed):

    python -m PyInstaller --clean --noconfirm DLSS5Tool_standalone.spec --distpath OUT/dist --workpath OUT/build

Set DLSS5_FFMPEG_DIR to ship a specific FFmpeg; otherwise the installation on PATH is
used, because a bundle has to carry the build machine's own.
"""
import os
import sys

if SPECPATH not in sys.path:
    sys.path.insert(0, SPECPATH)

import bundle_paths

# Resolved against this file's own directory, so a build from another working directory
# collects the same tree. This raises - naming the file and the layout it was looking
# for - rather than producing a bundle that only fails when somebody draws a frame.
BINARIES = bundle_paths.resolve_sources(SPECPATH)

# What has to be named here rather than left to the scan, and why.
#
# Two product modules are reached only through a literal ``__import__("name")`` call -
# ``pipeline.export_gpu_transfer_stream``'s HDR route and ``dlss_service``'s opt-in
# read - and PyInstaller's module scanner does not follow that form. This was measured
# rather than assumed: the graph rooted at ``gui.py`` (PyInstaller 6.22.3 on Python
# 3.11.15) collects ``gpu_transfer_export``, because gui.py names it in an import
# statement, and reaches neither module below. Left as it was, a frozen build resolves
# the HDR route at runtime and raises for the first job that opts into HDR10.
#
# Being collected is not being loaded: ``gpu_hdr_transfer_export`` imports
# ``hdr_gpu_engine``, which opens ``native/gpu_hdr_scrgb_bridge.dll`` only when a bridge
# is constructed, so the analysis-time import that hiddenimports triggers touches no
# device. ``tests/unit/test_packaged_hdr_imports.py`` is the focused check of both
# halves - named in the spec, and no native load at CPU import.
#
# ``hdr_fusion_engine`` is named beside it for the typed solver a dual or triple HDR
# job opens. The reason is different and narrower: the chain reaches it through an
# ordinary ``import`` inside two functions, so the scan does follow it once the chain
# itself is collected - the entry states the dependency where the build can read it
# rather than leaving it implied by another entry's reachability. Loading the module
# allocates ctypes structs and nothing else; the solver DLL is opened by ``load()``,
# which only a constructed bridge calls, and that file is already in the manifest
# ``bundle_paths.resolve_sources`` produces for the multiscale backend.
HIDDENIMPORTS = ['PIL.Image', 'PIL.ImageTk',
                 'gpu_hdr_transfer_export',
                 'hdr_gpu_engine',
                 'hdr_fusion_engine']
HIDDENIMPORTS += ['post_control', 'temporal_control']
HIDDENIMPORTS += ['gpu_preview_engine', 'native_effects', 'ordered_vfg', 'dlaa_gpu_engine',
                  'rtx_vsr_engine', 'fidelityfx_flow_engine',
                  'comparison_media', 'comparison_audio', 'comparison_playback', 'queue_payload']

a = Analysis(
    # Absolute, so the build reads this tree's gui.py whichever directory it was asked
    # from; a relative entry here resolves against the working directory instead.
    [os.path.join(SPECPATH, 'gui.py')],
    pathex=[SPECPATH],
    binaries=list(BINARIES),
    datas=[],
    hiddenimports=HIDDENIMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # heavy libs we never import — shrink + skip accidental collection.
        # NOTE: do NOT exclude sympy/networkx/etc — torch imports them at runtime.
        'matplotlib', 'scipy', 'pandas', 'sklearn', 'tqdm',
        'IPython', 'jupyter', 'notebook',
    ],
    noarchive=False,
    optimize=0,
)

# PyInstaller walks the dependencies of everything above and copies what it resolves,
# which is how a bundle ends up carrying somebody else's files. Two rules, in one place
# so the reason each exists stays readable:
#
#   Never ship a file the operating system or the graphics driver owns. The bridges load
#   nvcuda.dll and nvofapi64.dll by bare name on purpose, so the installed driver
#   answers for them; a copy inside the bundle would be found first and cannot be newer
#   than the driver it shadows. Anything under the Windows directory is the same fault
#   one step along - that directory belongs to the machine receiving the bundle.
#
#   Never ship a second, flattened copy of a bridge. A DLL declared into native/ gets
#   its own imports resolved by name, and the dependency walk would land
#   nvof_d3d12_interop.dll at the contents root as well - which is not where
#   dlss_gpu_host.dll, importing it at load time, looks first.
_WINDOWS_DIR = os.path.normcase(os.environ.get("SystemRoot")
                                or os.environ.get("WINDIR") or r"C:\Windows")
_NEVER = {name.lower() for name in bundle_paths.NEVER_BUNDLE}
_NATIVE_ONLY = {name.lower() for name in bundle_paths.NATIVE_BINARIES}


def _carried_by_the_machine(name, source):
    """Whether the bundle has to leave this file to the OS or the installed driver."""
    return name in _NEVER or os.path.normcase(source or "").startswith(_WINDOWS_DIR)


def _keep(entry):
    destination = str(entry[0]).replace("\\", "/")
    name = os.path.basename(destination).lower()
    if _carried_by_the_machine(name, entry[1]):
        return False
    if name in _NATIVE_ONLY:
        # Keep the copy this spec placed; drop the one the dependency walk flattened.
        return destination.lower().startswith("native/")
    return True


a.binaries = [entry for entry in a.binaries if _keep(entry)]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    [],
    [],
    name='DLSS5Tool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,               # keep UPX off: would break the 165MB signed NVIDIA DLL
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='DLSS5Tool',
)
