"""The DLSS host failure type, kept away from the host wrapper that used to own it.

``DLSSInferenceError`` was defined by ``dlss_engine``, the ctypes wrapper for the closed
``dlssnr_host.dll``. The service, the desktop window and the project GPU host all raise
and catch it, so it cannot live in a module whose whole purpose was to load a binary
this build no longer ships: importing that module to reach an error class is how a code
path that never asked for the old host ends up owning it again.
"""


class DLSSInferenceError(RuntimeError):
    """Raised when the DLSS runtime, its D3D12 device, or a Feature 18 frame fails."""
