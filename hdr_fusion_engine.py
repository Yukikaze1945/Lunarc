# -*- coding: utf-8 -*-
"""The typed linear-scRGB side of the multiscale fusion solver.

The solver in ``native/gpu_multiscale_fusion_bridge.cpp`` is one body of
mathematics - ``min_r sum_i w_i ||D_i r - r_i||^2 + beta ||r - r0||^2`` solved by
matrix-free CG, with the same float32 Lanczos taps, centres and clamp edges and the
same Catmull baseline - and it has always had two units it could be handed. The SDR
chain hands it 8-bit pixel samples through ``gpu_fusion_process``, and that entry's
byte semantics are fixed by every export that has ever run. The HDR chain hands it
signed linear scRGB, where 1.0 is 80 cd/m2, a component below 0 is out of gamut and
a component above 1 is headroom - neither of which an RGBA8 plane can carry. This
module speaks the second contract and only the second: ``gpu_fusion_open_hdr`` /
``gpu_fusion_process_hdr`` / ``gpu_fusion_get_hdr_counters``, and it never falls
back to the SDR entries when they are missing, because a byte-quantised composite
presented as an HDR one is the failure this split exists to prevent.

What is deliberately *not* here: a second set of tap plans. The plans, the
``GpuFusionOptions`` struct and the axis marshalling are the ones
``gpu_transfer_export`` already builds for the SDR chain, imported rather than
copied, so a fused HDR frame and a fused SDR frame are proved about the same
filter. A wrapper that carried its own plan code would let the two drift, and the
drift would show up as a difference in the picture rather than as a failing test.

Lifetime and completion, which the native header states and this module enforces:
the master plane and every observation plane are CUDA device pointers into bridges
that other threads own. ``gpu_hdr_decode``/``gpu_hdr_import_master`` and
``gpu_hdr_observe`` each wait their own pass fence before returning, so a plane
handed here is already device-visible, and the solver copies each plane
device-to-device inside the call - so a pointer only has to survive the call. The
*contents* belong to the observing bridge and are rewritten by its next
``observe()``, which is why each layer needs its own ``GpuHdrBridge`` and why the
descriptor returned here is consumed before the next frame is submitted.

The output is the bridge's own full-extent R16G16B16A16_FLOAT texture plus the
solver queue's real fence and the value that retires those writes. It is handed
back as the dict ``hdr_gpu_engine.GpuHdrBridge.set_composite`` takes, so the fused
frame goes to the existing encode without anything clamping it on the way.

Packaging, for the next stage: this module imports ``gpu_transfer_export`` and
``hdr_gpu_engine`` statically, and it loads ``gpu_multiscale_fusion_bridge.dll``,
which now links the RGBA16F row exports of ``nvof_d3d12_interop.dll`` directly. A
bundle has to carry both the module name (spec ``hiddenimports``, if it is resolved
lazily like the other HDR entries) and an interop DLL new enough to export them -
an old one fails at load, not at open.
"""
from __future__ import annotations

import ctypes as C
import math

import bundle_paths
from gpu_transfer_export import _FusionLayer, _FusionOptions, _fusion_axis
from hdr_gpu_engine import DXGI_R16G16B16A16_FLOAT
from hdr_gpu_engine import _require_scrgb16f_plane, _to_int_ptr

GPU_FUSION_HDR_ABI_VERSION = 1
GPU_FUSION_ABI_VERSION = 1
# GpuFusionOptions.layers is a fixed array of three; the bridge refuses anything
# else, so the refusal is stated here too rather than left for a native code.
LAYER_COUNT_MIN, LAYER_COUNT_MAX = 2, 3
HDR_PIXEL_BYTES = 8
D3D12_COPY_SOURCE = 0x800

# What each non-zero stage code of gpu_fusion_process_hdr means, in the caller's
# terms. A number with no story attached is how a bridge failure gets guessed at.
STAGE_CODES = {
    1: "arguments or the typed scRGB contract (including a bridge opened for SDR)",
    2: "an observation that does not describe its layer's residual plane",
    3: "a wide-row transport this build cannot make",
    4: "the CUDA context changed since open",
    5: "the full-resolution source plane copy failed",
    6: "an observation plane copy failed",
    7: "waiting a producer's own work failed",
    8: "a solver dispatch failed",
    9: "the initial residual norm could not be measured",
    10: "a conjugate-gradient step failed",
    11: "the operator stopped being positive definite",
    12: "the iteration budget ran out before the tolerance was met",
    13: "the composite store dispatch failed",
    14: "the store's counted fallback could not be read back",
    15: "a fused value the half grid cannot hold, with no master sample to fall "
       "back to",
    16: "flushing the store failed",
}

# gpu_fusion_open_hdr's codes. Code 3 is shared by two very different refusals -
# an unusable baseline plan and an interop build that cannot move a half row - and
# the bridge's own last message is what tells them apart, so both are named here
# rather than one being quietly dropped.
OPEN_STAGE_CODES = {
    1: "the typed options struct (its size, ABI version or reserved field) or a "
       "pixel_bytes that is not 8",
    2: "a layer plan that does not fit its extents, its weight or the tap budget",
    3: "the Catmull baseline plan, or a CUDA/D3D12 interop build with no RGBA16F "
       "row transport",
    4: "allocating the bridge",
    5: "the device side: queue, resources, root signature or kernels",
    6: "the completion event",
}

# Which of ``gpu_fusion_process_hdr``'s codes leave the bridge marked failed.
# Read off the native source rather than assumed: a refusal to even start a solve
# (bad arguments, a mis-describing observation, no wide-row transport, a changed
# CUDA context) returns before any state is written, so it poisons nothing and a
# corrected call on the same bridge is answered. Every code from the first copy
# onwards sets ``bridge->failed``.
#
# This is a description of the bridge's state, not a licence to keep exporting:
# the production chain aborts the job on any non-zero code, poisoned or not,
# because a frame that never reached the store has no composite to encode.
POISONING_STAGES = frozenset(range(5, 17))


class HdrFusionError(RuntimeError):
    """A typed HDR fusion call that will not be answered by the SDR entries."""


def _report_dict(struct, fields):
    return {name: getattr(struct, name) for name in fields}


class _HdrOptions(C.Structure):
    """Mirror of GpuFusionHdrOptions in native/gpu_multiscale_fusion_bridge.h."""
    _fields_ = [("struct_size", C.c_uint32), ("abi_version", C.c_uint32),
                ("pixel_bytes", C.c_uint32), ("reserved", C.c_uint32)]


class _HdrObservation(C.Structure):
    """One layer's signed working residual, exactly as ``gpu_hdr_observe``
    returned it: a CUDA device pointer, a byte pitch and the working extent. Not an
    R32 texture and not reinterpreted as one."""
    _fields_ = [("residual_ptr", C.c_uint64), ("residual_pitch", C.c_uint32),
                ("width", C.c_uint32), ("height", C.c_uint32)]


class _HdrReport(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("abi_version", C.c_uint32),
        ("iterations", C.c_uint32), ("initial_residual_norm", C.c_float),
        ("final_residual_norm", C.c_float), ("relative_residual", C.c_float),
        ("converged", C.c_uint32), ("unrepresentable_components", C.c_uint32),
        ("output_format", C.c_uint32), ("output_width", C.c_uint32),
        ("output_height", C.c_uint32), ("output_state", C.c_uint32),
        ("source_device_copy_bytes", C.c_uint64),
        ("observation_device_copy_bytes", C.c_uint64),
        ("scalar_readback_bytes", C.c_uint64)]


class _HdrCounters(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("abi_version", C.c_uint32),
        ("process_count", C.c_uint64), ("source_device_copy_bytes", C.c_uint64),
        ("observation_device_copy_bytes", C.c_uint64),
        ("scalar_readback_bytes", C.c_uint64),
        ("unrepresentable_components", C.c_uint64)]


REPORT_FIELDS = tuple(name for name, _ in _HdrReport._fields_[2:])
COUNTER_FIELDS = tuple(name for name, _ in _HdrCounters._fields_[2:])

_u64p = C.POINTER(C.c_uint64)
# name -> (argtypes, restype), in the style hdr_gpu_engine binds its own group.
_HDR_BRIDGE = [
    ("gpu_fusion_hdr_abi_version", [], C.c_uint32),
    ("gpu_fusion_open_hdr", [C.c_void_p, C.POINTER(_FusionOptions),
        C.POINTER(_HdrOptions), C.POINTER(C.c_void_p)], C.c_int32),
    ("gpu_fusion_process_hdr", [C.c_void_p, C.c_void_p, C.c_uint64, C.c_size_t,
        C.POINTER(_HdrObservation), C.c_uint32, C.POINTER(C.c_void_p),
        C.POINTER(C.c_void_p), _u64p, C.POINTER(_HdrReport)], C.c_int32),
    ("gpu_fusion_get_hdr_counters", [C.c_void_p, C.POINTER(_HdrCounters)],
        C.c_int32),
]
_HDR_SHARED = [
    ("gpu_fusion_abi_version", [], C.c_uint32),
    ("gpu_fusion_close", [C.c_void_p], None),
    ("gpu_fusion_last_error", [], C.c_char_p),
]
HDR_FUSION_SYMBOLS = tuple(name for name, _, _ in _HDR_BRIDGE)

_library = None


def _bind_all(library, table):
    for name, argtypes, restype in table:
        try:
            entry = getattr(library, name)
        except AttributeError:
            raise HdrFusionError(
                f"gpu_multiscale_fusion_bridge.dll does not export {name!r}")
        entry.argtypes = argtypes
        entry.restype = restype
    return library


def bind_shared(library):
    """The entries the typed group borrows from the bridge that has always been
    there: its ABI, its error text and its close. Bound without checking the ABI,
    so a probe can read the bridge's own answer before anything is refused."""
    return _bind_all(library, _HDR_SHARED)


def last_error_text(library):
    getter = getattr(library, "gpu_fusion_last_error", None)
    if getter is None:
        return ""
    detail = getter()
    return detail.decode("utf-8", "replace") if isinstance(detail, bytes) else (
        detail or "")


def probe_hdr_fusion(library=None):
    """Ask the loaded solver, by symbol name, whether it carries the typed HDR
    group. Returns ``{"supported", "version", "missing", "reason"}`` and never
    raises: a bundle whose fusion DLL predates these entries has to be able to be
    told that before anything is opened, not be handed a ctypes AttributeError.
    """
    if library is None:
        try:
            library = load()
        except (FileNotFoundError, OSError) as error:
            return {"supported": False, "version": 0,
                    "missing": list(HDR_FUSION_SYMBOLS),
                    "reason": f"gpu_multiscale_fusion_bridge.dll could not be "
                              f"loaded: {error}"}
    missing = []
    for name, _, _ in _HDR_BRIDGE:
        try:
            getattr(library, name)
        except AttributeError:
            missing.append(name)
    if missing:
        return {"supported": False, "version": 0, "missing": missing,
                "reason": "gpu_multiscale_fusion_bridge.dll does not export "
                          f"{', '.join(missing)}: this build has no typed linear "
                          "scRGB fusion contract, so an HDR frame would have to be "
                          "solved as RGBA8 bytes"}
    bind_shared(library)
    for name, argtypes, restype in _HDR_BRIDGE:
        entry = getattr(library, name)
        entry.argtypes = argtypes
        entry.restype = restype
    try:
        version = int(library.gpu_fusion_hdr_abi_version())
    except (TypeError, ValueError):
        version = -1
    if version != GPU_FUSION_HDR_ABI_VERSION:
        return {"supported": False, "version": version, "missing": [],
                "reason": f"the solver's typed HDR group reports version {version}, "
                          f"this wrapper speaks {GPU_FUSION_HDR_ABI_VERSION}"}
    sdr_abi = int(library.gpu_fusion_abi_version())
    if sdr_abi != GPU_FUSION_ABI_VERSION:
        return {"supported": False, "version": version, "missing": [],
                "reason": f"the solver speaks SDR ABI {sdr_abi}, this wrapper "
                          f"speaks {GPU_FUSION_ABI_VERSION}, and the two share one "
                          "GpuFusionOptions layout"}
    return {"supported": True, "version": version, "missing": [], "reason": ""}


def load(path=None):
    """The bundled solver, loaded once, with the typed group bound and the ABI
    checked. Import of this module never calls here.

    ``path`` is for a test that has to point at one explicitly named build of the
    DLL - the same wrapper code then drives a pre-A2 binary and reports what it is
    missing instead of driving it half-way.
    """
    global _library
    if _library is not None and path is None:
        return _library
    bundle_paths.add_dll_directory(bundle_paths.native_dir())
    bundled = path is None
    if bundled:
        path = bundle_paths.require_native(
            "gpu_multiscale_fusion_bridge.dll",
            "solve the multiscale HDR fusion on the GPU")
    library = C.CDLL(path)
    library._dll_path = path
    answer = probe_hdr_fusion(library)
    if not answer["supported"]:
        raise HdrFusionError(answer["reason"])
    if bundled:
        _library = library
    return library


def reset_loaded_library():
    """Drop the cached library handle, so an injected fake cannot leak into a later
    real load."""
    global _library
    _library = None


def _bounded_int(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int):
        raise HdrFusionError(f"{name} must be an integer, got {value!r}")
    if value < minimum or value > maximum:
        raise HdrFusionError(f"{name} {value} is outside {minimum}..{maximum}, "
                             f"which is what the fusion bridge accepts")
    return value


def _positive_finite(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise HdrFusionError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(number) or number <= 0:
        raise HdrFusionError(f"{name} must be a finite number above 0, got "
                             f"{value!r}")
    return number


def _nonnegative_finite(value, name):
    """The validator the residual strength answers to: finite and zero or above.

    The native contract is ``residual_strength >= 0`` - 0 is "solve, store the
    master alone", which the single-layer HDR chain has always accepted - so
    refusing it here would make the typed path narrower than the one it is
    replacing. Everything the objective needs strictly positive (the layer
    weights, beta, the stopping tolerance) still goes through
    ``_positive_finite``.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise HdrFusionError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(number) or number < 0:
        raise HdrFusionError(f"{name} must be a finite number at or above 0, "
                             f"got {value!r}")
    return number


def layer_options(*, full_width, full_height, layers):
    """The bridge's ``GpuFusionOptions`` for one fused HDR frame, built from the
    pipeline's own float32 plans.

    ``layers`` is a sequence of ``{"plan": TapPlans, "dims": (w, h),
    "weight": float}`` - the same three numbers the SDR chain's ``_open_fusion``
    marshals, through the same ``_fusion_axis``. The layer's ``D_i`` is its
    ``plan.working_color`` (full -> working) and ``r0`` is ``plan.upscale`` of the
    first layer (working -> full) as the Catmull baseline; the native bridge reads
    the baseline out of layer 0's extent, so every layer shares one baseline shape.
    Keep the returned struct's plans alive for the whole open session: the views
    carry raw pointers into the arrays inside them.
    """
    if len(layers) < LAYER_COUNT_MIN or len(layers) > LAYER_COUNT_MAX:
        raise HdrFusionError(
            f"the fusion solver takes {LAYER_COUNT_MIN} to {LAYER_COUNT_MAX} "
            f"layers, and {len(layers)} were given")
    options = _FusionOptions()
    options.struct_size = C.sizeof(_FusionOptions)
    options.full_width = _bounded_int(int(full_width), "full_width", 2, 16384)
    options.full_height = _bounded_int(int(full_height), "full_height", 2, 16384)
    options.layer_count = len(layers)
    for index, layer in enumerate(layers):
        try:
            plan, dims, weight = layer["plan"], layer["dims"], layer["weight"]
        except (KeyError, TypeError) as error:
            raise HdrFusionError(
                f"layer {index} must be a dict with plan, dims and weight: {error}")
        width, height = int(dims[0]), int(dims[1])
        if width < 1 or height < 1:
            raise HdrFusionError(
                f"layer {index}'s extent {width}x{height} has no samples in it")
        if width > options.full_width or height > options.full_height:
            raise HdrFusionError(
                f"layer {index}'s extent {width}x{height} is larger than the full "
                f"extent {options.full_width}x{options.full_height}")
        options.layers[index] = _FusionLayer(
            width, height, _fusion_axis(plan.working_color, True),
            _fusion_axis(plan.working_color, False), _positive_finite(
                weight, f"layer {index} weight"))
    baseline = layers[0]["plan"].upscale
    options.baseline_horizontal = _fusion_axis(baseline, True)
    options.baseline_vertical = _fusion_axis(baseline, False)
    return options


class HdrFusionBridge:
    """One open typed HDR fusion solver.

    The D3D12 device and the CUDA context are borrowed, as everywhere in this
    chain: the solver copies the caller's planes and shares its buffers with the
    bridges that own them, so it must be the device the Feature 18 host and the
    encoder use and the context the decode landed in, and the context may not
    change for the life of the bridge. Nothing here is destroyed on an arbitrary
    thread - cleanup is ``close()`` or a ``with`` block.
    """

    def __init__(self, *, d3d12_device, full_width, full_height, layers, beta,
                 residual_strength, relative_tolerance, max_iterations,
                 pixel_bytes=HDR_PIXEL_BYTES, library=None):
        self._lib = bind_shared(library if library is not None else load())
        if library is not None:
            answer = probe_hdr_fusion(self._lib)
            if not answer["supported"]:
                raise HdrFusionError(answer["reason"])
        self._options = layer_options(full_width=full_width,
                                      full_height=full_height, layers=layers)
        self._full = (self._options.full_width, self._options.full_height)
        self._layers = list(layers)
        self._options.beta = C.c_float(_positive_finite(beta, "beta"))
        self._options.residual_strength = C.c_float(_nonnegative_finite(
            residual_strength, "residual_strength"))
        relative_tolerance = _positive_finite(relative_tolerance,
                                              "relative_tolerance")
        if relative_tolerance >= 1.0:
            raise HdrFusionError(
                f"relative_tolerance {relative_tolerance} is not below 1, so the "
                "solve could never be asked to stop")
        self._options.relative_tolerance = C.c_float(relative_tolerance)
        self._options.max_iterations = _bounded_int(int(max_iterations),
                                                    "max_iterations", 1, 1000)
        if int(pixel_bytes) != HDR_PIXEL_BYTES:
            raise HdrFusionError(
                f"the typed HDR contract carries {HDR_PIXEL_BYTES} bytes per "
                f"sample (R16G16B16A16_FLOAT); {pixel_bytes} is an SDR plane and "
                "belongs to gpu_fusion_process")
        hdr = _HdrOptions()
        hdr.struct_size = C.sizeof(_HdrOptions)
        hdr.abi_version = GPU_FUSION_HDR_ABI_VERSION
        hdr.pixel_bytes = HDR_PIXEL_BYTES
        hdr.reserved = 0
        handle = C.c_void_p()
        code = self._lib.gpu_fusion_open_hdr(
            C.c_void_p(_to_int_ptr(d3d12_device)), C.byref(self._options),
            C.byref(hdr), C.byref(handle))
        if code:
            raise HdrFusionError(
                f"gpu_fusion_open_hdr reported {code} "
                f"({OPEN_STAGE_CODES.get(int(code), 'an unlisted stage')}): "
                f"{last_error_text(self._lib) or 'no detail'}")
        if not handle.value:
            # Success with no handle is not a bridge: adopting the null would let
            # every later call run against whatever the caller's zero means.
            raise HdrFusionError(
                "gpu_fusion_open_hdr reported success and handed back no bridge "
                "handle")
        self._handle = handle

    # -- introspection ------------------------------------------------------

    def update_parameters(self, config, strength):
        self._require_open()
        from fusion_runtime import update_parameters
        if not update_parameters(self._lib, self._handle, config, strength):
            return False
        for index, weight in enumerate(config['weights']):
            self._options.layers[index].weight = weight
        self._options.beta = config['beta']
        self._options.residual_strength = strength
        self._options.relative_tolerance = config['relative_tolerance']
        self._options.max_iterations = config['max_iterations']
        return True

    @property
    def full_extent(self):
        return self._full

    @property
    def working_extents(self):
        return [(int(layer["dims"][0]), int(layer["dims"][1]))
                for layer in self._layers]

    @property
    def relative_tolerance(self):
        """The stopping tolerance as it was actually marshalled: a float32.

        The solver compares its own ratio against this stored value, so this and
        not the caller's Python double is the standard a returned frame is judged
        by. ``0.001`` is not exactly representable in either format, and holding
        the report to the unrounded number would be a stricter contract than the
        bridge was given - a solve that met the one it was sent would be refused.
        """
        return float(self._options.relative_tolerance)

    @property
    def max_iterations(self):
        return int(self._options.max_iterations)

    def _fail(self, entry, code, poisoning=None):
        """One non-zero code, said in full: the stage, what it means, the bridge's
        own message, and - when the caller says which state to report - whether the
        bridge was actually poisoned by it."""
        code = int(code)
        detail = last_error_text(self._lib) or "no detail"
        state = ""
        if poisoning:
            state = "; this bridge is marked failed and will refuse further frames"
        elif poisoning is not None:
            state = ("; this bridge refused before writing any state, so it will "
                     "answer a corrected call - which is a fact about the handle, "
                     "not permission to keep exporting: the frame has no composite")
        return HdrFusionError(
            f"{entry} reported {code} "
            f"({STAGE_CODES.get(code, 'an unlisted stage')}): {detail}{state}")

    def _require_open(self):
        if self._handle is None:
            raise HdrFusionError("gpu_fusion_close: this bridge is closed")

    # -- the one call -------------------------------------------------------

    def process(self, *, cuda_context, master, observations):
        """Solve the fused residual and store the scRGB composite.

        ``master`` is the pristine full-resolution linear scRGB FP16 plane the
        composite is built on - the ``DevicePlane`` an ``hdr_gpu_engine`` bridge
        returned from ``decode()`` or ``import_master()``. ``observations`` is one
        signed working residual per layer, in the same order, each as that layer's
        own bridge returned it from ``observe()``.

        Both are validated as typed planes here rather than left for the solver to
        notice: an ``rgba8_cuda`` descriptor is refused before a byte of it can be
        copied as if it were half-float colour, and a residual whose extent or
        pitch does not describe the layer it is attributed to is refused the same
        way, because the solver would otherwise read samples off the end of a row
        and still produce a picture.

        Returns a dict naming the texture, the producer fence and the value that
        retires the store, which is the shape
        ``hdr_gpu_engine.GpuHdrBridge.set_composite`` takes, plus the solver's own
        report. It gets there by checking the report, not by trusting the zero: the
        solve has to be converged against the tolerance this bridge was given, the
        norms it reports have to be finite, and the fence value has to name real
        work. A non-zero ``unrepresentable_components`` is still a success: those
        samples fell back to the pristine master because the half grid cannot hold
        the fused value, exactly as the single-layer HDR composite counts it, and
        the number is reported instead of smoothed over. The texture belongs to this
        bridge and its contents are overwritten by the next call.
        """
        self._require_open()
        context = _to_int_ptr(cuda_context)
        if not context:
            raise HdrFusionError("process needs the CUDA context the planes live in")
        source = _require_scrgb16f_plane(master, extent=self._full,
                                        cuda_context=context,
                                        label="fused scRGB master")
        if len(observations) != len(self._layers):
            raise HdrFusionError(
                f"the bridge was opened for {len(self._layers)} layers and "
                f"{len(observations)} observations were given; the solver requires "
                "exactly one residual per layer")
        descriptors = (_HdrObservation * len(observations))()
        for index, observation in enumerate(observations):
            extent = self._layers[index]["dims"]
            plane = _require_scrgb16f_plane(
                observation, extent=(int(extent[0]), int(extent[1])),
                cuda_context=context,
                label=f"layer {index} signed working residual")
            descriptors[index] = _HdrObservation(
                plane["device_ptr"], plane["pitch_bytes"], plane["width"],
                plane["height"])
        texture, fence, value = C.c_void_p(), C.c_void_p(), C.c_uint64()
        report = _HdrReport()
        report.struct_size = C.sizeof(_HdrReport)
        report.abi_version = GPU_FUSION_HDR_ABI_VERSION
        code = self._lib.gpu_fusion_process_hdr(
            self._handle, C.c_void_p(context), C.c_uint64(source["device_ptr"]),
            C.c_size_t(source["pitch_bytes"]), descriptors, len(descriptors),
            C.byref(texture), C.byref(fence), C.byref(value), C.byref(report))
        if code:
            # Which state this left the handle in is a fact the bridge itself
            # decides, and the stages differ: the ones that refuse before writing
            # anything do not poison it, the ones past the first copy do. Either
            # way there is no composite for this frame, so the caller ends the job
            # rather than retrying the solve.
            raise self._fail("gpu_fusion_process_hdr", code,
                             poisoning=int(code) in POISONING_STAGES)
        if not texture.value or not fence.value:
            raise HdrFusionError(
                "gpu_fusion_process_hdr reported success and handed back no "
                "texture or no producer fence")
        if not int(value.value):
            raise HdrFusionError(
                "gpu_fusion_process_hdr reported success with fence value 0, which "
                "names no work to wait for: a consumer on the encode queue would "
                "read the composite before the store retired it, so a zero here is "
                "not a completed frame")
        if int(report.output_format) != DXGI_R16G16B16A16_FLOAT:
            raise HdrFusionError(
                f"the fused output is DXGI format {report.output_format}, not "
                f"R16G16B16A16_FLOAT ({DXGI_R16G16B16A16_FLOAT}); an 8-bit store "
                "would not carry the values this contract is about")
        if (int(report.output_width), int(report.output_height)) != self._full:
            raise HdrFusionError(
                f"the fused output is {report.output_width}x"
                f"{report.output_height}, not the full extent "
                f"{self._full[0]}x{self._full[1]}")
        if int(report.output_state) != D3D12_COPY_SOURCE:
            raise HdrFusionError(
                f"the fused output was left in D3D12 resource state "
                f"{report.output_state}, which is not the COPY_SOURCE a consumer "
                "on another queue may take")
        self._require_solved(report)
        from fusion_diagnostics import read_solver_diagnostics
        native_report = _report_dict(report, REPORT_FIELDS)
        diagnostics = read_solver_diagnostics(self._lib, self._handle)
        if diagnostics is not None:
            native_report["diagnostics"] = diagnostics
        return {
            "d3d12_resource": texture.value, "d3d12_fence": fence.value,
            "fence_value": value.value,
            "dxgi_format": int(report.output_format),
            "width": int(report.output_width),
            "height": int(report.output_height),
            "state": int(report.output_state),
            "report": native_report,
            "iterations": int(report.iterations),
            "converged": bool(report.converged),
            "unrepresentable_components": int(report.unrepresentable_components),
        }

    def _require_solved(self, report):
        """The reported evidence, checked rather than taken on trust.

        A zero code means the store completed; it is the report that says whether
        the picture in it was solved. Each check therefore names the number that
        failed, because the only other reading of an unconverged or non-finite
        success is a composite that quietly carries the Catmull baseline. The
        comparison is against the float32 tolerance this bridge was actually given:
        a ratio the report itself rounds above it is refused even though the
        solver's double-precision comparison met it, which is the strict direction
        and the one that cannot ship an un-converged frame.
        """
        initial = float(report.initial_residual_norm)
        final = float(report.final_residual_norm)
        relative = float(report.relative_residual)
        iterations = int(report.iterations)
        tolerance = self.relative_tolerance
        for name, number in (("initial_residual_norm", initial),
                             ("final_residual_norm", final),
                             ("relative_residual", relative)):
            if not math.isfinite(number) or number < 0.0:
                raise HdrFusionError(
                    f"gpu_fusion_process_hdr reported success with "
                    f"{name} {number!r}: a norm is a finite non-negative number, "
                    "and one that is not says nothing about the picture it claims "
                    "to have measured")
        if relative > tolerance:
            raise HdrFusionError(
                f"the solve reports relative residual {relative:.8g}, above the "
                f"{tolerance:.8g} it was asked for ({iterations} iterations "
                "reported); the composite would be a partial solution presented as "
                "a converged one")
        if not int(report.converged):
            raise HdrFusionError(
                f"gpu_fusion_process_hdr reported success without a converged "
                f"solve: relative residual {relative:.8g} after {iterations} of "
                f"{self.max_iterations} iterations against a tolerance of "
                f"{tolerance:.8g}; a frame is not exported on an unconverged "
                "residual")

    def counters(self):
        """Lifetime totals of the typed path. Every field is a count the solver
        keeps; none of them is a defaulted zero standing in for a measurement it
        does not make. ``scalar_readback_bytes`` is the CG dot reductions and the
        store's counted fallback - scalars, not pictures, and not a claim that the
        host crossed nothing."""
        self._require_open()
        block = _HdrCounters()
        block.struct_size = C.sizeof(_HdrCounters)
        block.abi_version = GPU_FUSION_HDR_ABI_VERSION
        code = self._lib.gpu_fusion_get_hdr_counters(self._handle, C.byref(block))
        if code:
            raise self._fail("gpu_fusion_get_hdr_counters", code)
        return _report_dict(block, COUNTER_FIELDS)

    def close(self):
        if self._handle is None:
            return
        self._lib.gpu_fusion_close(self._handle)
        self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()
        return False
