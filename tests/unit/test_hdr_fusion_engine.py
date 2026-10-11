"""CPU/contract tests for the typed linear-scRGB fusion wrapper.

No DLL is loaded, no GPU is touched and no pixel is read back. What is checked is
the wrapper's own set of promises: that its ctypes mirrors are the header's structs
field for field, that it asks the solver for the typed group instead of inferring it
from the SDR ABI, that it refuses an SDR-shaped plane before a byte of it can be
copied as half colour, that the plans it marshals are the pipeline's own float32
arrays rather than a second copy of them, and that a stage code arrives with the
bridge's message attached instead of as a number.
"""
import ctypes
import os
import pathlib
import re

import pytest

import gpu_transfer_export as gte
import hdr_fusion_engine as fusion
import hdr_gpu_engine as hdr


HEADER = pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent.parent \
    / "native" / "gpu_multiscale_fusion_bridge.h"

# The x64 layout the compiled bridge's struct_size guard requires.
EXPECTED_SIZES = {
    "GpuFusionHdrOptions": 16,
    "GpuFusionHdrObservation": 24,
    "GpuFusionHdrReport": 72,
    "GpuFusionHdrCounters": 48,
}
MIRRORS = {
    "GpuFusionHdrOptions": fusion._HdrOptions,
    "GpuFusionHdrObservation": fusion._HdrObservation,
    "GpuFusionHdrReport": fusion._HdrReport,
    "GpuFusionHdrCounters": fusion._HdrCounters,
}
FULL = (64, 48)
WORKING = [(32, 24), (16, 12)]


def header_fields(name):
    """The field names of one typedef in the native header, in declaration order,
    comments removed and comma-separated declarators expanded. Parsed from the
    header the bridge compiles against, so a renamed, split or reordered field in
    either place shows up here rather than as a garbage struct."""
    text = HEADER.read_text(encoding="utf-8")
    match = re.search(r"typedef struct " + name + r" \{(.*?)\} " + name + r";",
                      text, re.S)
    assert match, f"native/gpu_multiscale_fusion_bridge.h declares no {name}"
    body = re.sub(r"/\*.*?\*/", "", match.group(1), flags=re.S)
    fields = []
    for statement in body.split(";"):
        statement = statement.strip()
        if not statement:
            continue
        for declarator in statement.split()[1:]:
            fields.append(declarator.strip(",").split("[")[0])
    return fields


def planes(*, cuda_context=0x9999, full=FULL, working=WORKING):
    """The descriptors an HDR chain really hands over: the full-resolution master
    and one signed working residual per layer, all rgba16f_cuda."""
    master = hdr.DevicePlane(fmt="rgba16f_cuda",
        pixel_format="R16G16B16A16_FLOAT (scRGB)", width=full[0], height=full[1],
        cuda_context=cuda_context, device_ptr=0x1000,
        pitch_bytes=max(1, full[0] * 8), bytes_per_pixel=8)
    observations = [hdr.DevicePlane(fmt="rgba16f_cuda",
        pixel_format="R16G16B16A16_FLOAT (signed working residual, linear scRGB; "
                           "alpha 0.0 padding)", width=width, height=height,
        cuda_context=cuda_context, device_ptr=0x9000 + 0x1000 * index,
        pitch_bytes=max(1, width * 8), bytes_per_pixel=8)
        for index, (width, height) in enumerate(working)]
    return master, observations


class FakeFusion:
    """A stand-in for gpu_multiscale_fusion_bridge.dll.

    The typed group is present by default and can be taken away symbol by symbol,
    because the wrapper has to ask for it rather than infer it from the SDR ABI it
    also speaks. Every SDR entry is recorded too, so a test can prove the typed path
    never quietly reaches for one.
    """

    def __init__(self, hdr_abi=1, abi=1, missing=(), last_error=b"solve stopped",
                 open_code=0, process_code=0, fail_symbol=None,
                 output_format=10, output_state=0x800, output_size=FULL,
                 unrepresentable=0, handle=0xF0510, texture=0x2000, fence=0x3000,
                 fence_value=41, iterations=6, converged=1,
                 counters=None):
        self._hdr_abi, self._abi = hdr_abi, abi
        self._missing = set(missing)
        self._last_error = last_error
        self._open_code, self._process_code = open_code, process_code
        self._fail_symbol = fail_symbol
        self._report = {"output_format": output_format,
                        "output_state": output_state, "converged": converged,
                        "iterations": iterations,
                        "unrepresentable_components": unrepresentable,
                        "output_width": output_size[0],
                        "output_height": output_size[1],
                        "initial_residual_norm": 4.25,
                        "final_residual_norm": 0.00425,
                        "relative_residual": 0.001,
                        "source_device_copy_bytes": FULL[0] * FULL[1] * 8,
                        "observation_device_copy_bytes": sum(
                            width * height * 8 for width, height in WORKING),
                        "scalar_readback_bytes": 512}
        self._counters = dict(process_count=2, source_device_copy_bytes=2 * 24576,
            observation_device_copy_bytes=12288, scalar_readback_bytes=1024,
            unrepresentable_components=0, **(counters or {}))
        self._out = {"texture": texture, "fence": fence, "fence_value": fence_value,
                     "handle": handle}
        self.calls = []
        self.close_count = 0

    def __getattr__(self, name):
        if name in self._missing or name.startswith("_"):
            raise AttributeError(name)

        def call(*args):
            return self._dispatch(name, args)
        return call

    def _dispatch(self, symbol, args):
        self.calls.append((symbol, args))
        if symbol == "gpu_fusion_abi_version":
            return self._abi
        if symbol == "gpu_fusion_hdr_abi_version":
            return self._hdr_abi
        if symbol == "gpu_fusion_last_error":
            return self._last_error
        if symbol == "gpu_fusion_close":
            self.close_count += 1
            return None
        if self._fail_symbol == symbol:
            return 1
        if symbol == "gpu_fusion_open_hdr":
            if not self._open_code:
                args[3]._obj.value = self._out["handle"]
            return self._open_code
        if symbol == "gpu_fusion_process_hdr":
            if self._process_code:
                return self._process_code
            args[6]._obj.value = self._out["texture"]
            args[7]._obj.value = self._out["fence"]
            args[8]._obj.value = self._out["fence_value"]
            for name, value in self._report.items():
                setattr(args[9]._obj, name, value)
            return 0
        if symbol == "gpu_fusion_get_hdr_counters":
            for name, value in self._counters.items():
                setattr(args[1]._obj, name, value)
            return 0
        return 0

    def set_output(self, **fields):
        self._report.update(fields)

    def set_out(self, **fields):
        self._out.update(fields)


def _layers(plans=None):
    plans = plans or [gte.TapPlans(FULL[0], FULL[1], width, height)
                      for width, height in WORKING]
    return [{"plan": plan, "dims": dims, "weight": weight} for plan, dims, weight
            in zip(plans, WORKING, (1.0, 0.5))]


def _bridge(fake=None, **overrides):
    kwargs = dict(d3d12_device=0xBEEF, full_width=FULL[0], full_height=FULL[1],
                  layers=_layers(), beta=0.05, residual_strength=1.0,
                  relative_tolerance=0.01, max_iterations=32)
    kwargs.update(overrides)
    return fusion.HdrFusionBridge(library=fake or FakeFusion(), **kwargs)


# ---------------------------------------------------------------------------
# the mirrors are the header's, not an approximation of one
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,size", sorted(EXPECTED_SIZES.items()))
def test_typed_struct_sizes_match_the_compiled_abi(name, size):
    assert ctypes.sizeof(MIRRORS[name]) == size


@pytest.mark.parametrize("name", sorted(EXPECTED_SIZES))
def test_typed_mirrors_are_the_headers_fields_in_order(name):
    assert [field for field, _ in MIRRORS[name]._fields_] == header_fields(name)


def test_typed_structs_sized_like_the_mirrors_reach_the_bridge():
    """The guard the bridge checks is the mirror's own size, not a constant the
    wrapper carried from an older header."""
    fake = FakeFusion()
    _bridge(fake)
    sent = [a for s, a in fake.calls if s == "gpu_fusion_open_hdr"][0]
    assert sent[2]._obj.struct_size == ctypes.sizeof(fusion._HdrOptions)
    assert sent[2]._obj.abi_version == fusion.GPU_FUSION_HDR_ABI_VERSION
    assert sent[2]._obj.pixel_bytes == 8 and sent[2]._obj.reserved == 0
    assert sent[1]._obj.struct_size == ctypes.sizeof(gte._FusionOptions)


# ---------------------------------------------------------------------------
# the typed group is asked for, never inferred
# ---------------------------------------------------------------------------

def test_probe_reports_the_group_as_missing_by_symbol():
    answer = fusion.probe_hdr_fusion(FakeFusion(missing=["gpu_fusion_process_hdr"]))
    assert not answer["supported"] and answer["missing"] == ["gpu_fusion_process_hdr"]
    assert "typed linear" in answer["reason"]


def test_probe_refuses_a_group_of_another_version():
    answer = fusion.probe_hdr_fusion(FakeFusion(hdr_abi=2))
    assert not answer["supported"] and answer["version"] == 2


def test_probe_refuses_an_sdr_abi_it_shares_the_options_layout_with():
    answer = fusion.probe_hdr_fusion(FakeFusion(abi=2))
    assert not answer["supported"] and "GpuFusionOptions" in answer["reason"]


def test_probe_accepts_a_bridge_that_speaks_both_contracts():
    assert fusion.probe_hdr_fusion(FakeFusion()) == {
        "supported": True, "version": 1, "missing": [], "reason": ""}


# ---------------------------------------------------------------------------
# opening: the contract is checked before the bridge is asked
# ---------------------------------------------------------------------------

def test_open_sends_the_solvers_own_options_and_the_layer_plans():
    fake = FakeFusion()
    _bridge(fake)
    options = [a for s, a in fake.calls if s == "gpu_fusion_open_hdr"][0][1]._obj
    assert options.full_width, options.full_height == FULL
    assert options.layer_count == 2
    assert options.beta == pytest.approx(0.05, abs=1e-7)
    assert options.residual_strength == pytest.approx(1.0, abs=1e-7)
    assert options.max_iterations == 32
    assert [options.layers[i].weight for i in range(2)] == [
        pytest.approx(1.0), pytest.approx(0.5)]
    assert [(options.layers[i].width, options.layers[i].height) for i in range(2)] \
        == WORKING


def test_the_plans_marshalled_are_the_pipelines_own_arrays_not_a_copy():
    """One plan set, proved by pointer identity: the wrapper hands the bridge the
    ``tap_tables`` view the SDR chain already builds, so the fused HDR frame and a
    fused SDR frame are statements about the same filter."""
    plans = [gte.TapPlans(FULL[0], FULL[1], width, height)
             for width, height in WORKING]
    options = fusion.layer_options(full_width=FULL[0], full_height=FULL[1],
                                   layers=_layers(plans))
    def address(pointer):
        return ctypes.cast(pointer, ctypes.c_void_p).value
    first = plans[0].working_color
    assert address(options.layers[0].horizontal.indices) == \
        address(first.horizontal_indices)
    assert address(options.layers[0].vertical.weights) == \
        address(first.vertical_weights)
    assert options.layers[0].horizontal.taps == first.taps
    baseline = plans[0].upscale
    assert address(options.baseline_horizontal.indices) == \
        address(baseline.horizontal_indices)
    assert options.baseline_horizontal.dst_len == FULL[0]


@pytest.mark.parametrize("bad", [4, 2, 16, 0])
def test_a_sample_width_that_is_not_a_half_plane_is_refused(bad):
    with pytest.raises(fusion.HdrFusionError, match="R16G16B16A16_FLOAT"):
        _bridge(pixel_bytes=bad)


@pytest.mark.parametrize("count,why", [(1, "2 to 3"), (4, "2 to 3"),
                                       (0, "2 to 3")])
def test_a_layer_count_the_bridge_cannot_hold_is_refused(count, why):
    layers = [{"plan": gte.TapPlans(FULL[0], FULL[1], width, height),
               "dims": (width, height), "weight": 1.0}
              for width, height in (WORKING * 2)[:count]]
    with pytest.raises(fusion.HdrFusionError, match=why):
        fusion.layer_options(full_width=FULL[0], full_height=FULL[1],
                             layers=layers)


@pytest.mark.parametrize("overrides,why", [
    (dict(beta=float("nan")), "finite"),
    (dict(beta=0.0), "above 0"),
    (dict(residual_strength=float("inf")), "finite"),
    (dict(residual_strength=float("nan")), "finite"),
    (dict(residual_strength=-0.5), "at or above 0"),
    (dict(relative_tolerance=1.0), "below 1"),
    (dict(relative_tolerance=0.0), "above 0"),
    (dict(max_iterations=0), "outside 1..1000"),
    (dict(max_iterations=4000), "outside 1..1000"),
    (dict(full_width=1), "outside 2..16384"),
])
def test_a_scalar_the_solver_could_not_use_is_refused_before_open(overrides, why):
    fake = FakeFusion()
    with pytest.raises(fusion.HdrFusionError, match=why):
        _bridge(fake, **overrides)
    assert not [s for s, _ in fake.calls if s == "gpu_fusion_open_hdr"]


def test_zero_residual_strength_is_a_request_and_not_a_refusal():
    """``0`` means "solve, store the master alone", which the single-layer HDR
    chain has always accepted; the strength is not an inverse gain that a divide
    would choke on, and beta, the weights and the tolerance stay strictly
    positive."""
    fake = FakeFusion()
    bridge = _bridge(fake, residual_strength=0.0)
    options = [a for s, a in fake.calls if s == "gpu_fusion_open_hdr"][0][1]._obj
    assert options.residual_strength == 0.0
    master, observations = planes()
    assert bridge.process(cuda_context=0x9999, master=master,
                          observations=observations)["converged"]


def test_the_tolerance_a_solve_is_judged_by_is_the_one_it_was_sent():
    """The report is compared against the float32 that reached the bridge, so a
    caller asking for a tolerance that float32 cannot hold is judged by the one it
    actually got."""
    bridge = _bridge(FakeFusion(), relative_tolerance=0.001)
    assert bridge.relative_tolerance == ctypes.c_float(0.001).value


def test_a_layer_larger_than_the_full_extent_is_refused():
    with pytest.raises(fusion.HdrFusionError, match="larger than the full"):
        _bridge(layers=[{"plan": gte.TapPlans(64, 48, 64, 48), "dims": (128, 24),
                         "weight": 1.0},
                        {"plan": gte.TapPlans(64, 48, 16, 12), "dims": (16, 12),
                         "weight": 0.5}])


def test_open_failure_carries_the_stage_meaning_and_the_bridge_message():
    fake = FakeFusion(open_code=3, last_error=b"no RGBA16F row transport")
    with pytest.raises(fusion.HdrFusionError) as caught:
        _bridge(fake)
    message = str(caught.value)
    assert "reported 3" in message and "RGBA16F row transport" in message
    assert "interop build" in message


def test_a_successful_open_with_a_null_handle_is_refused_not_adopted():
    fake = FakeFusion(handle=0)
    with pytest.raises(fusion.HdrFusionError, match="no bridge handle"):
        _bridge(fake)


# ---------------------------------------------------------------------------
# the process call: typed planes in, typed store out
# ---------------------------------------------------------------------------

def test_process_sends_one_typed_observation_per_layer():
    fake = FakeFusion()
    bridge = _bridge(fake)
    master, observations = planes()
    bridge.process(cuda_context=0x9999, master=master, observations=observations)
    args = [a for s, a in fake.calls if s == "gpu_fusion_process_hdr"][0]
    assert args[0].value == 0xF0510 and args[1].value == 0x9999
    assert args[2].value == 0x1000 and args[3].value == FULL[0] * 8
    assert args[5] == 2
    descriptors = args[4]
    assert [descriptors[i].width for i in range(2)] == [width for width, _ in WORKING]
    assert [descriptors[i].residual_ptr for i in range(2)] == [0x9000, 0xA000]
    assert [descriptors[i].residual_pitch for i in range(2)] == [256, 128]


def test_the_typed_path_never_reaches_for_an_sdr_entry():
    fake = FakeFusion()
    master, observations = planes()
    _bridge(fake).process(cuda_context=0x9999, master=master,
                          observations=observations)
    assert not [s for s, _ in fake.calls
                if s in ("gpu_fusion_open", "gpu_fusion_process",
                         "gpu_fusion_get_transfer_counters")]


def test_process_returns_the_descriptor_set_composite_consumes():
    fake = FakeFusion()
    bridge = _bridge(fake)
    master, observations = planes()
    output = bridge.process(cuda_context=0x9999, master=master,
                            observations=observations)
    assert output["d3d12_resource"] == 0x2000 and output["d3d12_fence"] == 0x3000
    assert output["fence_value"] == 41
    assert output["dxgi_format"] == hdr.DXGI_R16G16B16A16_FLOAT
    assert (output["width"], output["height"]) == FULL
    assert output["iterations"] == 6 and output["converged"]
    assert output["report"]["source_device_copy_bytes"] == 64 * 48 * 8
    assert output["report"]["observation_device_copy_bytes"] == (32 * 24 + 16 * 12) * 8
    # The A1 consumer's own checks, run on the descriptor this returned.
    texture, fence, value = hdr.GpuHdrBridge._producer_texture(
        None, output, None, None, extent=FULL, extent_name="full",
        label="fused composite", role="composite import")
    assert (texture, fence, value) == (0x2000, 0x3000, 41)


@pytest.mark.parametrize("patch,why", [
    (dict(output_format=28), "R16G16B16A16_FLOAT"),
    (dict(output_state=0x40), "COPY_SOURCE"),
    (dict(output_width=32, output_height=24), "not the full extent"),
])
def test_a_success_that_is_not_a_typed_store_is_rejected(patch, why):
    fake = FakeFusion()
    fake.set_output(**patch)
    master, observations = planes()
    with pytest.raises(fusion.HdrFusionError, match=why):
        _bridge(fake).process(cuda_context=0x9999, master=master,
                              observations=observations)


@pytest.mark.parametrize("patch,why", [
    (dict(texture=0), "no texture"),
    (dict(fence=0), "no producer fence"),
    (dict(fence_value=0), "fence value 0"),
])
def test_a_success_that_handed_back_nothing_is_not_called_a_frame(patch, why):
    fake = FakeFusion()
    fake.set_out(**patch)
    master, observations = planes()
    with pytest.raises(fusion.HdrFusionError, match=why):
        _bridge(fake).process(cuda_context=0x9999, master=master,
                              observations=observations)


@pytest.mark.parametrize("patch,why", [
    (dict(converged=0), "without a converged"),
    (dict(relative_residual=0.05), "above the"),
    (dict(final_residual_norm=float("nan")), "finite non-negative"),
    (dict(initial_residual_norm=-1.0), "finite non-negative"),
    (dict(relative_residual=float("inf")), "finite non-negative"),
])
def test_a_success_whose_report_does_not_evidence_a_solve_is_not_a_frame(patch,
                                                                         why):
    """The zero code says the store completed; only the report says the picture in
    it was solved, so an unconverged or unmeasurable success is refused here rather
    than encoded as a frame."""
    fake = FakeFusion()
    fake.set_output(**patch)
    master, observations = planes()
    with pytest.raises(fusion.HdrFusionError) as caught:
        _bridge(fake).process(cuda_context=0x9999, master=master,
                              observations=observations)
    assert why in str(caught.value)
    assert not [s for s, _ in fake.calls if s == "gpu_fusion_close"]


def test_an_observation_per_layer_is_required():
    master, observations = planes()
    with pytest.raises(fusion.HdrFusionError, match="exactly one residual"):
        _bridge().process(cuda_context=0x9999, master=master,
                          observations=observations[:1])


def test_a_third_observation_for_a_two_layer_bridge_is_refused():
    master, observations = planes()
    with pytest.raises(fusion.HdrFusionError, match="2 layers"):
        _bridge().process(cuda_context=0x9999, master=master,
                          observations=observations + observations[:1])


def test_a_failed_solve_names_its_stage_and_says_the_bridge_is_poisoned():
    fake = FakeFusion(process_code=12)
    master, observations = planes()
    with pytest.raises(fusion.HdrFusionError) as caught:
        _bridge(fake).process(cuda_context=0x9999, master=master,
                              observations=observations)
    message = str(caught.value)
    assert "reported 12" in message and "iteration budget" in message
    assert "marked failed" in message


@pytest.mark.parametrize("code,expectation", [
    (1, "refused before writing any state"),
    (2, "refused before writing any state"),
    (3, "refused before writing any state"),
    (4, "refused before writing any state"),
    (5, "marked failed"),
    (6, "marked failed"),
    (7, "marked failed"),
    (8, "marked failed"),
    (13, "marked failed"),
    (16, "marked failed"),
])
def test_the_message_says_which_stages_actually_poison_the_bridge(code,
                                                                  expectation):
    """The bridge marks itself failed from the first copy onwards and not before,
    which the wrapper reports as it is. Either way the frame has no composite: a
    refusal to start is a fact about the handle, not permission to keep exporting.

    Read off native/gpu_multiscale_fusion_bridge.cpp's gpu_fusion_process_hdr, so a
    stage that changed its poisoning shows up here rather than as a caller relying
    on a state the bridge never entered.
    """
    fake = FakeFusion(process_code=code)
    master, observations = planes()
    with pytest.raises(fusion.HdrFusionError) as caught:
        _bridge(fake).process(cuda_context=0x9999, master=master,
                              observations=observations)
    message = str(caught.value)
    assert f"reported {code}" in message
    assert expectation in message
    assert ("marked failed" in message) is (code >= 5)


def test_a_poisoned_bridge_is_closed_not_retried():
    fake = FakeFusion(process_code=11)
    bridge = _bridge(fake)
    master, observations = planes()
    with pytest.raises(fusion.HdrFusionError):
        bridge.process(cuda_context=0x9999, master=master, observations=observations)
    bridge.close()
    assert fake.close_count == 1


# ---------------------------------------------------------------------------
# plane validation: nothing is reinterpreted, so nothing can be smuggled in
# ---------------------------------------------------------------------------

def test_an_rgba8_master_is_refused_rather_than_read_as_half_colour():
    master, observations = planes()
    master.format = "rgba8_cuda"
    with pytest.raises(hdr.HdrBridgeError, match="rgba16f_cuda"):
        _bridge().process(cuda_context=0x9999, master=master,
                          observations=observations)


def test_an_observation_at_the_wrong_extent_is_refused():
    master, observations = planes()
    observations[1].width, observations[1].height = 32, 24
    with pytest.raises(hdr.HdrBridgeError, match="does not match"):
        _bridge().process(cuda_context=0x9999, master=master,
                          observations=observations)


def test_an_observation_whose_pitch_cannot_hold_a_half_row_is_refused():
    master, observations = planes()
    observations[0].pitch_bytes = 32 * 4
    with pytest.raises(hdr.HdrBridgeError, match="under the 256 bytes"):
        _bridge().process(cuda_context=0x9999, master=master,
                          observations=observations)


def test_a_plane_from_another_cuda_context_is_refused():
    master, observations = planes()
    with pytest.raises(hdr.HdrBridgeError, match="CUDA context"):
        _bridge().process(cuda_context=0x7777, master=master,
                          observations=observations)


def test_a_missing_master_is_refused_before_anything_is_copied():
    _, observations = planes()
    with pytest.raises(hdr.HdrBridgeError, match="required"):
        _bridge().process(cuda_context=0x9999, master=None,
                          observations=observations)


# ---------------------------------------------------------------------------
# counters and lifetime
# ---------------------------------------------------------------------------

def test_counters_expose_only_fields_the_abi_carries():
    fake = FakeFusion()
    bridge = _bridge(fake)
    counters = bridge.counters()
    assert set(counters) == set(fusion.COUNTER_FIELDS)
    assert counters["process_count"] == 2
    assert counters["scalar_readback_bytes"] == 1024


def test_counters_report_a_counted_fallback_without_calling_it_a_failure():
    fake = FakeFusion(unrepresentable=7)
    master, observations = planes()
    output = _bridge(fake).process(cuda_context=0x9999, master=master,
                                   observations=observations)
    assert output["unrepresentable_components"] == 7


def test_close_is_idempotent_and_use_after_close_is_refused():
    fake = FakeFusion()
    bridge = _bridge(fake)
    bridge.close()
    bridge.close()
    assert fake.close_count == 1
    with pytest.raises(fusion.HdrFusionError, match="closed"):
        bridge.counters()


def test_the_context_manager_closes_on_the_error_path():
    fake = FakeFusion()
    master, observations = planes()
    with pytest.raises(RuntimeError, match="rejected upstream"):
        with fusion.HdrFusionBridge(d3d12_device=0xBEEF, full_width=FULL[0],
                                    full_height=FULL[1], layers=_layers(), beta=0.05,
                                    residual_strength=1.0, relative_tolerance=0.01,
                                    max_iterations=32, library=fake) as bridge:
            bridge.process(cuda_context=0x9999, master=master,
                           observations=observations)
            raise RuntimeError("the frame was rejected upstream")
    assert fake.close_count == 1
