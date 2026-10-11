"""Gate: rr_cpu_has_avx2() must answer about this machine, not about this build.

The function used to be

    #if defined(__AVX2__) -> 1
    else CPUID leaf 7 EBX bit 8

which is wrong twice over. The macro says what the compiler was allowed to emit, which
has nothing to do with where the DLL later runs - and bit 8 of leaf 7 EBX is ERMS
(enhanced REP MOVSB/STOSB), a string-operation optimisation that AMD and Intel ship on
CPUs with no AVX2 at all. The replaced probe therefore reported "AVX2 usable" on any ERMS
machine that also had leaf 7, and on this build the macro branch could never fire because
residual_bridge is compiled for the baseline x64 set.

What the replacement has to get right is checked here against an independent
implementation of the same question: NumPy walks CPUID and XGETBV itself at import time
and records the result in __cpu_features__. Two unrelated implementations agreeing on the
live machine is the evidence; the source guards below are what keep a later edit from
quietly going back to one bit and no OS check.
"""

import os
import re
import sys

import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

residual_native = pytest.importorskip(
    "residual_native", reason="residual_bridge extension missing")
if not residual_native.is_available():
    pytest.skip("residual_bridge could not be probed", allow_module_level=True)

BRIDGE_CPP = os.path.join(_ROOT, "native", "residual_bridge.cpp")
BUILD_BAT = os.path.join(_ROOT, "native", "build_residual.bat")


def _numpy_runtime_avx2():
    """NumPy's own answer, from the CPUID/XGETBV walk it did when it was imported."""
    from numpy._core._multiarray_umath import __cpu_features__ as features
    return bool(features.get("AVX2", False))


def _probe_body():
    """The text of rr_cpu_has_avx2, taken from the source that built the loaded DLL."""
    with open(BRIDGE_CPP, encoding="utf-8") as handle:
        text = handle.read()
    start = text.index("rr_cpu_has_avx2(void)")
    body = text[start:text.index("\n}", start) + 2]
    assert len(body) < 4000, "could not find the end of rr_cpu_has_avx2"
    return body


def test_probe_matches_numpy_runtime_detection():
    probed = residual_native.avx2_available()
    assert probed == _numpy_runtime_avx2(), (
        f"the bridge probe says AVX2={probed} but NumPy's own CPUID+XGETBV detection "
        f"says AVX2={_numpy_runtime_avx2()} on the same machine")
    assert probed is True, (
        "this CPU (Ryzen 7 5700X) has AVX2 and Windows enables YMM state saving, so a "
        "correct runtime probe has to return 1 here; 0 would mean the probe is broken")


def test_probe_answer_comes_from_the_cpu_not_from_the_build():
    """If the build cannot define __AVX2__, a truthful 1 can only have come from CPUID."""
    with open(BUILD_BAT, encoding="utf-8", errors="replace") as handle:
        build = handle.read()
    assert "/arch" not in build.lower(), (
        "build_residual.bat now compiles with /arch:AVX2 or similar; the source-level "
        "guard below would no longer exclude a compile-time macro as the answer's source")
    assert residual_native.avx2_available() == _numpy_runtime_avx2()
    # __AVX2__ is a GCC/Clang macro and MSVC only sets it under an explicit /arch:AVX2,
    # so on the shipped baseline build the old first branch could never have run. Only a
    # preprocessor test on it is a gate; the function's comment names it to explain why it
    # is not used.
    body = _probe_body()
    directive = re.search(r"^\s*#\s*(if|elif).*__AVX2__", body, re.MULTILINE)
    assert directive is None, (
        f"the probe gates on a compile-time macro again: {directive and directive.group(0)}")


def test_probe_walks_every_condition_the_instruction_needs():
    """AVX2 executes only if the CPU has it *and* the OS keeps YMM state for us."""
    body = _probe_body()
    assert re.search(r"__cpuid\s*\(\s*info\s*,\s*0\s*\)", body), (
        "the probe must first ask whether CPUID leaf 7 exists at all")
    assert "__cpuidex(info, 1, 0)" in body, "leaf 1 is where OSXSAVE and AVX live"
    assert "(1 << 27)" in body, "ECX bit 27 (OSXSAVE) gates whether XGETBV is usable"
    assert "(1 << 28)" in body, "ECX bit 28 (AVX) gates whether the hardware has AVX"
    assert "_xgetbv(0)" in body and "0x6" in body, (
        "XGETBV(0) bits 1 and 2 are how the OS says it saves XMM and YMM state; without "
        "them a CPU that has AVX2 still faults on the first VEX-encoded instruction")
    assert "__cpuidex(info, 7, 0)" in body, "the AVX2 bit is in leaf 7 subleaf 0"
    assert "(1 << 5)" in body, (
        "leaf 7 EBX bit 5 is AVX2; anything else being tested is not AVX2")


def test_probe_no_longer_mistakes_erms_for_avx2():
    """The specific old bug: bit 8 of leaf 7 EBX is ERMS, not AVX2."""
    body = _probe_body()
    assert "(1 << 8)" not in body, (
        "bit 8 of leaf 7 EBX is ERMS (enhanced REP MOVSB/STOSB). Testing it as if it were "
        "AVX2 makes the probe answer yes on CPUs that have string optimisations and no "
        "AVX2; the previous version of this function did exactly that")
    # And the returned value must be derived from the leaf-7 read, not an earlier one.
    avx2_line = [ln for ln in body.splitlines() if "(1 << 5)" in ln]
    assert len(avx2_line) == 1, f"expected one AVX2 bit test, found {avx2_line}"
    assert "info[1]" in avx2_line[0], "AVX2 must be read from EBX of leaf 7"


def test_probe_needs_no_plan_and_leaves_counters_alone():
    """Capability reporting is not frame work and must not be booked as anyone's work."""
    plan = residual_native.NativeResidualBackend(64, 48, 32, 24)
    try:
        before = plan.diagnostics()
        answers = {residual_native.avx2_available() for _ in range(50)}
        assert answers in ({True}, {False}), (
            f"the probe gave more than one answer to the same question on the same machine: "
            f"{answers}")
        after = plan.diagnostics()
        changed = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
        assert not changed, f"the capability probe moved this plan's counters: {changed}"
    finally:
        plan.close()
