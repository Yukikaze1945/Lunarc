"""Evidence: what rr_cpu_has_avx2() answers here, and what each version of it means.

    python tests/unit/avx2_probe_report.py --out avx2_probe.json

Two separate questions, and one machine can only answer one of them.

Measured: the loaded DLL's answer on this CPU under this OS, cross-checked against NumPy,
which walks CPUID and XGETBV itself when it is imported. Two independent implementations
agreeing is the strongest statement a single machine can make about a capability probe.

Not measurable here: the old probe's false positive. It tested leaf 7 EBX bit 8, which is
ERMS, and this Ryzen 7 5700X has ERMS and AVX2 both, so old and new answer 1 and agree. A
wrong bit needs a CPU where it differs from bit 5 to show up, and none is available. So the
rest of the report is a semantics table, derived from the bit each version reads and the
conditions each version checks - both extracted from source text, the current file and the
pre-fix file out of git - over feature rows this machine cannot provide. Those rows say what
the code means, not what this CPU is.
"""

import argparse
import json
import os
import re
import subprocess
import sys

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CURRENT_CPP = os.path.join(_ROOT, "native", "residual_bridge.cpp")
# The commit this cleanup round started from, whose probe is the thing being fixed.
LEGACY_REV = "7411d97"
LEGACY_PATH = "native/residual_bridge.cpp"
BUILD_BAT = os.path.join(_ROOT, "native", "build_residual.bat")
_REGISTERS = {0: "EAX", 1: "EBX", 2: "ECX", 3: "EDX"}
# (a condition the body can test, the row capability that condition has to pass)
_CONDITIONS = (("osxsave_bit27", "osxsave"), ("avx_bit28", "hw_avx"),
               ("xgetbv_saves_ymm", "os_saves_ymm"))


def _probe_body(text):
    start = text.index("rr_cpu_has_avx2(void)")
    return text[start:text.index("\n}", start) + 2]


def _legacy_text():
    """The pre-fix source straight out of git, so this needs no scratch build."""
    try:
        out = subprocess.run(["git", "--no-pager", "show", f"{LEGACY_REV}:{LEGACY_PATH}"],
                             capture_output=True, text=True, cwd=_ROOT, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    try:
        return out.stdout if _probe_body(out.stdout) else None
    except ValueError:
        return None


def _answers_with(body):
    """The bit test a probe body ends up returning, as leaf/register/bit."""
    for segment in body.split("return")[1:]:
        tested = segment.split(";")[0]
        bit = re.search(r"1\s*<<\s*(\d+)", tested)
        register = re.search(r"info\[(\d)\]", tested)
        if bit and register:
            prior = body[:body.index(tested)]
            leaves = re.findall(r"__cpuid(?:ex)?\(\s*info\s*,\s*(\d+)", prior)
            return {"leaf": int(leaves[-1]) if leaves else None,
                    "register": _REGISTERS[int(register.group(1))],
                    "bit": int(bit.group(1)),
                    "expression": " ".join(tested.split())}
    return None


def _gates(body):
    """Which conditions a probe body actually tests."""
    return {
        "leaf7_exists": bool(re.search(r"__cpuid\s*\(\s*info\s*,\s*0\s*\)", body)),
        "osxsave_bit27": "(1 << 27)" in body,
        "avx_bit28": "(1 << 28)" in body,
        "xgetbv_saves_ymm": "_xgetbv(0)" in body and "0x6" in body,
        "compile_time_avx2_macro": bool(
            re.search(r"^\s*#\s*(if|elif).*__AVX2__", body, re.MULTILINE)),
    }


# Bit 5 of leaf 7 EBX is AVX2, bit 8 is ERMS. Row 1 is this machine; the rest are what a
# capability probe has to survive and this machine cannot supply.
FEATURE_ROWS = [
    {"cpu": "measured here: AMD Ryzen 7 5700X (Zen 3)", "leaf7_exists": True,
     "osxsave": True, "hw_avx": True, "os_saves_ymm": True, "avx2": True, "erms": True},
    {"cpu": "AMD Phenom II / Core 2 class: ERMS present, AVX2 absent",
     "leaf7_exists": True, "osxsave": True, "hw_avx": True, "os_saves_ymm": True,
     "avx2": False, "erms": True},
    {"cpu": "Intel Haswell class under an OS that never enabled YMM state saving",
     "leaf7_exists": True, "osxsave": True, "hw_avx": True, "os_saves_ymm": False,
     "avx2": True, "erms": True},
    {"cpu": "pre-leaf-7 CPU (early x86-64)", "leaf7_exists": False, "osxsave": False,
     "hw_avx": False, "os_saves_ymm": False, "avx2": False, "erms": False},
]


def _would_execute(row):
    """The truth a probe is supposed to report: could AVX2 code run on this row."""
    return bool(row["avx2"] and row["hw_avx"] and row["osxsave"] and row["os_saves_ymm"])


def _answer(row, gates, bit):
    """What a probe checking these conditions and this one bit answers for this row.

    A condition the body does not test is simply not applied, which is how the table shows
    what a missing check costs rather than what an ideal implementation would say. None
    means the code has no determinate answer at all.
    """
    if gates.get("leaf7_exists") and not row["leaf7_exists"]:
        return 0
    if not gates.get("leaf7_exists") and not row["leaf7_exists"]:
        # CPUID with a leaf above the highest one the CPU implements returns whatever the
        # top leaf holds, so reading the feature bit there is not a no - it is undefined.
        return None
    for gate, capability in _CONDITIONS:
        if gates.get(gate) and not row[capability]:
            return 0
    if bit == 5:
        return 1 if row["avx2"] else 0
    if bit == 8:
        return 1 if row["erms"] else 0
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="", help="write the report here")
    args = parser.parse_args(argv)

    import residual_native
    with open(CURRENT_CPP, encoding="utf-8") as handle:
        current_text = handle.read()
    current_body = _probe_body(current_text)
    current = {"source": os.path.relpath(CURRENT_CPP, _ROOT),
               "answers_with": _answers_with(current_body), "gates": _gates(current_body),
               "probe_answer_on_this_machine": bool(residual_native.avx2_available())}

    legacy = {"source": f"git {LEGACY_REV}:{LEGACY_PATH}"}
    legacy_text = _legacy_text()
    if legacy_text is None:
        legacy["gates"] = {"leaf7_exists": False, "osxsave_bit27": False,
                           "avx_bit28": False, "xgetbv_saves_ymm": False,
                           "compile_time_avx2_macro": True}
        legacy["answers_with"] = {"bit": 8, "register": "EBX", "leaf": 7,
                                  "expression": "(info[1] & (1 << 8)) != 0"}
        legacy["source_unavailable"] = (
            f"git show {LEGACY_REV}:{LEGACY_PATH} did not work here, so the legacy columns "
            f"below are the pre-fix reading as recorded in the fix's commit message")
    else:
        legacy_body = _probe_body(legacy_text)
        legacy["answers_with"] = _answers_with(legacy_body)
        legacy["gates"] = _gates(legacy_body)

    from numpy._core._multiarray_umath import __cpu_features__ as features
    # NumPy's own names: SSE42 rather than SSE4_2, and X86_V3, which is the architecture
    # level NumPy selected for itself and is only reachable with AVX2 plus the OS saving
    # YMM state - a second, independent way of answering the same question.
    numpy_features = {k: bool(features.get(k, False))
                      for k in ("AVX", "AVX2", "AVX512F", "FMA3", "SSE42", "X86_V3")}
    assert "SSE4_2" not in features, "NumPy changed its feature names; re-check this list"

    current_bit = (current["answers_with"] or {}).get("bit")
    legacy_bit = (legacy["answers_with"] or {}).get("bit")
    table = []
    for row in FEATURE_ROWS:
        truth = _would_execute(row)
        old = _answer(row, legacy["gates"], legacy_bit)
        new = _answer(row, current["gates"], current_bit)
        table.append({"cpu": row["cpu"],
                      "avx2_in_hardware": row["avx2"],
                      "os_saves_ymm_state": row["os_saves_ymm"],
                      "avx2_would_actually_execute": truth,
                      "old_probe_would_answer": old,
                      "new_probe_would_answer": new,
                      "old_correct": old == int(truth),
                      "new_correct": new == int(truth)})

    report = {
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "native_probe": current,
        "legacy_probe": legacy,
        "legacy_macro_branch_note": "the pre-fix source opened with '#if defined(__AVX2__) "
                                    "return 1', so a build compiled for AVX2 answered yes "
                                    "on every CPU. MSVC does not define that macro for the "
                                    "baseline x64 set this DLL is built for, so on the "
                                    "shipped build the answer came from the bit test under "
                                    "it, which was ERMS.",
        "numpy_runtime_features": numpy_features,
        "measured_agreement_with_numpy": (current["probe_answer_on_this_machine"]
                                          == numpy_features["AVX2"]),
        "this_machine_can_show_the_old_bug": bool(
            table[0]["old_probe_would_answer"] != table[0]["new_probe_would_answer"]),
        "rows_where_old_and_new_disagree": [r["cpu"] for r in table
                                            if r["old_probe_would_answer"]
                                            != r["new_probe_would_answer"]],
        "rows_where_the_old_probe_was_wrong": [r["cpu"] for r in table
                                               if not r["old_correct"]],
        "rows_where_the_new_probe_is_wrong": [r["cpu"] for r in table
                                              if not r["new_correct"]],
        "semantics_table": table,
        "table_note": "every row but the first is a feature set this CPU does not have. The "
                      "answers are computed from each version's own source text, so the "
                      "table says what that code means; it is not a measurement.",
        "kernels_compiled_with_arch_flag": "/arch" in open(
            BUILD_BAT, encoding="utf-8", errors="replace").read().lower(),
        "avx2_intrinsic_used_by_kernels": bool(re.search(r"_mm256|__m256", current_text)),
        "capability_reported_only_this_round": True,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + chr(10))
    print(text)
    return 0 if report["measured_agreement_with_numpy"] else 1


if __name__ == "__main__":
    sys.exit(main())
