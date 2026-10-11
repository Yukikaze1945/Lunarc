#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Who is allowed to run the transfer benchmark, and whether the machine was quiet when it did.

Why this file exists is one number. The previous round pooled twelve counterbalanced executions and
came out at 3.5804 against 3.5983 FPS - a 0.5% decision - while its own per-execution spreads between
two kept runs of the same order ran to tens of percent, and the wall time of the identical unit
suite moved between 66 s and 134 s across the same session. A comparison that small cannot be read
off a machine doing other things, and nothing in the evidence said whether it was.

So this module answers two questions before a single frame is timed, and records the answers as
artifacts:

  * **May I run?** `BenchmarkLock` takes an OS-level mandatory lock on
    `bench_out/.transfer_benchmark.lock` - `msvcrt.locking` on Windows, `fcntl.flock` on POSIX. Not a
    lockfile-existence check: those lie in both directions, failing when a crashed process left a
    file behind and passing when two processes happened to look at different paths. A contender is
    refused within five seconds and the reason names the owner that holds it (§6-§9). This module
    never kills another process, never deletes a live lock, and never continues an unlocked
    benchmark.

  * **Was it quiet?** `quiet_preflight` samples CPU, RAM and the GPU at 1 Hz for a thirty-second
    window and accepts the first window in which every threshold holds. The thresholds are the ones
    §12 fixed before any result was seen, and they are constants here rather than arguments, so a
    later caller cannot tune them towards the answer it wants. Up to ten minutes of windows may be
    spent waiting for one; a window that never passes stops the round as NOT_READY instead of
    running the benchmark anyway (§15). What it records is §20's `benchmark_environment.json`, field
    names included.

Everything a test needs to disagree with is injected - `sample_fn`, `sleep`, `monotonic` - so §22's
nine regressions can hand it a busy machine and an idle one without either existing on this host.

One rule needed measuring before it could be written, and it is worth stating plainly because §14 as
literally worded cannot be satisfied on this machine. `nvidia-smi --query-compute-apps` on this
Windows driver lists every process holding a GPU context, not the CUDA compute subset: measured at
idle, 38 rows - explorer, Chrome, the terminal, this editor - and every one of them reports its
memory as `[N/A]`, unchanged across samples eight seconds apart. A rule that fails the preflight for
any compute process outside the benchmark tree therefore fails on an idle machine. The policy below
keeps §14's recording exactly (the whole table goes into the artifact, `[N/A]` unconverted) and
narrows which rows are treated as a workload to the three that can distinguish a competitor from a
desktop: a process the driver actually accounts GPU memory for, a process whose image or command
line looks like build or media or Python work, or a process that joined the list partway through the
window. Everything else was classified `desktop_context`, and the classification of every row is
published so a reader can check it rather than trust it. What a desktop player or a game would still
have to hide from is §12's utilisation thresholds, which stay in force unchanged.

Round C §5 narrowed the forgiving half of that by name. A row became one of five classes -
`benchmark_descendant`, `repo_conflict`, `known_compute_workload`, `ambient_os_graphics_client`,
`unknown_external_gpu_client` - and only the OS component case could be excused for having *arrived*:
that needed an exact identity (image name from a one-entry table, the full path it has to be at, no
GPU memory the driver accounts for, no workload in its command line), because §5 forbids the shortcut
of treating anything under `C:\\Windows\\System32\\` as harmless. A row whose identity could not be read
was an `unknown_external_gpu_client`, which is a conflict, not an omission.

Three things then had to be measured rather than reasoned about, and all three changed the rule.

**A row's presence is not evidence of anything (Round C, follow-up policy).** The class above was
reached by two branches that never named a workload: a client that *arrived* during the window, and a
client whose identity could not be read. That is what stopped the round run at `f7d35df` (evidence
commit `eb61bbc`) over `ChatGPT Classic.exe` - a desktop window that had opened its GPU context a second
after the sampling started, reported `[N/A]` for memory like the other thirty-four rows, and was
condemned for being new.
Being new is not using the card. So a conflict now requires one of four positive facts about *that
process*, held in at least three consecutive 1 Hz samples (the same `CONTAMINATION_CONSECUTIVE_SAMPLES`
§18 already applies to a repo workload): the driver accounts a non-zero amount of its GPU memory; it is
seen on a compute or codec engine; its image, path or command line names a known compute workload; or
its own measured share of the GPU is at or above `PER_PROCESS_GPU_COMPUTE_PERCENT_AT_LEAST`. A row that
satisfies none of those is `ambient_or_unproven_gpu_client`: recorded with its pid, path, memory field
and arrival, never a reason to refuse a window or stop a batch. `ambient_os_graphics_client` stays as
the narrower claim §5 asked for - the one OS component excused *by exact identity* - and the
wording of the four criteria is what keeps this from becoming a name list: there is no ChatGPT entry,
no System32 entry, and no directory prefix anywhere in this file. A row whose identity cannot be read
is now unproven rather than guilty, which is a deliberate reversal of the sentence above it, and the
regression that pinned the old behaviour was rewritten to say so rather than deleted.

**The second fact is that `pmon`'s `type` column proves nothing here.** The per-process engine column
was measured on this host (`nvidia-smi pmon -c 1 -s u`, three samples, 105 rows): every client reports
`C+G`, the shell and the editor included, so a rule that read "C" as compute would call `explorer.exe`
a CUDA competitor. The engine column is therefore recorded and never used as the second criterion; what
is used instead is a non-dash `enc`/`dec`/`jpg`/`ofa` figure at or above the same frozen threshold, which
is a coding engine being occupied rather than a context existing. The `sm` and `mem` columns *are*
usable, and measured: the busiest windowing clients here sat at 1-8 % while the whole-machine threshold
for a quiet window is a 10 % median, which is how the per-process limit below was fixed.

**A child of the benchmark was being condemned as a stranger (Round C §2, second half).** In the batch
whose evidence is `eb61bbc`, three of its own processes - `ffmpeg.exe` as the child of a `python.exe`
that was in that sample's tree, and two others - were written as `repo_conflict`, because a reading used
a process list taken *before* the GPU table it judged. A child spawned in between is in the GPU table
and not in the list, so it has no parent, no creation time, and no membership. §2's `sample()` now
queries the card first and the process list second, which makes the list always at least as recent as
the table it is matched against, and the two orderings are covered by
`test_a_child_spawned_between_the_two_reads_is_attributed`, which advances a fake clock between the two
calls so the property survives a refactor of this body rather than only a comment. On top of that, a
row whose own identity is not in the snapshot can still be attributed by the two facts the OS does give
for it: its parent is in this same sample's tree, and it was created after the process at the root of
that tree. Both are recorded per row as `descendant_attribution`, so the forgiveness is checkable and
not a name.
"""

import argparse
import datetime
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import threading
import time

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, os.path.join(_ROOT, "src"), _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:  # Windows and POSIX take their locks through different calls; both are mandatory.
    import msvcrt  # type: ignore
except ImportError:  # pragma: no cover - the POSIX branch
    msvcrt = None
try:
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - the Windows branch
    fcntl = None

import psutil  # noqa: E402
import ctypes  # noqa: E402  # §6: the power plan and the job object are only readable through WinAPI

# §9: where the lock lives. `bench_out/` is ignored by git, so neither the lock nor its metadata can
# be committed by accident, and both are named so a reader who finds them knows who wrote them.
LOCK_FILENAME = ".transfer_benchmark.lock"
LOCK_METADATA_FILENAME = ".transfer_benchmark.lock.json"
LOCK_TIMEOUT_SECONDS = 5.0

# §12, fixed before any measurement this round will report. Written as constants and not as keyword
# arguments with defaults that a caller is expected to pass: the only way these change is a commit.
THRESHOLDS = {
    "cpu_median_percent_at_most": 20.0,
    "cpu_p95_percent_at_most": 35.0,
    "gpu_utilization_median_percent_at_most": 10.0,
    "gpu_utilization_p95_percent_at_most": 25.0,
    "available_ram_at_least_bytes": 8 * 1024 * 1024 * 1024,
}
# Every one of those five can also fail by not being measurable: a threshold a machine cannot report
# is a window that cannot be certified quiet, so an unreadable figure is its own failure reason and
# never a default of zero.

WINDOW_SECONDS = 30          # §11: one window is thirty samples at 1 Hz
SAMPLE_INTERVAL_SECONDS = 1.0
QUIET_WAIT_SECONDS = 600     # §15: ten minutes for one passing window, then stop
# §18, and now §5's second half as well: an outsider has to stay three consecutive 1 Hz samples to
# count, both for the monitor over a batch and for the window that is being certified. One number for
# both because they are the same claim about the machine; it sits here rather than beside the watcher
# because `judge_window` needs it too, and a default argument is evaluated where it is written.
CONTAMINATION_CONSECUTIVE_SAMPLES = 3

# §13: a repository workload is another process doing this project's kind of work. The names are
# the ones a build, a test run or a transcode would show up as.
CONFLICTING_EXECUTABLES = ("python.exe", "python", "pytest", "pytest.exe", "ffmpeg.exe",
                           "ffprobe.exe", "cl.exe", "link.exe", "cmake.exe", "ninja.exe",
                           "msbuild.exe")
CONFLICTING_COMMAND_TOKENS = ("collect_transfer", "benchmark_e2e", "pytest", "tests/unit",
                              "nvidia-smi", "nvof", "residual_bridge", "dlss")

# §14 policy, part 2: an image or command line that says this GPU client is doing work rather than
# drawing a window. Kept separate from CONFLICTING_EXECUTABLES because a non-repo CUDA program is
# still a competing GPU load even though it has nothing to do with this repository.
GPU_WORKLOAD_TOKENS = CONFLICTING_EXECUTABLES + (
    "ffmpeg", "x264", "x265", "nvenc", "nvdec", "cuda", "cudart", "nvof", "tensorrt",
    "trtexec", "blender", "handbrake", "obs", "ollama", "llama", "stable-diffusion",
    "comfyui", "torch", "build", "cmake", "ninja", "cl.exe", "link.exe",
    # §5 names these as workloads that must still hard fail, and this predicate is the only place
    # their image name is read: `scsp_dlss_smoke.exe` belongs to no checkout of this repository, so
    # without the tokens it would be an unexplained long-lived client and the window would pass.
    # Matched against the image name alone, so the `dlss` in this repository's own directory name
    # cannot reach it.
    "dlss", "scsp", "smoke")

# The sampler's own query tool, named so a monitor can never contaminate a batch with itself. On
# this driver `--query-compute-apps` does not list nvidia-smi at all - measured as the same forty
# rows in every sample of a five-minute window, none of them named nvidia-smi - so nothing here
# depends on it. A driver that did list its own query would otherwise show a fresh pid arriving every
# second, which is exactly what the mid-window-arrival rule looks for, and the batch would stop
# because the instrument was reading the machine.
GPU_QUERY_TOOL_NAMES = ("nvidia-smi", "nvidia-smi.exe", "nvsmi")

# §5 (Round C) plus the follow-up policy: the names a GPU client is classified by, and which of them
# stop a window. The round before this one had one name for a workload and one for everything else, and
# this driver's `--query-compute-apps` answers with thirty-eight long-lived windowing clients - the
# shell, the browser, the terminal, a music app - none of which is doing compute. So the names below say
# *why* a row was forgiven or refused, which is the part that can be audited.
#
# `desktop_context`, the sixth name as of `3bca05d`, is gone rather than renamed in place: it was a
# bucket for "we could find no reason", and `ambient_or_unproven_gpu_client` says the same thing from
# the direction that matters - nothing about this row demonstrates compute, so its pid, path, memory
# field and arrival are written down and it is left alone.
GPU_CLIENT_CLASSIFICATIONS = ("benchmark_descendant", "repo_conflict", "known_compute_workload",
                              "ambient_os_graphics_client", "unknown_external_gpu_client",
                              "ambient_or_unproven_gpu_client")
GPU_CLIENT_CONFLICT_CLASSIFICATIONS = ("repo_conflict", "known_compute_workload",
                                       "unknown_external_gpu_client")
# The two names that may never refuse a window or contaminate a batch, however long the process stays.
GPU_CLIENT_NON_TRIGGERING_CLASSIFICATIONS = ("ambient_os_graphics_client",
                                             "ambient_or_unproven_gpu_client")

# The four positive facts that may make an outside GPU client a conflict, named so an artifact can say
# which one it relied on. Each is a property of the process rather than of the moment sampling happened
# to start, and none of them is "its name was not in a list somebody approves of".
EXTERNAL_GPU_COMPUTE_CRITERIA = ("driver_accounts_gpu_memory_in_use",
                                 "codec_engine_occupied",
                                 "image_path_or_command_names_a_compute_workload",
                                 "per_process_gpu_utilisation_at_or_above_the_frozen_limit")

# Criterion four, fixed here before the next round measures anything: one process holding at least this
# percent of the card, as `nvidia-smi pmon -s u` reports it for that pid, is a competitor even though
# the whole-machine gate might still be satisfied. Measured on this host over three samples and 105
# rows: the windowing clients that report any `sm` at all sit between 1 % and 8 %, and a quiet window
# already caps the *machine* at a 10 % median and a 25 % p95 - so a single process at 15 % is outside
# the ambient band while still being only a sixth of the card. The only way it changes is a commit.
PER_PROCESS_GPU_COMPUTE_PERCENT_AT_LEAST = 15.0

# The `pmon` engine columns. `type` is recorded and never used as a criterion: every client on this
# driver reports `C+G`, the shell included, so it names a context rather than work.
PMON_UTILISATION_COLUMNS = {"sm": "gpu_sm_utilisation_percent",
                            "mem": "gpu_memory_utilisation_percent",
                            "enc": "gpu_encoder_utilisation_percent",
                            "dec": "gpu_decoder_utilisation_percent",
                            "jpg": "gpu_jpeg_utilisation_percent",
                            "ofa": "gpu_ofa_utilisation_percent"}
PMON_CODEC_ENGINE_COLUMNS = ("enc", "dec", "jpg", "ofa")
# `-c 1` is one sample period, and this binary answers it in about a tenth of a second (measured: 0.097 s
# against 0.061 s for the compute-apps query), so a 1 Hz sample can carry it without the window drifting.
PMON_UTILISATION_ARGS = ("pmon", "-c", "1", "-s", "u")

# The tokens read against a *path or command line*. Narrower than `GPU_WORKLOAD_TOKENS` on purpose:
# those are matched against an image name, where a short token is safe, while the same three letters
# inside a directory name accuse a great deal - `Observations`, `distributor`, `mixed-build` - and every
# false hit is a refused window. Matched at word boundaries by `names_a_compute_workload`, with this
# repository's own path taken out first for the same reason `looks_like_a_workload` takes it out.
GPU_COMPUTE_COMMAND_TOKENS = ("cuda", "cudart", "nvof", "nvenc", "nvdec", "nvjpeg", "nvcodec",
                              "tensorrt", "trtexec", "ffmpeg", "ffprobe", "avcodec", "x264", "x265",
                              "ollama", "llama", "stable-diffusion", "comfyui", "torch", "pytest",
                              "collect_transfer", "residual_bridge", "scsp_dlss_smoke",
                              "handbrake", "blender", "obs64", "smoke", "dlss")
# `benchmark` is deliberately *not* in that list even though it is the word this whole suite is about:
# an editor opened on a file called `benchmark_notes.md` is not a second benchmark, and every false hit
# here refuses a window. The work this token would name is already caught twice over - by the image-name
# rule above, and by §13's repository-workload rule, which reads the same command line against this
# checkout's own path.

# §5's ambient class is an exact-identity test over a one-entry table, and the entry is the process
# that produced this round's proven false alarm: `C:\Windows\System32\ShellHost.exe` restarted mid
# batch, looked like an arrival, and stopped a run that had nothing arriving. There is deliberately
# no `C:\Windows\System32\*` in this file, and no plan to add one: a directory prefix forgives every
# executable that happens to live inside it, including a competing DLSS smoke test, and §5 forbids
# exactly that. The expected path is derived from the Windows directory the machine reports, so the
# match is one full file name against one full file name.
AMBIENT_OS_GRAPHICS_EXECUTABLES = ("shellhost.exe",)

# §5's last condition: the command line has to name none of the work an ambient shell component has
# no business doing. Matching these tokens is what makes "it is called ShellHost.exe" insufficient on
# its own - a reinterpreted name with a CUDA payload in its command line is not forgiven by §5.
AMBIENT_DISQUALIFYING_COMMAND_TOKENS = ("cuda", "dlss", "benchmark", "python", "ffmpeg", "smoke",
                                        "nvof", "nvenc", "nvdec", "pytest", "collect_transfer",
                                        "tests/unit")

# nvidia-smi answers "N/A" for fields a driver does not expose. Those become None here and are named
# in the sample's `unavailable_fields`; they never become 0, because 0 reads as "idle" and N/A means
# "nobody knows", which are different sentences (§11, §14).
NOT_REPORTED = ("n/a", "na", "[n/a]", "", "not supported", "undefined")

# §6: the word a field of the audit carries when the machine would not answer it. Distinct from
# `measured`, and never a number.
UNAVAILABLE = "unavailable"

GPU_FIELD_QUERY = ("utilization.gpu,utilization.memory,temperature.gpu,clocks.current.graphics,"
                   "clocks.current.sm,clocks.current.memory,power.draw,pstate,"
                   "utilization.encoder,utilization.decoder")
GPU_FIELD_NAMES = ("gpu_utilization_percent", "gpu_memory_utilization_percent",
                   "gpu_temperature_celsius", "gpu_graphics_clock_mhz", "gpu_sm_clock_mhz",
                   "gpu_memory_clock_mhz", "gpu_power_draw_watts", "gpu_pstate",
                   "gpu_encoder_utilization_percent", "gpu_decoder_utilization_percent")


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _number(text):
    """A reported figure as a float, or None when it was not reported. Never 0 for unknown."""
    if text is None:
        return None
    cleaned = str(text).strip().strip("[]")
    if cleaned.lower() in NOT_REPORTED:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _reported(text):
    return _number(text) is not None


# --------------------------------------------------------------------------------------------
# The lock (§6-§9)
# --------------------------------------------------------------------------------------------

class BenchmarkLockBusy(Exception):
    """Another holder has the lock. Named so a driver can turn it into `benchmark_lock_busy`."""

    def __init__(self, owner, waited_seconds):
        super().__init__(f"the transfer benchmark lock is held by {owner.get('pid') or 'unknown'} "
                         f"on {owner.get('hostname') or 'unknown host'} after "
                         f"{waited_seconds:.2f}s")
        self.owner = owner
        self.waited_seconds = waited_seconds


def read_lock_metadata(metadata_path):
    """The last owner's record, as diagnostics. A torn or absent file is `unknown`, not a failure."""
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            doc = json.load(handle)
        return doc if isinstance(doc, dict) else {"unparseable": str(doc)[:200]}
    except FileNotFoundError:
        return None
    except Exception as exc:  # a half-written record still says who was here
        return {"unparseable": type(exc).__name__, "raw_first_200": ""}


class BenchmarkLock:
    """A mandatory, per-process lock over the benchmark: one holder, or a refusal in five seconds.

    The lock is the byte range of `bench_out/.transfer_benchmark.lock`, held by the OS for as long
    as the descriptor is open. The JSON beside it records who took it and is *diagnostics only*:
    nothing about correctness depends on being able to read it, because a metadata file can be
    stale, truncated or deleted while its owner is still running. When a contender is refused, the
    metadata is what names the owner to the reader - and if the metadata is gone, the refusal still
    happens, because the OS lock does not care what the JSON says.

    Releasing does not delete either file. Deleting a lock file a waiter is blocked on is how a
    lock becomes advisory again.
    """

    POLL_SECONDS = 0.05

    def __init__(self, root=_ROOT, timeout_seconds=LOCK_TIMEOUT_SECONDS, code_under_test=None,
                 command=None, protocol=None, monotonic=time.monotonic, sleep=time.sleep):
        self.root = os.path.abspath(root)
        self.path = os.path.join(self.root, "bench_out", LOCK_FILENAME)
        self.metadata_path = os.path.join(self.root, "bench_out", LOCK_METADATA_FILENAME)
        self.timeout_seconds = float(timeout_seconds)
        self.monotonic = monotonic
        self.sleep = sleep
        self.fd = None
        self.acquired_at_utc = None
        self.owner = None
        self.waited_seconds = 0.0
        self.held_for_seconds = 0.0
        self.record = {"pid": os.getpid(), "started_utc": None,
                       "code_under_test": code_under_test, "command": command or " ".join(sys.argv),
                       "hostname": socket.gethostname(), "protocol": protocol,
                       "lock_mechanism": ("msvcrt.locking" if msvcrt is not None
                                          else "fcntl.flock")}

    # -- the two OS calls, kept small so a reader can see there is no existence check in here ----
    # `self.fd` is a raw file descriptor: `os.open` returns an int, and both locking calls want that
    # number rather than a buffered Python file object. The byte locked is the first byte of the
    # file, which is where `msvcrt.locking` starts from once the descriptor has been seeked there.
    def _try_lock(self):
        try:
            os.lseek(self.fd, 0, os.SEEK_SET)
        except OSError:
            pass
        if msvcrt is not None:
            try:
                msvcrt.locking(self.fd, msvcrt.LK_NBLCK, 1)
                return True
            except OSError:
                return False
        try:  # pragma: no cover - not this host
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _try_unlock(self):
        try:
            os.lseek(self.fd, 0, os.SEEK_SET)
            if msvcrt is not None:
                msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover - not this host
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            return True
        except OSError:
            return False

    @property
    def held(self):
        return self.fd is not None

    def acquire(self):
        """Take the lock, waiting at most `timeout_seconds`. Returns whether it was taken.

        Contention is a refusal, never an overwrite: no other process is signalled, no live lock is
        removed, and nothing about the benchmark proceeds on the assumption the machine is ours.
        """
        if self.held:
            return True
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
        started = self.monotonic()
        deadline = started + self.timeout_seconds
        while True:
            if self._try_lock():
                self.acquired_at_utc = utc_now()
                # Set here as well as in `__enter__`, so a holder that took the lock through
                # `acquire_or_raise()` still reports how long it kept it: the number is what tells a
                # reader whether the lock covered the batch or only the preflight.
                self._wall_started = time.time()
                self.waited_seconds = round(self.monotonic() - started, 3)
                self.record["started_utc"] = self.acquired_at_utc
                self.owner = dict(self.record)
                self._write_metadata()
                return True
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                break
            # Capped at the remaining budget, so the loop gives up *at* the timeout rather than one
            # poll interval after it: §9 asks for a refusal inside five seconds, and 5.0 + 0.1 is a
            # different sentence from 5.0 when the reader is deciding whether to wait longer.
            self.sleep(min(self.POLL_SECONDS, remaining))
        self.waited_seconds = round(self.monotonic() - started, 3)
        try:
            os.close(self.fd)
        except OSError:
            pass
        finally:
            self.fd = None
        self.owner = read_lock_metadata(self.metadata_path) or {"pid": None, "hostname": None,
                                                                "metadata_available": False}
        return False

    def acquire_or_raise(self):
        if not self.acquire():
            raise BenchmarkLockBusy(self.owner or {}, self.waited_seconds)
        return self

    def _write_metadata(self):
        try:
            with open(self.metadata_path, "w", encoding="utf-8") as handle:
                json.dump(self.record, handle, indent=2, sort_keys=True)
        except OSError:
            pass  # diagnostics never block the benchmark

    def release(self):
        """Give the lock back. Neither file is deleted: a waiter may be blocked on that descriptor.

        The metadata keeps naming the last owner, which is why it is diagnostics and the OS lock is
        the authority - a JSON file cannot say whether its writer is still running, and pretending it
        can is how an existence check gets mistaken for a lock.
        """
        if not self.held:
            return False
        if getattr(self, "_wall_started", None):
            self.held_for_seconds = round(time.time() - self._wall_started, 3)
        released = self._try_unlock()
        try:
            os.close(self.fd)
        except OSError:
            pass
        finally:
            self.fd = None
        return released

    def __enter__(self):
        self.acquire_or_raise()
        self._wall_started = time.time()
        return self

    def __exit__(self, *_exc):
        self.release()
        return False


# --------------------------------------------------------------------------------------------
# Reading the machine (§11, §13, §14)
# --------------------------------------------------------------------------------------------

def identity_of(pid, process_create_time=None):
    """§2: the pair that identifies a process, because a pid alone identifies only a number.

    Windows recycles pids, so a row that arrives after its predecessor exited can carry the same
    number for a different process. `process_create_time` is what tells them apart, and it is the
    half that has to survive: when the OS would not give one up, the pid is kept and the identity is
    explicitly half-known rather than silently widened to "any process with this number".
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if process_create_time is None:
        return (pid, None)
    try:
        return (pid, round(float(process_create_time), 6))
    except (TypeError, ValueError):
        return (pid, None)


def identity_key(item):
    """One process identity as a stable string, from a row, a pair or an already-written key.

    The artifact has to hold these in a JSON list, where a tuple becomes a string anyway; writing the
    same normalised form on both sides is what lets a reader compare "who was here at the start of
    the window" with "who is here now" without guessing at a pid.
    """
    if isinstance(item, dict):
        ident = identity_of(item.get("pid"), item.get("process_create_time"))
    elif isinstance(item, (tuple, list)):
        ident = identity_of(item[0] if item else None, item[1] if len(item) > 1 else None)
    elif isinstance(item, int):
        ident = (int(item), None)
    else:
        text = str(item or "")
        return text if "@" in text else ""
    return "" if ident is None else f"{ident[0]}@{'-' if ident[1] is None else ident[1]}"


def identity_set(items):
    """A collection of rows, pairs or written keys as one set of (pid, create_time) pairs."""
    out = set()
    for item in items or []:
        if isinstance(item, dict):
            ident = identity_of(item.get("pid"), item.get("process_create_time"))
        elif isinstance(item, (tuple, list)):
            ident = identity_of(item[0] if item else None, item[1] if len(item) > 1 else None)
        elif isinstance(item, int):
            ident = (int(item), None)
        else:
            pid, _, created = str(item or "").partition("@")
            try:
                ident = identity_of(pid, created or None)
            except (TypeError, ValueError):
                ident = None
        if ident is not None:
            out.add(ident)
    return out


def process_tree_snapshot(root_pid=None, rows=None):
    """§2: who was in the benchmark's process tree *at this instant*, with a reusable identity.

    This is deliberately not a set of pids accumulated over a run. The membership test walks each
    process' parent chain as it reads it now, so a child that has exited has no live row and is out
    of the snapshot, and a pid that has been handed to a new process is in or out according to that
    new process' own parent. `sample()` takes one of these per reading and nothing else, which is the
    semantics §2 asks for: a GPU event is compared with the tree of its own sample, never with the
    union of the batch.
    """
    root_pid = os.getpid() if root_pid is None else int(root_pid)
    rows = process_rows() if rows is None else rows
    by_pid = {int(row["pid"]): row for row in rows if row.get("pid") is not None}
    out = []
    for pid, row in sorted(by_pid.items()):
        if not _chain_reaches(pid, root_pid, by_pid):
            continue
        out.append({"pid": pid, "name": row.get("name") or "",
                    "process_create_time": row.get("process_create_time"),
                    "parent_pid": row.get("parent_pid"),
                    "is_current_benchmark_descendant": True})
    if not any(row["pid"] == root_pid for row in out):
        # The reader of the monitor is the root, whatever the OS says about who its parent is, and a
        # snapshot that lost the process taking it would call its own sampler external.
        root_row = by_pid.get(root_pid) or {}
        out.insert(0, {"pid": root_pid, "name": root_row.get("name") or "",
                       "process_create_time": root_row.get("process_create_time"),
                       "parent_pid": root_row.get("parent_pid"),
                       "is_current_benchmark_descendant": True})
    return out


def _chain_reaches(pid, root_pid, by_pid, limit=64):
    """Whether a process' own parent chain arrives at `root_pid` right now."""
    seen = set()
    current = int(pid)
    for _ in range(limit):
        if current == int(root_pid):
            return True
        if current in seen:
            return False
        seen.add(current)
        parent = (by_pid.get(current) or {}).get("parent_pid")
        try:
            parent = int(parent)
        except (TypeError, ValueError):
            return False
        current = parent
    return False


def tree_identities(snapshot):
    """The (pid, create_time) pairs of one snapshot, as a set for a same-sample membership test."""
    return {identity for identity in (identity_of(row.get("pid"), row.get("process_create_time"))
                                      for row in snapshot or []) if identity is not None}


def matches_identity(row, identities):
    """Whether one process row is one of a set of (pid, create_time) identities (§2).

    The pair is the identity, so a recycled pid cannot inherit a dead process' membership. When
    either side has no creation time to offer, the pid is all the evidence there is, and matching on
    it is the forgiving direction: it forgives a process the OS refused to date rather than charging
    it with being new.
    """
    ident = identity_of(row.get("pid"), row.get("process_create_time"))
    if ident is None:
        return False
    if ident in identities:
        return True
    return any(known[0] == ident[0] and (known[1] is None or ident[1] is None)
               for known in identities)


def in_this_tree(row, snapshot):
    """Whether a process row belongs to the tree snapshot it was read beside (§2)."""
    return matches_identity(row, tree_identities(snapshot))


def descendant_pids(root_pid=None):
    """Every pid below `root_pid`, including itself (§17: our own FFmpeg is not contamination).

    Kept for callers that want a pid list for one moment. It is *not* what contamination is judged
    on: §2's rule needs the snapshot above, taken per sample, because a pid list collected over a
    whole batch keeps naming processes that exited minutes ago.
    """
    root_pid = os.getpid() if root_pid is None else root_pid
    pids = {int(root_pid)}
    try:
        for child in psutil.Process(root_pid).children(recursive=True):
            try:
                pids.add(int(child.pid))
            except psutil.Error:
                continue
    except psutil.Error:
        pass
    return pids


def process_rows(fields=("pid", "name", "cmdline", "ppid", "create_time", "exe")):
    """Every process the OS will name, with the fields §2's identity and §13's verdict need."""
    rows = []
    for proc in psutil.process_iter(list(fields)):
        try:
            info = proc.info
            name = (info.get("name") or "").strip()
            cmd = " ".join(info.get("cmdline") or [])
            created = info.get("create_time")
            ppid = info.get("ppid")
            executable = info.get("exe")
            rows.append({"pid": int(info["pid"]), "name": name, "command_line": cmd,
                         "process_create_time": (None if created is None
                                                 else round(float(created), 6)),
                         "parent_pid": None if ppid is None else int(ppid),
                         "executable_path": (str(executable) if executable else None)})
        except (psutil.Error, TypeError, ValueError):
            continue
    return rows


def _path_spellings(path):
    """Every way this host's APIs might have written one path, lowercase, for substring matching.

    A process' command line is whatever its launcher typed: `F:\\repo`, `f:/repo`, `./repo` inside a
    shell that kept the relative form. Matching only the spelling `os.path.abspath` produced here
    would let a competing `pytest` run through the preflight because somebody started it from a
    different directory, which is the one thing §13 is for.
    """
    if not path:
        return []
    given = str(path)
    out = set()
    for candidate in {given, os.path.abspath(given), os.path.normpath(given)}:
        for variant in {candidate, candidate.replace("\\", "/"), candidate.replace("/", "\\")}:
            out.add(variant.lower().rstrip("/\\"))
    return sorted(out, key=len, reverse=True)


def _mentions_path(text, spellings):
    lowered = str(text or "").lower().replace("\\", "/")
    return any(spelling.replace("\\", "/") in lowered for spelling in spellings)


def looks_like_a_workload(name, command_line, exclude=None):
    """§13's second half, as a predicate a test can feed rows to without spawning anything.

    `exclude` is the repository path, and removing it before matching is not a detail: this
    repository is called `dlss5standaloneV2`, so a haystack that still contains the path matches the
    `dlss` token on its own, and every process merely *opened* in this directory - an editor, a
    shell, a chat client - would be reported as a competing workload. With the path taken out, what
    is left has to name a workload by itself.
    """
    text = f"{name} {command_line}"
    for spelling in _path_spellings(exclude):
        for variant in {spelling, spelling.replace("\\", "/"), spelling.replace("/", "\\")}:
            text = text.replace(variant, " ")
    haystack = text.lower()
    return (os.path.basename(str(name)).lower() in {e.lower() for e in CONFLICTING_EXECUTABLES}
            or any(token.lower() in haystack for token in CONFLICTING_COMMAND_TOKENS))


def repo_related_processes(repo_path=_ROOT, rows=None):
    """Every process whose image or command line names this repository (§11's second list).

    This is the raw list §11 asks a sample to record. Deciding which of them is a *workload* is
    `conflicting_repo_processes` below, and keeping the two apart is what lets a reader check the
    decision instead of inheriting it: a chat client opened in this directory shows up here and not
    there.
    """
    rows = process_rows() if rows is None else rows
    spellings = _path_spellings(repo_path)
    return [row for row in rows
            if _mentions_path(f"{row.get('name') or ''} {row.get('command_line') or ''}",
                              spellings)]


def conflicting_repo_processes(repo_path=_ROOT, tree_snapshot=None, rows=None):
    """§13: another process running this repository's work, from outside the benchmark's tree.

    Three things have to hold together for a row to count: it is not in the process tree of *this*
    sample (§2), its command line names this repository's path, and its image or command line is the
    kind that does build, test or transcode work. A chat client that happens to be open in this
    directory is not a conflict; a second `pytest` running the same tests is.
    """
    rows = process_rows() if rows is None else rows
    out = []
    for row in repo_related_processes(repo_path=repo_path, rows=rows):
        if in_this_tree(row, tree_snapshot) or row["pid"] == os.getpid():
            continue
        if looks_like_a_workload(row.get("name") or "", row.get("command_line") or "",
                                 exclude=repo_path):
            out.append({"pid": row["pid"], "name": row.get("name"),
                        "process_create_time": row.get("process_create_time"),
                        "parent_pid": row.get("parent_pid"),
                        "command_line": row.get("command_line"),
                        "is_current_benchmark_descendant": False})
    return out


def _smi_rows(query, compute_apps=False):
    """nvidia-smi rows for `query`, or None when the tool is absent or unhappy."""
    flag = "--query-compute-apps=" if compute_apps else "--query-gpu="
    try:
        out = subprocess.run(["nvidia-smi", flag + query, "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode != 0:
            return None
        return [[field.strip() for field in line.split(",")]
                for line in out.stdout.strip().splitlines() if line.strip()]
    except Exception:
        return None


def _pmon_table(args=PMON_UTILISATION_ARGS, timeout=20):
    """`nvidia-smi pmon` answers as rows of its own column names, or None when it will not answer.

    The tool prints two comment lines before its data - the column names, then the units - and pads the
    command name to a fixed width, so the header is what makes the columns addressable. Everything
    before the last column is positional and exactly as long as the header minus the name; the name is
    whatever is left, because `Qoder CN.exe` is one command with a space in it.

    A `-` in an engine column means "this pid did not occupy that engine during this sample period". It
    becomes None here rather than 0.0, which is §11's rule for the machine-level fields carried over: a
    zero would read as an idle measurement, and the two sentences are different.
    """
    try:
        out = subprocess.run(["nvidia-smi", *(list(args) if not isinstance(args, str) else [args])],
                             capture_output=True, text=True, timeout=timeout)
        if out.returncode != 0:
            return None
        lines = [line.strip() for line in (out.stdout or "").splitlines() if line.strip()]
    except Exception:
        return None
    header = next((line for line in lines if line.startswith("#") and "pid" in line), None)
    if not header:
        return None
    names = [name for name in header.lstrip("#").split() if name != "#"]
    if len(names) < 3:
        return None
    positional = len(names) - 1
    rows = []
    for line in lines:
        if line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) < positional:
            continue
        values = dict(zip(names[:positional], tokens[:positional]))
        values[names[-1]] = " ".join(tokens[positional:])
        rows.append(values)
    return rows


def per_process_gpu_utilisation(rows=None, args=PMON_UTILISATION_ARGS):
    """The only per-process GPU evidence this driver gives: `pmon`'s engine columns, keyed by pid.

    This is criterion four's measurement and criterion two's, and it is a separate reading from the
    compute-apps table because `--query-compute-apps` does not carry utilisation at all. The fields are
    named after what they are, and a pid that `pmon` did not list gets no entry rather than an entry of
    zeros - an absence a reader can see is not the same claim as an idle measurement.
    """
    rows = _pmon_table(args=args) if rows is None else rows
    availability = {"per_process_gpu_utilisation_answered": rows is not None,
                    "per_process_gpu_utilisation_row_count": len(rows or []),
                    "per_process_gpu_utilisation_unavailable_reason":
                        None if rows is not None else "nvidia-smi pmon did not answer"}
    if rows is None:
        return {}, availability
    out = {}
    for row in rows:
        try:
            pid = int(row.get("pid"))
        except (TypeError, ValueError):
            continue
        entry = {"pmon_engine_type_reported": (row.get("type") or None),
                 "pmon_command_name": (row.get("command") or None)}
        for column, field in PMON_UTILISATION_COLUMNS.items():
            entry[field] = _number(row.get(column))
        entry["utilisation_fields_unreadable"] = [
            field for field in PMON_UTILISATION_COLUMNS.values() if entry[field] is None]
        out[pid] = entry
    return out, availability


def names_a_compute_workload(process_name, command_line=None, executable_path=None, exclude=None):
    """Criterion three: does this process name, path or command line *say* it is a compute workload?

    Two matchings, because the two haystacks are not equally safe. Against the image name a substring is
    fine - `python.exe` contains `python` however it was launched - and this is the rule that has always
    been here. Against a path or a command line a substring is not: `C:\\Users\\dev\\Observations`
    contains `obs`, and the tokens in this file are short enough for that to matter. So the wider
    haystack is matched at word boundaries, with the repository's own path removed first for the same
    reason `looks_like_a_workload` removes it - this checkout is called `dlss5standaloneV2`, and a chat
    client opened in it must not be reported as a second DLSS implementation.
    """
    name = os.path.basename(str(process_name or ""))
    if any(token.lower() in name.lower() for token in GPU_WORKLOAD_TOKENS):
        return True
    haystacks = [str(command_line or ""), str(executable_path or "")]
    for spelling in _path_spellings(exclude):
        for variant in {spelling, spelling.replace("\\", "/"), spelling.replace("/", "\\")}:
            haystacks = [haystack.replace(variant, " ") for haystack in haystacks]
    for haystack in haystacks:
        if not haystack.strip():
            continue
        for token in GPU_COMPUTE_COMMAND_TOKENS:
            if re.search(rf"(?<![a-z0-9_]){re.escape(token.lower())}(?![a-z0-9_])",
                         haystack.lower().replace("\\", "/")):
                return True
    return False


def gpu_readings():
    """§11's GPU block. Unreadable fields are None and named in `unavailable_fields`."""
    fields = {"gpu_available": False}
    fields.update({name: None for name in GPU_FIELD_NAMES})
    fields["nvidia_smi_answered"] = False
    rows = _smi_rows(GPU_FIELD_QUERY)
    if not rows:
        fields["unavailable_fields"] = list(GPU_FIELD_NAMES) + ["gpu_available"]
        return fields
    fields["nvidia_smi_answered"] = True
    row = rows[0]
    unavailable = []
    for index, name in enumerate(GPU_FIELD_NAMES):
        value = _number(row[index]) if index < len(row) else None
        if name == "gpu_pstate":
            raw = (row[index] if index < len(row) else "").strip()
            fields[name] = None if raw.lower() in NOT_REPORTED else raw
        else:
            fields[name] = value
        if fields[name] is None:
            unavailable.append(name)
    fields["unavailable_fields"] = unavailable
    fields["gpu_available"] = fields["gpu_utilization_percent"] is not None
    return fields


def _root_create_time(tree_snapshot=None, process_rows_by_pid=None, root_pid=None):
    """When the process at the root of this sample's tree started, which is when the batch's own life
    began. None when neither the snapshot nor the process list can say, and then the create-time
    attribution below is simply unavailable rather than assumed."""
    root_pid = os.getpid() if root_pid is None else int(root_pid)
    for entry in tree_snapshot or []:
        try:
            pid = int(entry.get("pid") or -1)
        except (TypeError, ValueError):
            continue
        if pid != root_pid:
            continue
        value = entry.get("process_create_time")
        if value is not None:
            return float(value)
    row = (process_rows_by_pid or {}).get(root_pid) or {}
    value = row.get("process_create_time")
    return None if value is None else float(value)


def benchmark_descendant_attribution(row, tree_snapshot=None, process_rows_by_pid=None,
                                     root_pid=None):
    """Is this GPU-table row a process the benchmark started, and *which fact* says so? (§2, §19)

    Returns `(attributed, method)`. The first answer is the one §2 was written for: the row's own
    (pid, create_time) is in the tree of this same reading. The second is the one the torn read at
    `eb61bbc` needed, because a child spawned a moment after the process list was taken has a row in
    the GPU table and none in the list - so it has no parent, no creation time, and no membership, and
    the batch's own FFmpeg came out as a `repo_conflict`. Two changes fix that: `sample()` now reads the
    card before the list, which makes a listed child visible, and a row whose identity is not in the
    snapshot can still be attributed when the OS *does* answer the two questions that matter about it -
    its parent is in this same sample's tree, and it came into existence at or after the root process
    did. Both have to hold. A long-lived desktop client cannot satisfy them because it was created
    before the batch, and a recycled pid cannot because the pid it would inherit a membership from is
    the thing the identity check already refused.

    Nothing here is a name. A competitor's FFmpeg is not forgiven by this function; it has a parent
    outside the tree and it is caught by §13 and by criterion three.
    """
    root_pid = os.getpid() if root_pid is None else int(root_pid)
    try:
        pid = int(row.get("pid"))
    except (TypeError, ValueError):
        return False, None
    if pid == root_pid:
        return True, "this_is_the_reading_process"
    if in_this_tree(row, tree_snapshot):
        return True, "identity_in_this_sample_tree"
    parent_pid, created = row.get("parent_pid"), row.get("process_create_time")
    try:
        parent_pid = None if parent_pid is None else int(parent_pid)
        created = None if created is None else float(created)
    except (TypeError, ValueError):
        return False, None
    if parent_pid is None or created is None or not tree_snapshot:
        return False, None
    parent_in_tree = any(int(entry.get("pid") or -1) == parent_pid
                         for entry in tree_snapshot if entry.get("pid") is not None)
    if not parent_in_tree:
        return False, None
    started = _root_create_time(tree_snapshot, process_rows_by_pid, root_pid)
    if started is None or created < started:
        return False, None
    return True, "parent_in_this_sample_tree_and_created_after_the_root"


def external_gpu_compute_evidence(row, repo_path=_ROOT,
                                  threshold=PER_PROCESS_GPU_COMPUTE_PERCENT_AT_LEAST):
    """Which of the four criteria one GPU-table row satisfies *in this sample*, with their numbers.

    An empty list is the answer that matters: it means nothing about this row demonstrates compute, and
    the caller classifies it as ambient-or-unproven rather than inventing a reason. Each entry names its
    criterion and carries the figure it was reached from, so a reader can disagree with the measurement
    instead of having to take the verdict on trust.

    A field the driver did not report is not evidence. `[N/A]` memory, a `-` in an engine column and a
    missing pid in `pmon`'s list all produce no criterion rather than the criterion a zero would
    produce, which is §11's rule applied to the per-process case: an unmeasured process is an unproven
    one, and unproven is not idle.
    """
    out = []
    if row.get("memory_reported_by_driver") and (row.get("used_memory_bytes") or 0) > 0:
        out.append({"criterion": "driver_accounts_gpu_memory_in_use",
                    "detail": {"used_memory_as_reported": row.get("used_memory"),
                               "used_memory_bytes": row.get("used_memory_bytes")}})
    for column, field in PMON_UTILISATION_COLUMNS.items():
        value = row.get(field)
        if value is None:
            continue
        if column in PMON_CODEC_ENGINE_COLUMNS and value >= threshold:
            out.append({"criterion": "codec_engine_occupied",
                        "detail": {"engine": column, "percent": value, "at_least": threshold,
                                   "note": "a coding engine held at or above the frozen limit is work "
                                           "on the card, not a window being drawn"}})
        elif column in ("sm", "mem") and value >= threshold:
            out.append({"criterion": "per_process_gpu_utilisation_at_or_above_the_frozen_limit",
                        "detail": {"engine": column, "percent": value, "at_least": threshold}})
    if names_a_compute_workload(row.get("process_name"), row.get("command_line"),
                                row.get("executable_path"), exclude=repo_path):
        out.append({"criterion": "image_path_or_command_names_a_compute_workload",
                    "detail": {"process_name": row.get("process_name"),
                               "executable_path": row.get("executable_path"),
                               "command_line": row.get("command_line")}})
    return out


def nvidia_compute_processes(tree_snapshot=None, rows=None, process_rows_by_pid=None,
                             utilisation_by_pid=None, root_pid=None):
    """§14's table, verbatim: pid, process name, and `used_memory` exactly as it was reported.

    `[N/A]` stays `[N/A]` in `used_memory`, and `used_memory_bytes` is None beside it with
    `memory_reported_by_driver` false, so a reader can see the driver declined to account for that
    process rather than seeing a zero that would look like an idle one.

    §2: each row also carries the identity and the parent the OS reports for that pid *now*, and
    `is_current_benchmark_descendant` is decided against the tree snapshot of the same reading and
    recorded with the method that decided it (`descendant_attribution`), so the forgiveness a process
    was given can be checked against the sample rather than taken on trust.

    `utilisation_by_pid` is `per_process_gpu_utilisation()`'s answer for the same reading. A pid the
    tool did not list keeps None in every engine field and `pmon_listed_this_pid` false, which is the
    difference between "nobody measured it" and "it held nothing".
    """
    rows = _smi_rows("pid,process_name,used_memory", compute_apps=True) if rows is None else rows
    out = []
    if not rows:
        return out
    by_pid = {} if process_rows_by_pid is None else process_rows_by_pid
    utilisation = {} if utilisation_by_pid is None else utilisation_by_pid
    for row in rows:
        if len(row) < 3:
            continue
        try:
            pid = int(row[0])
        except ValueError:
            continue
        name, memory = row[1], row[2]
        megabytes = _number(memory)
        live = by_pid.get(pid) or {}
        entry = utilisation.get(pid) or {}
        candidate = {"pid": pid, "process_name": name, "used_memory": memory,
                     "used_memory_bytes": None if megabytes is None
                     else int(megabytes * 1024 * 1024),
                     "memory_reported_by_driver": megabytes is not None,
                     "process_create_time": live.get("process_create_time"),
                     "parent_pid": live.get("parent_pid"),
                     # §5: an ambient class is only as good as the identity behind it, so the row
                     # carries the path and the command line that were checked, not just the verdict.
                     "command_line": live.get("command_line"),
                     "executable_path": live.get("executable_path") or (name if "\\" in name
                                                                        else None),
                     "pmon_listed_this_pid": bool(entry),
                     "pmon_engine_type_reported": entry.get("pmon_engine_type_reported"),
                     "pmon_command_name": entry.get("pmon_command_name")}
        for field in PMON_UTILISATION_COLUMNS.values():
            candidate[field] = entry.get(field)
        attributed, method = benchmark_descendant_attribution(
            candidate, tree_snapshot=tree_snapshot, process_rows_by_pid=by_pid, root_pid=root_pid)
        candidate["is_current_benchmark_descendant"] = attributed
        candidate["descendant_attribution"] = method
        candidate["is_this_process"] = pid == (os.getpid() if root_pid is None else int(root_pid))
        out.append(candidate)
    return out


def looks_like_a_gpu_workload(process_name):
    """Whether an image name in nvidia-smi's table reads as work rather than as a window.

    §14 as written fails on an idle machine here, because this driver's `--query-compute-apps`
    answers with every process holding a GPU context - explorer, the browser, the terminal - and not
    with the CUDA subset. This predicate is the part of the narrowing that does not depend on a
    process having arrived during the window: a competitor that was already rendering when the
    sampling started is still a competitor, and the driver will not tell us its memory, so its name
    has to.
    """
    name = os.path.basename(str(process_name or "")).lower()
    return any(token.lower() in name for token in GPU_WORKLOAD_TOKENS)


def looks_like_the_query_tool(process_name):
    """Whether a GPU-table row is the very tool this monitor uses to read the table.

    A one-name predicate so the exemption is auditable on its own: the query tool is the monitor's
    own child, so it is classified with the benchmark and the list it matches is
    `GPU_QUERY_TOOL_NAMES`, nothing more.
    """
    name = os.path.basename(str(process_name or "")).lower()
    return name in {token.lower() for token in GPU_QUERY_TOOL_NAMES}


def windows_system_root(system_root=None):
    """The Windows directory this machine reports, which is where an OS component's path starts.

    Read from the environment rather than typed, because a path typed from memory is a claim about
    somebody else's install: on a machine whose Windows directory is on another drive it would make
    every genuine ShellHost fail §5's exact-path condition, and the row that should be forgiven
    would be the one reported.
    """
    given = (system_root or os.environ.get("SystemRoot") or os.environ.get("windir")
             or "").strip().rstrip("\\/")
    return given or None


def windows_ambient_component_path(basename, system_root=None):
    """The one full path an entry of §5's ambient table has to be at, or None if it has no root.

    `shellhost.exe` matches `<Windows>\\System32\\shellhost.exe` and nothing else. This builds a
    single file name per table entry - it is not a prefix, and no caller can pass a directory here
    and have it mean "everything inside".
    """
    root = windows_system_root(system_root)
    if not root or not basename:
        return None
    return os.path.join(root, "System32", basename)


def _same_path(one, other):
    """Two spellings of the same file, decided by exact normalised equality rather than by prefix.

    `_path_spellings` exists for substring matching in command lines, where `F:\\repo` has to catch
    `f:/repo/tests`. Identity is the opposite case: one path may only equal the other path, so this
    compares the full normalised forms against each other and nothing shorter.
    """
    if not one or not other:
        return False
    return bool(set(_path_spellings(one)) & set(_path_spellings(other)))


def ambient_os_graphics_identity(row, system_root=None):
    """§5's six conditions for `ambient_os_graphics_client`, and the names of the ones it failed.

    Returns `(ok, failed_conditions)`. The caller has already refused the descendant, repo and
    workload rules, so what is left here is the identity question: is this row demonstrably the
    Windows shell component §5 allows, or is it something wearing that name?

    Arrival is not one of the conditions, and that omission is the fix. The previous rule forgave a
    long-lived client and condemned one that had just started, which meant a shell host restarting
    mid-batch - the thing Windows does without asking - read as an outsider settling in for the whole
    run. An exactly identified OS component is ambient whether or not it was there at the start; a
    row whose identity cannot be read is not ambient at all.
    """
    failed = []
    name = str(row.get("process_name") or "")
    basename = os.path.basename(str(row.get("executable_path") or name))
    if not basename:
        return False, ["no_executable_identity"]
    if basename.lower() not in AMBIENT_OS_GRAPHICS_EXECUTABLES:
        failed.append("basename_is_not_in_the_ambient_table")
        return False, failed
    expected = windows_ambient_component_path(basename, system_root=system_root)
    if not expected:
        failed.append("no_windows_directory_to_match_the_path_against")
    elif not _same_path(row.get("executable_path") or name, expected):
        failed.append("executable_path_is_not_the_windows_component")
    if row.get("is_current_benchmark_descendant"):
        failed.append("is_a_benchmark_descendant")
    # A reported figure has to be a *quantity* to count against a row. `0` is the driver answering
    # "none", which is the ambient case rather than a disqualifier, and treating it as one would make
    # the forgiving rule depend on which of two idle numbers the driver chose to print.
    if row.get("memory_reported_by_driver") and (row.get("used_memory_bytes") or 0) > 0:
        failed.append("driver_accounts_gpu_memory_in_use_for_it")
    command_line = row.get("command_line")
    if command_line is None:
        failed.append("command_line_not_readable")
    elif any(token in str(command_line).lower()
             for token in AMBIENT_DISQUALIFYING_COMMAND_TOKENS):
        failed.append("command_line_names_a_workload")
    return not failed, failed


def classify_gpu_processes(processes, baseline_identities=None, workload_rows=None,
                           system_root=None, repo_path=_ROOT):
    """Apply §5's policy to the table, and say which rule classified each row.

    Three of the six names may stop a window (`repo_conflict`, `known_compute_workload`,
    `unknown_external_gpu_client`) and two may not (`ambient_os_graphics_client`,
    `ambient_or_unproven_gpu_client`); the sixth, `benchmark_descendant`, is the run itself. What
    changed at this revision is *how the third conflict name is reached*. It used to be reached by two
    branches that named no workload at all - a client that arrived during the window, and a client whose
    identity could not be read - and on a machine whose driver lists thirty-five windowing clients in
    the compute table, that meant a chat window opening its GPU context was grounds for refusing a
    round. Now `unknown_external_gpu_client` requires one of `EXTERNAL_GPU_COMPUTE_CRITERIA` to hold for
    that process, evaluated per sample by `external_gpu_compute_evidence`, and a row that satisfies none
    of them is recorded as unproven instead: its pid, path, memory field and arrival all stay in the
    artifact, and nothing downstream may treat them as a reason.

    The order is the argument, so it is worth reading as one. Own tree first, then the query tool, then
    §13's repository workload - a competitor running this project's work is a conflict whatever its
    utilisation looks like - and only then the criteria, with the narrow exact-identity OS excuse after
    them so that a ShellHost wearing a CUDA payload is still refused.

    `baseline_identities` are the (pid, create_time) pairs the window saw when it started. They are no
    longer a verdict input, and they stay because §5's "must record" half is not optional: a reader
    should be able to see that something joined the table mid-window, and decide from the recorded
    evidence rather than from the instrument's opinion whether it mattered.
    """
    baseline = identity_set(baseline_identities)
    workload_pids = set() if workload_rows is None else {int(r["pid"]) for r in workload_rows}
    out = []
    for row in processes:
        ambient_ok, ambient_failed = ambient_os_graphics_identity(row, system_root=system_root)
        arrived = bool(baseline) and not matches_identity(row, baseline)
        criteria = external_gpu_compute_evidence(row, repo_path=repo_path)
        named = [item for item in criteria
                 if item["criterion"] == "image_path_or_command_names_a_compute_workload"]
        proven = [item for item in criteria if item not in named]
        if row.get("is_this_process") or row.get("is_current_benchmark_descendant"):
            kind = "benchmark_descendant"
            reason = ("the benchmark's own process tree, in this sample"
                      + (f" (attributed by {row.get('descendant_attribution')})"
                         if row.get("descendant_attribution") else ""))
        elif looks_like_the_query_tool(row.get("process_name")):
            kind = "benchmark_descendant"
            reason = ("the nvidia-smi query this monitor launches, which is its own child rather "
                      "than a client of the card")
        elif row["pid"] in workload_pids:
            kind, reason = ("repo_conflict",
                            "§5: a process running this repository's work from outside the "
                            "benchmark's tree - a second benchmark, another pytest, another ffmpeg")
        elif named:
            kind = "known_compute_workload"
            reason = ("criterion %d (%s): its image name, its path or its command line names a "
                      "compute workload by itself - "
                      % (EXTERNAL_GPU_COMPUTE_CRITERIA.index(named[0]["criterion"]) + 1,
                         named[0]["criterion"])
                      + json.dumps(named[0]["detail"], sort_keys=True, default=str))
        elif proven:
            kind, reason = ("unknown_external_gpu_client",
                            "nobody can say what this client is, but "
                            + ", ".join(item["criterion"] for item in proven)
                            + " is evidence of the card being used: "
                            + json.dumps(proven[0]["detail"], sort_keys=True, default=str))
        elif ambient_ok:
            kind = "ambient_os_graphics_client"
            reason = ("§5: exactly identified as the Windows shell component it names - name, full "
                      "path, no GPU memory in use accounted for it, no workload in its command line - "
                      "and no criterion above was satisfied")
        else:
            kind = "ambient_or_unproven_gpu_client"
            reason = ("recorded, not refused: none of " + ", ".join(EXTERNAL_GPU_COMPUTE_CRITERIA)
                      + " holds for this row"
                      + (", and it did join the table during the window" if arrived
                         else ", and it was already in the table when the window started")
                      + (f"; §5's narrow ambient identity failed on {ambient_failed}"
                         if ambient_failed else ""))
        out.append({**row, "classification": kind, "classification_reason": reason,
                    "ambient_identity_failed_conditions": ambient_failed,
                    "compute_evidence": criteria,
                    "compute_criteria_met": [item["criterion"] for item in criteria],
                    # §5's "must record" half, kept as a field rather than as a verdict: a row whose
                    # identity could not be read used to be refused for that alone, and the reversal is
                    # only honest if the fact is still written down where a reader can find it.
                    "identity_readable": bool(row.get("executable_path") or row.get("process_name")),
                    "arrived_during_the_window": arrived if baseline else None})
    return out


def published_gpu_row(row, sample_utc=None):
    """One classified GPU-table row as §11 publishes it: the facts, then the verdict, then the moment.

    An explicit list rather than the classifier's own dictionary, for the same reason the memory figure
    is written unconverted: what leaves this function is the record a reader audits, so it is named here
    in one place instead of being whatever the classifier happened to add last. Every field §5's four
    criteria and §2's two attribution methods read is in it, including the ones that decided *against*
    refusing the row - `compute_criteria_met: []` is only checkable next to the `used_memory_bytes: null`
    and the `pmon_listed_this_pid: false` that produced it, and
    `ambient_identity_failed_conditions` is the difference between "this is the shell component" and "we
    could not prove anything about it".

    `sample_timestamp` is added here rather than in the classifier because it belongs to the reading, not
    to the process: §2's ownership claim is about one moment, and `collect_transfer_status` refuses a row
    that cannot say which moment it was judged at.
    """
    out = {"pid": row["pid"], "process_name": row["process_name"],
           "used_memory": row["used_memory"],
           "used_memory_bytes": row.get("used_memory_bytes"),
           "memory_reported_by_driver": row.get("memory_reported_by_driver"),
           "process_create_time": row.get("process_create_time"),
           "parent_pid": row.get("parent_pid"),
           "executable_path": row.get("executable_path"),
           "command_line": row.get("command_line"),
           "identity_readable": row.get("identity_readable"),
           "is_current_benchmark_descendant": row.get("is_current_benchmark_descendant"),
           "descendant_attribution": row.get("descendant_attribution"),
           "classification": row.get("classification"),
           "classification_reason": row.get("classification_reason"),
           "compute_criteria_met": row.get("compute_criteria_met"),
           "compute_evidence": row.get("compute_evidence"),
           "ambient_identity_failed_conditions": row.get("ambient_identity_failed_conditions"),
           "arrived_during_the_window": row.get("arrived_during_the_window"),
           "pmon_listed_this_pid": row.get("pmon_listed_this_pid"),
           "pmon_engine_type_reported": row.get("pmon_engine_type_reported"),
           "pmon_command_name": row.get("pmon_command_name"),
           "sample_timestamp": sample_utc or utc_now()}
    for field in PMON_UTILISATION_COLUMNS.values():
        out[field] = row.get(field)
    return out


def sample(tree_snapshot=None, repo_path=_ROOT, baseline_identities=None, root_pid=None,
           rows=None, compute_rows=None, utilisation_rows=None):
    """One 1 Hz reading of everything §11 names, in the order §11 names it.

    §11 asks each sample to record the NVIDIA compute process list and the repo-related process
    list, not merely a verdict about them, so both lists are in the row: `nvidia_compute_processes`
    is §14's table (pid, process name, `used_memory` as reported, and which rule classified it) and
    `repo_processes` is every process naming this repository, of which
    `conflicting_repo_processes` is the subset §13 calls a workload. `system_cpu_percent` keeps the
    brief's own name because the row is meant to be read without this file open.

    §2 is why the tree is read *here*, per sample, instead of being handed in as a list collected
    over a batch: `process_tree_snapshot` says who was in the benchmark's tree at the moment of this
    reading, `tree_of_this_sample` publishes it, and every process row in the reading carries the
    pid, creation time, parent and membership that let a reader check the claim afterwards.

    **The order of the four reads is the torn-read fix, and it is load-bearing.** The card is queried
    before the operating system's process list, twice over: the compute table, then `pmon`. That makes
    the list used for parentage always at least as recent as the table it is matched against, so a child
    the benchmark spawned while the first query was in flight is in both. Read the other way round -
    which is how this function worked at `eb61bbc`, and the reason its own FFmpeg was written up as a
    `repo_conflict` - the child is in the GPU table with no row in the older list, so it has no parent,
    no creation time and no membership, and no amount of care in the classifier can recover a fact that
    was never read. `test_a_child_spawned_between_the_two_reads_is_attributed` pins the ordering by
    advancing a fake clock between the two calls, so the property survives a refactor of this body
    rather than only a comment.
    """
    gpu_table = (_smi_rows("pid,process_name,used_memory", compute_apps=True)
                 if compute_rows is None else compute_rows)
    utilisation, utilisation_availability = per_process_gpu_utilisation(rows=utilisation_rows)
    rows = process_rows() if rows is None else rows
    by_pid = {int(row["pid"]): row for row in rows if row.get("pid") is not None}
    tree_snapshot = (process_tree_snapshot(root_pid=root_pid, rows=rows)
                     if tree_snapshot is None else list(tree_snapshot))
    processes = nvidia_compute_processes(rows=gpu_table or [], tree_snapshot=tree_snapshot,
                                        process_rows_by_pid=by_pid,
                                        utilisation_by_pid=utilisation, root_pid=root_pid)
    conflicts = conflicting_repo_processes(repo_path=repo_path, tree_snapshot=tree_snapshot,
                                          rows=rows)
    classified = classify_gpu_processes(processes, baseline_identities=baseline_identities,
                                        workload_rows=conflicts, repo_path=repo_path)
    reading = {"utc": utc_now(), "unix_seconds": round(time.time(), 3),
               "system_cpu_percent": psutil.cpu_percent(interval=None),
               "available_ram_bytes": int(psutil.virtual_memory().available),
               "benchmark_descendant_pids": len(tree_snapshot),
               "tree_of_this_sample": tree_snapshot}
    reading.update(gpu_readings())
    reading["repo_processes"] = [_who(row) for row in
                                 repo_related_processes(repo_path=repo_path, rows=rows)]
    reading["conflicting_repo_processes"] = [_who(row) for row in conflicts]
    reading["nvidia_compute_processes"] = [published_gpu_row(row, reading["utc"])
                                           for row in classified]
    reading["external_compute_processes"] = [
        published_gpu_row(row, reading["utc"]) for row in classified
        if row["classification"] in GPU_CLIENT_CONFLICT_CLASSIFICATIONS]
    reading["gpu_process_count"] = len(classified)
    reading["ambient_or_unproven_gpu_client_count"] = sum(
        1 for row in classified
        if row["classification"] == "ambient_or_unproven_gpu_client")
    # §5: the ambient clients are counted beside the unproven ones rather than mixed into them,
    # because "we forgave this by exact identity" and "nothing here demonstrates compute" are two
    # different claims, and the report has to be able to say which one it is making.
    reading["ambient_os_graphics_client_count"] = sum(
        1 for row in classified
        if row["classification"] == "ambient_os_graphics_client")
    for key, value in utilisation_availability.items():
        reading[key] = value
    reading["classification_counts"] = {
        kind: sum(1 for row in classified if row["classification"] == kind)
        for kind in GPU_CLIENT_CLASSIFICATIONS}
    return reading


def _who(row):
    """One process as §2 wants it remembered: who, since when, whose child, and ours or not."""
    return {"pid": row.get("pid"), "name": row.get("name"),
            "process_create_time": row.get("process_create_time"),
            "parent_pid": row.get("parent_pid"),
            "is_current_benchmark_descendant": bool(row.get("is_current_benchmark_descendant"))
            if row.get("is_current_benchmark_descendant") is not None else False}


def prime_cpu_percent():
    """psutil's first non-blocking cpu_percent() is always 0.0; this throws that reading away."""
    psutil.cpu_percent(interval=None)


# --------------------------------------------------------------------------------------------
# Judging a window (§12, §15)
# --------------------------------------------------------------------------------------------

def percentile(values, fraction):
    """A deterministic linear-interpolated percentile; None when there is nothing to rank."""
    ordered = sorted(v for v in values if v is not None)
    if not ordered:
        return None
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def conflict_streaks(samples, consecutive=CONTAMINATION_CONSECUTIVE_SAMPLES):
    """How long each conflict-class outsider stayed, and whether that reached §18's three in a row.

    §18's rule was already the monitor's: the same identity, in consecutive samples, three times. The
    window that *certifies* a machine as quiet now applies the same test to the same rows, because
    otherwise the two halves of the instrument disagree - a one-sample blip would refuse a preflight
    window while the monitor, on identical evidence, would not contaminate a batch. That asymmetry is
    what made a chat window that opened its GPU context during a sample cost a whole round.

    Only identities in `GPU_CLIENT_CONFLICT_CLASSIFICATIONS` can appear at all: `workload_sightings`
    reads `external_compute_processes`, which is that set and nothing else. A streak is broken by an
    absent sample, not by a shorter one, and the identity is §2's (kind, pid, create time, name), so a
    pid handed to a different process starts a new count rather than continuing one.
    """
    live, maxima, latest = {}, {}, {}
    for reading in samples:
        present = workload_sightings(reading)
        for identity, info in present.items():
            live[identity] = live.get(identity, 0) + 1
            maxima[identity] = max(maxima.get(identity, 0), live[identity])
            latest[identity] = info
        for identity in [key for key in live if key not in present]:
            del live[identity]
    return sorted(({"kind": identity[0], "pid": identity[1],
                    "name": latest[identity].get("name"),
                    "used_memory": latest[identity].get("used_memory"),
                    "classification": latest[identity].get("classification"),
                    "process_create_time": identity[2],
                    "consecutive_samples": maxima[identity],
                    "sustained": maxima[identity] >= consecutive,
                    "first_seen_utc": latest[identity].get("sample_timestamp"),
                    "source": latest[identity].get("source")}
                   for identity in maxima),
                  key=lambda row: (not row["sustained"], row["kind"], str(row["pid"])))


def summarise(samples):
    """The five §12 figures plus the outsider lists, from a window's samples and nothing else.

    `external_compute_processes` stays the union of every conflict-class row the window saw, because
    §14's recording obligation is not conditional on the row mattering. The two lists after it are the
    §18 streak decision applied to that same union: which outsiders stayed three consecutive samples,
    and which were there and gone. Only the first may refuse a window, and both are written down, so a
    reader can see the difference the rule drew rather than having to trust it.
    """
    cpu = [row.get("system_cpu_percent") for row in samples]
    gpu = [row.get("gpu_utilization_percent") for row in samples]
    ram = [row.get("available_ram_bytes") for row in samples]
    all_streaks = conflict_streaks(samples)
    streaks = [row for row in all_streaks if row["kind"] == "gpu_compute"]
    repo_streaks = [row for row in all_streaks if row["kind"] == "repo_workload"]

    def label(row):
        return " ".join(str(part) for part in
                        (row.get("pid"), row.get("name"), row.get("used_memory"),
                         row.get("classification")) if part is not None)

    return {"sample_count": len(samples),
            "cpu_median_percent": (round(statistics.median([v for v in cpu if v is not None]), 3)
                                   if any(v is not None for v in cpu) else None),
            "cpu_p95_percent": _round(percentile(cpu, 0.95), 3),
            "gpu_utilization_median_percent": (
                round(statistics.median([v for v in gpu if v is not None]), 3)
                if any(v is not None for v in gpu) else None),
            "gpu_utilization_p95_percent": _round(percentile(gpu, 0.95), 3),
            "gpu_utilization_samples_unreadable": sum(1 for v in gpu if v is None),
            "available_ram_min_bytes": min([v for v in ram if v is not None], default=None),
            "gpu_temperature_max_celsius": max(
                (row.get("gpu_temperature_celsius") for row in samples
                 if row.get("gpu_temperature_celsius") is not None), default=None),
            "gpu_pstate_seen": sorted({str(row.get("gpu_pstate")) for row in samples}),
            "external_compute_processes": sorted(
                {f'{row.get("pid")} {row.get("process_name")} '
                 f'{row.get("used_memory")}' for sample_row in samples
                 for row in sample_row.get("external_compute_processes") or []}),
            "sustained_external_compute_processes": sorted(
                {label(row) for row in streaks if row["sustained"]}),
            "transient_external_compute_process_sightings": sorted(
                {label(row) for row in streaks if not row["sustained"]}),
            "external_compute_conflict_streaks": streaks,
            "conflicting_repo_processes_streaks": repo_streaks,
            "sustained_conflicting_repo_processes": sorted(
                {label(row) for row in repo_streaks if row["sustained"]}),
            "conflicting_repo_processes": sorted(
                {f'{row.get("pid")} {row.get("name")}' for sample_row in samples
                 for row in sample_row.get("conflicting_repo_processes") or []})}


def _round(value, digits):
    return None if value is None else round(value, digits)


def window_sample_count(window_seconds=WINDOW_SECONDS, interval=SAMPLE_INTERVAL_SECONDS):
    """How many readings a window is made of: §11's thirty samples at 1 Hz."""
    return max(1, int(round(float(window_seconds) / float(interval))))


def judge_window(samples, thresholds=None, window_seconds=WINDOW_SECONDS,
                 interval=SAMPLE_INTERVAL_SECONDS):
    """§12's five thresholds plus §13's and §14's process rules, as named failures.

    A failure is a reason string and the numbers that produced it. Nothing here rounds an unreadable
    figure into a passing one: a threshold that could not be evaluated is its own failure.

    §13's repository-workload rule is unchanged from the brief's wording - one sighting refuses the
    window - because the follow-up directive that narrowed the GPU-client case did not touch it, and
    silently extending a loosening to a rule nobody relaxed is how an instrument stops meaning what
    its specification says. Its streak is still recorded beside it, in
    `sustained_conflicting_repo_processes`, so a reader can compare how the two rules behaved on the
    same machine rather than take the difference on trust.
    """
    thresholds = dict(THRESHOLDS if thresholds is None else thresholds)
    summary = summarise(samples)
    failures = []
    recorded_not_refused = []

    def compare(name, value, limit, at_least=False):
        """One threshold, and the two ways it can fail: too busy, or not measurable.

        A figure that was never read cannot certify quietness. It is reported as its own failure
        rather than skipped, which is the same rule §11 sets for the GPU fields and the reason no
        unreadable metric here defaults to zero.
        """
        if value is None:
            failures.append({"reason": f"{name}_could_not_be_read",
                             "detail": {"limit": limit,
                                        "note": "unreadable is not quiet: a threshold cannot be "
                                                "certified against a figure that was not reported"}})
        elif (value < limit) if at_least else (value > limit):
            failures.append({"reason": f"{name}_violated",
                             "detail": {"measured": value, "limit": limit}})

    compare("cpu_median_percent", summary["cpu_median_percent"],
            thresholds["cpu_median_percent_at_most"])
    compare("cpu_p95_percent", summary["cpu_p95_percent"], thresholds["cpu_p95_percent_at_most"])
    compare("gpu_utilization_median_percent", summary["gpu_utilization_median_percent"],
            thresholds["gpu_utilization_median_percent_at_most"])
    compare("gpu_utilization_p95_percent", summary["gpu_utilization_p95_percent"],
            thresholds["gpu_utilization_p95_percent_at_most"])
    ram = summary["available_ram_min_bytes"]
    compare("available_ram_bytes", ram, thresholds["available_ram_at_least_bytes"], at_least=True)
    if ram is not None and ram < thresholds["available_ram_at_least_bytes"]:
        failures[-1]["detail"]["measured_gib"] = round(ram / 2**30, 3)
        failures[-1]["detail"]["limit_gib"] = round(
            thresholds["available_ram_at_least_bytes"] / 2**30, 3)
    if summary["conflicting_repo_processes"]:
        failures.append({"reason": "conflicting_repo_process",
                         "detail": {"processes": summary["conflicting_repo_processes"],
                                    "note": "recorded, not killed (§13)"}})
    if summary["sustained_external_compute_processes"]:
        failures.append({"reason": "external_gpu_compute_workload",
                         "detail": {"processes": summary["sustained_external_compute_processes"],
                                    "streaks": [row for row
                                                in summary["external_compute_conflict_streaks"]
                                                if row["sustained"]],
                                    "consecutive_samples_required":
                                        CONTAMINATION_CONSECUTIVE_SAMPLES,
                                    "note": "recorded, not killed (§14); §18's three consecutive "
                                            "samples is what makes it a refusal here as well"}})
    if summary["transient_external_compute_process_sightings"]:
        recorded_not_refused.append(
            {"reason": "transient_external_gpu_client",
             "detail": {"sightings": summary["transient_external_compute_process_sightings"],
                        "streaks": [row for row in summary["external_compute_conflict_streaks"]
                                    if not row["sustained"]],
                        "note": "§5's follow-up rule as a written-down fact: a conflict-class client "
                                "that was on the card for fewer than three consecutive samples did not "
                                "refuse this window, and this entry exists so that the reason it did "
                                "not is checkable rather than silent"}})
    needed = window_sample_count(window_seconds, interval)
    if len(samples) < needed:
        failures.append({"reason": "window_has_fewer_samples_than_it_owes",
                         "detail": {"samples": len(samples), "required": needed,
                                    "window_seconds": window_seconds,
                                    "interval_seconds": interval}})
    return {"ok": not failures, "failures": failures, "recorded_not_refused": recorded_not_refused,
            "summary": summary, "thresholds": thresholds, "sample_count": len(samples)}


def run_window(sample_fn=None, window_seconds=WINDOW_SECONDS, interval=SAMPLE_INTERVAL_SECONDS,
               sleep=time.sleep, monotonic=time.monotonic, prime=prime_cpu_percent):
    """One preflight window: `window_seconds` of samples on an `interval` grid, and a verdict.

    Two things are deliberate about the loop. Samples are scheduled on a grid counted from the
    window's own start instead of sleeping an interval after each one, because a sample costs a
    couple of nvidia-smi calls and the drift would stretch thirty samples past thirty seconds; and
    the first sample comes one interval *after* the start, because `psutil.cpu_percent(None)`
    reports the share since its previous call, so a reading taken immediately is measured over
    nothing. Thirty readings at 1 Hz is then literally thirty seconds of machine.

    The count is capped by the schedule, so an injected clock that never advances cannot hang a test.

    The verdict is merged in by name rather than spread into the literal, because spreading it once
    turned this function's `samples` list into `judge_window`'s sample *count*: two producers, one
    key, and the artifact that read it back published the integer 30 where thirty measurements should
    have been. That is the same field-name collision the ffprobe aggregate was fixed for in §1, so
    the two are kept apart structurally now.
    """
    sample_fn = sample if sample_fn is None else sample_fn
    needed = window_sample_count(window_seconds, interval)
    rows = process_rows()
    by_pid = {int(row["pid"]): row for row in rows if row.get("pid") is not None}
    tree_snapshot = process_tree_snapshot(rows=rows)
    baseline = nvidia_compute_processes(tree_snapshot=tree_snapshot, process_rows_by_pid=by_pid)
    prime()
    samples = []
    started = monotonic()
    while len(samples) < needed:
        delay = started + ((len(samples) + 1) * interval) - monotonic()
        if delay > 0:
            sleep(delay)
        # §2: no tree is handed in. Each reading takes the tree of its own moment, so a child that
        # exited cannot keep representing the run in a later sample, and a recycled pid cannot
        # inherit one that it did.
        samples.append(sample_fn(baseline_identities=baseline))
    window = {"samples": samples, "started_utc": (samples or [{}])[0].get("utc"),
              "ended_utc": (samples or [{}])[-1].get("utc"),
              "compute_processes_at_window_start": sorted(
                  {identity_key(row) for row in baseline})}
    for key, value in judge_window(samples, window_seconds=window_seconds,
                                   interval=interval).items():
        window[key] = value
    assert isinstance(window["samples"], list), "the sample list must stay a list of samples"
    return window


def quiet_preflight(sample_fn=None, window_seconds=WINDOW_SECONDS,
                    interval=SAMPLE_INTERVAL_SECONDS, max_wait_seconds=QUIET_WAIT_SECONDS,
                    thresholds=None, sleep=time.sleep, monotonic=time.monotonic, log=None,
                    lock=None, code_under_test=None):
    """§11/§15: windows until one is quiet, and stop asking after ten minutes.

    The payload is `benchmark_environment.json` as §20 asks for it, field names included, so the
    artifact is this function's return value rather than a translation of it. `lock` is the
    `BenchmarkLock` the caller is holding: its state and owner record go into the artifact, because
    "the machine was quiet" and "nobody else was benchmarking" are two different claims and the
    round needs to be able to say both.

    Waiting is not a re-roll of the benchmark (§16): this only decides whether a batch may start.
    Once it does start, the batch is the batch.
    """
    thresholds = dict(THRESHOLDS if thresholds is None else thresholds)
    started = monotonic()
    start_utc = utc_now()
    windows = []
    passed = None
    # The budget is what §15 sets; the count is only a backstop so an injected clock that does not
    # advance can hang a test rather than the benchmark.
    window_cap = int(max_wait_seconds / window_seconds) + 2
    while len(windows) < window_cap:
        window = run_window(sample_fn=sample_fn, window_seconds=window_seconds,
                            interval=interval, sleep=sleep, monotonic=monotonic)
        # How far into the wait this window finished, on the clock §15's budget is measured by. The
        # sample rows carry their own timestamps, but the budget is about the wait, and the wait is
        # what this loop's clock counts.
        window["elapsed_seconds_at_end"] = round(monotonic() - started, 3)
        windows.append(window)
        if window["ok"]:
            passed = len(windows)
            break
        if monotonic() - started >= max_wait_seconds:
            break
    last = windows[-1]
    all_samples = [dict(row, preflight_window_attempt=index + 1)
                   for index, window in enumerate(windows) for row in window["samples"]]
    # §15's rule is that the round is certified by *one* window: the first in which every threshold
    # held. `selected_passing_window` is that window and nothing else, so the gate that decides
    # `quiet_preflight` has one object to read instead of a union of every window the wait went
    # through - and a machine that was busy for nine windows and idle for the tenth was idle when the
    # benchmark started. The failed windows are not deleted: they stay in `windows` below, with their
    # own failures, and the union of what they saw is published beside the selection.
    selected = None
    if passed:
        chosen = windows[passed - 1]
        selected = {
            "attempt": passed,
            "started_utc": chosen["started_utc"],
            "ended_utc": chosen["ended_utc"],
            "sample_count": len(chosen["samples"]),
            "ok": chosen["ok"],
            "failures": chosen["failures"],
            "summary": chosen["summary"],
            "external_compute_processes": chosen["summary"]["external_compute_processes"],
            "sustained_external_compute_processes":
                chosen["summary"]["sustained_external_compute_processes"],
            "transient_external_compute_process_sightings":
                chosen["summary"]["transient_external_compute_process_sightings"],
            "recorded_not_refused": chosen["recorded_not_refused"],
            "conflicting_repo_processes": chosen["summary"]["conflicting_repo_processes"],
            "appeared_after_seconds": chosen["elapsed_seconds_at_end"],
            "within_wait_budget": bool(chosen["elapsed_seconds_at_end"] is not None
                                       and chosen["elapsed_seconds_at_end"] <= max_wait_seconds),
        }
    historical_external = sorted({row for window in windows
                                  for row in window["summary"]["external_compute_processes"]})
    historical_repo = sorted({row for window in windows
                              for row in window["summary"]["conflicting_repo_processes"]})
    payload = {
        "generated_by": "tests/unit/benchmark_environment.py",
        "code_under_test": code_under_test,
        # §20's names, verbatim, so a reader checking the artifact against the brief finds the
        # fields under the names the brief gave them.
        "lock_acquired": bool(lock.held) if lock is not None else False,
        "lock_owner": (lock.owner if lock is not None and lock.held else None),
        "lock_path": lock.path if lock is not None else None,
        "lock_mechanism": (lock.record.get("lock_mechanism") if lock is not None else None),
        "preflight_start": start_utc,
        "preflight_end": utc_now(),
        "preflight_window_length_seconds": window_seconds,
        "preflight_sample_interval_seconds": interval,
        "preflight_window_count_attempted": len(windows),
        "preflight_window_that_passed": passed,
        "preflight_wait_budget_seconds": max_wait_seconds,
        "selected_passing_window": selected,
        "thresholds": thresholds,
        "all_preflight_samples": all_samples,
        "cpu_median": (selected or last)["summary"]["cpu_median_percent"],
        "cpu_p95": (selected or last)["summary"]["cpu_p95_percent"],
        "gpu_util_median": (selected or last)["summary"]["gpu_utilization_median_percent"],
        "gpu_util_p95": (selected or last)["summary"]["gpu_utilization_p95_percent"],
        "available_ram_min": (selected or last)["summary"]["available_ram_min_bytes"],
        "available_ram_min_gib": (_round(
            ((selected or last)["summary"]["available_ram_min_bytes"] or 0) / 2**30, 3)
            if (selected or last)["summary"]["available_ram_min_bytes"] is not None else None),
        "gpu_temperature_max_celsius": (selected or last)["summary"]["gpu_temperature_max_celsius"],
        "gpu_pstate_seen": (selected or last)["summary"]["gpu_pstate_seen"],
        # §20 names these two fields, and the question they answer is "what was in the way of the
        # window the round was certified on", so they carry the selection. The history of the windows
        # that were refused stays beside them under its own name instead of being mixed in.
        "external_compute_processes": (selected["external_compute_processes"] if selected
                                       else historical_external),
        "conflicting_repo_processes": (selected["conflicting_repo_processes"] if selected
                                       else historical_repo),
        "external_compute_processes_across_all_windows": historical_external,
        "conflicting_repo_processes_across_all_windows": historical_repo,
        "preflight_status": "PASS" if passed else "FAIL",
        "not_ready_reason": None if passed else "no_quiet_preflight_window",
        "windows": [{"attempt": index + 1, "started_utc": window["started_utc"],
                     "ended_utc": window["ended_utc"], "ok": window["ok"],
                     "elapsed_seconds_at_end": window["elapsed_seconds_at_end"],
                     "sample_count": len(window["samples"]),
                     "failures": window["failures"],
                     "recorded_not_refused": window["recorded_not_refused"],
                     "summary": window["summary"],
                     "compute_processes_at_window_start":
                         window["compute_processes_at_window_start"]}
                    for index, window in enumerate(windows)],
        "note": "the samples live in one list, `all_preflight_samples`, which is §20's name for "
                "them, and each row says which window it belongs to in `preflight_window_attempt` - "
                "duplicating them under each window too would put the same thirty numbers in the "
                "file twice. Each row carries §14's GPU client table (38 rows on this driver, one "
                "line per process with the memory exactly as the driver reported it), so the "
                "per-sample lists are the bulk of this file; the figures that decide the window are "
                "the five scalars above, recomputable from those rows. §15 accepts a window, not a "
                "wait: `selected_passing_window` is the one window in which every threshold held, "
                "and it is the only thing the gate judges. The windows that were refused stay in "
                "`windows` with their failures, and everything any of them saw stays in the "
                "`*_across_all_windows` lists, so a quiet tenth window after nine busy ones reads as "
                "a quiet tenth window and not as a machine that was never busy.",
    }
    if log:
        log(json.dumps({"preflight_status": payload["preflight_status"],
                        "window_that_passed": passed,
                        "windows_attempted": len(windows),
                        "failures_of_last_window": [row["reason"] for row in
                                                    last["failures"]]}, indent=2, sort_keys=True))
    return payload


# --------------------------------------------------------------------------------------------
# Watching the batch (§17-§19, §21)
# --------------------------------------------------------------------------------------------

# `CONTAMINATION_CONSECUTIVE_SAMPLES` is defined with the other frozen policy numbers, above.

# §17 forbids the monitor using CPU or GPU utilisation as a failure threshold, because the
# benchmark itself is the largest consumer on the machine while it runs. Nothing in
# `ContaminationLedger` reads those fields; they are recorded so a reader can see what the round
# chose not to fail on, and the check that says so is in §22's regressions.
MEDIA_CHILD_TOKENS = ("ffmpeg", "ffprobe")


def descendant_rows(root_pid=None):
    """(pid, name) for every process in the tree at this moment, including the root (§19's tracking).

    The names matter as well as the numbers: §21 asks the monitor to say which of the benchmark's own
    children were FFmpeg, so a reader can confirm the transcoder they see in the GPU table is the one
    this batch spawned rather than a neighbour's. This is a *reading* of the tree, taken now; the
    authority for whether a process belongs to the run is the snapshot in the sample that saw it.
    """
    return [(row["pid"], row["name"]) for row in process_tree_snapshot(root_pid=root_pid)]


# §21 asks the runtime artifact for its samples, and §11 lists what a sample carries. Keeping §11's
# measured fields is the point; keeping this driver's forty-row GPU client table in every one of the
# couple of thousand samples a full batch takes is not - it would put a ten-megabyte JSON file in the
# evidence commit. A runtime sample therefore records every number §11 names, plus the two outsider
# lists §18's rule is actually applied to, plus the counts that show the full table was read and
# classified each second. §17 does not let the monitor fail on utilisation, so nothing here reads the
# utilisation figures; they are recorded because the brief says to record them.
RUNTIME_SAMPLE_FIELDS = ("utc", "unix_seconds", "system_cpu_percent", "available_ram_bytes",
                         "gpu_utilization_percent", "gpu_memory_utilization_percent",
                         "gpu_temperature_celsius", "gpu_graphics_clock_mhz", "gpu_sm_clock_mhz",
                         "gpu_memory_clock_mhz", "gpu_power_draw_watts", "gpu_pstate",
                         "unavailable_fields", "benchmark_descendant_pids",
                         "gpu_process_count", "ambient_os_graphics_client_count",
                         "ambient_or_unproven_gpu_client_count", "per_process_gpu_utilisation_answered",
                         "tree_of_this_sample")


def runtime_reading(reading):
    """Project a full §11 sample down to what the runtime monitor keeps (§17, §21)."""
    out = {name: reading.get(name) for name in RUNTIME_SAMPLE_FIELDS}
    out["conflicting_repo_processes"] = [
        _who(row) for row in reading.get("conflicting_repo_processes") or []]
    out["external_compute_processes"] = [
        {"pid": row.get("pid"), "process_name": row.get("process_name"),
         "used_memory": row.get("used_memory"), "classification": row.get("classification"),
         "classification_reason": row.get("classification_reason"),
         # The evidence as well as the verdict: §18's three-sample rule is what makes an event count,
         # and which of the four criteria made it an event in the first place has to be readable from
         # the sample that recorded it, not inferred afterwards from a name.
         "compute_criteria_met": row.get("compute_criteria_met"),
         "descendant_attribution": row.get("descendant_attribution"),
         "process_create_time": row.get("process_create_time"),
         "parent_pid": row.get("parent_pid"),
         "is_current_benchmark_descendant": row.get("is_current_benchmark_descendant"),
         "sample_timestamp": row.get("sample_timestamp") or reading.get("utc")}
        for row in reading.get("external_compute_processes") or []]
    repo_rows = reading.get("repo_processes")
    out["repo_process_count"] = None if repo_rows is None else len(repo_rows)
    out["full_gpu_process_table_in_sample"] = False
    return out


def workload_sightings(reading):
    """§18's candidates in one sample, each with the identity §2 says a process has.

    Two kinds, matching the brief's two rules: a process doing this repository's work from outside
    the benchmark's tree, and an external NVIDIA compute process the §14 policy called a workload.
    The identity is (kind, pid, create_time, name) rather than a row or a bare pid, because the rule
    is about *one process* persisting across samples: a streak that continues after the pid it was
    built from has been handed to something else is not a streak.
    """
    out = {}
    for row in reading.get("conflicting_repo_processes") or []:
        if row.get("is_current_benchmark_descendant"):
            continue
        key = ("repo_workload", row.get("pid"), row.get("process_create_time"),
               row.get("name") or "")
        out[key] = {"kind": "repo_workload", "pid": row.get("pid"),
                    "name": row.get("name") or "",
                    "classification": row.get("classification") or "repo_conflict",
                    "process_create_time": row.get("process_create_time"),
                    "parent_pid": row.get("parent_pid"),
                    "sample_timestamp": reading.get("utc"),
                    "source": "§13: a repository workload outside the benchmark's process tree"}
    for row in reading.get("external_compute_processes") or []:
        if row.get("is_current_benchmark_descendant"):
            continue
        key = ("gpu_compute", row.get("pid"), row.get("process_create_time"),
               row.get("process_name") or "")
        out[key] = {"kind": "gpu_compute", "pid": row.get("pid"),
                    "name": row.get("process_name") or "",
                    "used_memory": row.get("used_memory"),
                    # §5: an event has to say which of the conflict classes raised it, so a reader
                    # can check that no ambient or desktop row ever produced one.
                    "classification": row.get("classification"),
                    # ... and which of the four criteria put it in that class, because "an external
                    # GPU compute workload stopped this batch" is only checkable next to the figure
                    # it was reached from.
                    "compute_criteria_met": row.get("compute_criteria_met"),
                    "process_create_time": row.get("process_create_time"),
                    "parent_pid": row.get("parent_pid"),
                    "sample_timestamp": reading.get("utc"),
                    "source": "§5/§14: " + (row.get("classification_reason")
                                            or "an NVIDIA compute process outside the benchmark's "
                                               "tree")}
    return out


class ContaminationLedger:
    """§18's rule as state: the same outsider, seen in three samples in a row.

    A sighting that disappears breaks the run, which is what makes the rule about *consecutive*
    samples rather than about a total. Two one-sample blips - a `git` wrapper, a thumbnail
    generator, a process that started and finished between two readings - therefore do not stop a
    round that has already begun, and a competitor that settles in for three seconds does. Both
    halves of that are §22's regressions.
    """

    def __init__(self, consecutive=CONTAMINATION_CONSECUTIVE_SAMPLES):
        self.consecutive = max(1, int(consecutive))
        self.streaks = {}
        self.seen = {}
        self.contaminated = False
        self.reason = None
        self.reason_source = None
        self.first_contamination_timestamp = None

    def add(self, reading):
        """Feed one sample in; returns the identities that crossed the line on this sample."""
        sample_utc = reading.get("utc")
        present = workload_sightings(reading)
        crossed = []
        for identity, info in present.items():
            streak = self.streaks.get(identity)
            record = self.seen.setdefault(identity, {
                **info, "first_seen_utc": sample_utc, "last_seen_utc": sample_utc,
                "sightings": 0, "consecutive_max": 0, "reached_contamination_threshold": False})
            record["sightings"] += 1
            record["last_seen_utc"] = sample_utc
            length = (streak + 1) if streak is not None else 1
            self.streaks[identity] = length
            record["consecutive_max"] = max(record["consecutive_max"], length)
            if length >= self.consecutive:
                record["reached_contamination_threshold"] = True
                if not self.contaminated:
                    self.contaminated = True
                    created = info.get("process_create_time")
                    self.reason = (f"{info['kind']} workload pid {info['pid']} "
                                   f"({info['name'] or 'unnamed'}, created "
                                   f"{'at an unreadable time' if created is None else created}) "
                                   f"was present in {length} consecutive 1 Hz samples")
                    self.first_contamination_timestamp = record["first_seen_utc"]
                    self.reason_source = info["source"]
                if length == self.consecutive:
                    crossed.append(identity)
        for identity in [key for key in self.streaks if key not in present]:
            del self.streaks[identity]
        return crossed

    def event_rows(self):
        """Every outsider the batch saw, with how long each one stayed and what it was called."""
        return sorted(({"kind": info["kind"], "pid": info["pid"], "name": info["name"],
                        "classification": info.get("classification"),
                        "used_memory": info.get("used_memory"),
                        "compute_criteria_met": info.get("compute_criteria_met"),
                        "process_create_time": info.get("process_create_time"),
                        "parent_pid": info.get("parent_pid"),
                        "identity": identity_key({"pid": info["pid"],
                                                  "process_create_time":
                                                      info.get("process_create_time")}),
                        "source": info["source"], "first_seen_utc": info["first_seen_utc"],
                        "last_seen_utc": info["last_seen_utc"], "sightings": info["sightings"],
                        "consecutive_samples": info["consecutive_max"],
                        "reached_contamination_threshold": info["reached_contamination_threshold"]}
                       for info in self.seen.values()),
                      key=lambda row: (not row["reached_contamination_threshold"],
                                       row["kind"], str(row["pid"])))


def judge_contamination(samples, consecutive=CONTAMINATION_CONSECUTIVE_SAMPLES):
    """§18 over a finished list of samples: the same `ContaminationLedger`, run in one go."""
    ledger = ContaminationLedger(consecutive=consecutive)
    for reading in samples:
        ledger.add(reading)
    return ledger


class RuntimeMonitor:
    """A 1 Hz watcher for the length of the batch, recording who else was on the machine.

    The monitor deliberately does not stop anything by itself and does not re-run anything: §18 says
    keep the completed executions, do not reroll, and let the driver stop launching *further*
    executions once the current child has ended. Its authority is the record it writes; the gate
    that reads NOT_READY from that record is in `collect_transfer_status`.

    Everything it needs from the outside world is injectable - `sample_fn`, `sleep`, `monotonic`,
    `thread_factory` - and `observe()` can be driven by hand, so §22's two contamination regressions
    run over synthetic samples in one thread and never need a busy machine.
    """

    def __init__(self, sample_fn=None, interval=SAMPLE_INTERVAL_SECONDS,
                 consecutive=CONTAMINATION_CONSECUTIVE_SAMPLES, root=_ROOT, log=None,
                 sleep=time.sleep, monotonic=time.monotonic,
                 thread_factory=None, descendants_fn=process_tree_snapshot,
                 project=runtime_reading):
        self.sample_fn = sample_fn or (lambda **kwargs: sample(**kwargs))
        self.interval = float(interval)
        self.root = root
        self.log = log
        self.sleep = sleep
        self.monotonic = monotonic
        self.thread_factory = thread_factory
        self.descendants_fn = descendants_fn
        self.project = project or (lambda reading: reading)
        self.ledger = ContaminationLedger(consecutive=consecutive)
        self.samples = []
        self.benchmark_child_pids = set()
        self.ffmpeg_child_pids = set()
        self.benchmark_descendant_identities = {}
        self.started_utc = None
        self.ended_utc = None
        self._stop = threading.Event()
        self._thread = None

    def observe(self, reading=None, tree_snapshot=None, baseline_identities=None):
        """Add one reading. Returns the sample row, and updates the contamination state.

        §2: `tree_snapshot` exists so a test can hand in who was in the tree at that instant. When it
        is not handed in, the reading takes its own - never a list carried over from an earlier tick,
        which is what made an exited child look like it was still here.
        """
        if reading is None:
            reading = self.sample_fn(tree_snapshot=tree_snapshot,
                                     baseline_identities=baseline_identities)
        reading = self.project(dict(reading))
        crossed = self.ledger.add(reading)
        self.samples.append(reading)
        return {"sample": reading, "newly_contaminated": [
            {"kind": kind[0], "pid": kind[1], "process_create_time": kind[2],
             "name": kind[3]} for kind in crossed]}

    def _track_descendants(self, snapshot=None):
        rows = (self.descendants_fn() if snapshot is None else snapshot) or []
        for row in rows:
            name = str(row.get("name") or "")
            ident = identity_key(row)
            self.benchmark_child_pids.add(int(row["pid"]))
            self.benchmark_descendant_identities[ident] = {
                "pid": int(row["pid"]), "name": name,
                "process_create_time": row.get("process_create_time"),
                "parent_pid": row.get("parent_pid"),
                "first_seen_utc": (self.benchmark_descendant_identities.get(ident) or {})
                .get("first_seen_utc") or utc_now()}
            if any(token in name.lower() for token in MEDIA_CHILD_TOKENS):
                self.ffmpeg_child_pids.add(int(row["pid"]))
        return [(row["pid"], row.get("name")) for row in rows]

    def _loop(self):
        rows = process_rows()
        baseline = nvidia_compute_processes(
            tree_snapshot=process_tree_snapshot(rows=rows),
            process_rows_by_pid={int(r["pid"]): r for r in rows if r.get("pid") is not None})
        prime_cpu_percent()  # the first non-blocking cpu_percent() is always 0.0; drop that one
        started = self.monotonic()
        while not self._stop.is_set():
            delay = self.interval - ((self.monotonic() - started) % self.interval)
            self._stop.wait(delay if 0 < delay <= self.interval else self.interval)
            if self._stop.is_set():
                break
            # One snapshot per tick, used for both the §21 child record and this tick's own
            # classification, so the two can never disagree about who was running.
            snapshot = process_tree_snapshot(rows=process_rows())
            self._track_descendants(snapshot)
            out = self.observe(tree_snapshot=snapshot, baseline_identities=baseline)
            if self.log and out["newly_contaminated"]:
                self.log(json.dumps({"contaminated_by": out["newly_contaminated"],
                                     "utc": out["sample"].get("utc")}, sort_keys=True))

    def start(self):
        self.started_utc = utc_now()
        if self.thread_factory is not None:  # a test's stand-in for a thread
            self._thread = self.thread_factory(self._loop)
        else:  # pragma: no cover - exercised by the driver, not by §22
            self._thread = threading.Thread(target=self._loop, name="benchmark-monitor",
                                            daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """End the watch. Whatever it saw stays in the artifact whether or not it was fatal."""
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and getattr(thread, "join", None):
            thread.join(timeout=self.interval * 5 + 5)
        self._track_descendants()
        self.ended_utc = utc_now()
        return self.artifact()

    @property
    def contaminated(self):
        return self.ledger.contaminated

    def artifact(self, code_under_test=None, batch=None):
        """§21's `benchmark_runtime_monitor.json`, from what this monitor actually saw."""
        payload = {
            "generated_by": "tests/unit/benchmark_environment.py",
            "code_under_test": code_under_test,
            "measurement_start": self.started_utc,
            "measurement_end": self.ended_utc,
            "sample_interval_seconds": self.interval,
            "sample_count": len(self.samples),
            "samples": self.samples,
            "external_workload_events": self.ledger.event_rows(),
            "runtime_contaminated": self.ledger.contaminated,
            "contamination_reason": self.ledger.reason,
            "contamination_rule": self.ledger.reason_source if self.ledger.contaminated else None,
            "contamination_consecutive_samples_required": self.ledger.consecutive,
            "first_contamination_timestamp": self.ledger.first_contamination_timestamp,
            "benchmark_child_pids": sorted(self.benchmark_child_pids),
            "ffmpeg_child_pids": sorted(self.ffmpeg_child_pids),
            "benchmark_descendant_identities": sorted(self.benchmark_descendant_identities.values(),
                                                      key=lambda row: (str(row["pid"]),
                                                                       str(row["parent_pid"]))),
            "utilisation_is_not_a_failure_threshold": True,
            "sample_fields": list(RUNTIME_SAMPLE_FIELDS),
            "note": "CPU and GPU utilisation are recorded in every sample and read by nothing here: "
                    "§17 forbids failing a batch because the benchmark is loading the machine, so "
                    "the only thing this monitor judges is whether a process outside the "
                    "benchmark's own tree showed up and stayed for "
                    f"{self.ledger.consecutive} consecutive samples. Ownership is decided per "
                    "sample, from the tree snapshot in that same sample, because a pid list "
                    "collected across the whole batch would keep naming children that exited "
                    "minutes ago and would let a recycled pid inherit a dead child's exemption "
                    "(§2). `benchmark_child_pids` and `ffmpeg_child_pids` are the history §21 "
                    "names; `benchmark_descendant_identities` is the same history with the creation "
                    "time that makes it an identity rather than a number. Each sample carries every "
                    "number §11 names, its own tree, plus the two outsider lists; the driver's full "
                    "GPU client table is read and classified every second, and what it concluded is "
                    "in these rows, but the forty rows themselves are not repeated in a file that "
                    "runs for the length of the batch.",
        }
        if batch:
            payload["batch"] = batch
        return payload


# --------------------------------------------------------------------------------------------
# §6: is this machine even allowed to run the benchmark at full speed?
# --------------------------------------------------------------------------------------------

CPU_RESTRICTION_FIELDS = ("system_logical_cpu_count", "active_processor_count",
                          "process_cpu_affinity", "process_cpu_affinity_count",
                          "process_priority", "process_priority_class", "active_power_scheme",
                          "processor_max_state_percent", "power_overlay_scheme",
                          "process_power_throttling", "job_object_membership",
                          "job_cpu_rate_limit_percent")

# Which key of the audit says how each §6 field was obtained. Every field needs one because the
# difference between "this machine has no cap" and "this machine would not say" is the only thing a
# reader of a null can ask, and a list of bare nulls cannot answer it.
CPU_RESTRICTION_FIELD_STATE_KEYS = {
    "system_logical_cpu_count": "logical_cpu_count_state",
    "active_processor_count": "active_processor_count_state",
    "process_cpu_affinity": "affinity_state",
    "process_cpu_affinity_count": "affinity_state",
    "process_priority": "process_priority_state",
    "process_priority_class": "process_priority_state",
    "active_power_scheme": "power_scheme_state",
    "processor_max_state_percent": "processor_max_state_state",
    "power_overlay_scheme": "power_overlay_state",
    "process_power_throttling": "process_power_throttling_state",
    "job_object_membership": "job_object_state_name",
    "job_cpu_rate_limit_percent": "job_object_state_name",
}

# The states the readers above can answer with. `none_applied` and `no_job_object` are here because
# they are answers, not refusals: the API said so. `unreadable` and `unavailable` are the refusals,
# and a refusals is what §6 forbids writing as a number.
CPU_RESTRICTION_STATES = ("measured", UNAVAILABLE, "unreadable", "none_applied", "overlay_applied",
                          "no_job_object", "in_job_and_its_limits_are_not_queryable",
                          "alias_not_found_in_active_scheme",
                          "max_state_default_not_overridden",
                          "not_queryable_by_documented_api")

# The states that mean "the machine refused", as opposed to "the machine answered and the answer is
# no such thing". A refusal is recorded; §6 forbids turning it into a number.
CPU_RESTRICTION_REFUSALS = (UNAVAILABLE, "unreadable", "not_queryable_by_documented_api")

# A hard cap on how many logical cpus the process may be scheduled on is the difference between a
# measurement and a rumour about one, so these three reasons stop the round. §6's other fields are
# recorded and reported, and an unreadable one is never a failure by itself.
CPU_RESTRICTION_REASONS = ("cpu_affinity_restricted", "cpu_rate_limited", "cpu_power_cap_detected")

ALL_PROCESSOR_GROUPS = 0xFFFF

# The power setting §6 cares about, by the alias Microsoft documents for it. Not by GUID: a GUID
# typed from memory that turns out to be wrong does not raise, it reads as "no such setting", and
# the reading that follows looks like a machine that will not answer. `PROCTHROTTLEMAX` is both the
# documented alias and the one an OEM-renamed plan keeps, so this lookup has nothing to get wrong.
PROCESSOR_MAX_STATE_ALIAS = "PROCTHROTTLEMAX"
_GUID_IN_LINE_RE = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
_HEX_VALUE_RE = re.compile(r"0x([0-9a-fA-F]{1,8})\b")
_ALIAS_TAIL_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")

# A field the machine will not answer, in §6's own words: null, and named. Never 0 - a zero in a
# report about limits reads as "capped at nothing", which is the opposite of "unknown".
#
# The Windows priority classes, by mask. `ABOVE_NORMAL` is 0x8000 and `BELOW_NORMAL` is 0x4000;
# they are not neighbours of `NORMAL` (0x20) the way the names suggest, and a table that guessed
# would report a real priority class as `unknown_class_32768`.
WINDOWS_PRIORITY_CLASSES = ((0x100, "REALTIME_PRIORITY_CLASS"), (0x80, "HIGH_PRIORITY_CLASS"),
                            (0x8000, "ABOVE_NORMAL_PRIORITY_CLASS"), (0x20, "NORMAL_PRIORITY_CLASS"),
                            (0x4000, "BELOW_NORMAL_PRIORITY_CLASS"), (0x40, "IDLE_PRIORITY_CLASS"))

_WIN_DLL_CACHE = {}


def _win_dll(name):
    """A system DLL, or None if this machine will not hand one over.

    Cached because the audit runs once per round and an unavailable DLL is a fact about the
    machine, not a transient failure to retry.
    """
    if name not in _WIN_DLL_CACHE:
        try:
            _WIN_DLL_CACHE[name] = ctypes.WinDLL(name)
        except Exception:
            _WIN_DLL_CACHE[name] = None
    return _WIN_DLL_CACHE[name]


def cpu_count_logical():
    """How many logical cpus the machine has, or None if the answer is not forthcoming.

    psutil can report None on a machine it cannot interrogate; a None here is not 0, because the
    gate that compares it against an affinity count must not read "unknown" as "one cpu".
    """
    try:
        return psutil.cpu_count(logical=True)
    except Exception:
        return None


class _Guid(ctypes.Structure):
    _fields_ = [("data1", ctypes.c_uint32), ("data2", ctypes.c_uint16),
                ("data3", ctypes.c_uint16), ("data4", ctypes.c_ubyte * 8)]

    @classmethod
    def from_hex(cls, text):
        hexed = str(text).replace("-", "").replace("{", "").replace("}", "")
        if len(hexed) != 32:
            raise ValueError(f"not a guid: {text!r}")
        return cls(int(hexed[0:8], 16), int(hexed[8:12], 16), int(hexed[12:16], 16),
                   (ctypes.c_ubyte * 8).from_buffer_copy(bytes.fromhex(hexed[16:])))

    def text(self):
        tail = bytes(self.data4)
        return ("{%08X-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X}"
                % (self.data1, self.data2, self.data3, *[byte for byte in tail]))


def windows_power_scheme():
    """Which power plan is active, by GUID and by the name the Control Panel shows.

    Read-only. §6 forbids changing the plan, and this function has no code path that could.
    """
    out = {"active_power_scheme": None, "active_power_scheme_name": None,
           "power_scheme_state": UNAVAILABLE, "power_scheme_detail": None}
    powrprof = _win_dll("powrprof.dll")
    if powrprof is None:
        out["power_scheme_detail"] = "powrprof.dll could not be loaded"
        return out
    try:
        handle = ctypes.POINTER(_Guid)()
        if powrprof.PowerGetActiveScheme(None, ctypes.byref(handle)) != 0 or not handle:
            out["power_scheme_detail"] = "PowerGetActiveScheme refused the query"
            return out
        guid = handle.contents
        out["active_power_scheme"] = guid.text()
        out["power_scheme_state"] = "measured"
        size = ctypes.c_uint32(0)
        if powrprof.PowerReadFriendlyName(None, ctypes.byref(guid), None, None, None,
                                         ctypes.byref(size)) == 0 and size.value:
            buffer = ctypes.create_unicode_buffer(max(size.value // 2, 1))
            if powrprof.PowerReadFriendlyName(None, ctypes.byref(guid), None, None, buffer,
                                             ctypes.byref(size)) == 0:
                out["active_power_scheme_name"] = buffer.value or None
    except Exception as exc:
        out["power_scheme_detail"] = f"{type(exc).__name__}: {exc}"
    return out


def _powercfg(arguments):
    """Run `powercfg` read-only, decoded so ASCII survives whatever UI language it answers in.

    The tool prints in the console codepage, which on a Chinese Windows is not UTF-8: decoding it as
    UTF-8 replaces the label text, and the labels are exactly what a parser must not depend on.
    latin-1 leaves the ASCII bytes - GUIDs, aliases, `0x` values - intact and turns everything else
    into harmless noise, which is the only reason this function is not locale-sensitive.

    A non-zero exit is reported as a failure rather than as empty output. `powercfg` writes nothing
    to stdout when it rejects an argument, and a caller that read that as "the plan has no such
    setting" would publish a finding about the machine that is really a finding about the call.
    """
    try:
        proc = subprocess.run(["powercfg"] + list(arguments), capture_output=True, timeout=60)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return None, (f"powercfg {' '.join(arguments)} exited {proc.returncode}: "
                      f"{proc.stderr.decode('latin-1', 'replace').strip()[:120]}")
    return proc.stdout.decode("latin-1", "replace"), None


def power_scheme_guid_for_cli(guid):
    """A scheme GUID in the one spelling `powercfg` accepts: bare, and it is case-insensitive.

    `PowerGetActiveScheme` hands back `{A1841308-…}` with the braces, and `powercfg /query` refuses
    exactly that string. The two Windows APIs disagree about their own shared format, so the audit
    normalises rather than discovering this at the point where it looks like a missing setting.
    """
    return str(guid or "").strip().strip("{}").lower() or None


def powercfg_setting_value(alias, scheme_guid, text=None):
    """The AC and DC values of a power setting, found by its alias rather than by its GUID.

    `powercfg /query <plan>` prints the active plan as one block per group and setting: a GUID line,
    then that object's alias, then its numeric lines. Splitting the text on the GUID lines is what
    makes the lookup independent of the console language - the labels are the only part that is
    translated, and they are the part this function never reads. The last two numbers in a setting's
    block are its AC and DC indices, in that order, however many range lines come before them.

    A wrong GUID does not raise, it reads as "no such setting", and a reader cannot tell that apart
    from a machine that will not answer - which is why this resolves by the alias instead. Returns
    `(ac, dc, guid)` with any of them None; None values beside a guid mean the setting exists in the
    plan but carries no values, and no guid at all means the plan does not list it, which is "left at
    its default" rather than "zero".
    """
    if text is None:
        text, _failure = _powercfg(["/query", power_scheme_guid_for_cli(scheme_guid)])
        if text is None:
            return None, None, None
    blocks = [[]]
    for line in text.splitlines():
        if _GUID_IN_LINE_RE.search(line):
            blocks.append([line])
        else:
            blocks[-1].append(line)
    wanted = str(alias).upper()
    for block in blocks:
        header = _GUID_IN_LINE_RE.search(block[0]) if block else None
        if not header:
            continue
        numbers, named = [], None
        for line in block[1:]:
            tail = line.rsplit(":", 1)[-1].strip()
            if _ALIAS_TAIL_RE.match(tail):
                named = tail
            value = _HEX_VALUE_RE.search(line)
            if value:
                numbers.append(int(value.group(1), 16))
        if named != wanted:
            continue
        if len(numbers) < 2:
            return None, None, header.group(0)
        return float(numbers[-2]), float(numbers[-1]), header.group(0)
    return None, None, None


def windows_processor_max_state(scheme_guid=None):
    """The plan's processor performance cap, on AC and on battery, as a percentage.

    A percentage rather than a load, because the thing §6 is worried about is a setting somebody
    left behind: "maximum processor state 99%" disables turbo and halves every frame time without
    ever making the machine look busy. Both supply states are read and the lower one is published,
    since a round that runs on battery is capped by the battery value.
    """
    out = {"processor_max_state_percent": None, "processor_max_state_ac": None,
           "processor_max_state_dc": None, "processor_max_state_state": UNAVAILABLE,
           "processor_max_state_setting_guid": None, "processor_max_state_detail": None}
    scheme_guid = scheme_guid or windows_power_scheme().get("active_power_scheme")
    if not scheme_guid:
        out["processor_max_state_detail"] = "no active scheme to read the setting from"
        return out
    text, failure = _powercfg(["/query", power_scheme_guid_for_cli(scheme_guid)])
    if text is None:
        out["processor_max_state_state"] = "unreadable"
        out["processor_max_state_detail"] = failure
        return out
    ac, dc, guid = powercfg_setting_value(PROCESSOR_MAX_STATE_ALIAS, scheme_guid, text=text)
    out["processor_max_state_setting_guid"] = guid
    percents = []
    for name, value in (("ac", ac), ("dc", dc)):
        if value is None:
            continue
        out[f"processor_max_state_{name}"] = float(value)
        percents.append(float(value))
    if percents:
        out["processor_max_state_percent"] = min(percents)
        out["processor_max_state_state"] = "measured"
    else:
        # The alias was not in the plan's own list, which means the setting is at its group default
        # rather than that the machine refused to answer. `max_state_default_not_overridden` says
        # which, and neither it nor `unreadable` ever yields a number.
        out["processor_max_state_state"] = ("max_state_default_not_overridden" if guid else
                                           "alias_not_found_in_active_scheme")
    return out


def windows_power_overlay_scheme():
    """Whether a Windows 11 power overlay - the efficiency mode a user can slide into - is applied.

    The documented contract is that these two calls succeed *and hand back a null pointer* when no
    overlay is in effect, so a `None` here is a measurement rather than a missing field. It is
    labelled `none_applied` for that reason; a reader must not have to guess which kind of `None`
    they are looking at.
    """
    out = {"power_overlay_scheme": None, "power_overlay_state": UNAVAILABLE, "power_overlay_detail": None}
    powrprof = _win_dll("powrprof.dll")
    kernel32 = _win_dll("kernel32.dll")
    if powrprof is None:
        out["power_overlay_detail"] = "powrprof.dll could not be loaded"
        return out
    try:
        handle = ctypes.POINTER(_Guid)()
        code = int(powrprof.PowerGetActualOverlayScheme(ctypes.byref(handle)))
        if code != 0:
            out["power_overlay_detail"] = f"PowerGetActualOverlayScheme returned {code}"
            out["power_overlay_state"] = "unreadable"
            return out
        if handle:
            out["power_overlay_scheme"] = handle.contents.text()
            out["power_overlay_state"] = "overlay_applied"
            if kernel32 is not None:
                kernel32.GlobalFree(handle)
        else:
            out["power_overlay_state"] = "none_applied"
    except Exception as exc:
        out["power_overlay_detail"] = f"{type(exc).__name__}: {exc}"
    return out


def job_object_state(pid=None):
    """Whether this process runs inside a job object, and what that job's cpu limit is if it has one.

    `IsProcessInJob` with a null job handle is the documented way to ask the first question. The
    second needs a handle *to the job*, which a process that did not create its own job has no way
    to obtain - the job is usually unnamed, and `OpenJobObject` needs a name. So membership is
    measured and the limit is reported as unobtainable when there is a job to read, which §6 allows
    precisely because an unavailable field cannot also be an accusation.
    """
    out = {"job_object_membership": None, "job_cpu_rate_limit_percent": None,
           "job_object_state_name": UNAVAILABLE, "job_object_detail": None}
    kernel32 = _win_dll("kernel32.dll")
    if kernel32 is None:
        out["job_object_detail"] = "kernel32.dll could not be loaded"
        return out
    try:
        # `GetCurrentProcess` hands back the pseudo-handle -1, not a real one, so the prototype is
        # declared as c_int: as a c_void_p the value arrives signed, truncated or None depending on
        # the build, and an argument type turns -1 into the HANDLE the API actually wants.
        kernel32.GetCurrentProcess.restype = ctypes.c_int
        kernel32.IsProcessInJob.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.POINTER(ctypes.c_int)]
        kernel32.IsProcessInJob.restype = ctypes.c_int
        inside = ctypes.c_int(-1)
        if not kernel32.IsProcessInJob(ctypes.c_void_p(kernel32.GetCurrentProcess()), None,
                                      ctypes.byref(inside)):
            out["job_object_detail"] = f"IsProcessInJob failed with {ctypes.GetLastError()}"
            out["job_object_state_name"] = "unreadable"
            return out
        out["job_object_membership"] = bool(inside.value)
        out["job_object_state_name"] = ("no_job_object" if not inside.value
                            else "in_job_and_its_limits_are_not_queryable")
    except Exception as exc:
        out["job_object_detail"] = f"{type(exc).__name__}: {exc}"
    return out


def process_priority_class(pid=None):
    """This process' Windows priority class, by name.

    psutil reports `nice()` on Windows as the class mask; the mask is mapped so the artifact says
    `IDLE_PRIORITY_CLASS` rather than a number a reader has to look up.
    """
    out = {"process_priority": None, "process_priority_class": None,
           "process_priority_state": UNAVAILABLE, "process_priority_detail": None}
    try:
        process = psutil.Process(os.getpid() if pid is None else int(pid))
        with process.oneshot():
            priority = process.nice()
        out["process_priority"] = int(priority)
        for mask, name in WINDOWS_PRIORITY_CLASSES:
            if int(priority) == mask:
                out["process_priority_class"] = name
                break
        else:
            out["process_priority_class"] = f"unknown_class_{int(priority)}"
        out["process_priority_state"] = "measured"
    except Exception as exc:
        out["process_priority_detail"] = f"{type(exc).__name__}: {exc}"
    return out


def process_power_throttling_state():
    """Whether this process is in EcoQoS / power-throttling mode, as far as Windows will say.

    `PROCESS_POWER_THROTTLING_STATE` is set through `SetProcessInformation` and has no documented
    counterpart for reading it back, which is why Task Manager's "Eco mode" column is not something
    this instrument can promise. So the query is *attempted* - the two information classes that
    would answer it, asked of ntdll - and whatever the kernel returned is published. An artifact
    that says `not_queryable_by_documented_api` with the NTSTATUS beside it can be checked; one that
    says it in a code comment cannot.
    """
    out = {"process_power_throttling": None, "process_power_throttling_state": UNAVAILABLE,
           "process_power_throttling_detail": None}
    ntdll = _win_dll("ntdll.dll")
    if ntdll is None:
        out["process_power_throttling_detail"] = "ntdll.dll could not be loaded"
        return out

    class ThrottlingState(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("control_flags", ctypes.c_uint32),
                    ("throttle_duration_ms", ctypes.c_uint32)]

    try:
        query = ntdll.NtQueryInformationProcess
        query.restype = ctypes.c_long
        returned = ctypes.c_size_t(0)
        attempts = {}
        for info_class in (38, 39):
            state = ThrottlingState(1, 0, 0)
            status = int(query(ctypes.c_void_p(-1), ctypes.c_uint(info_class), ctypes.byref(state),
                              ctypes.sizeof(state), ctypes.byref(returned)))
            attempts[f"class_{info_class}"] = f"0x{status & 0xFFFFFFFF:08X}"
            if status == 0:
                out["process_power_throttling"] = int(state.control_flags)
                out["process_power_throttling_state"] = "measured"
        out["process_power_throttling_detail"] = attempts
        if out["process_power_throttling_state"] == UNAVAILABLE:
            out["process_power_throttling_state"] = "not_queryable_by_documented_api"
    except Exception as exc:
        out["process_power_throttling_detail"] = f"{type(exc).__name__}: {exc}"
    return out


def cpu_restriction_audit(pid=None, now=None, code_under_test=None):
    """§6's audit: the fields that could make every timing on this machine a lie, read once.

    Low utilisation is not the question. A machine that has been left with eight of its sixteen
    cpus, or a plan capped at 99 percent, or a job object with a rate limit, produces a benchmark
    that runs - slowly, evenly, and with a ratio nobody can trust. So this reads the configuration,
    not the load, and it reads it without writing a single setting: nothing here calls a
    Set*Information function, and §6 asks for that to stay true.

    Every field is published with the state it came back in, because the whole value of the artifact
    is that a reader can tell "the cap is 100 percent" from "the machine would not say".
    """
    now = utc_now() if now is None else now
    logical = cpu_count_logical()
    affinity = {"process_cpu_affinity": None, "process_cpu_affinity_count": None,
                "affinity_state": UNAVAILABLE, "affinity_detail": None}
    try:
        process = psutil.Process(os.getpid() if pid is None else int(pid))
        with process.oneshot():
            cpus = sorted(int(cpu) for cpu in process.cpu_affinity())
        affinity = {"process_cpu_affinity": cpus, "process_cpu_affinity_count": len(cpus),
                    "affinity_state": "measured", "affinity_detail": None}
    except Exception as exc:
        affinity["affinity_detail"] = f"{type(exc).__name__}: {exc}"
    active = None
    active_state = UNAVAILABLE
    kernel32 = _win_dll("kernel32.dll")
    if kernel32 is not None:
        try:
            # `0` is what this API answers when it cannot answer, so it is recorded as a refusal
            # rather than as a machine with no processors.
            counted = int(kernel32.GetActiveProcessorCount(ALL_PROCESSOR_GROUPS))
            active = counted or None
            active_state = "measured" if counted else UNAVAILABLE
        except Exception:
            active = None
    else:
        active_state = "unreadable"
    scheme = windows_power_scheme()
    document = {
        "code_under_test": code_under_test,
        "audited_utc": now,
        "pid": os.getpid() if pid is None else int(pid),
        "system_logical_cpu_count": logical,
        "logical_cpu_count_state": "measured" if logical else UNAVAILABLE,
        "active_processor_count": active,
        "active_processor_count_state": active_state,
        **affinity,
        **process_priority_class(pid),
        **scheme,
        # The plan's own GUID is what the cap is looked up in, so the read that needs it is handed
        # the read that produced it rather than asking the kernel twice for the same answer.
        **windows_processor_max_state(scheme.get("active_power_scheme")),
        **windows_power_overlay_scheme(),
        **job_object_state(pid),
        # §6 names EcoQoS alongside the throttling state; the attempt and its outcome are in the
        # fields below rather than in a comment, because "we could not read it" has to be checkable.
        **process_power_throttling_state(),
    }
    document["restrictions_detected"] = judge_cpu_restriction(document)
    document["cpu_restriction_ok"] = not document["restrictions_detected"]
    return document


def judge_cpu_restriction(audit):
    """§6's three hard gates, applied to an audit document.

    Takes the document rather than the machine so the same function answers for the live round and
    for the collector re-reading the artifact afterwards - and so a test can hand it an affinity of
    eight cpus without having to take a laptop apart. An unavailable field is not a pass either: it
    is simply not a reason, and it stays visible in the artifact.
    """
    audit = audit or {}
    reasons = []
    count, total = audit.get("process_cpu_affinity_count"), audit.get("system_logical_cpu_count")
    if count is not None and total is not None and count < total:
        reasons.append({"reason": "cpu_affinity_restricted",
                        "field": "process_cpu_affinity_count",
                        "actual": count, "expected": total,
                        "note": "the process may be scheduled on fewer cpus than the machine has"})
    limit = audit.get("job_cpu_rate_limit_percent")
    if limit is not None and float(limit) < 100.0:
        reasons.append({"reason": "cpu_rate_limited", "field": "job_cpu_rate_limit_percent",
                        "actual": limit, "expected": 100.0,
                        "note": "a job object is metering this process"})
    cap = audit.get("processor_max_state_percent")
    if cap is not None and float(cap) < 100.0:
        reasons.append({"reason": "cpu_power_cap_detected",
                        "field": "processor_max_state_percent",
                        "actual": cap, "expected": 100.0,
                        "note": "the power plan limits processor performance below full"})
    return reasons


# --------------------------------------------------------------------------------------------
# Why the 150-frame corroboration is not in the verdict (§20, §30)
# --------------------------------------------------------------------------------------------

CORROBORATION_REASONS = ("corroboration_agrees_with_primary_run",
                         "corroboration_conflicts_with_primary_run",
                         "corroboration_incomplete",
                         "corroboration_not_run_due_runtime_contamination",
                         "corroboration_not_run_due_preflight_failure",
                         "corroboration_not_run")


def corroboration_reason(primary_execution_count, primary_expected,
                         corroborating_execution_count, corroborating_expected,
                         primary_direction=None, corroborating_direction=None,
                         runtime_contaminated=False, preflight_passed=True):
    """One name for why the corroboration did not corroborate, and only one for a real conflict.

    A batch that was never launched and a batch that disagreed with the 60-frame run are different
    facts, and reading them as the same one is how a measurement that stopped at seven executions
    gets described as "the corroboration conflicts". The last name is the one a verdict may be
    refused over; everything before it is a statement about the experiment, not about the code.
    """
    primary_complete = bool(primary_expected) and primary_execution_count >= primary_expected
    corroborating_complete = (bool(corroborating_expected)
                              and corroborating_execution_count >= corroborating_expected)
    if not corroborating_execution_count:
        if runtime_contaminated:
            return "corroboration_not_run_due_runtime_contamination"
        if not preflight_passed:
            return "corroboration_not_run_due_preflight_failure"
        return "corroboration_not_run"
    if not primary_complete or not corroborating_complete:
        return "corroboration_incomplete"
    if primary_direction is None or corroborating_direction is None:
        return "corroboration_incomplete"
    return ("corroboration_agrees_with_primary_run"
            if primary_direction == corroborating_direction
            else "corroboration_conflicts_with_primary_run")


# --------------------------------------------------------------------------------------------
# CLI: the standalone form a driver or a human can run
# --------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="where to write benchmark_environment.json")
    parser.add_argument("--root", default=_ROOT, help="repository root, for the lock path")
    parser.add_argument("--lock-only", action="store_true",
                        help="take the lock, print the owner, release it and exit")
    parser.add_argument("--sample-once", action="store_true", help="print one reading")
    parser.add_argument("--hold-seconds", type=float, default=0.0,
                        help="with --lock-only: hold the lock this long, so a second owner can be "
                             "made to fail against a real holder")
    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    parser.add_argument("--max-wait-seconds", type=float, default=QUIET_WAIT_SECONDS)
    parser.add_argument("--code-under-test", default=None)
    parser.add_argument("--protocol", default=None,
                        help="the benchmark protocol name/version this round runs, recorded in the "
                             "lock metadata (§9)")
    args = parser.parse_args(argv)

    prime_cpu_percent()
    lock = BenchmarkLock(root=args.root, code_under_test=args.code_under_test, protocol=args.protocol,
                         command=" ".join(sys.argv))
    if args.sample_once:
        print(json.dumps(sample(), indent=2, sort_keys=True, default=str))
        return 0
    took = lock.acquire()
    print(json.dumps({"lock_acquired": took, "lock_owner": lock.owner,
                      "lock_path": lock.path, "waited_seconds": lock.waited_seconds},
                     indent=2, sort_keys=True))
    if not took:
        # §9: contention is a refusal inside five seconds, and the reason names the holder. Nothing
        # below this line runs - no sampling, no benchmark, no "proceed anyway and note it".
        return 2
    try:
        if args.lock_only:
            if args.hold_seconds:
                time.sleep(args.hold_seconds)
            return 0
        payload = quiet_preflight(window_seconds=args.window_seconds,
                                  max_wait_seconds=args.max_wait_seconds, lock=lock,
                                  code_under_test=args.code_under_test)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        print(json.dumps({key: payload[key] for key in
                          ("preflight_status", "preflight_window_count_attempted",
                           "preflight_window_that_passed", "cpu_median", "cpu_p95",
                           "gpu_util_median", "gpu_util_p95", "available_ram_min",
                           "not_ready_reason")}, indent=2, sort_keys=True))
        return 0 if payload["preflight_status"] == "PASS" else 2
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
