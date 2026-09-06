"""Diagnose a hung training run without ptrace.

Kaggle's container blocks ptrace, so py-spy / gdb / faulthandler-via-signal on an
already-running old build are all unavailable. Everything here reads /proc and
/sys only, which is always permitted.

    python diagnose_hang.py                 # auto-detect run_experiments.py PIDs
    python diagnose_hang.py --pid 227 228
    python diagnose_hang.py --sample-seconds 5

What the output tells you:

* Thread STATE column
    R  running          - NCCL busy-waits in R. Many R threads at ~100% CPU is a
                          spinning collective, not a dead one.
    S  sleeping         - blocked on a futex, pipe or socket.
    D  uninterruptible  - blocked in the kernel on I/O; look at DISK below.
* CPU delta over the sample window
    ~0 on every thread   -> nothing is spinning: a lock/pipe/IO deadlock.
    high on one thread   -> a collective is spinning while its peer never posted.
* WCHAN - the kernel function the thread is parked in.
    futex_wait / do_epoll_wait / pipe_read -> waiting on another process
    (a DataLoader worker, or the peer rank).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _read(path: str) -> str:
    try:
        return Path(path).read_text(errors="replace").strip()
    except OSError:
        return ""


def _find_pids(pattern: str) -> list[int]:
    proc = Path("/proc")
    if not proc.is_dir():
        print("This tool reads /proc and only runs on Linux (Kaggle). "
              f"No /proc on {sys.platform}.")
        return []
    pids = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        cmdline = _read(f"/proc/{entry.name}/cmdline").replace("\x00", " ")
        if pattern in cmdline and "diagnose_hang" not in cmdline:
            pids.append(int(entry.name))
    return sorted(pids)


def _threads(pid: int) -> list[int]:
    task = Path(f"/proc/{pid}/task")
    if not task.is_dir():
        return []
    return sorted(int(t.name) for t in task.iterdir() if t.name.isdigit())


def _thread_stat(pid: int, tid: int) -> tuple[str, int, str]:
    """Return (state, utime+stime in ticks, thread name)."""
    raw = _read(f"/proc/{pid}/task/{tid}/stat")
    if not raw:
        return "?", 0, "?"
    # comm may contain spaces/parentheses; split on the last ')'.
    close = raw.rfind(")")
    name = raw[raw.find("(") + 1: close]
    fields = raw[close + 2:].split()
    state = fields[0]
    utime, stime = int(fields[11]), int(fields[12])
    return state, utime + stime, name


def sample(pids: list[int], seconds: float) -> None:
    ticks = os.sysconf("SC_CLK_TCK")
    first: dict[tuple[int, int], int] = {}
    for pid in pids:
        for tid in _threads(pid):
            first[(pid, tid)] = _thread_stat(pid, tid)[1]

    time.sleep(seconds)

    for pid in pids:
        cmdline = _read(f"/proc/{pid}/cmdline").replace("\x00", " ")
        print(f"\n=== pid {pid} ===")
        print(f"cmd: {cmdline[:160]}")
        status = dict(
            line.split(":", 1)
            for line in _read(f"/proc/{pid}/status").splitlines()
            if ":" in line
        )
        print(f"state: {status.get('State', '?').strip()}   "
              f"threads: {status.get('Threads', '?').strip()}   "
              f"RSS: {status.get('VmRSS', '?').strip()}")
        try:
            fds = len(list(Path(f"/proc/{pid}/fd").iterdir()))
        except OSError:
            fds = -1
        print(f"open fds: {fds}")

        print(f"{'TID':>8}  {'STATE':<5} {'CPU%':>7}  {'WCHAN':<24} NAME")
        busy = 0
        for tid in _threads(pid):
            state, now, name = _thread_stat(pid, tid)
            before = first.get((pid, tid), now)
            cpu = (now - before) / ticks / seconds * 100.0
            if cpu > 5.0:
                busy += 1
            wchan = _read(f"/proc/{pid}/task/{tid}/wchan") or "-"
            print(f"{tid:>8}  {state:<5} {cpu:>6.1f}%  {wchan[:24]:<24} {name}")
        print(f"threads burning CPU (>5%): {busy}")


def environment() -> None:
    print("\n=== environment ===")
    for name in (
        "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_DEBUG",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING", "TORCH_DISTRIBUTED_DEBUG",
        "MIPIC_NUM_WORKERS", "MIPIC_BATCH_SIZE", "MIPIC_DDP_TIMEOUT_MIN",
    ):
        print(f"  {name}={os.environ.get(name, '<unset>')}")

    print("\n=== /dev/shm (NCCL shared-memory transport + DataLoader workers) ===")
    try:
        usage = shutil.disk_usage("/dev/shm")
        print(f"  total {usage.total / 2**20:.0f} MiB, "
              f"used {(usage.total - usage.free) / 2**20:.0f} MiB, "
              f"free {usage.free / 2**20:.0f} MiB")
        if usage.free < 64 * 2**20:
            print("  WARNING: /dev/shm nearly full -> set NCCL_SHM_DISABLE=1 "
                  "and MIPIC_NUM_WORKERS=0")
    except OSError as exc:
        print(f"  unavailable: {exc}")

    for label, path in (("working dir", "/kaggle/working"), ("root", "/")):
        if Path(path).exists():
            usage = shutil.disk_usage(path)
            print(f"\n=== disk: {label} ({path}) ===")
            print(f"  free {usage.free / 2**30:.1f} GiB of {usage.total / 2**30:.1f} GiB")
            if usage.free < 2 * 2**30:
                print("  WARNING: low disk -> torch.save can block")

    print("\n=== GPUs ===")
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        print("  " + (out.stdout.strip().replace("\n", "\n  ") or "no output"))
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  nvidia-smi unavailable: {exc}")


def verdict(pids: list[int]) -> None:
    print("\n=== how to read this ===")
    print("  every thread ~0% CPU, states S/D   -> not a spinning collective.")
    print("     WCHAN futex/pipe/epoll on the main thread: DataLoader worker or")
    print("     peer-rank deadlock. Retry with MIPIC_NUM_WORKERS=0.")
    print("     WCHAN in a D state: blocked on disk. Check the DISK section.")
    print("  one or more threads pinned near 100% in state R")
    print("     -> a NCCL collective is spinning for a peer that never posted.")
    print("     Retry with MIPIC_DDP_STATIC_GRAPH=1 and NCCL_P2P_DISABLE=1.")
    print("  only one rank present -> the other died; scroll the log for its error.")
    if len(pids) == 1:
        print("\n  NOTE: only one training process found. Its peer is gone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, nargs="*", default=None)
    parser.add_argument("--pattern", default="run_experiments.py")
    parser.add_argument("--sample-seconds", type=float, default=3.0)
    args = parser.parse_args()

    pids = args.pid if args.pid else _find_pids(args.pattern)
    if not pids:
        print(f"No process matching {args.pattern!r} found.")
        sys.exit(1)
    print(f"Inspecting pids: {pids}  (sampling {args.sample_seconds:g}s)")
    sample(pids, args.sample_seconds)
    environment()
    verdict(pids)
