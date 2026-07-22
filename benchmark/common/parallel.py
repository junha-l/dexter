"""Parallel per-sample evaluation over GPU-sharing worker subprocesses.

Shared by the benchmark metric scripts. A metric stays a single file with a
``--worker`` mode. The orchestrator (`run_parallel`) splits the predictions into
``--parallel`` slices and re-spawns the same script with ``--worker`` per slice,
streams one ``@@RESULT@@`` json record per sample back over stdin/stdout,
aggregates under one progress bar, and returns the records. Worker processes get
their GPU, single-thread caps, and PYTHONPATH from the child environment set
here — so the caps land before the worker imports torch (avoiding the
parallel * cores OpenMP thread explosion).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil
from pathlib import Path

RESULT_MARKER = "@@RESULT@@"  # worker -> orchestrator per-sample result line prefix

_REPO_ROOT = str(Path(__file__).resolve().parents[2])


def emit(record: dict):
    """Worker side: stream one result record to the orchestrator."""
    sys.stdout.write(RESULT_MARKER + json.dumps(record) + "\n")
    sys.stdout.flush()


def read_slice() -> list:
    """Worker side: read this slice's samples from stdin."""
    return json.loads(sys.stdin.read())


def _worker_env(gpu: int) -> dict:
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONPATH": _REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(var, "1")
    return env


def run_parallel(script, samples, worker_args, gpu, parallel, label, fields) -> list:
    """Fan `samples` across `parallel` worker subprocesses and return their records.

    script:      __file__ of the calling metric; re-spawned with --worker per slice.
    worker_args: extra CLI args every worker needs (e.g. ["--data_path", ...]).
    fields:      {display_name: reducer(records) -> float} for the live progress bar.
    """
    from rich.progress import BarColumn, Progress, TextColumn

    total = len(samples)
    if total == 0:
        sys.exit("No predictions to evaluate.")

    parallel = min(parallel, total)
    per_proc = ceil(total / parallel)
    slices = [samples[i : i + per_proc] for i in range(0, total, per_proc)]
    print(f"Evaluating {total} samples across {len(slices)} workers on GPU {gpu}.")

    cmd = [sys.executable, os.path.abspath(script), "--worker", *worker_args]
    env = _worker_env(gpu)
    records, failed = [], []
    lock = threading.Lock()

    metrics_fmt = " • ".join(f"{k} {{task.fields[{k}]:.4f}}" for k in fields)
    progress = Progress(
        TextColumn(f"[bold]{label}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}" + (f" • {metrics_fmt}" if metrics_fmt else "")),
        transient=True,
    )

    def run_slice(slice_idx: int, slice_samples: list):
        proc = subprocess.Popen(
            cmd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )  # fmt: skip
        # Drain stderr in a side thread so a chatty worker can't deadlock on a full pipe.
        stderr_buf = []
        drainer = threading.Thread(target=lambda: stderr_buf.append(proc.stderr.read()), daemon=True)
        drainer.start()

        try:
            proc.stdin.write(json.dumps(slice_samples))
            proc.stdin.close()
        except BrokenPipeError:
            pass  # worker died before reading its slice; reported via returncode below

        for line in proc.stdout:
            if not line.startswith(RESULT_MARKER):
                continue  # ignore stray library output; only act on result markers
            rec = json.loads(line[len(RESULT_MARKER) :])
            with lock:
                records.append(rec)
                progress.update(task, advance=1, **{k: fn(records) for k, fn in fields.items()})

        proc.wait()
        drainer.join()
        if proc.returncode != 0:
            tail = ((stderr_buf[0] if stderr_buf else "") or "").strip().splitlines()[-1:] or [""]
            with lock:
                failed.append(slice_idx)
                progress.console.print(f"[red]worker {slice_idx} FAILED[/] {tail[0]}")

    with progress:
        task = progress.add_task("", total=total, **{k: 0.0 for k in fields})
        with ThreadPoolExecutor(max_workers=len(slices)) as ex:
            for fut in as_completed([ex.submit(run_slice, i, s) for i, s in enumerate(slices)]):
                fut.result()

    if failed:
        print(f"{len(failed)} worker(s) failed: {sorted(failed)}")
    return records
