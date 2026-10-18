"""Parallel execution with crash isolation.

Verovio is C++ and can segfault, abort or loop forever on a malformed score,
so each task runs in its own short-lived process: a crash or a timeout costs
that one task, never the run. Processes come from a forkserver with the heavy
modules preloaded, so starting one costs a fork, not an interpreter boot.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from multiprocessing.connection import Connection
from typing import Any

logger = logging.getLogger(__name__)

WORKERS_ENV = "MUSICSET_WORKERS"
_PRELOAD = ["musicset.pipeline", "verovio", "cairosvg", "svgpathtools"]


def worker_count(reserve: int = 2, fraction: float = 0.5) -> int:
    """Workers for one stage: ``$MUSICSET_WORKERS``, else half the spare cores.

    Half, not all: a sustained full-core render build has taken a machine down
    with a hardware fault before (see gelato's worker policy).
    """
    env = os.environ.get(WORKERS_ENV, "").strip()
    if env:
        try:
            if (n := int(env)) > 0:
                return n
        except ValueError:
            pass
        logger.warning("%s=%r is not a positive integer; ignoring it", WORKERS_ENV, env)
    return max(1, int(((os.cpu_count() or 1) - reserve) * fraction))


@dataclass(slots=True)
class Outcome[T]:
    ok: bool
    value: T | None = None
    error: str | None = None


def _context() -> Any:
    ctx = mp.get_context("forkserver")
    ctx.set_forkserver_preload(_PRELOAD)
    return ctx


def _child(conn: Connection, fn: Callable[[Any], Any], arg: Any) -> None:
    try:
        conn.send((True, fn(arg)))
    except BaseException as e:
        conn.send((False, f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


def run_isolated[A, T](
    fn: Callable[[A], T], arg: A, *, timeout: float = 60.0, retries: int = 0, ctx: Any = None
) -> Outcome[T]:
    """Run ``fn(arg)`` in a fresh process. ``fn`` must be a module-level function."""
    ctx = ctx or _context()
    error = "not run"
    for _ in range(retries + 1):
        receiver, sender = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_child, args=(sender, fn, arg), daemon=True)
        proc.start()
        sender.close()
        try:
            # Receive before joining: a child blocks on exit until its result
            # has been read, so join-then-read deadlocks on large results
            if receiver.poll(timeout):
                ok, payload = receiver.recv()
                proc.join()
                if ok:
                    return Outcome(True, value=payload)
                # A Python exception is deterministic: retrying will not help
                return Outcome(False, error=payload)
            error = f"timed out after {timeout:.0f}s"
        except EOFError:
            proc.join()
            error = f"crashed (exit code {proc.exitcode})"
        finally:
            receiver.close()
            if proc.is_alive():
                proc.kill()
                proc.join()
    return Outcome(False, error=error)


def _guarded[A, T](fn: Callable[[A], T], arg: A) -> Outcome[T]:
    try:
        return Outcome(True, value=fn(arg))
    except Exception as e:
        return Outcome(False, error=f"{type(e).__name__}: {e}")


def map_pool[A, T](
    fn: Callable[[A], T],
    args: Iterable[A],
    *,
    workers: int | None = None,
    chunksize: int = 16,
) -> Iterator[tuple[A, Outcome[T]]]:
    """Plain process pool for pure-Python work that cannot crash the interpreter."""
    items = list(args)
    with ProcessPoolExecutor(max_workers=workers or worker_count(), mp_context=_context()) as pool:
        results = pool.map(partial(_guarded, fn), items, chunksize=chunksize)
        yield from zip(items, results, strict=True)


def map_isolated[A, T](
    fn: Callable[[A], T],
    args: Iterable[A],
    *,
    workers: int | None = None,
    timeout: float = 60.0,
    retries: int = 1,
) -> Iterator[tuple[A, Outcome[T]]]:
    """``run_isolated`` over many inputs, ``workers`` at a time, yielding as they finish."""
    ctx = _context()

    def run(arg: A) -> Outcome[T]:
        return run_isolated(fn, arg, timeout=timeout, retries=retries, ctx=ctx)

    with ThreadPoolExecutor(max_workers=workers or worker_count()) as pool:
        futures = {pool.submit(run, arg): arg for arg in args}
        for future in as_completed(futures):
            yield futures[future], future.result()
