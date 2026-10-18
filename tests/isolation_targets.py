"""Module-level task functions for the crash-isolation tests (children import them)."""

import os
import time


def square(x: int) -> int:
    return x * x


def fail(_: int) -> None:
    raise ValueError("bad score")


def crash(_: int) -> None:
    os.abort()


def hang(_: int) -> None:
    time.sleep(60)
