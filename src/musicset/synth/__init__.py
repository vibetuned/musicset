"""Synthetic scores: MusicXML to feed ``musicset build``.

Generators (``musicset synth KIND OUT -n N``):

* ``exercise``: scales and chord progressions for piano (kakigori's and
  gelato's synthetic writer, curriculum switches included);
* ``piano``: a melody, chords or thirds over a left-hand accompaniment, on a
  functional chord progression (chirality's harmony and arpeggio rules);
* ``counterpoint``: two-voice species counterpoint from a pool composed by
  ``musicset counterpoint-pool`` (chirality's VNS), as piano, duet, one
  staff or choir, in any key;
* ``easy`` / ``melody``: one-measure and four-to-eight-measure scores for the
  first curriculum stages (gelato's easy and intermediate writers);
* ``mutate`` / ``arrange``: real scores with rare notation injected
  (kakigori's synthetic generator and arranger).

Every file has its own seed, derived from (kind, seed, index), so a batch is
reproducible and can be resumed or extended. Notation rates come from a
``Profile`` (``synth.decorate``): plain, default (gelato) or detector
(kakigori).

Requires the ``synth`` extra (music21).
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

KINDS = ("exercise", "piano", "counterpoint", "easy", "melody", "mutate", "arrange")
SOURCE_KINDS = ("mutate", "arrange")


@dataclass(frozen=True)
class SynthTask:
    kind: str
    index: int
    out: Path
    seed: int = 0
    profile: str = "default"
    options: tuple[tuple[str, Any], ...] = ()
    """Generator options as sorted (name, value) pairs."""
    source: Path | None = None
    """The real score, for mutate and arrange."""
    pool: Path | None = None
    """The counterpoint pool."""

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def path(self) -> Path:
        if self.source is not None:
            return self.out / f"{self.kind}_{self.index:06d}_{self.source.stem}.mxl"
        return self.out / f"{self.kind}_{self.index:06d}.mxl"

    def rng(self) -> random.Random:
        return random.Random(f"{self.kind}/{self.seed}/{self.index}")


@cache
def _pool(path: Path) -> list[dict]:
    from musicset.synth.species import load_pool

    pool = load_pool(path)
    if not pool:
        raise ValueError(f"empty counterpoint pool {path}")
    return pool


def generate(task: SynthTask):
    """The task's score as a music21 Score (not written)."""
    import music21

    from musicset.synth import exercises, mutate, piano, simple, species
    from musicset.synth.decorate import profile

    rng = task.rng()
    options = dict(task.options)
    if task.kind == "exercise":
        return exercises.generate(
            rng, exercises.ExerciseConfig.from_options(profile=task.profile, **options)
        )
    if task.kind == "piano":
        return piano.generate(rng, profile(task.profile), piano.PianoConfig(**options))
    if task.kind == "counterpoint":
        if task.pool is None:
            raise ValueError("counterpoint needs a pool (musicset counterpoint-pool)")
        pool = _pool(task.pool)
        return species.piece_score(rng.choice(pool), rng, profile(task.profile))
    if task.kind == "easy":
        return simple.easy(rng)
    if task.kind == "melody":
        return simple.melody(rng)
    if task.kind in SOURCE_KINDS:
        if task.source is None:
            raise ValueError(f"{task.kind} needs a source score")
        score = music21.converter.parse(task.source)
        if not isinstance(score, music21.stream.Score):
            raise ValueError(f"{task.source.name}: not a score")
        if task.kind == "mutate":
            return mutate.mutate(score, rng, **options)
        return mutate.arrange(score, rng)
    raise ValueError(f"unknown kind {task.kind!r}: {KINDS}")


def synthesize(task: SynthTask) -> str:
    """Generate and write one file; returns its name (process-pool entry point)."""
    warnings.filterwarnings("ignore")
    path = task.path
    if path.exists():
        return path.name
    from musicset.synth.write import title, write_score

    score = generate(task)
    if task.kind in SOURCE_KINDS:
        write_score(score, path, notation="music21" if task.kind == "mutate" else "none")
    else:
        title(score, random.Random(f"title/{task.kind}/{task.seed}/{task.index}"))
        write_score(score, path)
    return path.name


__all__ = ["KINDS", "SOURCE_KINDS", "SynthTask", "generate", "synthesize"]
