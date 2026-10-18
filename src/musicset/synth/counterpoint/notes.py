"""
Rhythmic note model and species templates for the counterpoint engine
(Herremans & Sorensen 2012, Table 1 + Table 2 + Sec. 4.1).

A voice is a list of Note objects; rests are implicit gaps (only the
species opening measures have one). First-species voices may still be
given as plain MIDI-pitch sequences - `as_notes` normalizes them to
whole notes, which keeps the entire first-species API and its validated
behavior intact.

Durations are in quarter-note beats of a 4/4 measure (whole 4, half 2,
quarter 1, eighth 0.5); `beat` is the 1-based quarter-note offset
within the measure (the paper's Table 2 says "beat 1 to 16" without
defining the unit - a documented deviation).

The species ladder is encoded as rhythm templates: species 2-4 are
rhythm-restricted special cases of the fifth-species machinery,
evaluated under the same objective (V2/V3 already quantify per-value
dissonance treatment). Their cadence measures follow Fux / Salzer &
Schachter since the paper only defines the fifth-species one (Table 1
rule 5) - see chirality's docs/rules_herremans12.md for the deviations.

Table 1 feasibility (fifth species) implemented by `sample_skeleton`:
  1. key membership (enforced by the VNS pitch domain, not here);
  2. only allowed rhythmic patterns per measure - the paper never
     publishes the set; ours is {[2,2], [2,1,1], [1,1,2], [1,1,1,1]}
     plus one weak-beat quarter optionally split into an eighth pair;
  3. no immediate basic-pattern repetition (basic = ignoring ties and
     eighth decorations), and no more than two consecutive measures
     with the same basic pattern;
  4. first measure: half rest + half note;
  5. penultimate measure: tied quarter + two eighths + half (the
     preceding measure must therefore end with a tied half);
  6. last measure: whole note;
  7. ties connect notes of the same pitch (guaranteed by construction:
     a tie chain is a single pitch variable);
  8. a half may tie only to a half or a quarter;
  9. "maximum two measures of the same note value" - read as no more
     than two consecutive measures, eighth variations not counting.
"""

from collections.abc import Sequence
from enum import IntEnum
from typing import NamedTuple, cast

import numpy as np

TIE_NONE, TIE_START, TIE_END = 0, 1, 2

WHOLE, HALF, QUARTER, EIGHTH = 4.0, 2.0, 1.0, 0.5

BEATS_PER_MEASURE = 4.0


class Note(NamedTuple):
    """One note of a voice (the paper's Table 2, beat unit = quarters)."""

    pitch: int
    duration: float = WHOLE
    measure: int = 0
    beat: float = 1.0
    tied: int = TIE_NONE


type Voice = Sequence[int] | Sequence[Note]


def as_notes(voice: Voice) -> list[Note]:
    """Normalize a voice: plain pitches become one whole note per measure."""
    notes = list(voice)
    if not notes or isinstance(notes[0], Note):
        return cast(list[Note], notes)
    return [
        Note(pitch=int(cast(int, p)), duration=WHOLE, measure=i, beat=1.0)
        for i, p in enumerate(notes)
    ]


class Attack(NamedTuple):
    """A sounding event: a tie chain collapsed to one melodic note."""

    pitch: int
    duration: float  # summed over the tie chain
    measure: int  # onset measure
    beat: float  # onset beat
    indices: tuple[int, ...]  # positions of the member notes in the voice


def attacks(notes: Sequence[Note]) -> list[Attack]:
    """Collapse tie chains; H rules and VNS variables live on attacks."""
    result: list[Attack] = []
    for i, note in enumerate(notes):
        if note.tied == TIE_END and result:
            last = result[-1]
            result[-1] = last._replace(
                duration=last.duration + note.duration, indices=(*last.indices, i)
            )
        else:
            result.append(
                Attack(
                    pitch=note.pitch,
                    duration=note.duration,
                    measure=note.measure,
                    beat=note.beat,
                    indices=(i,),
                )
            )
    return result


def assign_pitches(skeleton: Sequence[Note], pitches: Sequence[int]) -> list[Note]:
    """Give each attack of a skeleton a pitch (tie chains stay unison)."""
    voice: list[Note] = []
    k = -1
    for note in skeleton:
        if note.tied != TIE_END:
            k += 1
        voice.append(note._replace(pitch=int(pitches[k])))
    if k + 1 != len(pitches):
        raise ValueError(f"{k + 1} attacks but {len(pitches)} pitches")
    return voice


def n_measures(notes: Sequence[Note]) -> int:
    return notes[-1].measure + 1 if notes else 0


# =============================================================================
# SPECIES TEMPLATES
# =============================================================================


class Species(IntEnum):
    FIRST = 1
    SECOND = 2
    THIRD = 3
    FOURTH = 4
    FIFTH = 5


# Fifth-species middle-measure base patterns (durations per measure).
_FIFTH_PATTERNS: tuple[tuple[float, ...], ...] = (
    (HALF, HALF),
    (HALF, QUARTER, QUARTER),
    (QUARTER, QUARTER, HALF),
    (QUARTER, QUARTER, QUARTER, QUARTER),
)


def _measure(
    measure: int,
    durations: Sequence[float],
    start_beat: float = 1.0,
    tie_first: int = TIE_NONE,
    tie_last: int = TIE_NONE,
) -> list[Note]:
    notes = []
    beat = start_beat
    for i, duration in enumerate(durations):
        tied = TIE_NONE
        if i == 0 and tie_first:
            tied = tie_first
        if i == len(durations) - 1 and tie_last:
            tied = tie_last
        notes.append(Note(pitch=0, duration=duration, measure=measure, beat=beat, tied=tied))
        beat += duration
    return notes


def _basic(durations: Sequence[float]) -> tuple[float, ...]:
    """Basic rhythmic pattern: eighth pairs read as their quarter."""
    merged: list[float] = []
    pending = 0.0
    for duration in durations:
        if duration == EIGHTH:
            pending += EIGHTH
            if pending == QUARTER:
                merged.append(QUARTER)
                pending = 0.0
        else:
            merged.append(duration)
    return tuple(merged)


def sample_skeleton(species: Species, measures: int, rng: np.random.Generator) -> list[Note]:
    """
    A feasible rhythmic skeleton (pitches 0): Table 1 for the fifth
    species, the classical templates for species 1-4.
    """
    minimum = 4 if species in (Species.FOURTH, Species.FIFTH) else 3
    if measures < minimum:
        raise ValueError(f"species {int(species)} needs >= {minimum} measures")

    if species == Species.FIRST:
        return [Note(pitch=0, duration=WHOLE, measure=i) for i in range(measures)]

    last = [Note(pitch=0, duration=WHOLE, measure=measures - 1)]

    if species == Species.SECOND:
        notes = _measure(0, [HALF], start_beat=3.0)  # half rest first
        for m in range(1, measures - 1):
            notes += _measure(m, [HALF, HALF])
        return notes + last

    if species == Species.THIRD:
        notes = _measure(0, [QUARTER] * 3, start_beat=2.0)  # quarter rest first
        for m in range(1, measures - 1):
            notes += _measure(m, [QUARTER] * 4)
        return notes + last

    if species == Species.FOURTH:
        # Syncopated chain: every barline crossed by a tie.
        notes = _measure(0, [HALF], start_beat=3.0, tie_last=TIE_START)
        for m in range(1, measures - 2):
            notes += _measure(m, [HALF, HALF], tie_first=TIE_END, tie_last=TIE_START)
        notes += _measure(measures - 2, [HALF, HALF], tie_first=TIE_END)
        return notes + last

    # Fifth species (Table 1). The measure before the penultimate must
    # end with a half tied into the suspension quarter (rules 5 + 8).
    notes = _measure(0, [HALF], start_beat=3.0)
    history: list[tuple[float, ...]] = []
    for m in range(1, measures - 2):
        final_half_forced = m == measures - 3
        while True:
            base = _FIFTH_PATTERNS[int(rng.integers(len(_FIFTH_PATTERNS)))]
            if final_half_forced and base[-1] != HALF:
                continue
            if history and base == history[-1]:
                continue  # no immediate basic-pattern repetition
            if len(history) >= 2 and base == history[-1] == history[-2]:
                continue  # never three in a row (rules 3 + 9)
            break
        history.append(base)

        durations = list(base)
        # One weak-beat quarter may split into an eighth pair.
        quarter_positions = [i for i, d in enumerate(durations) if d == QUARTER and i > 0]
        if quarter_positions and rng.random() < 0.3:
            at = quarter_positions[int(rng.integers(len(quarter_positions)))]
            durations[at : at + 1] = [EIGHTH, EIGHTH]

        tie_last = TIE_NONE
        if final_half_forced:
            tie_last = TIE_START  # feeds the penultimate suspension quarter
        elif durations[-1] == HALF and rng.random() < 0.25:
            tie_last = TIE_START  # rule 8: a half may tie into the next measure

        tie_first = TIE_END if notes[-1].tied == TIE_START else TIE_NONE
        notes += _measure(m, durations, tie_first=tie_first, tie_last=tie_last)

    # Penultimate measure (Table 1 rule 5): tied quarter, two eighths, half.
    notes += _measure(measures - 2, [QUARTER, EIGHTH, EIGHTH, HALF], tie_first=TIE_END)
    return notes + last
