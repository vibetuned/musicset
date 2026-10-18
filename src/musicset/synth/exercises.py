"""Piano exercises: scales and chord progressions, one or two staves.

The synthetic writer of kakigori and gelato, merged: gelato's curriculum
switches (``ExerciseConfig``) with the notation rates of a ``Profile``
(kakigori's writer is the ``detector`` profile).

A scale exercise would be fixed by its scale, key, meter and hand, so its
register, span, direction and length are drawn per score: the entropy sits in
the notes, where no curriculum switch can strip it out.
"""

from __future__ import annotations

import contextlib
import copy
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import music21

from musicset.synth.decorate import (
    Profile,
    add_clef_changes,
    add_double_barlines,
    add_octave_clefs,
    add_ottavas,
    add_slurs,
    decorate_chord,
    finger,
    ornament_note,
    pick_duration,
    profile,
)
from musicset.synth.theory import (
    MAJOR_KEYS,
    METER_SETS,
    MINOR_KEYS,
    SCALE_KINDS,
    CircleOfFifths,
    bar_length,
    chord_name,
    chord_pitches,
    octave,
    scale_fingering,
    scale_run,
)

CONTENT_TYPES = (*SCALE_KINDS, "chord_progression")
MAX_MEASURES = 64


@dataclass(frozen=True)
class ExerciseConfig:
    """Curriculum switches: what may appear at all."""

    staves: int = 2
    """1: right hand only; 2: grand staff."""
    fingerings: bool = True
    ornaments: bool = True
    """Ornaments, articulations and arpeggio signs."""
    dynamics: bool = True
    slurs: bool = True
    ties: bool = True
    meters: tuple[str, ...] | None = None
    """None: the profile's meters."""
    contents: tuple[str, ...] = CONTENT_TYPES
    min_measures: int = 12
    max_measures: int = 32
    profile: Profile = field(default_factory=lambda: profile("default"))

    @classmethod
    def from_options(cls, **options) -> ExerciseConfig:
        """Build from CLI-style options: ``meters`` may name a set (simple,
        medium, full, detector), ``profile`` a profile name."""
        if isinstance(options.get("meters"), str):
            options["meters"] = METER_SETS[options["meters"]]
        if isinstance(options.get("profile"), str):
            options["profile"] = profile(options["profile"])
        return cls(**{k: v for k, v in options.items() if v is not None})


@dataclass(frozen=True)
class _Shape:
    one_octave: bool
    octave_shift: int
    direction: str
    measures: int


def _shape(cfg: ExerciseConfig, rng: random.Random) -> _Shape:
    """Register, span, direction and length, shared by both hands."""
    return _Shape(
        one_octave=rng.random() < 0.40,
        octave_shift=rng.choice((-1, 0, 0, 1)),
        direction=rng.choice(("both", "both", "up", "down")),
        measures=rng.randint(cfg.min_measures, max(cfg.min_measures, cfg.max_measures)),
    )


# Written range of a scale run, MIDI: up to A6 / G4 above, down to G3 / E1 below
RUN_RANGE = {"RH": (55, 93), "LH": (28, 67)}


def _apply_shape(
    pitches: list, fingers: list[int], shape: _Shape, hand: str
) -> tuple[list, list[int]]:
    n_up = (len(pitches) + 1) // 2
    up_p, up_f = list(pitches[:n_up]), list(fingers[:n_up])
    if shape.one_octave and len(up_p) >= 8:
        up_p, up_f = up_p[:8], up_f[:8]
    # the drawn octave shift, or the nearest that keeps the run on the staff
    # (a B major run shifted up would climb to B7, seven ledger lines high)
    low, high = RUN_RANGE[hand]
    shifts = sorted(range(-2, 3), key=lambda s: (abs(s - shape.octave_shift), abs(s)))
    shift = next(
        (s for s in shifts if up_p[0].midi + 12 * s >= low and up_p[-1].midi + 12 * s <= high),
        0,
    )
    if shift:
        up_p = [octave(p, shift) for p in up_p]
    if shape.direction == "up":
        return up_p, up_f
    if shape.direction == "down":
        return up_p[::-1], up_f[::-1]
    return up_p + up_p[-2::-1], up_f + up_f[-2::-1]


def _repeat_to(pitches: list, fingers: list[int], per_measure: float, measures: int):
    want = round(measures * per_measure)
    if per_measure <= 0 or not pitches or want <= len(pitches):
        return pitches, fingers
    reps = -(-want // len(pitches))
    return (pitches * reps)[:want], (fingers * reps)[:want]


def _first_measure(meter: str, key: music21.key.Key, hand: str) -> music21.stream.Measure:
    m = music21.stream.Measure(number=1)
    m.timeSignature = music21.meter.TimeSignature(meter)
    m.keySignature = music21.key.KeySignature(key.sharps)
    m.clef = music21.clef.TrebleClef() if hand == "RH" else music21.clef.BassClef()
    return m


def _pack(
    events: Sequence[music21.note.GeneralNote], meter: str, key: music21.key.Key, hand: str
) -> list[music21.stream.Measure]:
    """Fill measures in order, padding the last one with a rest."""
    bar = bar_length(meter)
    measures = [_first_measure(meter, key, hand)]
    offset = 0.0
    for e in events:
        if offset + e.quarterLength > bar + 1e-3:
            measures.append(music21.stream.Measure(number=len(measures) + 1))
            offset = 0.0
        measures[-1].insert(offset, e)
        offset += e.quarterLength
    if bar - offset > 1e-3:
        measures[-1].insert(offset, music21.note.Rest(quarterLength=bar - offset))
    return measures


def _scale_measures(
    pitches: list,
    fingers: list[int],
    meter: str,
    key: music21.key.Key,
    hand: str,
    cfg: ExerciseConfig,
    fingered: bool,
    measures: int,
    rng: random.Random,
) -> list[music21.stream.Measure]:
    p = cfg.profile
    bar = bar_length(meter)
    value = pick_duration(p, bar, rng)
    pitches, fingers = _repeat_to(pitches, fingers, bar / value, measures)
    events: list[music21.note.Note] = []
    for pitch, f in zip(pitches, fingers, strict=True):
        n = music21.note.Note(copy.deepcopy(pitch), quarterLength=value)
        if fingered:
            finger(n, p, rng, [f])
        if cfg.ornaments:
            ornament_note(n, p, rng)
        prev = events[-1] if events else None
        if (
            cfg.ties
            and prev is not None
            and prev.pitch.midi == n.pitch.midi
            and rng.random() < p.tie
        ):
            prev.tie, n.tie = music21.tie.Tie("start"), music21.tie.Tie("stop")
        events.append(n)
    return _pack(events, meter, key, hand)


def _chord_measures(
    progression: list[str],
    meter: str,
    key: music21.key.Key,
    hand: str,
    cfg: ExerciseConfig,
    fingered: bool,
    rng: random.Random,
) -> list[music21.stream.Measure]:
    p = cfg.profile
    bar = bar_length(meter)
    value = pick_duration(p, bar, rng)
    octave = 4 if hand == "RH" else 3
    events: list[music21.note.GeneralNote] = []
    offset = 0.0
    previous = None
    for name in progression:
        c = music21.chord.Chord(chord_pitches(*chord_name(name), octave), quarterLength=value)
        if fingered:
            finger(c, p, rng)
        if cfg.ornaments:
            decorate_chord(c, p, rng)
        if offset + value > bar + 1e-3:
            offset, previous = 0.0, None  # repeated chords tie within a measure only
        if cfg.ties and previous == name and rng.random() < p.chord_tie:
            events[-1].tie, c.tie = music21.tie.Tie("start"), music21.tie.Tie("stop")
        events.append(c)
        offset += value
        previous = name
    return _pack(events, meter, key, hand)


def _equalize(a: list, b: list) -> tuple[list, list]:
    """Repeat each hand's measures to a common length (their LCM, capped)."""
    if len(a) == len(b):
        return a, b
    target = math.lcm(len(a), len(b))
    if target > MAX_MEASURES:
        target = max(len(a), len(b))

    def grow(ms: list) -> list:
        out = list(ms)
        while len(out) < target:
            for m in ms:
                if len(out) >= target:
                    break
                c = copy.deepcopy(m)
                c.number = len(out) + 1
                # the opening clef, key and meter are not restated
                c.removeByClass(
                    [music21.clef.Clef, music21.key.KeySignature, music21.meter.TimeSignature]
                )
                out.append(c)
        return out

    return grow(a), grow(b)


def _dynamics(measures: list[music21.stream.Measure], p: Profile, rng: random.Random) -> None:
    for m in measures:
        notes = list(m.notes)
        if notes and rng.random() < p.dynamic:
            m.insert(rng.choice(notes).offset, music21.dynamics.Dynamic(rng.choice(p.dynamics)))


def generate(rng: random.Random, cfg: ExerciseConfig | None = None) -> music21.stream.Score:
    cfg = cfg or ExerciseConfig()
    p = cfg.profile
    fingered = cfg.fingerings and rng.random() < p.fingered
    content = rng.choice(cfg.contents)
    if content in ("harmonic_minor", "melodic_minor"):
        key = music21.key.Key(rng.choice(MINOR_KEYS), "minor")
    else:
        key = music21.key.Key(rng.choice(MAJOR_KEYS))
    meter = rng.choice(cfg.meters or p.meters)
    hands = ["RH", "LH"] if cfg.staves == 2 else ["RH"]

    by_hand: dict[str, list[music21.stream.Measure]] = {}
    if content == "chord_progression":
        count = rng.randint(max(2, cfg.min_measures), max(cfg.min_measures + 1, cfg.max_measures))
        chords = CircleOfFifths().walk(key.tonic.name.replace("-", "b"), count, rng)
        for hand in hands:
            by_hand[hand] = _chord_measures(chords, meter, key, hand, cfg, fingered, rng)
    else:
        shape = _shape(cfg, rng)
        for hand in hands:
            pitches = scale_run(key, content, hand)
            fingers = scale_fingering(key, content, hand, pitches)
            pitches, fingers = _apply_shape(pitches, fingers, shape, hand)
            by_hand[hand] = _scale_measures(
                pitches, fingers, meter, key, hand, cfg, fingered, shape.measures, rng
            )
    if len(hands) == 2:
        by_hand["RH"], by_hand["LH"] = _equalize(by_hand["RH"], by_hand["LH"])

    score = music21.stream.Score()
    # a grand staff is one part on two staves, braced
    staves = [
        music21.stream.PartStaff() if len(hands) == 2 else music21.stream.Part() for _ in hands
    ]
    staves[0].insert(0, music21.instrument.Piano())
    for hand, part in zip(hands, staves, strict=True):
        if cfg.dynamics:
            _dynamics(by_hand[hand], p, rng)
        for m in by_hand[hand]:
            part.append(m)
        if cfg.slurs:
            add_slurs(part, p, rng)
        with contextlib.suppress(music21.exceptions21.Music21Exception):
            part.makeBeams(inPlace=True)
        score.insert(0, part)
    if len(staves) == 2:
        score.insert(0, music21.layout.StaffGroup(staves, symbol="brace", barTogether=True))
    add_double_barlines(score, p.double_barline, rng)
    add_clef_changes(score, p.clef_change, rng)
    add_octave_clefs(score, p.octave_clef, p.octave_8va, rng)
    add_ottavas(score, p.ottava, rng)
    return score
