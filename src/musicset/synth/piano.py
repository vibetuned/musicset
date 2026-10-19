"""Piano pieces over a functional chord progression.

The harmony comes from ``theory.progression`` (chirality's rules: cadences,
secondary dominants, sevenths, suspensions, pivot modulations) voiced with
minimal motion. Over it the right hand plays a melody (chord tones on the
beats, steps between), block chords or a melody in thirds, and the left hand
an accompaniment pattern: block chords, Alberti bass, broken chords, oom-pah,
octaves or rising arpeggios (chirality's arpeggio patterns).
"""

from __future__ import annotations

import contextlib
import copy
import random
from collections.abc import Iterator
from dataclasses import dataclass

import music21

from musicset.synth.decorate import Profile, decorate, profile
from musicset.synth.theory import (
    ALL_MAJOR_KEYS,
    ALL_MINOR_KEYS,
    Harmony,
    bar_length,
    chord_pitches,
    is_compound,
    octave,
    progression,
    voice_chord,
)

METERS = (
    ("4/4", 0.28), ("3/4", 0.22), ("2/4", 0.14), ("6/8", 0.14), ("C", 0.06), ("cut", 0.04),
    ("3/8", 0.04), ("12/8", 0.04), ("2/2", 0.04),
)  # fmt: skip
RIGHT = (("melody", 0.6), ("chords", 0.2), ("thirds", 0.2))
LEFT = (
    ("block", 0.2), ("alberti", 0.2), ("broken", 0.15), ("oompah", 0.2), ("octaves", 0.1),
    ("arpeggio", 0.15),
)  # fmt: skip
TEMPI = (
    ("Largo", 50), ("Adagio", 66), ("Andante", 80), ("Moderato", 100), ("Allegretto", 112),
    ("Allegro", 132), ("Vivace", 152), ("Presto", 176),
)  # fmt: skip
RH_RANGE = (60, 84)
LH_RANGE = (36, 62)

# One beat's rhythm: simple beats are quarters, compound beats dotted quarters
SIMPLE_BEAT = (
    ((1.0,), 0.3), ((0.5, 0.5), 0.3), ((0.75, 0.25), 0.1), ((0.25, 0.25, 0.25, 0.25), 0.1),
    ((0.5, 0.25, 0.25), 0.1), ((0.25, 0.25, 0.5), 0.1),
    ((1 / 3,) * 3, 0.08),  # triplet eighths (music21 reads 1/3 as a triplet)
)  # fmt: skip
COMPOUND_BEAT = (
    ((1.5,), 0.3), ((0.5, 0.5, 0.5), 0.3), ((1.0, 0.5), 0.25), ((0.75, 0.25, 0.5), 0.1),
    ((0.5, 1.0), 0.05),
)  # fmt: skip


@dataclass(frozen=True)
class PianoConfig:
    min_measures: int = 8
    max_measures: int = 24
    minor: float = 0.3
    """Share of pieces in a minor key."""
    tempo: float = 0.5
    """Share of pieces with a tempo mark."""


def _pick(rng: random.Random, items: tuple) -> str:
    return rng.choices([i[0] for i in items], weights=[i[1] for i in items])[0]


def _beat(meter: str) -> float:
    return 1.5 if is_compound(meter) or meter == "3/8" else 1.0


def _rhythm(bar: float, beat: float, rng: random.Random, final: bool = False) -> list[float]:
    if final:
        return [bar]
    patterns = COMPOUND_BEAT if beat == 1.5 else SIMPLE_BEAT
    out: list[float] = []
    left = bar
    while left > 1e-6:
        if left >= 2 * beat and rng.random() < 0.15:
            out.append(2 * beat)
            left -= 2 * beat
        elif left >= beat:
            out.extend(rng.choices([p for p, _ in patterns], weights=[w for _, w in patterns])[0])
            left -= beat
        else:
            out.append(left)
            left = 0
    return out


def _in_range(pitches: list[music21.pitch.Pitch], low: int, high: int) -> list[music21.pitch.Pitch]:
    """Every octave of ``pitches`` inside MIDI [low, high], low to high."""
    out = []
    for p in pitches:
        for n in range(9):
            q = music21.pitch.Pitch(p.name)
            q.octave = n
            if low <= q.midi <= high:
                out.append(q)
    return sorted(out, key=lambda q: q.midi)


def _near(
    candidates: list[music21.pitch.Pitch], target: int, rng: random.Random
) -> music21.pitch.Pitch:
    ranked = sorted(candidates, key=lambda q: abs(q.midi - target))
    return rng.choice(ranked[:2]) if len(ranked) > 1 else ranked[0]


class _Melody:
    """Chord tones on the beats, scale steps between them."""

    def __init__(self, key: music21.key.Key, rng: random.Random) -> None:
        self.scale = key.getScale().getPitches(
            music21.pitch.Pitch(midi=RH_RANGE[0]), music21.pitch.Pitch(midi=RH_RANGE[1])
        )
        self.rng = rng
        self.current = rng.choice(self.scale[len(self.scale) // 4 : 3 * len(self.scale) // 4])

    def chord_tone(self, harmony: Harmony) -> music21.pitch.Pitch:
        tones = _in_range(chord_pitches(harmony.root, harmony.quality), *RH_RANGE)
        self.current = _near(tones, self.current.midi + self.rng.choice((-3, 0, 3)), self.rng)
        return self.current

    def step(self) -> music21.pitch.Pitch:
        i = min(range(len(self.scale)), key=lambda j: abs(self.scale[j].midi - self.current.midi))
        i = max(0, min(len(self.scale) - 1, i + self.rng.choice((-2, -1, -1, 1, 1, 2))))
        self.current = self.scale[i]
        return self.current

    def third_below(self, p: music21.pitch.Pitch) -> music21.pitch.Pitch:
        names = [q.name for q in self.scale[:7]]
        i = names.index(p.name) if p.name in names else 0
        q = music21.pitch.Pitch(names[(i - 2) % 7])
        q.octave = p.octave
        while q.midi >= p.midi:
            q.octave = (q.octave or 4) - 1
        return q


def _slot_harmonies(chords: list[Harmony]) -> Iterator[tuple[float, Harmony]]:
    offset = 0.0
    for h in chords:
        yield offset, h
        offset += h.beats


def _harmony_at(chords: list[Harmony], offset: float) -> Harmony:
    found = chords[0]
    for start, h in _slot_harmonies(chords):
        if start <= offset + 1e-6:
            found = h
    return found


def _right(
    texture: str,
    chords: list[Harmony],
    bar: float,
    beat: float,
    melody: _Melody,
    final: bool,
    rng: random.Random,
) -> list[tuple[float, float, list[music21.pitch.Pitch]]]:
    events = []
    offset = 0.0
    previous = None
    for value in _rhythm(bar, beat, rng, final):
        h = _harmony_at(chords, offset)
        on_beat = abs(offset / beat - round(offset / beat)) < 1e-6
        if texture == "chords":
            previous = voice_chord(h, previous, 60, 81, tones=3)
            events.append((offset, value, previous))
        else:
            p = melody.chord_tone(h) if on_beat or final else melody.step()
            pitches = [melody.third_below(p), p] if texture == "thirds" else [p]
            if not final and rng.random() < 0.06:
                pitches = []  # a rest
            events.append((offset, value, pitches))
        offset += value
    return events


def _left(
    texture: str, chords: list[Harmony], beat: float, previous: list | None, final: bool
) -> tuple[list[tuple[float, float, list[music21.pitch.Pitch]]], list | None]:
    events = []
    for start, h in _slot_harmonies(chords):
        voiced = voice_chord(h, previous, LH_RANGE[0], LH_RANGE[1] - 12, tones=3)
        previous = voiced
        root = (
            voiced[0]
            if h.quality == "sus4"
            else min(
                _in_range([music21.pitch.Pitch(h.root)], LH_RANGE[0], LH_RANGE[0] + 12),
                key=lambda q: q.midi,
            )
        )
        style = "block" if final else texture
        if style == "block":
            events.append((start, h.beats, voiced))
        elif style == "octaves":
            events.append((start, h.beats, [root, octave(root, 1)]))
        elif style == "oompah":
            upper = [octave(p, 1) if p.midi < root.midi + 3 else p for p in voiced[1:]] or voiced
            steps = _steps(h.beats, beat if beat == 1.0 else 0.5)
            group = 3 if len(steps) % 3 == 0 else 2  # oom-pah-pah in threes
            for i, (offset, value) in enumerate(steps):
                events.append((start + offset, value, [root] if i % group == 0 else upper))
        else:
            order = {"alberti": (0, 2, 1, 2), "broken": (0, 1, 2, 1), "arpeggio": (0, 1, 2, 3)}[
                style
            ]
            tones = [*voiced, octave(voiced[0], 1)]
            step = 0.25 if style == "arpeggio" and beat == 1.0 else 0.5
            for i, (offset, value) in enumerate(_steps(h.beats, step)):
                events.append((start + offset, value, [tones[order[i % 4]]]))
    return events, previous


def _steps(total: float, step: float) -> list[tuple[float, float]]:
    """(offset, duration) pulses filling ``total``; the last one may be shorter."""
    out = []
    offset = 0.0
    while total - offset > 1e-6:
        out.append((offset, min(step, total - offset)))
        offset += step
    return out


def _fill(
    m: music21.stream.Measure, events: list[tuple[float, float, list[music21.pitch.Pitch]]]
) -> None:
    for offset, value, pitches in events:
        if not pitches:
            e: music21.note.GeneralNote = music21.note.Rest(quarterLength=value)
        elif len(pitches) == 1:
            e = music21.note.Note(copy.deepcopy(pitches[0]), quarterLength=value)
        else:
            e = music21.chord.Chord(copy.deepcopy(pitches), quarterLength=value)
        m.insert(offset, e)


def generate(
    rng: random.Random, p: Profile | str = "default", cfg: PianoConfig | None = None
) -> music21.stream.Score:
    cfg = cfg or PianoConfig()
    p = profile(p)
    minor = rng.random() < cfg.minor
    key = music21.key.Key(
        rng.choice(ALL_MINOR_KEYS if minor else ALL_MAJOR_KEYS), "minor" if minor else "major"
    )
    meter = _pick(rng, METERS)
    bar, beat = bar_length(meter), _beat(meter)
    count = rng.randint(cfg.min_measures, max(cfg.min_measures, cfg.max_measures))
    per_measure = 2 if bar >= 3.0 and rng.random() < 0.4 else 1
    harmony = progression(key, count, bar, rng, per_measure=per_measure)
    right, left = _pick(rng, RIGHT), _pick(rng, LEFT)

    rh, lh = music21.stream.PartStaff(), music21.stream.PartStaff()
    rh.insert(0, music21.instrument.Piano())
    melody = _Melody(key, rng)
    previous = None
    for i, chords in enumerate(harmony):
        final = i == count - 1
        top, bottom = music21.stream.Measure(number=i + 1), music21.stream.Measure(number=i + 1)
        if i == 0:
            for m, clef in ((top, music21.clef.TrebleClef()), (bottom, music21.clef.BassClef())):
                m.insert(0.0, clef)
                m.insert(0.0, music21.key.Key(key.tonic.name, key.mode))
                m.insert(0.0, music21.meter.TimeSignature(meter))
            if rng.random() < cfg.tempo:
                text, bpm = rng.choice(TEMPI)
                mark = music21.tempo.MetronomeMark(
                    text=text, number=bpm if rng.random() < 0.5 else None
                )
                top.insert(0.0, mark)
        _fill(top, _right(right, chords, bar, beat, melody, final, rng))
        events, previous = _left(left, chords, beat, previous, final)
        _fill(bottom, events)
        if final:
            top.rightBarline = music21.bar.Barline("final")
            bottom.rightBarline = music21.bar.Barline("final")
        rh.append(top)
        lh.append(bottom)

    score = music21.stream.Score()
    score.insert(0, rh)
    score.insert(0, lh)
    score.insert(0, music21.layout.StaffGroup([rh, lh], symbol="brace", barTogether=True))
    for staff in (rh, lh):
        with contextlib.suppress(music21.exceptions21.Music21Exception):
            staff.makeBeams(inPlace=True)
    return decorate(score, p, rng)
