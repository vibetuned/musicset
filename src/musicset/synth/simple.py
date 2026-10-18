"""Small scores for the first curriculum stages (gelato's easy and
intermediate writers): one staff, common keys and meters, no decoration."""

from __future__ import annotations

import random

import music21

from musicset.synth.theory import bar_length, octave

EASY_KEYS = ("C", "G", "F", "D", "Bb")
EASY_METERS = ("4/4", "3/4", "2/4")
EASY_DURATIONS = (4.0, 2.0, 1.0, 0.5)

MELODY_KEYS = ("C", "G", "F", "D", "Bb", "A", "Eb")
MELODY_METERS = ("4/4", "3/4")
# Rhythms that fill a measure exactly
MELODY_RHYTHMS = {
    "4/4": (
        (1.0, 1.0, 1.0, 1.0),
        (2.0, 2.0),
        (0.5, 0.5, 1.0, 2.0),
        (1.0, 0.5, 0.5, 2.0),
        (0.5, 0.5, 0.5, 0.5, 2.0),
        (1.0, 2.0, 1.0),
        (1.5, 0.5, 2.0),
        (3.0, 1.0),
    ),
    "3/4": ((1.0, 1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (0.5, 0.5, 1.0, 1.0)),
}


def _staff(rng: random.Random) -> music21.clef.Clef:
    return music21.clef.TrebleClef() if rng.random() < 0.5 else music21.clef.BassClef()


def _degree(key: music21.key.Key, degree: int, start: int) -> music21.pitch.Pitch:
    """Scale degree 1-14 from the tonic of octave ``start`` upward."""
    p = key.pitchFromDegree((degree - 1) % 7 + 1)
    p.octave = start + (degree - 1) // 7
    if p.pitchClass < key.tonic.pitchClass and (degree - 1) % 7:
        p.octave += 1  # degrees past B wrap into the next octave
    return p


def easy(rng: random.Random) -> music21.stream.Score:
    """One measure: a note, a diatonic chord, or a short scale walk."""
    meter = rng.choice(EASY_METERS)
    key = music21.key.Key(rng.choice(EASY_KEYS))
    clef = _staff(rng)
    treble = isinstance(clef, music21.clef.TrebleClef)
    bar = bar_length(meter)

    m = music21.stream.Measure(number=1)
    m.timeSignature = music21.meter.TimeSignature(meter)
    m.keySignature = music21.key.KeySignature(key.sharps)
    m.clef = clef
    mode = rng.choice(("note", "chord", "melody"))
    if mode == "melody":
        count = rng.randint(2, 4)
        start, step = rng.randint(1, 5), rng.choice((1, -1))
        if step < 0:
            start += 3  # stay at or above the tonic
        for i in range(count):
            n = music21.note.Note(_degree(key, start + i * step, 4 if treble else 3))
            n.quarterLength = bar / count
            m.insert(i * bar / count, n)
    else:
        value = rng.choice([d for d in EASY_DURATIONS if d <= bar])
        if mode == "chord":
            rn = music21.roman.RomanNumeral(rng.choice(("I", "ii", "IV", "V", "vi")), key)
            shift = 0 if treble else -1
            e = music21.chord.Chord([octave(p, shift) for p in rn.pitches], quarterLength=value)
        elif rng.random() < 0.2:
            e = music21.note.Rest(quarterLength=value)
        else:
            register = rng.choice((4, 5) if treble else (2, 3))
            e = music21.note.Note(_degree(key, rng.randint(1, 7), register), quarterLength=value)
        m.insert(0.0, e)
        if bar > value:
            m.insert(value, music21.note.Rest(quarterLength=bar - value))

    part = music21.stream.Part()
    part.append(m)
    score = music21.stream.Score()
    score.insert(0, part)
    return score


def melody(rng: random.Random) -> music21.stream.Score:
    """Four to eight measures of diatonic walk ending on a long tonic."""
    meter = rng.choice(MELODY_METERS)
    key = music21.key.Key(rng.choice(MELODY_KEYS))
    clef = _staff(rng)
    register = 4 if isinstance(clef, music21.clef.TrebleClef) else 3
    count = rng.randint(4, 8)
    degree = rng.choice((1, 3, 5))

    part = music21.stream.Part()
    for number in range(1, count + 1):
        m = music21.stream.Measure(number=number)
        if number == 1:
            m.timeSignature = music21.meter.TimeSignature(meter)
            m.keySignature = music21.key.KeySignature(key.sharps)
            m.clef = clef
        if number == count:
            m.insert(
                0.0, music21.note.Note(_degree(key, 1, register), quarterLength=bar_length(meter))
            )
        else:
            offset = 0.0
            for value in rng.choice(MELODY_RHYTHMS[meter]):
                if rng.random() < 0.10:
                    m.insert(offset, music21.note.Rest(quarterLength=value))
                else:
                    degree = max(1, min(10, degree + rng.choice((-2, -1, 1, 2))))
                    m.insert(
                        offset,
                        music21.note.Note(_degree(key, degree, register), quarterLength=value),
                    )
                offset += value
        part.append(m)
    score = music21.stream.Score()
    score.insert(0, part)
    return score
