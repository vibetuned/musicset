"""Mutations of real scores: rare notation injected into real music.

* ``mutate`` (kakigori's synthetic generator): denser rhythms (16ths, dotted
  pairs, 32nds, triplets), ornaments, articulations, fingerings, dynamics,
  chords and arpeggios, chord-name text, rehearsal marks, forced stems, slurs,
  ottava brackets, double barlines, clef changes and octave clefs;
* ``arrange`` (kakigori's synthetic arranger): one mutation per measure: an
  articulation on every note, an ornament, a dynamic, an ottava with the notes
  moved to compensate, a new rhythm on the same pitches, or two triplets.

Notes under a tie or a spanner are left alone by rhythmic changes.
"""

from __future__ import annotations

import contextlib
import copy
import random
from collections.abc import Callable, Sequence
from fractions import Fraction

import music21

from musicset.synth.decorate import (
    ARTICULATIONS,
    ORNAMENTS,
    add_clef_changes,
    add_double_barlines,
    add_octave_clefs,
)

_KEEP = (music21.meter.TimeSignature, music21.key.KeySignature, music21.clef.Clef)


def _free(n: music21.note.GeneralNote) -> bool:
    """Not tied and not under a spanner."""
    return getattr(n, "tie", None) is None and not n.getSpannerSites()


def _replace(
    parent: music21.stream.Stream,
    old: music21.base.Music21Object,
    new: Sequence[tuple[float | Fraction, music21.base.Music21Object]],
) -> None:
    try:
        parent.remove(old)
        for offset, e in new:
            parent.insert(offset, e)
    except music21.exceptions21.StreamException:
        pass


def _rhythms(score: music21.stream.Score, rng: random.Random) -> None:
    for measure in score.recurse().getElementsByClass(music21.stream.Measure):
        for n in list(measure.notes):
            if not isinstance(n, music21.note.Note) or not _free(n) or n.activeSite is None:
                continue
            parent, start, ql = n.activeSite, n.offset, n.duration.quarterLength
            if ql >= 1.0 and rng.random() < 0.10:  # run of 16ths
                _replace(
                    parent,
                    n,
                    [
                        (
                            start + i * 0.25,
                            music21.note.Rest(quarterLength=0.25)
                            if rng.random() < 0.2
                            else music21.note.Note(copy.deepcopy(n.pitch), quarterLength=0.25),
                        )
                        for i in range(int(ql / 0.25))
                    ],
                )
            elif ql == 1.0 and rng.random() < 0.15:  # dotted 8th + 16th
                _replace(
                    parent,
                    n,
                    [
                        (start, music21.note.Note(copy.deepcopy(n.pitch), quarterLength=0.75)),
                        (
                            start + 0.75,
                            music21.note.Note(copy.deepcopy(n.pitch), quarterLength=0.25),
                        ),
                    ],
                )
            elif ql == 0.25 and rng.random() < 0.20:  # two 32nds
                _replace(
                    parent,
                    n,
                    [
                        (
                            start + i * 0.125,
                            music21.note.Rest(quarterLength=0.125)
                            if rng.random() < 0.3
                            else music21.note.Note(copy.deepcopy(n.pitch), quarterLength=0.125),
                        )
                        for i in range(2)
                    ],
                )
            elif ql >= 0.5 and rng.random() < 0.15:  # triplet
                written = Fraction(ql) / 2
                notes = [
                    music21.note.Note(copy.deepcopy(n.pitch), quarterLength=written)
                    for _ in range(3)
                ]
                for note, kind in zip(notes, ("start", None, "stop"), strict=True):
                    t = music21.duration.Tuplet(3, 2)
                    if kind:
                        t.type = kind
                    note.duration.appendTuplet(t)
                step = Fraction(ql) / 3
                _replace(parent, n, [(start + i * step, note) for i, note in enumerate(notes)])


def _marks(score: music21.stream.Score, rng: random.Random) -> None:
    for note in list(score.recurse().notes):
        if not isinstance(note, music21.note.Note):
            continue
        if rng.random() < 0.05:
            note.expressions.append(ORNAMENTS["trill"]())
        elif rng.random() < 0.05:
            note.expressions.append(ORNAMENTS["mordent"]())
        elif rng.random() < 0.2:
            note.articulations.append(music21.articulations.Fingering(rng.randint(1, 5)))
        elif rng.random() < 0.05:
            note.expressions.append(ORNAMENTS["turn"]())
        if rng.random() < 0.15:
            note.articulations.append(rng.choice(ARTICULATIONS)())
        if rng.random() < 0.05 and note.activeSite is not None:
            dynamic = rng.choice(("pp", "pp", "mp", "mp", "ff", "ff", "p", "mf", "f", "sfz", "fz"))
            with contextlib.suppress(music21.exceptions21.StreamException):
                note.activeSite.insert(note.offset, music21.dynamics.Dynamic(dynamic))

        element: music21.note.GeneralNote = note
        if rng.random() < 0.10 and _free(note) and note.quarterLength >= 0.5:
            pitches = [note.pitch, note.pitch.transpose("P4"), note.pitch.transpose("M6")]
            if rng.random() < 0.5:
                pitches.append(note.pitch.transpose("P8"))
            chord = music21.chord.Chord(pitches)
            chord.duration = copy.deepcopy(note.duration)
            chord.expressions = copy.deepcopy(note.expressions)
            chord.articulations = copy.deepcopy(note.articulations)
            if not any(p.accidental is not None for p in pitches) and rng.random() < 0.5:
                chord.expressions.append(music21.expressions.ArpeggioMark())
            if note.activeSite is not None:
                try:
                    note.activeSite.replace(note, chord)
                    element = chord
                except music21.exceptions21.StreamException:
                    pass

        if rng.random() < 0.1:
            element.lyric = rng.choice(("C", "G7", "Amin", "F#dim"))
        if rng.random() < 0.02:
            element.expressions.append(
                music21.expressions.RehearsalMark(rng.choice(("A", "B", "C", "D")))
            )
        if rng.random() < 0.05 and element.quarterLength >= 1.0:
            element.stemDirection = "up" if rng.random() < 0.5 else "down"


def _spanners(score: music21.stream.Score, rng: random.Random) -> None:
    """A slur, or else an ottava bracket, per part."""
    for part in score.parts:
        notes = [n for n in part.recurse().notesAndRests if not n.isRest]
        if len(notes) >= 3 and rng.random() < 0.3:
            i = rng.randint(0, len(notes) - 3)
            part.insert(
                0, music21.spanner.Slur(notes[i], notes[rng.randint(i + 2, len(notes) - 1)])
            )
        elif len(notes) >= 4 and rng.random() < 0.30:
            i = rng.randint(0, len(notes) - 4)
            ottava = music21.spanner.Ottava(type=rng.choice(("8va", "8vb")))
            ottava.addSpannedElements(notes[i], notes[rng.randint(i + 3, len(notes) - 1)])
            part.insert(0, ottava)


def _beam(score: music21.stream.Score) -> None:
    for part in score.parts:
        try:
            part.makeBeams(inPlace=True)
        except music21.exceptions21.Music21Exception:
            for n in part.recurse().notes:
                n.beams.beamsList = []


def mutate(
    score: music21.stream.Score,
    rng: random.Random,
    *,
    octave_clef: float = 0.40,
    octave_8va: float = 0.65,
) -> music21.stream.Score:
    _rhythms(score, rng)
    _marks(score, rng)
    _spanners(score, rng)
    add_double_barlines(score, 0.06, rng, first=1)
    add_clef_changes(score, 0.25, rng)
    add_octave_clefs(score, octave_clef, octave_8va, rng)
    _beam(score)
    return score


# =============================================================================
# Arranger: one mutation per measure
# =============================================================================


def _pitches(measure: music21.stream.Measure) -> list[music21.pitch.Pitch]:
    out = []
    for el in measure.recurse().notes:
        if isinstance(el, music21.chord.Chord):
            if el.pitches:
                out.append(el.pitches[0])
        elif hasattr(el, "pitch"):
            out.append(el.pitch)
    return out


def _clear(measure: music21.stream.Measure) -> None:
    """Empty a measure, keeping its clef, key and meter."""
    keep = [(el.offset, el) for el in measure if isinstance(el, _KEEP)]
    measure.clear()
    for offset, el in keep:
        measure.insert(offset, el)


def _bar(measure: music21.stream.Measure) -> float:
    return measure.barDuration.quarterLength if measure.barDuration else 4.0


def _articulate(measure: music21.stream.Measure, rng: random.Random) -> None:
    kind = rng.choice(ARTICULATIONS)
    for el in measure.recurse().notes:
        el.articulations.append(kind())


def _ornament(measure: music21.stream.Measure, rng: random.Random) -> None:
    notes = list(measure.recurse().notes)
    if notes:
        target = rng.choice(notes)
        target.expressions.append(ORNAMENTS[rng.choice(tuple(ORNAMENTS))]())
        if isinstance(target, music21.chord.Chord) and rng.random() < 0.5:
            target.expressions.append(music21.expressions.ArpeggioMark())


def _dynamic(measure: music21.stream.Measure, rng: random.Random) -> None:
    notes = list(measure.recurse().notes)
    if notes:
        offset = rng.choice(notes).offset
        measure.insert(offset, music21.dynamics.Dynamic(rng.choice(("mf", "p", "sfz", "z"))))


def _octave(measure: music21.stream.Measure, rng: random.Random) -> None:
    """An ottava over the measure: the notes sound where they did and are
    drawn an octave toward the staff (8va over high passages, 8vb under low).

    music21's non-transposing Ottava holds sounding pitches and shifts only
    the display; kakigori also moved the pitches, which shifted the display
    twice: 8vb passages drawn two octaves up, seven ledger lines high.
    """
    notes = [
        el
        for el in measure.recurse().notes
        if isinstance(el, music21.note.Note | music21.chord.Chord)
    ]
    if len(notes) < 2:
        return
    midis = [p.midi for el in notes for p in el.pitches]
    mean = sum(midis) / len(midis)
    kind = "8va" if mean >= 72 else "8vb" if mean <= 48 else rng.choice(("8va", "8vb"))
    ottava = music21.spanner.Ottava(type=kind, transposing=False)
    ottava.addSpannedElements(notes[0], notes[-1])
    measure.insert(0, ottava)


# (quarter length, rest?); "pair" is a 16th then a 32nd rest, which never beams
_BLOCKS = (("pair", 0.375, False), ("half", 2.0, False), ("whole", 4.0, False),
           ("half rest", 2.0, True), ("whole rest", 4.0, True))  # fmt: skip


def _rhythm(measure: music21.stream.Measure, rng: random.Random) -> None:
    """The measure's pitches again, as 16ths, halves and wholes."""
    pitches = _pitches(measure)
    if not pitches:
        return
    total = _bar(measure)
    _clear(measure)
    offset, k = 0.0, 0
    while total - offset > 1e-3:
        fitting = [b for b in _BLOCKS if b[1] <= total - offset + 1e-3]
        if not fitting:
            measure.insert(offset, music21.note.Rest(quarterLength=total - offset))
            break
        name, ql, rest = rng.choice(fitting)
        if name == "pair":
            measure.insert(
                offset,
                music21.note.Note(copy.deepcopy(pitches[k % len(pitches)]), quarterLength=0.25),
            )
            measure.insert(offset + 0.25, music21.note.Rest(quarterLength=0.125))
            k += 1
        elif rest:
            measure.insert(offset, music21.note.Rest(quarterLength=ql))
        else:
            measure.insert(
                offset,
                music21.note.Note(copy.deepcopy(pitches[k % len(pitches)]), quarterLength=ql),
            )
            k += 1
        offset += ql


def _triplets(measure: music21.stream.Measure, rng: random.Random) -> None:
    """Two triplet groups filling a full measure of even length."""
    total = _bar(measure)
    half = total / 2
    if abs(measure.duration.quarterLength - total) > 1e-3 or abs(half - round(half)) > 1e-3:
        return
    pitches = _pitches(measure)
    if not pitches:
        return
    _clear(measure)
    offset = Fraction(0)
    step = Fraction(half) / 3
    for _ in range(2):
        # one Tuplet per note, typed start/stop: three notes sharing one
        # untyped Tuplet came out of MusicXML as three plain notes and three
        # one-note tuplets (a 5/4 measure in the MEI)
        for kind in ("start", None, "stop"):
            n = music21.note.Note(copy.deepcopy(rng.choice(pitches)), quarterLength=half / 2)
            tuplet = music21.duration.Tuplet(3, 2)
            if kind:
                tuplet.type = kind
            n.duration.appendTuplet(tuplet)
            measure.insert(offset, n)
            offset += step
    measure.makeBeams(inPlace=True)


MEASURE_MUTATIONS: tuple[Callable[[music21.stream.Measure, random.Random], None], ...] = (
    _articulate,
    _ornament,
    _dynamic,
    _octave,
    _rhythm,
    _triplets,
)


def arrange(score: music21.stream.Score, rng: random.Random) -> music21.stream.Score:
    for part in score.parts:
        for measure in part.getElementsByClass(music21.stream.Measure):
            rng.choice(MEASURE_MUTATIONS)(measure, rng)
    return score
