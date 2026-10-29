"""Notation layered onto scores: ornaments, articulations, dynamics,
fingerings, slurs, ties, double barlines, clef changes and octave signs.

How much of each appears is a ``Profile``; the generators take one, and
``decorate`` applies one to any score (generated or real):

* ``plain``: notes only, for the first curriculum stages;
* ``default``: gelato's synthetic writer;
* ``detector``: kakigori's, weighted toward the classes its detector reads
  worst (32nd notes, trills, pp/mp/ff, double barlines, common and cut time,
  clef changes, octave clefs and ottava brackets).
"""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass, replace

import music21

from musicset.synth.theory import ALL_METERS, DETECTOR_METERS, octave

ORNAMENTS = {
    "trill": music21.expressions.Trill,
    "mordent": music21.expressions.Mordent,
    "turn": music21.expressions.Turn,
    "fermata": music21.expressions.Fermata,
}
ARTICULATIONS = (
    music21.articulations.Staccato,
    music21.articulations.Accent,
    music21.articulations.Staccatissimo,
    music21.articulations.Tenuto,
    music21.articulations.StrongAccent,  # marcato
)


@dataclass(frozen=True, slots=True)
class Profile:
    name: str = "default"
    meters: tuple[str, ...] = ALL_METERS
    """Meters to draw from when a generator is not given one; duplicates weight."""
    durations: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5, 0.25)
    duration_weights: tuple[float, ...] = (0.10, 0.15, 0.35, 0.25, 0.15)
    ornament: float = 0.06
    """Per note: one of ``ornaments`` (duplicates weight)."""
    ornaments: tuple[str, ...] = ("trill", "mordent", "turn")
    articulation: float = 0.12
    """Per note."""
    chord_articulation: float = 0.08
    arpeggio: float = 0.25
    """Per chord."""
    dynamic: float = 0.5
    """Per measure, at one of its notes."""
    dynamics: tuple[str, ...] = ("p", "pp", "z", "sfz", "fz", "mf", "f")
    slur: float = 0.15
    """Per note: a slur over the next 2-6 notes starts here."""
    tie: float = 0.15
    """Per pair of adjacent equal notes."""
    chord_tie: float = 0.30
    fingered: float = 0.2
    """Share of scores that carry fingerings at all."""
    fingering: float = 0.5
    """Per note of a fingered score."""
    double_barline: float = 0.0
    """Per measure (same positions in every part)."""
    dashed_barline: float = 0.0
    """Per measure without a double barline."""
    clef_change: float = 0.0
    """Per part: a treble/bass switch for a measure or two."""
    c_clef_change: float = 0.0
    """Share of those switches to (or, on a C-clef part, from) an alto or
    tenor clef: the cClefChange glyph, rare in real lite scores."""
    octave_clef: float = 0.0
    """Per treble part: switch to an 8va/8vb treble clef partway."""
    octave_8va: float = 0.65
    """Share of octave clefs that are 8va."""
    c_clef: float = 0.0
    """Per single-staff score: an alto or tenor clef, the music moved by
    octaves to sit on it (real lite scores have a C clef on ~1% of pages)."""
    ottava: float = 0.0
    """Per part: an 8va/8vb bracket over a few notes."""
    key_change: float = 0.0
    """Per score: a key change partway (cancellation naturals), half the time
    back to the opening key later."""
    measure_rest: float = 0.0
    """Per measure of each part (not the first or last): a whole-measure rest."""

    def with_(self, **changes) -> Profile:
        return replace(self, **changes)


PROFILES: dict[str, Profile] = {
    "plain": Profile(
        name="plain",
        ornament=0.0,
        articulation=0.0,
        chord_articulation=0.0,
        arpeggio=0.0,
        dynamic=0.0,
        slur=0.0,
        tie=0.0,
        chord_tie=0.0,
        fingered=0.0,
    ),
    "default": Profile(),
    "detector": Profile(
        name="detector",
        meters=DETECTOR_METERS,
        durations=(4.0, 2.0, 1.0, 0.5, 0.25, 0.125),
        duration_weights=(0.08, 0.14, 0.33, 0.24, 0.14, 0.07),
        ornaments=("trill", "trill", "mordent", "mordent", "turn", "fermata"),
        # mfp and sfffz have no combined glyph: Verovio sets their letters,
        # the only source of the lone m and s glyphs
        dynamics=(
            "pp",
            "pp",
            "mp",
            "mp",
            "ff",
            "ff",
            "fff",
            "p",
            "z",
            "sfz",
            "fz",
            "mf",
            "f",
            "mfp",
            "sfffz",
        ),
        fingered=1.0,
        fingering=1.0,
        double_barline=0.10,
        dashed_barline=0.03,
        clef_change=0.25,
        c_clef_change=0.3,
        octave_clef=0.40,
        c_clef=0.15,
        ottava=0.30,
        key_change=0.35,
        measure_rest=0.04,
    ),
}
# The detector profile's rare-class weighting without its fingering flood
# (a digit on every note: 41k boxes in 120 scores, a class nothing reads) and
# with a third of its ornaments: one note in 17 ornamented taught the detector
# to read real fermatas as mordents, turns and trills
PROFILES["rare"] = PROFILES["detector"].with_(
    name="rare", fingered=0.2, fingering=0.5, ornament=0.02
)


def profile(name: str | Profile) -> Profile:
    if isinstance(name, Profile):
        return name
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown profile {name!r}: {list(PROFILES)}") from None


def pick_duration(p: Profile, bar: float, rng: random.Random) -> float:
    """A note value from the profile that fits in one bar."""
    valid = [(d, w) for d, w in zip(p.durations, p.duration_weights, strict=True) if d <= bar]
    if not valid:
        return bar
    values, weights = zip(*valid, strict=True)
    return rng.choices(values, weights=weights)[0]


# =============================================================================
# Per note
# =============================================================================


def ornament_note(n: music21.note.GeneralNote, p: Profile, rng: random.Random) -> None:
    """An ornament and/or an articulation on a note."""
    if p.ornament and rng.random() < p.ornament:
        n.expressions.append(ORNAMENTS[rng.choice(p.ornaments)]())
    if p.articulation and rng.random() < p.articulation:
        n.articulations.append(rng.choice(ARTICULATIONS)())


def decorate_chord(c: music21.chord.Chord, p: Profile, rng: random.Random) -> None:
    """An arpeggio sign and/or an articulation on a chord."""
    if p.arpeggio and rng.random() < p.arpeggio:
        c.expressions.append(music21.expressions.ArpeggioMark())
    if p.chord_articulation and rng.random() < p.chord_articulation:
        c.articulations.append(rng.choice(ARTICULATIONS)())


def chord_fingers(c: music21.chord.Chord, rng: random.Random) -> list[int]:
    """Black-key roots take 2-3-5, white ones 1-3-5 or 1-2-4."""
    base = (
        [2, 3, 5]
        if c.pitches[0].pitchClass in (1, 3, 6, 8, 10)
        else rng.choice(([1, 3, 5], [1, 2, 4]))
    )
    while len(base) < len(c.pitches):
        base.append(min(base[-1] + 1, 5))
    return base[: len(c.pitches)]


def finger(
    n: music21.note.GeneralNote, p: Profile, rng: random.Random, fingers: list[int] | None = None
) -> None:
    """Fingering on a note (``fingers[0]`` or random) or on each chord tone."""
    if isinstance(n, music21.chord.Chord):
        for f in fingers or chord_fingers(n, rng):
            if rng.random() < p.fingering:
                n.articulations.append(music21.articulations.Fingering(f))
    elif isinstance(n, music21.note.Note) and rng.random() < p.fingering:
        n.articulations.append(
            music21.articulations.Fingering(fingers[0] if fingers else rng.randint(1, 5))
        )


def _has_fingering(n: music21.note.GeneralNote) -> bool:
    return any(isinstance(a, music21.articulations.Fingering) for a in n.articulations)


# =============================================================================
# Per part / score
# =============================================================================


def _parts(score: music21.stream.Score) -> list[music21.stream.Part]:
    return list(score.parts)


def _measures(part: music21.stream.Stream) -> list[music21.stream.Measure]:
    return list(part.getElementsByClass(music21.stream.Measure))


def add_dynamics(part: music21.stream.Part, p: Profile, rng: random.Random) -> None:
    for m in _measures(part):
        notes = list(m.recurse().notes)
        if notes and rng.random() < p.dynamic:
            n = rng.choice(notes)
            with contextlib.suppress(music21.exceptions21.Music21Exception):
                m.insert(
                    n.getOffsetInHierarchy(m), music21.dynamics.Dynamic(rng.choice(p.dynamics))
                )


def add_slurs(part: music21.stream.Part, p: Profile, rng: random.Random) -> None:
    notes = [n for n in part.recurse().notes if not n.isRest]
    if len(notes) < 4 or not p.slur:
        return
    i = 0
    while i < len(notes) - 2:
        if rng.random() < p.slur:
            span = rng.randint(2, min(6, len(notes) - i))
            part.insert(0, music21.spanner.Slur(notes[i], notes[i + span - 1]))
            i += span
        else:
            i += 1


def add_ties(part: music21.stream.Part, p: Profile, rng: random.Random) -> None:
    """Tie adjacent notes (or chords) of equal pitch, barlines included."""
    previous = None
    for n in part.recurse().notesAndRests:
        if (
            previous is not None
            and not n.isRest
            and previous.tie is None
            and n.tie is None
            and _same_pitches(previous, n)
            and rng.random() < (p.chord_tie if n.isChord else p.tie)
        ):
            previous.tie = music21.tie.Tie("start")
            n.tie = music21.tie.Tie("stop")
        previous = None if n.isRest else n


def _same_pitches(a: music21.note.GeneralNote, b: music21.note.GeneralNote) -> bool:
    return a.isChord == b.isChord and [x.midi for x in a.pitches] == [x.midi for x in b.pitches]


def add_double_barlines(
    score: music21.stream.Score,
    rate: float,
    rng: random.Random,
    first: int = 3,
    dashed: float = 0.0,
) -> None:
    """Double barlines at phrase boundaries, and (``dashed`` per measure)
    dashed ones elsewhere, same positions in every part."""
    parts = _parts(score)
    if not parts or not (rate or dashed):
        return
    n = len(_measures(parts[0]))
    styles = {}
    for i in range(first, n - 1):
        r = rng.random()
        if r < rate:
            styles[i] = "double"
        elif r < rate + dashed:
            styles[i] = "dashed"
    for part in parts:
        for i, m in enumerate(_measures(part)):
            if i in styles:
                m.rightBarline = music21.bar.Barline(styles[i])


def _first_clef(part: music21.stream.Part) -> music21.clef.Clef | None:
    return part.recurse().getElementsByClass(music21.clef.Clef).first()


def add_clef_changes(
    score: music21.stream.Score, rate: float, rng: random.Random, c_share: float = 0.0
) -> None:
    """Switch clef for a measure or two, then switch back: treble/bass, or
    (``c_share`` of the switches) to an alto/tenor clef; a part on a C clef
    switches to treble or bass."""
    for part in _parts(score):
        measures = _measures(part)
        if len(measures) < 6 or rng.random() >= rate:
            continue
        current = _first_clef(part)
        to_c = c_share > 0 and rng.random() < c_share
        if isinstance(current, music21.clef.TrebleClef) and not current.octaveChange:
            change = music21.clef.AltoClef() if to_c else music21.clef.BassClef()
            restore = music21.clef.TrebleClef()
        elif isinstance(current, music21.clef.BassClef):
            change = music21.clef.TenorClef() if to_c else music21.clef.TrebleClef()
            restore = music21.clef.BassClef()
        elif isinstance(current, music21.clef.AltoClef):
            change, restore = music21.clef.TrebleClef(), music21.clef.AltoClef()
        elif isinstance(current, music21.clef.TenorClef):
            change, restore = music21.clef.BassClef(), music21.clef.TenorClef()
        else:
            continue
        i = rng.randint(2, len(measures) - 3)
        measures[i].insert(0.0, change)
        measures[min(i + rng.randint(1, 2), len(measures) - 1)].insert(0.0, restore)


def add_c_clefs(score: music21.stream.Score, rate: float, rng: random.Random) -> None:
    """Set a single-staff score on an alto or tenor clef, every note moved by
    whole octaves (spelling kept) so the music sits on that staff."""
    parts = _parts(score)
    if len(parts) != 1 or rng.random() >= rate:
        return
    part = parts[0]
    current = _first_clef(part)
    if current is None or current.activeSite is None or getattr(current, "octaveChange", 0):
        return
    if not isinstance(current, (music21.clef.TrebleClef, music21.clef.BassClef)):
        return
    pitches = sorted(p.midi for n in part.recurse().notes for p in n.pitches)
    if not pitches:
        return
    # middle line: C4 on the alto staff, A3 on the tenor staff
    clef, middle = rng.choice(((music21.clef.AltoClef(), 60), (music21.clef.TenorClef(), 57)))
    shift = round((middle - pitches[len(pitches) // 2]) / 12)
    if shift:
        for n in part.recurse().notes:
            if n.isChord:
                n.pitches = tuple(octave(q, shift) for q in n.pitches)
            else:
                n.pitch = octave(n.pitch, shift)
    current.activeSite.replace(current, clef)


def add_octave_clefs(
    score: music21.stream.Score, rate: float, weight_8va: float, rng: random.Random
) -> None:
    """Put a treble part on an 8va/8vb treble clef, which then repeats at
    every system start; sometimes back to a plain treble clef later on.

    The octave clef opens the part rather than arriving mid-system: a
    mid-system change glyph is labelled gClefChange, which can only mean a
    plain treble clef, so the measures up to the next system would read an
    octave off in the groundtruth.
    """
    for part in _parts(score):
        measures = _measures(part)
        if len(measures) < 4 or rng.random() >= rate:
            continue
        current = _first_clef(part)
        if not isinstance(current, music21.clef.TrebleClef) or current.octaveChange:
            continue
        clef = (
            music21.clef.Treble8vaClef()
            if rng.random() < weight_8va
            else music21.clef.Treble8vbClef()
        )
        site = current.activeSite
        if site is None:
            continue
        site.replace(current, clef)
        if rng.random() < 0.4:
            measures[rng.randint(len(measures) // 2, len(measures) - 1)].insert(
                0.0, music21.clef.TrebleClef()
            )


def add_ottavas(score: music21.stream.Score, rate: float, rng: random.Random) -> None:
    """An 8va or 8vb bracket over four or more consecutive notes."""
    for part in _parts(score):
        notes = [n for n in part.recurse().notes if not n.isRest]
        if len(notes) < 4 or rng.random() >= rate:
            continue
        start = rng.randint(0, len(notes) - 4)
        end = rng.randint(start + 3, min(len(notes) - 1, start + 12))
        ottava = music21.spanner.Ottava(type=rng.choice(("8va", "8vb")))
        ottava.addSpannedElements(notes[start], notes[end])
        part.insert(0, ottava)


def _key_of(score: music21.stream.Score) -> music21.key.KeySignature | None:
    return score.recurse().getElementsByClass(music21.key.KeySignature).first()


def add_key_changes(score: music21.stream.Score, rate: float, rng: random.Random) -> None:
    """A key change at a measure in every part, drawn as the new signature
    after naturals cancelling the old one; half the time back home later."""
    parts = _parts(score)
    if not parts or rng.random() >= rate:
        return
    measures = [_measures(part) for part in parts]
    count = min(len(ms) for ms in measures)
    current = _key_of(score)
    if count < 6 or current is None:
        return
    # a different key with accidentals the old one had to cancel: two or
    # more fewer sharps or flats, or the other side of the circle
    sharps = current.sharps or 0
    choices = [k for k in range(-6, 7) if k != sharps and (abs(k) < abs(sharps) or k * sharps < 0)]
    if not choices:
        choices = [k for k in range(-6, 7) if k != sharps]
    new = rng.choice(choices)
    at = rng.randint(2, count - 3)
    back = rng.randint(at + 1, count - 1) if rng.random() < 0.5 and at + 1 < count - 1 else None
    for ms in measures:
        ms[at].insert(0.0, music21.key.KeySignature(new))
        if back is not None:
            ms[back].insert(0.0, music21.key.KeySignature(sharps))


def add_measure_rests(score: music21.stream.Score, rate: float, rng: random.Random) -> None:
    """Empty some measures (not the first or last) into whole-measure rests;
    measures holding a tied or spanned note keep their notes."""
    if not rate:
        return
    for part in _parts(score):
        measures = _measures(part)
        for m in measures[1:-1]:
            if rng.random() >= rate:
                continue
            events = list(m.recurse().notesAndRests)
            if not events or any(
                getattr(e, "tie", None) is not None or e.getSpannerSites() for e in events
            ):
                continue
            bar = m.barDuration.quarterLength
            for e in events:
                e.activeSite.remove(e)
            for v in list(m.voices):
                m.remove(v)
            rest = music21.note.Rest(quarterLength=bar)
            rest.fullMeasure = True
            m.insert(0.0, rest)


def decorate(
    score: music21.stream.Score,
    p: Profile,
    rng: random.Random,
    *,
    notes: bool = True,
    fingerings: bool | None = None,
    ties: bool = True,
) -> music21.stream.Score:
    """Apply a whole profile to a score in place.

    ``notes`` turns the per-note marks on; ``fingerings`` forces fingering on
    or off (default: the profile's share of fingered scores).
    """
    fingered = rng.random() < p.fingered if fingerings is None else fingerings
    add_measure_rests(score, p.measure_rest, rng)  # before ties and spanners
    for part in _parts(score):
        if notes:
            for n in part.recurse().notes:
                if n.isChord:
                    decorate_chord(n, p, rng)
                else:
                    ornament_note(n, p, rng)
                if fingered and not _has_fingering(n):
                    finger(n, p, rng)
        if ties:
            add_ties(part, p, rng)
        add_dynamics(part, p, rng)
        add_slurs(part, p, rng)
    add_double_barlines(score, p.double_barline, rng, dashed=p.dashed_barline)
    add_c_clefs(score, p.c_clef, rng)
    add_clef_changes(score, p.clef_change, rng, c_share=p.c_clef_change)
    add_octave_clefs(score, p.octave_clef, p.octave_8va, rng)
    add_ottavas(score, p.ottava, rng)
    add_key_changes(score, p.key_change, rng)
    return score
