"""Music theory shared by the generators: keys, meters, scales and their
fingerings, chords, and functional harmony with voice leading.

Pitches are music21 objects spelled from the key (a chord is built from its
root by interval names, so E-flat minor is E-flat, G-flat, B-flat, never
E-flat, F-sharp, B-flat). The harmony rules come from chirality's chord
generator: secondary dominants, dominant sevenths, sus4 resolutions, vii as a
dominant substitute, cadences at phrase ends and pivot modulations.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import music21

# =============================================================================
# Keys and meters
# =============================================================================

MAJOR_KEYS = ("C", "G", "D", "F", "Bb", "A", "E", "Eb", "Ab")
MINOR_KEYS = ("A", "E", "B", "D", "G", "C", "F", "F#", "C#")
# Up to six accidentals each way: for transposing finished pieces
ALL_MAJOR_KEYS = ("C", "G", "D", "A", "E", "B", "F#", "F", "Bb", "Eb", "Ab", "Db", "Gb")
ALL_MINOR_KEYS = ("A", "E", "B", "F#", "C#", "G#", "D#", "D", "G", "C", "F", "Bb", "Eb")

SIMPLE_METERS = ("4/4", "3/4", "2/4")
MEDIUM_METERS = ("4/4", "3/4", "2/4", "6/8", "C", "cut")
ALL_METERS = (
    "1/4", "2/4", "4/4", "6/8", "8/8", "C", "cut", "16/8", "16/16", "16/12", "9/8", "12/8",
)  # fmt: skip
# Duplicates weight the meters whose digits the detector reads worst
# (timeSig1/2/3/6/8, common and cut time)
DETECTOR_METERS = (
    "1/4", "1/4", "2/4", "2/2", "3/4", "3/8", "6/8", "6/4", "12/8", "12/8", "16/12",
    "8/8", "16/8", "C", "C", "cut", "cut", "4/4", "9/8", "16/16",
)  # fmt: skip
METER_SETS = {
    "simple": SIMPLE_METERS,
    "medium": MEDIUM_METERS,
    "full": ALL_METERS,
    "detector": DETECTOR_METERS,
}


def octave(p: music21.pitch.Pitch, n: int) -> music21.pitch.Pitch:
    """``p`` moved ``n`` octaves, spelling kept (``transpose(12)`` respells
    G-flat as F-sharp)."""
    q = copy.deepcopy(p)
    q.octave = (p.octave if p.octave is not None else p.implicitOctave) + n
    return q


def key_of(tonic: str, mode: str = "major") -> music21.key.Key:
    return music21.key.Key(tonic, mode)


def bar_length(meter: str) -> float:
    return float(music21.meter.TimeSignature(meter).barDuration.quarterLength)


def is_compound(meter: str) -> bool:
    """6/8, 9/8, 12/8, 6/4: beats divide in three."""
    num, _, den = meter.partition("/")
    return den in ("8", "4") and num in ("6", "9", "12")


# =============================================================================
# Scales and fingerings
# =============================================================================

SCALE_KINDS = ("major", "harmonic_minor", "melodic_minor", "chromatic")

# Two octaves ascending, 1 = thumb; descending reverses the ascent
_STANDARD: dict[str, tuple[int, ...]] = {
    "RH": (1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5),
    "LH": (5, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1),
}
_F_RH = (1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4)
_B_LH = (4, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1)
_FLAT_LH = (3, 2, 1, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3)
MAJOR_FINGERINGS: dict[str, dict[str, tuple[int, ...]]] = {
    **{k: _STANDARD for k in ("C", "G", "D", "A", "E")},
    "B": {"RH": _STANDARD["RH"], "LH": _B_LH},
    "G-": {
        "RH": (2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2),
        "LH": (4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1, 4),
    },
    "F": {"RH": _F_RH, "LH": _STANDARD["LH"]},
    "B-": {"RH": (2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4), "LH": _FLAT_LH},
    "E-": {"RH": (3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3), "LH": _FLAT_LH},
    "A-": {"RH": (3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3), "LH": _FLAT_LH},
    "D-": {"RH": (2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2), "LH": _FLAT_LH},
}
MINOR_FINGERINGS: dict[str, dict[str, tuple[int, ...]]] = {
    **{k: _STANDARD for k in ("A", "E", "D", "G", "C")},
    "B": {"RH": _STANDARD["RH"], "LH": _B_LH},
    "F": {"RH": _F_RH, "LH": _STANDARD["LH"]},
    "F#": {
        "RH": (2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2),
        "LH": (4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1, 4),
    },
    "C#": {"RH": (3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3), "LH": _FLAT_LH},
}


def _ascending(scale: music21.scale.ConcreteScale, octave: int) -> list[music21.pitch.Pitch]:
    """Fifteen degrees (two octaves) up from the tonic in ``octave``."""
    out: list[music21.pitch.Pitch] = []
    for i in range(1, 16):
        p = scale.pitchFromDegree(i)
        p.octave = octave + (i - 1) // 7
        if out and p.midi <= out[-1].midi:
            p.octave += 1
        out.append(p)
    return out


def scale_run(key: music21.key.Key, kind: str, hand: str) -> list[music21.pitch.Pitch]:
    """Two octaves up and back down: from octave 4 (right hand) or 2 (left).

    Melodic minor rises with raised sixth and seventh and falls natural.
    """
    octave = 4 if hand == "RH" else 2
    tonic = key.tonic
    if kind == "chromatic":
        start = music21.pitch.Pitch(tonic.name)
        start.octave = octave
        up = [start.transpose(i) for i in range(25)]
        return up + up[-2::-1]
    if kind == "major":
        up = _ascending(music21.scale.MajorScale(tonic), octave)
    elif kind == "harmonic_minor":
        up = _ascending(music21.scale.HarmonicMinorScale(tonic), octave)
    elif kind == "melodic_minor":
        up = _ascending(music21.scale.MelodicMinorScale(tonic), octave)
        down = _ascending(music21.scale.MinorScale(tonic), octave)
        return up + down[-2::-1]
    else:
        raise ValueError(f"unknown scale kind {kind!r}: {SCALE_KINDS}")
    return up + up[-2::-1]


def scale_fingering(
    key: music21.key.Key, kind: str, hand: str, pitches: list[music21.pitch.Pitch]
) -> list[int]:
    """Fingers for a ``scale_run``: the classical patterns, thumb on C and F
    in chromatic scales."""
    if kind == "chromatic":
        return [1 if p.pitchClass in (0, 5) else 3 for p in pitches]
    table = MAJOR_FINGERINGS if kind == "major" else MINOR_FINGERINGS
    name = key.tonic.name
    if name not in table:
        enharmonic = key.tonic.getEnharmonic()
        name = enharmonic.name if enharmonic is not None else name
    up = table.get(name, next(iter(table.values())))[hand]
    return [*up, *up[-2::-1]]


# =============================================================================
# Chords
# =============================================================================

CHORD_INTERVALS: dict[str, tuple[str, ...]] = {
    "major": ("P1", "M3", "P5"),
    "minor": ("P1", "m3", "P5"),
    "dim": ("P1", "m3", "d5"),
    "aug": ("P1", "M3", "A5"),
    "sus2": ("P1", "M2", "P5"),
    "sus4": ("P1", "P4", "P5"),
    "maj7": ("P1", "M3", "P5", "M7"),
    "dom7": ("P1", "M3", "P5", "m7"),
    "min7": ("P1", "m3", "P5", "m7"),
    "dim7": ("P1", "m3", "d5", "d7"),
    "half_dim7": ("P1", "m3", "d5", "m7"),
    "maj6": ("P1", "M3", "P5", "M6"),
    "min6": ("P1", "m3", "P5", "M6"),
    "add9": ("P1", "M3", "P5", "M9"),
}

# (scale degree, quality, numeral); minor uses the harmonic-minor dominant
MAJOR_DIATONIC = (
    (1, "major", "I"), (2, "minor", "ii"), (3, "minor", "iii"), (4, "major", "IV"),
    (5, "major", "V"), (6, "minor", "vi"), (7, "dim", "vii°"),
)  # fmt: skip
MINOR_DIATONIC = (
    (1, "minor", "i"), (2, "dim", "ii°"), (3, "major", "III"), (4, "minor", "iv"),
    (5, "major", "V"), (6, "major", "VI"), (7, "dim", "vii°"),
)  # fmt: skip
# How often each degree is chosen: I, IV and V carry most progressions
DEGREE_WEIGHTS = (0.25, 0.12, 0.10, 0.18, 0.20, 0.12, 0.03)


def chord_pitches(root: str | music21.pitch.Pitch, quality: str, octave: int = 4):
    """Root-position chord tones, spelled from the root."""
    r = music21.pitch.Pitch(root) if isinstance(root, str) else music21.pitch.Pitch(root.name)
    r.octave = octave
    return [r.transpose(iv) for iv in CHORD_INTERVALS.get(quality, CHORD_INTERVALS["major"])]


def chord_name(name: str) -> tuple[str, str]:
    """'C' -> ('C', 'major'), 'F#m' -> ('F#', 'minor'), 'B°' -> ('B', 'dim')."""
    if name.endswith("°"):
        return name[:-1], "dim"
    if name.endswith("m"):
        return name[:-1], "minor"
    return name, "major"


class CircleOfFifths:
    """The seven diatonic chords around a key: a chord's neighbours are the
    other chords of its parent major key."""

    MAJORS = ("C", "G", "D", "A", "E", "B", "Gb", "Db", "Ab", "Eb", "Bb", "F")
    MINORS = ("Am", "Em", "Bm", "F#m", "C#m", "G#m", "Ebm", "Bbm", "Fm", "Cm", "Gm", "Dm")
    # vii° of MAJORS[(i - 5) % 12]: leading tones, so F#° (of G), never Gb°
    DIMINISHED = ("C°", "G°", "D°", "A°", "E°", "B°", "F#°", "C#°", "G#°", "D#°", "A#°", "F°")

    def neighbors(self, chord: str) -> dict[str, str]:
        if chord in self.DIMINISHED:
            parent = (self.DIMINISHED.index(chord) - 5) % 12
        elif chord in self.MINORS:
            parent = self.MINORS.index(chord)
        elif chord in self.MAJORS:
            parent = self.MAJORS.index(chord)
        else:
            raise ValueError(f"chord {chord!r} not on the circle")
        left, right = (parent - 1) % 12, (parent + 1) % 12
        diatonic = {
            "I": self.MAJORS[parent],
            "IV": self.MAJORS[left],
            "V": self.MAJORS[right],
            "vi": self.MINORS[parent],
            "ii": self.MINORS[left],
            "iii": self.MINORS[right],
            "vii°": self.DIMINISHED[(parent + 5) % 12],
        }
        return {role: name for role, name in diatonic.items() if name != chord}

    def walk(self, start: str, length: int, rng: random.Random, repeat: float = 0.2) -> list[str]:
        """A progression wandering between neighbours, repeating a chord
        ``repeat`` of the time (repeated chords can be tied)."""
        out = [start]
        current = start
        for _ in range(length - 1):
            if rng.random() >= repeat:
                try:
                    current = rng.choice(list(self.neighbors(current).values()))
                except ValueError:
                    current = start
            out.append(current)
        return out


# =============================================================================
# Functional harmony (after chirality's ChordProgressionGenerator)
# =============================================================================


@dataclass(frozen=True, slots=True)
class Harmony:
    root: str
    """music21 pitch name, no octave."""
    quality: str
    beats: float
    """Quarter lengths."""
    numeral: str


def _diatonic(key: music21.key.Key) -> list[tuple[str, str, str]]:
    table = MINOR_DIATONIC if key.mode == "minor" else MAJOR_DIATONIC
    out = []
    for degree, quality, numeral in table:
        root = key.pitchFromDegree(degree)
        if key.mode == "minor" and degree == 7:
            root = root.transpose("A1")  # the raised leading tone
        out.append((root.name, quality, numeral))
    return out


def _secondary_dominant(root: str) -> str:
    """Root of V/x: a fifth above x."""
    return music21.pitch.Pitch(root).transpose("P5").name


def progression(
    key: music21.key.Key,
    measures: int,
    bar: float,
    rng: random.Random,
    *,
    per_measure: int = 1,
    phrase: int = 4,
) -> list[list[Harmony]]:
    """Chords per measure: starts and ends on the tonic, cadences every
    ``phrase`` measures, with secondary dominants, sevenths, suspensions and
    the odd pivot modulation to a neighbouring key."""
    home = key
    diatonic = _diatonic(key)
    slot = bar / per_measure
    out: list[list[Harmony]] = []
    for m in range(measures):
        chords: list[Harmony] = []
        last, phrase_end = m == measures - 1, (m + 1) % phrase == 0
        for s in range(per_measure):
            final_slot = s == per_measure - 1
            if m == 0 and s == 0:
                idx = 0
            elif last:
                idx = 4 if (per_measure > 1 and not final_slot) else 0
            elif phrase_end and final_slot:
                idx = rng.choice((0, 4))  # authentic or half cadence
            elif phrase_end or (m == measures - 2 and final_slot):
                idx = rng.choice((1, 3, 4, 4))  # pre-dominant or dominant
            else:
                idx = rng.choices(range(7), weights=DEGREE_WEIGHTS)[0]
            root, quality, numeral = diatonic[idx]
            if idx == 4 and rng.random() < 0.1 and not last:
                root, quality, numeral = diatonic[6]  # vii° for V
            elif idx == 4 and rng.random() < 0.5:
                quality, numeral = "dom7", numeral + "7"
            beats = slot
            if idx not in (0, 4) and rng.random() < 0.15 and slot >= 2:
                # V/x then x, sharing the slot
                chords.append(Harmony(_secondary_dominant(root), "major", slot / 2, f"V/{numeral}"))
                beats = slot / 2
            elif quality == "major" and rng.random() < 0.1 and slot >= 2:
                chords.append(Harmony(root, "sus4", slot / 2, numeral + "sus4"))
                beats = slot / 2
            chords.append(Harmony(root, quality, beats, numeral))
        out.append(chords)
        if not last and not phrase_end and rng.random() < 0.05:
            # pivot to the dominant or subdominant key, back home at phrase end
            tonic = key.tonic.transpose(rng.choice(("P5", "P4"))).name
            key = music21.key.Key(tonic, key.mode)
            diatonic = _diatonic(key)
        if phrase_end:
            key, diatonic = home, _diatonic(home)
    return out


def voice_chord(
    harmony: Harmony,
    previous: list[music21.pitch.Pitch] | None,
    low: int,
    high: int,
    *,
    tones: int | None = None,
) -> list[music21.pitch.Pitch]:
    """Voice a chord within MIDI [low, high], choosing the inversion and
    octave that move least from ``previous`` (chirality's voice leading)."""
    base = chord_pitches(harmony.root, harmony.quality, 4)
    if tones is not None and tones < len(base):
        # root, third, fifth: the seventh goes first
        keep = list(range(tones))
        base = [base[i] for i in sorted(keep)]
    candidates = []
    for inv in range(len(base)):
        rotated = [music21.pitch.Pitch(p.nameWithOctave) for p in base[inv:] + base[:inv]]
        for i in range(1, len(rotated)):
            while rotated[i].midi <= rotated[i - 1].midi:
                rotated[i].octave = (rotated[i].octave or 4) + 1
        for shift in range(-4, 4):
            voiced = [octave(p, shift) for p in rotated]
            if voiced[0].midi >= low and voiced[-1].midi <= high:
                candidates.append(voiced)
    if not candidates:
        return [octave(p, round((low + high) / 2 / 12 - 5)) for p in base]
    if not previous:
        center = (low + high) / 2
        return min(candidates, key=lambda v: abs(sum(p.midi for p in v) / len(v) - center))

    def distance(v: list[music21.pitch.Pitch]) -> float:
        a = sorted(p.midi for p in previous)
        b = sorted(p.midi for p in v)
        n = max(len(a), len(b))
        a += [a[-1]] * (n - len(a))
        b += [b[-1]] * (n - len(b))
        return sum(abs(x - y) for x, y in zip(a, b, strict=True))

    return min(candidates, key=distance)
