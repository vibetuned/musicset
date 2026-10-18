"""Species counterpoint scores, from a pool of composed pieces.

Composing a piece (``synth.counterpoint``, Herremans & Sorensen's VNS) takes
4-90 seconds, so pieces are composed once into a pool (JSON lines) and each
score is a cheap variation of one: transposed to any key (the rules are
interval-based, so a transposed piece is exactly as good), written in 4/4,
common time, cut time or halved into 2/4, and laid out as a piano grand
staff, an instrumental duet (alto and tenor clefs included), two voices on
one staff, or a two-part choir with lyrics.

Pool pieces are composed in C major and A minor: their pitches are then
spelled correctly by MIDI number alone, and music21's interval transposition
keeps the spelling right in the target key.
"""

from __future__ import annotations

import contextlib
import json
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import music21

from musicset.synth.counterpoint.generate import generate_piece
from musicset.synth.counterpoint.herremans12 import Key
from musicset.synth.counterpoint.notes import TIE_END, TIE_START, Note, Species, as_notes
from musicset.synth.decorate import Profile, decorate, profile
from musicset.synth.theory import ALL_MAJOR_KEYS, ALL_MINOR_KEYS

POOL_TONIC = {"major": "C", "minor": "A"}


@dataclass(frozen=True, slots=True)
class PoolTask:
    seed: int
    species: int
    measures: int
    mode: str
    maxiters: int = 2


def compose(task: PoolTask) -> dict:
    """One pool piece (a module-level function, for process pools)."""
    key = Key(POOL_TONIC[task.mode], task.mode)
    piece = generate_piece(
        key,
        measures=task.measures,
        seed=task.seed,
        species=Species(task.species),
        maxiters=task.maxiters,
    )
    return {
        "seed": task.seed,
        "species": task.species,
        "mode": task.mode,
        "measures": task.measures,
        "cf": [int(p) for p in piece.cf],
        "cp": [list(n) for n in as_notes(piece.cp)],
        "cf_cost": round(float(piece.cf_cost), 6),
        "cp_cost": round(float(piece.cp_cost), 6),
    }


def pool_tasks(
    count: int,
    seed: int = 0,
    species: tuple[int, ...] = (1, 2, 3, 4, 5),
    measures: tuple[int, ...] = (8, 10, 12),
    modes: tuple[str, ...] = ("major", "minor"),
    maxiters: int = 2,
) -> list[PoolTask]:
    rng = random.Random(f"pool/{seed}")
    return [
        PoolTask(
            seed=seed * 1_000_000 + i,
            species=species[i % len(species)],
            measures=rng.choice(measures),
            mode=rng.choice(modes),
            maxiters=maxiters,
        )
        for i in range(count)
    ]


def load_pool(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# =============================================================================
# Pieces to scores
# =============================================================================

METERS = (
    ("4/4", 1.0, 0.35),
    ("C", 1.0, 0.2),
    ("cut", 1.0, 0.15),
    ("2/2", 1.0, 0.1),
    ("2/4", 0.5, 0.2),
)
LAYOUTS = (("piano", 0.4), ("duet", 0.3), ("single", 0.15), ("choir", 0.15))
# (instrument, clef) for the upper and the lower voice of a duet
UPPER = (
    (music21.instrument.Violin, music21.clef.TrebleClef),
    (music21.instrument.Flute, music21.clef.TrebleClef),
    (music21.instrument.Oboe, music21.clef.TrebleClef),
    (music21.instrument.Clarinet, music21.clef.TrebleClef),
)
LOWER = (
    (music21.instrument.Violoncello, music21.clef.BassClef),
    (music21.instrument.Violoncello, music21.clef.TenorClef),
    (music21.instrument.Viola, music21.clef.AltoClef),
    (music21.instrument.Bassoon, music21.clef.BassClef),
    (music21.instrument.Trombone, music21.clef.TenorClef),
)
TEXTS = (
    "Ky-ri-e e-le-i-son Chri-ste e-le-i-son",
    "A-ve Ma-ri-a gra-ti-a ple-na Do-mi-nus te-cum",
    "Glo-ri-a in ex-cel-sis De-o et in ter-ra pax",
    "San-ctus San-ctus Do-mi-nus De-us Sa-ba-oth",
    "Al-le-lu-ia al-le-lu-ia",
)


def _syllables(text: str) -> Iterator[tuple[str, str]]:
    while True:
        for word in text.split():
            parts = word.split("-")
            for i, s in enumerate(parts):
                if len(parts) == 1:
                    kind = "single"
                else:
                    kind = "begin" if i == 0 else "end" if i == len(parts) - 1 else "middle"
                yield s, kind


def _weighted[T](rng: random.Random, items: tuple[tuple, ...]) -> tuple:
    return rng.choices(items, weights=[it[-1] for it in items])[0]


def _measures(
    notes: list[Note],
    count: int,
    scale: float,
    lyrics: Iterator[tuple[str, str]] | None = None,
) -> list[music21.stream.Measure]:
    """One voice as measures; a late first entry gets its rest."""
    by_measure: list[list[Note]] = [[] for _ in range(count)]
    for n in notes:
        by_measure[n.measure].append(n)
    out = []
    for i, ns in enumerate(by_measure):
        m = music21.stream.Measure(number=i + 1)
        if ns and ns[0].beat > 1.0:
            m.insert(0.0, music21.note.Rest(quarterLength=(ns[0].beat - 1.0) * scale))
        for n in ns:
            pitch = music21.pitch.Pitch(midi=n.pitch)
            pitch.spellingIsInferred = False  # else transposing respells it
            e = music21.note.Note(pitch, quarterLength=n.duration * scale)
            if n.tied == TIE_START:
                e.tie = music21.tie.Tie("start")
            elif n.tied == TIE_END:
                e.tie = music21.tie.Tie("stop")
            elif lyrics is not None:
                text, kind = next(lyrics)
                e.lyrics.append(music21.note.Lyric(text, syllabic=kind))  # ty: ignore[invalid-argument-type]
            m.insert((n.beat - 1.0) * scale, e)
        out.append(m)
    return out


def _head(
    m: music21.stream.Measure, meter: str, key: music21.key.Key, clef: music21.clef.Clef
) -> None:
    m.insert(0.0, clef)
    m.insert(0.0, music21.key.Key(key.tonic.name, key.mode))
    m.insert(0.0, music21.meter.TimeSignature(meter))


# Middle line of each staff, MIDI: B4 and D3
STAFF_CENTERS = {"treble": 71, "bass": 50}


def _one_staff(cf: list[Note], cp: list[Note]) -> tuple[str, int] | None:
    """(clef, octaves to move the piece) to put both voices on one staff
    with at most a few ledger lines; None when they span too wide."""
    low, high = min(n.pitch for n in cf), max(n.pitch for n in cp)
    if high - low > 26:
        return None
    middle = (low + high) / 2
    fits = {c: round((center - middle) / 12) for c, center in STAFF_CENTERS.items()}
    clef = min(fits, key=lambda c: abs(fits[c] * 12 + middle - STAFF_CENTERS[c]))
    return clef, fits[clef]


def _transposition(mode: str, rng: random.Random) -> music21.interval.Interval:
    """From the pool key to a random key of the same mode, within a tritone."""
    tonic = rng.choice(ALL_MAJOR_KEYS if mode == "major" else ALL_MINOR_KEYS)
    start = music21.pitch.Pitch(POOL_TONIC[mode] + "4")
    target = music21.pitch.Pitch(tonic + "4")
    interval = music21.interval.Interval(start, target)
    if interval.semitones > 6:
        interval = music21.interval.Interval(start, target.transpose("P-8"))
    elif interval.semitones < -6:
        interval = music21.interval.Interval(start, target.transpose("P8"))
    return interval


def piece_score(
    piece: dict, rng: random.Random, p: Profile | str = "default"
) -> music21.stream.Score:
    p = profile(p)
    cf = as_notes(piece["cf"])
    cp = [Note(*n) for n in piece["cp"]]
    count = piece["measures"]
    meter, scale, _ = _weighted(rng, METERS)
    layout = _weighted(rng, LAYOUTS)[0]
    staff = _one_staff(cf, cp) if layout == "single" else None
    if layout == "single" and staff is None:
        layout = "piano"
    if staff is not None and staff[1]:
        cf = [n._replace(pitch=n.pitch + 12 * staff[1]) for n in cf]
        cp = [n._replace(pitch=n.pitch + 12 * staff[1]) for n in cp]
    key = music21.key.Key(POOL_TONIC[piece["mode"]], piece["mode"])

    score = music21.stream.Score()
    lyrics = rng.choice(TEXTS) if layout == "choir" else None
    upper = _measures(cp, count, scale, _syllables(lyrics) if lyrics else None)
    lower = _measures(cf, count, scale, _syllables(lyrics) if lyrics else None)
    if layout == "single":
        part = music21.stream.Part()
        part.insert(0, music21.instrument.Piano())
        for i, (a, b) in enumerate(zip(upper, lower, strict=True)):
            m = music21.stream.Measure(number=i + 1)
            if i == 0:
                clef = music21.clef.BassClef() if staff and staff[0] == "bass" else None
                _head(m, meter, key, clef or music21.clef.TrebleClef())
            for n, (voice, stem) in enumerate(((a, "up"), (b, "down")), start=1):
                v = music21.stream.Voice(id=str(n))
                for e in voice.notesAndRests:
                    if isinstance(e, music21.note.NotRest):
                        e.stemDirection = stem
                    v.insert(e.offset, e)
                m.insert(0.0, v)
            part.append(m)
        score.insert(0, part)
    else:
        if layout == "piano":
            staves: list[music21.stream.Part] = [
                music21.stream.PartStaff(),
                music21.stream.PartStaff(),
            ]
            staves[0].insert(0, music21.instrument.Piano())
            clefs: list[music21.clef.Clef] = [music21.clef.TrebleClef(), music21.clef.BassClef()]
        elif layout == "duet":
            (hi, hi_clef), (lo, lo_clef) = rng.choice(UPPER), rng.choice(LOWER)
            staves = [music21.stream.Part(), music21.stream.Part()]
            staves[0].insert(0, hi())
            staves[1].insert(0, lo())
            clefs = [hi_clef(), lo_clef()]
        else:
            staves = [music21.stream.Part(), music21.stream.Part()]
            for staff, voice in zip(
                staves, (music21.instrument.Soprano(), music21.instrument.Bass()), strict=True
            ):
                staff.insert(0, voice)
            clefs = [music21.clef.TrebleClef(), music21.clef.BassClef()]
        for staff, measures, clef in zip(staves, (upper, lower), clefs, strict=True):
            _head(measures[0], meter, key, clef)
            for m in measures:
                staff.append(m)
            score.insert(0, staff)
        symbol = "brace" if layout == "piano" else "bracket"
        score.insert(
            0, music21.layout.StaffGroup(staves, symbol=symbol, barTogether=layout != "choir")
        )

    for staff in score.parts:
        last = list(staff.getElementsByClass(music21.stream.Measure))[-1]
        last.rightBarline = music21.bar.Barline("final")
    interval = _transposition(piece["mode"], rng)
    if interval.semitones:
        score = score.transpose(interval)
    for staff in score.parts:
        with contextlib.suppress(music21.exceptions21.Music21Exception):
            staff.makeBeams(inPlace=True)
    return decorate(score, p, rng, notes=layout != "choir", ties=False)


def generator(pool: list[dict]) -> Callable[[random.Random, Profile | str], music21.stream.Score]:
    def make(rng: random.Random, p: Profile | str = "default") -> music21.stream.Score:
        return piece_score(rng.choice(pool), rng, p)

    return make
