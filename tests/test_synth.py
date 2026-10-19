import json
import random
import shutil
import zipfile
from pathlib import Path

import music21
import numpy as np
import pytest
from conftest import FIXTURES

from musicset.cli import main
from musicset.labels import LabelSet
from musicset.pipeline import BuildConfig, Layout, build_score
from musicset.score import read_musicxml
from musicset.synth import SynthTask, generate, synthesize
from musicset.synth.counterpoint.generate import generate_piece
from musicset.synth.counterpoint.herremans12 import Key
from musicset.synth.counterpoint.notes import Species, attacks, sample_skeleton
from musicset.synth.decorate import PROFILES, add_octave_clefs
from musicset.synth.exercises import RUN_RANGE
from musicset.synth.species import PoolTask, compose, piece_score
from musicset.synth.theory import CircleOfFifths, Harmony, octave, voice_chord
from musicset.synth.write import musical_digest, write_score

KINDS = ("exercise", "piano", "easy", "melody")


@pytest.fixture(scope="module")
def pool() -> list[dict]:
    return [
        compose(PoolTask(seed=s, species=sp, measures=5, mode=m, maxiters=1))
        for s, sp, m in ((1, 1, "major"), (2, 2, "minor"), (3, 4, "major"))
    ]


# --- counterpoint engine (chirality port) ---


def test_counterpoint_matches_chirality():
    """Pinned from chirality: the port composes the very same pieces."""
    p = generate_piece(Key("C", "major"), measures=6, seed=3, species=Species.FIRST, maxiters=1)
    assert p.cf == [52, 50, 52, 53, 50, 48]
    assert p.cp == [79, 81, 84, 84, 83, 84]
    p = generate_piece(Key("A", "minor"), measures=5, seed=4, species=Species.FOURTH, maxiters=1)
    assert p.cf == [68, 72, 71, 71, 69]
    assert [n.pitch for n in p.cp] == [76, 76, 79, 79, 81, 81, 80, 81]
    assert round(p.cp_cost, 6) == 1.642857


@pytest.mark.parametrize("species", list(Species))
def test_skeleton_measures_are_full(species):
    notes = sample_skeleton(species, 6, np.random.default_rng(0))
    for m in range(1, 6):
        total = sum(n.duration for n in notes if n.measure == m)
        assert total == 4.0
    assert len(attacks(notes)) <= len(notes)


# --- theory ---


def test_octave_keeps_spelling():
    p = music21.pitch.Pitch("G-4")
    assert p.transpose(12).name == "F#"  # why octave() exists
    assert octave(p, 1).nameWithOctave == "G-5"
    assert octave(p, -2).nameWithOctave == "G-2"


def test_chords_are_spelled_from_their_root():
    voiced = voice_chord(Harmony("E-", "minor", 4.0, "i"), None, 48, 72)
    assert sorted(p.name for p in voiced) == ["B-", "E-", "G-"]


def test_leading_tone_chords():
    c = CircleOfFifths()
    assert c.neighbors("G")["vii°"] == "F#°"
    assert c.neighbors("Db")["vii°"] == "C°"


# --- generators ---


@pytest.mark.parametrize("kind", KINDS)
def test_generators_write_readable_reproducible_scores(tmp_path: Path, kind: str):
    for i in range(3):
        task = SynthTask(kind, i, tmp_path / "a", profile="detector")
        synthesize(task)
        again = SynthTask(kind, i, tmp_path / "b", profile="detector")
        synthesize(again)
        assert musical_digest(task.path) == musical_digest(again.path)
        xml = read_musicxml(task.path)
        assert xml.name == f"{task.path.stem}.musicxml"
        assert "Music21" not in xml.text  # no placeholder title or composer


@pytest.mark.parametrize("kind", KINDS)
def test_notes_never_share_pitch_objects(kind: str):
    """music21 transposes ottava notes in place: a shared Pitch moves elsewhere too."""
    for i in range(4):
        score = generate(SynthTask(kind, i, Path("unused"), profile="detector"))
        pitches = [id(p) for n in score.recurse().notes for p in n.pitches]
        assert len(pitches) == len(set(pitches))


def test_scale_runs_stay_in_range():
    for i in range(30):
        score = generate(
            SynthTask("exercise", i, Path("unused"), options=(("contents", ("major",)),))
        )
        for part, hand in zip(score.parts, ("RH", "LH"), strict=False):
            low, high = RUN_RANGE[hand]
            midis = [p.midi for n in part.recurse().notes for p in n.pitches]
            assert low <= min(midis) and max(midis) <= high


def test_exercise_grand_staff_is_one_braced_part(tmp_path: Path):
    task = SynthTask("exercise", 0, tmp_path, options=(("staves", 2),))
    synthesize(task)
    xml = read_musicxml(task.path).text
    assert xml.count("<score-part ") == 1
    assert "<staves>2</staves>" in xml


def test_octave_clefs_open_the_part():
    score = generate(SynthTask("melody", 0, Path("unused")))
    clef = score.recurse().getElementsByClass(music21.clef.Clef).first()
    clef.activeSite.replace(clef, music21.clef.TrebleClef())
    add_octave_clefs(score, 1.0, 1.0, random.Random(0))
    first = score.parts[0].recurse().getElementsByClass(music21.clef.Clef).first()
    assert isinstance(first, music21.clef.Treble8vaClef)


def test_counterpoint_scores_stay_in_key(pool: list[dict]):
    layouts = set()
    for i in range(24):
        score = piece_score(pool[i % len(pool)], random.Random(i), PROFILES["plain"])
        key = score.recurse().getElementsByClass(music21.key.Key).first()
        assert key is not None
        allowed = {p.name for p in key.pitches}
        if key.mode == "minor":
            allowed.add(key.pitchFromDegree(7).transpose("A1").name)  # leading tone
        assert {n.pitch.name for n in score.recurse().notes} <= allowed
        layouts.add((len(score.parts), bool(list(score.recurse().getElementsByClass("Voice")))))
    assert len(layouts) >= 2


def test_write_keeps_exactly_the_generator_ties(tmp_path: Path):
    part = music21.stream.Part()
    for number, pitches in ((1, ("C4", "C4", "D4", "E4")), (2, ("E4", "F4", "F4", "G4"))):
        m = music21.stream.Measure(number=number)
        for p in pitches:
            m.append(music21.note.Note(p, quarterLength=1.0))
        part.append(m)
    notes = list(part.recurse().notes)
    notes[3].tie, notes[4].tie = music21.tie.Tie("start"), music21.tie.Tie("stop")  # over the bar
    score = music21.stream.Score([part])
    write_score(score, tmp_path / "t.mxl")
    xml = read_musicxml(tmp_path / "t.mxl").text
    assert xml.count('<tie type="start"') == 1
    assert xml.count('<tie type="stop"') == 1


def test_mutations_of_a_real_score(tmp_path: Path):
    for kind in ("mutate", "arrange"):
        task = SynthTask(kind, 0, tmp_path, source=FIXTURES / "tiny.musicxml")
        synthesize(task)
        assert task.path.name == f"{kind}_000000_tiny.mxl"
        assert zipfile.is_zipfile(task.path)


# --- CLI and pipeline ---


def test_cli_synth_resumes_and_extends(tmp_path: Path):
    out = tmp_path / "easy"
    main(["synth", "easy", str(out), "-n", "3", "--workers", "1"])
    first = {p.name: p.stat().st_mtime_ns for p in out.glob("*.mxl")}
    main(["synth", "easy", str(out), "-n", "5", "--workers", "1"])
    files = {p.name: p.stat().st_mtime_ns for p in out.glob("*.mxl")}
    assert len(files) == 5
    assert all(files[name] == t for name, t in first.items())  # untouched


def test_cli_counterpoint_pool_and_scores(tmp_path: Path):
    pool = tmp_path / "pool.jsonl"
    args = ["counterpoint-pool", str(pool), "-n", "2", "--species", "1", "--measures", "4"]
    main([*args, "--maxiters", "1", "--workers", "2"])
    assert len(pool.read_text().splitlines()) == 2
    main([*args, "--maxiters", "1", "--workers", "2"])  # resumed: nothing new
    assert len(pool.read_text().splitlines()) == 2
    assert {json.loads(line)["species"] for line in pool.read_text().splitlines()} == {1}
    out = tmp_path / "cp"
    main(["synth", "counterpoint", str(out), "-n", "3", "--pool", str(pool), "--workers", "1"])
    assert len(list(out.glob("counterpoint_*.mxl"))) == 3


def test_synthetic_scores_build_with_kern(tmp_path: Path):
    src = tmp_path / "src"
    for kind in ("piano", "exercise"):
        synthesize(SynthTask(kind, 1, src, profile="detector"))
    layout = Layout(tmp_path / "out")
    layout.create()
    cfg = BuildConfig(labels=LabelSet.load("lite"), write_kern=True)
    for path in sorted(src.glob("*.mxl")):
        result = build_score((path, layout, cfg, 0))
        assert result.status == "built"
        assert (layout.kern / f"{path.stem}.krn").read_text().startswith("**kern")
    shutil.rmtree(src)


def test_arranged_triplets_are_proper_tuplets():
    from musicset.synth.mutate import _octave, _triplets

    m = music21.stream.Measure(number=1)
    m.timeSignature = music21.meter.TimeSignature("4/4")
    for p in ("C5", "D5", "E5", "F5"):
        m.append(music21.note.Note(p, quarterLength=1.0))
    _triplets(m, random.Random(0))
    notes = list(m.notes)
    assert len(notes) == 6
    assert [n.duration.tuplets[0].type for n in notes] == ["start", None, "stop"] * 2
    assert sum(n.quarterLength for n in notes) == 4

    m = music21.stream.Measure(number=1)
    for p in ("C6", "D6", "E6"):
        m.append(music21.note.Note(p, quarterLength=1.0))
    _octave(m, random.Random(0))
    assert [p.nameWithOctave for n in m.notes for p in n.pitches] == ["C6", "D6", "E6"]
    (ottava,) = m.getElementsByClass(music21.spanner.Ottava)
    assert ottava.type == "8va" and not ottava.transposing


def _eight_measures() -> music21.stream.Score:
    part = music21.stream.Part()
    for i in range(1, 9):
        m = music21.stream.Measure(number=i)
        if i == 1:
            m.insert(0, music21.key.Key("D"))
            m.insert(0, music21.meter.TimeSignature("4/4"))
        for p in ("D5", "E5", "F#5", "G5"):
            m.append(music21.note.Note(p, quarterLength=1.0))
        part.append(m)
    return music21.stream.Score([part])


def test_key_changes_cancel_the_old_signature():
    from musicset.synth.decorate import add_key_changes

    for seed in range(5):
        score = _eight_measures()
        add_key_changes(score, 1.0, random.Random(seed))
        keys = list(score.recurse().getElementsByClass(music21.key.KeySignature))
        assert len(keys) >= 2
        assert (keys[1].sharps or 0) < 2  # fewer sharps (or flats): naturals to draw


def test_measure_rests_keep_the_ends_and_tied_notes():
    from musicset.synth.decorate import add_measure_rests

    score = _eight_measures()
    notes = list(score.recurse().notes)
    notes[8].tie, notes[9].tie = music21.tie.Tie("start"), music21.tie.Tie("stop")  # measure 3
    add_measure_rests(score, 1.0, random.Random(0))
    measures = list(score.parts[0].getElementsByClass(music21.stream.Measure))
    rested = [i for i, m in enumerate(measures) if not list(m.notes)]
    assert rested == [1, 3, 4, 5, 6]  # not the first, last, or the one with a tie
    rest = next(iter(measures[1].notesAndRests))
    assert isinstance(rest, music21.note.Rest)
    assert rest.fullMeasure is True and rest.quarterLength == 4
