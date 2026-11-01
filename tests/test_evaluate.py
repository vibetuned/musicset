import shutil
from pathlib import Path

from conftest import FIXTURES

from musicset.evaluate import score_kern
from musicset.labels import LabelSet
from musicset.pipeline import BuildConfig, Layout, build_score


def _build(tmp_path: Path) -> Layout:
    (tmp_path / "src").mkdir()
    shutil.copy(FIXTURES / "tiny.musicxml", tmp_path / "src" / "tiny.musicxml")
    layout = Layout(tmp_path / "out")
    layout.create()
    cfg = BuildConfig(labels=LabelSet.load("lite"), write_kern=True)
    build_score((tmp_path / "src" / "tiny.musicxml", layout, cfg, 0))
    return layout


def test_ground_truth_kern_matches_its_mei(tmp_path: Path):
    layout = _build(tmp_path)
    scores = score_kern(layout.mei / "tiny.mei", layout.kern / "tiny.krn")
    assert scores["notes"] > 0
    assert scores["pitch"] == scores["rhythm"] == scores["precision"] == 100.0
    assert scores["ratio"] == 1.0


def test_wrong_and_extra_notes_cost_recall_and_precision(tmp_path: Path):
    layout = _build(tmp_path)
    lines = (layout.kern / "tiny.krn").read_text().splitlines()
    # the first note (G4) read as a B2 nowhere in the score, and the C5 doubled
    first = next(i for i, line in enumerate(lines) if line == "8gL")
    lines[first] = "8BBL"
    c5 = next(i for i, line in enumerate(lines) if line == "16cc")
    lines[c5] = "16cc 16cc"
    wrong = tmp_path / "wrong.krn"
    wrong.write_text("\n".join(lines) + "\n")
    scores = score_kern(layout.mei / "tiny.mei", wrong)
    n = scores["notes"]
    assert scores["pitch"] == 100.0 * (n - 1) / n  # recall: one note missed
    assert scores["precision"] == 100.0 * (n - 1) / (n + 1)  # two notes out are wrong
    assert scores["ratio"] == (n + 1) / n


def test_breves_and_longs_score_as_two_and_four_whole_notes(tmp_path: Path):
    mei = tmp_path / "long.mei"
    mei.write_text(
        '<mei xmlns="http://www.music-encoding.org/ns/mei"><music><body><mdiv><score>'
        '<scoreDef><staffGrp><staffDef n="1" lines="5" clef.shape="G" clef.line="2"/></staffGrp></scoreDef>'
        '<section><measure n="1"><staff n="1"><layer n="1">'
        '<note pname="c" oct="4" dur="breve"/><note pname="d" oct="4" dur="long"/>'
        "</layer></staff></measure></section></score></mdiv></body></music></mei>"
    )
    krn = tmp_path / "long.krn"
    krn.write_text("**kern\n*staff1\n*clefG2\n=1\n0c\n00d\n*-\n")
    scores = score_kern(mei, krn)
    assert scores["notes"] == 2
    assert scores["pitch"] == scores["rhythm"] == 100.0
