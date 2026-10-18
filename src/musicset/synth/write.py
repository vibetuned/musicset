"""Writing generated scores, and checking that a batch is actually varied."""

from __future__ import annotations

import hashlib
import random
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path

import music21
from music21.musicxml.m21ToXml import GeneralObjectExporter


def _tie_pairs(score: music21.stream.Score) -> list[tuple[int, int]]:
    """Each intended tie as (start, stop) object ids, in reading order."""
    elements = [e for e in score.recurse() if getattr(e, "tie", None) is not None]
    pairs = []
    for i, el in enumerate(elements):
        if el.tie.type == "start":
            stop = next((e for e in elements[i + 1 :] if e.tie.type == "stop"), None)
            if stop is not None:
                pairs.append((id(el), id(stop)))
    return pairs


def _set_tie(el: music21.note.GeneralNote, kind: str) -> None:
    el.tie = music21.tie.Tie(kind)
    if isinstance(el, music21.chord.Chord):
        for n in el.notes:
            n.tie = music21.tie.Tie(kind)


_CONTAINER = """<?xml version="1.0" encoding="UTF-8"?>
<container>
  <rootfiles>
    <rootfile full-path="{name}" media-type="application/vnd.recordare.musicxml+xml"/>
  </rootfiles>
</container>
"""
# What music21 writes when a score has no title or composer of its own
_DEFAULT_CREDITS = re.compile(
    r"\s*<(?:movement-title>Music21 Fragment</movement-title"
    r"|creator type=\"composer\">Music21</creator)>"
)


def _keep_ties(score: music21.stream.Score) -> None:
    """``makeNotation`` keeping exactly the ties the generator placed.

    ``makeNotation`` splits durations at barlines and beams, but it also ties
    and unties notes on its own; so the intended ties are recorded first,
    every tie is stripped after notation, and only the recorded ones come back.
    """
    pairs = _tie_pairs(score)
    score.makeNotation(inPlace=True)
    for measure in score.recurse().getElementsByClass(music21.stream.Measure):
        measure.splitAtDurations()
    for el in score.recurse():
        if getattr(el, "tie", None) is not None:
            el.tie = None
        if isinstance(el, music21.chord.Chord):
            for n in el.notes:
                n.tie = None
    by_id = {id(el): el for el in score.recurse()}
    for start, stop in pairs:
        if start in by_id and stop in by_id:
            _set_tie(by_id[start], "start")
            _set_tie(by_id[stop], "stop")


def write_score(score: music21.stream.Score, path: Path, notation: str = "keep-ties") -> Path:
    """Write ``path`` (.mxl compressed, else plain MusicXML).

    ``notation``: ``keep-ties`` (generated scores: notation made, the
    generator's ties exactly), ``music21`` (music21's own notation pass, for
    edited real scores) or ``none`` (written as is). music21's placeholder
    title and composer are left out.
    """
    if notation == "keep-ties":
        _keep_ties(score)
    exporter = GeneralObjectExporter(score)
    exporter.makeNotation = notation == "music21"
    xml = _DEFAULT_CREDITS.sub("", exporter.parse().decode("utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    if path.suffix == ".mxl":
        name = f"{path.stem}.musicxml"
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(name, xml)
            zf.writestr("META-INF/container.xml", _CONTAINER.format(name=name))
    else:
        tmp.write_text(xml, encoding="utf-8")
    tmp.replace(path)
    return path


_UUID = re.compile(r'"[PI][0-9a-f]{32}"')  # music21's random part/instrument ids
_IDENTIFICATION = re.compile(r"<identification>.*?</identification>", re.S)


def musical_digest(path: Path) -> str | None:
    """A hash of the score's music, blind to ids and encoding dates."""
    try:
        if path.suffix == ".mxl":
            with zipfile.ZipFile(path) as z:
                name = next(
                    n
                    for n in z.namelist()
                    if n.lower().endswith((".musicxml", ".xml")) and "META-INF" not in n
                )
                xml = z.read(name).decode("utf-8", "replace")
        else:
            xml = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, zipfile.BadZipFile, StopIteration):
        return None
    xml = _UUID.sub('"X"', _IDENTIFICATION.sub("", xml))
    return hashlib.md5(xml.encode()).hexdigest()


def diversity(paths: Iterable[Path]) -> tuple[int, int]:
    """(distinct scores, readable scores).

    Restricting a generator (no ornaments, dynamics, slurs or fingerings, few
    meters) shrinks its space faster than it looks: a scale exercise is then
    fixed by its scale, key and meter, and a large batch silently repeats a
    few hundred pieces.
    """
    digests = [d for d in map(musical_digest, paths) if d is not None]
    return len(set(digests)), len(digests)


FORMS = (
    "Exercise", "Study", "Etude", "Sonatina", "Minuet", "Prelude", "Invention", "Chorale",
    "Scale Study", "Little Piece", "Waltz", "Song", "Canon", "Duet", "Allemande", "Bagatelle",
    "Lesson", "Air", "Romance", "Rondo", "Gavotte", "Melody", "Arpeggio Study", "Two-Part Study",
)  # fmt: skip


def _key_words(key: music21.key.Key) -> str:
    name = key.tonic.name.replace("-", "-flat").replace("#", "-sharp")
    return f"{name} {key.mode}" if key.mode == "minor" else f"{name} major"


def title(score: music21.stream.Score, rng: random.Random, untitled: float = 0.2) -> None:
    """A plausible title (form, number, key), or none ``untitled`` of the time."""
    metadata = score.metadata or music21.metadata.Metadata()
    score.metadata = metadata
    if rng.random() < untitled:
        return
    words = [rng.choice(FORMS)]
    if rng.random() < 0.4:
        words.append(f"No. {rng.randint(1, 30)}")
    key = score.recurse().getElementsByClass(music21.key.Key).first()
    if key is not None and rng.random() < 0.4:
        words.append(f"in {_key_words(key)}")
    metadata.title = " ".join(words)
