"""Choosing which source scores go into a dataset, before any rendering.

Everything here reads the MusicXML once, so a 250k-file corpus is scanned in
minutes. Gates that need the engraving (page count, layout density) are build
options instead (:class:`musicset.pipeline.BuildConfig`).

Merges kakigori's ``filter-mxl --lite`` gates and gelato's voice-count sampling
(``select_real_scores``).
"""

from __future__ import annotations

import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from musicset.score import read_musicxml

_FIFTHS = re.compile(r"<fifths>(-?\d+)</fifths>")


@dataclass(slots=True)
class ScoreStats:
    path: Path
    parts: int = 0
    max_staves: int = 1
    """Most staves in any one part."""
    voices: int = 0
    """Distinct (part, staff, voice) triples: the voice count xml2abc will emit."""
    max_voices_per_staff: int = 0
    notes: int = 0
    tab_or_percussion: bool = False
    gestural_accidentals: bool = False
    """Many alterations, none printed and no key signature: pitches not on the page."""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        return d


def score_stats(path: Path) -> ScoreStats:
    stats = ScoreStats(path)
    try:
        xml = read_musicxml(path).text
        root = ET.fromstring(xml)
    except Exception as e:
        stats.error = f"{type(e).__name__}: {e}"
        return stats

    stats.parts = len(root.findall("part-list/score-part"))
    for part in root.iter("part"):
        staves = [int(s.text) for s in part.iter("staves") if s.text and s.text.isdigit()]
        stats.max_staves = max([stats.max_staves, *staves])
        per_staff: dict[str, set[str]] = defaultdict(set)
        for note in part.iter("note"):
            stats.notes += 1
            per_staff[note.findtext("staff") or "1"].add(note.findtext("voice") or "1")
        stats.voices += max(1, sum(len(v) for v in per_staff.values()))
        stats.max_voices_per_staff = max(
            [stats.max_voices_per_staff, *(len(v) for v in per_staff.values())]
        )

    stats.tab_or_percussion = ">TAB<" in xml or "percussion" in xml.lower()
    alterations = xml.count("<alter>") - xml.count("<alter>0<")
    stats.gestural_accidentals = (
        alterations > 10
        and "<accidental" not in xml
        and not any(f != "0" for f in _FIFTHS.findall(xml))
    )
    return stats


def lite_ok(stats: ScoreStats) -> bool:
    """kakigori's LITE corpus: one part on one staff or a piano grand staff, one
    voice per staff (several are mostly editor artifacts in PDMX), no tablature
    or percussion, and every pitch printed."""
    return (
        stats.error is None
        and stats.parts == 1
        and stats.max_staves <= 2
        and stats.max_voices_per_staff <= 1
        and not stats.tab_or_percussion
        and not stats.gestural_accidentals
    )


@dataclass(frozen=True, slots=True)
class Selection:
    lite: bool = False
    max_voices: int = 0
    """Keep scores with at most this many voices (0: any)."""
    per_voice: tuple[int, ...] = ()
    """Cap per voice count (1, 2, 3...): balances a corpus skewed to one voice."""
    limit: int = 0
    exclude: frozenset[str] = frozenset()
    """Stems never to select (e.g. another split, for holdout safety)."""
    seed: int = 42

    def accepts(self, stats: ScoreStats) -> bool:
        if stats.error is not None or stats.path.stem in self.exclude:
            return False
        if self.lite and not lite_ok(stats):
            return False
        return not (self.max_voices and not 0 < stats.voices <= self.max_voices)

    def choose(self, candidates: list[ScoreStats]) -> list[ScoreStats]:
        """Accepted scores, shuffled reproducibly, capped per voice count and in total."""
        pool = sorted((s for s in candidates if self.accepts(s)), key=lambda s: str(s.path))
        random.Random(self.seed).shuffle(pool)
        if self.per_voice:
            taken: dict[int, int] = defaultdict(int)
            capped = []
            for s in pool:
                cap = self.per_voice[s.voices - 1] if s.voices <= len(self.per_voice) else None
                if cap is None or taken[s.voices] < cap:
                    taken[s.voices] += 1
                    capped.append(s)
            pool = capped
        return pool[: self.limit] if self.limit else pool


def place(chosen: list[ScoreStats], out_dir: Path, *, copy: bool = False) -> list[Path]:
    """Link (or copy) chosen sources into ``out_dir``; name clashes get the parent dir appended."""
    out_dir.mkdir(parents=True, exist_ok=True)
    placed = []
    for s in chosen:
        dst = out_dir / s.path.name
        if dst.exists() or dst.is_symlink():
            if dst.resolve() == s.path.resolve():
                placed.append(dst)
                continue
            dst = out_dir / f"{s.path.stem}_{s.path.parent.name}{s.path.suffix}"
        if copy:
            shutil.copy2(s.path, dst)
        else:
            dst.symlink_to(s.path.resolve())
        placed.append(dst)
    return placed
