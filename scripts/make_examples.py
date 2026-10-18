"""Pick a small, varied set of PDMX scores into examples/ (gitignored) to try
the pipeline on.

    uv run python scripts/make_examples.py [--pdmx ../music_data] [--out examples]

Seeded and stratified, public domain without license conflicts, short enough
to build in seconds. Writes examples/scores/*.mxl and examples/examples.jsonl.
"""

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

from musicset.score import read_musicxml
from musicset.select import ScoreStats, lite_ok, score_stats

# (name, how many, test on stats + raw MusicXML)
STRATA = [
    ("single-staff", 6, lambda s, x: lite_ok(s) and s.max_staves == 1),
    ("grand-staff", 4, lambda s, x: lite_ok(s) and s.max_staves == 2),
    ("voices-per-staff", 4, lambda s, x: s.parts == 1 and s.max_voices_per_staff >= 2),
    ("lyrics", 4, lambda s, x: s.parts <= 3 and "<lyric" in x),
    ("ensemble-transposing", 4, lambda s, x: s.parts >= 4 and "<transpose" in x),
]
MAX_NOTES = 1200


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdmx", type=Path, default=Path(__file__).resolve().parents[2] / "music_data")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "examples")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    with (args.pdmx / "PDMX.csv").open(newline="") as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if r["license"] == "publicdomain"
            and r["subset:no_license_conflict"] == "True"
            and r["n_notes"].isdigit()
            and 0 < int(r["n_notes"]) <= MAX_NOTES
        ]
    random.Random(args.seed).shuffle(rows)

    scores = args.out / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    need = {name: n for name, n, _ in STRATA}
    chosen: list[tuple[str, ScoreStats]] = []
    for row in rows:
        if not any(need.values()):
            break
        path = args.pdmx / row["mxl"]
        if not path.exists():
            continue
        stats = score_stats(path)
        if stats.error or stats.tab_or_percussion or stats.gestural_accidentals:
            continue
        xml = read_musicxml(path).text
        for name, _, test in STRATA:
            if need[name] and test(stats, xml):
                need[name] -= 1
                chosen.append((name, stats))
                shutil.copy2(path, scores / path.name)
                break

    with (args.out / "examples.jsonl").open("w") as f:
        for name, stats in chosen:
            f.write(json.dumps({"category": name, **stats.to_dict()}) + "\n")
    print(f"{len(chosen)} scores in {scores}")
    for name, n, _ in STRATA:
        print(f"  {name:22} {n - need[name]}/{n}")


if __name__ == "__main__":
    main()
