"""Score **kern against MEI: per staff and measure, multisets of pitches and of
pitch + duration (kakigori's compare-kern metric), with Humdrum spines tracked.

kakigori's compare_kern keys measures by barline number and maps columns to
staves once, which fits its serializer but not general Humdrum (pickups,
restarted numbering, voice sub-spines). Here measures are counted in order and
spine splits/merges are followed; without ``*staffN`` tokens, staves come from
spine order (Humdrum lists staves bottom to top).

    uv run python scripts/compare_kern.py MEI_DIR KRN_DIR [KRN_DIR...]
"""

import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

KAKIGORI = Path(__file__).resolve().parents[2] / "kakigori/src/kakigori/graph/compare_kern.py"
_spec = importlib.util.spec_from_file_location("kakigori_compare_kern", KAKIGORI)
assert _spec is not None and _spec.loader is not None
_ck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ck)
mei_bags = _ck.mei_bags  # kakigori's validated MEI reading (key signatures, accidental carry)

KERN_ALTER = {"#": 1, "##": 2, "###": 3, "-": -1, "--": -2, "n": 0, "": 0}
NOTE = re.compile(r"([a-gA-G]+)(#{1,3}|-{1,2}|n)?")


def kern_bags(text: str) -> tuple[dict, dict]:
    lines = [line for line in text.splitlines() if line and not line.startswith("!!")]
    header = next(line for line in lines if line.startswith("**"))
    kinds = header.split("\t")
    n_kern = kinds.count("**kern")
    # columns: [spine kind, staff]; staves default to reverse spine order
    cols: list[list] = []
    k = 0
    for kind in kinds:
        staff = None
        if kind == "**kern":
            staff = str(n_kern - k)
            k += 1
        cols.append([kind, staff])

    pitch, rhythm = {}, {}
    measure, seen_data = 1, False
    for line in lines[lines.index(header) + 1 :]:
        toks = line.split("\t")
        if line.startswith("="):
            if seen_data:
                measure += 1
                seen_data = False
            continue
        if line.startswith("!"):
            continue
        if line.startswith("*"):
            new: list[list] = []
            i = 0
            while i < len(toks):
                tok, col = toks[i], cols[i]
                if m := re.fullmatch(r"\*staff(\d+)", tok):
                    col[1] = m.group(1)
                if tok == "*^":
                    new += [list(col), list(col)]
                elif tok == "*v":
                    j = i
                    while j < len(toks) and toks[j] == "*v":
                        j += 1
                    new.append(col)
                    i = j
                    continue
                elif tok == "*-":
                    pass
                elif tok == "*+":
                    new += [col, ["**?", None]]
                else:
                    new.append(col)
                i += 1
            if "*x" in toks:
                a = toks.index("*x")
                new[a], new[a + 1] = new[a + 1], new[a]
            cols = new
            continue
        for tok, (kind, staff) in zip(toks, cols, strict=False):
            if kind != "**kern" or staff is None or tok == ".":
                continue
            seen_data = True
            key = (staff, measure)
            p_bag = pitch.setdefault(key, Counter())
            r_bag = rhythm.setdefault(key, Counter())
            for sub in tok.split(" "):
                if "r" in sub or sub == ".":
                    continue
                m = NOTE.search(re.sub(r"^\D*\d+", "", sub))
                if not m or len(set(m.group(1).upper())) != 1:
                    continue
                letters = m.group(1)
                octave = len(letters) + 3 if letters.islower() else 4 - len(letters)
                note = (letters[0].upper(), octave, KERN_ALTER[m.group(2) or ""])
                p_bag[note] += 1
                if "q" in sub.lower():
                    continue
                d = re.match(r"\D*(\d+)(%(\d+))?", sub)
                if not d or int(d.group(1)) == 0:
                    continue
                value = 1.0 / int(d.group(1)) * (int(d.group(3)) if d.group(3) else 1)
                dots = re.sub(r"[^.]", "", sub.split(" ")[0])
                if dots:
                    value *= 2.0 - 0.5 ** len(dots)
                r_bag[(*note, round(value, 4))] += 1
    return pitch, rhythm


def match(gt: dict, out: dict) -> tuple[int, int]:
    total = sum(sum(b.values()) for b in gt.values())
    hit = sum(sum((b & out.get(k, Counter())).values()) for k, b in gt.items())
    return total, hit


def main() -> None:
    mei_dir = Path(sys.argv[1])
    for krn_dir in map(Path, sys.argv[2:]):
        pt = ph = rt = rh = files = perfect = 0
        for mei in sorted(mei_dir.glob("*.mei")):
            krn = krn_dir / f"{mei.stem}.krn"
            if not krn.exists():
                continue
            gp, gr = mei_bags(mei)
            kp, kr = kern_bags(krn.read_text(encoding="utf-8"))
            a, b = match(gp, kp)
            c, d = match(gr, kr)
            pt, ph, rt, rh = pt + a, ph + b, rt + c, rh + d
            files += 1
            perfect += a == b and c == d
        print(
            f"{krn_dir.name:24} files {files:4}  pitch {100 * ph / max(pt, 1):6.2f}%  "
            f"rhythm {100 * rh / max(rt, 1):6.2f}%  perfect files {perfect}/{files}  ({pt} notes)"
        )


if __name__ == "__main__":
    main()
