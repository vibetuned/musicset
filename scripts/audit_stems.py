"""Audit geometric stem classes against the written duration of their note.

    uv run python scripts/audit_stems.py PAGES_DIR [PAGES_DIR...]

Needs pages rendered with musicset's default svg_attributes (data-dur, data-grace).
"""

import sys
from collections import Counter
from pathlib import Path

from musicset.annotate.schema import PageAnnotation

EXPECTED = {"1": "stem4", "2": "stem4", "4": "stem4", "8": "stem8", "16": "stem16"}


def main() -> None:
    table: Counter[tuple[str, str, bool]] = Counter()
    examples: dict[tuple[str, str, bool], str] = {}
    for root in map(Path, sys.argv[1:]):
        for path in sorted(root.glob("*.json.gz")):
            page = PageAnnotation.load(path)
            for el in page.elements:
                if "stem" not in el.classes or el.parent is None:
                    continue
                holder = page.elements[el.parent]
                dur = holder.attrs.get("dur")
                if dur is None:
                    continue
                expected = EXPECTED.get(dur, "stem32")
                got = el.refined[0]
                key = (expected, got, "grace" in holder.attrs)
                table[key] += 1
                examples.setdefault(key, f"{path.name} {holder.id}")
    total = sum(table.values())
    wrong = {k: v for k, v in table.items() if k[0] != k[1]}
    print(f"stems: {total}  mismatched: {sum(wrong.values())}")
    for (exp, got, grace), n in sorted(wrong.items(), key=lambda kv: -kv[1]):
        tag = " (grace)" if grace else ""
        print(f"  expected {exp:7} got {got:7}{tag}: {n:6}   e.g. {examples[(exp, got, grace)]}")


if __name__ == "__main__":
    main()
