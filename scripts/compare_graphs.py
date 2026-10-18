"""Compare musicset's graph view with kakigori reference edges.

uv run python scripts/compare_graphs.py MEI_DIR ANN_DIR STRUCTURE_JSON REF_DIR
"""

import json
import sys
from collections import Counter
from pathlib import Path

from musicset.views.graph import build_edges, page_sort_key


def main() -> None:
    mei_dir, ann_dir, roles_file, ref_dir = map(Path, sys.argv[1:5])
    roles = json.loads(roles_file.read_text())["node_roles"]
    scores = identical = edges_total = 0
    kinds: Counter[int] = Counter()
    for ref_path in sorted(ref_dir.glob("*.json")):
        ref = [tuple(e) for e in json.loads(ref_path.read_text())]
        pages = sorted(ann_dir.glob(f"{ref_path.stem}_page*.json"), key=page_sort_key)
        anns = [json.loads(p.read_text())["annotations"] for p in pages]
        new = build_edges(mei_dir / f"{ref_path.stem}.mei", anns, roles)
        scores += 1
        edges_total += len(ref)
        kinds.update(k for _, _, k in ref)
        if new == ref:
            identical += 1
        else:
            print(f"{ref_path.stem}: {len(ref)} ref vs {len(new)} new edges")
    print(f"scores {scores}: identical edge lists {identical}")
    print(f"edges {edges_total}: {dict(kinds)}")


if __name__ == "__main__":
    main()
