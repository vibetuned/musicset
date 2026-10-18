"""musicset's kern view vs kakigori reference kern, byte for byte.

uv run python scripts/compare_kern_port.py MEI_DIR ANN_DIR LABELS REF_DIR
"""

import json
import sys
from pathlib import Path

from musicset.labels import LabelSet
from musicset.views.graph import build_edges, page_sort_key
from musicset.views.kern import score_kern


def main() -> None:
    mei_dir, ann_dir, labels, ref_dir = (
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        sys.argv[3],
        Path(sys.argv[4]),
    )
    roles = LabelSet.load(labels).node_roles
    same = total = 0
    for ref in sorted(ref_dir.glob("*.krn")):
        paths = sorted(ann_dir.glob(f"{ref.stem}_page*.json"), key=lambda p: page_sort_key(p))
        pages = [json.loads(p.read_text())["annotations"] for p in paths]
        edges = build_edges(mei_dir / f"{ref.stem}.mei", pages, roles)
        total += 1
        if score_kern(pages, edges, roles) == ref.read_text():
            same += 1
        elif total - same <= 3:
            print("differs:", ref.stem)
    print(f"kern files identical: {same}/{total}")


if __name__ == "__main__":
    main()
