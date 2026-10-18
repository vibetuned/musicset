"""Dump kakigori's graph groundtruth edges for a dataset (run in kakigori's env).

    uv run --project ../kakigori python scripts/kakigori_graph_reference.py \
        MEI_DIR ANN_DIR STRUCTURE_JSON OUT_DIR
"""

import json
import sys
from pathlib import Path

from kakigori.graph.parsers import GroundTruthGraphBuilder, page_sort_key


def main() -> None:
    mei_dir, ann_dir, roles_file, out_dir = map(Path, sys.argv[1:5])
    roles = json.loads(roles_file.read_text())["node_roles"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for mei in sorted(mei_dir.glob("*.mei")):
        pages = sorted(ann_dir.glob(f"{mei.stem}_page*.json"), key=page_sort_key)
        if not pages:
            continue
        edges = GroundTruthGraphBuilder(str(mei), [str(p) for p in pages], roles).build_edges()
        (out_dir / f"{mei.stem}.json").write_text(json.dumps(edges))


if __name__ == "__main__":
    main()
