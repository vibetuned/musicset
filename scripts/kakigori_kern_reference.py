"""kakigori's serializer on groundtruth graphs, with nodes in document order
(run in kakigori's env). validate_groundtruth feeds nodes from a set, whose
order changes with the hash seed; this reference fixes it for byte comparison.

    uv run --project ../kakigori python scripts/kakigori_kern_reference.py \
        MEI_DIR ANN_DIR STRUCTURE_JSON OUT_DIR
"""

import json
import sys
from pathlib import Path

import torch
from kakigori.graph.parsers import GroundTruthGraphBuilder, page_sort_key
from kakigori.graph.serializers import MinimalHumdrumSerializer


def main() -> None:
    mei_dir, ann_dir, roles_file, out_dir = map(Path, sys.argv[1:5])
    roles = json.loads(roles_file.read_text())["node_roles"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for mei in sorted(mei_dir.glob("*.mei")):
        paths = sorted(ann_dir.glob(f"{mei.stem}_page*.json"), key=page_sort_key)
        if not paths:
            continue
        edges = GroundTruthGraphBuilder(str(mei), [str(p) for p in paths], roles).build_edges()
        node_ids = sorted({u for u, _, _ in edges} | {v for _, v, _ in edges})
        index = {n: i for i, n in enumerate(node_ids)}
        edge_index = torch.tensor(
            [[index[u] for u, _, _ in edges], [index[v] for _, v, _ in edges]]
        )
        y = torch.tensor([k for _, _, k in edges])
        serializer = MinimalHumdrumSerializer(edge_index, y, roles, node_ids)

        node_map: dict = {}
        valid = set(node_ids)
        for path in paths:
            page_nodes = []
            for ann in json.loads(path.read_text())["annotations"]:
                if "id" not in ann:
                    continue
                if ann["id"] in node_map:
                    ann["id"] = f"{ann['id']}_{ann['class']}_{len(node_map)}"
                node_map[ann["id"]] = ann
                if ann["id"] in valid:
                    x1, y1, x2, y2 = ann["bbox"]
                    ann["cx"], ann["cy"] = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    page_nodes.append(ann)
            if page_nodes:
                serializer.add_page(page_nodes)
        kern = serializer.export_to_krn()
        if not kern.startswith("Error"):
            (out_dir / f"{mei.stem}.krn").write_text(kern + "\n")


if __name__ == "__main__":
    main()
