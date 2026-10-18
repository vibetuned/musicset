"""Compare musicset's detection view with kakigori reference annotations.

uv run python scripts/compare_detection.py SVG_DIR IMG_DIR REF_DIR CONF_DIR
"""

import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

from musicset.annotate.extract import annotate_svg
from musicset.views.detection import ClassMap, detection_annotations


def key(a: dict) -> tuple:
    return a["class"], tuple(a["bbox"]), a.get("id")


def main() -> None:
    svg_dir, img_dir, ref_dir, conf = map(Path, sys.argv[1:5])
    class_map = ClassMap.from_files(conf / "config.json", conf / "smufl_mapping.json")
    pages = identical = 0
    only_ref: Counter[str] = Counter()
    only_new: Counter[str] = Counter()
    order_diff = 0
    for ref_path in sorted(ref_dir.glob("*.json")):
        ref = json.loads(ref_path.read_text())
        with Image.open(img_dir / ref["image"]) as im:
            size = im.size
        page = annotate_svg(svg_dir / f"{ref_path.stem}.svg", score=ref_path.stem)
        new = detection_annotations(page, class_map, size)
        pages += 1
        ref_keys = [key(a) for a in ref["annotations"]]
        new_keys = [key(a) for a in new]
        if ref_keys == new_keys:
            identical += 1
            continue
        rc, nc = Counter(ref_keys), Counter(new_keys)
        if rc == nc:
            order_diff += 1
        for k in rc - nc:
            only_ref[k[0]] += 1
        for k in nc - rc:
            only_new[k[0]] += 1
    print(f"pages: {pages}  identical (order + values): {identical}")
    print(f"same boxes, other order: {order_diff}")
    print("only in kakigori:", dict(only_ref))
    print("only in musicset:", dict(only_new))


if __name__ == "__main__":
    main()
