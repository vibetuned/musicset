"""Dump kakigori's extractor output for a directory of SVG/PNG pairs.

Run inside the kakigori environment:
    uv run --project ../kakigori python scripts/kakigori_reference.py \
        SVG_DIR IMG_DIR OUT_DIR CONF_DIR
"""

import json
import sys
from pathlib import Path

from kakigori.dataset.extract_annotations import (
    extract_from_svg,
    load_smufl_mapping,
    load_target_classes,
)
from PIL import Image


def main() -> None:
    svg_dir, img_dir, out_dir, conf = map(Path, sys.argv[1:5])
    targets = load_target_classes(str(conf / "config.json"))
    mapping = load_smufl_mapping(str(conf / "smufl_mapping.json"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for svg in sorted(svg_dir.glob("*.svg")):
        img = img_dir / f"{svg.stem}.png"
        if not img.exists():
            continue
        with Image.open(img) as im:
            w, h = im.size
        anns = extract_from_svg(svg, w, h, targets, mapping)
        record = {"image": img.name, "width": w, "height": h, "annotations": anns}
        (out_dir / f"{svg.stem}.json").write_text(json.dumps(record))


if __name__ == "__main__":
    main()
