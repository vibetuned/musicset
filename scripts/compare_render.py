"""Check id injection and engraving against an existing kakigori/gelato dataset.

    uv run --with verovio==6.0.1 python scripts/compare_render.py DATASET_DIR

DATASET_DIR holds filter/ (sources), injected/ and svgs/ from the old pipeline.
"""

import sys
import zipfile
from pathlib import Path

from musicset.ids import id_prefix, inject_ids
from musicset.render.engrave import Engraving
from musicset.render.options import RenderOptions
from musicset.score import read_musicxml


def main() -> None:
    root = Path(sys.argv[1])
    same_ids = same_svg = pages = scores = 0
    for src in sorted((root / "filter").glob("*.mxl")):
        old = root / "injected" / src.name
        if not old.exists():
            continue
        scores += 1
        new_text = inject_ids(read_musicxml(src).text, id_prefix(src.stem))
        with zipfile.ZipFile(old) as zf:
            name = next(n for n in zf.namelist() if n.endswith((".xml", ".musicxml")))
            old_text = zf.read(name).decode()
        same_ids += new_text == old_text

        engraving = Engraving.load(old_text, RenderOptions(svg_attributes=()))
        for number, svg in engraving.pages():
            ref = root / "svgs" / f"{src.stem}_page{number}.svg"
            if ref.exists():
                pages += 1
                same_svg += svg == ref.read_text(encoding="utf-8")
    print(f"scores {scores}: injected MusicXML identical {same_ids}")
    print(f"pages {pages}: SVG byte-identical {same_svg}")


if __name__ == "__main__":
    main()
