"""Compare the page view (per-page MusicXML, ABC) with a gelato dataset.

    uv run python scripts/compare_pages.py GELATO_DATASET_DIR

The dataset holds injected/, svgs/, paginated/ and abcs/ from gelato's pipeline.
"""

import copy
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from musicset.annotate.extract import annotate_svg
from musicset.score import read_musicxml
from musicset.views.abc import AbcError, to_abc
from musicset.views.paginate import slice_page


def gelato_prune(xml_root, valid_ids):
    """gelato.dataset.paginate_mxl.prune_musicxml, verbatim."""
    new_root = copy.deepcopy(xml_root)
    for part in new_root.findall(".//part"):
        running_attrs = ET.Element("attributes")
        measures_to_remove = []
        first_measure_kept = False
        for measure in part.findall("measure"):
            for attrs in measure.findall("attributes"):
                for child in attrs:
                    existing = running_attrs.find(child.tag)
                    if existing is not None:
                        running_attrs.remove(existing)
                    running_attrs.append(copy.deepcopy(child))
            measure_ids = {el.get("id") for el in measure.iter() if "id" in el.attrib}
            if measure_ids.intersection(valid_ids):
                if not first_measure_kept:
                    if len(running_attrs) > 0:
                        measure.insert(0, copy.deepcopy(running_attrs))
                    first_measure_kept = True
            else:
                measures_to_remove.append(measure)
        for m in measures_to_remove:
            part.remove(m)
    return new_root


def kept_measures(root):
    return [m.get("id") for m in root.iter("measure")]


def main() -> None:
    root = Path(sys.argv[1])
    stats: Counter[str] = Counter()
    for svg in sorted((root / "svgs").glob("*.svg")):
        ref_xml = root / "paginated" / f"{svg.stem}.mxl"
        score = svg.stem.rsplit("_page", 1)[0]
        source = root / "injected" / f"{score}.mxl"
        if not (ref_xml.exists() and source.exists()):
            continue
        stats["pages"] += 1
        tree = ET.fromstring(read_musicxml(source).text)
        svg_text = svg.read_text(encoding="utf-8")
        regex_ids = set(re.findall(r'id="([^"]+)"', svg_text))
        page_ids = annotate_svg(svg_text).ids()

        reference = read_musicxml(ref_xml).text
        stats["reference copy reproduces gelato"] += (
            ET.tostring(gelato_prune(tree, regex_ids), encoding="unicode") == reference
        )
        stats["page.ids() keeps the same measures"] += kept_measures(
            gelato_prune(tree, page_ids)
        ) == kept_measures(gelato_prune(tree, regex_ids))

        mine = ET.tostring(slice_page(tree, page_ids), encoding="unicode")
        if mine == reference:
            stats["MusicXML identical"] += 1
        else:
            multi = "<staves>" in reference
            stats[f"MusicXML differs ({'multi-staff' if multi else 'single staff'})"] += 1

        ref_abc = root / "abcs" / f"{svg.stem}.abc"
        if ref_abc.exists():
            stats["ABC pages"] += 1
            try:
                stats["ABC identical"] += to_abc(mine) == ref_abc.read_text(encoding="utf-8")
                stats["ABC identical (gelato prune)"] += to_abc(reference) == ref_abc.read_text(
                    encoding="utf-8"
                )
            except AbcError:
                stats["ABC failed"] += 1
    for k, v in stats.items():
        print(f"{k:40} {v}")


if __name__ == "__main__":
    main()
