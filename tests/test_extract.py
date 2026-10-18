from collections import Counter

import pytest

from musicset import PageAnnotation
from musicset.annotate.smufl import glyph_name

# Stem duration a note's written value implies (half and whole notes share stem4)
_STEM_FOR_DUR = {"1": 4, "2": 4, "4": 4, "8": 8, "16": 16, "32": 32}


def test_page_geometry(tiny_page: PageAnnotation):
    assert (tiny_page.width, tiny_page.height) == (21000.0, 29700.0)
    assert (tiny_page.pixel_width, tiny_page.pixel_height) == (2100.0, 2970.0)
    assert tiny_page.meta["renderer"].startswith("Verovio")
    for el in tiny_page.elements:
        if el.bbox is not None:
            x1, y1, x2, y2 = el.bbox
            assert 0 <= x1 <= x2 <= tiny_page.width
            assert 0 <= y1 <= y2 <= tiny_page.height


@pytest.mark.parametrize("fixture", ["tiny_page", "tiny_page_geometric"])
def test_refined_classes(fixture: str, request: pytest.FixtureRequest):
    page = request.getfixturevalue(fixture)
    has_durations = any("dur" in el.attrs for el in page.elements)
    assert has_durations == (fixture == "tiny_page")
    refined = Counter(c for el in page.elements for c in el.refined)
    assert refined == {
        "stem4": 1,
        "stem8": 2,
        "stem16": 4,
        "stem32": 1,
        "beam8": 1,
        "beam16": 1,
        "barlineFinal": 1,
    }


def test_stem_class_matches_written_duration(tiny_page: PageAnnotation):
    for el in tiny_page.elements:
        if "stem" in el.classes:
            assert el.parent is not None
            note = tiny_page.elements[el.parent]
            assert el.refined == (f"stem{_STEM_FOR_DUR[note.attrs['dur']]}",)


def test_stem_box_excludes_its_flag(tiny_page: PageAnnotation):
    flagged = next(el for el in tiny_page.elements if el.refined == ("stem32",))
    flag = next(el for el in tiny_page.elements if "flag" in el.classes)
    assert flagged.bbox is not None and flag.bbox is not None
    assert flag.bbox[2] > flagged.bbox[2]


def test_notes_carry_pitch_and_hierarchy(tiny_page: PageAnnotation):
    notes = [el for el in tiny_page.elements if el.classes[0] == "note"]
    assert [(n.attrs["pname"], n.attrs["oct"]) for n in notes] == [
        ("g", "4"),
        ("a", "4"),
        ("b", "4"),
        ("c", "5"),
        ("d", "5"),
        ("e", "5"),
        ("f", "4"),
        ("g", "4"),
    ]
    kinds = [el.classes[0] for el in tiny_page.ancestors(notes[-1].parent)]
    assert kinds[:4] == ["layer", "staff", "measure", "system"]


def test_glyphs_are_owned_by_their_element(tiny_page: PageAnnotation):
    names = Counter(glyph_name(g.code) for g in tiny_page.glyphs)
    assert names["noteheadBlack"] == 8
    assert names["gClef"] == 1
    assert names["flag32ndUp"] == 1
    clef = next(g for g in tiny_page.glyphs if g.code == "E050")
    assert clef.element is not None
    assert "clef" in tiny_page.elements[clef.element].classes


def test_system_staff_is_synthetic_and_precedes_its_system(tiny_page: PageAnnotation):
    i, staff = next((i, el) for i, el in enumerate(tiny_page.elements) if el.synthetic)
    assert staff.classes == ("system-staff",)
    assert staff.parent == i + 1
    assert "system" in tiny_page.elements[i + 1].classes


def test_round_trip(tmp_path, tiny_page: PageAnnotation):
    for name in ("p.json", "p.json.gz"):
        tiny_page.save(tmp_path / name)
        assert PageAnnotation.load(tmp_path / name) == tiny_page
