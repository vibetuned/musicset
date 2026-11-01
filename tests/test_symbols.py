import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from musicset.annotate.schema import Element, Glyph, PageAnnotation
from musicset.cli import main
from musicset.pipeline import PAGE_SUFFIX
from musicset.symbols import (
    PRESETS,
    SYMBOL_BY_NAME,
    SYMBOL_GROUPS,
    SYMBOLS,
    _handwritten_strokes,
    _render_strokes,
    build_symbol_dataset,
    extract_dataset_symbols,
    extract_page_symbols,
    generate_symbol_samples,
    load_symbol_dataset,
    page_symbol_boxes,
    render_symbol,
    symbol_specs,
)


def test_symbol_catalog_covers_fingerings_1_to_5_and_all_groups():
    names = [s.name for s in SYMBOLS]
    assert len(names) == len(set(names)) == 33
    assert [s.name for s in symbol_specs(("fingering",))] == [
        "fing1",
        "fing2",
        "fing3",
        "fing4",
        "fing5",
    ]
    assert {s.group for s in SYMBOLS} == set(SYMBOL_GROUPS)
    with pytest.raises(ValueError, match="unknown symbol group"):
        symbol_specs(("unknown",))


@pytest.mark.parametrize("spec", SYMBOLS, ids=lambda s: s.name)
def test_every_symbol_renders_ink_on_paper(spec):
    img = render_symbol(spec, size=32, preset="clean", rng=0)
    assert img.shape == (32, 32) and img.dtype == np.uint8
    # Black ink (< 128) on white paper (255), with borders mostly paper
    ink = (img < 180).sum()
    assert 4 <= ink <= 32 * 32 * 0.75, f"{spec.name} ink pixels: {ink}"
    assert int(img[0, 0]) == 255


@pytest.mark.parametrize("spec", [s for s in SYMBOLS if s.handwritten], ids=lambda s: s.name)
def test_handwritten_stroke_templates_draw_ink(spec):
    rng = np.random.default_rng(7)
    strokes = _handwritten_strokes(spec.name, rng)
    canvas = _render_strokes(strokes, 96, 0.7, rng)
    assert canvas.shape == (96, 96)
    assert 0.005 < (canvas < 0.5).mean() < 0.60


@pytest.mark.parametrize("preset", PRESETS)
def test_presets_and_polarity_and_reproducibility(preset: str):
    a = render_symbol("trill", size=28, preset=preset, rng=42)
    b = render_symbol("trill", size=28, preset=preset, rng=42)
    inv = render_symbol("trill", size=28, preset=preset, invert=True, rng=42)
    assert a.shape == (28, 28) and np.array_equal(a, b)
    assert np.array_equal(inv, 255 - a)


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="unknown preset"):
        render_symbol("fing1", preset="watercolor")


def test_generate_symbol_samples_shape_and_labels():
    xs, ys, classes = generate_symbol_samples(
        3, groups=("fingering", "ornament"), size=32, preset="mixed", seed=5
    )
    assert len(classes) == 10
    assert xs.shape == (30, 32, 32) and xs.dtype == np.uint8
    assert ys.shape == (30,) and set(ys.tolist()) == set(range(10))


def test_page_symbol_extraction_splits_fingerings_and_skips_key_sig():
    page = PageAnnotation(
        score="s",
        page=1,
        width=1000.0,
        height=1000.0,
        pixel_width=200.0,
        pixel_height=200.0,
        elements=[
            Element("f1", ("fing",), (100, 100, 160, 180), text="1"),
            Element("f5", ("fing",), (200, 100, 260, 180), text="5"),
            Element("f0", ("fing",), (300, 100, 360, 180), text="0"),  # ignored: only 1-5
            Element("tr", ("trill",), (400, 100, 500, 180)),
            Element("ks", ("keySig",), (50, 50, 90, 150)),
            Element("ka", ("keyAccid",), (50, 50, 90, 150), parent=4),
            Element("ac", ("accid",), (600, 100, 660, 200)),
            Element("ar", ("artic",), (700, 100, 740, 140)),
        ],
        glyphs=[
            # Inside trill element: must not be emitted a second time
            Glyph("E566", (400, 100, 500, 180), element=3, owner="tr"),
            # Inside key signature: must be skipped
            Glyph("E262", (50, 50, 90, 150), element=5, owner="ka"),
            # Regular accidental and staccato articulation: must be extracted
            Glyph("E262", (600, 100, 660, 200), element=6, owner="ac"),
            Glyph("E4A2", (700, 100, 740, 140), element=7, owner="ar"),
        ],
    )
    boxes = page_symbol_boxes(page, (200, 200))
    assert [cls for cls, _ in boxes] == [
        "fing1",
        "fing5",
        "trill",
        "accidentalSharp",
        "articStaccatoAbove",
    ]

    canvas = np.full((200, 200), 255, dtype=np.uint8)
    canvas[20:36, 20:32] = 0
    crops = extract_page_symbols(page, canvas, size=32)
    assert len(crops) == 5
    assert crops[0][0] == "fing1" and crops[0][1].shape == (32, 32)
    assert (crops[0][1] < 200).any()


def test_extract_dataset_symbols_from_pages_and_annotations(tmp_path: Path):
    ds = tmp_path / "built"
    (ds / "pages").mkdir(parents=True)
    (ds / "imgs").mkdir(parents=True)
    (ds / "annotations").mkdir(parents=True)

    page = PageAnnotation(
        score="piece",
        page=1,
        width=200.0,
        height=200.0,
        pixel_width=200.0,
        pixel_height=200.0,
        elements=[
            Element("fg2", ("fing",), (20, 20, 45, 50), text="2"),
            Element("mrd", ("mordent",), (80, 20, 120, 50)),
        ],
        glyphs=[Glyph("E520", (130, 20, 170, 60), owner="dyn1")],
    )
    page.save(ds / "pages" / f"piece_page1{PAGE_SUFFIX}")
    img = np.full((200, 200), 255, dtype=np.uint8)
    img[25:45, 25:40] = 0
    img[25:45, 85:115] = 0
    img[25:55, 135:165] = 0
    Image.fromarray(img, mode="L").save(ds / "imgs" / "piece_page1.png")

    xs, ys, classes = extract_dataset_symbols(ds, size=32)
    extracted_names = [classes[i] for i in ys.tolist()]
    assert sorted(extracted_names) == ["dynamicPiano", "fing2", "mordent"]
    assert xs.shape == (3, 32, 32)

    # Fallback path: annotations/*.json without pages/*.json.gz
    aug = tmp_path / "aug"
    (aug / "imgs").mkdir(parents=True)
    (aug / "annotations").mkdir(parents=True)
    Image.fromarray(img, mode="L").save(aug / "imgs" / "piece_page1.png")
    ann = {
        "image": "piece_page1.png",
        "width": 200,
        "height": 200,
        "annotations": [
            {"class": "fing", "bbox": [20, 20, 45, 50], "text": "3"},
            {"class": "turn", "bbox": [80, 20, 120, 50]},
        ],
    }
    (aug / "annotations" / "piece_page1.json").write_text(json.dumps(ann))
    xs2, ys2, classes2 = extract_dataset_symbols(aug, groups=("fingering", "ornament"), size=28)
    assert sorted(classes2[i] for i in ys2.tolist()) == ["fing3", "turn"]
    assert xs2.shape == (2, 28, 28)


def test_build_and_cli_symbols_writes_npz_and_image_folder(tmp_path: Path):
    out = tmp_path / "symbols_out"
    main(
        [
            "symbols",
            str(out),
            "-n",
            "6",
            "--groups",
            "fingering",
            "ornament",
            "--size",
            "32",
            "--seed",
            "9",
        ]
    )
    assert (out / "dataset.npz").exists()
    assert (out / "labels.json").exists()
    meta = json.loads((out / "labels.json").read_text())
    assert len(meta["classes"]) == 10
    assert sum(meta["splits"].values()) == 60
    assert meta["splits"]["train"] > 0 and meta["splits"]["val"] > 0 and meta["splits"]["test"] > 0

    x_tr, _y_tr, classes = load_symbol_dataset(out, "train")
    x_va, _y_va, _ = load_symbol_dataset(out, "val")
    x_te, _y_te, _ = load_symbol_dataset(out, "test")
    assert len(x_tr) + len(x_va) + len(x_te) == 60
    assert x_tr.shape[1:] == (32, 32)
    assert len(classes) == 10
    assert SYMBOL_BY_NAME["fing1"].name in classes

    pngs = list((out / "train").rglob("*.png"))
    assert len(pngs) == len(x_tr)

    # --no-png mode
    out_npz = tmp_path / "npz_only"
    report = build_symbol_dataset(out_npz, samples_per_class=4, groups=("pedal",), write_png=False)
    assert report.counts == {"pedal": 4}
    assert (out_npz / "dataset.npz").exists()
    assert not (out_npz / "train").exists()
