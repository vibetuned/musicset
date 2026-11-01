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


def test_render_symbol_with_source_and_sample_sources():
    from musicset.symbols import (
        SOURCE_FONT,
        SOURCE_STROKE,
        generate_symbol_samples_with_sources,
        render_symbol_with_source,
    )

    img, src = render_symbol_with_source("fing1", size=32, preset="tablet", rng=3)
    assert np.array_equal(img, render_symbol("fing1", size=32, preset="tablet", rng=3))
    assert src in (SOURCE_FONT, SOURCE_STROKE)
    _, src_clean = render_symbol_with_source("fing1", size=32, preset="clean", rng=3)
    assert src_clean == SOURCE_FONT

    xs, ys, _classes, sources = generate_symbol_samples_with_sources(
        20, groups=("fingering",), preset="tablet", seed=1
    )
    assert xs.shape == (100, 32, 32) and sources.shape == (100,) and sources.dtype == np.uint8
    assert set(sources.tolist()) == {SOURCE_FONT, SOURCE_STROKE}
    plain = generate_symbol_samples(20, groups=("fingering",), preset="tablet", seed=1)
    assert np.array_equal(plain[0], xs) and np.array_equal(plain[1], ys)


def test_group_split_keeps_each_score_in_one_split():
    from musicset.symbols import _group_split_indices

    rng = np.random.default_rng(0)
    groups = np.array([f"score{i // 7}" for i in range(140)])  # 20 scores x 7 crops
    idx = _group_split_indices(groups, (0.8, 0.1, 0.1), rng)
    assert sorted(np.concatenate(list(idx.values())).tolist()) == list(range(140))
    seen = {name: set(groups[i].tolist()) for name, i in idx.items()}
    assert not (seen["train"] & seen["val"])
    assert not (seen["train"] & seen["test"])
    assert not (seen["val"] & seen["test"])
    assert len(idx["test"]) >= 7 and len(idx["val"]) >= 7 and len(idx["train"]) >= 70


def test_build_dataset_drops_blank_and_cross_split_duplicate_crops(tmp_path: Path):
    from musicset.symbols import SOURCE_CROP, load_symbol_sources

    ds = tmp_path / "built"
    (ds / "pages").mkdir(parents=True)
    (ds / "imgs").mkdir(parents=True)
    img = np.full((200, 200), 255, dtype=np.uint8)
    img[25:45, 25:40] = 0  # the engraved fingering digit
    # Four scores, each engraving the identical fingering, plus a mordent box on blank paper
    for k in range(4):
        page = PageAnnotation(
            score=f"s{k}",
            page=1,
            width=200.0,
            height=200.0,
            pixel_width=200.0,
            pixel_height=200.0,
            elements=[
                Element("fg", ("fing",), (20, 20, 45, 50), text="2"),
                Element("mrd", ("mordent",), (120, 120, 160, 150)),
            ],
            glyphs=[],
        )
        page.save(ds / "pages" / f"s{k}_page1{PAGE_SUFFIX}")
        Image.fromarray(img, mode="L").save(ds / "imgs" / f"s{k}_page1.png")

    out = tmp_path / "out"
    report = build_symbol_dataset(
        out,
        samples_per_class=0,
        from_dataset=ds,
        groups=("fingering", "ornament"),
        split=(0.5, 0.25, 0.25),
        write_png=False,
    )
    assert report.removed == {"blank": 4, "duplicate": 3}
    assert sum(report.splits.values()) == 1
    assert report.source_counts["test"] == {"font": 0, "stroke": 0, "crop": 1}
    with np.load(out / "dataset.npz") as data:
        for split in ("train", "val", "test"):
            assert data[f"s_{split}"].shape == data[f"y_{split}"].shape
            assert set(data[f"s_{split}"].tolist()) <= {SOURCE_CROP}
    assert load_symbol_sources(out, "test").tolist() == [SOURCE_CROP]
    meta = json.loads((out / "labels.json").read_text())
    assert meta["source_names"] == ["font", "stroke", "crop"]
    assert meta["removed"] == {"blank": 4, "duplicate": 3}
    assert meta["from_dataset"] == [str(ds)]
    assert meta["crop_scores"] == {"train": 2, "val": 1, "test": 1}

    # --no-dedup keeps all four identical crops, one score per split at most
    kept = build_symbol_dataset(
        tmp_path / "out2",
        samples_per_class=0,
        from_dataset=ds,
        groups=("fingering",),
        split=(0.5, 0.25, 0.25),
        dedup=False,
        write_png=False,
    )
    assert kept.removed == {"blank": 0, "duplicate": 0} and sum(kept.splits.values()) == 4
