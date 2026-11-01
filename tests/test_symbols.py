import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from musicset.annotate.schema import Element, Glyph, PageAnnotation
from musicset.cli import main
from musicset.context import CONTEXT_DIM
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
    assert len(names) == len(set(names)) == 51
    assert (
        names[:33] == [s.name for s in SYMBOLS[:33]] and names[32] == "pedal"
    )  # old indices stable
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


def test_page_symbol_extraction_notes_and_timesigs_with_context():
    from musicset.symbols import page_symbol_boxes_with_context

    page = PageAnnotation(
        score="s",
        page=1,
        width=1000.0,
        height=400.0,
        pixel_width=1000.0,
        pixel_height=400.0,
        elements=[
            Element("sys", ("system-staff",), (0, 100, 1000, 180), synthetic=True),
            Element("m", ("measure",), (0, 80, 1000, 220)),
            # 12/8 stacked time signature (element 2)
            Element("ts", ("meterSig",), (40, 100, 90, 180)),
            # Standalone quarter note with an attached staccato (elements 3..6)
            Element("nq", ("note",), (150, 80, 175, 200), attrs={"dur": "4"}),
            Element(None, ("notehead",), (150, 130, 172, 150), parent=3),
            Element("st1", ("stem",), (168, 80, 172, 140), parent=3),
            Element("ar1", ("artic",), (155, 180, 168, 200), parent=3),
            # Standalone 8th note inferred from flag glyph without data-dur (elements 7..10)
            Element("n8", ("note",), (240, 80, 275, 150)),
            Element(None, ("notehead",), (240, 130, 262, 150), parent=7),
            Element("st2", ("stem",), (258, 80, 262, 140), parent=7),
            Element("fl2", ("flag",), (258, 80, 275, 120), parent=9),
            # Beamed note: must NOT be emitted as an isolated note (elements 11..14)
            Element("bm", ("beam",), (350, 80, 450, 150)),
            Element("nb", ("note",), (350, 80, 372, 150), parent=11, attrs={"dur": "8"}),
            Element(None, ("notehead",), (350, 130, 372, 150), parent=12),
            Element("st3", ("stem",), (368, 80, 372, 140), parent=12),
        ],
        glyphs=[
            Glyph("E081", (40, 100, 62, 138), element=2),
            Glyph("E082", (64, 100, 90, 138), element=2),
            Glyph("E088", (50, 142, 78, 180), element=2),
            Glyph("E0A4", (150, 130, 172, 150), element=4, owner="nq"),
            Glyph("E4A2", (155, 180, 168, 200), element=6, owner="ar1"),
            Glyph("E0A4", (240, 130, 262, 150), element=8, owner="n8"),
            Glyph("E240", (258, 80, 275, 120), element=10, owner="n8"),
            Glyph("E0A4", (350, 130, 372, 150), element=13, owner="nb"),
        ],
    )
    items = page_symbol_boxes_with_context(page, (1000, 400))
    assert [(cls, box) for cls, box, _ in items] == [
        ("noteQuarter", (150.0, 80.0, 172.0, 150.0)),  # notehead+stem only, excludes artic
        ("note8th", (240.0, 80.0, 275.0, 150.0)),  # notehead+stem+flag
        ("articStaccatoAbove", (155.0, 180.0, 168.0, 200.0)),
    ]
    assert all(ctx is not None for _, _, ctx in items)


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
                Element("sys", ("system-staff",), (0, 60, 200, 100)),  # staff space 10
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
    assert report.source_counts["test"] == {
        "font": 0,
        "stroke": 0,
        "crop": 1,
        "hand": 0,
        "emnist": 0,
    }
    with np.load(out / "dataset.npz") as data:
        for split in ("train", "val", "test"):
            assert data[f"s_{split}"].shape == data[f"y_{split}"].shape
            assert (data[f"w_{split}"] == -1).all()
            assert data[f"c_{split}"].shape == (len(data[f"y_{split}"]), CONTEXT_DIM)
            assert set(data[f"s_{split}"].tolist()) <= {SOURCE_CROP}
        # the surviving crop has a real context: centre 4.5 spaces above the middle line
        c = data["c_test"]
        assert c.shape[0] == 1 and c[0, 0] == 1.0 and c[0, 2] == 1.0  # above the staff
        assert abs(c[0, 4] - (-4.5 / 6)) < 1e-6
    assert load_symbol_sources(out, "test").tolist() == [SOURCE_CROP]
    meta = json.loads((out / "labels.json").read_text())
    assert meta["source_names"] == ["font", "stroke", "crop", "hand", "emnist"]
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


def test_barline_renders_as_a_bar():
    bar = render_symbol("barlineSingle", size=32, preset="clean", rng=1)
    cols = (bar < 128).any(axis=0).nonzero()[0]
    rows = (bar < 128).any(axis=1).nonzero()[0]
    assert rows.max() - rows.min() >= 14 and cols.max() - cols.min() <= 8


def test_homus_reader_rendering_and_writer_split(tmp_path: Path):
    import shutil

    from musicset.homus import HOMUS_CLASSES, load_homus, parse_homus, stroke_fraction
    from musicset.symbols import SOURCE_HAND, render_strokes_sample

    assert len(HOMUS_CLASSES) == 22 and set(HOMUS_CLASSES.values()) <= set(SYMBOL_BY_NAME)
    label, strokes = parse_homus("G-Clef\n10,10;12,40;14,90;\n\n30,20;40,20;\n")
    assert label == "G-Clef" and strokes == (
        ((10.0, 10.0), (12.0, 40.0), (14.0, 90.0)),
        ((30.0, 20.0), (40.0, 20.0)),
    )
    assert stroke_fraction(strokes) == 80 / 150
    assert stroke_fraction([[(0, 0), (3, 3)]]) == 0.08  # a dot stays small

    root = tmp_path / "HOMUS"
    for w in (1, 2, 3):
        d = root / str(w)
        d.mkdir(parents=True)
        # each writer draws a little differently, as real writers do
        (d / f"{w}-1.txt").write_text(
            f"G-Clef\n{10 + w},10;12,{40 + w};14,90;\n30,20;40,{20 + w};\n"
        )
        (d / f"{w}-2.txt").write_text(f"Dot\n{50 + w},50;{52 + w},{52 + w};\n")
        (d / f"{w}-3.txt").write_text("Not-A-Class\n1,1;2,2;\n")
    samples = load_homus(root)
    assert len(samples) == 6 and {s.writer for s in samples} == {1, 2, 3}
    assert {s.label for s in samples} == {"gClef", "augmentationDot"}
    zpath = shutil.make_archive(str(tmp_path / "homus"), "zip", tmp_path, "HOMUS")
    assert len(load_homus(Path(zpath))) == 6

    clean = render_strokes_sample(samples[0].strokes, size=32, fraction=0.7, width=0.085)
    again = render_strokes_sample(samples[0].strokes, size=32, fraction=0.7, width=0.085)
    assert clean.shape == (32, 32) and np.array_equal(clean, again) and int(clean[0, 0]) == 255
    rows = (clean < 128).any(axis=1).nonzero()[0]
    assert 18 <= rows.max() - rows.min() + 1 <= 27  # ~70% of the canvas plus pen caps
    dot = render_strokes_sample(samples[1].strokes, size=32, fraction=0.08, width=0.085)
    assert 0 < (dot < 128).sum() < 40
    augmented = render_strokes_sample(samples[0].strokes, size=32, preset="tablet", rng=3)
    assert augmented.shape == (32, 32)
    brushed = render_strokes_sample(samples[0].strokes, size=32, fraction=0.7, brush=True)
    assert brushed.shape == (32, 32) and not np.array_equal(brushed, clean)
    assert len(np.unique(brushed)) > len(np.unique(clean))  # soft edge adds grey levels

    out = tmp_path / "out"
    report = build_symbol_dataset(
        out,
        samples_per_class=0,
        homus=root,
        homus_copies=2,
        homus_brush=0.5,
        groups=("clef", "note"),
        split=(0.34, 0.33, 0.33),
        min_ink=1,
        write_png=False,
    )
    assert report.splits == {"train": 4, "val": 2, "test": 2}  # train writer rendered twice
    assert report.source_counts["test"] == {
        "font": 0,
        "stroke": 0,
        "crop": 0,
        "hand": 2,
        "emnist": 0,
    }
    meta = json.loads((out / "labels.json").read_text())
    assert meta["homus_writers"] == {"train": 1, "val": 1, "test": 1}
    assert meta["source_names"] == ["font", "stroke", "crop", "hand", "emnist"]
    with np.load(out / "dataset.npz") as data:
        assert set(data["s_test"].tolist()) == {SOURCE_HAND}
        assert set(data["w_test"].tolist()) <= {1, 2, 3} and len(set(data["w_train"].tolist())) == 1
    records = json.loads((out / "strokes_test.json").read_text())
    assert len(records) == 2 and {"writer", "label", "fraction", "strokes"} <= set(records[0])
    assert (out / "strokes_val.json").is_file() and not (out / "strokes_train.json").exists()


def test_context_vector_and_page_geometry():
    from musicset.context import (
        CONTEXT_DIM,
        PageGeometry,
        SymbolContext,
        context_from_record,
        context_vector,
    )

    assert context_vector(None).tolist() == [0.0] * CONTEXT_DIM
    page = PageAnnotation(
        score="s",
        page=1,
        width=1000.0,
        height=400.0,
        pixel_width=1000.0,
        pixel_height=400.0,
        elements=[
            Element("sys", ("system-staff",), (0, 100, 1000, 180)),  # staff space 20, middle 140
            Element("m", ("measure",), (0, 50, 500, 250)),
        ],
        glyphs=[Glyph("E0A4", (300, 130, 320, 150), owner="n")],  # notehead centred (310, 140)
    )
    geo = PageGeometry(page)
    ctx = geo.context((330, 100, 340, 110))  # 10x10 box centred (335, 105)
    assert ctx is not None
    assert abs(ctx.y_staff - (-1.75)) < 1e-9
    assert abs(ctx.x_measure - 0.67) < 1e-9
    assert abs(ctx.barline - 8.25) < 1e-9
    assert ctx.note_dx is not None and abs(ctx.note_dx - 1.25) < 1e-9
    assert ctx.note_dy is not None and abs(ctx.note_dy - (-1.75)) < 1e-9
    assert abs(ctx.width - 0.5) < 1e-9 and abs(ctx.height - 0.5) < 1e-9
    v = context_vector(ctx)
    assert v.shape == (CONTEXT_DIM,) and v.dtype == np.float32
    assert v[0] == 1.0 and v[1] == 1.0 and v[2] == 0.0 and v[3] == 0.0  # inside the staff lines
    assert abs(v[4] - (-1.75 / 6)) < 1e-6
    assert abs(v[5] - 0.34) < 1e-6 and abs(v[6] - 1.0) < 1e-6  # barline clipped at 8
    # the note is 1.25 spaces to the left: beside it, not in its column
    assert v[7] == 1.0 and (v[8], v[9], v[10], v[11]) == (0.0, 0.0, 0.0, 1.0)
    assert abs(v[12] - 1.25 / 8) < 1e-6 and abs(v[13] - (-1.75 / 8)) < 1e-6
    # a staccato-like mark straight above the notehead, outside the staff
    above = context_vector(SymbolContext(-3.0, 0.5, 4.0, 0.2, -2.5, 0.4, 0.4))
    assert (above[1], above[2], above[3]) == (0.0, 1.0, 0.0)
    assert (above[8], above[9], above[10], above[11]) == (1.0, 0.0, 0.0, 0.0)
    # far from any notehead and outside every measure
    far = geo.context((800, 300, 820, 320))
    assert far is not None and far.note_dx is None and far.x_measure == 0.5
    assert context_vector(far)[7] == 0.0 and context_vector(far)[3] == 1.0  # below the staff
    # a box whose own notehead must not count as its neighbour
    own = geo.context((300, 130, 320, 150))
    assert own is not None and own.note_dx is None
    # no staff on the page: unknown
    empty = PageAnnotation(
        score="e", page=1, width=10.0, height=10.0, pixel_width=10.0, pixel_height=10.0
    )
    assert PageGeometry(empty).context((1, 1, 2, 2)) is None
    # contributed record
    rec = {
        "context": {
            "y_staff": -2.0,
            "x_measure": 0.1,
            "barline": 1.0,
            "note_dx": None,
            "note_dy": None,
        }
    }
    got = context_from_record(rec, (0, 0, 30, 15), 10.0)
    assert got == SymbolContext(-2.0, 0.1, 1.0, None, None, 3.0, 1.5)
    assert context_from_record({}, (0, 0, 1, 1), 10.0) is None


def test_stroke_contributions_with_context(tmp_path: Path):
    from musicset.strokes import load_stroke_contributions

    f = tmp_path / "battuta.jsonl"
    f.write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                {
                    "label": "noteQuarter",
                    "writer": "alice",
                    "staff_space": 10.0,
                    "strokes": [[[0, 0, 1, 0.5], [5, 35, 20, 0.6]]],
                    "context": {
                        "y_staff": 1.0,
                        "x_measure": 0.3,
                        "barline": 2.0,
                        "note_dx": 2.5,
                        "note_dy": 0.0,
                    },
                },
                {
                    "label": "noteQuarter",
                    "writer": "bob",
                    "staff_space": 10.0,
                    "strokes": [[[0, 0], [4, 30]]],
                },
                {
                    "label": "not-a-class",
                    "writer": "bob",
                    "staff_space": 10.0,
                    "strokes": [[[0, 0], [1, 1]]],
                },
            )
        )
        + "\n"
    )
    samples = load_stroke_contributions(f, classes={"noteQuarter"})
    assert [s.writer for s in samples] == [1000, 1001]
    assert samples[0].context is not None and samples[0].context.note_dx == 2.5
    assert samples[0].fraction == 35 / (8.8 * 10)
    assert samples[1].context is None
    out = tmp_path / "out"
    report = build_symbol_dataset(
        out,
        samples_per_class=0,
        strokes=[f],
        groups=("note",),
        split=(0.5, 0.0, 0.5),
        min_ink=1,
        write_png=False,
    )
    assert report.source_counts["train"]["hand"] + report.source_counts["test"]["hand"] == 2
    with np.load(out / "dataset.npz") as data:
        c = np.concatenate([data["c_train"], data["c_test"]])
        w = np.concatenate([data["w_train"], data["w_test"]])
        assert set(w.tolist()) == {1000, 1001}
        assert c[w == 1000][0, 0] == 1.0 and c[w == 1000][0, 7] == 1.0  # alice's real context
        assert (c[w == 1001] == 0).all()  # bob's unknown (no crop pool to draw from)
    meta = json.loads((out / "labels.json").read_text())
    assert meta["context_features"][0] == "valid" and meta["strokes"] == [str(f)]


def _write_idx(path: Path, array: np.ndarray) -> None:
    import gzip
    import struct

    code = {np.uint8: 8}[array.dtype.type]
    header = (
        b"\x00\x00" + bytes([code, array.ndim]) + struct.pack(">" + "I" * array.ndim, *array.shape)
    )
    with gzip.open(path, "wb") as f:
        f.write(header + array.tobytes())


def test_emnist_reader_and_word_composition(tmp_path: Path):
    from musicset.emnist import (
        BYCLASS_CHARS,
        EMNIST_SYMBOLS,
        compose_word,
        ink_bbox,
        load_emnist,
        read_idx,
    )
    from musicset.symbols import SYMBOL_BY_NAME

    assert set(EMNIST_SYMBOLS) <= set(SYMBOL_BY_NAME)
    assert all(c in BYCLASS_CHARS for words in EMNIST_SYMBOLS.values() for w in words for c in w)

    # two fake EMNIST samples: a "1" and a "p", stored transposed and white-on-black
    one = np.zeros((28, 28), dtype=np.uint8)
    one[4:24, 13:16] = 255  # a vertical bar, upright
    pee = np.zeros((28, 28), dtype=np.uint8)
    pee[6:26, 8:11] = 255  # stem
    pee[6:16, 8:20] = 255  # bowl
    images = np.stack([one.T, pee.T])  # EMNIST layout
    labels = np.array([BYCLASS_CHARS.index("1"), BYCLASS_CHARS.index("p")], dtype=np.uint8)
    _write_idx(tmp_path / "emnist-byclass-train-images-idx3-ubyte.gz", images)
    _write_idx(tmp_path / "emnist-byclass-train-labels-idx1-ubyte.gz", labels)
    (tmp_path / "emnist-byclass-mapping.txt").write_text(
        "\n".join(f"{i} {ord(c)}" for i, c in enumerate(BYCLASS_CHARS))
    )
    assert (
        read_idx(tmp_path / "emnist-byclass-train-labels-idx1-ubyte.gz").tolist() == labels.tolist()
    )

    xs, chars = load_emnist(tmp_path, "train")
    assert xs.shape == (2, 28, 28) and xs.dtype == np.uint8 and chars.tolist() == ["1", "p"]
    assert np.array_equal(xs[0], 255 - one)  # upright again, black ink on white paper
    assert ink_bbox(xs[0]) == (13, 4, 16, 24)
    only_p, only_chars = load_emnist(tmp_path, "train", chars="p")
    assert only_chars.tolist() == ["p"] and len(only_p) == 1

    word = compose_word([xs[1], xs[1]], np.random.default_rng(0), gap=(0.1, 0.1), jitter=0.0)
    assert word.ndim == 2 and word.dtype == np.uint8 and int(word[0, 0]) in (0, 255)
    assert word.shape[0] == 20 and word.shape[1] >= 2 * 12 + 2  # two 12-wide p's plus a gap
    with pytest.raises(ValueError, match="unknown EMNIST split"):
        load_emnist(tmp_path, "valid")
    with pytest.raises(ValueError, match="at least one inked"):
        compose_word([np.full((28, 28), 255, dtype=np.uint8)], np.random.default_rng(0))


def test_build_dataset_with_emnist(tmp_path: Path):
    from musicset.emnist import BYCLASS_CHARS
    from musicset.symbols import SOURCE_EMNIST, render_ink_sample

    root = tmp_path / "emnist"
    root.mkdir()

    def glyph(char: str) -> np.ndarray:  # a distinct blob per character, EMNIST layout
        img = np.zeros((28, 28), dtype=np.uint8)
        k = BYCLASS_CHARS.index(char) % 10
        img[4 + k % 3 : 24, 6 + k % 4 : 10 + k % 4 + 8] = 255
        return img.T

    for split in ("train", "test"):
        chars = list("12345pmfsz") * 6
        _write_idx(
            root / f"emnist-byclass-{split}-images-idx3-ubyte.gz",
            np.stack([glyph(c) for c in chars]),
        )
        _write_idx(
            root / f"emnist-byclass-{split}-labels-idx1-ubyte.gz",
            np.array([BYCLASS_CHARS.index(c) for c in chars], dtype=np.uint8),
        )
    ink = render_ink_sample(255 - glyph("p").T, size=32, fraction=0.6)
    assert ink.shape == (32, 32) and (ink < 128).sum() > 20 and int(ink[0, 0]) == 255

    out = tmp_path / "out"
    report = build_symbol_dataset(
        out,
        samples_per_class=0,
        emnist=root,
        emnist_per_class=6,
        groups=("fingering", "dynamic"),
        split=(0.5, 0.25, 0.25),
        min_ink=1,
        write_png=False,
    )
    counts = report.source_counts
    assert counts["train"]["emnist"] > 0 and counts["val"]["emnist"] > 0
    assert counts["test"]["emnist"] > 0
    assert sum(counts[s]["emnist"] for s in counts) == sum(report.splits.values())
    with np.load(out / "dataset.npz") as data:
        assert set(data["s_train"].tolist()) == {SOURCE_EMNIST}
        assert data["y_train"].max() <= 16  # 5 fingerings + 12 dynamics
    meta = json.loads((out / "labels.json").read_text())
    assert meta["emnist"] == str(root) and meta["source_names"][-1] == "emnist"
