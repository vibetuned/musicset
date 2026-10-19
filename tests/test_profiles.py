import json
import random
import shutil
from pathlib import Path

import pytest
import verovio
from conftest import FIXTURES

from musicset.pipeline import BuildConfig, Layout, build_score
from musicset.render.profiles import has_encoded_breaks, sample_render_options


def _limits() -> dict:
    opts = verovio.toolkit().getAvailableOptions()
    opts = json.loads(opts) if isinstance(opts, str) else opts
    return {k: v for group in opts["groups"].values() for k, v in group["options"].items()}


def test_sampled_options_stay_inside_verovio_limits():
    limits = _limits()
    for seed in range(40):
        opts = sample_render_options(random.Random(seed), encoded_breaks=True)
        for key, value in opts.extra.items():
            spec = limits[key]
            if "min" in spec and not isinstance(value, bool):
                assert spec["min"] <= value <= spec["max"], (key, value)
        assert opts.breaks in ("auto", "line")


def test_sampling_is_seeded_and_breaks_need_a_source_that_has_them():
    a = sample_render_options(random.Random(5))
    assert a == sample_render_options(random.Random(5))
    assert all(sample_render_options(random.Random(s)).breaks == "auto" for s in range(30))
    assert has_encoded_breaks('<print new-system="yes"/>') and not has_encoded_breaks("<print/>")


def test_varied_build_writes_distinct_variants(tmp_path: Path):
    (tmp_path / "src").mkdir()
    shutil.copy(FIXTURES / "tiny.musicxml", tmp_path / "src" / "tiny.musicxml")
    layout = Layout(tmp_path / "out")
    layout.create()
    cfg = BuildConfig(profile="varied", variants=2, seed=4)
    for v in (1, 2):
        assert build_score((tmp_path / "src" / "tiny.musicxml", layout, cfg, v)).status == "built"
    m1 = json.loads((layout.manifest / "tiny__v1.json").read_text())
    m2 = json.loads((layout.manifest / "tiny__v2.json").read_text())
    assert m1["render_profile"] == "varied" and m1["render_options"] != m2["render_options"]
    assert (layout.imgs / "tiny__v2_page1.png").exists()


def test_variants_need_the_varied_profile():
    with pytest.raises(ValueError, match="varied"):
        BuildConfig(variants=2)
