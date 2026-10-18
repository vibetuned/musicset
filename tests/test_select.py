from pathlib import Path

from musicset.select import Selection, lite_ok, place, score_stats

STAVES_2 = "<attributes><staves>2</staves></attributes>"


def _score(parts: list[str], part_list: int | None = None) -> str:
    n = part_list if part_list is not None else len(parts)
    plist = "".join(
        f'<score-part id="P{i}"><part-name>x</part-name></score-part>' for i in range(n)
    )
    body = "".join(f'<part id="P{i}">{p}</part>' for i, p in enumerate(parts))
    return f"<score-partwise><part-list>{plist}</part-list>{body}</score-partwise>"


def _note(staff: str = "1", voice: str = "1", alter: str = "") -> str:
    pitch = f"<step>C</step>{f'<alter>{alter}</alter>' if alter else ''}<octave>4</octave>"
    tail = f"<duration>1</duration><voice>{voice}</voice><staff>{staff}</staff>"
    return f"<note><pitch>{pitch}</pitch>{tail}</note>"


def _write(tmp: Path, name: str, xml: str) -> Path:
    path = tmp / f"{name}.musicxml"
    path.write_text(xml)
    return path


def test_stats_and_lite_gates(tmp_path: Path):
    single = _write(tmp_path, "single", _score([f"<measure>{_note()}{_note()}</measure>"]))
    grand = _write(
        tmp_path,
        "grand",
        _score([f"<measure>{STAVES_2}{_note('1')}{_note('2', '2')}</measure>"]),
    )
    duet = _write(tmp_path, "duet", _score([f"<measure>{_note()}</measure>"] * 2))
    two_voices = _write(
        tmp_path, "voices", _score([f"<measure>{_note(voice='1')}{_note(voice='2')}</measure>"])
    )
    gestural = _write(tmp_path, "gestural", _score([f"<measure>{_note(alter='1') * 11}</measure>"]))

    stats = {p.stem: score_stats(p) for p in (single, grand, duet, two_voices, gestural)}
    assert {k: s.voices for k, s in stats.items()} == {
        "single": 1,
        "grand": 2,
        "duet": 2,
        "voices": 2,
        "gestural": 1,
    }
    assert {k for k, s in stats.items() if lite_ok(s)} == {"single", "grand"}
    assert stats["gestural"].gestural_accidentals


def test_unreadable_scores_are_never_chosen(tmp_path: Path):
    broken = score_stats(_write(tmp_path, "broken", "<score-partwise>"))
    assert broken.error is not None
    assert Selection().choose([broken]) == []


def test_choose_is_reproducible_capped_and_excludes(tmp_path: Path):
    one = [
        score_stats(_write(tmp_path, f"a{i}", _score([f"<measure>{_note()}</measure>"])))
        for i in range(6)
    ]
    two = [
        score_stats(_write(tmp_path, f"b{i}", _score([f"<measure>{_note()}</measure>"] * 2)))
        for i in range(3)
    ]
    sel = Selection(per_voice=(2, 1), exclude=frozenset({"b0"}), seed=1)
    chosen = sel.choose(one + two)
    assert sorted(s.voices for s in chosen) == [1, 1, 2]
    assert "b0" not in {s.path.stem for s in chosen}
    assert [s.path for s in chosen] == [s.path for s in sel.choose(list(reversed(one + two)))]
    assert len(Selection(limit=4).choose(one + two)) == 4
    assert all(s.voices == 1 for s in Selection(max_voices=1).choose(one + two))


def test_place_links_and_resolves_name_clashes(tmp_path: Path):
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    a = score_stats(_write(tmp_path / "x", "same", _score([f"<measure>{_note()}</measure>"])))
    b = score_stats(_write(tmp_path / "y", "same", _score([f"<measure>{_note()}</measure>"])))
    placed = place([a, b], tmp_path / "out")
    assert [p.name for p in placed] == ["same.musicxml", "same_y.musicxml"]
    assert all(p.is_symlink() for p in placed)
    assert place([a], tmp_path / "out") == [tmp_path / "out" / "same.musicxml"]
