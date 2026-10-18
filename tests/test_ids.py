import xml.etree.ElementTree as ET

from musicset.ids import id_prefix, inject_ids


def test_prefix_uses_last_three_alphanumerics():
    assert id_prefix("QmbbGKtZ9G6DkWxvSeU516c1ktWiFJmEbHGmR3JFtLAPyC") == "PyC"
    assert id_prefix("piece_a-b") == "ab"


def test_prefix_always_starts_with_a_letter():
    assert id_prefix("synthetic_score_000123") == "m123"
    assert id_prefix("score_-_") == "m"


def test_inject_keeps_existing_ids_and_numbers_the_rest():
    xml = '<score-partwise><part id="P1"><measure><note/></measure></part></score-partwise>'
    root = ET.fromstring(inject_ids(xml, "abc"))
    ids = [el.get("id") for el in root.iter()]
    assert ids == ["abc-1", "P1", "abc-2", "abc-3"]
