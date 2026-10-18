"""Stable element ids for MusicXML.

Verovio carries MusicXML ``id`` attributes through to its SVG and MEI output,
so giving every element an id up front is what links a drawn glyph back to
the note, measure or direction it came from.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET


def id_prefix(stem: str) -> str:
    """Short per-score prefix: the alphanumerics among the last three characters.

    Matches the ids of kakigori/gelato datasets, except that a prefix not
    starting with a letter gets an ``m`` in front: XML ids must start with a
    letter, and stems like ``synthetic_score_000123`` produced ``123-1``.
    """
    prefix = "".join(c for c in stem[-3:] if c.isalnum())
    if not prefix[:1].isalpha():
        prefix = "m" + prefix
    return prefix


def inject_ids(xml: str, prefix: str) -> str:
    """Give every element without an ``id`` the id ``{prefix}-{n}``, in document order."""
    root = ET.fromstring(xml)
    counter = 1
    for el in root.iter():
        if "id" not in el.attrib:
            el.attrib["id"] = f"{prefix}-{counter}"
            counter += 1
    return ET.tostring(root, encoding="unicode")
