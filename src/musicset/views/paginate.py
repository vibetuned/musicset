"""Page view: the part of the score drawn on one page, as MusicXML.

A measure belongs to a page when any of its elements was drawn there (ids are
shared between the MusicXML and the engraving, see :mod:`musicset.ids`). The
first measure of each page receives the attributes in force at that point
(clefs, key, time, divisions), so every page reads as a standalone score.

Ported from gelato's paginate step, with one fix: the running attributes were
keyed by tag only, so on a two-staff part ``<clef number="2">`` evicted
``<clef number="1">`` and every later page lost staff 1's clef (likewise
numbered keys and staff details). State is now keyed by tag and staff number.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET

# Attribute children that apply per staff; without @number they mean staff 1
_PER_STAFF = frozenset({"clef", "staff-details", "measure-style", "transpose"})


def _state_key(el: ET.Element) -> tuple[str, str | None]:
    number = el.get("number")
    if number is None and el.tag in _PER_STAFF:
        number = "1"
    return el.tag, number


def slice_page(root: ET.Element, page_ids: set[str]) -> ET.Element:
    """Copy of a ``score-partwise`` tree keeping only the measures drawn on a page."""
    sliced = copy.deepcopy(root)
    for part in sliced.findall(".//part"):
        state: dict[tuple[str, str | None], ET.Element] = {}
        dropped = []
        first_kept = False
        for measure in part.findall("measure"):
            for attrs in measure.findall("attributes"):
                for child in attrs:
                    state[_state_key(child)] = copy.deepcopy(child)

            if any(el.get("id") in page_ids for el in measure.iter()):
                if not first_kept:
                    if state:
                        carried = ET.Element("attributes")
                        carried.extend(copy.deepcopy(list(state.values())))
                        measure.insert(0, carried)
                    first_kept = True
            else:
                dropped.append(measure)
        for measure in dropped:
            part.remove(measure)
    return sliced


def page_musicxml(xml: str, page_ids: set[str]) -> str:
    return ET.tostring(slice_page(ET.fromstring(xml), page_ids), encoding="unicode")
