"""Reading and writing MusicXML sources (.mxl, .musicxml, .xml)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

SOURCE_SUFFIXES = (".mxl", ".musicxml", ".xml")
_CONTAINER = "META-INF/container.xml"


@dataclass(slots=True)
class MusicXML:
    """A MusicXML document plus what is needed to write it back as it came."""

    text: str
    name: str = "score.xml"
    """Member name inside the .mxl archive."""
    container: bytes | None = None
    """Original META-INF/container.xml, preserved on write."""


def find_sources(root: Path) -> list[Path]:
    """Every score file under ``root``, sorted for reproducible processing order."""
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in SOURCE_SUFFIXES and p.is_file())


def _rootfile(zf: zipfile.ZipFile) -> str | None:
    """The main score member: container.xml's rootfile, else the first XML member."""
    names = zf.namelist()
    if _CONTAINER in names:
        try:
            container = ET.fromstring(zf.read(_CONTAINER))
            for el in container.iter():
                if el.tag.endswith("rootfile") and el.get("full-path") in names:
                    return el.get("full-path")
        except ET.ParseError:
            pass
    return next(
        (n for n in names if n.endswith((".xml", ".musicxml")) and not n.startswith("META-INF")),
        None,
    )


def read_musicxml(path: Path) -> MusicXML:
    if path.suffix.lower() != ".mxl":
        return MusicXML(path.read_text(encoding="utf-8"), name=path.name)
    with zipfile.ZipFile(path) as zf:
        name = _rootfile(zf)
        if name is None:
            raise ValueError(f"{path.name}: no MusicXML document in archive")
        container = zf.read(_CONTAINER) if _CONTAINER in zf.namelist() else None
        return MusicXML(zf.read(name).decode("utf-8"), name=name, container=container)


def write_mxl(path: Path, score: MusicXML) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(score.name, score.text)
        if score.container is not None:
            zf.writestr(_CONTAINER, score.container)
