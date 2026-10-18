"""ABC view: MusicXML as ABC notation, through W.G. Vree's xml2abc (vendored, LGPL)."""

from __future__ import annotations

from musicset.vendor import xml2abc


class AbcError(ValueError):
    pass


def to_abc(xml: str) -> str:
    """ABC transcription of a MusicXML document, with xml2abc's default options."""
    abc, diagnostics = xml2abc.vertaal(xml)
    if not abc:
        raise AbcError(diagnostics.strip() or "xml2abc produced no output")
    return abc
