"""
High-level species-counterpoint generation, after Herremans & Sorensen
(2012): compose a cantus firmus (whole notes, horizontal rules only),
then a counterpoint against it (horizontal + vertical rules) in the
requested species, each with its own VNS run.
"""

from dataclasses import dataclass

from musicset.synth.counterpoint.herremans12 import (
    DEFAULT_WEIGHTS_V,
    Key,
    horizontal_subscores,
    vertical_subscores,
)
from musicset.synth.counterpoint.notes import Species, as_notes, attacks
from musicset.synth.counterpoint.vns import CounterpointVNS

# Default vocal-ish registers: cantus in the C3-C5 octave pair, the
# counterpoint one octave up (crossing is discouraged by rule V7).
CF_RANGE = (48, 72)
CP_RANGE = (60, 84)


def _weights_v_for(species: Species, weights_v: dict[str, float] | None) -> dict[str, float] | None:
    """
    V18 (ending suspension) defaults to weight 0 because it cannot
    apply without ties; enable it for the tied species unless the
    caller supplied explicit weights.
    """
    if weights_v is not None:
        return weights_v
    if species in (Species.FOURTH, Species.FIFTH):
        return dict(DEFAULT_WEIGHTS_V, v18_ending_suspension=1.0)
    return None


@dataclass
class Piece:
    """A generated cantus firmus + counterpoint pair."""

    key: Key
    cf: list[int]
    cp: list  # pitches (first species) or Notes
    cf_cost: float
    cp_cost: float
    species: Species = Species.FIRST

    @property
    def cp_attack_pitches(self) -> list[int]:
        """The counterpoint's melodic line, one pitch per attack."""
        return [a.pitch for a in attacks(as_notes(self.cp))]

    def subscores(self) -> dict[str, float]:
        """Nonzero subscores of the counterpoint (its full objective)."""
        subs = horizontal_subscores(self.cp, self.key, "cp")
        subs.update(vertical_subscores(self.cf, self.cp, self.key))
        return {name: value for name, value in subs.items() if value > 0}


def generate_cantus_firmus(
    key: Key,
    measures: int = 16,
    pitch_range: tuple[int, int] = CF_RANGE,
    seed: int | None = None,
    **vns_options,
) -> tuple[list[int], float]:
    """Compose a cantus firmus (whole notes, horizontal rules only)."""
    solver = CounterpointVNS(key, seed=seed, **vns_options)
    return solver.solve(measures=measures, role="cf", pitch_range=pitch_range)


def generate_counterpoint(
    cf: list[int],
    key: Key,
    pitch_range: tuple[int, int] = CP_RANGE,
    seed: int | None = None,
    species: Species = Species.FIRST,
    weights_v: dict[str, float] | None = None,
    **vns_options,
) -> tuple[list, float]:
    """Compose a counterpoint of the given species above a cantus."""
    solver = CounterpointVNS(
        key, seed=seed, weights_v=_weights_v_for(species, weights_v), **vns_options
    )
    return solver.solve(cf=cf, role="cp", pitch_range=pitch_range, species=species)


def generate_piece(
    key: Key | None = None,
    measures: int = 16,
    cf_range: tuple[int, int] = CF_RANGE,
    cp_range: tuple[int, int] = CP_RANGE,
    seed: int | None = None,
    cf: list[int] | None = None,
    species: Species = Species.FIRST,
    **vns_options,
) -> Piece:
    """
    Compose a whole two-voice piece: the cantus firmus first (unless
    one is supplied), then the counterpoint against it.
    """
    key = key or Key("C", "major")
    species = Species(species)
    if cf is None:
        cf, cf_cost = generate_cantus_firmus(
            key, measures=measures, pitch_range=cf_range, seed=seed, **vns_options
        )
    else:
        cf = list(cf)
        cf_cost = sum(horizontal_subscores(cf, key, "cf").values())
    cp_seed = None if seed is None else seed + 1
    cp, cp_cost = generate_counterpoint(
        cf, key, pitch_range=cp_range, seed=cp_seed, species=species, **vns_options
    )
    return Piece(key=key, cf=cf, cp=cp, cf_cost=cf_cost, cp_cost=cp_cost, species=species)
