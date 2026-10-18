"""
Variable Neighborhood Search counterpoint composer, after Herremans &
Sorensen (2012, Section 4 and Figure 5). It shares its skeleton with
the fingering VNS of chirality (random feasible start, swap / change1 /
change2, perturbation); this solver mutates NOTE PITCHES and follows the
counterpoint paper where the two algorithms differ:

  - steepest descent inside each neighborhood (the fingering VNS uses
    first descent, after Balliauw et al.);
  - a fixed neighborhood cycle swap -> change1 -> change2, restarted
    from swap whenever a cycle improved the current fragment;
  - one tabu list per neighborhood over recently changed note
    positions (tenure a fraction of the piece length);
  - adaptive weight adjustment: whenever the search is perturbed, the
    weight of the worst-valued subscore is increased by one; local
    search selects moves on the adapted score f^a while the best
    fragment is tracked on the original objective f;
  - perturbation restarts from the best fragment with a fraction r of
    the notes re-randomized;
  - stop at f(s_best) = 0 or after `maxiters` successive perturbations
    without improving s_best.

Defaults follow the paper's optimal parameter settings (Table 6): all
three neighborhoods active, tabu tenure 1/16, r = 1/8, adaptive
weights on, maxiters 50.
"""

from collections.abc import Sequence

import numpy as np

from musicset.synth.counterpoint.herremans12 import (
    DEFAULT_WEIGHTS_H,
    DEFAULT_WEIGHTS_V,
    HORIZONTAL_RULES,
    VERTICAL_RULES,
    Key,
    horizontal_subscores,
    vertical_subscores,
)
from musicset.synth.counterpoint.notes import (
    TIE_END,
    Note,
    Species,
    sample_skeleton,
)


class CounterpointVNS:
    """Species counterpoint composer over a key's pitch material.

    With `cf` given, optimizes a counterpoint against the fixed cantus
    (horizontal + vertical aspects of Eq. 1); without it, optimizes a
    standalone line (horizontal only) - i.e. a cantus firmus when
    role="cf". The rhythm comes from a per-species feasible skeleton
    (notes.sample_skeleton) fixed for the run; the search itself is a
    pitch-vector problem with one variable per attack (tie chains keep
    a single pitch by construction).
    """

    def __init__(
        self,
        key: Key,
        weights_h: dict[str, float] | None = None,
        weights_v: dict[str, float] | None = None,
        maxiters: int = 50,
        perturb_fraction: float = 1 / 8,
        tabu_fraction: float = 1 / 16,
        adaptive_weights: bool = True,
        seed: int | None = None,
    ):
        self.key = key
        self.weights_h = dict(weights_h or DEFAULT_WEIGHTS_H)
        self.weights_v = dict(weights_v or DEFAULT_WEIGHTS_V)
        self.maxiters = maxiters
        self.perturb_fraction = perturb_fraction
        self.tabu_fraction = tabu_fraction
        self.adaptive_weights = adaptive_weights
        self.rng = np.random.default_rng(seed)
        self._static = None

    # ---- objective ---------------------------------------------------
    # The search state is always a plain pitch vector, one pitch per
    # ATTACK; `_realize` places it on the rhythmic skeleton (identity
    # for first species) before the subscores see it.

    def _realize(self, pitches: Sequence[int]):
        if self._static is None:
            return pitches
        return [
            Note(pitches[k], duration, measure, beat, tied)
            for k, duration, measure, beat, tied in self._static
        ]

    def _subscores(self, pitches: Sequence[int]) -> dict[str, float]:
        voice = self._realize(pitches)
        subs = horizontal_subscores(voice, self.key, self._role)
        if self._cf is not None:
            subs.update(vertical_subscores(self._cf, voice, self.key))
        return subs

    def _values(self, pitches: Sequence[int]) -> list[float]:
        """Subscore values in `self._rule_names` order (the hot path)."""
        voice = self._realize(pitches)
        key, role, cf = self.key, self._role, self._cf
        values = [fn(voice, key, role) for fn in HORIZONTAL_RULES.values()]
        if cf is not None:
            values += [fn(cf, voice, key) for fn in VERTICAL_RULES.values()]
        return values

    def _cost(self, pitches: Sequence[int], adapted: bool = False) -> float:
        weights = self._adapted_flat if adapted else self._weights_flat
        return sum(w * v for w, v in zip(weights, self._values(pitches), strict=False) if v)

    # ---- neighborhoods (steepest descent each) -----------------------

    def _descend(self, voice: list[int], moves, tabu) -> tuple[list[int], bool]:
        """Repeat steepest-descent steps in one neighborhood until stuck."""
        improved_any = False
        while True:
            cost = self._cost(voice, adapted=True)
            best_move, best_cost = None, cost
            for positions, candidate in moves(voice):
                if any(p in tabu for p in positions):
                    continue
                c = self._cost(candidate, adapted=True)
                if c < best_cost:
                    best_move, best_cost = (positions, candidate), c
            if best_move is None:
                return voice, improved_any
            positions, voice = best_move
            improved_any = True
            for p in positions:
                self._push_tabu(tabu, p)
            self._track_best(voice)

    def _swap_moves(self, voice):
        n = len(voice)
        for i in range(n):
            for j in range(i + 1, n):
                if voice[i] == voice[j]:
                    continue
                candidate = list(voice)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                yield (i, j), candidate

    def _change1_moves(self, voice):
        for i in range(len(voice)):
            for p in self._domain:
                if p == voice[i]:
                    continue
                candidate = list(voice)
                candidate[i] = p
                yield (i,), candidate

    def _change2_moves(self, voice):
        for i in range(len(voice) - 1):
            for p1 in self._domain:
                for p2 in self._domain:
                    if p1 == voice[i] and p2 == voice[i + 1]:
                        continue
                    candidate = list(voice)
                    candidate[i], candidate[i + 1] = p1, p2
                    yield (i, i + 1), candidate

    # ---- tabu / adaptive weights / perturbation ----------------------

    def _push_tabu(self, tabu: list[int], position: int) -> None:
        tabu.append(position)
        while len(tabu) > self._tenure:
            tabu.pop(0)

    def _bump_worst_subscore(self, pitches: list[int]) -> None:
        """Increase the adapted weight of the worst subscore by one."""
        values = self._values(pitches)
        worst = max(range(len(values)), key=values.__getitem__)
        if values[worst] > 0.0:
            self._adapted_flat[worst] += 1.0

    def _perturb(self, voice: list[int]) -> list[int]:
        candidate = list(voice)
        count = max(1, round(len(voice) * self.perturb_fraction))
        for i in self.rng.choice(len(voice), size=count, replace=False):
            candidate[int(i)] = int(self.rng.choice(self._domain))
        return candidate

    def _track_best(self, voice: list[int]) -> None:
        cost = self._cost(voice, adapted=False)
        if cost < self._best_cost:
            self._best, self._best_cost = list(voice), cost
            self._stale = False

    # ---- main loop (paper Figure 5) ----------------------------------

    def solve(
        self,
        measures: int | None = None,
        cf: Sequence[int] | None = None,
        role: str = "cp",
        pitch_range: tuple[int, int] = (60, 84),
        initial: Sequence[int] | None = None,
        species: Species = Species.FIRST,
        skeleton: list[Note] | None = None,
        restarts: int = 1,
    ) -> tuple[list, float]:
        """
        Compose one voice; returns (voice, cost). The voice is a plain
        pitch list for first species and a list of Notes otherwise.

        measures defaults to len(cf). The pitch domain is the key's
        scale material inside `pitch_range` (inclusive). The rhythmic
        skeleton is sampled per species (Table 1 feasibility) at the
        start of each run and stays fixed - the paper's moves mutate
        pitches only; `initial` is a per-attack pitch vector for the
        first run. `restarts` runs the whole VNS that many times
        (re-sampling the skeleton unless one was given) and keeps the
        best piece; 1 is the paper's behavior.
        """
        self._cf = list(cf) if cf is not None else None
        self._role = role
        if measures is None:
            if self._cf is None:
                raise ValueError("measures is required without a cantus firmus")
            measures = len(self._cf)
        if self._cf is not None and len(self._cf) != measures:
            raise ValueError("measures must match the cantus firmus length")
        self._domain = self.key.scale_pitches(*pitch_range)
        if not self._domain:
            raise ValueError(f"No {self.key} pitches inside {pitch_range}")
        if skeleton is not None and skeleton[-1].measure + 1 != measures:
            raise ValueError("skeleton must span exactly `measures` measures")

        overall_voice, overall_cost = None, float("inf")
        for run in range(max(1, restarts)):
            if skeleton is not None:
                the_skeleton = list(skeleton)
            elif species == Species.FIRST:
                the_skeleton = None  # plain pitch vector, one per measure
            else:
                the_skeleton = sample_skeleton(species, measures, self.rng)
            if the_skeleton is None:
                self._static = None
                n = measures
            else:
                # (attack index, duration, measure, beat, tied) per note:
                # the fixed rhythm; _realize only fills in the pitches.
                static = []
                k = -1
                for note in the_skeleton:
                    if note.tied != TIE_END:
                        k += 1
                    static.append((k, note.duration, note.measure, note.beat, note.tied))
                self._static = static
                n = k + 1
            self._run(n, initial if run == 0 else None)
            if self._best_cost < overall_cost:
                overall_voice = self._realize(self._best)
                overall_cost = self._best_cost
            if overall_cost == 0:
                break
        assert overall_voice is not None
        return list(overall_voice), overall_cost

    def _run(self, n_attacks: int, initial: Sequence[int] | None) -> None:
        """One full VNS run over a fixed skeleton (paper Figure 5)."""
        self._rule_names = list(HORIZONTAL_RULES)
        self._weights_flat = [self.weights_h.get(n, 1.0) for n in HORIZONTAL_RULES]
        if self._cf is not None:
            self._rule_names += list(VERTICAL_RULES)
            self._weights_flat += [self.weights_v.get(n, 1.0) for n in VERTICAL_RULES]
        self._adapted_flat = list(self._weights_flat)
        self._tenure = max(1, round(n_attacks * self.tabu_fraction))
        tabus = {"swap": [], "change1": [], "change2": []}
        neighborhoods = (
            ("swap", self._swap_moves),
            ("change1", self._change1_moves),
            ("change2", self._change2_moves),
        )

        if initial is not None:
            current = list(initial)
        else:
            current = [int(p) for p in self.rng.choice(self._domain, size=n_attacks)]
        self._best, self._best_cost = list(current), self._cost(current)
        self._stale = True

        iters = 0
        while self._best_cost > 0 and iters < self.maxiters:
            cycled = True
            while cycled:
                cycled = False
                for name, moves in neighborhoods:
                    current, improved = self._descend(current, moves, tabus[name])
                    cycled = cycled or improved

            if not self._stale:  # s_best improved since last check
                self._stale = True
                iters = 0
            else:
                iters += 1

            if self._best_cost == 0 or iters >= self.maxiters:
                break
            if self.adaptive_weights:
                self._bump_worst_subscore(current)
            current = self._perturb(self._best)
