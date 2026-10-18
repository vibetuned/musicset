"""
Herremans & Sorensen - "Composing Fifth Species Counterpoint Music With
Variable Neighborhood Search" (University of Antwerp working paper
2012-020), the expanded version
of their first-species paper (Journal of Mathematics and the Arts, 2012).

The paper quantifies the species counterpoint rules of Salzer &
Schachter (1969) into 19 *horizontal* (melodic) and 19 *vertical*
(harmonic) subscores, each in [0, 1] with 0 = perfect conformity
(appendix Eqs. 2-39, Tables 7-8). The objective for a fragment s is
Eq. 1:

    f(s) = sum_i a_i * subscore_H_i(s) + sum_j b_j * subscore_V_j(s)

minimized by a VNS (see counterpoint/vns.py). The cantus firmus (always
whole notes) is scored on the horizontal aspect only; the counterpoint
on horizontal + vertical against the cantus.

Voices are lists of `notes.Note` (pitch/duration/measure/beat/tied);
plain MIDI-pitch sequences are accepted everywhere and normalized to
whole notes (first species). Horizontal rules and melodic notions run
on the ATTACK sequence (tie chains collapsed to one melodic note);
vertical rules classify each counterpoint note by duration against the
cantus pitch of its measure. The species templates and the feasibility
rules of the paper's Table 1 live in `notes.py`.

Deviations / interpretation notes (the paper has no reproduction
tables; formulas are implemented literally from the appendix with these
choices, in the spirit of balliauw17's deviations section):
  - Eq. 6 (H5, leap count) is implemented exactly as printed, including
    its odd third branch: with fewer than 2L leaps the penalty is
    #leaps / (length-1-4L), so an all-stepwise melody scores 0 and the
    penalty *rises* toward the 2L boundary before dropping to 0 inside
    [2L, 4L]. Likely an erratum for (2L - #leaps), but we reproduce the
    paper.
  - The formulas are stated for L x 16 measures; L = measures / 16 and
    "length" = number of attacks (identical in first species).
  - Eq. 38 (V18) as printed awards 1 (= worst) to the ending Table 8
    calls best - an evident erratum; we implement the corrected
    polarity (0 = dissonant suspension into the leading tone, 0.5 = any
    suspension, 1 = otherwise) for rhythmic pieces and return 0 for
    all-whole-note pieces, where no suspension can exist. The default
    weight stays 0; generate.py enables it for the tied species (4, 5).
  - H2's neighboring-tone clause ("the two repeated notes must always
    be vertically consonant") needs the other voice; the horizontal
    subscore checks the melodic pattern only (approached and left by
    step), and V1 already penalizes the dissonant case.
  - H3: strong beats are 1 and 3 of the 4/4 measure; a whole-note voice
    scores 0 by construction.
  - H14 "preferably in the same octave as the ending note" is read as
    the penultimate lying a step from the final (leading tone below /
    supertonic above); otherwise the degree earns 0.5.
  - H17 counts every occurrence of a pitch that appears more than 2L
    times (Table 7's tie limit reread for repeated whole notes).
  - H18 marks notes belonging to a repeated pitch bigram whose two
    occurrences start within a 16-attack window (Table 7's "16 note
    interval"; any repeated sequence of length >= 2 contains a repeated
    bigram).
  - V2/V3 dissonance exceptions: a suspension is a tie-end on beat 1
    resolving down by step to the next attack; a passing tone is
    approached and left by step in the same direction; a neighbor tone
    is approached and left by step returning to the same pitch. Weak
    beats for halves: beat 3; for quarters: beats 2-4.
  - V5: whole-note unisons are allowed on the first and last measure
    only (first species has no suspensions); shorter unisons follow
    Table 8 rule 5: allowed on beat 1 when tied over or left by step.
  - V12 as printed counts opposite-direction simultaneous leaps of
    any size but same-direction leaps only when large (> 6 semitones in
    either voice); implemented literally.
  - V15: successive same-class perfect first-beat intervals count
    unless separated by at least three quarter notes (Table 8 rule 15);
    in first species no separation is possible.
  - V16: successive same-class perfect off-beat intervals count unless
    separated by at least two notes; the paper's afterbeat consonant-
    suspension exception (rule 16, second sentence) is omitted.
  - V17 (battuta) requires contrary motion with both voices leaping
    into a unison/octave or fifth.
  - Motion classification: oblique when either voice holds its pitch,
    parallel for the same signed interval, similar for the same
    direction otherwise, contrary for opposite directions.
  - Minor mode pitch material is the natural minor plus the raised
    leading tone (Table 7 rule 14 requires the raised form at the
    cadence).
"""

import itertools
from collections import Counter

from musicset.synth.counterpoint.notes import (
    EIGHTH,
    HALF,
    QUARTER,
    TIE_END,
    WHOLE,
    Note,
    Voice,
    as_notes,
    attacks,
    n_measures,
)

# =============================================================================
# KEYS AND SCALES
# =============================================================================

PITCH_CLASSES = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}

_MAJOR_DEGREES = (0, 2, 4, 5, 7, 9, 11)
# Natural minor plus the raised leading tone (Table 7 rule 14: "In the
# case of a minor scale it should be raised by half a tone").
_MINOR_DEGREES = (0, 2, 3, 5, 7, 8, 10, 11)


class Key:
    """A tonic pitch class + mode, with the pitch material of the key."""

    def __init__(self, tonic: str = "C", mode: str = "major"):
        if tonic not in PITCH_CLASSES:
            raise ValueError(f"Unknown tonic '{tonic}'. Known: {sorted(PITCH_CLASSES)}")
        if mode not in ("major", "minor"):
            raise ValueError(f"Unknown mode '{mode}' (major or minor)")
        self.tonic_name = tonic
        self.mode = mode
        self.tonic = PITCH_CLASSES[tonic]
        degrees = _MAJOR_DEGREES if mode == "major" else _MINOR_DEGREES
        self.pitch_classes = frozenset((self.tonic + d) % 12 for d in degrees)

    @property
    def leading_tone(self) -> int:
        """Pitch class a semitone below the tonic (raised in minor)."""
        return (self.tonic - 1) % 12

    @property
    def supertonic(self) -> int:
        return (self.tonic + 2) % 12

    def scale_pitches(self, low: int, high: int) -> list[int]:
        """All key pitches in the inclusive MIDI range [low, high]."""
        return [p for p in range(low, high + 1) if p % 12 in self.pitch_classes]

    def __contains__(self, pitch: int) -> bool:
        return pitch % 12 in self.pitch_classes

    def __repr__(self) -> str:
        return f"Key({self.tonic_name} {self.mode})"


# =============================================================================
# INTERVAL VOCABULARY
# =============================================================================

# Consonant *horizontal* intervals (Table 7 rule 4): 0, 1, 2, 3, 4, 5,
# 7, 8, 9 and 12 semitones "(or more octaves)".
_MELODIC_CONSONANT = frozenset((0, 1, 2, 3, 4, 5, 7, 8, 9, 12))

# Consonant *vertical* interval classes: unison/octave, thirds, fifth,
# sixths (P4 is dissonant in two-voice counterpoint).
_VERTICAL_CONSONANT_CLASSES = frozenset((0, 3, 4, 7, 8, 9))

# Perfect interval classes (Table 8 rule 11: 0, 5, 7 and 12 semitones).
_PERFECT_CLASSES = frozenset((0, 5, 7))

# Thirds, sixths (and their octave compounds: tenths etc.) for V19.
_IMPERFECT_CONSONANT_CLASSES = frozenset((3, 4, 8, 9))


def is_step(interval: int) -> bool:
    """Stepwise motion is an interval of one or two semitones (Table 7 rule 5)."""
    return 1 <= abs(interval) <= 2


def is_leap(interval: int) -> bool:
    """A leap is an interval of more than two semitones (Table 7 rule 6)."""
    return abs(interval) > 2


def is_large_leap(interval: int) -> bool:
    """A large leap is an interval of more than 4 semitones (Table 7 rule 6)."""
    return abs(interval) > 4


def is_melodic_consonant(interval: int) -> bool:
    """Table 7 rule 4's set; above the octave only whole octaves qualify."""
    iv = abs(interval)
    if iv <= 12:
        return iv in _MELODIC_CONSONANT
    return iv % 12 == 0


def is_vertical_consonant(interval: int) -> bool:
    return abs(interval) % 12 in _VERTICAL_CONSONANT_CLASSES


def is_perfect(interval: int) -> bool:
    return abs(interval) % 12 in _PERFECT_CLASSES


def intervals(pitches: list[int]) -> list[int]:
    """Signed melodic intervals between successive attack pitches."""
    return [b - a for a, b in itertools.pairwise(pitches)]


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _memo1(builder):
    """
    Cache the last result by argument identity: within one aggregate
    call every subscore receives the very same voice objects, so the
    expensive normalization runs once. Callers must not mutate a voice
    in place between calls (the VNS builds fresh candidates).
    """
    last_key = last_value = None

    def get(arg):
        nonlocal last_key, last_value
        if arg is last_key:
            return last_value
        value = builder(arg)  # build first: never cache a raiser
        last_key, last_value = arg, value
        return value

    return get


def _memo2(builder):
    """Two-argument variant of _memo1."""
    key_a = key_b = last_value = None

    def get(a, b):
        nonlocal key_a, key_b, last_value
        if a is key_a and b is key_b:
            return last_value
        value = builder(a, b)
        key_a, key_b, last_value = a, b, value
        return value

    return get


class _Melody:
    """Normalized horizontal context of one voice."""

    __slots__ = ("L", "atk", "measures", "notes", "pitches")

    def __init__(self, voice: Voice):
        self.notes = as_notes(voice)
        self.atk = attacks(self.notes)
        self.pitches = [a.pitch for a in self.atk]
        self.measures = n_measures(self.notes)
        self.L = self.measures / 16.0


_melody = _memo1(_Melody)


def _pitches(voice: Voice) -> list[int]:
    """The melodic (attack) pitch sequence of a voice."""
    return _melody(voice).pitches


def climax_index(pitches: list[int]) -> int:
    """Index of the (first occurrence of the) highest attack pitch."""
    return max(range(len(pitches)), key=lambda i: (pitches[i], -i))


def motion_intervals(voice: Voice) -> list[int]:
    """
    Interval between the start and end note of every maximal upward or
    downward movement (Table 7 rule 15). A repeated pitch ends the
    current movement.
    """
    pitches = _pitches(voice)
    motions = []
    start = 0
    direction = 0
    for i, iv in enumerate(intervals(pitches)):
        s = _sign(iv)
        if s == 0:
            if direction != 0:
                motions.append(pitches[i] - pitches[start])
            start, direction = i + 1, 0
        elif direction == 0:
            start, direction = i, s
        elif s != direction:
            motions.append(pitches[i] - pitches[start])
            start, direction = i, s
    if direction != 0:
        motions.append(pitches[-1] - pitches[start])
    return motions


def direction_changes(voice: Voice) -> int:
    """Number of times the melodic direction flips (repeats ignored)."""
    changes = 0
    last = 0
    for iv in intervals(_pitches(voice)):
        s = _sign(iv)
        if s == 0:
            continue
        if last != 0 and s != last:
            changes += 1
        last = s
    return changes


# Motion types between two voices (V11, V13, V17).
PARALLEL, SIMILAR, OBLIQUE, CONTRARY = "parallel", "similar", "oblique", "contrary"


def motion_type(d_lower: int, d_upper: int) -> str:
    if d_lower == 0 or d_upper == 0:
        return OBLIQUE
    if _sign(d_lower) != _sign(d_upper):
        return CONTRARY
    return PARALLEL if d_lower == d_upper else SIMILAR


def _neighbor_positions(pitches: list[int]) -> set:
    """
    Attack indices taking part in a neighboring-tone figure: a note
    positioned between two repeated notes, approached and left by
    stepwise motion (Table 7 rule 2).
    """
    hits = set()
    for i in range(len(pitches) - 2):
        if pitches[i] == pitches[i + 2] and is_step(pitches[i + 1] - pitches[i]):
            hits.update((i, i + 1, i + 2))
    return hits


def _clamp(x: float) -> float:
    return min(1.0, max(0.0, x))


# =============================================================================
# HORIZONTAL SUBSCORES (appendix Eqs. 2-20, Table 7)
# Every function takes (voice, key, role) -> float in [0, 1]; role is
# "cf" or "cp" and only affects H14. L = measures / 16, length = the
# number of attacks (the formulas are stated for L x 16 measures).
# =============================================================================


def h1_eighths_stepwise(voice: Voice, key: Key, role: str = "cp") -> float:
    """Eq. 2: (#8ths not preceded by step + #8ths not left by step) / (#8ths x 2)."""
    m = _melody(voice)
    atk, pitches = m.atk, m.pitches
    eighth_idx = [i for i, a in enumerate(atk) if a.duration == EIGHTH]
    if not eighth_idx:
        return 0.0
    bad = 0
    for i in eighth_idx:
        if i == 0 or not is_step(pitches[i] - pitches[i - 1]):
            bad += 1
        if i == len(pitches) - 1 or not is_step(pitches[i + 1] - pitches[i]):
            bad += 1
    return _clamp(bad / (2 * len(eighth_idx)))


def h2_one_climax(voice: Voice, key: Key, role: str = "cp") -> float:
    """Eq. 3: (#highest note not as neighboring tone - 1) / length."""
    pitches = _pitches(voice)
    if not pitches:
        return 0.0
    top = max(pitches)
    neighbors = _neighbor_positions(pitches)
    count = sum(1 for i, p in enumerate(pitches) if p == top and i not in neighbors)
    return _clamp((count - 1) / len(pitches))


def h3_climax_strong_beat(voice: Voice, key: Key, role: str = "cp") -> float:
    """Eq. 4: #highest notes on weak beats / #highest notes (neighbors excluded)."""
    m = _melody(voice)
    atk, pitches = m.atk, m.pitches
    if not atk:
        return 0.0
    top = max(pitches)
    neighbors = _neighbor_positions(pitches)
    candidates = [i for i, p in enumerate(pitches) if p == top and i not in neighbors]
    if not candidates:
        return 0.0
    weak = sum(1 for i in candidates if atk[i].beat not in (1.0, 3.0))
    return _clamp(weak / len(candidates))


def h4_consonant_intervals(voice: Voice, key: Key, role: str = "cp") -> float:
    """Eq. 5: 1 - #consonant horizontal intervals / #intervals."""
    ivs = intervals(_pitches(voice))
    if not ivs:
        return 0.0
    return _clamp(1.0 - sum(map(is_melodic_consonant, ivs)) / len(ivs))


def h5_stepwise_predominates(voice: Voice, key: Key, role: str = "cp") -> float:
    """
    Eq. 6, literally as printed (see module deviations): 2L..4L leaps
    is optimal; above 4L the excess is penalized; below 2L the count
    itself is (paper's third branch).
    """
    m = _melody(voice)
    length, L = len(m.pitches), m.L
    leaps = sum(1 for iv in intervals(m.pitches) if is_leap(iv))
    denom = length - 1 - 4 * L
    if denom <= 0:
        return 0.0
    if 2 * L <= leaps <= 4 * L:
        return 0.0
    if leaps > 4 * L:
        return _clamp((leaps - 4 * L) / denom)
    return _clamp(leaps / denom)


def h6_leap_then_step(voice: Voice, key: Key, role: str = "cp") -> float:
    """Eq. 7: #large leaps not followed by stepwise motion / #large leaps."""
    ivs = intervals(_pitches(voice))
    large = [i for i, iv in enumerate(ivs) if is_large_leap(iv)]
    if not large:
        return 0.0
    bad = sum(1 for i in large if i + 1 >= len(ivs) or not is_step(ivs[i + 1]))
    return _clamp(bad / len(large))


def h7_leap_changes_direction(voice: Voice, key: Key, role: str = "cp") -> float:
    """Eq. 8: #large leaps not followed by a direction change / #large leaps."""
    ivs = intervals(_pitches(voice))
    large = [i for i, iv in enumerate(ivs) if is_large_leap(iv)]
    if not large:
        return 0.0
    bad = sum(
        1
        for i in large
        if i + 1 >= len(ivs) or _sign(ivs[i + 1]) == _sign(ivs[i]) or ivs[i + 1] == 0
    )
    return _clamp(bad / len(large))


def h8_climax_consonant_with_tonic(voice: Voice, key: Key, role: str = "cp") -> float:
    """Eq. 9: 0 if the climax is melodically consonant with the tonic."""
    pitches = _pitches(voice)
    if not pitches:
        return 0.0
    iv_class = (max(pitches) - key.tonic) % 12
    return 0.0 if iv_class in (_MELODIC_CONSONANT - {12}) else 1.0


def h9_two_consecutive_leaps(voice: Voice, key: Key, role: str = "cp") -> float:
    """Eq. 10: # of 3 consecutive leaps / (length - 3)."""
    pitches = _pitches(voice)
    ivs = intervals(pitches)
    if len(pitches) <= 3:
        return 0.0
    triples = sum(1 for i in range(len(ivs) - 2) if all(is_leap(ivs[i + k]) for k in range(3)))
    return _clamp(triples / (len(pitches) - 3))


def h10_large_leaps_per_l(voice: Voice, key: Key, role: str = "cp") -> float:
    """Eq. 11: 0 if #large leaps <= 2L, else the excess normalized."""
    m = _melody(voice)
    length, L = len(m.pitches), m.L
    large = sum(1 for iv in intervals(m.pitches) if is_large_leap(iv))
    if large <= 2 * L:
        return 0.0
    denom = length - 1 - 2 * L
    if denom <= 0:
        return 1.0
    return _clamp((large - 2 * L) / denom)


def h11_long_stepwise_run(voice: Voice, key: Key, role: str = "cp") -> float:
    """Eq. 12: penalize a longest same-direction stepwise run above 5 notes."""
    pitches = _pitches(voice)
    longest = 0
    run = 1
    last = 0
    for iv in intervals(pitches):
        if is_step(iv) and (_sign(iv) == last or last == 0):
            run += 1
        elif is_step(iv):
            run = 2
        else:
            run = 1
        last = _sign(iv) if is_step(iv) else 0
        longest = max(longest, run)
    if longest <= 5:
        return 0.0
    return _clamp(longest / len(pitches))


def h12_direction_changes(voice: Voice, key: Key, role: str = "cp") -> float:
    """Eq. 13: at least 3 direction changes per L."""
    m = _melody(voice)
    L = m.L
    if L <= 0:
        return 0.0
    changes = direction_changes(voice)
    if changes >= 3 * L:
        return 0.0
    return _clamp(1.0 - changes / (3 * L))


def h13_ends_on_tonic(voice: Voice, key: Key, role: str = "cp") -> float:
    """Eq. 14: 0 if the ending note is the tonic."""
    pitches = _pitches(voice)
    if not pitches:
        return 0.0
    return 0.0 if pitches[-1] % 12 == key.tonic else 1.0


def h14_penultimate(voice: Voice, key: Key, role: str = "cp") -> float:
    """
    Eq. 15: the penultimate should be the leading tone (CP) or the
    supertonic (CF); 0 when a step from the final, 0.5 when only the
    degree is right, 1 otherwise.
    """
    pitches = _pitches(voice)
    if len(pitches) < 2:
        return 0.0
    pen, end = pitches[-2], pitches[-1]
    wanted = key.leading_tone if role == "cp" else key.supertonic
    if pen % 12 != wanted:
        return 1.0
    return 0.0 if abs(end - pen) <= 2 else 0.5


def h15_motion_consonant(voice: Voice, key: Key, role: str = "cp") -> float:
    """Eq. 16: #dissonant motion intervals / #motion intervals."""
    motions = motion_intervals(voice)
    if not motions:
        return 0.0
    bad = sum(1 for m in motions if not is_melodic_consonant(m))
    return _clamp(bad / len(motions))


def h16_motion_span(voice: Voice, key: Key, role: str = "cp") -> float:
    """Eq. 17: #motion intervals > 9 semitones / #motion intervals."""
    motions = motion_intervals(voice)
    if not motions:
        return 0.0
    return _clamp(sum(1 for m in motions if abs(m) > 9) / len(motions))


def h17_repeated_notes(voice: Voice, key: Key, role: str = "cp") -> float:
    """Eq. 18: #notes repeated more than 2L times / length."""
    m = _melody(voice)
    pitches = m.pitches
    if not pitches:
        return 0.0
    L = m.L
    counts = Counter(pitches)
    repeated = sum(c for c in counts.values() if c > 2 * L)
    return _clamp(repeated / len(pitches))


def h18_repeated_sequences(voice: Voice, key: Key, role: str = "cp") -> float:
    """Eq. 19: #notes in a repeated sequence within a 16-note window / length."""
    pitches = _pitches(voice)
    n = len(pitches)
    if n < 4:
        return 0.0
    marked = set()
    for i in range(n - 1):
        for j in range(i + 1, min(i + 16, n - 1)):
            if pitches[i] == pitches[j] and pitches[i + 1] == pitches[j + 1]:
                marked.update((i, i + 1, j, j + 1))
    return _clamp(len(marked) / n)


def h19_max_octave(voice: Voice, key: Key, role: str = "cp") -> float:
    """Eq. 20: #intervals > 12 semitones / #intervals."""
    ivs = intervals(_pitches(voice))
    if not ivs:
        return 0.0
    return _clamp(sum(1 for iv in ivs if abs(iv) > 12) / len(ivs))


# =============================================================================
# VERTICAL SUBSCORES (appendix Eqs. 21-39, Table 8)
# Every function takes (cf, cp, key) -> float in [0, 1]; the cantus is
# whole notes, the counterpoint's notes are classified by duration and
# read against the cantus pitch of their measure. cp is the upper voice.
# =============================================================================


class _Aligned:
    """The cp notes with their vertical intervals and attack context."""

    def __init__(self, cf: Voice, cp: Voice):
        self.cf_notes = as_notes(cf)
        if any(n.duration != WHOLE for n in self.cf_notes):
            raise ValueError("the cantus firmus must be whole notes")
        melody = _melody(cp)
        self.cp_notes = melody.notes
        self.measures = n_measures(self.cf_notes)
        if melody.measures != self.measures:
            raise ValueError(
                f"cf and cp lengths differ: {self.measures} vs {melody.measures} measures"
            )
        self.cf_pitch = [n.pitch for n in self.cf_notes]
        self.atk = melody.atk
        self.atk_pitches = melody.pitches
        # Attack position of every cp note index.
        self.attack_of: dict[int, int] = {}
        for pos, a in enumerate(self.atk):
            for idx in a.indices:
                self.attack_of[idx] = pos
        self.pairs: list[tuple[Note, int]] = [
            (n, n.pitch - self.cf_pitch[n.measure]) for n in self.cp_notes
        ]

    def interval(self, note: Note) -> int:
        return note.pitch - self.cf_pitch[note.measure]

    def attack_interval(self, pos: int) -> int:
        a = self.atk[pos]
        return a.pitch - self.cf_pitch[a.measure]

    def notes_with_intervals(self) -> list[tuple[Note, int]]:
        return self.pairs

    # ---- melodic context of a note's attack --------------------------

    def _around(self, note_idx: int) -> tuple[int | None, int, int | None]:
        pos = self.attack_of[note_idx]
        prev = self.atk_pitches[pos - 1] if pos > 0 else None
        nxt = self.atk_pitches[pos + 1] if pos + 1 < len(self.atk) else None
        return prev, self.atk_pitches[pos], nxt

    def is_suspension(self, note_idx: int) -> bool:
        """Tie-end on the first beat, resolving down by step."""
        note = self.cp_notes[note_idx]
        if note.tied != TIE_END or note.beat != 1.0:
            return False
        _, here, nxt = self._around(note_idx)
        return nxt is not None and is_step(nxt - here) and nxt < here

    def is_passing(self, note_idx: int) -> bool:
        prev, here, nxt = self._around(note_idx)
        if prev is None or nxt is None:
            return False
        return (
            is_step(here - prev) and is_step(nxt - here) and _sign(here - prev) == _sign(nxt - here)
        )

    def is_neighbor(self, note_idx: int) -> bool:
        prev, here, nxt = self._around(note_idx)
        if prev is None or nxt is None:
            return False
        return is_step(here - prev) and is_step(nxt - here) and nxt == prev

    def left_by_step(self, note_idx: int) -> bool:
        _, here, nxt = self._around(note_idx)
        return nxt is not None and is_step(nxt - here)


_aligned = _memo2(_Aligned)


def v1_consonant_whole_notes(cf: Voice, cp: Voice, key: Key) -> float:
    """Eq. 21: #dissonant whole notes / #whole notes."""
    al = _aligned(cf, cp)
    wholes = [(n, iv) for n, iv in al.notes_with_intervals() if n.duration == WHOLE]
    if not wholes:
        return 0.0
    bad = sum(1 for _, iv in wholes if not is_vertical_consonant(iv))
    return _clamp(bad / len(wholes))


def v2_half_notes(cf: Voice, cp: Voice, key: Key) -> float:
    """
    Eq. 22: dissonant halves are allowed as a beat-1 suspension
    (continued stepwise and downward) or as a weak-beat passing tone.
    """
    al = _aligned(cf, cp)
    halves = [(i, n) for i, n in enumerate(al.cp_notes) if n.duration == HALF]
    if not halves:
        return 0.0
    bad = 0
    for i, note in halves:
        if is_vertical_consonant(al.interval(note)):
            continue
        if note.beat == 1.0 and al.is_suspension(i):
            continue
        if note.beat != 1.0 and al.is_passing(i):
            continue
        bad += 1
    return _clamp(bad / len(halves))


def v3_quarter_notes(cf: Voice, cp: Voice, key: Key) -> float:
    """
    Eq. 23: dissonant quarters are allowed as a beat-1 suspension or,
    off the first beat, as passing or neighbor tones.
    """
    al = _aligned(cf, cp)
    quarters = [(i, n) for i, n in enumerate(al.cp_notes) if n.duration == QUARTER]
    if not quarters:
        return 0.0
    bad = 0
    for i, note in quarters:
        if is_vertical_consonant(al.interval(note)):
            continue
        if note.beat == 1.0 and al.is_suspension(i):
            continue
        if note.beat != 1.0 and (al.is_passing(i) or al.is_neighbor(i)):
            continue
        bad += 1
    return _clamp(bad / len(quarters))


def v4_eighth_pairs(cf: Voice, cp: Voice, key: Key) -> float:
    """Eq. 24: eighth notes occur in pairs; at least one must be consonant."""
    al = _aligned(cf, cp)
    eighths = [n for n in al.cp_notes if n.duration == EIGHTH]
    pairs = list(zip(eighths[0::2], eighths[1::2], strict=False))
    if not pairs:
        return 0.0
    bad = sum(
        1
        for a, b in pairs
        if not is_vertical_consonant(al.interval(a)) and not is_vertical_consonant(al.interval(b))
    )
    return _clamp(bad / len(pairs))


def v5_unisons(cf: Voice, cp: Voice, key: Key) -> float:
    """
    Eq. 25: #non-allowed unisons / #notes. Whole-note unisons are
    allowed at the extremes; shorter ones on beat 1 when tied over or
    left by step (Table 8 rule 5).
    """
    al = _aligned(cf, cp)
    if not al.cp_notes:
        return 0.0
    bad = 0
    for i, note in enumerate(al.cp_notes):
        if al.interval(note) != 0:
            continue
        if note.duration == WHOLE:
            if note.measure in (0, al.measures - 1):
                continue
        elif note.beat == 1.0 and (note.tied == TIE_END or al.left_by_step(i)):
            continue
        bad += 1
    return _clamp(bad / len(al.cp_notes))


def v6_max_distance(cf: Voice, cp: Voice, key: Key) -> float:
    """Eq. 26: #distances > 16 semitones / length, except on the climax."""
    al = _aligned(cf, cp)
    if not al.cp_notes:
        return 0.0
    peak = al.atk[climax_index(al.atk_pitches)]
    bad = sum(
        1
        for i, (n, iv) in enumerate(al.notes_with_intervals())
        if abs(iv) > 16 and i not in peak.indices
    )
    return _clamp(bad / len(al.cp_notes))


def v7_crossing(cf: Voice, cp: Voice, key: Key) -> float:
    """Eq. 27: #crossed notes / length."""
    al = _aligned(cf, cp)
    if not al.cp_notes:
        return 0.0
    bad = sum(1 for _, iv in al.notes_with_intervals() if iv < 0)
    return _clamp(bad / len(al.cp_notes))


def v8_overlap(cf: Voice, cp: Voice, key: Key) -> float:
    """Eq. 28: #overlaps / (length - 1), over successive attacks."""
    al = _aligned(cf, cp)
    n = len(al.atk)
    if n < 2:
        return 0.0
    bad = 0
    for i in range(1, n):
        prev_cf = al.cf_pitch[al.atk[i - 1].measure]
        here_cf = al.cf_pitch[al.atk[i].measure]
        if al.atk[i].pitch < prev_cf or here_cf > al.atk[i - 1].pitch:
            bad += 1
    return _clamp(bad / (n - 1))


def v9_unison_to_octave(cf: Voice, cp: Voice, key: Key) -> float:
    """Eq. 29: #motions from unison to octave (and vice versa) / #intervals."""
    al = _aligned(cf, cp)
    verts = [al.attack_interval(i) for i in range(len(al.atk))]
    if len(verts) < 2:
        return 0.0
    bad = sum(1 for a, b in itertools.pairwise(verts) if {abs(a), abs(b)} == {0, 12})
    return _clamp(bad / (len(verts) - 1))


def v10_ends_perfect(cf: Voice, cp: Voice, key: Key) -> float:
    """Eq. 30: 0 if the ending is a unison or octave."""
    al = _aligned(cf, cp)
    if not al.cp_notes:
        return 0.0
    return 0.0 if abs(al.interval(al.cp_notes[-1])) % 12 == 0 else 1.0


def _attack_motions(al: _Aligned):
    """(d_lower, d_upper, arrival interval) per successive attack pair."""
    for i in range(1, len(al.atk)):
        d_cf = al.cf_pitch[al.atk[i].measure] - al.cf_pitch[al.atk[i - 1].measure]
        d_cp = al.atk[i].pitch - al.atk[i - 1].pitch
        yield d_cf, d_cp, al.attack_interval(i)


def v11_perfect_approach(cf: Voice, cp: Voice, key: Key) -> float:
    """
    Eq. 31: #perfect intervals not approached by contrary or oblique
    motion / #intervals. Covers parallel and direct fifths/octaves.
    """
    al = _aligned(cf, cp)
    if len(al.atk) < 2:
        return 0.0
    bad = sum(
        1
        for d_cf, d_cp, iv in _attack_motions(al)
        if is_perfect(iv) and motion_type(d_cf, d_cp) not in (CONTRARY, OBLIQUE)
    )
    return _clamp(bad / (len(al.atk) - 1))


def v12_simultaneous_leaps(cf: Voice, cp: Voice, key: Key) -> float:
    """Eq. 32: (#sim leaps not in same dir + #large sim leaps in same dir) / (2 x #intervals)."""
    al = _aligned(cf, cp)
    if len(al.atk) < 2:
        return 0.0
    bad = 0
    for d_cf, d_cp, _ in _attack_motions(al):
        if (is_leap(d_cf) and is_leap(d_cp)) and (
            _sign(d_cf) != _sign(d_cp) or abs(d_cf) > 6 or abs(d_cp) > 6
        ):
            bad += 1
    return _clamp(bad / (2 * (len(al.atk) - 1)))


def v13_all_motion_types(cf: Voice, cp: Voice, key: Key) -> float:
    """Eq. 33: 1 - #different motion types used / 4."""
    al = _aligned(cf, cp)
    if len(al.atk) < 2:
        return 0.0
    used = {motion_type(d_cf, d_cp) for d_cf, d_cp, _ in _attack_motions(al)}
    return _clamp(1.0 - len(used) / 4.0)


def v14_climaxes_disjoint(cf: Voice, cp: Voice, key: Key) -> float:
    """Eq. 34: 1 if the climaxes of CF and CP fall in the same measure."""
    al = _aligned(cf, cp)
    if not al.cf_pitch or not al.atk:
        return 0.0
    cf_peak = climax_index(al.cf_pitch)
    cp_peak = al.atk[climax_index(al.atk_pitches)].measure
    return 1.0 if cf_peak == cp_peak else 0.0


def v15_successive_perfect(cf: Voice, cp: Voice, key: Key) -> float:
    """
    Eq. 35: successive same-class unisons/octaves/fifths on first beats
    not separated by three quarter notes / (#measures - 1).
    """
    al = _aligned(cf, cp)
    if al.measures < 2:
        return 0.0
    first_beats: dict[int, tuple[int, int]] = {}
    for i, note in enumerate(al.cp_notes):
        if note.beat == 1.0 and note.measure not in first_beats:
            first_beats[note.measure] = (i, al.interval(note))
    bad = 0
    for m in range(al.measures - 1):
        if m not in first_beats or m + 1 not in first_beats:
            continue
        i_a, iv_a = first_beats[m]
        i_b, iv_b = first_beats[m + 1]
        if abs(iv_a) % 12 not in (0, 7) or abs(iv_a) % 12 != abs(iv_b) % 12:
            continue
        between = al.cp_notes[i_a + 1 : i_b]
        if sum(1 for n in between if n.duration == QUARTER) >= 3:
            continue
        bad += 1
    return _clamp(bad / (al.measures - 1))


def v16_afterbeat_perfect(cf: Voice, cp: Voice, key: Key) -> float:
    """
    Eq. 36: successive same-class perfect intervals off the first beat
    separated by fewer than two notes / #(notes - measures). The
    paper's afterbeat consonant-suspension exception is omitted (see
    module deviations).
    """
    al = _aligned(cf, cp)
    off = [(i, al.interval(n)) for i, n in enumerate(al.cp_notes) if n.beat != 1.0]
    if not off:
        return 0.0
    perfect = [(i, abs(iv) % 12) for i, iv in off if abs(iv) % 12 in (0, 7)]
    bad = 0
    for (i_a, cls_a), (i_b, cls_b) in itertools.pairwise(perfect):
        if cls_a == cls_b and i_b - i_a - 1 < 2:
            bad += 1
    return _clamp(bad / len(off))


def v17_battuta(cf: Voice, cp: Voice, key: Key) -> float:
    """Eq. 37: #ottava and quinta battutas / #measures."""
    al = _aligned(cf, cp)
    if len(al.atk) < 2 or al.measures == 0:
        return 0.0
    bad = sum(
        1
        for d_cf, d_cp, iv in _attack_motions(al)
        if motion_type(d_cf, d_cp) == CONTRARY
        and is_leap(d_cf)
        and is_leap(d_cp)
        and abs(iv) % 12 in (0, 7)
    )
    return _clamp(bad / al.measures)


def v18_ending_suspension(cf: Voice, cp: Voice, key: Key) -> float:
    """
    Eq. 38 with corrected polarity (the printed formula awards 1 to
    Table 8's best ending): 0 for a dissonant suspension resolving into
    the leading tone in the penultimate measure, 0.5 for any other
    suspension there, 1 otherwise. All-whole-note voices (first
    species) cannot carry a suspension, so the rule reads 0 there and
    its default weight stays 0; generate.py enables it for the tied
    species.
    """
    al = _aligned(cf, cp)
    if all(n.duration == WHOLE for n in al.cp_notes) or al.measures < 2:
        return 0.0
    pen = al.measures - 2
    suspension = None
    for i, note in enumerate(al.cp_notes):
        if note.measure == pen and note.beat == 1.0 and note.tied == TIE_END:
            suspension = i
            break
    if suspension is None:
        return 1.0
    dissonant = not is_vertical_consonant(al.interval(al.cp_notes[suspension]))
    pen_attacks = [
        a
        for a in al.atk
        if a.measure == pen or (pen in {al.cp_notes[i].measure for i in a.indices})
    ]
    resolves_to_leading = pen_attacks and pen_attacks[-1].pitch % 12 == key.leading_tone
    return 0.0 if dissonant and resolves_to_leading else 0.5


def v19_imperfect_predominate(cf: Voice, cp: Voice, key: Key) -> float:
    """Eq. 39: 0 if thirds/sixths/tenths are at least half the notes."""
    al = _aligned(cf, cp)
    if not al.cp_notes:
        return 0.0
    good = sum(
        1 for _, iv in al.notes_with_intervals() if abs(iv) % 12 in _IMPERFECT_CONSONANT_CLASSES
    )
    if good >= 0.5 * len(al.cp_notes):
        return 0.0
    return _clamp(1.0 - 2.0 * good / len(al.cp_notes))


# =============================================================================
# REGISTRIES AND AGGREGATES
# =============================================================================

HORIZONTAL_RULES = {
    "h1_eighths_stepwise": h1_eighths_stepwise,
    "h2_one_climax": h2_one_climax,
    "h3_climax_strong_beat": h3_climax_strong_beat,
    "h4_consonant_intervals": h4_consonant_intervals,
    "h5_stepwise_predominates": h5_stepwise_predominates,
    "h6_leap_then_step": h6_leap_then_step,
    "h7_leap_changes_direction": h7_leap_changes_direction,
    "h8_climax_consonant_with_tonic": h8_climax_consonant_with_tonic,
    "h9_two_consecutive_leaps": h9_two_consecutive_leaps,
    "h10_large_leaps_per_l": h10_large_leaps_per_l,
    "h11_long_stepwise_run": h11_long_stepwise_run,
    "h12_direction_changes": h12_direction_changes,
    "h13_ends_on_tonic": h13_ends_on_tonic,
    "h14_penultimate": h14_penultimate,
    "h15_motion_consonant": h15_motion_consonant,
    "h16_motion_span": h16_motion_span,
    "h17_repeated_notes": h17_repeated_notes,
    "h18_repeated_sequences": h18_repeated_sequences,
    "h19_max_octave": h19_max_octave,
}

VERTICAL_RULES = {
    "v1_consonant_whole_notes": v1_consonant_whole_notes,
    "v2_half_notes": v2_half_notes,
    "v3_quarter_notes": v3_quarter_notes,
    "v4_eighth_pairs": v4_eighth_pairs,
    "v5_unisons": v5_unisons,
    "v6_max_distance": v6_max_distance,
    "v7_crossing": v7_crossing,
    "v8_overlap": v8_overlap,
    "v9_unison_to_octave": v9_unison_to_octave,
    "v10_ends_perfect": v10_ends_perfect,
    "v11_perfect_approach": v11_perfect_approach,
    "v12_simultaneous_leaps": v12_simultaneous_leaps,
    "v13_all_motion_types": v13_all_motion_types,
    "v14_climaxes_disjoint": v14_climaxes_disjoint,
    "v15_successive_perfect": v15_successive_perfect,
    "v16_afterbeat_perfect": v16_afterbeat_perfect,
    "v17_battuta": v17_battuta,
    "v18_ending_suspension": v18_ending_suspension,
    "v19_imperfect_predominate": v19_imperfect_predominate,
}

H_KEYS = tuple(HORIZONTAL_RULES)
V_KEYS = tuple(VERTICAL_RULES)

# Paper default: every weight 1, set by the user (Eq. 1). V18 defaults
# to 0 (it cannot apply without ties; the printed-formula erratum is
# documented above) - generate.py raises it to 1 for species 4 and 5.
DEFAULT_WEIGHTS_H = {name: 1.0 for name in H_KEYS}
DEFAULT_WEIGHTS_V = {name: 1.0 for name in V_KEYS}
DEFAULT_WEIGHTS_V["v18_ending_suspension"] = 0.0


def horizontal_subscores(voice: Voice, key: Key, role: str = "cp") -> dict[str, float]:
    return {name: fn(voice, key, role) for name, fn in HORIZONTAL_RULES.items()}


def vertical_subscores(cf: Voice, cp: Voice, key: Key) -> dict[str, float]:
    return {name: fn(cf, cp, key) for name, fn in VERTICAL_RULES.items()}


def horizontal_cost(
    voice: Voice, key: Key, role: str = "cp", weights: dict[str, float] | None = None
) -> float:
    """Weighted horizontal aspect of Eq. 1 (the whole objective for a CF)."""
    w = weights or DEFAULT_WEIGHTS_H
    return sum(w.get(name, 1.0) * fn(voice, key, role) for name, fn in HORIZONTAL_RULES.items())


def vertical_cost(cf: Voice, cp: Voice, key: Key, weights: dict[str, float] | None = None) -> float:
    """Weighted vertical aspect of Eq. 1."""
    w = weights or DEFAULT_WEIGHTS_V
    return sum(w.get(name, 1.0) * fn(cf, cp, key) for name, fn in VERTICAL_RULES.items())


def counterpoint_cost(
    cf: Voice,
    cp: Voice,
    key: Key,
    weights_h: dict[str, float] | None = None,
    weights_v: dict[str, float] | None = None,
) -> float:
    """
    Eq. 1 objective for a counterpoint against a cantus firmus: the
    CP's horizontal aspect plus the vertical aspect. (The CF's own
    horizontal score is a separate `horizontal_cost(cf, key, "cf")`.)
    """
    return horizontal_cost(cp, key, "cp", weights_h) + vertical_cost(cf, cp, key, weights_v)
