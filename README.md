# musicset

Score datasets for optical music recognition. One library for what kakigori and
gelato each re-implemented: turning MusicXML into rendered pages with complete
annotations, varying the engraving and degrading the images, and generating
synthetic scores.

| Stage | Status |
|---|---|
| 1. Dataset transformation: sources → pages + annotations | **done** |
| 2. Render variety (fonts, layout, spacing) + augmentation | **done**: varied profiles, scan presets, visualizer |
| 3. Synthetic score generation | **started**: generators, counterpoint pool, notation profiles |

## Install

```bash
uv sync --all-extras                      # here, for development
uv add --editable ../musicset             # from kakigori or gelato
uv add --editable '../musicset[viz]'      # with the dataset viewer (PySide6, networkx)
uv add --editable '../musicset[synth]'    # with the synthetic score generators (music21)
```

Python ≥ 3.13, Verovio ≥ 6.3. Tooling: `uv run ruff check`, `uv run ty check src tests`, `uv run pytest`.

Music fonts: SMuFL characters inside text (metronome marks, figured bass) are
drawn by cairosvg from installed fonts named after the engraving font. Install
the five fonts Verovio engraves with (from the matching Verovio release, into
`~/.local/share/fonts/musicset`) on every machine that builds datasets:

```bash
scripts/install-fonts.sh            # --force to reinstall; FONT_DIR=... for system-wide
```

## Usage

```bash
# 1. Choose sources (one XML parse per file, ~45 s for all 254k PDMX scores)
uv run musicset select ../music_data/mxl data/piano/sources --lite --limit 10000
uv run musicset select ../music_data/mxl data/real/sources --max-voices 3 \
    --per-voice 12000 16352 13504 --exclude-from data/real-eval/sources

# 2. Engrave and annotate: kakigori outputs (boxes + graphs), gelato outputs (per-page ABC)
uv run musicset build data/piano/sources data/piano --labels lite --graph --abc \
    --max-measures-per-system 15 --max-notes-per-measure 40 \
    --profile varied --variants 3 --seed 0    # 3 engravings per score, varied styles

# 3. Scan-like copies (or use musicset.augment.Augmenter online in a training dataset)
uv run musicset augment data/piano data/piano-scans --preset mixed --copies 2

# 4. Look at it, next to real scans
uv run musicset view data/piano --labels lite --scans examples/imslp
uv run musicset view data/validation-lite --labels lite --pred-kern ../kakigori/inference_out/validation_det/krn
```

`scripts/make_examples.py` fills the gitignored `examples/` with 22 varied
public-domain PDMX scores (single staff, grand staff, voices, lyrics,
transposing ensembles) to try every command on in seconds.

| Command | Does |
|---|---|
| `select` | filter and sample sources: `--lite` gates (one part, ≤2 staves, one voice per staff, no tab/percussion, no gestural-only accidentals), voice-count caps, holdout exclusion; links (or `--copy`) into a directory with `selection.jsonl` |
| `build` | engrave, rasterize, annotate every page; `--labels` (detection view), `--graph`, `--kern`, `--mei`, `--paginated`, `--abc`; layout gates `--max-pages`, `--max-measures-per-system`, `--max-notes-per-measure`; pages with a staff off the page are always rejected |
| `annotate` | canonical annotations for SVGs rendered by the old pipelines |
| `detection` | re-project stored canonical pages under another label set |
| `graph` | graph groundtruth from MEI + detection files (old datasets) |
| `kern` | **kern from MEI + detection files (old datasets) |
| `augment` | scan-like copies of a built dataset: images, moved boxes, graph copies |
| `synth` | synthetic MusicXML: `exercise`, `piano`, `counterpoint`, `easy`, `melody`, `mutate`, `arrange` (see below) |
| `counterpoint-pool` | compose species counterpoint for `synth counterpoint` (slow; resumable JSON lines) |
| `view` | Qt viewer: page with boxes, graph edges and text target; the score graph as a network (ForceAtlas2, spring or page-position layout, nodes drawn as their SMuFL glyph or page crop, class/id labels, hover highlights neighbours, drag moves nodes); augmentation next to real scans; source pages, ground-truth and predicted **kern (`--pred-kern DIR`) engraved side by side with their match against the MEI, ordered worst first on request |

Workers default to half the spare cores; set `MUSICSET_WORKERS` or `--workers`.
Each score runs in its own process from a preloaded forkserver, so a Verovio
segfault or hang costs that score only (`failures.jsonl` lists failures and
rejections).

A build writes the directory names kakigori and gelato already read:

```
injected/<score>.mxl              source with stable element ids
svgs/<score>_page<N>.svg          engraved pages
imgs/<score>_page<N>.png          rasterized pages
pages/<score>_page<N>.json.gz     canonical annotation (~22 KB/page)
annotations/<score>_page<N>.json  detection view (kakigori format, + parent) --labels
graphs/<score>.json               graph view (kakigori .pt layout, JSON)  --graph
kern/<score>.krn                  **kern read off annotations + graph     --kern
mei/<score>.mei                   MEI, ids shared with the SVG            --mei / --graph / --kern
paginated/<score>_page<N>.mxl     the measures drawn on the page          --paginated
abcs/<score>_page<N>.abc          the same as ABC (gelato's target)       --abc
manifest/<score>.json             settings + fingerprints; written last
```

Reruns are resumable and never stale: a score is skipped only when its manifest
matches the current settings; if only the label set changed, the detection and
graph views are re-projected from `pages/` and `mei/` without rendering.

## Label sets

A label set is a directory in kakigori's conf format (`config.json`,
`smufl_mapping.json`, `hierarchy.json`, `structure.json`). Built in:

| Name | Classes | |
|---|---|---|
| `full` | 107 | kakigori's full set without the `pgHead`/`autogenerated` page furniture |
| `lite` | 99 | piano / single staff |
| `lite-pruned` | 91 | lite without classes no consumer reads (stage-C best model) |

`--labels` also takes any directory, e.g. kakigori's own `conf/`.

## Canonical annotation

`musicset.annotate` reads everything off a rendered page once, independent of
any model's labels (`PageAnnotation`, see `annotate/schema.py`):

- **elements**: every classed SVG group (note, staff, beam, dynam, ...) in
  document order, with its ink box, its parent in the engraving tree, the
  classes derived from geometry (`barlineFinal`, `beam16`, `stem8`, ...), text,
  and the semantic attributes Verovio attaches (`pname`, `oct`, `dur`, `grace`...);
- **glyphs**: every SMuFL glyph by codepoint, with its owning element and id.

Coordinates are SVG user units, so one annotation serves any raster size.
Text boxes are measured: cairosvg's text drawing is hooked during rasterization
and each text element's glyph outlines are read back (IoU 0.88–0.97 with the
true ink; the old character-count estimate scored 0.55–0.67).

Label sets are applied by *views* (`musicset.views`): `detection` (kakigori
boxes), `graph` (kakigori groundtruth graph), `kern` (kakigori's serializer,
**kern read off the annotations and graph), `paginate` (per-page MusicXML)
and `abc` (per-page ABC, through W.G. Vree's xml2abc, vendored under LGPL).

### Why kern comes from the serializer

Measured on kakigori's validation sets against MEI, per staff and measure:

| Route | Result |
|---|---|
| Verovio MEI→Humdrum (kakigori's old `krn/`) | still broken in 6.3.0: tokens like `.ZZZ8e`, `32%-3ryy`, no clef/key/meter |
| humlib musicxml2hum (Verovio `musicxml-hum` input) | clean, 98.4% pitch on lite, but crashes on every music21-written score tested (synthetic data) and ~2.5% of real ones, and writes transposing instruments at concert pitch while the page shows written pitch |
| **serializer** (graph → kern) | reads what is drawn; fresh lite build: 98.42% pitch, 98.17% rhythm, 464/500 files perfect; pure Python |

The serializer (`MinimalHumdrumSerializer`) also takes model predictions
(`from_pyg`), so kakigori's inference can import it from here.

## Parity with the old pipelines

Checked with `scripts/` against kakigori's and gelato's datasets:

- **Engraving**: 35/35 validation-test pages byte-identical SVG (Verovio 6.0.1).
  Builds now use Verovio 6.3, whose layout can differ from 6.0.1.
- **Ids**: identical injected MusicXML, except stems ending in a digit get an
  `m` prefix (`123-1` is not a valid XML id; it is now `m123-1`).
- **Detection**: 3,342 pages from four kakigori datasets identical in order and
  values (before the measure-box change below), except for these intentional fixes:
  - page headers: kakigori picked between `pgHead` and `autogenerated` through a
    set intersection whose order changes per Python process, so train-small and
    the validation sets disagree. Both classes are dropped from `full`, and the
    element kind now always wins over modifier classes;
  - flagged 32nd-or-shorter stems were `stem4` (no 32nd flags in the glyph map);
  - stem classes come from the note's written duration; the geometric rule
    labelled every stem under a beam by the beam's line count, mislabelling
    0.37% of stems in mixed groups (16th + 8th + 16th);
  - text boxes are measured (above), when rendered by musicset.
- **Graph**: 512 scores (validation-test + validation-lite, 322,962 edges)
  with edge lists identical to kakigori's builder, given the same inputs.
- **Pages / ABC**: gelato's eval-stage-3 and -7 (278 pages): the same measures
  on every page; ABC identical except 3 pages, from a fix: gelato carried the
  running attributes by tag only, so on two-staff parts one clef evicted the
  other and later pages were transcribed in the wrong clef (73 pages lost a
  clef; 3 had their ABC changed, all now matching the image).
- **Lite gates**: agree with kakigori's on 4,000 random PDMX scores, and accept
  all 9,500 of `train-lite`.
- **Kern**: same pitch/rhythm scores as kakigori's serializer on validation-test,
  -small and -lite (99.93/99.88, 98.34/97.64, 98.42/97.90), now deterministic:
  kakigori's output changed with Python's hash seed (19-20 of 201 files), and a
  staff's opening key/meter/clef could be a mid-system change (a 6/8 score
  opened as `*M1/8 *M3/4`). With musicset's stem labels, rhythm on a fresh lite
  build rises to 98.17% and perfect files from 442 to 464 of 500.

## Render variety and augmentation

Real scans differ from Verovio renders most in **engraving weight**. Measured
on IMSLP plate engravings (Schumann op. 68, Schmitt, Pischna; `musicset.scans`),
staff lines are 0.14-0.33 of a staff space thick (median 0.25); default Verovio
renders draw 0.056, and Verovio's heaviest setting reaches only 0.15.

- **Render profiles** (`--profile varied`, `musicset.render.profiles`): per
  score and variant, a font (Leipzig, Bravura, Gootville, Leland, Petaluma),
  staff size, line/stem/barline/ledger weights leaning heavy, slur and hairpin
  thickness, spacing, page size and margins, header/footer, and MuseScore's
  encoded system breaks half of the time when the source has them (Verovio
  `line`: its own page breaks, since encoded page breaks overflow a sampled
  page). Seeded, inside
  Verovio's published option limits, recorded in each manifest.
- **Augmentation** (`musicset.augment`): degradation steps modelled on how a
  scan is made (ink spread, faded ink, show-through, paper, skew, perspective,
  page bend, scanner shadow, blur, resolution loss, light, noise, dust,
  binarization, JPEG); geometric steps move the boxes with the ink, ink spread
  grows them. Presets `engraved`, `aged`, `photo`, `mixed`.
  `scripts/calibrate_augment.py` measures them against reference scans:
  `engraved` on varied renders is median 0.23 (p10-p90 0.15-0.29) against the
  scans' 0.25 (0.14-0.33). Augment full-resolution pages, before downscaling to
  a model's input: a 3x3 max filter after kakigori's 640 px letterbox erased
  ~95% of stems, barlines and staff lines
  (`kakigori/docs/experiments/vision/augmentation_dilate.ipynb`).
- **Visualizer** (Augment tab of `musicset view`): clean, augmented and a real
  scan scaled to the same staff space, panned and zoomed together; preset,
  strength and seed controls, re-engraving with a sampled profile, the applied
  steps, and the line/space ratio of all three (pages are deskewed first).

A layout bug surfaced on varied renders: a long editorial note set as a
direction ran four page widths off the page, and since a measure's box was the
union of everything in it, the measure spanned the system and broke measure
order in the kern serializer. A measure's box now covers its staves and barline
only; kern improved on every set (fresh lite build 98.49/98.24, 466/500 perfect;
varied examples 96.95 → 98.77 pitch).

Fonts: Gootville has no metronome glyphs, so such text runs use Verovio's
fallback font (Leipzig), as Verovio does for notation glyphs. Leland and
Petaluma space metronome notes wider (their space glyph), as Verovio's own
output does.

## Synthetic scores

`musicset synth KIND OUT -n N` writes MusicXML (`.mxl`) to feed `build`; every
file has its own seed from (kind, `--seed`, index), so batches are reproducible,
resumable and extendable (`--start`), and each run reports how many of its
scores are musically distinct.

```bash
uv run musicset synth exercise data/synth/src -n 20000 --profile detector
uv run musicset synth piano data/synth/src -n 20000
uv run musicset counterpoint-pool data/synth/pool.jsonl -n 1000      # ~30 min, 10 workers
uv run musicset synth counterpoint data/synth/src -n 20000 --pool data/synth/pool.jsonl
uv run musicset synth mutate data/synth/src --sources data/piano/sources
uv run musicset build data/synth/src data/synth --labels lite --kern --profile varied
```

| Kind | What |
|---|---|
| `exercise` | scales (major, harmonic/melodic minor, chromatic) and chord progressions, one or two staves, with curriculum switches (`--staves`, `--no-fingerings`, `--no-ornaments`, `--no-dynamics`, `--no-slurs`, `--no-ties`, `--meters`, `--contents`) |
| `piano` | a melody, chords or thirds over block chords, Alberti bass, broken chords, oom-pah, octaves or arpeggios, on a functional progression (cadences, secondary dominants, sevenths, sus4, pivot modulations) voiced with minimal motion |
| `counterpoint` | two-voice species counterpoint (1st-5th species, Herremans & Sörensen's VNS in `musicset.synth.counterpoint`) from the pool, in any key, in 4/4, C, cut time or halved into 2/4, as a piano grand staff, an instrumental duet (alto and tenor clefs), two voices on one staff, or a choir with Latin lyrics |
| `easy`, `melody` | one-measure and 4-8 measure single-staff scores for first curriculum stages |
| `mutate`, `arrange` | real scores with rare notation injected (32nds, triplets, ornaments, ottavas, octave clefs, clef changes, double barlines...), or one mutation per measure |

Notation rates are a `Profile` (`musicset.synth.decorate`): `plain` (notes
only), `default` (moderate rates), `detector` (weighted toward the classes a
detector reads worst: 32nds, trills, mordents, fermatas, pp/mp/ff, double
barlines, common/cut and 5/7/9/10 meters, clef changes, octave clefs, ottava
brackets, key changes drawing cancellation naturals, whole-measure rests,
fingering on every note) or `rare` (the same without the fingering flood).
`--meters rare` with short pieces (`--min-measures 4 --max-measures 8`)
multiplies the rare time signatures, which a piece draws only once. On a
sample build every one of the 87 `lite` classes appears; C clefs come from the
counterpoint duets, double sharps from the leading tones of sharp minor keys.

**kern on synthetic builds** (serializer vs MEI, pitch/rhythm): counterpoint,
easy, melody 100/100; exercise 99.76/99.76; piano 99.59/99.57 (detector
profile: octave clefs, clef changes, ottavas); arrange 99.37/98.99; mutate
98.33/97.97. Their crowded measures exposed faults real scores hit rarely,
now fixed:

- **Serializer**: closely set noteheads of one layer were sliced into one row
  (a kern chord); tuplet ratios came from the event count, so a triplet of an
  eighth, two sixteenths and an eighth read as 4:2, and a quarter-eighth
  triplet as nothing. Now one layer never shares a row, and the ratio comes
  from the written total (3/8: a triplet).
- **Graph groundtruth**: dots, accidentals, articulations and stems were
  attached to the nearest event or the touching notehead, and in a crowded
  measure that is the neighbour (a dotted eighth lost its dot to the next
  sixteenth). The detection view now records each glyph's `parent` event from
  the SVG nesting and the graph view attaches by it (older datasets without
  it fall back to geometry).
- **Build**: Verovio occasionally sets a last system taller than the room
  left on the page; pages with a staff off the page are rejected instead of
  yielding clipped groundtruth.

On the real sets the same fixes move validation-small rhythm 97.64 -> 97.71,
validation-lite to 467/500 perfect files and varied examples to 34/44; nothing
regresses. Detection and graph views carry `VIEWS_VERSION` in their
fingerprint, so rebuilding an existing dataset re-projects them from `pages/`
and `mei/` without rendering.

**Known limitation: octave clefs changed mid-system.** A switch to an 8va/8vb
treble clef inside a system is drawn as a small clef-change glyph, which the
label sets can only call `gClefChange` (a plain treble clef). The kern
serializer then reads every measure up to the next system an octave off (from
the next system on, the full-size `clefG8va`/`clefG8vb` is read correctly).
The generators therefore put octave clefs at the start of a part and only
switch back to a plain treble clef mid-piece. Real scores rarely change to an
octave clef mid-system; if they turn up, the fix is a `gClef8vaChange` /
`gClef8vbChange` class pair (or reading the "8" off the glyph).
