"""``musicset`` command line."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from musicset.annotate.smufl import available_fonts
from musicset.labels import LabelSet, builtin_names
from musicset.parallel import map_isolated, map_pool
from musicset.pipeline import (
    PAGE_SUFFIX,
    BuildConfig,
    Layout,
    annotate_file,
    augment_page,
    augmented_graphs,
    build_score,
    detection_file,
    graph_file,
    kern_file,
    page_file_stem,
)
from musicset.render.options import RenderOptions
from musicset.score import find_sources
from musicset.select import Selection, place, score_stats
from musicset.views.graph import page_sort_key

logger = logging.getLogger("musicset")


def _labels(args: argparse.Namespace) -> LabelSet | None:
    return LabelSet.load(args.labels) if args.labels else None


def _add_labels_arg(p: argparse.ArgumentParser, *, required: bool = False) -> None:
    p.add_argument(
        "--labels",
        required=required,
        help=f"label set: {', '.join(builtin_names())}, or a conf directory (config.json, ...)",
    )


def _report(results, total: int, desc: str, unit: str) -> list[str]:
    """Drain ``(task, Outcome)`` pairs with a progress bar; returns the error lines."""
    errors = []
    for task, outcome in tqdm(results, total=total, desc=desc, unit=unit):
        if not outcome.ok:
            first = task[0] if isinstance(task, tuple) else task
            errors.append(f"{getattr(first, 'name', first)}: {outcome.error}")
    for line in errors[:20]:
        logger.warning(line)
    return errors


def _stem_list(path: Path) -> list[str]:
    """Stems listed one per line (blank lines and # comments skipped)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [s for line in lines if (s := line.split("#", 1)[0].strip())]


def cmd_select(args: argparse.Namespace) -> None:
    sources = find_sources(args.sources)
    exclude = frozenset(p.stem for d in args.exclude_from for p in find_sources(d))
    if args.exclude_list:
        exclude |= frozenset(_stem_list(args.exclude_list))
    selection = Selection(
        lite=args.lite,
        max_voices=args.max_voices,
        per_voice=tuple(args.per_voice),
        per_staves=tuple(args.per_staves),
        limit=args.limit,
        exclude=exclude,
        only=frozenset(_stem_list(args.only)) if args.only else None,
        dedup=args.dedup,
        seed=args.seed,
    )
    stats = []
    for _, outcome in tqdm(
        map_pool(score_stats, sources, workers=args.workers, chunksize=64),
        total=len(sources),
        desc="Scanning",
        unit="score",
    ):
        if outcome.ok and outcome.value is not None:
            stats.append(outcome.value)
    chosen = selection.choose(stats)
    place(chosen, args.out, copy=args.copy)
    with (args.out / "selection.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(s.to_dict()) + "\n" for s in chosen)
    accepted = Counter(s.max_staves for s in stats if selection.accepts(s))
    by_voices = Counter(s.voices for s in chosen)
    by_staves = Counter(s.max_staves for s in chosen)
    logger.info(
        "selected %d of %d scores into %s (by voice count: %s; by staff count: %s of %s accepted)",
        len(chosen),
        len(sources),
        args.out,
        dict(sorted(by_voices.items())),
        dict(sorted(by_staves.items())),
        dict(sorted(accepted.items())),
    )


def cmd_build(args: argparse.Namespace) -> None:
    layout = Layout(args.out)
    # A build's own outputs must never be read back as sources (sources may live
    # inside the dataset directory, e.g. data/x/sources)
    generated = [d.resolve() for d in (layout.injected, layout.paginated)]
    sources = [
        s
        for s in find_sources(args.sources)
        if not any(s.resolve().is_relative_to(d) for d in generated)
    ]
    if args.limit:
        sources = sources[: args.limit]
    if not sources:
        raise SystemExit(f"no .mxl/.musicxml/.xml files under {args.sources}")

    layout.create()
    cfg = BuildConfig(
        render=RenderOptions(font=args.font, breaks=args.breaks),
        raster_scale=args.raster_scale,
        write_svg=not args.no_svg,
        write_png=not args.no_png,
        write_mei=args.mei,
        write_paginated=args.paginated,
        write_abc=args.abc,
        labels=_labels(args),
        write_graph=args.graph,
        write_kern=args.kern,
        max_pages=args.max_pages,
        max_measures_per_system=args.max_measures_per_system,
        max_notes_per_measure=args.max_notes_per_measure,
        min_kern_pitch=args.min_kern_pitch,
        min_kern_rhythm=args.min_kern_rhythm,
        profile=args.profile,
        seed=args.seed,
        variants=args.variants,
    )

    statuses: Counter[str] = Counter()
    failures = []
    tasks = [(src, layout, cfg, v) for src in sources for v in range(1, args.variants + 1)]
    if args.variants == 1:
        tasks = [(src, layout, cfg, 0) for src in sources]
    results = map_isolated(build_score, tasks, workers=args.workers, timeout=args.timeout)
    with tqdm(total=len(tasks), desc="Building", unit="score") as bar:
        for (src, _, _, _), outcome in results:
            if outcome.ok and outcome.value is not None:
                statuses[outcome.value.status] += 1
            else:
                kind = "rejected" if (outcome.error or "").startswith("Rejected") else "failed"
                statuses[kind] += 1
                failures.append({"source": str(src), "status": kind, "error": outcome.error})
            bar.set_postfix(statuses)
            bar.update()

    with (layout.root / "failures.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in failures)
    logger.info("done: %s", dict(statuses))
    if failures:
        logger.info("failures and rejections listed in %s", layout.root / "failures.jsonl")


def cmd_annotate(args: argparse.Namespace) -> None:
    svgs = sorted(args.svg_dir.glob("*.svg"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(s, args.out_dir) for s in svgs]
    errors = _report(
        map_pool(annotate_file, tasks, workers=args.workers), len(tasks), "Annotating", "page"
    )
    logger.info("annotated %d pages, %d failed", len(svgs) - len(errors), len(errors))


def cmd_detection(args: argparse.Namespace) -> None:
    labels = _labels(args)
    assert labels is not None
    pages = sorted(args.pages_dir.glob(f"*{PAGE_SUFFIX}"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for page in pages:
        size = None
        if args.img_dir is not None:
            with Image.open(args.img_dir / f"{page_file_stem(page)}.png") as im:
                size = im.size
        tasks.append((page, args.out_dir, labels.classes, size))
    errors = _report(
        map_pool(detection_file, tasks, workers=args.workers), len(tasks), "Projecting", "page"
    )
    logger.info("wrote %d detection files, %d failed", len(tasks) - len(errors), len(errors))


def _score_tasks(args: argparse.Namespace) -> tuple[LabelSet, list]:
    """(mei, its pages' detection files, out dir, node roles) per score."""
    labels = _labels(args)
    assert labels is not None
    if not labels.has_graph_roles:
        raise SystemExit(f"label set {labels.name!r} has no node roles (structure.json)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for mei in sorted(args.mei_dir.glob("*.mei")):
        pages = sorted(args.ann_dir.glob(f"{mei.stem}_page*.json"), key=lambda p: page_sort_key(p))
        if pages:
            tasks.append((mei, pages, args.out_dir, labels.node_roles))
    return labels, tasks


def cmd_graph(args: argparse.Namespace) -> None:
    _, tasks = _score_tasks(args)
    results = map_pool(graph_file, tasks, workers=args.workers, chunksize=4)
    errors = _report(results, len(tasks), "Graphs", "score")
    logger.info("wrote %d graphs, %d failed", len(tasks) - len(errors), len(errors))


def cmd_kern(args: argparse.Namespace) -> None:
    _, tasks = _score_tasks(args)
    results = map_pool(kern_file, tasks, workers=args.workers, chunksize=4)
    errors = _report(results, len(tasks), "Kern", "score")
    logger.info("wrote %d kern files, %d failed", len(tasks) - len(errors), len(errors))


def cmd_augment(args: argparse.Namespace) -> None:
    imgs = sorted((args.dataset / "imgs").glob("*.png"))
    tasks = []
    for img in imgs:
        ann = args.dataset / "annotations" / f"{img.stem}.json"
        if ann.exists():
            tasks.append((img, ann, args.out, args.preset, args.strength, args.seed, args.copies))
    if not tasks:
        raise SystemExit(f"no imgs/*.png with annotations/*.json in {args.dataset}")
    for sub in ("imgs", "annotations", "graphs"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    results = map_pool(augment_page, tasks, workers=args.workers, chunksize=2)
    errors = _report(results, len(tasks), "Augmenting", "page")
    graphs = augmented_graphs(args.dataset, args.out, args.copies)
    logger.info(
        "wrote %d augmented pages (%d failed) and %d graphs to %s",
        (len(tasks) - len(errors)) * args.copies,
        len(errors),
        graphs,
        args.out,
    )


def cmd_view(args: argparse.Namespace) -> None:
    try:
        from musicset.viewer import run
    except ImportError as e:
        raise SystemExit(f"the viewer needs the viz extra: uv add 'musicset[viz]' ({e})") from e
    scans = []
    if args.scans is not None:
        scans = sorted(
            p for p in args.scans.iterdir() if p.suffix.lower() in (".pdf", ".png", ".jpg")
        )
    run(args.dataset, _labels(args), scans, args.pred_kern)


def cmd_synth(args: argparse.Namespace) -> None:
    import random

    from musicset.synth import SOURCE_KINDS, SynthTask, synthesize
    from musicset.synth.write import diversity

    options: dict = {}
    if args.kind == "exercise":
        options = {
            "staves": args.staves,
            "fingerings": False if args.no_fingerings else None,
            "ornaments": False if args.no_ornaments else None,
            "dynamics": False if args.no_dynamics else None,
            "slurs": False if args.no_slurs else None,
            "ties": False if args.no_ties else None,
            "meters": args.meters,
            "contents": tuple(args.contents) if args.contents else None,
            "min_measures": args.min_measures,
            "max_measures": args.max_measures,
        }
    elif args.kind == "piano":
        options = {"min_measures": args.min_measures, "max_measures": args.max_measures}
    elif args.kind == "mutate":
        options = {"octave_clef": args.octave_clef, "octave_8va": args.octave_8va}
    options = {k: v for k, v in options.items() if v is not None}

    sources: list[Path] = []
    if args.kind in SOURCE_KINDS:
        if args.sources is None:
            raise SystemExit(f"{args.kind} needs --sources")
        sources = find_sources(args.sources)
        random.Random(args.seed).shuffle(sources)
        if not sources:
            raise SystemExit(f"no scores under {args.sources}")
    if args.kind == "counterpoint" and args.pool is None:
        raise SystemExit("counterpoint needs --pool (see musicset counterpoint-pool)")
    count = args.count or (len(sources) if sources else 0)
    if not count:
        raise SystemExit("how many? -n N")

    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [
        SynthTask(
            kind=args.kind,
            index=i,
            out=args.out,
            seed=args.seed,
            profile=args.profile,
            options=tuple(sorted(options.items())),
            source=sources[i % len(sources)] if sources else None,
            pool=args.pool,
        )
        for i in range(args.start, args.start + count)
    ]
    pending = [t for t in tasks if not t.path.exists()]
    logger.info("%d to generate (%d already present)", len(pending), len(tasks) - len(pending))
    errors = _report(
        map_pool(synthesize, pending, workers=args.workers, chunksize=4),
        len(pending),
        f"Synthesizing {args.kind}",
        "score",
    )
    distinct, total = diversity(t.path for t in tasks[:3000] if t.path.exists())
    level = logging.WARNING if total and distinct < 0.9 * total else logging.INFO
    logger.log(
        level,
        "%d failed; %d distinct scores in a sample of %d%s",
        len(errors),
        distinct,
        total,
        " - these settings cannot vary that much, loosen them" if level == logging.WARNING else "",
    )


def cmd_counterpoint_pool(args: argparse.Namespace) -> None:
    from musicset.synth.species import compose, load_pool, pool_tasks

    done = {piece["seed"] for piece in load_pool(args.pool)} if args.pool.exists() else set()
    tasks = pool_tasks(
        args.count,
        seed=args.seed,
        species=tuple(args.species),
        measures=tuple(args.measures),
        modes=tuple(args.modes),
        maxiters=args.maxiters,
    )
    pending = [t for t in tasks if t.seed not in done]
    logger.info(
        "%d pieces to compose (%d already in the pool)", len(pending), len(tasks) - len(pending)
    )
    args.pool.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    with args.pool.open("a", encoding="utf-8") as f:
        results = map_isolated(
            compose, pending, workers=args.workers, timeout=args.timeout, retries=0
        )
        for task, outcome in tqdm(results, total=len(pending), desc="Composing", unit="piece"):
            if outcome.ok and outcome.value is not None:
                f.write(json.dumps(outcome.value) + "\n")
                f.flush()
            else:
                failed += 1
                logger.warning("seed %d (species %d): %s", task.seed, task.species, outcome.error)
    logger.info(
        "pool %s: %d pieces, %d failed", args.pool, len(done) + len(pending) - failed, failed
    )


def cmd_symbols(args: argparse.Namespace) -> None:
    from musicset.symbols import SYMBOL_GROUPS, build_symbol_dataset

    if args.samples_per_class == 0 and not args.from_dataset:
        raise SystemExit("need -n > 0 or --from-dataset")
    report = build_symbol_dataset(
        args.out,
        samples_per_class=args.samples_per_class,
        from_dataset=args.from_dataset,
        max_crops_per_class=args.max_crops_per_class,
        homus=args.homus,
        homus_copies=args.homus_copies,
        homus_brush=args.homus_brush,
        groups=tuple(args.groups) if args.groups else SYMBOL_GROUPS,
        size=args.size,
        preset=args.preset,
        strength=args.strength,
        invert=args.invert,
        split=(args.split[0], args.split[1], args.split[2]),
        min_ink=args.min_ink,
        dedup=not args.no_dedup,
        write_png=not args.no_png,
        seed=args.seed,
        workers=args.workers,
    )
    logger.info(
        "wrote %d symbols (%d classes, splits=%s, sources=%s, removed=%s) into %s",
        sum(report.splits.values()),
        len(report.classes),
        report.splits,
        report.source_counts,
        report.removed,
        report.out,
    )


def main(argv: list[str] | None = None) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workers", type=int, default=None, help="default: $MUSICSET_WORKERS")
    parser = argparse.ArgumentParser(prog="musicset", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("select", parents=[common], help="choose source scores (no rendering)")
    p.add_argument("sources", type=Path, help="directory of .mxl/.musicxml/.xml scores")
    p.add_argument("out", type=Path, help="directory to link (or copy) the chosen scores into")
    p.add_argument("--lite", action="store_true", help="one part, <=2 staves, 1 voice per staff")
    p.add_argument("--max-voices", type=int, default=0, help="at most N voices (0: any)")
    p.add_argument(
        "--per-voice", type=int, nargs="*", default=[], help="cap per voice count 1, 2, 3..."
    )
    p.add_argument(
        "--per-staves", type=int, nargs="*", default=[], help="cap per staff count 1, 2..."
    )
    p.add_argument(
        "--only", type=Path, help="file of stems (one per line): select among these only"
    )
    p.add_argument("--limit", type=int, default=0, help="at most N scores in total")
    p.add_argument(
        "--exclude-from",
        type=Path,
        nargs="*",
        default=[],
        help="never pick scores present in these directories (holdout safety)",
    )
    p.add_argument("--exclude-list", type=Path, help="file of stems (one per line) never to pick")
    p.add_argument(
        "--dedup",
        action="store_true",
        help="one score per music content; none duplicating an excluded score",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy", action="store_true", help="copy instead of symlinking")
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("build", parents=[common], help="engrave scores and annotate every page")
    p.add_argument("sources", type=Path, help="directory of .mxl/.musicxml/.xml scores")
    p.add_argument("out", type=Path, help="dataset directory")
    _add_labels_arg(p)
    p.add_argument("--graph", action="store_true", help="graph groundtruth (needs --labels)")
    p.add_argument("--kern", action="store_true", help="**kern from the graph (needs --labels)")
    p.add_argument("--mei", action="store_true", help="MEI groundtruth (implied by --graph/--kern)")
    p.add_argument("--paginated", action="store_true", help="per-page MusicXML")
    p.add_argument("--abc", action="store_true", help="per-page ABC")
    p.add_argument(
        "--profile",
        choices=("fixed", "varied"),
        default="fixed",
        help="varied: sample font, staff size, line weights, spacing, page per score",
    )
    p.add_argument("--seed", type=int, default=0, help="seed of the varied profile")
    p.add_argument("--variants", type=int, default=1, help="engravings per source (varied)")
    p.add_argument("--font", default="Leipzig", choices=available_fonts())
    p.add_argument("--breaks", default="auto", help="Verovio breaks: auto, encoded, none...")
    p.add_argument("--raster-scale", type=float, default=1.0)
    p.add_argument("--no-svg", action="store_true", help="do not keep SVG pages")
    p.add_argument("--no-png", action="store_true", help="do not keep PNG pages")
    p.add_argument("--max-pages", type=int, default=0, help="reject longer scores (0: no limit)")
    p.add_argument(
        "--max-measures-per-system", type=float, default=0, help="reject smeared layouts (try 15)"
    )
    p.add_argument(
        "--max-notes-per-measure", type=float, default=0, help="reject barless sources (try 40)"
    )
    p.add_argument(
        "--min-kern-pitch",
        type=float,
        default=0,
        help="reject scores whose kern scores below this pitch recall/precision vs their MEI",
    )
    p.add_argument(
        "--min-kern-rhythm",
        type=float,
        default=0,
        help="reject scores whose kern scores below this rhythm recall vs their MEI",
    )
    p.add_argument("--limit", type=int, default=0, help="only the first N sources")
    p.add_argument("--timeout", type=float, default=120.0, help="seconds per score")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser(
        "annotate", parents=[common], help="canonical annotations from existing Verovio SVGs"
    )
    p.add_argument("svg_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.set_defaults(func=cmd_annotate)

    p = sub.add_parser("detection", parents=[common], help="detection view of canonical pages")
    p.add_argument("pages_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    _add_labels_arg(p, required=True)
    p.add_argument("--img-dir", type=Path, help="take image sizes from these PNGs")
    p.set_defaults(func=cmd_detection)

    p = sub.add_parser("graph", parents=[common], help="graph view from MEI + detection files")
    p.add_argument("mei_dir", type=Path)
    p.add_argument("ann_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    _add_labels_arg(p, required=True)
    p.set_defaults(func=cmd_graph)

    p = sub.add_parser("kern", parents=[common], help="**kern from MEI + detection files")
    p.add_argument("mei_dir", type=Path)
    p.add_argument("ann_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    _add_labels_arg(p, required=True)
    p.set_defaults(func=cmd_kern)

    p = sub.add_parser("augment", parents=[common], help="scan-like copies of a built dataset")
    p.add_argument("dataset", type=Path, help="built dataset (imgs/, annotations/)")
    p.add_argument("out", type=Path, help="output directory (imgs/, annotations/, graphs/)")
    p.add_argument("--preset", default="mixed", help="engraved, aged, photo, mixed, clean")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--copies", type=int, default=1, help="augmented copies per page")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_augment)

    p = sub.add_parser("synth", parents=[common], help="generate synthetic MusicXML scores")
    p.add_argument(
        "kind",
        choices=("exercise", "piano", "counterpoint", "easy", "melody", "mutate", "arrange"),
    )
    p.add_argument("out", type=Path, help="directory for the .mxl files")
    p.add_argument(
        "-n", "--count", type=int, default=0, help="scores (mutate/arrange: one per source)"
    )
    p.add_argument("--start", type=int, default=0, help="first index (to extend a batch)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--profile", default="default", help="notation rates: plain, default, detector, rare"
    )
    p.add_argument("--pool", type=Path, help="counterpoint pool (.jsonl)")
    p.add_argument("--sources", type=Path, help="real scores to mutate or arrange")
    g = p.add_argument_group("exercise")
    g.add_argument("--staves", type=int, choices=(1, 2))
    g.add_argument("--no-fingerings", action="store_true")
    g.add_argument("--no-ornaments", action="store_true", help="nor articulations, arpeggios")
    g.add_argument("--no-dynamics", action="store_true")
    g.add_argument("--no-slurs", action="store_true")
    g.add_argument("--no-ties", action="store_true")
    g.add_argument("--meters", choices=("simple", "medium", "full", "detector", "rare"))
    g.add_argument(
        "--contents",
        nargs="+",
        choices=("major", "harmonic_minor", "melodic_minor", "chromatic", "chord_progression"),
    )
    g = p.add_argument_group("exercise and piano")
    g.add_argument("--min-measures", type=int)
    g.add_argument("--max-measures", type=int)
    g = p.add_argument_group("mutate")
    g.add_argument("--octave-clef", type=float, help="per-part rate of octave clefs (0.4)")
    g.add_argument("--octave-8va", type=float, help="share of 8va among them (0.65)")
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser(
        "counterpoint-pool", parents=[common], help="compose species counterpoint (slow) for synth"
    )
    p.add_argument("pool", type=Path, help=".jsonl file, appended to (resumable)")
    p.add_argument("-n", "--count", type=int, required=True, help="pieces in the pool")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--species", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--measures", type=int, nargs="+", default=[8, 10, 12])
    p.add_argument("--modes", nargs="+", default=["major", "minor"], choices=("major", "minor"))
    p.add_argument("--maxiters", type=int, default=2, help="VNS restarts without improvement")
    p.add_argument("--timeout", type=float, default=900.0, help="seconds per piece")
    p.set_defaults(func=cmd_counterpoint_pool)

    p = sub.add_parser("view", help="browse a dataset: pages, boxes, graphs, targets")
    p.add_argument("dataset", type=Path, help="dataset directory (imgs/, annotations/, ...)")
    _add_labels_arg(p)
    p.add_argument(
        "--scans", type=Path, help="directory of reference scans (PDF/PNG) to compare with"
    )
    p.add_argument(
        "--pred-kern",
        type=Path,
        help="directory of predicted **kern (<score>.krn) to compare in the Kern tab",
    )
    p.set_defaults(func=cmd_view)

    p = sub.add_parser(
        "symbols",
        parents=[common],
        help="MNIST-like isolated symbol dataset (fingerings 1-5, ornaments, articulations, ...)",
    )
    p.add_argument("out", type=Path, help="output directory (dataset.npz, labels.json, train/...)")
    p.add_argument(
        "-n",
        "--samples-per-class",
        type=int,
        default=200,
        help="synthetic samples per class (0: extract from --from-dataset only)",
    )
    p.add_argument(
        "--from-dataset",
        type=Path,
        nargs="*",
        default=[],
        help="built dataset(s) to extract real page crops from",
    )
    p.add_argument(
        "--max-crops-per-class",
        type=int,
        default=0,
        help="at most N extracted crops per class (0: all)",
    )
    p.add_argument(
        "--homus",
        type=Path,
        default=None,
        help="HOMUS.zip or its extracted directory: real pen strokes, split by writer",
    )
    p.add_argument(
        "--homus-copies",
        type=int,
        default=1,
        help="augmented renderings of each HOMUS training sample",
    )
    p.add_argument(
        "--homus-brush",
        type=float,
        default=0.0,
        help="probability of rendering a HOMUS training sample with a pressure brush",
    )
    p.add_argument(
        "--groups",
        nargs="+",
        choices=(
            "fingering",
            "ornament",
            "articulation",
            "accidental",
            "dynamic",
            "pedal",
            "note",
            "rest",
            "clef",
            "timesig",
            "barline",
        ),
        help="symbol groups to include (default: all)",
    )
    p.add_argument(
        "--preset",
        choices=("mixed", "tablet", "score", "clean"),
        default="mixed",
        help="synthetic style and augmentation preset",
    )
    p.add_argument("--strength", type=float, default=1.0, help="augmentation strength")
    p.add_argument("--size", type=int, default=32, help="patch width and height in pixels")
    p.add_argument(
        "--invert",
        action="store_true",
        help="white ink on black background (classic MNIST); default is black on white",
    )
    p.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="train/val/test split fractions",
    )
    p.add_argument(
        "--min-ink", type=int, default=4, help="drop patches with fewer ink pixels (0: keep all)"
    )
    p.add_argument(
        "--no-dedup",
        action="store_true",
        help="keep byte-identical images in several splits (default: one split only)",
    )
    p.add_argument("--no-png", action="store_true", help="write dataset.npz and labels.json only")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_symbols)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
