"""Evolutionary loop: select survivors from a simulated generation, breed the next.

Driven entirely by ``config.json`` at the project root (selection
weights, breeding/mutation settings, output options). Typical use::

    # Inspect selection only (no files written)
    python evolution.py --generation 0 --select-only

    # Breed generation_01 (design_010..design_019 + OpenFOAM run/ folder)
    python evolution.py --generation 0

``--generation N`` chooses the SOURCE generation: 0 = the genesis batch
(``genesis-batch/``), 1 = ``generation_01/`` and so on. The source generation
must contain ``run/all_results_summary.csv`` brought back from the remote
machine (produced there by ``collect_results.py``).

Flow per invocation:
  1. Read every design's ``parameters.json`` and its 5-height CFD sweep results.
  2. Score designs with a weighted sum of min-max normalised metrics
     (CL up, CD down, L/D up, ground-effect gain up, |Cm| down, span/(L/D) down).
  3. Survivors = ``num_best`` top designs + ``num_runner_ups`` runner-ups.
  4. Survivors pair randomly (disjoint pairs); each pair breeds
     ``children_per_pair`` children by uniform crossover whose granularity is
     set per gene group in ``breeding.crossover_units``: the airfoil group is
     inherited whole, the per-section wing arrays (sweep/dihedral slopes,
     scale-down, pitch twist) mix element-by-element (each section's gene from
     parent A or B), and every winglet parameter is drawn from A or B
     independently.
  5. Each child gene then mutates with probability ``mutation.rate`` by a
     gaussian step of ``mutation.scale`` x the gene's configured range.
  6. Children are validated (same geometry guards as the genesis population),
     decoded, exported (OBJ/STL) and a full OpenFOAM case set is generated with
     exactly the same settings as the genesis batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .cfd_config import OpenFoamCaseConfig
from .config import DEFAULT_CONFIG, GeneratorConfig
from .exporter import ExportOptions, export_design
from .genome import WingGenome, decode_genome, genome_fingerprint
from .geometry_generator import write_design_files
from .motorbike_case_generator import generate_motorbike_cases, validate_reference_case
from .population_generator import validate_design, winglet_stations, write_population_manifest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVOLUTION_CONFIG = PROJECT_ROOT / "config.json"

# Gene groups; how each one crosses over is set in breeding.crossover_units
# (whole / per_gene / per_section — see crossover()).
GENE_GROUPS: dict[str, tuple[str, ...]] = {
    "airfoil": ("root_airfoil", "tip_airfoil", "transition_section"),
    "wing_pitch": ("root_pitch_deg", "pitch_change_deg"),
    "wing_horizontal": ("horizontal_slope",),
    "wing_vertical": ("vertical_slope",),
    "wing_scale": ("scale_down",),
    "winglet": (
        "winglet_is_pointy",
        "winglet_ellipse_length",
        "winglet_ellipse_radius",
        "winglet_ellipse_rotation_deg",
        "winglet_arc_angle_deg",
        "winglet_horizontal_rate",
        "winglet_scale_rate",
        "winglet_flat_tip_chord_m",
        "winglet_pointy_tip_x_frac",
    ),
}

ARRAY_GENES = ("horizontal_slope", "vertical_slope", "pitch_change_deg", "scale_down")
# Per-section change genes where index 0 is the root and must stay at 0.
ROOT_LOCKED_ARRAYS = ("pitch_change_deg", "scale_down")


# ─────────────────────────────────────────────────────────────────────────────
# Config / generation folders
# ─────────────────────────────────────────────────────────────────────────────
def load_evolution_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    def strip(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if not k.startswith("_")}
        return node

    return strip(raw)


def generation_dir(number: int, cfg: dict[str, Any]) -> Path:
    gens = cfg["generations"]
    if number == 0:
        return PROJECT_ROOT / gens["genesis_dir"]
    return PROJECT_ROOT / gens["generation_dir_pattern"].format(number=number)


def load_designs(gen_dir: Path) -> dict[str, dict[str, Any]]:
    designs: dict[str, dict[str, Any]] = {}
    for design_dir in sorted(gen_dir.glob("design_*")):
        params = design_dir / "parameters.json"
        if design_dir.is_dir() and params.exists():
            design = json.loads(params.read_text(encoding="utf-8"))
            designs[design["design_id"]] = design
    if not designs:
        raise FileNotFoundError(f"No design_*/parameters.json found in {gen_dir}")
    return designs


def load_results(gen_dir: Path, cfg: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    csv_path = gen_dir / "run" / cfg["generations"]["results_csv_name"]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Results CSV not found: {csv_path}\n"
            "Bring all_results_summary.csv back from the remote run first."
        )
    rows_by_design: dict[str, list[dict[str, str]]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows_by_design.setdefault(row["design_id"], []).append(row)
    return rows_by_design


# ─────────────────────────────────────────────────────────────────────────────
# Fitness
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DesignScore:
    design_id: str
    eligible: bool
    reason: str = ""
    valid_cases: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    normalized: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    rank: int = 0
    role: str = ""  # best / runner_up / ""


def _row_is_valid(row: dict[str, str], validity: dict[str, Any]) -> bool:
    try:
        final_time = float(row.get("final_time", "nan"))
        cd = float(row.get("Cd_avg_last_window") or row.get("Cd") or "nan")
        cl = float(row.get("Cl_avg_last_window") or row.get("Cl") or "nan")
        cm = float(row.get("CmPitch_avg_last_window") or row.get("CmPitch") or "nan")
    except ValueError:
        return False
    limit = float(validity["max_abs_coefficient"])
    return (
        math.isfinite(final_time)
        and final_time >= float(validity["required_final_time"])
        and all(math.isfinite(v) for v in (cd, cl, cm))
        and abs(cd) < limit
        and abs(cl) < limit
        and abs(cm) < limit
    )


def _least_squares_slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom <= 0.0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom


def total_span_m(design: dict[str, Any]) -> float:
    """Full half-model span including the winglet's spanwise reach."""
    wingtip = design.get("wingtip", {})
    length = float(wingtip.get("ellipse_length", 0.0))
    arc = math.radians(abs(float(wingtip.get("arc_angle_deg", 0.0))))
    return float(design.get("total_span", 0.0)) + length * math.sin(arc)


def design_metrics(
    design: dict[str, Any], rows: list[dict[str, str]], cfg: dict[str, Any]
) -> DesignScore:
    validity = cfg["results_validity"]
    design_id = design["design_id"]
    valid = [row for row in rows if _row_is_valid(row, validity)]

    score = DesignScore(design_id=design_id, eligible=False, valid_cases=len(valid))
    if validity["require_full_sweep"] and len(valid) < len(rows):
        score.reason = f"only {len(valid)}/{len(rows)} cases converged"
        return score
    if len(valid) < int(validity["min_valid_cases"]):
        score.reason = f"only {len(valid)} valid cases (need {validity['min_valid_cases']})"
        return score

    points = []
    for row in valid:
        h = float(row["nominal_height_above_ground_m"])
        cd = float(row.get("Cd_avg_last_window") or row["Cd"])
        cl = float(row.get("Cl_avg_last_window") or row["Cl"])
        cm = float(row.get("CmPitch_avg_last_window") or row["CmPitch"])
        points.append((h, cl, cd, cm))
    points.sort(key=lambda p: p[0])

    eval_height = cfg["fitness"].get("evaluation_height_m")
    if eval_height is not None:
        pool = [p for p in points if abs(p[0] - float(eval_height)) < 1e-6] or points
    else:
        pool = points

    lds = [cl / cd for _, cl, cd, _ in pool if cd > 1e-9]
    if not lds:
        score.reason = "no usable Cd values"
        return score

    heights = [p[0] for p in points]
    ld_curve = [cl / cd for _, cl, cd, _ in points if cd > 1e-9]
    ge_gain = -_least_squares_slope(heights[: len(ld_curve)], ld_curve) if len(ld_curve) >= 2 else 0.0

    cl_mean = sum(p[1] for p in pool) / len(pool)
    cd_mean = sum(p[2] for p in pool) / len(pool)
    ld_mean = sum(lds) / len(lds)
    abs_cm_mean = sum(abs(p[3]) for p in pool) / len(pool)
    span = total_span_m(design)
    lowest = points[0]

    score.metrics = {
        "cl": cl_mean,
        "cd": cd_mean,
        "lift_to_drag": ld_mean,
        "ground_effect_gain": ge_gain,
        "abs_cm": abs_cm_mean,
        "span_per_ld": (span / ld_mean) if ld_mean > 1e-6 else float("inf"),
        "ld_at_lowest_height": (lowest[1] / lowest[2]) if lowest[2] > 1e-9 else 0.0,
        "cl_at_lowest_height": lowest[1],
        "span_m": span,
    }
    score.eligible = True
    return score


def score_generation(
    designs: dict[str, dict[str, Any]],
    results: dict[str, list[dict[str, str]]],
    cfg: dict[str, Any],
) -> list[DesignScore]:
    scores = []
    for design_id, design in designs.items():
        rows = results.get(design_id, [])
        if not rows:
            scores.append(DesignScore(design_id=design_id, eligible=False, reason="no result rows"))
            continue
        scores.append(design_metrics(design, rows, cfg))

    weights: dict[str, float] = cfg["fitness"]["weights"]
    eligible = [s for s in scores if s.eligible]

    # Min-max normalise each weighted metric across the eligible designs. A design
    # with an infinite penalty metric (e.g. span_per_ld at L/D<=0) normalises to 1.
    for name, weight in weights.items():
        if weight == 0.0:
            continue
        values = [s.metrics[name] for s in eligible if math.isfinite(s.metrics[name])]
        lo = min(values) if values else 0.0
        hi = max(values) if values else 0.0
        for s in eligible:
            value = s.metrics[name]
            if not math.isfinite(value):
                s.normalized[name] = 1.0
            elif hi > lo:
                s.normalized[name] = (value - lo) / (hi - lo)
            else:
                s.normalized[name] = 0.5

    for s in eligible:
        s.score = sum(weights[name] * s.normalized.get(name, 0.0) for name in weights if weights[name] != 0.0)

    eligible.sort(key=lambda s: s.score, reverse=True)
    for rank, s in enumerate(eligible, start=1):
        s.rank = rank
    return sorted(scores, key=lambda s: (not s.eligible, s.rank or 1_000_000))


# ─────────────────────────────────────────────────────────────────────────────
# Selection
# ─────────────────────────────────────────────────────────────────────────────
def genome_from_design(design: dict[str, Any]) -> WingGenome:
    g = dict(design["genome"])
    for name in ARRAY_GENES:
        g[name] = tuple(float(v) for v in g[name])
    return WingGenome(**g)


def _genome_distance(a: WingGenome, b: WingGenome, config: GeneratorConfig) -> float:
    """Rough normalised genetic distance, used by the 'diverse' runner-up strategy."""
    g = config.genes
    distance = 0.0
    distance += 0.0 if a.root_airfoil == b.root_airfoil else 1.0
    distance += 0.0 if a.tip_airfoil == b.tip_airfoil else 1.0
    distance += 0.0 if a.winglet_is_pointy == b.winglet_is_pointy else 1.0
    scalar_bounds = {
        "root_pitch_deg": g.root_pitch_deg,
        "winglet_ellipse_length": g.winglet_ellipse_length,
        "winglet_ellipse_radius": g.winglet_ellipse_radius,
        "winglet_ellipse_rotation_deg": g.winglet_ellipse_rotation_deg,
        "winglet_arc_angle_deg": g.winglet_arc_angle_deg,
        "winglet_horizontal_rate": g.winglet_horizontal_rate,
        "winglet_scale_rate": g.winglet_scale_rate,
        "winglet_flat_tip_chord_m": g.winglet_flat_tip_chord_m,
        "winglet_pointy_tip_x_frac": g.winglet_pointy_tip_x_frac,
    }
    for name, bounds in scalar_bounds.items():
        span = max(bounds.span, 1e-9)
        distance += abs(getattr(a, name) - getattr(b, name)) / span
    array_bounds = {
        "horizontal_slope": g.horizontal_slope,
        "vertical_slope": g.vertical_slope,
        "pitch_change_deg": g.pitch_change_deg,
        "scale_down": g.scale_down,
    }
    for name, bounds in array_bounds.items():
        span = max(bounds.span, 1e-9)
        va, vb = getattr(a, name), getattr(b, name)
        distance += sum(abs(x - y) for x, y in zip(va, vb)) / (len(va) * span)
    return distance


def select_survivors(
    scores: list[DesignScore],
    designs: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    config: GeneratorConfig,
) -> list[str]:
    sel = cfg["selection"]
    ranked = [s for s in scores if s.eligible]
    needed = int(sel["num_best"]) + int(sel["num_runner_ups"])
    if len(ranked) < needed:
        raise SystemExit(
            f"Only {len(ranked)} eligible designs but {needed} survivors requested. "
            "Relax results_validity or selection counts in config.json."
        )

    best = ranked[: int(sel["num_best"])]
    for s in best:
        s.role = "best"

    if sel["runner_up_strategy"] == "diverse":
        pool = ranked[int(sel["num_best"]) : int(sel["num_best"]) + int(sel["diverse_pool_size"])]
        chosen: list[DesignScore] = []
        selected_genomes = [genome_from_design(designs[s.design_id]) for s in best]
        for _ in range(int(sel["num_runner_ups"])):
            def min_dist(candidate: DesignScore) -> float:
                cg = genome_from_design(designs[candidate.design_id])
                return min(_genome_distance(cg, g, config) for g in selected_genomes)

            pick = max((c for c in pool if c not in chosen), key=min_dist)
            chosen.append(pick)
            selected_genomes.append(genome_from_design(designs[pick.design_id]))
        runner_ups = chosen
    else:  # next_best
        runner_ups = ranked[int(sel["num_best"]) : needed]

    for s in runner_ups:
        s.role = "runner_up"
    return [s.design_id for s in best + runner_ups]


# ─────────────────────────────────────────────────────────────────────────────
# Crossover + mutation
# ─────────────────────────────────────────────────────────────────────────────
def _repair_slope_walk(values: list[float], bounds, max_step: float) -> list[float]:
    """Re-impose the random-walk smoothness invariant on a slope-gene array.

    The decoder/validator expect adjacent slope genes to differ by at most
    ``max_step``; element-wise crossover or mutation can break that, so each
    element is pulled back within reach of its predecessor (and the bounds).
    """
    repaired = list(values)
    for i in range(1, len(repaired)):
        lo = repaired[i - 1] - max_step
        hi = repaired[i - 1] + max_step
        repaired[i] = min(max(repaired[i], lo), hi)
        repaired[i] = min(bounds.maximum, max(bounds.minimum, repaired[i]))
    return repaired


def crossover(
    a: WingGenome,
    b: WingGenome,
    rng: random.Random,
    cfg: dict[str, Any],
    config: GeneratorConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Uniform crossover with per-group granularity from ``breeding.crossover_units``.

    * ``whole``       — the group is inherited intact from parent A or B.
    * ``per_gene``    — every gene in the group is drawn from A or B independently
                        (used for the winglet, so each winglet parameter evolves
                        on its own).
    * ``per_section`` — per-section array genes are mixed element-by-element (each
                        section's value from A or B); scalar genes in the group are
                        drawn independently. Mixed slope arrays are smoothness-
                        repaired afterwards.
    """
    units: dict[str, str] = cfg["breeding"]["crossover_units"]
    child: dict[str, Any] = {}
    choices: dict[str, Any] = {}
    mixed_arrays: list[str] = []

    for group, fields in GENE_GROUPS.items():
        mode = units.get(group, "whole")
        if mode == "whole":
            parent = a if rng.random() < 0.5 else b
            for name in fields:
                child[name] = getattr(parent, name)
            choices[group] = "A" if parent is a else "B"
            continue

        group_choices: dict[str, str] = {}
        for name in fields:
            value_a, value_b = getattr(a, name), getattr(b, name)
            if mode == "per_section" and name in ARRAY_GENES:
                picks = ["A" if rng.random() < 0.5 else "B" for _ in value_a]
                child[name] = tuple(
                    va if pick == "A" else vb for pick, va, vb in zip(picks, value_a, value_b)
                )
                group_choices[name] = "".join(picks)
                mixed_arrays.append(name)
            else:
                pick = "A" if rng.random() < 0.5 else "B"
                child[name] = value_a if pick == "A" else value_b
                group_choices[name] = pick
        choices[group] = group_choices

    # Keep the airfoil genes structurally consistent: a single-airfoil wing has
    # no transition, and a transition requires differing root/tip profiles.
    if child["transition_section"] is None or child["root_airfoil"] == child["tip_airfoil"]:
        child["tip_airfoil"] = child["root_airfoil"]
        child["transition_section"] = None

    # Element-wise mixing of two valid slope walks can produce adjacent steps
    # larger than the decoder allows; repair them back into a smooth walk.
    g = config.genes
    max_step = config.validation.max_slope_change_per_section
    slope_bounds = {"horizontal_slope": g.horizontal_slope, "vertical_slope": g.vertical_slope}
    for name, bounds in slope_bounds.items():
        if name in mixed_arrays:
            child[name] = tuple(
                round(v, 6) for v in _repair_slope_walk(list(child[name]), bounds, max_step)
            )
    return child, choices


def mutate(
    child: dict[str, Any],
    rng: random.Random,
    cfg: dict[str, Any],
    config: GeneratorConfig,
) -> tuple[dict[str, Any], list[str]]:
    rate = float(cfg["mutation"]["rate"])
    scale = float(cfg["mutation"]["scale"])
    g = config.genes
    log: list[str] = []

    def perturb(name: str, value: float, bounds) -> float:
        sigma = scale * max(bounds.span, 1e-9)
        new = min(bounds.maximum, max(bounds.minimum, value + rng.gauss(0.0, sigma)))
        log.append(f"{name}: {value:.6g} -> {new:.6g}")
        return new

    scalar_bounds = {
        "root_pitch_deg": g.root_pitch_deg,
        "winglet_ellipse_length": g.winglet_ellipse_length,
        "winglet_ellipse_radius": g.winglet_ellipse_radius,
        "winglet_ellipse_rotation_deg": g.winglet_ellipse_rotation_deg,
        "winglet_arc_angle_deg": g.winglet_arc_angle_deg,
        "winglet_horizontal_rate": g.winglet_horizontal_rate,
        "winglet_scale_rate": g.winglet_scale_rate,
        "winglet_flat_tip_chord_m": g.winglet_flat_tip_chord_m,
        "winglet_pointy_tip_x_frac": g.winglet_pointy_tip_x_frac,
    }
    for name, bounds in scalar_bounds.items():
        if rng.random() < rate:
            child[name] = round(perturb(name, float(child[name]), bounds), 6)

    array_bounds = {
        "horizontal_slope": g.horizontal_slope,
        "vertical_slope": g.vertical_slope,
        "pitch_change_deg": g.pitch_change_deg,
        "scale_down": g.scale_down,
    }
    max_step = config.validation.max_slope_change_per_section
    for name, bounds in array_bounds.items():
        values = list(child[name])
        mutated = False
        for i in range(len(values)):
            if name in ROOT_LOCKED_ARRAYS and i == 0:
                continue  # the root section's change genes stay 0
            if rng.random() < rate:
                values[i] = perturb(f"{name}[{i}]", float(values[i]), bounds)
                mutated = True
        if mutated and name in ("horizontal_slope", "vertical_slope"):
            values = _repair_slope_walk(values, bounds, max_step)
        child[name] = tuple(round(v, 6) for v in values)

    # Discrete genes.
    airfoils = config.gene_space.allowed_airfoils
    if rng.random() < rate and len(airfoils) > 1:
        new_root = rng.choice([a for a in airfoils if a != child["root_airfoil"]])
        log.append(f"root_airfoil: {child['root_airfoil']} -> {new_root}")
        child["root_airfoil"] = new_root
        if child["transition_section"] is None:
            child["tip_airfoil"] = new_root
    if rng.random() < rate:
        log.append(f"winglet_is_pointy: {child['winglet_is_pointy']} -> {not child['winglet_is_pointy']}")
        child["winglet_is_pointy"] = not child["winglet_is_pointy"]
    return child, log


def winglet_min_normal_thickness_m(design: dict[str, Any], config: GeneratorConfig) -> float:
    """Analytic estimate of the thinnest perpendicular winglet wall the loft builds.

    The loft stacks span-normal sections, so where the winglet path climbs by
    `slope` per metre of span the built wall thins to section_thickness /
    sqrt(1 + slope^2). Steep arcs (near 90 deg) can shear the winglet into a
    sub-millimetre sheet that snappyHexMesh turns into solver-killing slivers
    (this is what sank design_006/design_009 in the genesis batch).
    """
    stations = winglet_stations(design["sections"], design["wingtip"], config)
    previous_y = design["sections"][-1]["spanwise_position"]
    previous_z = design["sections"][-1]["vertical_displacement"]
    thinnest = float("inf")
    for station in stations:
        dy = max(station["spanwise_position"] - previous_y, 1e-9)
        dz = station["vertical_displacement"] - previous_z
        slope = abs(dz) / dy
        normal = station["target_thickness_m"] / math.sqrt(1.0 + slope * slope)
        thinnest = min(thinnest, normal)
        previous_y = station["spanwise_position"]
        previous_z = station["vertical_displacement"]
    return thinnest


# ─────────────────────────────────────────────────────────────────────────────
# Breeding
# ─────────────────────────────────────────────────────────────────────────────
def breed_generation(
    source_generation: int,
    cfg: dict[str, Any],
    *,
    config: GeneratorConfig = DEFAULT_CONFIG,
    select_only: bool = False,
    force: bool = False,
    seed: int | None = None,
    reference_case_dir: Path | None = None,
    n_subdomains: int | None = None,
) -> None:
    src_dir = generation_dir(source_generation, cfg)
    designs = load_designs(src_dir)
    results = load_results(src_dir, cfg)
    scores = score_generation(designs, results, cfg)
    survivors = select_survivors(scores, designs, cfg, config)
    _print_selection_table(scores, cfg)
    print(f"\nSurvivors: {', '.join(survivors)}")

    if select_only:
        return

    next_generation = source_generation + 1
    out_dir = generation_dir(next_generation, cfg)
    case_config = OpenFoamCaseConfig(population_dir=out_dir, output_dir=out_dir / "run")
    if reference_case_dir is not None:
        case_config = replace(case_config, reference_case_dir=reference_case_dir)
    if n_subdomains is not None:
        if n_subdomains < 1:
            raise ValueError("The core count must be at least 1.")
        case_config = replace(case_config, solver=replace(case_config.solver, n_subdomains=n_subdomains))
    if cfg["output"]["generate_openfoam_cases"]:
        validate_reference_case(case_config.reference_case_dir)
    per_gen = int(cfg["generations"]["designs_per_generation"])
    pairs_needed = len(survivors) // 2
    children_per_pair = int(cfg["breeding"]["children_per_pair"])
    if pairs_needed * children_per_pair != per_gen:
        raise SystemExit(
            f"{len(survivors)} survivors -> {pairs_needed} pairs x {children_per_pair} children "
            f"!= designs_per_generation ({per_gen}). Fix config.json."
        )

    if out_dir.exists():
        if not force:
            raise SystemExit(f"{out_dir} already exists. Re-run with --force to overwrite it.")
        _clear_directory(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_seed = seed if seed is not None else int(cfg["output"]["random_seed"]) + 1000 * next_generation
    rng = random.Random(base_seed)

    shuffled = list(survivors)
    rng.shuffle(shuffled)
    pairs = [(shuffled[i], shuffled[i + 1]) for i in range(0, len(shuffled), 2)]
    print(f"Parent pairs: {pairs} (seed {base_seed})")

    guards = cfg["guards"]
    fingerprints = {genome_fingerprint(genome_from_design(designs[d])) for d in survivors}
    child_designs: list[dict[str, Any]] = []
    breeding_records: list[dict[str, Any]] = []
    first_index = next_generation * per_gen

    export_options = ExportOptions(
        validate_cad=False,
        export_stl=bool(cfg["output"]["export_stl"]),
        export_step=False,
        render_preview=bool(cfg["output"]["render_previews"]),
    )

    child_number = 0
    for pair in pairs:
        parent_a = genome_from_design(designs[pair[0]])
        parent_b = genome_from_design(designs[pair[1]])
        for _ in range(children_per_pair):
            design_index = first_index + child_number
            record = _breed_one_child(
                parent_a, parent_b, pair, design_index, rng, cfg, config, fingerprints, guards
            )
            design = record.pop("design")
            design_dir = out_dir / design["design_id"]
            write_design_files(design, design_dir, config)
            export_design(design_dir, config, export_options)
            child_designs.append(design)
            breeding_records.append(record)
            print(
                f"  {design['design_id']}  <- {pair[0]} x {pair[1]}  "
                f"(attempts {record['attempts']}, mutations {len(record['mutations'])})"
            )
            child_number += 1

    write_population_manifest(
        out_dir,
        child_designs,
        config.with_overrides(num_designs=per_gen, output_dir=out_dir, random_seed=base_seed),
    )
    _write_reports(out_dir, source_generation, scores, survivors, pairs, breeding_records, base_seed, cfg)

    if cfg["output"]["generate_openfoam_cases"]:
        print("\nGenerating OpenFOAM cases (same settings as the genesis batch)...")
        cases = generate_motorbike_cases(case_config)
        print(f"Generated {len(cases)} OpenFOAM cases in {out_dir / 'run'}")

    print(f"\nGeneration {next_generation:02d} written to {out_dir}")


def _clear_directory(path: Path) -> None:
    """Empty a directory tree, keeping the top-level folder itself.

    On Windows an open Explorer window (or shell) can hold a handle on the
    folder, which makes deleting the folder itself fail even though all of its
    contents can be removed; emptying and reusing it avoids that.
    """
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _breed_one_child(
    parent_a: WingGenome,
    parent_b: WingGenome,
    pair: tuple[str, str],
    design_index: int,
    rng: random.Random,
    cfg: dict[str, Any],
    config: GeneratorConfig,
    fingerprints: set[str],
    guards: dict[str, Any],
) -> dict[str, Any]:
    max_attempts = int(guards["max_attempts_per_child"])
    for attempt in range(1, max_attempts + 1):
        child_fields, choices = crossover(parent_a, parent_b, rng, cfg, config)
        child_fields, mutations = mutate(child_fields, rng, cfg, config)
        genome = WingGenome(**child_fields)

        fingerprint = genome_fingerprint(genome)
        if fingerprint in fingerprints:
            continue
        design = decode_genome(genome, config, design_index)
        if guards["validate_children"]:
            validation = validate_design(design, config)
            if not validation.ok:
                continue

        thinnest = winglet_min_normal_thickness_m(design, config)
        if thinnest < float(guards["min_winglet_normal_thickness_m"]):
            if guards["reject_thin_winglet"]:
                continue
            if guards["warn_thin_winglet"]:
                print(
                    f"  WARNING {design['design_id']}: estimated thinnest winglet wall "
                    f"{thinnest * 1000:.2f} mm < {float(guards['min_winglet_normal_thickness_m']) * 1000:.2f} mm "
                    "- this case may mesh badly and diverge (see design_009 in the genesis batch)."
                )

        fingerprints.add(fingerprint)
        return {
            "design": design,
            "design_id": design["design_id"],
            "parents": list(pair),
            "crossover": choices,
            "mutations": mutations,
            "attempts": attempt,
            "winglet_min_normal_thickness_mm": round(thinnest * 1000, 3),
        }
    raise SystemExit(
        f"Could not breed a valid child from {pair} after {max_attempts} attempts. "
        "Loosen the geometry guards or mutation settings."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
def _print_selection_table(scores: list[DesignScore], cfg: dict[str, Any]) -> None:
    weights = cfg["fitness"]["weights"]
    active = [name for name, w in weights.items() if w != 0.0]
    header = f"{'design':<11}{'rank':>5}{'score':>8}  " + "".join(f"{m:>18}" for m in active)
    print("\nSelection report (raw metric values; score = weighted sum of normalised metrics)")
    print(header)
    for s in scores:
        if s.eligible:
            cells = "".join(f"{s.metrics[m]:>18.5g}" for m in active)
            role = f"  <- {s.role}" if s.role else ""
            print(f"{s.design_id:<11}{s.rank:>5}{s.score:>8.3f}  {cells}{role}")
        else:
            print(f"{s.design_id:<11}{'-':>5}{'-':>8}  INELIGIBLE: {s.reason}")


def _write_reports(
    out_dir: Path,
    source_generation: int,
    scores: list[DesignScore],
    survivors: list[str],
    pairs: list[tuple[str, str]],
    breeding_records: list[dict[str, Any]],
    seed: int,
    cfg: dict[str, Any],
) -> None:
    selection = {
        "source_generation": source_generation,
        "fitness_weights": cfg["fitness"]["weights"],
        "survivors": survivors,
        "designs": [
            {
                "design_id": s.design_id,
                "eligible": s.eligible,
                "reason": s.reason,
                "valid_cases": s.valid_cases,
                "rank": s.rank,
                "role": s.role,
                "score": s.score,
                "metrics": s.metrics,
                "normalized": s.normalized,
            }
            for s in scores
        ],
    }
    (out_dir / "selection_report.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )

    weights = cfg["fitness"]["weights"]
    active = [name for name, w in weights.items() if w != 0.0]
    with (out_dir / "selection_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["design_id", "rank", "role", "score", "eligible", "reason", *active])
        for s in scores:
            writer.writerow(
                [
                    s.design_id,
                    s.rank or "",
                    s.role,
                    f"{s.score:.6f}" if s.eligible else "",
                    s.eligible,
                    s.reason,
                    *[f"{s.metrics.get(m, ''):.6g}" if s.eligible else "" for m in active],
                ]
            )

    breeding = {
        "source_generation": source_generation,
        "random_seed": seed,
        "parent_pairs": [list(p) for p in pairs],
        "children": breeding_records,
    }
    (out_dir / "breeding_log.json").write_text(json.dumps(breeding, indent=2) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select survivors from a simulated generation and breed the next one. "
            "--generation chooses the SOURCE generation (0 = genesis-batch)."
        )
    )
    parser.add_argument("--generation", type=int, required=True, help="Source generation number (0 = genesis batch).")
    parser.add_argument("--config", type=Path, default=DEFAULT_EVOLUTION_CONFIG, help="Path to config.json.")
    parser.add_argument("--select-only", action="store_true", help="Print the selection table and exit; write nothing.")
    parser.add_argument("--force", action="store_true", help="Overwrite the output generation folder if it exists.")
    parser.add_argument("--seed", type=int, default=None, help="Override the breeding RNG seed.")
    parser.add_argument("--reference-case", type=Path, help="Path to the OpenFOAM v2412 motorBike tutorial.")
    parser.add_argument("--cores", type=int, help="Parallel ranks per case (default: cfd_config.py).")
    args = parser.parse_args()

    cfg = load_evolution_config(args.config)
    breed_generation(
        args.generation,
        cfg,
        select_only=args.select_only,
        force=args.force,
        seed=args.seed,
        reference_case_dir=args.reference_case,
        n_subdomains=args.cores,
    )


if __name__ == "__main__":
    main()
