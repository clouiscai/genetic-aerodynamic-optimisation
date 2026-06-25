"""Human-readable report of the design genome and all tunable settings.

Generated directly from :class:`~wing_generator.config.GeneratorConfig`, so it
always reflects the live bounds, evolution settings and fitness weights.
"""

from __future__ import annotations

from .config import GeneratorConfig


def _row(name: str, rng: str, meaning: str) -> str:
    return f"| {name} | {rng} | {meaning} |"


def build_parameter_report(config: GeneratorConfig) -> str:
    g = config.genes
    gs = config.gene_space
    v = config.validation
    evo = config.evolution
    fit = config.fitness

    def rng(bounds) -> str:
        lo, hi = bounds.as_tuple()
        return f"{lo:g} … {hi:g}"

    lines: list[str] = []
    add = lines.append

    add("# Wing Design Genome & Configuration Report")
    add("")
    add(
        "Each wing is a **genome** (`wing_generator/genome.py`) decoded into geometry. "
        "The evolutionary optimiser will mutate and recombine these genes; the CFD "
        "**survival test** (not yet implemented) will score them. Everything below is "
        "tuned in `wing_generator/config.py`. Lengths are metres, angles degrees."
    )
    add("")

    # ---- Fixed -------------------------------------------------------------
    add("## 1. Fixed airframe (NOT genes)")
    add("")
    add("| Setting | Value | Meaning |")
    add("| --- | --- | --- |")
    add(f"| `total_span_m` | {config.total_span_m:g} | wing span, root → tip, excludes winglet |")
    add(f"| `root_chord_m` | {config.root_chord_m:g} | chord at the root; taper scales down from here |")
    add(f"| `num_sections` | {config.num_sections} | wing cross-sections |")
    add(f"| `wingtip_sections` | {config.wingtip_sections} | winglet cross-sections |")
    add(f"| `airfoils` | {', '.join(gs.allowed_airfoils)} | available profiles |")
    add(f"| `scale_min` | {config.scale_min:g} | chord-scale floor (max taper) |")
    add(f"| `winglet_tip_thickness_m` | {config.winglet_tip_thickness_m:g} | thin tip edge thickness |")
    add(f"| `winglet_min_tip_chord_m` | {config.winglet_min_tip_chord_m:g} | loft-safety floor, tip chord |")
    add("")

    # ---- Wing genes --------------------------------------------------------
    add("## 2. Wing genes")
    add("")
    add(
        "The sweep (horizontal) and dihedral (vertical) curves are stored as **per-section "
        "rates of change of offset** (local slopes). The decoder integrates them from the "
        "root, so each curve is a smooth spline that may carry **several inflection points** — "
        f"bounded by the slope ranges and by `max_slope_change_per_section = {v.max_slope_change_per_section:g}`."
    )
    add("")
    add("| Gene | Range / choices | Meaning |")
    add("| --- | --- | --- |")
    add(_row("`airfoil`", f"{len(gs.allowed_airfoils)} profiles", f"single profile w.p. {gs.single_airfoil_probability:g}, else 2 with one transition"))
    add(_row("`root_pitch_deg`", rng(g.root_pitch_deg), "incidence at the root"))
    add(_row("`horizontal_slope[i]`", rng(g.horizontal_slope), "per-section rate of change of sweep offset"))
    add(_row("`vertical_slope[i]`", rng(g.vertical_slope), "per-section rate of change of dihedral offset"))
    add(_row("`pitch_change_deg[i]`", rng(g.pitch_change_deg), "per-section change of pitch"))
    add(_row("`scale_down[i]`", rng(g.scale_down), "per-section chord scale-down magnitude"))
    add("")

    # ---- Winglet genes -----------------------------------------------------
    pct = round(gs.winglet_pointy_probability * 100)
    add("## 3. Winglet genes (elliptical arc)")
    add("")
    add(
        "The winglet is an elliptical arc joined C1-smoothly to the wing. It is "
        f"**{pct}% pointy** (tapers to a point) or **{100 - pct}% flat** (ends at a tip chord)."
    )
    add("")
    add("```")
    add("span_offset = ellipse_length · sin(phi)")
    add("bend        = ellipse_radius · (1 - cos phi)          # zero start slope -> C1")
    add("vertical    = dihedral_line + cos(rotation)·bend")
    add("horizontal  = sweep_line + sin(rotation)·bend + horizontal_rate·length·(1-cos phi) + apex")
    add("chord       = chord · (1 - scale_rate)^station        # pointy->point, flat->tip chord")
    add("# every added term has zero slope at phi=0 ⇒ C1-smooth join (no crease)")
    add("```")
    add("")
    add("| Gene | Range / choices | Meaning |")
    add("| --- | --- | --- |")
    add(_row("`is_pointy`", f"{pct}% / {100 - pct}%", "pointy tip vs flat tip"))
    add(_row("`ellipse_length`", rng(g.winglet_ellipse_length), "spanwise semi-axis"))
    add(_row("`ellipse_radius`", rng(g.winglet_ellipse_radius), "bend semi-axis"))
    add(_row("`ellipse_rotation_deg`", rng(g.winglet_ellipse_rotation_deg), "cant of bend: 0 up, 90 outboard, 180 down"))
    add(_row("`arc_angle_deg`", rng(g.winglet_arc_angle_deg), "arc swept along the ellipse"))
    add(_row("`horizontal_rate`", rng(g.winglet_horizontal_rate), "extra sweep slope along the winglet"))
    add(_row("`scale_rate`", rng(g.winglet_scale_rate), "per-station chord scale-down rate"))
    add(_row("`flat_tip_chord_m`", rng(g.winglet_flat_tip_chord_m), "tip chord — flat winglets only"))
    add(_row("`pointy_tip_x_frac`", rng(g.winglet_pointy_tip_x_frac), "apex chordwise lean as fraction of winglet reach — pointy only"))
    add("")

    # ---- Validation --------------------------------------------------------
    add("## 4. Geometry-quality guards")
    add("")
    add("| Guard | Limit |")
    add("| --- | --- |")
    add(f"| max total twist | {v.max_total_pitch_twist_deg:g}° |")
    add(f"| max pitch step / section | {v.max_pitch_delta_deg_per_section:g}° |")
    add(f"| max chord change / section | {v.max_chord_delta_fraction_per_section:g} |")
    add(f"| max sweep step / section | {v.max_horizontal_delta_m_per_section:g} m |")
    add(f"| max dihedral step / section | {v.max_vertical_delta_m_per_section:g} m |")
    add(f"| max slope change / section | {v.max_slope_change_per_section:g} |")
    add(f"| max winglet sweep / dihedral / scale step | {v.wingtip_max_horizontal_delta_m_per_section:g} m / {v.wingtip_max_vertical_delta_m_per_section:g} m / {v.wingtip_max_scale_delta_per_section:g} |")
    add("")

    # ---- Evolution (not active yet) ----------------------------------------
    add("## 5. Evolution settings — *defined, not active yet*")
    add("")
    add("Wired in after review, once the CFD survival test exists.")
    add("")
    add("| Setting | Value |")
    add("| --- | --- |")
    add(f"| `population_size` | {evo.population_size} |")
    add(f"| `generations` | {evo.generations} |")
    add(f"| `elitism_fraction` | {evo.elitism_fraction:g} |")
    add(f"| `crossover_rate` (random assortment) | {evo.crossover_rate:g} |")
    add(f"| `mutation_rate` | {evo.mutation_rate:g} |")
    add(f"| `mutation_scale` | {evo.mutation_scale:g} |")
    add(f"| `tournament_size` | {evo.tournament_size} |")
    add("")
    add("Gene-group dominance weights (relative inheritance during assortment):")
    add("")
    add("| Gene group | Weight |")
    add("| --- | --- |")
    for group, weight in evo.gene_group_dominance.items():
        add(f"| `{group}` | {weight:g} |")
    add("")

    # ---- Fitness (CFD survival test, not implemented) ----------------------
    add("## 6. Fitness / CFD survival test — *not implemented yet*")
    add("")
    add(
        "The CFD evaluation is the next stage (after review). These weights define how "
        f"its metrics combine into one score (maximize = {fit.maximize})."
    )
    add("")
    add("| Metric | Weight |")
    add("| --- | --- |")
    for metric, weight in fit.metric_weights.items():
        add(f"| `{metric}` | {weight:g} |")
    add("")
    add("_Generated from `GeneratorConfig`; edit `config.py` and re-run to update._")
    add("")

    return "\n".join(lines)
