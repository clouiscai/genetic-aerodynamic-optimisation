"""The design **genome** and its decoder.

A :class:`WingGenome` is the full set of tweakable genes for one wing. The
evolutionary process will mutate and recombine these; here we only define them,
sample them at random within :class:`~wing_generator.config.GeneBounds`, and
*decode* them into a concrete wing (the "phenotype") that the CAD generator and
the future CFD survival test consume.

Gene groups (see also ``config.GeneBounds``)
--------------------------------------------
Wing:
  * ``root_airfoil`` / ``tip_airfoil`` / ``transition_section`` — the airfoil(s).
  * ``root_pitch_deg`` — incidence at the root.
  * ``horizontal_slope[i]`` — per-section *rate of change* of the sweep offset.
  * ``vertical_slope[i]``   — per-section *rate of change* of the dihedral offset.
  * ``pitch_change_deg[i]`` — per-section change of pitch.
  * ``scale_down[i]``       — per-section chord scale-down magnitude.

Winglet (elliptical arc):
  * ``winglet_is_pointy``           — pointy tip vs flat tip.
  * ``winglet_ellipse_length``      — spanwise semi-axis (m).
  * ``winglet_ellipse_radius``      — bend semi-axis (m).
  * ``winglet_ellipse_rotation_deg``— cant of the bend (0 up, 90 outboard, 180 down).
  * ``winglet_arc_angle_deg``       — arc swept along the ellipse.
  * ``winglet_horizontal_rate``     — extra sweep slope along the winglet.
  * ``winglet_scale_rate``          — per-station chord scale-down rate.
  * ``winglet_flat_tip_chord_m``    — tip chord, used only when *flat*.
  * ``winglet_pointy_tip_x_frac``   — apex chordwise lean as a fraction of the winglet
    spanwise reach, used only when *pointy* (kept fractional so it can't spike).

Fixed (NOT genes): wing span (``config.total_span_m`` = 1.5 m) and root chord
(``config.root_chord_m`` = 1.2 m).

Wing curve reconstruction
-------------------------
``horizontal_slope`` and ``vertical_slope`` are the *derivatives* of the offset
curves. The decoder integrates them (trapezoidal rule) from the root outward, so
the offset curve is a smooth C1 spline that may contain **several inflection
points** — bounded only by the slope bounds and ``max_slope_change_per_section``.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from .config import GeneratorConfig


@dataclass(frozen=True)
class WingGenome:
    """The complete, mutable gene vector for one wing."""

    # ---- airfoil (discrete) ----
    root_airfoil: str
    tip_airfoil: str
    transition_section: int | None

    # ---- wing, global ----
    root_pitch_deg: float

    # ---- wing, per-section rate-of-change genes (length == num_sections) ----
    horizontal_slope: tuple[float, ...]
    vertical_slope: tuple[float, ...]
    pitch_change_deg: tuple[float, ...]
    scale_down: tuple[float, ...]

    # ---- winglet ----
    winglet_is_pointy: bool
    winglet_ellipse_length: float
    winglet_ellipse_radius: float
    winglet_ellipse_rotation_deg: float
    winglet_arc_angle_deg: float
    winglet_horizontal_rate: float
    winglet_scale_rate: float
    winglet_flat_tip_chord_m: float
    winglet_pointy_tip_x_frac: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────
def random_genome(rng: random.Random, config: GeneratorConfig) -> WingGenome:
    """Draw a random genome uniformly within the configured bounds.

    The per-section slope genes are generated as a bounded random walk so that the
    slope changes smoothly (``|slope[i] - slope[i-1]| <= max_slope_change_per_section``).
    This keeps the integrated curve smooth while still allowing inflection points.
    """

    g = config.genes
    n = config.num_sections
    max_dslope = config.validation.max_slope_change_per_section

    airfoils = config.gene_space.allowed_airfoils
    root_airfoil = rng.choice(airfoils)
    single = len(airfoils) == 1 or rng.random() < config.gene_space.single_airfoil_probability
    if single:
        tip_airfoil = root_airfoil
        transition_section: int | None = None
    else:
        tip_airfoil = rng.choice([a for a in airfoils if a != root_airfoil])
        transition_section = rng.randint(max(1, n // 3), min(n - 1, (n * 2) // 3))

    horizontal_slope = _random_walk(rng, n, g.horizontal_slope.as_tuple(), max_dslope)
    vertical_slope = _random_walk(rng, n, g.vertical_slope.as_tuple(), max_dslope)

    # Per-section changes (index 0 is the root and stays at its base value).
    pitch_change = (0.0,) + tuple(
        rng.uniform(*g.pitch_change_deg.as_tuple()) for _ in range(n - 1)
    )
    scale_down = (0.0,) + tuple(rng.uniform(*g.scale_down.as_tuple()) for _ in range(n - 1))

    return WingGenome(
        root_airfoil=root_airfoil,
        tip_airfoil=tip_airfoil,
        transition_section=transition_section,
        root_pitch_deg=rng.uniform(*g.root_pitch_deg.as_tuple()),
        horizontal_slope=horizontal_slope,
        vertical_slope=vertical_slope,
        pitch_change_deg=pitch_change,
        scale_down=scale_down,
        winglet_is_pointy=rng.random() < config.gene_space.winglet_pointy_probability,
        winglet_ellipse_length=rng.uniform(*g.winglet_ellipse_length.as_tuple()),
        winglet_ellipse_radius=rng.uniform(*g.winglet_ellipse_radius.as_tuple()),
        winglet_ellipse_rotation_deg=rng.uniform(*g.winglet_ellipse_rotation_deg.as_tuple()),
        winglet_arc_angle_deg=rng.uniform(*g.winglet_arc_angle_deg.as_tuple()),
        winglet_horizontal_rate=rng.uniform(*g.winglet_horizontal_rate.as_tuple()),
        winglet_scale_rate=rng.uniform(*g.winglet_scale_rate.as_tuple()),
        winglet_flat_tip_chord_m=rng.uniform(*g.winglet_flat_tip_chord_m.as_tuple()),
        winglet_pointy_tip_x_frac=rng.uniform(*g.winglet_pointy_tip_x_frac.as_tuple()),
    )


def _random_walk(
    rng: random.Random, n: int, bounds: tuple[float, float], max_step: float
) -> tuple[float, ...]:
    lo, hi = bounds
    value = rng.uniform(lo, hi)
    out = [value]
    for _ in range(n - 1):
        value = _clamp(value + rng.uniform(-max_step, max_step), lo, hi)
        out.append(value)
    return tuple(round(v, 6) for v in out)


# ─────────────────────────────────────────────────────────────────────────────
# Decoding (genome → phenotype / design dict)
# ─────────────────────────────────────────────────────────────────────────────
def decode_genome(
    genome: WingGenome, config: GeneratorConfig, design_index: int
) -> dict[str, Any]:
    """Decode a genome into a concrete design dict (sections + winglet).

    The returned schema is what ``geometry_generator`` and the downstream CFD-prep
    pipeline consume, plus a ``genome`` block carrying the raw genes for traceability
    and for the evolutionary loop.
    """

    n = config.num_sections
    span_m = config.total_span_m
    root_chord = config.root_chord_m
    scale_min = config.scale_min
    twist_limit = config.validation.max_total_pitch_twist_deg

    spans = [span_m * i / (n - 1) for i in range(n)]
    ds = span_m / (n - 1)

    smoothing_passes = max(0, config.validation.section_curve_smoothing_passes)

    # Integrate per-section slopes (trapezoidal) into smooth offset curves.
    horizontal = _smooth_curve(_integrate_slopes(genome.horizontal_slope, ds), smoothing_passes)
    vertical = _smooth_curve(_integrate_slopes(genome.vertical_slope, ds), smoothing_passes)
    horizontal[0] = 0.0
    vertical[0] = 0.0

    # Section airfoils.
    if genome.transition_section is None:
        section_airfoils = [genome.root_airfoil] * n
    else:
        section_airfoils = [
            genome.root_airfoil if i < genome.transition_section else genome.tip_airfoil
            for i in range(n)
        ]

    raw_pitches: list[float] = []
    raw_scales: list[float] = []
    pitch = genome.root_pitch_deg
    scale = 1.0
    for i in range(n):
        if i > 0:
            pitch = _clamp(
                pitch + genome.pitch_change_deg[i],
                genome.root_pitch_deg - twist_limit,
                genome.root_pitch_deg + twist_limit,
            )
            scale = max(scale_min, scale - genome.scale_down[i])
        raw_pitches.append(pitch)
        raw_scales.append(scale)

    pitches = _smooth_curve(raw_pitches, smoothing_passes)
    pitches[0] = genome.root_pitch_deg
    for i, value in enumerate(pitches):
        pitches[i] = _clamp(
            value,
            genome.root_pitch_deg - twist_limit,
            genome.root_pitch_deg + twist_limit,
        )

    scales = _smooth_monotone_decreasing(raw_scales, scale_min, smoothing_passes)
    scales[0] = 1.0

    sections: list[dict[str, Any]] = []
    for i in range(n):
        pitch = pitches[i]
        scale = scales[i]
        chord = root_chord * scale
        sections.append(
            {
                "section_index": i,
                "airfoil_type": section_airfoils[i],
                "spanwise_position": round(spans[i], 6),
                "section_scale": round(scale, 6),
                "chord_length": round(chord, 6),
                "root_pitch_angle_deg": round(pitch, 6),
                "horizontal_displacement": round(0.0 if i == 0 else horizontal[i], 6),
                "vertical_displacement": round(0.0 if i == 0 else vertical[i], 6),
            }
        )

    winglet = {
        "shape": "elliptical_arc",
        "is_pointy": genome.winglet_is_pointy,
        "ellipse_length": round(genome.winglet_ellipse_length, 6),
        "ellipse_radius": round(genome.winglet_ellipse_radius, 6),
        "ellipse_rotation_deg": round(genome.winglet_ellipse_rotation_deg, 6),
        "arc_angle_deg": round(genome.winglet_arc_angle_deg, 6),
        "horizontal_rate": round(genome.winglet_horizontal_rate, 6),
        "scale_rate": round(genome.winglet_scale_rate, 6),
        "flat_tip_chord_m": round(genome.winglet_flat_tip_chord_m, 6),
        "pointy_tip_x_frac": round(genome.winglet_pointy_tip_x_frac, 6),
        "tip_thickness_m": config.winglet_tip_thickness_m,
        "min_tip_chord_m": config.winglet_min_tip_chord_m,
        "min_tip_thickness_m": config.winglet_min_tip_thickness_m,
    }

    transitions = 0 if genome.transition_section is None else 1
    return {
        "design_id": f"design_{design_index:03d}",
        "units": {"length": "m", "angle": "deg", "forgecad_length": "mm"},
        "fixed": {"wing_span_m": span_m, "root_chord_m": root_chord},
        "total_span": span_m,
        "num_sections": n,
        "genome": genome.to_dict(),
        "airfoil_population": list(config.gene_space.allowed_airfoils),
        "airfoil_coordinate_source": "UIUC Airfoil Coordinates Database",
        "airfoil_profile_rules": {
            "profile_point_count": config.airfoil_points_per_profile,
            "max_profiles_per_wing": 2,
            "single_profile_allowed": True,
            "selected_profiles": list(dict.fromkeys(section_airfoils)),
            "transition_count": transitions,
            "transition_section_index": genome.transition_section,
        },
        "sections": sections,
        "wingtip_sections": config.wingtip_sections,
        "wingtip": winglet,
        "forgecad": {
            "mesh_edge_length_mm": config.forgecad.mesh_edge_length_mm,
            "mesh_samples": config.forgecad.mesh_samples,
            "bounds_padding_mm": config.forgecad.bounds_padding_mm,
        },
    }


def _integrate_slopes(slopes: tuple[float, ...], ds: float) -> list[float]:
    """Trapezoidal integral of per-section slopes → offset curve (root offset = 0)."""
    offsets = [0.0]
    for i in range(1, len(slopes)):
        offsets.append(offsets[-1] + 0.5 * (slopes[i - 1] + slopes[i]) * ds)
    return offsets


def _smooth_curve(values: list[float], passes: int) -> list[float]:
    """Repeated binomial smoothing, preserving endpoints."""

    if len(values) < 3 or passes <= 0:
        return list(values)
    out = list(values)
    first = out[0]
    last = out[-1]
    for _ in range(passes):
        next_values = list(out)
        for i in range(1, len(out) - 1):
            next_values[i] = 0.25 * out[i - 1] + 0.5 * out[i] + 0.25 * out[i + 1]
        next_values[0] = first
        next_values[-1] = last
        out = next_values
    return out


def _smooth_monotone_decreasing(values: list[float], minimum: float, passes: int) -> list[float]:
    if len(values) < 3 or passes <= 0:
        return [_clamp(v, minimum, 1.0) for v in values]

    out = _smooth_curve(values, passes)
    out[0] = 1.0
    out[-1] = _clamp(out[-1], minimum, out[-2] if len(out) > 1 else 1.0)
    for i in range(1, len(out)):
        out[i] = _clamp(out[i], minimum, out[i - 1])
    for i in range(len(out) - 2, -1, -1):
        out[i] = max(out[i], out[i + 1])
    out[0] = 1.0
    return out


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def genome_fingerprint(genome: WingGenome) -> str:
    """Stable, low-precision fingerprint used to reject duplicate designs."""
    import json

    compact = {
        "af": (genome.root_airfoil, genome.tip_airfoil, genome.transition_section),
        "rp": round(genome.root_pitch_deg, 3),
        "hs": [round(v, 3) for v in genome.horizontal_slope],
        "vs": [round(v, 3) for v in genome.vertical_slope],
        "pc": [round(v, 3) for v in genome.pitch_change_deg],
        "sd": [round(v, 4) for v in genome.scale_down],
        "wg": [
            genome.winglet_is_pointy,
            round(genome.winglet_ellipse_length, 3),
            round(genome.winglet_ellipse_radius, 3),
            round(genome.winglet_ellipse_rotation_deg, 1),
            round(genome.winglet_arc_angle_deg, 1),
            round(genome.winglet_horizontal_rate, 3),
            round(genome.winglet_scale_rate, 3),
            round(genome.winglet_flat_tip_chord_m, 3),
            round(genome.winglet_pointy_tip_x_frac, 3),
        ],
    }
    return json.dumps(compact, sort_keys=True)
