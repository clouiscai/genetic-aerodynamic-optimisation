"""Generate and validate a population of wing designs.

Each design is produced by sampling a :class:`~wing_generator.genome.WingGenome`
(``genome.random_genome``) and decoding it into a phenotype (``genome.decode_genome``).
The phenotype is then checked by :func:`validate_design`, which guards geometry
quality (smooth curves, monotonic span, well-behaved winglet) before the expensive
CAD step. The same validation re-simulates the winglet stations so they exactly
mirror ``geometry_generator``'s ForgeCAD code.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from math import cos, pi, sin
from pathlib import Path
from typing import Any

from .airfoils import airfoil_names, normalized_airfoil
from .config import GeneratorConfig
from .genome import decode_genome, genome_fingerprint, random_genome


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: tuple[str, ...] = ()


class PopulationGenerator:
    """Samples unique, valid designs from the genome search space."""

    def __init__(self, config: GeneratorConfig):
        unknown = set(config.allowed_airfoils) - set(airfoil_names())
        if unknown:
            raise ValueError(f"Unknown airfoil(s): {', '.join(sorted(unknown))}")
        if not config.allowed_airfoils:
            raise ValueError("At least one allowed airfoil is required")
        self.config = config
        self.rng = random.Random(config.random_seed)
        self._fingerprints: set[str] = set()

    def next_valid_design(self, design_index: int, max_attempts: int = 200) -> dict[str, Any]:
        for _ in range(max_attempts):
            genome = random_genome(self.rng, self.config)
            fingerprint = genome_fingerprint(genome)
            if fingerprint in self._fingerprints:
                continue
            design = decode_genome(genome, self.config, design_index)
            validation = validate_design(design, self.config)
            if not validation.ok:
                continue
            self._fingerprints.add(fingerprint)
            return design
        raise RuntimeError(
            f"Could not generate a valid unique design_{design_index:03d} "
            f"after {max_attempts} attempts"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Winglet station re-simulation (must mirror geometry_generator.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
def winglet_stations(
    sections: list[dict[str, Any]], wingtip: dict[str, Any], config: GeneratorConfig
) -> list[dict[str, Any]]:
    """Reproduce the winglet cross-sections the CAD loft will create.

    Elliptical arc swept through ``phi in [0, arc]``:
      * ``span_offset = ellipse_length * sin(phi)``
      * ``bend = ellipse_radius * (1 - cos phi)`` (zero initial slope ⇒ C1 with wing)
      * the bend is rotated by ``ellipse_rotation`` into vertical & horizontal parts
      * the wing's exit sweep/dihedral slopes are continued for a smooth join
      * chord scales down per station at ``scale_rate``; pointy → point (with apex
        shifted to ``pointy_tip_x``), flat → stops at ``flat_tip_chord``.
    """

    n = config.num_sections
    w = config.wingtip_sections
    root_chord = config.root_chord_m
    f, p = sections[-1], sections[-2]
    ds = max(f["spanwise_position"] - p["spanwise_position"], 1e-6)

    sweep_slope = (f["horizontal_displacement"] - p["horizontal_displacement"]) / ds
    dihedral_slope = (f["vertical_displacement"] - p["vertical_displacement"]) / ds

    length = float(wingtip["ellipse_length"])
    radius = float(wingtip["ellipse_radius"])
    theta = float(wingtip["ellipse_rotation_deg"]) * pi / 180.0
    arc = abs(float(wingtip["arc_angle_deg"])) * pi / 180.0
    h_rate = float(wingtip["horizontal_rate"])
    s_rate = float(wingtip["scale_rate"])
    is_pointy = bool(wingtip["is_pointy"])
    flat_tip_chord = float(wingtip["flat_tip_chord_m"])
    pointy_tip_x_frac = float(wingtip["pointy_tip_x_frac"])
    span_reach = length * sin(arc)  # total winglet spanwise reach
    min_tip_chord = float(wingtip["min_tip_chord_m"])
    min_tip_thickness = float(wingtip.get("min_tip_thickness_m", config.winglet_min_tip_thickness_m))
    tip_thickness = float(wingtip.get("tip_thickness_m", config.winglet_tip_thickness_m))

    final_chord = f["chord_length"]
    final_coords = normalized_airfoil(
        f["airfoil_type"], config.airfoil_points_per_profile
    )
    base_profile_thickness = max(y for _, y in final_coords) - min(y for _, y in final_coords)
    base_thickness = max(base_profile_thickness * final_chord, 1e-9)
    stations: list[dict[str, Any]] = []
    for i in range(1, w + 1):
        t = i / w
        phi = arc * t
        one_minus_cos = 1.0 - cos(phi)
        eased_t = 3.0 * t * t - 2.0 * t * t * t
        span_offset = length * sin(phi)
        bend = radius * one_minus_cos
        vertical_bend = cos(theta) * bend
        horizontal_bend = sin(theta) * bend
        # Extra winglet sweep with a zero-initial-slope (1 - cos) ramp ⇒ C1 at the join.
        horizontal_rate_bend = h_rate * length * one_minus_cos
        apex = pointy_tip_x_frac * span_reach * eased_t if is_pointy else 0.0

        chord_factor = (1.0 - s_rate) ** i
        if is_pointy:
            # Normalised taper that always reaches the point (min_tip_chord) at the tip.
            raw_end = (1.0 - s_rate) ** w
            norm = (chord_factor - raw_end) / max(1e-6, 1.0 - raw_end)
            chord = min_tip_chord + (final_chord - min_tip_chord) * norm
        else:
            chord = max(flat_tip_chord, final_chord * chord_factor)

        stations.append(
            {
                "section_index": n - 1 + i,
                "airfoil_type": f["airfoil_type"],
                "spanwise_position": f["spanwise_position"] + span_offset,
                "chord_length": chord,
                "section_scale": chord / root_chord,
                "root_pitch_angle_deg": f["root_pitch_angle_deg"],
                "target_thickness_m": max(
                    min_tip_thickness,
                    base_thickness + (tip_thickness - base_thickness) * eased_t,
                ),
                "horizontal_displacement": (
                    f["horizontal_displacement"]
                    + sweep_slope * span_offset
                    + horizontal_rate_bend
                    + horizontal_bend
                    + apex
                ),
                "vertical_displacement": (
                    f["vertical_displacement"] + dihedral_slope * span_offset + vertical_bend
                ),
            }
        )
    return stations


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
def validate_design(design: dict[str, Any], config: GeneratorConfig) -> ValidationResult:
    errors: list[str] = []
    g = config.genes
    v = config.validation
    sections = design.get("sections", [])

    if design.get("num_sections") != config.num_sections:
        errors.append("metadata section count does not match config")
    if len(sections) != config.num_sections:
        errors.append(f"expected {config.num_sections} sections, found {len(sections)}")
    if abs(float(design.get("total_span", -1)) - config.total_span_m) > v.span_tolerance_m:
        errors.append("total span does not match the fixed wing span")

    positions = [s["spanwise_position"] for s in sections]
    if positions and abs(positions[0]) > v.span_tolerance_m:
        errors.append("root section is not at y=0")
    if positions and abs(positions[-1] - config.total_span_m) > v.span_tolerance_m:
        errors.append("tip section is not at the fixed wing span")
    if any(b <= a for a, b in zip(positions, positions[1:])):
        errors.append("section spanwise positions are not strictly increasing")

    # Per-section geometry sanity + smoothness.
    for s in sections:
        if s["chord_length"] <= 0:
            errors.append(f"section {s['section_index']}: chord must be positive")
        if not (config.scale_min - 1e-6 <= s["section_scale"] <= 1.0 + 1e-6):
            errors.append(f"section {s['section_index']}: scale out of [{config.scale_min}, 1]")
        if s["airfoil_type"] not in config.allowed_airfoils:
            errors.append(f"section {s['section_index']}: unknown airfoil {s['airfoil_type']}")

    for a, b in zip(sections, sections[1:]):
        if abs(b["root_pitch_angle_deg"] - a["root_pitch_angle_deg"]) > v.max_pitch_delta_deg_per_section:
            errors.append(f"pitch jump between sections {a['section_index']} and {b['section_index']}")
        frac = abs(b["chord_length"] - a["chord_length"]) / max(a["chord_length"], 1e-9)
        if frac > v.max_chord_delta_fraction_per_section:
            errors.append(f"chord jump between sections {a['section_index']} and {b['section_index']}")
        if abs(b["horizontal_displacement"] - a["horizontal_displacement"]) > v.max_horizontal_delta_m_per_section:
            errors.append(f"sweep jump between sections {a['section_index']} and {b['section_index']}")
        if abs(b["vertical_displacement"] - a["vertical_displacement"]) > v.max_vertical_delta_m_per_section:
            errors.append(f"dihedral jump between sections {a['section_index']} and {b['section_index']}")

    if sections:
        pitches = [s["root_pitch_angle_deg"] for s in sections]
        if max(pitches) - min(pitches) > v.max_total_pitch_twist_deg + 1e-6:
            errors.append("total pitch twist exceeds the configured limit")

    # Winglet gene bounds.
    if design.get("wingtip_sections") != config.wingtip_sections:
        errors.append("wingtip section count metadata does not match config")
    wingtip = design.get("wingtip", {})
    _check_bound(errors, "winglet ellipse length", wingtip.get("ellipse_length"), g.winglet_ellipse_length)
    _check_bound(errors, "winglet ellipse radius", wingtip.get("ellipse_radius"), g.winglet_ellipse_radius)
    _check_bound(errors, "winglet ellipse rotation", wingtip.get("ellipse_rotation_deg"), g.winglet_ellipse_rotation_deg)
    _check_bound(errors, "winglet arc angle", wingtip.get("arc_angle_deg"), g.winglet_arc_angle_deg)
    _check_bound(errors, "winglet horizontal rate", wingtip.get("horizontal_rate"), g.winglet_horizontal_rate)
    _check_bound(errors, "winglet scale rate", wingtip.get("scale_rate"), g.winglet_scale_rate)
    if abs(float(wingtip.get("arc_angle_deg", 0.0))) > 90.0 + 1e-6:
        errors.append("winglet arc angle must be ≤ 90° for monotonic loft stations")

    # Winglet smoothness (re-simulated stations).
    if len(sections) >= 2:
        stations = [sections[-1]] + winglet_stations(sections, wingtip, config)
        for a, b in zip(stations, stations[1:]):
            if abs(b["horizontal_displacement"] - a["horizontal_displacement"]) > v.wingtip_max_horizontal_delta_m_per_section:
                errors.append(
                    f"winglet sweep jump between stations {a['section_index']} and {b['section_index']}"
                )
            if abs(b["vertical_displacement"] - a["vertical_displacement"]) > v.wingtip_max_vertical_delta_m_per_section:
                errors.append(
                    f"winglet dihedral jump between stations {a['section_index']} and {b['section_index']}"
                )
            if abs(b["section_scale"] - a["section_scale"]) > v.wingtip_max_scale_delta_per_section:
                errors.append(
                    f"winglet scale jump between stations {a['section_index']} and {b['section_index']}"
                )

    # Loft-quality guard (prevents the wrap/fan "360 twist"): across the FULL set of
    # loft profiles (wing sections + winglet stations), no profile may shift laterally
    # (sweep + dihedral) by more than `max_loft_lateral_move_chord_ratio` of its own
    # chord between consecutive stations, or the ruled surface folds over itself.
    if len(sections) >= 2:
        all_profiles = sections + winglet_stations(sections, wingtip, config)
        for a, b in zip(all_profiles, all_profiles[1:]):
            dh = b["horizontal_displacement"] - a["horizontal_displacement"]
            dz = b["vertical_displacement"] - a["vertical_displacement"]
            lateral_move = (dh * dh + dz * dz) ** 0.5
            local_chord = max(min(a["chord_length"], b["chord_length"]), 1e-6)
            if lateral_move > v.max_loft_lateral_move_chord_ratio * local_chord:
                errors.append(
                    f"loft would wrap/twist between stations {a['section_index']} and "
                    f"{b['section_index']}: lateral move {lateral_move * 1000:.0f} mm exceeds "
                    f"{v.max_loft_lateral_move_chord_ratio:g} x chord {local_chord * 1000:.0f} mm"
                )
                break

    return ValidationResult(ok=not errors, errors=tuple(errors))


def _check_bound(errors: list[str], label: str, value: Any, bound) -> None:
    if value is None or not bound.contains(float(value)):
        errors.append(f"{label} out of bounds")


def write_population_manifest(
    output_dir: Path, designs: list[dict[str, Any]], config: GeneratorConfig
) -> None:
    manifest = {
        "num_designs": len(designs),
        "requested_num_designs": config.num_designs,
        "random_seed": config.random_seed,
        "wing_span_m": config.total_span_m,
        "root_chord_m": config.root_chord_m,
        "num_sections": config.num_sections,
        "wingtip_sections": config.wingtip_sections,
        "design_ids": [d["design_id"] for d in designs],
        "pointy_winglets": sum(1 for d in designs if d["wingtip"]["is_pointy"]),
    }
    (output_dir / "population_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
