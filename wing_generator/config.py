"""Central configuration — the single file you edit by hand.

Everything the system does is driven from :class:`GeneratorConfig`:

* **Fixed airframe constants** (``WING_SPAN_M``, ``ROOT_CHORD_M``) — never genes.
* **Gene bounds** (:class:`GeneBounds`) — the search space the population generator
  samples and the evolutionary process will later mutate. Tighten a bound to
  constrain that gene; e.g. shrinking the *slope* bounds limits how many inflection
  points a wing curve can develop.
* **Gene space** (:class:`GeneSpace`) — discrete/structural choices (airfoils, pointy
  probability).
* **Validation** (:class:`ValidationSettings`) — geometry-quality guards.
* **Evolution** (:class:`EvolutionConfig`) — population size, dominance weighting,
  assortment/mutation. *Defined now, not executed yet* (wired in after review).
* **Fitness** (:class:`FitnessConfig`) — how CFD "survival test" metrics combine.
  *The CFD itself is intentionally not implemented yet.*
* **ForgeCAD** (:class:`ForgeCadSettings`) — CAD/preview CLI options.

See ``genome.py`` for how these genes are decoded into a wing, and
``PARAMETER_REPORT.md`` (generated each run) for a human-readable gene list.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

from .airfoils import AIRFOIL_POINT_COUNT


# ─────────────────────────────────────────────────────────────────────────────
# Fixed airframe constants. These are NOT genes — the search never changes them.
# ─────────────────────────────────────────────────────────────────────────────
WING_SPAN_M = 1.5      # root → wing tip, EXCLUDING the winglet
ROOT_CHORD_M = 1.2     # chord at the root section (taper scales down from here)


@dataclass(frozen=True)
class Bounds:
    """Closed numeric bounds for random sampling and validation."""

    minimum: float
    maximum: float

    def contains(self, value: float, tolerance: float = 1e-9) -> bool:
        return self.minimum - tolerance <= value <= self.maximum + tolerance

    def as_tuple(self) -> tuple[float, float]:
        return self.minimum, self.maximum

    @property
    def span(self) -> float:
        return self.maximum - self.minimum


@dataclass(frozen=True)
class GeneBounds:
    """Inclusive bounds for every continuous gene.

    The wing's sweep (horizontal) and dihedral (vertical) curves are stored in the
    genome as *per-section rates of change* (local slopes). The phenotype curve is
    rebuilt by integrating those slopes, so it behaves like a Bezier that may carry
    **more than one inflection point**. How wavy it can get is governed entirely by
    the slope bounds below and ``ValidationSettings.max_slope_change_per_section``.
    """

    # ---- Wing, global -------------------------------------------------------
    root_pitch_deg: Bounds = Bounds(-2.0, 5.0)               # incidence at the root

    # ---- Wing, per-section rate-of-change genes -----------------------------
    horizontal_slope: Bounds = Bounds(-0.3, 0.3)           # d(x_offset)/d(span)
    vertical_slope: Bounds = Bounds(-0.15, 0.15)             # d(z_offset)/d(span)
    pitch_change_deg: Bounds = Bounds(-0.50, 0.50)           # Δpitch per section
    scale_down: Bounds = Bounds(0.0, 0.05)                   # chord-scale drop per section

    # ---- Winglet — elliptical arc -------------------------------------------
    winglet_ellipse_length: Bounds = Bounds(0.10, 0.40)      # spanwise semi-axis (m)
    winglet_ellipse_radius: Bounds = Bounds(0.10, 0.40)      # bend semi-axis (m)
    winglet_ellipse_rotation_deg: Bounds = Bounds(0.0, 180.0)  # 0=up, 90=outboard, 180=down
    winglet_arc_angle_deg: Bounds = Bounds(10.0, 90.0)       # arc swept along the ellipse
    winglet_horizontal_rate: Bounds = Bounds(-0.30, 0.30)    # extra sweep slope along winglet
    winglet_scale_rate: Bounds = Bounds(0.01, 0.1)           # chord scale-down rate per station
    winglet_flat_tip_chord_m: Bounds = Bounds(0.05, 0.40)    # tip chord (flat winglets)
    # Pointy apex chordwise lean, as a FRACTION of the winglet's spanwise reach.
    # Keeping it a fraction (not absolute metres) prevents a small winglet from being
    # shifted into a thin sideways spike. ±1.0 ⇒ at most a 45° lean.
    winglet_pointy_tip_x_frac: Bounds = Bounds(-1.0, 1.0)


@dataclass(frozen=True)
class GeneSpace:
    """Discrete / structural gene choices."""

    allowed_airfoils: tuple[str, ...] = (
        "ClarkY",
        "NACA4412",
        "Eppler817",
        "Eppler387",
    )
    # Probability a wing uses a single airfoil (1.0 = always single profile).
    single_airfoil_probability: float = 1.0
    # Probability a winglet is "pointy" (tapers to a point); otherwise "flat".
    winglet_pointy_probability: float = 0.50


@dataclass(frozen=True)
class ValidationSettings:
    """Geometry-quality guards applied to the decoded phenotype before CAD export."""

    span_tolerance_m: float = 1e-9
    max_total_pitch_twist_deg: float = 5.0
    max_pitch_delta_deg_per_section: float = 5.0
    max_chord_delta_fraction_per_section: float = 0.30
    max_horizontal_delta_m_per_section: float = 0.20
    max_vertical_delta_m_per_section: float = 0.50
    # THE smoothness knob. Limits how fast a wing curve's slope may change per section
    # (|slope[i] - slope[i-1]|). The slope is a random walk with this as its step, so a
    # LARGE value lets the slope swing its whole range in ~1 section → a pleated /
    # rippled (accordion) wing that looks "twisted/broken". A small value drifts slowly
    # → a smooth wing with at most 1-2 gentle inflections. Raise it for wilder shapes.
    max_slope_change_per_section: float = 0.03
    # Loft-quality guard: reject any design where a profile shifts laterally (sweep +
    # dihedral) by more than this fraction of its own chord between consecutive loft
    # stations. Beyond ~0.5 the ruled surface wraps/fans and self-intersects (the
    # "360 twist"); 0.35 keeps a safe margin so every accepted model lofts smoothly.
    max_loft_lateral_move_chord_ratio: float = 0.35
    # Winglet smoothness guards.
    wingtip_max_horizontal_delta_m_per_section: float = 0.15
    wingtip_max_vertical_delta_m_per_section: float = 0.12
    wingtip_max_scale_delta_per_section: float = 0.30
    # Phenotype smoothing: repeated [1,2,1] filtering on decoded section curves.
    # This keeps the genetic variation while removing visual kinks between adjacent
    # loft stations in Fusion/mesh imports.
    section_curve_smoothing_passes: int = 4


@dataclass(frozen=True)
class EvolutionConfig:
    """Settings for the evolutionary loop.

    NOT executed yet — these are defined now so the genome, batches and reports are
    ready for the optimiser that follows the CFD survival test (after review).
    """

    population_size: int = 10
    generations: int = 0                 # 0 = generate the initial population only
    elitism_fraction: float = 0.10        # fittest fraction copied unchanged
    crossover_rate: float = 0.70          # chance two parents' genes are assorted
    mutation_rate: float = 0.15           # per-gene mutation probability
    mutation_scale: float = 0.10          # mutation step as a fraction of each gene's range
    tournament_size: int = 3              # selection pressure
    # Relative "dominance" weight of each gene group during crossover/assortment.
    # Higher weight ⇒ that group is more likely to be inherited intact.
    gene_group_dominance: dict[str, float] = field(
        default_factory=lambda: {
            "airfoil": 1.0,
            "wing_pitch": 1.0,
            "wing_horizontal": 1.0,
            "wing_vertical": 1.0,
            "wing_scale": 1.0,
            "winglet": 1.0,
        }
    )


@dataclass(frozen=True)
class FitnessConfig:
    """How CFD 'survival test' metrics combine into one fitness score.

    The CFD evaluation itself is intentionally NOT implemented yet (it is the next
    stage, after review). These weights are the contract the survival test will use.
    """

    # Weighted sum of (normalised) metrics. Positive weight rewards, negative penalises.
    metric_weights: dict[str, float] = field(
        default_factory=lambda: {
            "lift_to_drag": 1.0,
            "lift": 0.0,
            "drag": -1.0,
            "structural_mass": 0.0,
            "stall_margin": 0.0,
        }
    )
    maximize: bool = True  # higher weighted score = fitter


@dataclass(frozen=True)
class ForgeCadSettings:
    """ForgeCAD CLI settings."""

    backend: str = "manifold"
    mesh_edge_length_mm: float = 20.0
    mesh_samples: int = 96
    # STL/OBJ export uses the same 200-point airfoil profile rule as ForgeCAD.
    # Keep the span interpolation dense enough for smooth transitions, but not so
    # dense that Fusion displays thousands of near-coincident sliver facets as gaps.
    stitched_stl_subdivisions_per_interval: int = 32
    stitched_stl_points_per_profile: int = AIRFOIL_POINT_COUNT
    bounds_padding_mm: float = 35.0
    preview_camera: str = "45:35"
    preview_size_px: int = 1200
    command_timeout_s: int = 300


@dataclass(frozen=True)
class GeneratorConfig:
    """Top-level configuration assembled from the groups above."""

    # Batch / structure
    num_designs: int = 10
    num_sections: int = 25
    wingtip_sections: int = 10
    output_dir: Path = Path("genesis-batch")
    random_seed: int = 42
    airfoil_points_per_profile: int = AIRFOIL_POINT_COUNT

    # Fixed airframe (mirrors the module constants; not genes)
    total_span_m: float = WING_SPAN_M
    root_chord_m: float = ROOT_CHORD_M

    # Chord-scale floor (taper cannot shrink a section below this fraction)
    scale_min: float = 0.40

    # Winglet tip-closure helpers (loft-safety + thin edge thickness).
    # NOTE: the exact (truck) B-rep kernel used by the interactive viewer cannot loft a
    # near-razor point — it throws "memory access out of bounds". ~60 mm chord is the
    # safe threshold. The manifold/mesh kernel (STL/CFD) tolerates far smaller, but we
    # keep one floor so both backends agree. A pointy tip at 60 mm still reads clearly
    # sharper than a flat tip (which stays 150-400 mm).
    winglet_tip_thickness_m: float = 0.010
    winglet_min_tip_chord_m: float = 0.06
    winglet_min_tip_thickness_m: float = 0.010

    genes: GeneBounds = field(default_factory=GeneBounds)
    gene_space: GeneSpace = field(default_factory=GeneSpace)
    validation: ValidationSettings = field(default_factory=ValidationSettings)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    forgecad: ForgeCadSettings = field(default_factory=ForgeCadSettings)

    @property
    def allowed_airfoils(self) -> tuple[str, ...]:
        return self.gene_space.allowed_airfoils

    def with_overrides(
        self,
        *,
        num_designs: int | None = None,
        num_sections: int | None = None,
        total_span_m: float | None = None,
        output_dir: str | Path | None = None,
        random_seed: int | None = None,
        allowed_airfoils: Iterable[str] | None = None,
    ) -> "GeneratorConfig":
        gene_space = self.gene_space
        if allowed_airfoils is not None:
            cleaned = tuple(a.strip() for a in allowed_airfoils if a.strip())
            gene_space = replace(gene_space, allowed_airfoils=cleaned)
        return replace(
            self,
            num_designs=self.num_designs if num_designs is None else num_designs,
            num_sections=self.num_sections if num_sections is None else num_sections,
            total_span_m=self.total_span_m if total_span_m is None else total_span_m,
            output_dir=self.output_dir if output_dir is None else Path(output_dir),
            random_seed=self.random_seed if random_seed is None else random_seed,
            gene_space=gene_space,
        )


DEFAULT_CONFIG = GeneratorConfig()
