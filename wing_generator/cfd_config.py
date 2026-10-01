from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DomainSettings:
    """Far-field distances in metres around the wing.

    OpenFOAM axes follow the working example case:
    x = chordwise/freestream direction, y = spanwise, z = vertical.
    """

    front_m: float = 4.0
    side_m: float = 3.0
    top_m: float = 4.0
    back_m: float = 8.0


@dataclass(frozen=True)
class RefinementSettings:
    """snappyHexMesh refinement controls."""

    front_m: float = 0.25
    side_m: float = 0.25
    top_m: float = 0.55
    back_m: float = 0.75
    base_cell_size_m: float = 0.16
    refinement_box_level: int = 3
    # Thin ground-hugging band under the wing, refined finer than the box so the
    # ground prism layers can extrude on cells comparable to the layer stack.
    ground_refinement_level: int = 4
    # Level 6 keeps the wing/winglet surface fine enough for lower y+ while
    # staying below the much heavier level-7 meshes.
    surface_min_level: int = 6
    surface_max_level: int = 6
    feature_level: int = 6
    max_global_cells: int = 8_000_000
    max_local_cells: int = 1_000_000

    @property
    def actual_refined_to_base_ratio(self) -> int:
        return 2**self.refinement_box_level


@dataclass(frozen=True)
class BoundaryLayerSettings:
    """Cell-relative snappyHexMesh layers on the wing and moving ground."""

    # Used to size the ground refinement band, not to set the first prism height.
    ground_first_layer_height_m: float = 0.0003
    growth_rate: float = 1.12
    # Cell-relative layer settings used by the motorBike-derived generator.
    # Layer-survival preset: keep 7 layers, but avoid an overly thin/strict
    # stack that snappy rejects across the leading/trailing edges.
    n_layers: int = 7
    final_layer_fraction: float = 0.2
    min_thickness_fraction: float = 0.001


@dataclass(frozen=True)
class SolverSettings:
    """Steady incompressible RANS settings."""

    velocity_mps: float = 25.0
    # In incompressible simpleFoam, p is kinematic pressure, so forceCoeffs uses
    # rhoInf=1 just like the validated example case.
    rho_kg_per_m3: float = 1.0
    nu_m2_per_s: float = 1.5e-5
    turbulent_intensity: float = 0.016
    turbulent_length_scale_m: float = 0.5
    end_time: int = 300
    # write_interval == end_time → only the final iteration's fields are written
    # (saves storage). forceCoeffs still logs every step (its own writeControl).
    write_interval: int = 300
    n_subdomains: int = 32

    @property
    def turbulent_ke(self) -> float:
        return 1.5 * (self.velocity_mps * self.turbulent_intensity) ** 2

    @property
    def turbulent_omega(self) -> float:
        beta_star = 0.09
        return self.turbulent_ke**0.5 / (beta_star**0.25 * self.turbulent_length_scale_m)


def default_reference_case_dir() -> Path:
    explicit = os.environ.get("OPENFOAM_REFERENCE_CASE")
    if explicit:
        return Path(explicit).expanduser()
    tutorials = os.environ.get("FOAM_TUTORIALS")
    if tutorials:
        return Path(tutorials) / "incompressible" / "simpleFoam" / "motorBike"
    return Path("ref") / "motorBike"


@dataclass(frozen=True)
class OpenFoamCaseConfig:
    """Inputs for generating OpenFOAM ground-effect cases."""

    population_dir: Path = Path("genesis-batch")
    output_dir: Path = Path("genesis-batch/run")
    reference_case_dir: Path = field(default_factory=default_reference_case_dir)
    # Ground plane z values. The wing/root reference remains around z=0; a value
    # of -0.4 means the ground is 0.4 m below the reference, matching the example.
    ground_clearances_m: tuple[float, ...] = (-0.2, -0.4, -0.6, -0.8, -1.0)
    stl_input_scale_to_m: float = 0.001
    root_plane_tolerance_m: float = 1e-6
    domain: DomainSettings = field(default_factory=DomainSettings)
    refinement: RefinementSettings = field(default_factory=RefinementSettings)
    boundary_layer: BoundaryLayerSettings = field(default_factory=BoundaryLayerSettings)
    solver: SolverSettings = field(default_factory=SolverSettings)


DEFAULT_CFD_CONFIG = OpenFoamCaseConfig()
