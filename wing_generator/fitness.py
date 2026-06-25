"""Fitness / CFD "survival test" interface — **NOT implemented yet**.

This module only defines the *contract* between the design population and the CFD
evaluation so the rest of the system is ready for it. The actual CFD (OpenFOAM)
run is the next stage and will be built **after review**.

Planned flow (future):
    design (genome → geometry → STL) → OpenFOAM case → solve → aerodynamic metrics
    → :func:`combine_metrics` → fitness score → selection in the evolutionary loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import GeneratorConfig


@dataclass(frozen=True)
class FitnessResult:
    """Outcome of one design's survival test."""

    design_id: str
    metrics: dict[str, float] = field(default_factory=dict)  # raw CFD outputs
    score: float = 0.0                                       # weighted fitness
    feasible: bool = True                                    # passed/failed the test
    notes: str = ""


def combine_metrics(metrics: dict[str, float], config: GeneratorConfig) -> float:
    """Combine raw CFD metrics into one fitness score using configured weights.

    This part *is* defined (it is just arithmetic on the config weights); only the
    metric production (the CFD run) is pending.
    """

    weights = config.fitness.metric_weights
    score = sum(weights.get(name, 0.0) * value for name, value in metrics.items())
    return score if config.fitness.maximize else -score


def evaluate_fitness(design: dict[str, Any], config: GeneratorConfig) -> FitnessResult:
    """Run the CFD survival test for a design. Not implemented yet (pending review)."""

    raise NotImplementedError(
        "The CFD survival test is not implemented yet. It will mesh each design's STL, "
        "run OpenFOAM, extract aerodynamic metrics, and score them via combine_metrics(). "
        "This stage is intentionally deferred until after review."
    )
