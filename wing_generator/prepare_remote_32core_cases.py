from __future__ import annotations

import argparse
from pathlib import Path

from .cfd_config import OpenFoamCaseConfig
from .motorbike_case_generator import generate_motorbike_cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare motorBike-derived 32-core OpenFOAM wing cases."
    )
    parser.add_argument("--population-dir", type=Path, default=Path("genesis-batch"))
    parser.add_argument("--output-dir", type=Path, default=Path("genesis-batch/run"))
    parser.add_argument(
        "--ground-clearances",
        default="-0.2,-0.4,-0.6,-0.8,-1.0",
        help="Comma-separated ground-plane z values in metres.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = OpenFoamCaseConfig(
        population_dir=args.population_dir,
        output_dir=args.output_dir,
        ground_clearances_m=tuple(float(value) for value in args.ground_clearances.split(",")),
    )
    cases = generate_motorbike_cases(config)
    print(
        f"Generated {len(cases)} motorBike-derived 32-core OpenFOAM cases "
        f"in {config.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
