from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .config import DEFAULT_CONFIG
from .exporter import ExportOptions, export_design, validate_forgecad_model
from .geometry_generator import write_design_files
from .population_generator import PopulationGenerator, validate_design, write_population_manifest
from .report import build_parameter_report


def main() -> None:
    args = _parse_args()
    allowed_airfoils = None
    if args.airfoils:
        allowed_airfoils = tuple(part.strip() for part in args.airfoils.split(",") if part.strip())

    config = DEFAULT_CONFIG.with_overrides(
        num_designs=args.num_designs,
        num_sections=args.num_sections,
        total_span_m=args.total_span,
        output_dir=args.output_dir,
        random_seed=args.seed,
        allowed_airfoils=allowed_airfoils,
    )

    export_options = ExportOptions(
        validate_cad=not args.skip_cad_validation,
        export_stl=not args.skip_export and not args.skip_stl,
        export_step=not args.skip_export and not args.skip_step,
        render_preview=not args.skip_preview,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "PARAMETER_REPORT.md").write_text(
        build_parameter_report(config) + "\n", encoding="utf-8"
    )
    generator = PopulationGenerator(config)
    accepted_designs = []
    log_rows: list[dict[str, str | int]] = []

    design_index = 0
    while design_index < config.num_designs:
        print(f"Generating design_{design_index:03d}...")
        attempts = 0
        while attempts < args.max_attempts_per_design:
            attempts += 1
            design = generator.next_valid_design(design_index)
            design_dir = config.output_dir / design["design_id"]
            write_design_files(design, design_dir, config)

            validation = validate_design(design, config)
            if not validation.ok:
                log_rows.append(
                    {
                        "design_id": design["design_id"],
                        "attempt": attempts,
                        "status": "rejected_parameter_validation",
                        "message": "; ".join(validation.errors),
                    }
                )
                continue

            if export_options.validate_cad:
                cad_result = validate_forgecad_model(design_dir, config)
                if not cad_result.ok:
                    log_rows.append(
                        {
                            "design_id": design["design_id"],
                            "attempt": attempts,
                            "status": "rejected_forgecad_validation",
                            "message": "ForgeCAD run failed; see forgecad_run.log",
                        }
                    )
                    continue

            export_design(design_dir, config, export_options)
            accepted_designs.append(design)
            log_rows.append(
                {
                    "design_id": design["design_id"],
                    "attempt": attempts,
                    "status": "accepted",
                    "message": "generated",
                }
            )
            design_index += 1
            break
        else:
            raise RuntimeError(
                f"Could not produce design_{design_index:03d} after "
                f"{args.max_attempts_per_design} attempts"
            )

    write_population_manifest(config.output_dir, accepted_designs, config)
    _write_generation_log(config.output_dir / "generation_log.csv", log_rows)
    print(f"Done. Generated {len(accepted_designs)} designs in {config.output_dir.resolve()}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the initial ForgeCAD wing population for later CFD evaluation."
    )
    parser.add_argument("--num-designs", type=int, default=DEFAULT_CONFIG.num_designs)
    parser.add_argument("--num-sections", type=int, default=DEFAULT_CONFIG.num_sections)
    parser.add_argument("--total-span", type=float, default=DEFAULT_CONFIG.total_span_m)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CONFIG.output_dir)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.random_seed)
    parser.add_argument(
        "--airfoils",
        type=str,
        default="",
        help="Comma-separated airfoil keys. Defaults to ClarkY,NACA4412,Eppler817,Eppler387.",
    )
    parser.add_argument("--max-attempts-per-design", type=int, default=200)
    parser.add_argument("--skip-cad-validation", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-stl", action="store_true")
    parser.add_argument("--skip-step", action="store_true")
    parser.add_argument("--skip-preview", action="store_true")
    return parser.parse_args()


def _write_generation_log(path: Path, rows: list[dict[str, str | int]]) -> None:
    fieldnames = ["design_id", "attempt", "status", "message"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
