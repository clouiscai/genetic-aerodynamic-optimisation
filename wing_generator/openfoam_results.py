from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_COEFF_COLUMNS = ("Time", "Cd", "Cl", "CmPitch")


def collect_force_coefficients(cases_dir: Path, output_csv: Path | None = None) -> list[dict[str, str]]:
    cases_dir = cases_dir.resolve()
    rows: list[dict[str, str]] = []
    for coefficient_path in coefficient_files(cases_dir):
        case_dir = coefficient_path.parents[3]
        ground_dir = case_dir.name
        design_id = case_dir.parent.name
        final_values = read_final_coefficient_row(coefficient_path)
        if final_values is None:
            continue
        row = {
            "design_id": design_id,
            "ground_case": ground_dir,
            "case_dir": str(case_dir),
        }
        row.update(final_values)
        rows.append(row)

    target = output_csv or cases_dir / "force_coefficients_summary.csv"
    write_summary_csv(target, rows)
    return rows


def collect_force_coefficient_iterations(
    cases_dir: Path, output_csv: Path | None = None
) -> list[dict[str, str]]:
    cases_dir = cases_dir.resolve()
    rows: list[dict[str, str]] = []
    for coefficient_path in coefficient_files(cases_dir):
        case_dir = coefficient_path.parents[3]
        ground_dir = case_dir.name
        design_id = case_dir.parent.name
        for values in read_coefficient_rows(coefficient_path):
            row = {
                "design_id": design_id,
                "ground_case": ground_dir,
                "case_dir": str(case_dir),
            }
            row.update(values)
            rows.append(row)

    target = output_csv or cases_dir / "force_coefficients_iterations.csv"
    write_summary_csv(target, rows)
    return rows


def coefficient_files(cases_dir: Path) -> list[Path]:
    return sorted(
        cases_dir.glob("design_*/ground_*/postProcessing/wingForceCoeffs/*/coefficient.dat")
    )


def read_final_coefficient_row(path: Path) -> dict[str, str] | None:
    rows = read_coefficient_rows(path)
    return rows[-1] if rows else None


def read_coefficient_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    columns = DEFAULT_COEFF_COLUMNS
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            header = stripped.lstrip("#").strip().split()
            if header and header[0] == "Time":
                columns = tuple(header)
            continue
        parts = stripped.split()
        rows.append(
            {
                column: parts[index] if index < len(parts) else ""
                for index, column in enumerate(columns)
            }
        )
    return rows


def write_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    coefficient_columns = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in {"design_id", "ground_case", "case_dir"}
        },
        key=lambda key: (key != "Time", key),
    )
    fieldnames = ["design_id", "ground_case", "case_dir", *coefficient_columns]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect OpenFOAM wing force coefficients.")
    parser.add_argument("--cases-dir", type=Path, default=Path("genesis-batch/run"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--iterations-output",
        type=Path,
        default=None,
        help="CSV path for every forceCoeffs iteration row.",
    )
    args = parser.parse_args()
    rows = collect_force_coefficients(args.cases_dir, args.output)
    target = args.output or args.cases_dir / "force_coefficients_summary.csv"
    print(f"Wrote {len(rows)} final force coefficient rows to {target.resolve()}")
    iteration_rows = collect_force_coefficient_iterations(args.cases_dir, args.iterations_output)
    iteration_target = args.iterations_output or args.cases_dir / "force_coefficients_iterations.csv"
    print(f"Wrote {len(iteration_rows)} per-iteration force coefficient rows to {iteration_target.resolve()}")


if __name__ == "__main__":
    main()
