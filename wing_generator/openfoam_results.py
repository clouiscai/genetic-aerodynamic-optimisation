#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from statistics import mean


COEFF_GLOB = "postProcessing/wingForceCoeffs/*/coefficient.dat"
YPLUS_GLOB = "postProcessing/yPlus/*/yPlus.dat"
BASE_COLUMNS = [
    "design_id",
    "ground_case",
    "ground_plane_z_m",
    "nominal_height_above_ground_m",
    "status",
    "case_dir",
    "final_time",
    "Cd",
    "Cl",
    "CmPitch",
    "Cd_avg_last_window",
    "Cl_avg_last_window",
    "CmPitch_avg_last_window",
    "wing_yplus_min",
    "wing_yplus_max",
    "wing_yplus_average",
    "lowerWall_yplus_min",
    "lowerWall_yplus_max",
    "lowerWall_yplus_average",
    "coefficient_file",
    "yplus_file",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect final OpenFOAM force coefficients and y+ stats from a "
            "generation run folder into CSV files."
        )
    )
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=Path("."),
        help="Batch run directory containing design_*/ground_* cases. Default: current directory.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("all_results_summary.csv"),
        help="CSV file for one summary row per case.",
    )
    parser.add_argument(
        "--iterations-output",
        type=Path,
        default=Path("force_coefficients_iterations.csv"),
        help="CSV file for every force-coefficient iteration row.",
    )
    parser.add_argument(
        "--average-window",
        type=int,
        default=50,
        help="Number of final coefficient rows used for *_avg_last_window columns.",
    )
    parser.add_argument(
        "--no-iterations",
        action="store_true",
        help="Only write the per-case summary CSV.",
    )
    args = parser.parse_args()

    cases_dir = args.cases_dir.resolve()
    manifest = read_manifest(cases_dir / "case_manifest.csv")
    case_dirs = discover_case_dirs(cases_dir, manifest)

    summary_rows: list[dict[str, str]] = []
    iteration_rows: list[dict[str, str]] = []
    for case_dir in case_dirs:
        case_key = case_dir_key(case_dir)
        manifest_row = manifest.get(case_key, {})
        coeff_file = newest_matching_file(case_dir, COEFF_GLOB)
        yplus_file = newest_matching_file(case_dir, YPLUS_GLOB)

        coeff_rows = read_table(coeff_file) if coeff_file else []
        yplus_rows = read_table(yplus_file) if yplus_file else []

        final_coeffs = coeff_rows[-1] if coeff_rows else {}
        row = {
            "design_id": case_dir.parent.name,
            "ground_case": case_dir.name,
            "ground_plane_z_m": manifest_row.get("ground_plane_z_m", ground_plane_from_name(case_dir.name)),
            "nominal_height_above_ground_m": manifest_row.get(
                "nominal_height_above_ground_m", nominal_height_from_name(case_dir.name)
            ),
            "status": case_status(case_dir, coeff_rows),
            "case_dir": case_dir.relative_to(cases_dir).as_posix(),
            "final_time": final_coeffs.get("Time", ""),
            "Cd": final_coeffs.get("Cd", ""),
            "Cl": final_coeffs.get("Cl", ""),
            "CmPitch": final_coeffs.get("CmPitch", ""),
            "coefficient_file": relative_or_empty(coeff_file, cases_dir),
            "yplus_file": relative_or_empty(yplus_file, cases_dir),
        }
        row.update(averages_for_window(coeff_rows, args.average_window))
        row.update(yplus_by_patch(yplus_rows, "wing"))
        row.update(yplus_by_patch(yplus_rows, "lowerWall"))
        summary_rows.append(row)

        if not args.no_iterations:
            for coeff_row in coeff_rows:
                iteration_rows.append(
                    {
                        "design_id": case_dir.parent.name,
                        "ground_case": case_dir.name,
                        "case_dir": case_dir.relative_to(cases_dir).as_posix(),
                        **coeff_row,
                    }
                )

    write_csv(resolve_output(cases_dir, args.summary_output), summary_rows, BASE_COLUMNS)
    print(f"Wrote {len(summary_rows)} case rows to {resolve_output(cases_dir, args.summary_output)}")

    if not args.no_iterations:
        iteration_columns = columns_for_rows(iteration_rows, ["design_id", "ground_case", "case_dir", "Time"])
        write_csv(resolve_output(cases_dir, args.iterations_output), iteration_rows, iteration_columns)
        print(
            f"Wrote {len(iteration_rows)} coefficient iteration rows to "
            f"{resolve_output(cases_dir, args.iterations_output)}"
        )


def read_manifest(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            case_path = Path(row.get("case_dir", "").replace("\\", "/"))
            ground_case = case_path.name
            design_id = row.get("design_id", "") or case_path.parent.name
            if design_id and ground_case:
                rows[(design_id, ground_case)] = row
    return rows


def discover_case_dirs(cases_dir: Path, manifest: dict[tuple[str, str], dict[str, str]]) -> list[Path]:
    found = {path for path in cases_dir.glob("design_*/ground_*") if path.is_dir()}
    for design_id, ground_case in manifest:
        path = cases_dir / design_id / ground_case
        if path.is_dir():
            found.add(path)
    return sorted(found, key=lambda path: (path.parent.name, path.name))


def newest_matching_file(case_dir: Path, pattern: str) -> Path | None:
    matches = sorted(case_dir.glob(pattern), key=lambda path: (numeric_parent_time(path), str(path)))
    return matches[-1] if matches else None


def numeric_parent_time(path: Path) -> float:
    try:
        return float(path.parent.name)
    except ValueError:
        return -1.0


def read_table(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    columns: tuple[str, ...] = ()
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                candidate = stripped.lstrip("#").strip().split()
                if candidate and candidate[0] == "Time":
                    columns = tuple(candidate)
                continue
            parts = stripped.split()
            if not columns:
                columns = tuple(f"col_{index}" for index in range(len(parts)))
            rows.append({column: parts[index] if index < len(parts) else "" for index, column in enumerate(columns)})
    return rows


def case_status(case_dir: Path, coeff_rows: list[dict[str, str]]) -> str:
    if not coeff_rows:
        return "missing_coefficients"
    log_path = case_dir / "log.simpleFoam"
    if not log_path.exists():
        return "coefficients_found_no_log"
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    final_time = coeff_rows[-1].get("Time", "")
    if final_time and re.search(rf"^\s*Time\s*=\s*{re.escape(final_time)}\s*$", text, flags=re.MULTILINE):
        return "ok"
    return "coefficients_found_check_log"


def averages_for_window(rows: list[dict[str, str]], window: int) -> dict[str, str]:
    if not rows or window <= 0:
        return {
            "Cd_avg_last_window": "",
            "Cl_avg_last_window": "",
            "CmPitch_avg_last_window": "",
        }
    tail = rows[-window:]
    return {
        "Cd_avg_last_window": average_column(tail, "Cd"),
        "Cl_avg_last_window": average_column(tail, "Cl"),
        "CmPitch_avg_last_window": average_column(tail, "CmPitch"),
    }


def average_column(rows: list[dict[str, str]], column: str) -> str:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get(column, "")))
        except ValueError:
            pass
    return f"{mean(values):.9g}" if values else ""


def yplus_by_patch(rows: list[dict[str, str]], patch: str) -> dict[str, str]:
    result = {
        f"{patch}_yplus_min": "",
        f"{patch}_yplus_max": "",
        f"{patch}_yplus_average": "",
    }
    for row in reversed(rows):
        if row.get("patch") == patch:
            result[f"{patch}_yplus_min"] = row.get("min", "")
            result[f"{patch}_yplus_max"] = row.get("max", "")
            result[f"{patch}_yplus_average"] = row.get("average", "")
            break
    return result


def case_dir_key(case_dir: Path) -> tuple[str, str]:
    return case_dir.parent.name, case_dir.name


def ground_plane_from_name(name: str) -> str:
    value = nominal_height_from_name(name)
    return f"-{value}" if value else ""


def nominal_height_from_name(name: str) -> str:
    match = re.fullmatch(r"ground_m(\d+)p(\d+)", name)
    if not match:
        return ""
    return f"{int(match.group(1))}.{match.group(2)}"


def relative_or_empty(path: Path | None, base: Path) -> str:
    return path.relative_to(base).as_posix() if path else ""


def resolve_output(cases_dir: Path, output: Path) -> Path:
    return output if output.is_absolute() else cases_dir / output


def columns_for_rows(rows: list[dict[str, str]], preferred: list[str]) -> list[str]:
    all_columns = {key for row in rows for key in row}
    columns = list(preferred)
    columns.extend(sorted(all_columns - set(columns)))
    return columns


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extras = sorted({key for row in rows for key in row} - set(columns))
    fieldnames = [*columns, *extras]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
