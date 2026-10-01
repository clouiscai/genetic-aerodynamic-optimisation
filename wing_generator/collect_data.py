from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_REQUIRED_TIME = 300.0
DEFAULT_MAX_ABS_COEFFICIENT = 10.0
DEFAULT_MIN_CASES_PER_DESIGN = 5

OUTPUT_COLUMNS = [
    "generation",
    "generation_number",
    "source_batch",
    "design_id",
    "ground_case",
    "ground_plane_z_m",
    "nominal_height_above_ground_m",
    "completion_reason",
    "final_time",
    "reached_required_time",
    "normal_solver_end",
    "summary_status",
    "coefficient_source",
    "C_l",
    "C_d",
    "C_m",
    "lift_to_drag",
    "C_l_final",
    "C_d_final",
    "C_m_final",
    "C_l_avg_last_window",
    "C_d_avg_last_window",
    "C_m_avg_last_window",
    "main_wing_span_m",
    "winglet_spanwise_reach_m",
    "actual_half_span_m",
    "reference_planform_area_m2",
    "reference_mean_chord_m",
    "root_chord_m",
    "tip_chord_m",
    "chordwise_extent_m",
    "vertical_extent_m",
    "root_airfoil",
    "tip_airfoil",
    "root_pitch_deg",
    "tip_pitch_deg",
    "winglet_shape",
    "winglet_is_pointy",
    "winglet_ellipse_length_m",
    "winglet_ellipse_radius_m",
    "winglet_rotation_deg",
    "winglet_arc_angle_deg",
    "winglet_horizontal_rate",
    "winglet_scale_rate",
    "reference_length_m",
    "reynolds_number",
    "wing_faces",
    "root_faces",
    "source_summary_csv",
    "parameters_json",
    "case_info_json",
]

EXCLUDED_COLUMNS = [
    "generation",
    "source_batch",
    "design_id",
    "ground_case",
    "final_time",
    "summary_status",
    "reason",
    "C_l_final",
    "C_d_final",
    "C_m_final",
    "source_summary_csv",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Combine OpenFOAM all_results_summary.csv files with generated wing "
            "geometry data, keeping only completed and physically sane cases."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing genesis-batch and generation_* folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("combined_data.csv"),
        help="Combined qualifying case CSV, relative to --root by default.",
    )
    parser.add_argument(
        "--excluded-output",
        type=Path,
        default=Path("combined_excluded_aero_cases.csv"),
        help="Audit CSV containing rejected cases.",
    )
    parser.add_argument(
        "--required-time",
        type=float,
        default=DEFAULT_REQUIRED_TIME,
        help="A case qualifies when final_time reaches this value.",
    )
    parser.add_argument(
        "--max-abs-coefficient",
        type=float,
        default=DEFAULT_MAX_ABS_COEFFICIENT,
        help="Reject non-finite coefficients or coefficients at/above this magnitude.",
    )
    parser.add_argument(
        "--coefficient-source",
        choices=("average", "final"),
        default="average",
        help="Use last-window averages when available, or always use final values.",
    )
    parser.add_argument(
        "--require-full-sweep",
        action="store_true",
        help="Keep a design only when all its cases qualify and it has enough cases.",
    )
    parser.add_argument(
        "--min-cases-per-design",
        type=int,
        default=DEFAULT_MIN_CASES_PER_DESIGN,
        help="Minimum qualifying cases when --require-full-sweep is used.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output = resolve_path(root, args.output)
    excluded_output = resolve_path(root, args.excluded_output)
    summaries = discover_summaries(root)
    if not summaries:
        raise SystemExit(f"No run/all_results_summary.csv files found under {root}")

    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    all_case_counts: dict[tuple[str, str], int] = defaultdict(int)
    accepted_case_counts: dict[tuple[str, str], int] = defaultdict(int)

    for batch_dir, summary_path in summaries:
        generation_label, generation_number = generation_info(batch_dir.name)
        run_dir = summary_path.parent
        manifest = read_case_manifest(run_dir / "case_manifest.csv")

        for summary_row in read_csv(summary_path):
            design_id = summary_row.get("design_id", "").strip()
            ground_case = summary_row.get("ground_case", "").strip()
            design_key = (batch_dir.name, design_id)
            all_case_counts[design_key] += 1

            parameters_path = batch_dir / design_id / "parameters.json"
            parameters = read_json(parameters_path)
            case_dir = case_directory(run_dir, summary_row, design_id, ground_case)
            case_info_path = case_dir / "case_info.json"
            case_info = read_json(case_info_path)
            manifest_row = manifest.get((design_id, ground_case), {})
            log_path = case_dir / "log.simpleFoam"

            reason, completion_reason = qualification(
                summary_row,
                log_path=log_path,
                required_time=args.required_time,
                max_abs_coefficient=args.max_abs_coefficient,
                coefficient_source=args.coefficient_source,
            )
            if reason:
                excluded.append(
                    excluded_row(
                        generation_label,
                        batch_dir.name,
                        summary_row,
                        reason,
                        summary_path,
                    )
                )
                continue

            row = combined_row(
                generation_label=generation_label,
                generation_number=generation_number,
                batch_dir=batch_dir,
                summary_path=summary_path,
                summary_row=summary_row,
                parameters_path=parameters_path,
                parameters=parameters,
                case_info_path=case_info_path,
                case_info=case_info,
                manifest_row=manifest_row,
                completion_reason=completion_reason,
                log_path=log_path,
                required_time=args.required_time,
                coefficient_source=args.coefficient_source,
            )
            accepted.append(row)
            accepted_case_counts[design_key] += 1

    if args.require_full_sweep:
        complete_designs = {
            key
            for key, total in all_case_counts.items()
            if total >= args.min_cases_per_design and accepted_case_counts[key] == total
        }
        retained: list[dict[str, Any]] = []
        for row in accepted:
            key = (str(row["source_batch"]), str(row["design_id"]))
            if key in complete_designs:
                retained.append(row)
            else:
                excluded.append(
                    {
                        "generation": row["generation"],
                        "source_batch": row["source_batch"],
                        "design_id": row["design_id"],
                        "ground_case": row["ground_case"],
                        "final_time": row["final_time"],
                        "summary_status": row["summary_status"],
                        "reason": "design_incomplete_sweep",
                        "C_l_final": row["C_l_final"],
                        "C_d_final": row["C_d_final"],
                        "C_m_final": row["C_m_final"],
                        "source_summary_csv": row["source_summary_csv"],
                    }
                )
        accepted = retained

    accepted.sort(key=case_sort_key)
    excluded.sort(key=case_sort_key)
    write_csv(output, accepted, OUTPUT_COLUMNS)
    write_csv(excluded_output, excluded, EXCLUDED_COLUMNS)

    batches = sorted({row["source_batch"] for row in accepted})
    designs = {(row["source_batch"], row["design_id"]) for row in accepted}
    print(f"Read {len(summaries)} summary files: {', '.join(path.parent.parent.name for _, path in summaries)}")
    print(f"Wrote {len(accepted)} qualifying case rows from {len(designs)} designs to {output}")
    print(f"Wrote {len(excluded)} excluded case rows to {excluded_output}")
    print(f"Included batches: {', '.join(batches) if batches else '(none)'}")
    return 0


def discover_summaries(root: Path) -> list[tuple[Path, Path]]:
    batch_dirs = []
    genesis = root / "genesis-batch"
    if genesis.is_dir():
        batch_dirs.append(genesis)
    batch_dirs.extend(sorted(path for path in root.glob("generation_*") if path.is_dir()))

    found = []
    for batch_dir in batch_dirs:
        summary = batch_dir / "run" / "all_results_summary.csv"
        if summary.is_file():
            found.append((batch_dir, summary))
    return sorted(found, key=lambda item: generation_info(item[0].name)[1])


def generation_info(batch_name: str) -> tuple[str, int]:
    if batch_name == "genesis-batch":
        return "genesis", 0
    match = re.fullmatch(r"generation_(\d+)", batch_name)
    if match:
        number = int(match.group(1))
        return f"generation_{number:02d}", number
    return batch_name, 999999


def qualification(
    row: dict[str, str],
    *,
    log_path: Path,
    required_time: float,
    max_abs_coefficient: float,
    coefficient_source: str = "average",
) -> tuple[str, str]:
    final_time = parse_float(row.get("final_time"))
    final_coefficients = [
        parse_float(row.get("Cl")),
        parse_float(row.get("Cd")),
        parse_float(row.get("CmPitch")),
    ]
    if final_time is None or not math.isfinite(final_time):
        return "missing_final_time", ""
    if any(value is None or not math.isfinite(value) for value in final_coefficients):
        return "missing_or_nonfinite_coefficient", ""
    if any(abs(float(value)) >= max_abs_coefficient for value in final_coefficients):
        return "coefficient_out_of_range", ""
    averages = [
        parse_float(row.get(name))
        for name in ("Cl_avg_last_window", "Cd_avg_last_window", "CmPitch_avg_last_window")
    ]
    if coefficient_source == "average" and all(value is not None for value in averages):
        if any(not math.isfinite(value) or abs(value) >= max_abs_coefficient for value in averages):
            return "invalid_average_coefficient", ""

    reached_time = final_time + 1e-9 >= required_time
    normal_end = solver_ended_normally(log_path)
    if reached_time:
        return "", f"final_time>={format_number(required_time)}"
    if normal_end:
        return "", "normal_solver_end"
    return f"incomplete_final_time<{format_number(required_time)}", ""


def combined_row(
    *,
    generation_label: str,
    generation_number: int,
    batch_dir: Path,
    summary_path: Path,
    summary_row: dict[str, str],
    parameters_path: Path,
    parameters: dict[str, Any],
    case_info_path: Path,
    case_info: dict[str, Any],
    manifest_row: dict[str, str],
    completion_reason: str,
    log_path: Path,
    required_time: float,
    coefficient_source: str,
) -> dict[str, Any]:
    final_cl = parse_float(summary_row.get("Cl"))
    final_cd = parse_float(summary_row.get("Cd"))
    final_cm = parse_float(summary_row.get("CmPitch"))
    average_cl = parse_float(summary_row.get("Cl_avg_last_window"))
    average_cd = parse_float(summary_row.get("Cd_avg_last_window"))
    average_cm = parse_float(summary_row.get("CmPitch_avg_last_window"))

    if coefficient_source == "average" and all(
        value is not None for value in (average_cl, average_cd, average_cm)
    ):
        cl, cd, cm = average_cl, average_cd, average_cm
        used_source = "last_window_average"
    else:
        cl, cd, cm = final_cl, final_cd, final_cm
        used_source = "final"

    geometry = geometry_data(parameters, case_info, manifest_row)
    final_time = parse_float(summary_row.get("final_time"))
    normal_end = solver_ended_normally(log_path)

    return {
        "generation": generation_label,
        "generation_number": generation_number,
        "source_batch": batch_dir.name,
        "design_id": summary_row.get("design_id", ""),
        "ground_case": summary_row.get("ground_case", ""),
        "ground_plane_z_m": first_value(
            summary_row.get("ground_plane_z_m"),
            case_info.get("ground_plane_z_m"),
            manifest_row.get("ground_plane_z_m"),
        ),
        "nominal_height_above_ground_m": first_value(
            summary_row.get("nominal_height_above_ground_m"),
            case_info.get("nominal_height_above_ground_m"),
            manifest_row.get("nominal_height_above_ground_m"),
        ),
        "completion_reason": completion_reason,
        "final_time": final_time,
        "reached_required_time": bool(final_time is not None and final_time + 1e-9 >= required_time),
        "normal_solver_end": normal_end,
        "summary_status": summary_row.get("status", ""),
        "coefficient_source": used_source,
        "C_l": cl,
        "C_d": cd,
        "C_m": cm,
        "lift_to_drag": safe_divide(cl, cd),
        "C_l_final": final_cl,
        "C_d_final": final_cd,
        "C_m_final": final_cm,
        "C_l_avg_last_window": average_cl,
        "C_d_avg_last_window": average_cd,
        "C_m_avg_last_window": average_cm,
        **geometry,
        "source_summary_csv": relative_text(summary_path, batch_dir.parent),
        "parameters_json": relative_text(parameters_path, batch_dir.parent) if parameters_path.is_file() else "",
        "case_info_json": relative_text(case_info_path, batch_dir.parent) if case_info_path.is_file() else "",
    }


def geometry_data(
    parameters: dict[str, Any],
    case_info: dict[str, Any],
    manifest_row: dict[str, str],
) -> dict[str, Any]:
    sections = parameters.get("sections") or []
    wingtip = parameters.get("wingtip") or {}
    genome = parameters.get("genome") or {}
    bounds = case_info.get("wing_bounds_m") or {}
    minimum = bounds.get("minimum") or []
    maximum = bounds.get("maximum") or []

    main_span = parse_float(parameters.get("total_span"))
    if main_span is None:
        main_span = parse_float((parameters.get("fixed") or {}).get("wing_span_m"))

    winglet_length = parse_float(wingtip.get("ellipse_length")) or 0.0
    winglet_arc_deg = parse_float(wingtip.get("arc_angle_deg")) or 0.0
    formula_reach = winglet_length * math.sin(math.radians(abs(winglet_arc_deg)))
    formula_actual_span = (main_span or 0.0) + formula_reach

    bounds_span = axis_extent(minimum, maximum, 1)
    actual_span = bounds_span if bounds_span is not None else formula_actual_span
    winglet_reach = (
        max(0.0, actual_span - main_span)
        if main_span is not None and actual_span is not None
        else formula_reach
    )

    area = first_float(
        case_info.get("area_ref_m2"),
        manifest_row.get("area_ref_m2"),
        reference_area(parameters),
    )
    root_chord = section_float(sections, 0, "chord_length")
    tip_chord = section_float(sections, -1, "chord_length")
    root_airfoil = section_value(sections, 0, "airfoil_type") or genome.get("root_airfoil", "")
    tip_airfoil = section_value(sections, -1, "airfoil_type") or genome.get("tip_airfoil", "")
    root_pitch = first_float(
        section_value(sections, 0, "root_pitch_angle_deg"),
        genome.get("root_pitch_deg"),
    )
    tip_pitch = section_float(sections, -1, "root_pitch_angle_deg")
    l_ref = first_float(
        case_info.get("l_ref_m"),
        manifest_row.get("l_ref_m"),
        root_chord,
    )

    return {
        "main_wing_span_m": main_span,
        "winglet_spanwise_reach_m": winglet_reach,
        "actual_half_span_m": actual_span,
        "reference_planform_area_m2": area,
        "reference_mean_chord_m": safe_divide(area, main_span),
        "root_chord_m": root_chord,
        "tip_chord_m": tip_chord,
        "chordwise_extent_m": axis_extent(minimum, maximum, 0),
        "vertical_extent_m": axis_extent(minimum, maximum, 2),
        "root_airfoil": root_airfoil,
        "tip_airfoil": tip_airfoil,
        "root_pitch_deg": root_pitch,
        "tip_pitch_deg": tip_pitch,
        "winglet_shape": wingtip.get("shape", ""),
        "winglet_is_pointy": wingtip.get("is_pointy", ""),
        "winglet_ellipse_length_m": parse_float(wingtip.get("ellipse_length")),
        "winglet_ellipse_radius_m": parse_float(wingtip.get("ellipse_radius")),
        "winglet_rotation_deg": parse_float(wingtip.get("ellipse_rotation_deg")),
        "winglet_arc_angle_deg": parse_float(wingtip.get("arc_angle_deg")),
        "winglet_horizontal_rate": parse_float(wingtip.get("horizontal_rate")),
        "winglet_scale_rate": parse_float(wingtip.get("scale_rate")),
        "reference_length_m": l_ref,
        "reynolds_number": first_float(
            case_info.get("reynolds_number"),
            manifest_row.get("reynolds_number"),
        ),
        "wing_faces": first_value(case_info.get("wing_faces"), manifest_row.get("wing_faces")),
        "root_faces": first_value(case_info.get("root_faces"), manifest_row.get("root_faces")),
    }


def excluded_row(
    generation_label: str,
    batch_name: str,
    row: dict[str, str],
    reason: str,
    summary_path: Path,
) -> dict[str, Any]:
    return {
        "generation": generation_label,
        "source_batch": batch_name,
        "design_id": row.get("design_id", ""),
        "ground_case": row.get("ground_case", ""),
        "final_time": row.get("final_time", ""),
        "summary_status": row.get("status", ""),
        "reason": reason,
        "C_l_final": row.get("Cl", ""),
        "C_d_final": row.get("Cd", ""),
        "C_m_final": row.get("CmPitch", ""),
        "source_summary_csv": str(summary_path),
    }


def read_case_manifest(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    if not path.is_file():
        return rows
    for row in read_csv(path):
        design_id = row.get("design_id", "").strip()
        case_path = Path(row.get("case_dir", "").replace("\\", "/"))
        ground_case = case_path.name
        if design_id and ground_case:
            rows[(design_id, ground_case)] = row
    return rows


def case_directory(
    run_dir: Path,
    row: dict[str, str],
    design_id: str,
    ground_case: str,
) -> Path:
    local = run_dir / design_id / ground_case
    if local.is_dir():
        return local
    case_value = row.get("case_dir", "").strip()
    if case_value:
        candidate = Path(case_value.replace("\\", "/"))
        if candidate.is_absolute():
            return candidate
        return run_dir / candidate
    return run_dir / design_id / ground_case


def solver_ended_normally(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    try:
        tail = log_path.read_text(encoding="utf-8", errors="ignore")[-16384:]
    except OSError:
        return False
    return bool(re.search(r"(?m)^\s*End\s*$", tail))


def reference_area(parameters: dict[str, Any]) -> float | None:
    sections = parameters.get("sections") or []
    if len(sections) < 2:
        root_chord = first_float(
            (parameters.get("fixed") or {}).get("root_chord_m"),
            section_value(sections, 0, "chord_length"),
        )
        span = first_float(
            parameters.get("total_span"),
            (parameters.get("fixed") or {}).get("wing_span_m"),
        )
        return root_chord * span if root_chord is not None and span is not None else None

    area = 0.0
    try:
        for left, right in zip(sections, sections[1:]):
            dy = float(right["spanwise_position"]) - float(left["spanwise_position"])
            area += 0.5 * (float(left["chord_length"]) + float(right["chord_length"])) * dy
    except (KeyError, TypeError, ValueError):
        return None
    return max(area, 1e-9)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: csv_value(row.get(column)) for column in columns})


def resolve_path(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def relative_text(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_float(*values: Any) -> float | None:
    for value in values:
        parsed = parse_float(value)
        if parsed is not None:
            return parsed
    return None


def first_value(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return ""


def section_value(sections: list[dict[str, Any]], index: int, key: str) -> Any:
    if not sections:
        return None
    try:
        return sections[index].get(key)
    except (IndexError, TypeError):
        return None


def section_float(sections: list[dict[str, Any]], index: int, key: str) -> float | None:
    return parse_float(section_value(sections, index, key))


def axis_extent(minimum: list[Any], maximum: list[Any], axis: int) -> float | None:
    try:
        return float(maximum[axis]) - float(minimum[axis])
    except (IndexError, TypeError, ValueError):
        return None


def safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".12g")
    return value


def format_number(value: float) -> str:
    return format(value, "g")


def case_sort_key(row: dict[str, Any]) -> tuple[int, int, float, str]:
    generation_number = generation_info(str(row.get("source_batch", "")))[1]
    design_match = re.search(r"(\d+)$", str(row.get("design_id", "")))
    design_number = int(design_match.group(1)) if design_match else 999999
    height = parse_float(row.get("nominal_height_above_ground_m"))
    if height is None:
        ground_match = re.search(r"m(\d+)p(\d+)", str(row.get("ground_case", "")))
        height = (
            float(f"{ground_match.group(1)}.{ground_match.group(2)}")
            if ground_match
            else 999999.0
        )
    return generation_number, design_number, height, str(row.get("ground_case", ""))


if __name__ == "__main__":
    raise SystemExit(main())
