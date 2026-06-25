from __future__ import annotations

import json
import struct
from collections import Counter, deque
from dataclasses import asdict, dataclass
from math import acos, cos, degrees, sin, sqrt
from pathlib import Path
from typing import Any

from .airfoils import normalized_airfoil
from .config import GeneratorConfig
from .population_generator import winglet_stations


Vec3 = tuple[float, float, float]
Triangle = tuple[int, int, int]


@dataclass(frozen=True)
class StitchedStlStats:
    profiles: int
    profile_points: int
    vertices: int
    triangles: int
    boundary_edges: int
    nonmanifold_edges: int
    oriented_edge_conflicts: int
    connected_components: int
    min_triangle_area_mm2: float
    triangle_aspect_ratio_p95: float
    triangle_aspect_ratio_p99: float
    triangle_aspect_ratio_max: float
    station_band_normal_angle_p95_deg: float
    station_band_normal_angle_p99_deg: float
    station_band_normal_angle_max_deg: float
    volume_mm3: float


def write_stitched_wing_stl(
    design: dict[str, Any],
    config: GeneratorConfig,
    output_path: Path,
) -> StitchedStlStats:
    """Write a deterministic, watertight wing STL from indexed airfoil sections.

    ForgeCAD's level-set loft is closed, but Fusion can display apparent gaps
    where the implicit mesh jumps between profile stations. This exporter uses
    the same design parameters and airfoil point order, then explicitly stitches
    each profile point to the same-index point on the next smoothed station.
    """

    vertices, triangles, stats = build_stitched_wing_mesh(design, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_binary_stl(output_path, vertices, triangles)
    return stats


def write_stitched_wing_files(
    design: dict[str, Any],
    config: GeneratorConfig,
    stl_path: Path,
    obj_path: Path | None = None,
) -> StitchedStlStats:
    vertices, triangles, stats = build_stitched_wing_mesh(design, config)
    stl_path.parent.mkdir(parents=True, exist_ok=True)
    _write_binary_stl(stl_path, vertices, triangles)
    if obj_path is not None:
        obj_path.parent.mkdir(parents=True, exist_ok=True)
        _write_obj(obj_path, vertices, triangles)
    return stats


def write_stitched_wing_obj(
    design: dict[str, Any],
    config: GeneratorConfig,
    output_path: Path,
) -> StitchedStlStats:
    vertices, triangles, stats = build_stitched_wing_mesh(design, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_obj(output_path, vertices, triangles)
    return stats


def build_stitched_wing_mesh(
    design: dict[str, Any],
    config: GeneratorConfig,
) -> tuple[list[Vec3], list[Triangle], StitchedStlStats]:
    control_profiles = _control_profiles(design, config)
    smooth_profiles = _smooth_profiles(
        control_profiles,
        max(1, int(config.forgecad.stitched_stl_subdivisions_per_interval)),
    )
    vertices, triangles = _triangulate_profiles(smooth_profiles)

    volume = _signed_volume(vertices, triangles)
    if volume < 0:
        triangles = [(a, c, b) for a, b, c in triangles]
        volume = -volume
    stats = _mesh_stats(vertices, triangles, len(smooth_profiles), len(smooth_profiles[0]), volume)
    return vertices, triangles, stats


def write_stitched_stl_log(path: Path, stats: StitchedStlStats) -> None:
    path.write_text(
        "returncode: 0\n\n"
        "STDOUT:\n"
        "Wrote deterministic stitched STL from section profiles.\n"
        f"{json.dumps(asdict(stats), indent=2)}\n\n"
        "STDERR:\n",
        encoding="utf-8",
    )


def _control_profiles(design: dict[str, Any], config: GeneratorConfig) -> list[list[Vec3]]:
    sections = list(design["sections"])
    wingtip = design["wingtip"]
    stations = sections + winglet_stations(sections, wingtip, config)
    return [_section_profile_points(station, config) for station in stations]


def _section_profile_points(section: dict[str, Any], config: GeneratorConfig) -> list[Vec3]:
    coords = normalized_airfoil(
        section["airfoil_type"], config.forgecad.stitched_stl_points_per_profile
    )
    thickness = max(y for _, y in coords) - min(y for _, y in coords)
    chord_mm = float(section["chord_length"]) * 1000.0
    target_thickness_m = section.get("target_thickness_m")
    if target_thickness_m is None:
        z_scale = chord_mm
    else:
        z_scale = max(float(target_thickness_m), 1e-9) * 1000.0 / max(thickness, 1e-9)

    pitch = float(section["root_pitch_angle_deg"]) * 3.141592653589793 / 180.0
    c = cos(pitch)
    s = sin(pitch)
    x_offset = float(section["horizontal_displacement"]) * 1000.0
    y_span = float(section["spanwise_position"]) * 1000.0
    z_offset = float(section["vertical_displacement"]) * 1000.0

    points: list[Vec3] = []
    for x, z in coords:
        sx = x * chord_mm
        sz = z * z_scale
        rx = sx * c - sz * s
        rz = sx * s + sz * c
        points.append((rx + x_offset, y_span, rz + z_offset))
    return points


def _smooth_profiles(control_profiles: list[list[Vec3]], subdivisions: int) -> list[list[Vec3]]:
    if len(control_profiles) < 2:
        raise ValueError("Need at least two profiles for STL loft")

    point_count = len(control_profiles[0])
    if any(len(profile) != point_count for profile in control_profiles):
        raise ValueError("All profile stations must have the same point count")

    y_positions = [profile[0][1] for profile in control_profiles]
    if any(b <= a for a, b in zip(y_positions, y_positions[1:])):
        raise ValueError("Profile stations must be strictly increasing in span")

    tangents: list[list[tuple[float, float]]] = []
    for station_index, profile in enumerate(control_profiles):
        station_tangents: list[tuple[float, float]] = []
        for point_index in range(point_count):
            if station_index == 0:
                dy = y_positions[1] - y_positions[0]
                left = control_profiles[0][point_index]
                right = control_profiles[1][point_index]
            elif station_index == len(control_profiles) - 1:
                dy = y_positions[-1] - y_positions[-2]
                left = control_profiles[-2][point_index]
                right = control_profiles[-1][point_index]
            else:
                dy = y_positions[station_index + 1] - y_positions[station_index - 1]
                left = control_profiles[station_index - 1][point_index]
                right = control_profiles[station_index + 1][point_index]
            dy = max(dy, 1e-9)
            station_tangents.append(((right[0] - left[0]) / dy, (right[2] - left[2]) / dy))
        tangents.append(station_tangents)

    profiles: list[list[Vec3]] = []
    for station_index in range(len(control_profiles) - 1):
        a = control_profiles[station_index]
        b = control_profiles[station_index + 1]
        y0 = y_positions[station_index]
        y1 = y_positions[station_index + 1]
        dy = y1 - y0
        start_sample = 0 if station_index == 0 else 1
        for sample in range(start_sample, subdivisions + 1):
            t = sample / subdivisions
            y = y0 + dy * t
            profile: list[Vec3] = []
            for point_index in range(point_count):
                x = _hermite(
                    t,
                    a[point_index][0],
                    tangents[station_index][point_index][0],
                    b[point_index][0],
                    tangents[station_index + 1][point_index][0],
                    dy,
                )
                z = _hermite(
                    t,
                    a[point_index][2],
                    tangents[station_index][point_index][1],
                    b[point_index][2],
                    tangents[station_index + 1][point_index][1],
                    dy,
                )
                profile.append((x, y, z))
            profiles.append(profile)
    return profiles


def _hermite(t: float, p0: float, m0: float, p1: float, m1: float, dx: float) -> float:
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * p0 + h10 * dx * m0 + h01 * p1 + h11 * dx * m1


def _triangulate_profiles(profiles: list[list[Vec3]]) -> tuple[list[Vec3], list[Triangle]]:
    point_count = len(profiles[0])
    vertices = [point for profile in profiles for point in profile]

    def vid(profile_index: int, point_index: int) -> int:
        return profile_index * point_count + point_index % point_count

    triangles: list[Triangle] = []
    for profile_index in range(len(profiles) - 1):
        for point_index in range(point_count):
            a = vid(profile_index, point_index)
            b = vid(profile_index, point_index + 1)
            c = vid(profile_index + 1, point_index)
            d = vid(profile_index + 1, point_index + 1)
            triangles.append((a, c, b))
            triangles.append((b, c, d))

    _add_strip_cap(triangles, 0, point_count, vid, reverse=False)
    _add_strip_cap(triangles, len(profiles) - 1, point_count, vid, reverse=True)

    return vertices, triangles


def _add_strip_cap(
    triangles: list[Triangle],
    profile_index: int,
    point_count: int,
    vid,
    *,
    reverse: bool,
) -> None:
    half = point_count // 2
    for chord_index in range(half - 1):
        upper_a = vid(profile_index, chord_index)
        upper_b = vid(profile_index, chord_index + 1)
        lower_a = vid(profile_index, point_count - 1 - chord_index)
        lower_b = vid(profile_index, point_count - 2 - chord_index)
        if reverse:
            triangles.append((upper_a, lower_a, lower_b))
            triangles.append((upper_a, lower_b, upper_b))
        else:
            triangles.append((upper_a, upper_b, lower_b))
            triangles.append((upper_a, lower_b, lower_a))



def _write_binary_stl(path: Path, vertices: list[Vec3], triangles: list[Triangle]) -> None:
    header = b"genetic-opt-aero stitched wing STL".ljust(80, b" ")
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(struct.pack("<I", len(triangles)))
        for tri in triangles:
            a, b, c = (vertices[index] for index in tri)
            normal = _triangle_normal(a, b, c)
            handle.write(struct.pack("<3f", *normal))
            handle.write(struct.pack("<3f", *a))
            handle.write(struct.pack("<3f", *b))
            handle.write(struct.pack("<3f", *c))
            handle.write(struct.pack("<H", 0))


def _write_obj(path: Path, vertices: list[Vec3], triangles: list[Triangle]) -> None:
    vertex_normals = _vertex_normals(vertices, triangles)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# genetic-opt-aero stitched wing OBJ\n")
        handle.write("# Shared vertices plus averaged normals for Fusion visual QA.\n")
        for x, y, z in vertices:
            handle.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
        for nx, ny, nz in vertex_normals:
            handle.write(f"vn {nx:.9f} {ny:.9f} {nz:.9f}\n")
        handle.write("s 1\n")
        for a, b, c in triangles:
            # OBJ is 1-indexed. Use the matching vertex normal index for smooth display.
            handle.write(
                f"f {a + 1}//{a + 1} {b + 1}//{b + 1} {c + 1}//{c + 1}\n"
            )


def _vertex_normals(vertices: list[Vec3], triangles: list[Triangle]) -> list[Vec3]:
    accum = [[0.0, 0.0, 0.0] for _ in vertices]
    for a, b, c in triangles:
        normal = _triangle_normal_raw(vertices[a], vertices[b], vertices[c])
        for index in (a, b, c):
            accum[index][0] += normal[0]
            accum[index][1] += normal[1]
            accum[index][2] += normal[2]

    normals: list[Vec3] = []
    for x, y, z in accum:
        length = sqrt(x * x + y * y + z * z)
        if length <= 1e-12:
            normals.append((0.0, 0.0, 1.0))
        else:
            normals.append((x / length, y / length, z / length))
    return normals


def _triangle_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3:
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    normal = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    length = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])
    if length <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (normal[0] / length, normal[1] / length, normal[2] / length)


def _triangle_area(a: Vec3, b: Vec3, c: Vec3) -> float:
    normal = _triangle_normal_raw(a, b, c)
    return 0.5 * sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])


def _triangle_normal_raw(a: Vec3, b: Vec3, c: Vec3) -> Vec3:
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    return (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )


def _signed_volume(vertices: list[Vec3], triangles: list[Triangle]) -> float:
    volume = 0.0
    for tri in triangles:
        a, b, c = (vertices[index] for index in tri)
        volume += (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        ) / 6.0
    return volume


def _mesh_stats(
    vertices: list[Vec3],
    triangles: list[Triangle],
    profile_count: int,
    point_count: int,
    volume: float,
) -> StitchedStlStats:
    edge_counts: Counter[tuple[int, int]] = Counter()
    directed_edges: Counter[tuple[int, int]] = Counter()
    edge_faces: dict[tuple[int, int], list[int]] = {}
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(vertices))}
    areas: list[float] = []
    aspect_ratios: list[float] = []
    normals: list[Vec3] = []

    for triangle_index, (a, b, c) in enumerate(triangles):
        areas.append(_triangle_area(vertices[a], vertices[b], vertices[c]))
        aspect_ratios.append(_triangle_aspect_ratio(vertices[a], vertices[b], vertices[c]))
        normals.append(_triangle_normal(vertices[a], vertices[b], vertices[c]))
        for u, v in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((u, v)))
            edge_counts[edge] += 1
            edge_faces.setdefault(edge, []).append(triangle_index)
            directed_edges[(u, v)] += 1
            adjacency[u].add(v)
            adjacency[v].add(u)

    oriented_conflicts = 0
    for u, v in edge_counts:
        if directed_edges[(u, v)] != directed_edges[(v, u)]:
            oriented_conflicts += 1

    station_band_angles: list[float] = []
    side_triangle_count = 2 * (profile_count - 1) * point_count
    for (u, v), faces in edge_faces.items():
        if len(faces) != 2:
            continue
        left, right = faces
        if left >= side_triangle_count or right >= side_triangle_count:
            continue
        if abs(vertices[u][1] - vertices[v][1]) > 1e-6:
            continue
        dot = max(-1.0, min(1.0, sum(normals[left][axis] * normals[right][axis] for axis in range(3))))
        station_band_angles.append(degrees(acos(dot)))

    return StitchedStlStats(
        profiles=profile_count,
        profile_points=point_count,
        vertices=len(vertices),
        triangles=len(triangles),
        boundary_edges=sum(1 for count in edge_counts.values() if count == 1),
        nonmanifold_edges=sum(1 for count in edge_counts.values() if count > 2),
        oriented_edge_conflicts=oriented_conflicts,
        connected_components=_connected_component_count(adjacency),
        min_triangle_area_mm2=min(areas) if areas else 0.0,
        triangle_aspect_ratio_p95=_percentile(aspect_ratios, 95),
        triangle_aspect_ratio_p99=_percentile(aspect_ratios, 99),
        triangle_aspect_ratio_max=max(aspect_ratios) if aspect_ratios else 0.0,
        station_band_normal_angle_p95_deg=_percentile(station_band_angles, 95),
        station_band_normal_angle_p99_deg=_percentile(station_band_angles, 99),
        station_band_normal_angle_max_deg=max(station_band_angles) if station_band_angles else 0.0,
        volume_mm3=volume,
    )


def _triangle_aspect_ratio(a: Vec3, b: Vec3, c: Vec3) -> float:
    lengths = [
        sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2),
        sqrt((b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2 + (b[2] - c[2]) ** 2),
        sqrt((c[0] - a[0]) ** 2 + (c[1] - a[1]) ** 2 + (c[2] - a[2]) ** 2),
    ]
    return max(lengths) / max(min(lengths), 1e-12)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _connected_component_count(adjacency: dict[int, set[int]]) -> int:
    seen: set[int] = set()
    components = 0
    for start in adjacency:
        if start in seen:
            continue
        components += 1
        queue: deque[int] = deque([start])
        seen.add(start)
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
    return components
