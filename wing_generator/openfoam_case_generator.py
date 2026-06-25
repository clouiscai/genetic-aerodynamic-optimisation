from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .cfd_config import DEFAULT_CFD_CONFIG, OpenFoamCaseConfig


@dataclass(frozen=True)
class Bounds3D:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    @property
    def size(self) -> tuple[float, float, float]:
        return tuple(self.maximum[i] - self.minimum[i] for i in range(3))

    @property
    def center(self) -> tuple[float, float, float]:
        return tuple((self.minimum[i] + self.maximum[i]) / 2.0 for i in range(3))


@dataclass(frozen=True)
class CaseInfo:
    design_id: str
    ground_clearance_m: float
    case_dir: Path
    geometry_stl: Path
    geometry_scaled_stl: Path
    wing_bounds_m: Bounds3D
    domain_bounds_m: Bounds3D
    refinement_box_m: Bounds3D
    l_ref_m: float
    area_ref_m2: float
    reynolds_number: float
    kept_triangles: int
    removed_root_triangles: int


def generate_openfoam_cases(config: OpenFoamCaseConfig = DEFAULT_CFD_CONFIG) -> list[CaseInfo]:
    population_dir = config.population_dir.resolve()
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    design_dirs = sorted(path for path in population_dir.glob("design_*") if path.is_dir())
    if not design_dirs:
        raise FileNotFoundError(f"No design_* folders found in {population_dir}")

    cases: list[CaseInfo] = []
    for design_dir in design_dirs:
        stl_path = design_dir / "geometry.stl"
        parameters_path = design_dir / "parameters.json"
        if not stl_path.exists():
            continue
        if not parameters_path.exists():
            raise FileNotFoundError(f"Missing parameters.json beside {stl_path}")

        parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
        triangles = read_stl_triangles(stl_path)
        for clearance_m in config.ground_clearances_m:
            case_dir = output_dir / design_dir.name / ground_level_label(clearance_m)
            if case_dir.exists():
                shutil.rmtree(case_dir)
            case_info = create_single_case(
                case_dir=case_dir,
                design_id=design_dir.name,
                source_stl=stl_path,
                parameters=parameters,
                triangles=triangles,
                ground_clearance_m=clearance_m,
                config=config,
            )
            cases.append(case_info)

    write_case_manifest(output_dir, cases, config)
    return cases


def create_single_case(
    *,
    case_dir: Path,
    design_id: str,
    source_stl: Path,
    parameters: dict,
    triangles: list[tuple[tuple[float, float, float], ...]],
    ground_clearance_m: float,
    config: OpenFoamCaseConfig,
) -> CaseInfo:
    system_dir = case_dir / "system"
    constant_dir = case_dir / "constant"
    tri_surface_dir = constant_dir / "triSurface"
    zero_dir = case_dir / "0"
    include_dir = zero_dir / "include"
    for directory in (system_dir, tri_surface_dir, include_dir):
        directory.mkdir(parents=True, exist_ok=True)

    geometry_triangles_mm, geometry_triangles_m, wing_bounds, kept_count, removed_count = transform_stl_triangles(
        triangles,
        input_scale=config.stl_input_scale_to_m,
        ground_clearance_m=ground_clearance_m,
        root_plane_tolerance_m=config.root_plane_tolerance_m,
    )
    geometry_stl = tri_surface_dir / "geometry.stl"
    geometry_scaled_stl = tri_surface_dir / "geometry_scaled.stl"
    split_no_layer_faces = True
    write_layer_split_ascii_stl(
        geometry_stl,
        geometry_triangles_mm,
        split_no_layer_faces=split_no_layer_faces,
    )
    write_layer_split_ascii_stl(
        geometry_scaled_stl,
        geometry_triangles_m,
        split_no_layer_faces=split_no_layer_faces,
    )

    domain_bounds = domain_bounds_for_wing(wing_bounds, ground_clearance_m, config)
    refinement_box = refinement_box_for_wing(wing_bounds, ground_clearance_m, config)
    l_ref = reference_length(parameters)
    area_ref = reference_area(parameters)
    reynolds = config.solver.velocity_mps * l_ref / config.solver.nu_m2_per_s

    write_text(system_dir / "blockMeshDict", block_mesh_dict(domain_bounds, config))
    write_text(system_dir / "snappyHexMeshDict", snappy_hex_mesh_dict(refinement_box, domain_bounds, config))
    write_text(system_dir / "surfaceFeatureExtractDict", surface_feature_extract_dict())
    write_text(system_dir / "controlDict", control_dict(wing_bounds, l_ref, area_ref, config))
    write_text(system_dir / "fvSchemes", fv_schemes())
    write_text(system_dir / "fvSolution", fv_solution())
    write_text(system_dir / "meshQualityDict", mesh_quality_dict(config))
    write_text(system_dir / "decomposeParDict", decompose_par_dict(config))
    write_text(constant_dir / "transportProperties", transport_properties(config))
    write_text(constant_dir / "turbulenceProperties", turbulence_properties())
    write_zero_files(zero_dir, include_dir, config)
    write_text(case_dir / "Allrun", allrun_script())
    write_text(case_dir / "Allrun-parallel", allrun_parallel_script(config))
    write_text(case_dir / "Allclean", allclean_script())
    write_text(case_dir / "wing.foam", "")

    case_info = CaseInfo(
        design_id=design_id,
        ground_clearance_m=ground_clearance_m,
        case_dir=case_dir,
        geometry_stl=geometry_stl,
        geometry_scaled_stl=geometry_scaled_stl,
        wing_bounds_m=wing_bounds,
        domain_bounds_m=domain_bounds,
        refinement_box_m=refinement_box,
        l_ref_m=l_ref,
        area_ref_m2=area_ref,
        reynolds_number=reynolds,
        kept_triangles=kept_count,
        removed_root_triangles=removed_count,
    )
    write_case_info(case_dir / "case_info.json", case_info, config)
    return case_info


def read_stl_triangles(path: Path) -> list[tuple[tuple[float, float, float], ...]]:
    data = path.read_bytes()
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        expected_size = 84 + triangle_count * 50
        if expected_size == len(data):
            return read_binary_stl(data, triangle_count)
    return read_ascii_stl(data.decode("utf-8", errors="ignore"))


def read_binary_stl(
    data: bytes, triangle_count: int
) -> list[tuple[tuple[float, float, float], ...]]:
    triangles = []
    offset = 84
    for _ in range(triangle_count):
        values = struct.unpack_from("<12fH", data, offset)
        triangles.append(
            (
                (values[3], values[4], values[5]),
                (values[6], values[7], values[8]),
                (values[9], values[10], values[11]),
            )
        )
        offset += 50
    return triangles


def read_ascii_stl(text: str) -> list[tuple[tuple[float, float, float], ...]]:
    vertices: list[tuple[float, float, float]] = []
    triangles = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0] == "vertex":
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            if len(vertices) == 3:
                triangles.append(tuple(vertices))
                vertices = []
    return triangles


def transform_stl_triangles(
    triangles: list[tuple[tuple[float, float, float], ...]],
    *,
    input_scale: float,
    ground_clearance_m: float,
    root_plane_tolerance_m: float,
) -> tuple[
    list[tuple[tuple[float, float, float], ...]],
    list[tuple[tuple[float, float, float], ...]],
    Bounds3D,
    int,
    int,
]:
    del ground_clearance_m

    transformed_mm = []
    transformed_m = []
    removed_root = 0
    for triangle in triangles:
        converted_mm = tuple(forgecad_to_openfoam(vertex, 1.0) for vertex in triangle)
        converted_m = tuple(forgecad_to_openfoam(vertex, input_scale) for vertex in triangle)
        if all(vertex[1] <= root_plane_tolerance_m for vertex in converted_m):
            removed_root += 1
        transformed_mm.append(converted_mm)
        transformed_m.append(converted_m)

    if not transformed_m:
        raise ValueError("All STL triangles were removed as root-plane faces")
    wing_bounds = bounds_from_vertices(vertex for triangle in transformed_m for vertex in triangle)
    return transformed_mm, transformed_m, wing_bounds, len(transformed_m), removed_root


def forgecad_to_openfoam(
    vertex: tuple[float, float, float], input_scale: float
) -> tuple[float, float, float]:
    chord_x, span_y, vertical_z = vertex
    return (
        chord_x * input_scale,
        span_y * input_scale,
        vertical_z * input_scale,
    )


def add_vec(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def bounds_from_vertices(vertices: Iterable[tuple[float, float, float]]) -> Bounds3D:
    iterator = iter(vertices)
    first = next(iterator)
    mins = [first[0], first[1], first[2]]
    maxs = [first[0], first[1], first[2]]
    for vertex in iterator:
        for axis in range(3):
            mins[axis] = min(mins[axis], vertex[axis])
            maxs[axis] = max(maxs[axis], vertex[axis])
    return Bounds3D(tuple(mins), tuple(maxs))


def write_ascii_stl(
    path: Path, triangles: list[tuple[tuple[float, float, float], ...]], solid_name: str
) -> None:
    lines = [f"solid {solid_name}"]
    for triangle in triangles:
        normal = triangle_normal(*triangle)
        lines.append(f"  facet normal {normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g}")
        lines.append("    outer loop")
        for vertex in triangle:
            lines.append(f"      vertex {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {solid_name}")
    write_text(path, "\n".join(lines) + "\n")


def write_layer_split_ascii_stl(
    path: Path,
    triangles: list[tuple[tuple[float, float, float], ...]],
    *,
    split_no_layer_faces: bool = True,
) -> None:
    """Write aerodynamic skin and no-layer closure STL regions.

    The wing skin remains one patch. Root and trailing-edge closure faces keep
    the triSurface closed for snappy, but they do not receive prism layers.
    """

    if not split_no_layer_faces:
        write_ascii_stl(path, triangles, solid_name="wing")
        return

    regions: dict[str, list[tuple[tuple[float, float, float], ...]]] = {
        "wing": [],
        "wingRoot": [],
    }
    x_min, x_max = _triangle_bounds(triangles, axis=0)
    y_min, y_max = _triangle_bounds(triangles, axis=1)
    z_min, z_max = _triangle_bounds(triangles, axis=2)
    chord = x_max - x_min
    le_threshold = x_min + 0.015 * chord
    te_threshold = x_min + 0.985 * chord
    for triangle in triangles:
        if all(abs(vertex[1]) <= 1e-9 for vertex in triangle):
            regions["wingRoot"].append(triangle)
        else:
            regions["wing"].append(triangle)

    lines: list[str] = []
    for solid_name, region_triangles in regions.items():
        lines.append(f"solid {solid_name}")
        for triangle in region_triangles:
            normal = triangle_normal(*triangle)
            lines.append(f"  facet normal {normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g}")
            lines.append("    outer loop")
            for vertex in triangle:
                lines.append(f"      vertex {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}")
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append(f"endsolid {solid_name}")
    write_text(path, "\n".join(lines) + "\n")


def _triangle_bounds(
    triangles: list[tuple[tuple[float, float, float], ...]], axis: int
) -> tuple[float, float]:
    values = [vertex[axis] for triangle in triangles for vertex in triangle]
    return min(values), max(values)


def _is_local_layer_problem_area(
    triangle: tuple[tuple[float, float, float], ...],
    *,
    x_min: float,
    x_span: float,
    y_min: float,
    y_span: float,
    z_min: float,
    z_span: float,
) -> bool:
    centroid = tuple(sum(vertex[axis] for vertex in triangle) / 3.0 for axis in range(3))
    xn = (centroid[0] - x_min) / x_span
    yn = (centroid[1] - y_min) / y_span
    zn = (centroid[2] - z_min) / z_span
    if zn > 0.45:
        return False
    local_boxes = (
        (0.025, 0.125, 0.810, 0.980),
        (0.035, 0.075, 0.690, 0.740),
        (0.070, 0.105, 0.580, 0.620),
        (0.185, 0.260, 0.925, 1.000),
        (0.285, 0.320, 0.850, 0.895),
        (0.375, 0.415, 0.770, 0.805),
        (0.47, 0.49, 0.86, 0.89),
        (0.545, 0.575, 0.590, 0.615),
        (0.55, 0.59, 0.69, 0.72),
        (0.640, 0.660, 0.460, 0.480),
        (0.825, 0.855, 0.195, 0.245),
        (0.85, 0.88, 0.18, 0.21),
        (0.975, 1.000, 0.400, 0.430),
        (0.94, 0.97, 0.00, 0.02),
        (0.86, 0.97, 0.05, 0.18),
    )
    return any(x0 <= xn <= x1 and y0 <= yn <= y1 for x0, x1, y0, y1 in local_boxes)


def triangle_normal(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> tuple[float, float, float]:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    normal = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
    length = math.sqrt(sum(component * component for component in normal))
    if length <= 1e-20:
        return (0.0, 0.0, 0.0)
    return tuple(component / length for component in normal)


def domain_bounds_for_wing(
    wing_bounds: Bounds3D, ground_z_m: float, config: OpenFoamCaseConfig
) -> Bounds3D:
    return Bounds3D(
        (wing_bounds.minimum[0] - config.domain.front_m, 0.0, ground_z_m),
        (
            wing_bounds.maximum[0] + config.domain.back_m,
            wing_bounds.maximum[1] + config.domain.side_m,
            max(0.0, wing_bounds.maximum[2]) + config.domain.top_m,
        ),
    )


def refinement_box_for_wing(
    wing_bounds: Bounds3D, ground_z_m: float, config: OpenFoamCaseConfig
) -> Bounds3D:
    x_min = wing_bounds.minimum[0] - config.refinement.front_m
    x_max = wing_bounds.maximum[0] + config.refinement.back_m
    y_min = 0.0
    y_max = wing_bounds.maximum[1] + config.refinement.side_m
    z_min = ground_z_m
    z_max = wing_bounds.maximum[2] + config.refinement.top_m
    return Bounds3D(
        (x_min, y_min, z_min),
        (x_max, y_max, z_max),
    )


def reference_length(parameters: dict) -> float:
    sections = parameters.get("sections", [])
    if not sections:
        return 1.2
    return float(sections[0].get("chord_length", 1.2))


def reference_area(parameters: dict) -> float:
    sections = parameters.get("sections", [])
    if len(sections) < 2:
        return reference_length(parameters) * float(parameters.get("total_span", 1.5))
    area = 0.0
    for left, right in zip(sections, sections[1:]):
        dy = float(right["spanwise_position"]) - float(left["spanwise_position"])
        area += 0.5 * (float(left["chord_length"]) + float(right["chord_length"])) * dy
    return max(area, 1e-9)


def block_mesh_dict(domain: Bounds3D, config: OpenFoamCaseConfig) -> str:
    nx, ny, nz = cell_counts(domain, config)
    grading = block_mesh_grading(domain, config)
    x0, y0, z0 = domain.minimum
    x1, y1, z1 = domain.maximum
    return foam_header("dictionary", "blockMeshDict") + f"""
scale   1;

vertices
(
    ({x0:.9g} {y0:.9g} {z0:.9g})
    ({x1:.9g} {y0:.9g} {z0:.9g})
    ({x1:.9g} {y1:.9g} {z0:.9g})
    ({x0:.9g} {y1:.9g} {z0:.9g})
    ({x0:.9g} {y0:.9g} {z1:.9g})
    ({x1:.9g} {y0:.9g} {z1:.9g})
    ({x1:.9g} {y1:.9g} {z1:.9g})
    ({x0:.9g} {y1:.9g} {z1:.9g})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading {grading}
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces ((0 4 7 3));
    }}
    outlet
    {{
        type patch;
        faces ((2 6 5 1));
    }}
    ground
    {{
        type wall;
        faces ((0 3 2 1));
    }}
    slippingWalls
    {{
        type patch;
        faces
        (
            (4 5 6 7)
            (3 7 6 2)
        );
    }}
    symmetry
    {{
        type symmetry;
        faces ((1 5 4 0));
    }}
);
"""


def _ground_z_distribution(height: float, config: OpenFoamCaseConfig) -> tuple[int, float, int]:
    """Vertical cell layout for a smoothly graded ground boundary layer.

    Region 1 grows geometrically from ``ground_first_layer_height_m`` up to the
    base cell size, so its last cell matches the bulk and there is no size jump.
    Region 2 fills the remaining height with uniform base cells. Returns
    ``(n_bl_cells, bl_height_m, n_uniform_cells)``.
    """
    bl = config.boundary_layer
    base = config.refinement.base_cell_size_m
    first = bl.ground_first_layer_height_m
    growth = bl.growth_rate
    if first <= 0.0 or first >= base or growth <= 1.0:
        return 0, 0.0, max(1, math.ceil(height / base))
    n_bl = max(1, math.ceil(math.log(base / first) / math.log(growth)))
    bl_height = min(geometric_series_total(first, growth, n_bl), 0.6 * height)
    n_uniform = max(1, round((height - bl_height) / base))
    return n_bl, bl_height, n_uniform


def cell_counts(domain: Bounds3D, config: OpenFoamCaseConfig) -> tuple[int, int, int]:
    base = config.refinement.base_cell_size_m
    nx = max(1, math.ceil(domain.size[0] / base))
    ny = max(1, math.ceil(domain.size[1] / base))
    if config.boundary_layer.ground_layers_from_block_mesh:
        n_bl, _, n_uniform = _ground_z_distribution(domain.size[2], config)
        nz = n_bl + n_uniform
    else:
        nz = max(1, math.ceil(domain.size[2] / base))
    return nx, ny, nz


def block_mesh_grading(domain: Bounds3D, config: OpenFoamCaseConfig) -> str:
    if not config.boundary_layer.ground_layers_from_block_mesh:
        return "(1 1 1)"

    height = domain.size[2]
    n_bl, bl_height, n_uniform = _ground_z_distribution(height, config)
    if n_bl <= 0:
        return "(1 1 1)"

    growth = config.boundary_layer.growth_rate
    total_cells = n_bl + n_uniform
    bl_length_fraction = bl_height / height
    far_length_fraction = 1.0 - bl_length_fraction
    bl_cell_fraction = n_bl / total_cells
    far_cell_fraction = n_uniform / total_cells
    bl_expansion = growth ** max(0, n_bl - 1)

    return f"""(
        1
        1
        (
            ({bl_length_fraction:.9g} {bl_cell_fraction:.9g} {bl_expansion:.9g})
            ({far_length_fraction:.9g} {far_cell_fraction:.9g} 1)
        )
    )"""


def geometric_series_total(first: float, growth: float, count: int) -> float:
    if count <= 0:
        return 0.0
    if abs(growth - 1.0) < 1e-12:
        return first * count
    return first * (growth**count - 1.0) / (growth - 1.0)


def snappy_hex_mesh_dict(
    refinement_box: Bounds3D, domain: Bounds3D, config: OpenFoamCaseConfig
) -> str:
    r = config.refinement
    bl = config.boundary_layer
    add_layers = bl.n_layers > 0 or bl.snappy_ground_layers
    ground_layer_entry = f"""
        ground
        {{
            nSurfaceLayers {bl.n_layers};
            firstLayerThickness {bl.ground_first_layer_height_m};
            minThickness {max(bl.ground_first_layer_height_m * 0.1, 1e-8)};
            expansionRatio {bl.growth_rate};
        }}""" if bl.snappy_ground_layers else ""
    split_no_layer_faces = True
    geometry_regions = """
        regions
        {
            wing { name wing; }
            wingRoot { name wingRoot; }
        }""" if split_no_layer_faces else ""
    surface_regions = f"""
            regions
            {{
                wing
                {{
                    level ({r.surface_min_level} {r.surface_max_level});
                    patchInfo
                    {{
                        type wall;
                        inGroups (wingGroup);
                    }}
                }}
                wingRoot
                {{
                    level ({r.surface_min_level} {r.surface_max_level});
                    patchInfo
                    {{
                        type symmetry;
                    }}
                }}
            }}""" if split_no_layer_faces else ""
    location = (
        domain.minimum[0] + min(0.5, 0.1 * domain.size[0]),
        domain.maximum[1] - min(0.5, 0.1 * domain.size[1]),
        domain.minimum[2] + min(0.5, 0.1 * domain.size[2]),
    )
    return foam_header("dictionary", "snappyHexMeshDict") + f"""
castellatedMesh true;
snap            true;
addLayers       {str(add_layers).lower()};

geometry
{{
    geometry_scaled.stl
    {{
        type triSurfaceMesh;
        name wing;
{geometry_regions}
    }}

    refinementBox
    {{
        type box;
        min {foam_vec(refinement_box.minimum)};
        max {foam_vec(refinement_box.maximum)};
    }}
}}

castellatedMeshControls
{{
    maxLocalCells       {r.max_local_cells};
    maxGlobalCells      {r.max_global_cells};
    minRefinementCells  10;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 3;

    features
    (
    );

    refinementSurfaces
    {{
        wing
        {{
            level ({r.surface_min_level} {r.surface_max_level});
{surface_regions}
            patchInfo
            {{
                type wall;
                inGroups (wingGroup);
            }}
        }}
    }}

    resolveFeatureAngle 60;

    refinementRegions
    {{
        refinementBox
        {{
            mode inside;
            levels ((1e15 {r.refinement_box_level}));
        }}
    }}

    locationInMesh {foam_vec(location)};
    allowFreeStandingZoneFaces false;
}}

snapControls
{{
    nSmoothPatch        3;
    tolerance           4.0;
    nSolveIter          30;
    nRelaxIter          5;
}}

addLayersControls
{{
    relativeSizes       false;
    expansionRatio      {bl.growth_rate};
    firstLayerThickness {bl.first_layer_height_m};
    minThickness        {max(bl.first_layer_height_m * 0.1, 1e-8)};

    layers
    {{
        wing
        {{
            nSurfaceLayers {bl.n_layers};
            firstLayerThickness {bl.first_layer_height_m};
            minThickness {max(bl.first_layer_height_m * 0.1, 1e-8)};
            expansionRatio {bl.growth_rate};
        }}
{ground_layer_entry}
    }}

    nGrow                   0;
    featureAngle            60;
    nRelaxIter              3;
    nSmoothSurfaceNormals   1;
    nSmoothNormals          3;
    nSmoothThickness        10;
    maxFaceThicknessRatio   0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle        90;
    nBufferCellsNoExtrude   2;
    nLayerIter              50;
    meshShrinker            displacementMotionSolver;
    solver                  displacementLaplacian;
    displacementLaplacianCoeffs
    {{
        diffusivity quadratic inverseDistance (ground wing);
    }}
}}

meshQualityControls
{{
    #include "meshQualityDict"
    maxNonOrtho 85;
    maxInternalSkewness 4;
    maxBoundarySkewness 4;
    minVol 1e-20;
    minTetQuality -1e30;
    minDeterminant -1e30;
    minFaceWeight -1;
    minVolRatio 1e-12;
    minTwist -1;
}}

writeFlags
(
    scalarLevels
    layerSets
    layerFields
);

mergeTolerance 1e-6;
"""


def surface_feature_extract_dict() -> str:
    return foam_header("dictionary", "surfaceFeatureExtractDict") + """
geometry_scaled.stl
{
    extractionMethod    extractFromSurface;
    includedAngle       150;

    subsetFeatures
    {
        nonManifoldEdges no;
        openEdges        yes;
    }

    writeObj            yes;
}
"""


def control_dict(
    wing_bounds: Bounds3D, l_ref: float, area_ref: float, config: OpenFoamCaseConfig
) -> str:
    solver = config.solver
    cof_r = (0.25 * l_ref, 0.0, 0.0)
    return foam_header("dictionary", "controlDict") + f"""
application     simpleFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {solver.end_time};
deltaT          1;

writeControl    timeStep;
writeInterval   {solver.write_interval};
purgeWrite      0;

writeFormat     ascii;
writePrecision  8;
writeCompression off;

timeFormat      general;
timePrecision   8;

runTimeModifiable true;

functions
{{
    wingForceCoeffs
    {{
        type            forceCoeffs;
        libs            (forces);
        writeControl    timeStep;
        timeInterval    1;
        log             yes;

        patches         (wing);
        coefficients    (Cd Cl CmPitch);
        rho             rhoInf;
        rhoInf          {solver.rho_kg_per_m3:.9g};
        liftDir         (0 0 1);
        dragDir         (1 0 0);
        CofR            {foam_vec(cof_r)};
        pitchAxis       (0 1 0);
        magUInf         {solver.velocity_mps:.9g};
        lRef            {l_ref:.9g};
        Aref            {area_ref:.9g};
    }}

    yPlus
    {{
        type            yPlus;
        libs            (fieldFunctionObjects);
        patches         (wing ground);
        writeFields     yes;
        writeControl    writeTime;
    }}

    wallShearStress
    {{
        type            wallShearStress;
        libs            (fieldFunctionObjects);
        patches         (wing ground);
        writeControl    writeTime;
    }}
}}
"""


def write_zero_files(zero_dir: Path, include_dir: Path, config: OpenFoamCaseConfig) -> None:
    solver = config.solver
    write_text(
        include_dir / "initialConditions",
        foam_header("", "")[:0]
        + f"""
flowVelocity         ({solver.velocity_mps:.9g} 0 0);
pressure             0;
turbulentKE          {solver.turbulent_ke:.9g};
turbulentOmega       {solver.turbulent_omega:.9g};
""".lstrip(),
    )
    write_text(zero_dir / "U", field_u())
    write_text(zero_dir / "p", field_p())
    write_text(zero_dir / "k", field_k())
    write_text(zero_dir / "omega", field_omega())
    write_text(zero_dir / "nut", field_nut())


def field_u() -> str:
    return field_header("volVectorField", "U") + """
#include "include/initialConditions"

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform $flowVelocity;

boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"

    inlet          { type fixedValue; value uniform $flowVelocity; }
    outlet         { type inletOutlet; inletValue uniform (0 0 0); value $internalField; }
    ground         { type fixedValue; value uniform $flowVelocity; }
    slippingWalls  { type slip; }
    symmetry       { type symmetry; value $internalField; }
    wingRoot       { type symmetry; value $internalField; }
    wing           { type noSlip; }
}
"""


def field_p() -> str:
    return field_header("volScalarField", "p") + """
#include "include/initialConditions"

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform $pressure;

boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"

    inlet          { type zeroGradient; }
    outlet         { type fixedValue; value uniform $pressure; }
    ground         { type zeroGradient; }
    slippingWalls  { type zeroGradient; }
    symmetry       { type symmetry; }
    wingRoot       { type symmetry; }
    wing           { type zeroGradient; }
}
"""


def field_k() -> str:
    return field_header("volScalarField", "k") + """
#include "include/initialConditions"

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform $turbulentKE;

boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"

    inlet          { type fixedValue; value uniform $turbulentKE; }
    outlet         { type inletOutlet; inletValue uniform $turbulentKE; value uniform $turbulentKE; }
    ground         { type kqRWallFunction; value uniform $turbulentKE; }
    slippingWalls  { type slip; }
    symmetry       { type symmetry; }
    wingRoot       { type symmetry; }
    wing           { type kqRWallFunction; value uniform $turbulentKE; }
}
"""


def field_omega() -> str:
    return field_header("volScalarField", "omega") + """
#include "include/initialConditions"

dimensions      [0 0 -1 0 0 0 0];

internalField   uniform $turbulentOmega;

boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"

    inlet          { type fixedValue; value uniform $turbulentOmega; }
    outlet         { type inletOutlet; inletValue uniform $turbulentOmega; value uniform $turbulentOmega; }
    ground         { type omegaWallFunction; value uniform $turbulentOmega; }
    slippingWalls  { type slip; }
    symmetry       { type symmetry; }
    wingRoot       { type symmetry; }
    wing           { type omegaWallFunction; value uniform $turbulentOmega; }
}
"""


def field_nut() -> str:
    return field_header("volScalarField", "nut") + """
dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"

    inlet          { type calculated; value uniform 0; }
    outlet         { type calculated; value uniform 0; }
    ground         { type nutUSpaldingWallFunction; value uniform 0; }
    slippingWalls  { type slip; }
    symmetry       { type symmetry; }
    wingRoot       { type symmetry; }
    wing           { type nutUSpaldingWallFunction; value uniform 0; }
}
"""


def fv_schemes() -> str:
    return foam_header("dictionary", "fvSchemes") + """
ddtSchemes
{
    default steadyState;
}

gradSchemes
{
    default Gauss linear;
    grad(U) cellLimited Gauss linear 1;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss upwind;
    turbulence      bounded Gauss upwind;
    div(phi,k)      $turbulence;
    div(phi,omega)  $turbulence;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default Gauss linear limited corrected 0.333;
}

interpolationSchemes
{
    default linear;
}

snGradSchemes
{
    default limited corrected 0.333;
}

wallDist
{
    method meshWave;
}
"""


def fv_solution() -> str:
    return foam_header("dictionary", "fvSolution") + """
solvers
{
    "cellDisplacement.*"
    {
        solver          GAMG;
        tolerance       1e-8;
        relTol          0;
        smoother        GaussSeidel;
    }

    p
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-7;
        relTol          0.01;
        maxIter         2000;
    }

    Phi { $p; }

    U
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
        nSweeps         1;
    }

    k
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
        nSweeps         1;
    }

    omega
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
        nSweeps         1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 2;
    consistent no;
}

potentialFlow
{
    nNonOrthogonalCorrectors 10;
}

relaxationFactors
{
    equations
    {
        U     0.2;
        k     0.1;
        omega 0.1;
    }
    fields
    {
        p     0.05;
    }
}

cache
{
    grad(U);
}
"""


def mesh_quality_dict(config: OpenFoamCaseConfig) -> str:
    return """
maxNonOrtho 65;
maxBoundarySkewness 4;
maxInternalSkewness 4;
maxConcave 80;
minFlatness 0.5;
minVol 1e-13;
minTetQuality 1e-30;
minArea -1;
minTwist 0.02;
minDeterminant 0.001;
minFaceWeight 0.02;
minVolRatio 0.01;
minTriangleTwist -1;
nSmoothScale 4;
errorReduction 0.75;
"""


def decompose_par_dict(config: OpenFoamCaseConfig) -> str:
    return foam_header("dictionary", "decomposeParDict") + f"""
numberOfSubdomains {config.solver.n_subdomains};

method          scotch;
"""


def transport_properties(config: OpenFoamCaseConfig) -> str:
    return foam_header("dictionary", "transportProperties") + f"""
transportModel  Newtonian;

nu              {config.solver.nu_m2_per_s:.9g};
"""


def turbulence_properties() -> str:
    return foam_header("dictionary", "turbulenceProperties") + """
simulationType RAS;

RAS
{
    RASModel        kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}
"""


def allrun_script() -> str:
    return """#!/usr/bin/env bash
cd "${0%/*}" || exit
. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions
set -eo pipefail
#------------------------------------------------------------------------------

bash ./Allclean
touch wing.foam

runApplication surfaceConvert -scale 0.001 constant/triSurface/geometry.stl constant/triSurface/geometry_scaled.stl
runApplication surfaceFeatureExtract
runApplication blockMesh
checkMesh 2>&1 | tee log.checkMesh.initial || true
runApplication snappyHexMesh -overwrite
checkMesh -constant 2>&1 | tee log.checkMesh.constant || true
runApplication potentialFoam -writephi
simpleFoam 2>&1 | tee log.simpleFoam

#------------------------------------------------------------------------------
"""


def allrun_parallel_script(config: OpenFoamCaseConfig) -> str:
    n = config.solver.n_subdomains
    return f"""#!/usr/bin/env bash
cd "${{0%/*}}" || exit
. ${{WM_PROJECT_DIR:?}}/bin/tools/RunFunctions
set -eo pipefail
#------------------------------------------------------------------------------

bash ./Allclean
touch wing.foam

runApplication surfaceConvert -scale 0.001 constant/triSurface/geometry.stl constant/triSurface/geometry_scaled.stl
runApplication surfaceFeatureExtract
runApplication blockMesh
checkMesh 2>&1 | tee log.checkMesh.initial || true
decomposePar 2>&1 | tee log.decomposePar.initial
mpirun -np {n} snappyHexMesh -parallel -overwrite 2>&1 | tee log.snappyHexMesh
runApplication reconstructParMesh -constant
checkMesh -constant 2>&1 | tee log.checkMesh.constant || true
rm -rf processor*
decomposePar -force 2>&1 | tee log.decomposePar.final
mpirun -np {n} potentialFoam -parallel -writephi 2>&1 | tee log.potentialFoam
mpirun -np {n} simpleFoam -parallel 2>&1 | tee log.simpleFoam
runApplication reconstructPar -latestTime

#------------------------------------------------------------------------------
"""


def allclean_script() -> str:
    return """#!/usr/bin/env bash
cd "${0%/*}" || exit
set -euo pipefail

# Remove artifacts created by Allrun/Allrun-parallel while keeping the case
# inputs: 0/, constant/triSurface/geometry.stl, system/, and run scripts.
rm -rf processor* postProcessing dynamicCode VTK
rm -rf constant/polyMesh constant/extendedFeatureEdgeMesh
rm -f log.* wing.foam
rm -f constant/triSurface/geometry_scaled.stl
rm -f constant/triSurface/geometry_scaled.eMesh
rm -f constant/triSurface/geometry_scaled*.obj

for time_dir in ./*; do
    name="${time_dir#./}"
    if [[ -d "$time_dir" && "$name" =~ ^[0-9]+([.][0-9]+)?$ && "$name" != "0" ]]; then
        rm -rf -- "$time_dir"
    fi
done
"""


def foam_header(foam_class: str, object_name: str) -> str:
    return f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2412                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {foam_class};
    object      {object_name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""


def field_header(foam_class: str, object_name: str) -> str:
    return foam_header(foam_class, object_name)


def foam_vec(values: tuple[float, float, float]) -> str:
    return f"({values[0]:.9g} {values[1]:.9g} {values[2]:.9g})"


def ground_level_label(ground_z_m: float) -> str:
    prefix = "ground_m" if ground_z_m < 0 else "ground_p"
    return prefix + f"{abs(ground_z_m):.2f}".replace(".", "p")


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8", newline="\n")


def write_case_info(path: Path, case_info: CaseInfo, config: OpenFoamCaseConfig) -> None:
    payload = {
        "design_id": case_info.design_id,
        "ground_plane_z_m": case_info.ground_clearance_m,
        "nominal_height_above_ground_m": abs(case_info.ground_clearance_m),
        "case_dir": str(case_info.case_dir),
        "geometry_stl": str(case_info.geometry_stl),
        "geometry_scaled_stl": str(case_info.geometry_scaled_stl),
        "wing_bounds_m": asdict(case_info.wing_bounds_m),
        "domain_bounds_m": asdict(case_info.domain_bounds_m),
        "refinement_box_m": asdict(case_info.refinement_box_m),
        "l_ref_m": case_info.l_ref_m,
        "area_ref_m2": case_info.area_ref_m2,
        "reynolds_number": case_info.reynolds_number,
        "kept_triangles": case_info.kept_triangles,
        "removed_root_triangles": case_info.removed_root_triangles,
        "solver": asdict(config.solver),
        "refinement": {
            **asdict(config.refinement),
            "actual_refined_to_base_ratio": config.refinement.actual_refined_to_base_ratio,
        },
        "boundary_layer": asdict(config.boundary_layer),
    }
    write_text(path, json.dumps(payload, indent=2))


def write_case_manifest(
    output_dir: Path, cases: list[CaseInfo], config: OpenFoamCaseConfig
) -> None:
    rows = []
    for case in cases:
        rows.append(
            {
                "design_id": case.design_id,
                "ground_plane_z_m": case.ground_clearance_m,
                "nominal_height_above_ground_m": abs(case.ground_clearance_m),
                "case_dir": str(case.case_dir),
                "l_ref_m": case.l_ref_m,
                "area_ref_m2": case.area_ref_m2,
                "reynolds_number": case.reynolds_number,
                "kept_triangles": case.kept_triangles,
                "removed_root_triangles": case.removed_root_triangles,
            }
        )
    write_text(
        output_dir / "case_manifest.json",
        json.dumps(
            {
                "num_cases": len(cases),
                "population_dir": str(config.population_dir.resolve()),
                "ground_plane_z_values_m": list(config.ground_clearances_m),
                "settings": {
                    "domain": asdict(config.domain),
                    "refinement": {
                        **asdict(config.refinement),
                        "actual_refined_to_base_ratio": config.refinement.actual_refined_to_base_ratio,
                    },
                    "boundary_layer": asdict(config.boundary_layer),
                    "solver": asdict(config.solver),
                },
                "cases": rows,
            },
            indent=2,
        ),
    )
    with (output_dir / "case_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["design_id", "ground_clearance_m", "case_dir"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_text(output_dir / "run_all_cases.sh", run_all_cases_script(cases, output_dir))


def run_all_cases_script(cases: list[CaseInfo], output_dir: Path) -> str:
    lines = [
        "#!/usr/bin/env bash",
        'cd "${0%/*}" || exit',
        "set -euo pipefail",
        "",
    ]
    for case in cases:
        rel = case.case_dir.relative_to(output_dir)
        lines.append(f'(cd "{rel.as_posix()}" && bash ./Allrun-parallel)')
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate OpenFOAM cases for wing STL population.")
    parser.add_argument("--population-dir", type=Path, default=DEFAULT_CFD_CONFIG.population_dir)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CFD_CONFIG.output_dir)
    parser.add_argument(
        "--ground-clearances",
        default=",".join(str(value) for value in DEFAULT_CFD_CONFIG.ground_clearances_m),
        help="Comma-separated ground clearances in metres.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = OpenFoamCaseConfig(
        population_dir=args.population_dir,
        output_dir=args.output_dir,
        ground_clearances_m=tuple(float(value) for value in args.ground_clearances.split(",")),
    )
    cases = generate_openfoam_cases(config)
    print(f"Generated {len(cases)} OpenFOAM cases in {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
