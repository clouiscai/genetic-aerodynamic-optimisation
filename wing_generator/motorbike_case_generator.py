from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass, replace
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


@dataclass(frozen=True)
class ObjMesh:
    vertices: list[tuple[float, float, float]]
    faces: list[tuple[int, int, int]]


@dataclass(frozen=True)
class MotorBikeCaseInfo:
    design_id: str
    ground_clearance_m: float
    case_dir: Path
    source_obj: Path
    wing_obj: Path
    wing_bounds_m: Bounds3D
    domain_bounds_m: Bounds3D
    refinement_box_m: Bounds3D
    l_ref_m: float
    area_ref_m2: float
    reynolds_number: float
    wing_faces: int
    root_faces: int


def generate_motorbike_cases(
    config: OpenFoamCaseConfig = DEFAULT_CFD_CONFIG,
) -> list[MotorBikeCaseInfo]:
    population_dir = config.population_dir.resolve()
    output_dir = config.output_dir.resolve()
    validate_reference_case(config.reference_case_dir)
    if config.solver.n_subdomains < 1:
        raise ValueError("The core count must be at least 1.")
    output_dir.mkdir(parents=True, exist_ok=True)

    design_dirs = sorted(path for path in population_dir.glob("design_*") if path.is_dir())
    if not design_dirs:
        raise FileNotFoundError(f"No design_* folders found in {population_dir}")

    cases: list[MotorBikeCaseInfo] = []
    for design_dir in design_dirs:
        source_obj = design_dir / "geometry.obj"
        parameters_path = design_dir / "parameters.json"
        if not source_obj.exists():
            raise FileNotFoundError(f"Missing geometry.obj in {design_dir}")
        if not parameters_path.exists():
            raise FileNotFoundError(f"Missing parameters.json beside {source_obj}")

        parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
        mesh = read_obj_mesh(source_obj)
        for clearance_m in config.ground_clearances_m:
            case_dir = output_dir / design_dir.name / ground_level_label(clearance_m)
            case_info = create_motorbike_case(
                case_dir=case_dir,
                design_id=design_dir.name,
                source_obj=source_obj,
                parameters=parameters,
                mesh=mesh,
                ground_clearance_m=clearance_m,
                config=config,
            )
            cases.append(case_info)

    write_case_manifest(output_dir, cases, config)
    write_root_scripts(output_dir, config)
    return cases


def create_motorbike_case(
    *,
    case_dir: Path,
    design_id: str,
    source_obj: Path,
    parameters: dict,
    mesh: ObjMesh,
    ground_clearance_m: float,
    config: OpenFoamCaseConfig,
) -> MotorBikeCaseInfo:
    if case_dir.exists():
        shutil.rmtree(case_dir)
    copy_motorbike_reference(config.reference_case_dir.resolve(), case_dir)

    system_dir = case_dir / "system"
    constant_dir = case_dir / "constant"
    tri_surface_dir = constant_dir / "triSurface"
    zero_orig_dir = case_dir / "0.orig"
    include_dir = zero_orig_dir / "include"

    tri_surface_dir.mkdir(parents=True, exist_ok=True)
    include_dir.mkdir(parents=True, exist_ok=True)

    wing_obj = tri_surface_dir / "wing.obj"
    wing_bounds, wing_faces, root_faces = write_scaled_grouped_obj(
        wing_obj,
        mesh,
        input_scale=config.stl_input_scale_to_m,
        root_plane_tolerance_m=config.root_plane_tolerance_m,
    )
    domain_bounds = domain_bounds_for_wing(wing_bounds, ground_clearance_m, config)
    refinement_box = refinement_box_for_wing(wing_bounds, ground_clearance_m, config)
    l_ref = reference_length(parameters)
    area_ref = reference_area(parameters)
    reynolds = config.solver.velocity_mps * l_ref / config.solver.nu_m2_per_s

    write_text(system_dir / "blockMeshDict", block_mesh_dict(domain_bounds, config))
    write_text(
        system_dir / "snappyHexMeshDict",
        snappy_hex_mesh_dict(refinement_box, domain_bounds, wing_bounds, ground_clearance_m, config),
    )
    write_text(system_dir / "surfaceFeatureExtractDict", surface_feature_extract_dict())
    write_text(system_dir / "controlDict", control_dict(config))
    write_text(system_dir / "forceCoeffs", force_coeffs_dict(wing_bounds, l_ref, area_ref, config))
    write_text(system_dir / "topoSetDict", topo_set_dict(refinement_box))
    write_text(system_dir / "decomposeParDict", decompose_par_dict(config))
    write_text(constant_dir / "transportProperties", transport_properties(config))
    write_text(zero_orig_dir / "include" / "initialConditions", initial_conditions(config))
    write_zero_files(zero_orig_dir, config)
    write_text(case_dir / "Allrun", allrun_script())
    write_text(case_dir / "Allclean", allclean_script())
    write_text(case_dir / "wing.foam", "")

    case_info = MotorBikeCaseInfo(
        design_id=design_id,
        ground_clearance_m=ground_clearance_m,
        case_dir=case_dir,
        source_obj=source_obj,
        wing_obj=wing_obj,
        wing_bounds_m=wing_bounds,
        domain_bounds_m=domain_bounds,
        refinement_box_m=refinement_box,
        l_ref_m=l_ref,
        area_ref_m2=area_ref,
        reynolds_number=reynolds,
        wing_faces=wing_faces,
        root_faces=root_faces,
    )
    write_case_info(case_dir / "case_info.json", case_info, config)
    return case_info


def copy_motorbike_reference(reference_dir: Path, case_dir: Path) -> None:
    if not reference_dir.exists():
        raise FileNotFoundError(f"motorBike reference case not found: {reference_dir}")
    shutil.copytree(reference_dir, case_dir)

    removable_dirs = [
        case_dir / "0",
        case_dir / "constant" / "polyMesh",
        case_dir / "constant" / "extendedFeatureEdgeMesh",
        case_dir / "constant" / "triSurface",
        case_dir / "postProcessing",
        case_dir / "dynamicCode",
        case_dir / "VTK",
    ]
    removable_dirs.extend(case_dir.glob("processor*"))
    for path in removable_dirs:
        if path.exists():
            shutil.rmtree(path)

    for child in case_dir.iterdir():
        if child.is_dir() and is_time_directory(child.name):
            shutil.rmtree(child)
    for log_file in case_dir.glob("log.*"):
        log_file.unlink()


def validate_reference_case(reference_dir: Path) -> None:
    required = ("0.orig", "constant", "system/fvSchemes", "system/fvSolution", "system/meshQualityDict")
    missing = [name for name in required if not (reference_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete motorBike reference case at {reference_dir}: missing {', '.join(missing)}. "
            "Pass --reference-case, set OPENFOAM_REFERENCE_CASE, or source OpenFOAM "
            "to set FOAM_TUTORIALS."
        )


def is_time_directory(name: str) -> bool:
    try:
        float(name)
    except ValueError:
        return False
    return True


def read_obj_mesh(path: Path) -> ObjMesh:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                indices = [parse_obj_vertex_index(token, len(vertices)) for token in line.split()[1:]]
                if len(indices) < 3:
                    continue
                for i in range(1, len(indices) - 1):
                    faces.append((indices[0], indices[i], indices[i + 1]))

    if not vertices or not faces:
        raise ValueError(f"OBJ file did not contain vertices and triangular faces: {path}")
    return ObjMesh(vertices=vertices, faces=faces)


def parse_obj_vertex_index(token: str, vertex_count: int) -> int:
    raw = token.split("/", 1)[0]
    index = int(raw)
    if index < 0:
        return vertex_count + index + 1
    return index


def write_scaled_grouped_obj(
    path: Path,
    mesh: ObjMesh,
    *,
    input_scale: float,
    root_plane_tolerance_m: float,
) -> tuple[Bounds3D, int, int]:
    scaled_vertices = [
        (vertex[0] * input_scale, vertex[1] * input_scale, vertex[2] * input_scale)
        for vertex in mesh.vertices
    ]

    wing_faces: list[tuple[int, int, int]] = []
    root_faces: list[tuple[int, int, int]] = []
    for face in mesh.faces:
        if all(abs(scaled_vertices[index - 1][1]) <= root_plane_tolerance_m for index in face):
            root_faces.append(face)
        else:
            wing_faces.append(face)

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# motorBike-derived wing case geometry, scaled to metres\n")
        handle.write("o wingAssembly\n")
        for vertex in scaled_vertices:
            handle.write(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
        handle.write("g wing\n")
        for face in wing_faces:
            handle.write(f"f {face[0]} {face[1]} {face[2]}\n")
        handle.write("g wingRoot\n")
        for face in root_faces:
            handle.write(f"f {face[0]} {face[1]} {face[2]}\n")

    return bounds_from_vertices(scaled_vertices), len(wing_faces), len(root_faces)


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
    return Bounds3D(
        (
            wing_bounds.minimum[0] - config.refinement.front_m,
            0.0,
            ground_z_m,
        ),
        (
            wing_bounds.maximum[0] + config.refinement.back_m,
            wing_bounds.maximum[1] + config.refinement.side_m,
            wing_bounds.maximum[2] + config.refinement.top_m,
        ),
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
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    frontAndBack
    {{
        type patch;
        faces ((3 7 6 2));
    }}
    symmetry
    {{
        type symmetry;
        faces ((1 5 4 0));
    }}
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
    lowerWall
    {{
        type wall;
        faces ((0 3 2 1));
    }}
    upperWall
    {{
        type patch;
        faces ((4 5 6 7));
    }}
);
"""


def cell_counts(domain: Bounds3D, config: OpenFoamCaseConfig) -> tuple[int, int, int]:
    base = config.refinement.base_cell_size_m
    return (
        max(1, math.ceil(domain.size[0] / base)),
        max(1, math.ceil(domain.size[1] / base)),
        max(1, math.ceil(domain.size[2] / base)),
    )


def snappy_hex_mesh_dict(
    refinement_box: Bounds3D,
    domain: Bounds3D,
    wing_bounds: Bounds3D,
    ground_z_m: float,
    config: OpenFoamCaseConfig,
) -> str:
    del wing_bounds
    r = config.refinement
    bl = config.boundary_layer
    # Ground-hugging refinement band: tall enough to contain the ground layer
    # stack (~3x) so layers extrude on cells sized close to the stack.
    ground_stack_m = bl.ground_first_layer_height_m * (
        (bl.growth_rate ** bl.n_layers - 1) / (bl.growth_rate - 1)
    )
    ground_band_height_m = max(3.0 * ground_stack_m, 0.05)
    ground_band_min = refinement_box.minimum
    ground_band_max = (
        refinement_box.maximum[0],
        refinement_box.maximum[1],
        ground_z_m + ground_band_height_m,
    )
    location = (
        domain.minimum[0] + min(0.5, 0.15 * domain.size[0]),
        domain.maximum[1] - min(0.25, 0.1 * domain.size[1]),
        ground_z_m + min(0.1, 0.1 * domain.size[2]),
    )
    return foam_header("dictionary", "snappyHexMeshDict") + f"""
castellatedMesh true;
snap            true;
addLayers       true;

geometry
{{
    wing.obj
    {{
        type triSurfaceMesh;
        name wing;
        regions
        {{
            wing {{ name wing; }}
            wingRoot {{ name wingRoot; }}
        }}
    }}

    refinementBox
    {{
        type box;
        min {foam_vec(refinement_box.minimum)};
        max {foam_vec(refinement_box.maximum)};
    }}

    groundBand
    {{
        type box;
        min {foam_vec(ground_band_min)};
        max {foam_vec(ground_band_max)};
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
        {{
            file "wing.eMesh";
            level {r.feature_level};
        }}
    );

    refinementSurfaces
    {{
        wing
        {{
            level ({r.surface_min_level} {r.surface_max_level});
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
            }}
            patchInfo
            {{
                type wall;
                inGroups (wingGroup);
            }}
        }}
    }}

    resolveFeatureAngle 30;

    refinementRegions
    {{
        refinementBox
        {{
            mode inside;
            levels ((1E15 {r.refinement_box_level}));
        }}

        groundBand
        {{
            mode inside;
            levels ((1E15 {r.ground_refinement_level}));
        }}
    }}

    locationInMesh {foam_vec(location)};
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance    2.0;
    nSolveIter   30;
    nRelaxIter   5;

    nFeatureSnapIter     10;
    implicitFeatureSnap  false;
    explicitFeatureSnap  true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    // Layer-survival preset: keep level-6 surface refinement and 7 layers,
    // but use a less fragile stack so snappy keeps layers on the wing.
    relativeSizes true;
    expansionRatio {bl.growth_rate};
    finalLayerThickness {bl.final_layer_fraction};
    minThickness {bl.min_thickness_fraction};

    layers
    {{
        wing
        {{
            nSurfaceLayers {bl.n_layers};
        }}
        lowerWall
        {{
            nSurfaceLayers {bl.n_layers};
        }}
    }}

    nGrow 0;
    // Allow layers to continue over aerofoil curvature; layer quality limits
    // below are relaxed enough to avoid deleting most wing layers.
    featureAngle 170;
    slipFeatureAngle 60;
    nRelaxIter 10;
    nSmoothSurfaceNormals 3;
    nSmoothNormals 5;
    nSmoothThickness 15;
    maxFaceThicknessRatio 0.7;
    maxThicknessToMedialRatio 0.6;
    minMedialAxisAngle 80;
    nBufferCellsNoExtrude 0;
    nLayerIter 120;
}}

meshQualityControls
{{
    #include "meshQualityDict"

    nSmoothScale 4;
    errorReduction 0.75;
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
wing.obj
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


def control_dict(config: OpenFoamCaseConfig) -> str:
    solver = config.solver
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

writeFormat     binary;
writePrecision  6;
writeCompression off;

timeFormat      general;
timePrecision   6;

runTimeModifiable true;

functions
{{
    #include "forceCoeffs"

    yPlus
    {{
        type            yPlus;
        libs            (fieldFunctionObjects);
        patches         (wing lowerWall);
        writeFields     yes;
        writeControl    writeTime;
    }}

    wallShearStress
    {{
        type            wallShearStress;
        libs            (fieldFunctionObjects);
        patches         (wing lowerWall);
        writeControl    writeTime;
    }}
}}
"""


def force_coeffs_dict(
    wing_bounds: Bounds3D, l_ref: float, area_ref: float, config: OpenFoamCaseConfig
) -> str:
    del wing_bounds
    solver = config.solver
    cof_r = (0.25 * l_ref, 0.0, 0.0)
    return foam_header("", "") + f"""
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
"""


def topo_set_dict(refinement_box: Bounds3D) -> str:
    return foam_header("dictionary", "topoSetDict") + f"""
actions
(
    {{
        name    inner;
        type    cellZoneSet;
        action  new;
        source  boxToCell;
        min     {foam_vec(refinement_box.minimum)};
        max     {foam_vec(refinement_box.maximum)};
    }}
);
"""


def decompose_par_dict(config: OpenFoamCaseConfig) -> str:
    return foam_header("dictionary", "decomposeParDict") + f"""
numberOfSubdomains {config.solver.n_subdomains};

method          hierarchical;

coeffs
{{
    n           (4 4 2);
}}
"""


def transport_properties(config: OpenFoamCaseConfig) -> str:
    return foam_header("dictionary", "transportProperties") + f"""
transportModel  Newtonian;

nu              {config.solver.nu_m2_per_s:.9g};
"""


def initial_conditions(config: OpenFoamCaseConfig) -> str:
    solver = config.solver
    return foam_header("", "") + f"""
flowVelocity         ({solver.velocity_mps:.9g} 0 0);
pressure             0;
turbulentKE          {solver.turbulent_ke:.9g};
turbulentOmega       {solver.turbulent_omega:.9g};
"""


def write_zero_files(zero_dir: Path, config: OpenFoamCaseConfig) -> None:
    write_text(zero_dir / "U", field_u())
    write_text(zero_dir / "p", field_p())
    write_text(zero_dir / "k", field_k())
    write_text(zero_dir / "omega", field_omega())
    write_text(zero_dir / "nut", field_nut())


def field_u() -> str:
    return field_header("volVectorField", "U") + """
#include        "include/initialConditions"

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform $flowVelocity;

boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"

    inlet          { type fixedValue; value $internalField; }
    outlet         { type inletOutlet; inletValue uniform (0 0 0); value $internalField; }
    lowerWall      { type fixedValue; value $internalField; }
    upperWall      { type slip; }
    frontAndBack   { type slip; }
    symmetry       { type symmetry; value $internalField; }
    wingRoot       { type symmetry; value $internalField; }
    wing           { type noSlip; }
}
"""


def field_p() -> str:
    return field_header("volScalarField", "p") + """
#include        "include/initialConditions"

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform $pressure;

boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"

    inlet          { type zeroGradient; }
    outlet         { type fixedValue; value $internalField; }
    lowerWall      { type zeroGradient; }
    upperWall      { type zeroGradient; }
    frontAndBack   { type zeroGradient; }
    symmetry       { type symmetry; }
    wingRoot       { type symmetry; }
    wing           { type zeroGradient; }
}
"""


def field_k() -> str:
    return field_header("volScalarField", "k") + """
#include        "include/initialConditions"

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform $turbulentKE;

boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"

    inlet          { type fixedValue; value $internalField; }
    outlet         { type inletOutlet; inletValue $internalField; value $internalField; }
    lowerWall      { type kqRWallFunction; value $internalField; }
    upperWall      { type slip; }
    frontAndBack   { type slip; }
    symmetry       { type symmetry; }
    wingRoot       { type symmetry; }
    wing           { type kqRWallFunction; value $internalField; }
}
"""


def field_omega() -> str:
    return field_header("volScalarField", "omega") + """
#include        "include/initialConditions"

dimensions      [0 0 -1 0 0 0 0];

internalField   uniform $turbulentOmega;

boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"

    inlet          { type fixedValue; value $internalField; }
    outlet         { type inletOutlet; inletValue $internalField; value $internalField; }
    lowerWall      { type omegaWallFunction; value $internalField; }
    upperWall      { type slip; }
    frontAndBack   { type slip; }
    symmetry       { type symmetry; }
    wingRoot       { type symmetry; }
    wing           { type omegaWallFunction; value $internalField; }
}
"""


def field_nut() -> str:
    return field_header("volScalarField", "nut") + """
dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"

    frontAndBack   { type calculated; value uniform 0; }
    symmetry       { type symmetry; }
    inlet          { type calculated; value uniform 0; }
    outlet         { type calculated; value uniform 0; }
    lowerWall      { type nutUSpaldingWallFunction; value uniform 0; }
    upperWall      { type calculated; value uniform 0; }
    wingRoot       { type symmetry; }
    wing           { type nutUSpaldingWallFunction; value uniform 0; }
}
"""


def allrun_script() -> str:
    return """#!/usr/bin/env bash
cd "${0%/*}" || exit
. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions
set -e
#------------------------------------------------------------------------------

touch wing.foam

runApplication surfaceFeatureExtract

runApplication blockMesh

runApplication decomposePar

if foamDictionary -entry geometry -value system/snappyHexMeshDict | \\
   grep -q distributedTriSurfaceMesh
then
    echo "surfaceRedistributePar does not need to be run anymore"
    echo " - distributedTriSurfaceMesh will do on-the-fly redistribution"
fi

runParallel snappyHexMesh -overwrite

runParallel topoSet

restore0Dir -processor

runParallel patchSummary

runParallel potentialFoam -writephi

runParallel checkMesh -writeFields '(nonOrthoAngle)' -constant

runParallel $(getApplication)

runApplication reconstructParMesh -constant

runApplication reconstructPar -latestTime

#------------------------------------------------------------------------------
"""


def allclean_script() -> str:
    return """#!/usr/bin/env bash
cd "${0%/*}" || exit
. ${WM_PROJECT_DIR:?}/bin/tools/CleanFunctions
set -e
#------------------------------------------------------------------------------

cleanCase0

rm -rf processor* postProcessing dynamicCode VTK
rm -rf constant/polyMesh constant/extendedFeatureEdgeMesh
rm -f log.* wing.foam
rm -f constant/triSurface/wing.eMesh
rm -f constant/triSurface/wing_*.obj
rm -f constant/triSurface/wing*.eMesh

#------------------------------------------------------------------------------
"""


def write_case_info(path: Path, case_info: MotorBikeCaseInfo, config: OpenFoamCaseConfig) -> None:
    payload = {
        "design_id": case_info.design_id,
        "ground_plane_z_m": case_info.ground_clearance_m,
        "nominal_height_above_ground_m": abs(case_info.ground_clearance_m),
        "case_dir": str(case_info.case_dir),
        "source_obj": str(case_info.source_obj),
        "wing_obj": str(case_info.wing_obj),
        "wing_bounds_m": asdict(case_info.wing_bounds_m),
        "domain_bounds_m": asdict(case_info.domain_bounds_m),
        "refinement_box_m": asdict(case_info.refinement_box_m),
        "l_ref_m": case_info.l_ref_m,
        "area_ref_m2": case_info.area_ref_m2,
        "reynolds_number": case_info.reynolds_number,
        "wing_faces": case_info.wing_faces,
        "root_faces": case_info.root_faces,
        "refined_cell_size_m": config.refinement.base_cell_size_m
        / (2**config.refinement.refinement_box_level),
        "solver": asdict(config.solver),
        "domain": asdict(config.domain),
        "refinement": {
            **asdict(config.refinement),
            "actual_refined_to_base_ratio": config.refinement.actual_refined_to_base_ratio,
        },
        "boundary_layer": asdict(config.boundary_layer),
        "reference_case_dir": str(config.reference_case_dir),
    }
    write_text(path, json.dumps(payload, indent=2))


def write_case_manifest(
    output_dir: Path, cases: list[MotorBikeCaseInfo], config: OpenFoamCaseConfig
) -> None:
    rows = [
        {
            "design_id": case.design_id,
            "ground_plane_z_m": case.ground_clearance_m,
            "nominal_height_above_ground_m": abs(case.ground_clearance_m),
            "case_dir": case.case_dir.relative_to(output_dir).as_posix(),
            "l_ref_m": case.l_ref_m,
            "area_ref_m2": case.area_ref_m2,
            "reynolds_number": case.reynolds_number,
            "wing_faces": case.wing_faces,
            "root_faces": case.root_faces,
        }
        for case in cases
    ]
    write_text(
        output_dir / "case_manifest.json",
        json.dumps(
            {
                "num_cases": len(cases),
                "population_dir": str(config.population_dir.resolve()),
                "reference_case_dir": str(config.reference_case_dir),
                "ground_plane_z_values_m": list(config.ground_clearances_m),
                "refined_cell_size_m": config.refinement.base_cell_size_m
                / (2**config.refinement.refinement_box_level),
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
        fieldnames = list(rows[0].keys()) if rows else ["design_id", "ground_plane_z_m", "case_dir"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_text(output_dir / "run_all_cases.sh", run_all_cases_script(cases, output_dir))


def write_root_scripts(output_dir: Path, config: OpenFoamCaseConfig) -> None:
    shutil.copy2(Path(__file__).with_name("openfoam_results.py"), output_dir / "collect_results.py")
    write_text(
        output_dir / "run_all_cases_tracked.sh",
        tracked_runner_script(config.solver.end_time),
    )
    write_text(output_dir / "run_remote_batch.sh", remote_launcher_script("run_all_cases_tracked.sh"))
    write_text(output_dir / "README_RUN.md", readme_text(config))


def run_all_cases_script(cases: list[MotorBikeCaseInfo], output_dir: Path) -> str:
    lines = [
        "#!/usr/bin/env bash",
        'cd "${0%/*}" || exit',
        "set -euo pipefail",
        "",
    ]
    for case in cases:
        rel = case.case_dir.relative_to(output_dir).as_posix()
        lines.append(f'(cd "{rel}" && bash ./Allrun)')
    lines.append("")
    lines.extend(result_collector_lines())
    return "\n".join(lines) + "\n"


def tracked_runner_script(end_time: int) -> str:
    return ("""#!/usr/bin/env bash
set -uo pipefail

cd "${0%/*}" || exit 1

status_file="batch_status.csv"
if [[ ! -f "$status_file" ]]; then
    printf "case,status,start,end,exit_code\\n" > "$status_file"
fi

case_paths=$(sed -n 's/^(cd "\\([^"]*\\)".*/\\1/p' run_all_cases.sh)

for case_dir in $case_paths; do
    start=$(date -Is)
    if [[ -f "$case_dir/log.simpleFoam" ]] && grep -q "Time = __END_TIME__" "$case_dir/log.simpleFoam"; then
        end=$(date -Is)
        printf "%s,skipped_complete,%s,%s,0\\n" "$case_dir" "$start" "$end" >> "$status_file"
        continue
    fi

    echo "=== Running $case_dir ==="
    (
        cd "$case_dir" || exit 1
        bash ./Allrun
    )
    exit_code=$?
    end=$(date -Is)

    if [[ $exit_code -eq 0 ]]; then
        printf "%s,ok,%s,%s,0\\n" "$case_dir" "$start" "$end" >> "$status_file"
    else
        printf "%s,failed,%s,%s,%s\\n" "$case_dir" "$start" "$end" "$exit_code" >> "$status_file"
    fi
done

run_result_collector() {
    if [[ ! -f collect_results.py ]]; then
        echo "WARNING: collect_results.py not found; skipping CSV summary."
        return 0
    fi

    if command -v python3 >/dev/null 2>&1; then
        python3 collect_results.py
    elif command -v python >/dev/null 2>&1; then
        python collect_results.py
    else
        echo "WARNING: Python not found; skipping CSV summary."
    fi
}

run_result_collector
""").replace("__END_TIME__", str(end_time))


def result_collector_lines() -> list[str]:
    return [
        'if [[ -f collect_results.py ]]; then',
        '    if command -v python3 >/dev/null 2>&1; then',
        '        python3 collect_results.py',
        '    elif command -v python >/dev/null 2>&1; then',
        '        python collect_results.py',
        '    else',
        '        echo "WARNING: Python not found; skipping CSV summary."',
        '    fi',
        'else',
        '    echo "WARNING: collect_results.py not found; skipping CSV summary."',
        'fi',
    ]


def remote_launcher_script(script_name: str) -> str:
    return f"""#!/usr/bin/env bash
set -eo pipefail

cd "${{0%/*}}" || exit 1

if [[ -z "${{WM_PROJECT_DIR:-}}" ]]; then
    for openfoam_rc in \\
        /usr/lib/openfoam/openfoam2412/etc/bashrc \\
        /usr/lib/openfoam/openfoam*/etc/bashrc \\
        /opt/openfoam*/etc/bashrc
    do
        if [[ -f "$openfoam_rc" ]]; then
            # shellcheck source=/dev/null
            . "$openfoam_rc" || true
            break
        fi
    done
fi

if [[ -z "${{WM_PROJECT_DIR:-}}" && -d /usr/lib/openfoam/openfoam2412 ]]; then
    export WM_PROJECT_DIR=/usr/lib/openfoam/openfoam2412
    export WM_PROJECT=OpenFOAM
    export WM_PROJECT_VERSION=2412
    export FOAM_APPBIN="${{WM_PROJECT_DIR}}/platforms/linux64GccDPInt32Opt/bin"
    export FOAM_LIBBIN="${{WM_PROJECT_DIR}}/platforms/linux64GccDPInt32Opt/lib"
    export FOAM_ETC="${{WM_PROJECT_DIR}}/etc"
    export PATH="${{FOAM_APPBIN}}:${{WM_PROJECT_DIR}}/bin:${{PATH}}"
    export LD_LIBRARY_PATH="${{FOAM_LIBBIN}}:${{FOAM_LIBBIN}}/sys-openmpi:${{FOAM_LIBBIN}}/dummy:${{LD_LIBRARY_PATH:-}}"
fi

if [[ -z "${{WM_PROJECT_DIR:-}}" ]]; then
    echo "OpenFOAM environment is not loaded. Source your OpenFOAM bashrc first, then rerun this script." >&2
    exit 1
fi

bash ./{script_name}
"""


def readme_text(config: OpenFoamCaseConfig) -> str:
    r = config.refinement
    bl = config.boundary_layer
    solver = config.solver
    return f"""# Generated Wing Cases

Generated from the OpenFOAM v2412 simpleFoam/motorBike tutorial.
See case_manifest.csv for the designs and ground heights in this batch.

Settings used:
- Parallel ranks per case: {solver.n_subdomains}; cases run sequentially.
- Flow and moving-ground speed: {solver.velocity_mps:g} m/s.
- End iteration: {solver.end_time}.
- Wing surface refinement: ({r.surface_min_level} {r.surface_max_level}).
- Refinement box level: {r.refinement_box_level}; ground band level: {r.ground_refinement_level}.
- Castellated mesh cell cap: {r.max_global_cells}; layers can add further cells.
- Requested layers on wing and lowerWall: {bl.n_layers}.
- relativeSizes: true; expansionRatio: {bl.growth_rate:g}.
- finalLayerThickness: {bl.final_layer_fraction:g}; minThickness: {bl.min_thickness_fraction:g}.
- Check layer coverage and measured y+ after meshing and solving.

In a sourced OpenFOAM shell, run one case from its folder with
`bash ./Allrun`. From this run folder, use `bash ./run_all_cases_tracked.sh`
for the full batch. It logs batch_status.csv, continues after failed cases,
and calls collect_results.py at the end. A Python 3 interpreter is required.

run_all_cases.sh stops on the first failing case.
run_remote_batch.sh attempts to load OpenFOAM before calling the tracked runner.
To collect results again: `python3 collect_results.py --no-iterations`.

Outputs: all_results_summary.csv and force_coefficients_iterations.csv.
The collector's status column reports available output, not numerical convergence.
"""


def foam_header(foam_class: str, object_name: str) -> str:
    class_line = f"    class       {foam_class};\n" if foam_class else ""
    object_line = f"    object      {object_name};\n" if object_name else ""
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
{class_line}{object_line}}}
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8", newline="\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate motorBike-derived wing OpenFOAM cases.")
    parser.add_argument("--population-dir", type=Path, default=DEFAULT_CFD_CONFIG.population_dir)
    parser.add_argument("--output-dir", type=Path, help="Defaults to <population-dir>/run.")
    parser.add_argument("--reference-case", type=Path, help="Path to the OpenFOAM v2412 motorBike tutorial.")
    parser.add_argument("--cores", type=int, default=DEFAULT_CFD_CONFIG.solver.n_subdomains)
    parser.add_argument(
        "--ground-clearances",
        default=",".join(str(value) for value in DEFAULT_CFD_CONFIG.ground_clearances_m),
        help="Comma-separated ground-plane z values in metres.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = OpenFoamCaseConfig(
        population_dir=args.population_dir,
        output_dir=args.output_dir or args.population_dir / "run",
        reference_case_dir=args.reference_case or OpenFoamCaseConfig().reference_case_dir,
        solver=replace(DEFAULT_CFD_CONFIG.solver, n_subdomains=args.cores),
        ground_clearances_m=tuple(float(value) for value in args.ground_clearances.split(",")),
    )
    cases = generate_motorbike_cases(config)
    print(f"Generated {len(cases)} motorBike-derived OpenFOAM cases in {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
