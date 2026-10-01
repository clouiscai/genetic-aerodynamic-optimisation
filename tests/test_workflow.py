from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from wing_generator import evolution
from wing_generator.cfd_config import OpenFoamCaseConfig, default_reference_case_dir
from wing_generator.collect_data import case_directory, qualification
from wing_generator.config import DEFAULT_CONFIG
from wing_generator.exporter import ExportOptions, export_design
from wing_generator.motorbike_case_generator import generate_motorbike_cases
from wing_generator.openfoam_results import read_manifest
from wing_generator.population_generator import PopulationGenerator, validate_design


REPO = Path(__file__).resolve().parents[1]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_python(*args: str, cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, *args], cwd=cwd, env=env,
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    return result


class WorkflowTests(unittest.TestCase):
    def test_default_seed_population_and_breeding_without_cad(self):
        with tempfile.TemporaryDirectory(prefix="wing workflow ") as tmp:
            root = Path(tmp)
            genesis = root / "genesis-batch"
            # An empty PATH proves the offline path does not require ForgeCAD.
            env = {**os.environ, "PATH": "", "PYTHONPATH": ""}
            run_python(
                str(REPO / "seed_generator.py"), "--output-dir", str(genesis),
                "--skip-cad-validation", "--skip-export", "--skip-preview",
                cwd=root, env=env,
            )
            manifest = json.loads((genesis / "population_manifest.json").read_text())
            self.assertEqual(manifest["num_designs"], 10)
            self.assertEqual(manifest["design_ids"], [f"design_{i:03d}" for i in range(10)])
            self.assertFalse((genesis / "interactive_genome.forge.js").exists())
            self.assertFalse((genesis / "manual_genome.forge.js").exists())
            designs = evolution.load_designs(genesis)
            for design in designs.values():
                self.assertTrue(validate_design(design, DEFAULT_CONFIG).ok)

            generator = PopulationGenerator(DEFAULT_CONFIG)
            repeated = generator.next_valid_design(0)
            self.assertEqual(designs["design_000"]["genome"], json.loads(json.dumps(repeated["genome"])))
            rows = []
            for index, design_id in enumerate(designs):
                for height in (0.2, 0.4, 0.6, 0.8, 1.0):
                    rows.append({
                        "design_id": design_id, "final_time": "300",
                        "nominal_height_above_ground_m": str(height),
                        "Cl": str(0.8 + index * 0.02 + 0.1 / height),
                        "Cd": str(0.08 + index * 0.001), "CmPitch": "-0.03",
                    })
            write_csv(genesis / "run" / "all_results_summary.csv", rows)
            cfg = evolution.load_evolution_config(REPO / "config.json")
            cfg["output"].update(
                export_stl=False, render_previews=False, generate_openfoam_cases=False,
            )
            with patch.object(evolution, "PROJECT_ROOT", root), contextlib.redirect_stdout(io.StringIO()):
                evolution.breed_generation(0, cfg, select_only=True)
                self.assertFalse((root / "generation_01").exists())
                evolution.breed_generation(0, cfg, seed=123)

            children = evolution.load_designs(root / "generation_01")
            self.assertEqual(set(children), {f"design_{i:03d}" for i in range(10, 20)})
            self.assertTrue(all(validate_design(d, DEFAULT_CONFIG).ok for d in children.values()))
            breeding = json.loads((root / "generation_01" / "breeding_log.json").read_text())
            self.assertEqual(len(breeding["parent_pairs"]), 2)
            self.assertEqual(len({p for pair in breeding["parent_pairs"] for p in pair}), 4)
            self.assertEqual(len(breeding["children"]), 10)
            child_manifest = json.loads((root / "generation_01" / "population_manifest.json").read_text())
            self.assertEqual(child_manifest["random_seed"], breeding["random_seed"])
            fingerprints = {
                evolution.genome_fingerprint(evolution.genome_from_design(d))
                for d in children.values()
            }
            self.assertEqual(len(fingerprints), 10)

    def test_mesh_only_export_without_forgecad(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            design = PopulationGenerator(DEFAULT_CONFIG).next_valid_design(0)
            (folder / "parameters.json").write_text(json.dumps(design))
            config = replace(
                DEFAULT_CONFIG,
                forgecad=replace(DEFAULT_CONFIG.forgecad, stitched_stl_subdivisions_per_interval=1),
            )
            with patch("wing_generator.exporter._forgecad_command", side_effect=AssertionError("CAD invoked")):
                export_design(folder, config, ExportOptions(
                    validate_cad=False, export_stl=True, export_step=False, render_preview=False,
                ))
            self.assertGreater((folder / "geometry.stl").stat().st_size, 84)
            self.assertGreater((folder / "geometry.obj").stat().st_size, 0)
            self.assertFalse((folder / "geometry.stl.todo.txt").exists())

    def test_portable_case_batch_and_standalone_collector(self):
        with tempfile.TemporaryDirectory(prefix="wing transfer ") as tmp:
            root = Path(tmp)
            population = root / "genesis-batch"
            design_dir = population / "design_000"
            design_dir.mkdir(parents=True)
            design = PopulationGenerator(DEFAULT_CONFIG).next_valid_design(0)
            (design_dir / "parameters.json").write_text(json.dumps(design))
            # Tiny closed mesh and tutorial fixture exercise packaging, not CFD.
            (design_dir / "geometry.obj").write_text(
                "v 0 0 0\nv 1000 0 0\nv 0 1000 0\nv 0 0 100\n"
                "f 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n"
            )
            reference = root / "tutorial fixture"
            for name in ("0.orig", "constant", "system"):
                (reference / name).mkdir(parents=True)
            for name in ("fvSchemes", "fvSolution", "meshQualityDict"):
                (reference / "system" / name).write_text("// test fixture only\n")
            run_dir = population / "run"
            cfg = OpenFoamCaseConfig(
                population_dir=population, output_dir=run_dir, reference_case_dir=reference,
            )
            cfg = replace(cfg, solver=replace(cfg.solver, n_subdomains=4))
            cases = generate_motorbike_cases(cfg)
            self.assertEqual(len(cases), 5)
            self.assertEqual(read_manifest(run_dir / "case_manifest.csv").keys(), {
                ("design_000", c.case_dir.name) for c in cases
            })
            for case in cases:
                self.assertIn("numberOfSubdomains 4;", (case.case_dir / "system" / "decomposeParDict").read_text())
                coeff = case.case_dir / "postProcessing" / "wingForceCoeffs" / "0" / "coefficient.dat"
                coeff.parent.mkdir(parents=True)
                coeff.write_text("# Time Cd Cl CmPitch\n299 0.08 1.0 -0.03\n300 0.08 1.0 -0.03\n")
                (case.case_dir / "log.simpleFoam").write_text("Time = 300\nEnd\n")
                yplus = case.case_dir / "postProcessing" / "yPlus" / "0" / "yPlus.dat"
                yplus.parent.mkdir(parents=True)
                yplus.write_text("# Time patch min max average\n300 wing 0.5 20 5\n300 lowerWall 1 50 10\n")
            run_python("collect_results.py", cwd=run_dir, env={**os.environ, "PYTHONPATH": ""})
            with (run_dir / "all_results_summary.csv").open(newline="") as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(len(summary), 5)
            self.assertEqual(summary[0]["Cl_avg_last_window"], "1")
            self.assertEqual(summary[0]["wing_yplus_average"], "5")
            self.assertFalse(Path(summary[0]["case_dir"]).is_absolute())
            for script in ("run_all_cases.sh", "run_all_cases_tracked.sh"):
                self.assertIn("collect_results.py", (run_dir / script).read_text())
            self.assertTrue((run_dir / "run_remote_batch.sh").is_file())
            self.assertIn("Parallel ranks per case: 4", (run_dir / "README_RUN.md").read_text())

            run_python(
                "-m", "wing_generator.collect_data", "--root", str(root),
                "--output", "joined.csv", cwd=REPO,
            )
            with (root / "joined.csv").open(newline="") as handle:
                joined = list(csv.DictReader(handle))
            self.assertEqual(len(joined), 5)
            self.assertEqual(float(joined[0]["C_l"]), 1.0)
            self.assertEqual(float(joined[0]["main_wing_span_m"]), 1.5)
            self.assertGreater(float(joined[0]["reference_planform_area_m2"]), 0)

    def test_reference_discovery_is_portable(self):
        with patch.dict(os.environ, {"OPENFOAM_REFERENCE_CASE": "my template", "FOAM_TUTORIALS": "tutorials"}, clear=True):
            self.assertEqual(default_reference_case_dir(), Path("my template"))
        with patch.dict(os.environ, {"FOAM_TUTORIALS": "tutorials"}, clear=True):
            self.assertEqual(default_reference_case_dir(), Path("tutorials/incompressible/simpleFoam/motorBike"))
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(default_reference_case_dir(), Path("ref/motorBike"))

    def test_missing_reference_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = OpenFoamCaseConfig(
                population_dir=root / "population", output_dir=root / "run",
                reference_case_dir=root / "missing",
            )
            with self.assertRaisesRegex(FileNotFoundError, "--reference-case"):
                generate_motorbike_cases(cfg)
            self.assertFalse(cfg.output_dir.exists())

    def test_eligibility_distinguishes_completion_from_sane_coefficients(self):
        validity = evolution.load_evolution_config(REPO / "config.json")["results_validity"]
        row = {"final_time": "300", "Cl": "1", "Cd": "0.1", "CmPitch": "-0.02"}
        self.assertTrue(evolution._row_is_valid(row, validity))
        for invalid in (
            {"final_time": "299"}, {"final_time": "inf"},
            {"Cl": "nan"}, {"Cd": "1e20"}, {"Cl_avg_last_window": "inf"},
        ):
            self.assertFalse(evolution._row_is_valid({**row, **invalid}, validity))
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.simpleFoam"
            log.write_text("Time = 120\nEnd\n")
            early = {**row, "final_time": "120"}
            reason, completion = qualification(early, log_path=log, required_time=300, max_abs_coefficient=10)
            self.assertEqual(reason, "")
            self.assertEqual(completion, "normal_solver_end")
            log.unlink()
            self.assertTrue(qualification(early, log_path=log, required_time=300, max_abs_coefficient=10)[0])
            bad_average = {**row, "Cl_avg_last_window": "nan", "Cd_avg_last_window": "0.1", "CmPitch_avg_last_window": "0"}
            self.assertEqual(
                qualification(bad_average, log_path=log, required_time=300, max_abs_coefficient=10)[0],
                "invalid_average_coefficient",
            )

    def test_imported_summaries_resolve_local_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = root / "design_000" / "ground_m0p40"
            local.mkdir(parents=True)
            row = {"case_dir": "/remote/experiment/design_000/ground_m0p40"}
            self.assertEqual(case_directory(root, row, "design_000", "ground_m0p40"), local)
            manifest = root / "manifest.csv"
            write_csv(manifest, [{
                "design_id": "design_000",
                "case_dir": "C:\\old\\run\\design_000\\ground_m0p40",
            }])
            self.assertIn(("design_000", "ground_m0p40"), read_manifest(manifest))


if __name__ == "__main__":
    unittest.main()
