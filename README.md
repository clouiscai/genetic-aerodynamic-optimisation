# Genetic Ground-Effect Wing Optimisation

A research project exploring **biologically inspired design optimisation**:
encode a design as a genome, evaluate it in a simulated environment, select
parents, and produce new designs through crossover and mutation. The case study
is a one-sided wing with a rounded winglet operating near the ground.

The genesis population contains **10 designs**, and each subsequent generation
contains **10 children**. Five ground heights per design give **50 CFD cases per
generation**. Sections, genes, designs and simulation cases are different counts.

The main contribution is the evolutionary workflow and its traceable design
history. An LLM can help adapt the code to another design problem; the optimisation
loop itself runs in Python and does not require an LLM or API key.

## Methodology

1. **Encode:** represent airfoil choice, incidence, section sweep, dihedral,
   twist, taper and winglet shape as bounded genes.
2. **Develop:** decode each genome into section geometry, apply smoothness
   constraints, reject invalid or duplicate designs, and export meshes.
3. **Evaluate:** simulate each design across five nominal ground heights with
   OpenFOAM and collect aerodynamic coefficients.
4. **Select:** filter incomplete or implausible results, then rank eligible
   designs using a weighted sum of metrics normalised within that generation.
5. **Reproduce:** select two best designs and two runners-up, shuffle them into
   two disjoint pairs, and produce five children per pair.
6. **Vary and repeat:** mix parent genes, mutate them, repair constrained arrays,
   validate the children, and evaluate the next generation.

With the default configuration, the four selected parents contribute genes;
they are not copied unchanged into the next population. Mutation has a 0.10
probability per scalar gene or array element, with Gaussian scale 0.20 of the
gene's allowed range. Crossover granularity is configurable by gene group.

This is an evolutionary search, not proof of a global optimum. Generation-local
normalisation means fitness scores are useful for ranking that generation;
compare raw metrics under the same evaluation conditions across generations.

## Requirements

- Python 3.10 or newer. The core Python workflow uses the standard library.
- OpenFOAM **v2412**, Bash and MPI on the simulation machine.
- A local copy of the v2412 `incompressible/simpleFoam/motorBike` tutorial.
  Its solver dictionaries and mesh quality settings form the case template.
- Optional: ForgeCAD CLI for CAD validation, STEP export and previews, plus
  Chrome/Chromium for previews. STL/OBJ generation can run with Python alone.

Generated populations, simulation fields, images and local reference cases are
excluded from Git. A fresh clone contains code and the published CSV, not the
original per-design `parameters.json` files or CFD run directories.

## Try the Seed Generator

Run commands from the repository root. First try a single design without CAD
execution, mesh export or simulation:

```sh
python seed_generator.py --num-designs 1 --output-dir genesis-batch-test --skip-cad-validation --skip-export --skip-preview
```

This writes the genome and decoded sections in `parameters.json`, a
self-contained `geometry.forge.js`, a manifest and a parameter report.
The default seed population can then be generated with STL/OBJ meshes:

```sh
python seed_generator.py --num-designs 10 --skip-cad-validation --skip-step --skip-preview
```

Output is `genesis-batch/design_000` through `design_009`. Each mesh is
stitched from corresponding airfoil points along smoothed sections.
A bare `python seed_generator.py` also requests ForgeCAD validation, STEP
export and previews. STEP export depends on the CAD backend and can fail on
complex lofts; check export logs and `*.todo.txt` markers before using outputs.

Geometry defaults are a 1.5 m main half-span, 1.2 m root chord, 25 main-wing
sections and 10 additional winglet stations. Profiles use 200 points.
Each wing uses a single airfoil by default; the genome also supports a two-profile
transition. The main-wing chord floor is 0.4 times root chord. See
`wing_generator/config.py` for bounds and validation settings.

## Prepare and Run CFD

The active case generator is `wing_generator.motorbike_case_generator`, which
reads the generated OBJ meshes. Point it at your own tutorial directory:

```sh
python -m wing_generator.motorbike_case_generator --population-dir genesis-batch --reference-case "/path/to/tutorials/incompressible/simpleFoam/motorBike" --cores 32
```

The same command accepts a quoted Windows path when preparing cases locally.
The output defaults to `<population-dir>/run`; use `--output-dir` to override.
The template can also be set with `OPENFOAM_REFERENCE_CASE`, discovered beneath
`FOAM_TUTORIALS` in a sourced OpenFOAM shell, or placed in `ref/motorBike`.
An explicit `--reference-case` takes precedence.

Ground planes default to z = -0.2, -0.4, -0.6, -0.8 and -1.0 m.
For another sweep, pass e.g. `--ground-clearances=-0.2,-0.4,-0.6`.
These are nominal distances from z = 0, not guaranteed clearance beneath the
lowest winglet point. Geometry is in millimetres for CAD and metres for OpenFOAM;
the flow, span and vertical axes are X, Y and Z.

Current CFD defaults include 25 m/s flow, a moving ground, root symmetry,
simpleFoam with k-omega SST, 300 iterations and 32 parallel ranks per case.
Surface/feature refinement is level 6, with 7 requested layers on the wing and
ground, `relativeSizes true`, growth 1.12, final thickness fraction 0.2 and
minimum thickness fraction 0.001. Layer coverage and y+ must be measured for
each mesh; these settings do not guarantee a particular y+.

Transfer the generated `run` folder to the simulation machine. In a sourced
OpenFOAM shell:

```sh
cd genesis-batch/run
bash ./run_all_cases_tracked.sh
```

Cases run one after another; each case uses the configured parallel ranks.
The tracked runner records `batch_status.csv`, skips cases whose solver log
contains the configured final iteration, continues after failures, and calls the
bundled `collect_results.py` at the end. Its skip check is not a convergence test.

For one case, enter its folder and run `bash ./Allrun`.
`run_all_cases.sh` stops at the first failure; `run_remote_batch.sh` attempts
to source OpenFOAM before invoking the tracked runner. Generated
`README_RUN.md` records the settings actually used.

The collector writes `all_results_summary.csv` and
`force_coefficients_iterations.csv`, including final coefficients, averages of
the last 50 recorded coefficient rows, and available y+ statistics.
To collect again without rerunning CFD:

```sh
python -m wing_generator.openfoam_results --cases-dir genesis-batch/run --no-iterations
```

On a remote machine with only the run folder, use
`python3 collect_results.py --no-iterations` inside that folder.

## Breed the Next Generation

Bring the summary CSV back to the source generation's `run` folder.
Keep the matching design folders and `parameters.json` files locally.

```sh
python evolution.py --generation 0 --select-only
python evolution.py --generation 0 --reference-case "/path/to/tutorials/incompressible/simpleFoam/motorBike" --cores 32
```

`--generation` identifies the **source**: 0 is `genesis-batch`, 1 is
`generation_01`, and so on. The second command writes `generation_01`
with ten children, STL/OBJ meshes, selection reports, a breeding log and CFD
cases. Repeat simulation and collection before advancing again.

Selection defaults require at least five valid case rows per design, no invalid
rows in the supplied sweep, final iteration >= 300, and finite selected
coefficients with magnitude below 10. Averages are preferred when available.
This is an eligibility filter, not evidence of mesh independence or numerical
convergence. Check residuals, coefficient histories, mesh quality and y+ separately.

Use `--config path/to/config.json` for a different evolutionary configuration
and `--seed` to override the breeding seed. To breed without preparing cases,
set `output.generate_openfoam_cases` to false. Use a new output generation for
a new experiment; `--force` replaces an existing output generation.

## Configuration and Source

| Concern | File |
| --- | --- |
| Seed count, geometry bounds, airfoils, smoothing and mesh sampling | `wing_generator/config.py` |
| Fitness weights, selection, crossover, mutation and generation naming | `config.json` |
| Domain, ground sweep, refinement, layers, solver and core count | `wing_generator/cfd_config.py` |
| Genome sampling, decoding and identity | `wing_generator/genome.py` |
| Geometry validation and population manifests | `wing_generator/population_generator.py` |
| ForgeCAD source and deterministic STL/OBJ export | `wing_generator/geometry_generator.py`, `stl_mesh.py`, `exporter.py` |
| Evaluation setup and per-case result collection | `wing_generator/motorbike_case_generator.py`, `openfoam_results.py` |
| Eligibility, scoring, selection and breeding | `wing_generator/evolution.py` |
| Join aerodynamic results with geometry | `wing_generator/collect_data.py` |

Seed population size and offspring count are separate settings. When changing
the experiment size, keep the number of selected parents even and ensure
`(num_best + num_runner_ups) / 2 * children_per_pair == designs_per_generation`.
Update `min_valid_cases` when changing the ground-height sweep. Use the same
geometry and CFD settings for comparable evaluations within an experiment.

## Published Data

`combined_data.csv` currently contains **536 qualifying case rows** across
genesis and generations 01-10, with 51 columns. One row is one design at one
ground height, not a whole generation. It includes `C_l`, `C_d`, `C_m`,
L/D, span, chord, airfoil and winglet information, and source references.

The span includes only the modelled half-wing. Reference planform area is the
main-wing section integral used for force coefficients, excluding the winglet;
it is not a full two-sided wing area.

To build a new joined dataset from local summaries and matching design metadata:

```sh
python -m wing_generator.collect_data --root . --output combined_data_new.csv
```

The default output name is `combined_data.csv`. The collector accepts sane
coefficients from cases reaching 300 iterations or ending normally with an
`End` marker in a locally available solver log. It writes an exclusion audit
to `combined_excluded_aero_cases.csv`. Early normal termination therefore needs
the log as well as the remote summary. Use `--require-full-sweep` to require a
complete design sweep. This dataset filter is distinct from the stricter
iteration requirement used for breeding.

## Adapt the Methodology with Your LLM

Read [the adaptation guide](docs/adapting-the-methodology.md) for a prompt you
can give your own LLM, the interfaces to replace, and a small validation workflow.
Keep the genome -> evaluation -> selection -> reproduction loop and substitute
your own geometry, simulator, operating conditions and objectives.

A reusable [adapt-evolution-workflow skill](.agents/skills/adapt-evolution-workflow/SKILL.md)
is included. Point a coding agent at that file, or attach it alongside the
adaptation guide in a chat-based LLM. It explains the implemented method and
asks the LLM to build around your actual workflow.

## Development Checks

```sh
python -m unittest discover -s tests -v
```

The tests use temporary designs, a small template fixture and synthetic
coefficients. They exercise generation, portable case preparation, collection,
filtering and breeding without launching CAD or CFD.
