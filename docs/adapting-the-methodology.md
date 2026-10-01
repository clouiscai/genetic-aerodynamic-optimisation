# Adapt the Evolutionary Methodology

This repository demonstrates how to turn a design problem into a repeatable
evolutionary experiment. The wing is the example; the reusable method is bounded
variation, evaluation under consistent conditions, parent selection and
reproduction with an auditable history.

An LLM helps implement the experiment and its tool integrations. The simulator
or measured data supplies performance evidence. The current optimiser has no
LLM dependency.

## Start with Your Own Experiment

Give your LLM a compact brief:

| Question | Example from this repository |
| --- | --- |
| What is one individual? | One half-wing and rounded winglet |
| What can vary? | Airfoil, incidence, section slopes, twist, taper, winglet parameters |
| What stays fixed? | Main half-span, root chord, section count within an experiment |
| What makes it invalid? | Out-of-bounds genes, invalid loft geometry, missing evaluation |
| How is it evaluated? | Five ground heights with identical CFD settings |
| What is rewarded or penalised? | Lift, drag, L/D, pitching moment, ground-effect response, span efficiency |
| What is the budget? | Ten designs and fifty cases per generation |
| How is it reproduced? | Four parents, two disjoint pairs, five children per pair |
| When does the experiment stop? | Define a generation budget or a raw-metric improvement criterion |

Supply tool versions, file formats, coordinate axes, units, execution commands,
an example valid input and result, and practical compute limits. If these are
unknown, ask the LLM to identify the missing contract before integrating the tool.

## Prompt for a Coding Agent

Replace the bracketed fields and give this prompt to your own LLM with access
to the repository. A chat-only LLM can use uploaded source files and example data.

```text
Read README.md, docs/adapting-the-methodology.md and
.agents/skills/adapt-evolution-workflow/SKILL.md.
Inspect the implementation before proposing changes.

Adapt this evolutionary design method to:
- Design problem and existing workflow: [describe it]
- Geometry/data representation and tools: [tools, versions, file formats]
- Adjustable variables, fixed dimensions, bounds and units: [list]
- Evaluation tool, operating conditions and example output: [provide]
- Objectives and feasibility constraints: [define]
- Population, compute budget, reproducibility seed and stopping rule: [define]

Map my workflow to genome, decoder, geometry validation, evaluation,
result eligibility, fitness, selection, crossover and mutation.
Reuse the relevant logic, and identify the wing-specific parts to replace.
Implement a minimal local example in my chosen workflow.
Verify one candidate through the evaluation/result interface, then one small
generation with deterministic test results. Label synthetic data explicitly.
Document commands, dependencies, assumptions, result units, and where I tune
the experiment. Keep real simulation runs within the compute budget I specify.
```

## Implementation Map

| Boundary | Current implementation | What your adaptation must define |
| --- | --- | --- |
| Genome | WingGenome in wing_generator/genome.py | Serialisable genes, bounds, discrete choices and identity |
| Decoder | decode_genome and winglet_stations | Deterministic genome-to-model mapping with units and constraints |
| Feasibility | validate_design in population_generator.py | Reject, repair or penalise invalid candidates, with reasons |
| Model export | geometry_generator.py and stl_mesh.py | Files accepted by your modeller, simulator or measurement tool |
| Evaluation | motorbike_case_generator.py | Operating conditions, execution, expected outputs and failure handling |
| Results | openfoam_results.py and collect_data.py | Stable IDs, condition IDs, completion evidence, metrics and provenance |
| Fitness | design_metrics and score_generation in evolution.py | Metric direction, scaling, aggregation and treatment of invalid values |
| Reproduction | select_survivors, crossover, mutate and _breed_one_child | Parent strategy, gene dependencies, mutation and duplicate handling |
| History | parameters.json, manifests, selection_report and breeding_log | Seeds, parents, inherited genes, mutations and settings |

The current implementation is specialised Python code, not a universal plugin
API. Adding a new gene requires updating sampling, decoding, validation,
crossover groups, mutation bounds and serialisation together.
Another application may need a different topology or different crossover rules.

The collector emitted into a run directory is a standalone copy of
wing_generator/openfoam_results.py. Change the tracked source when modifying
collection, then regenerate the run scripts.

## Result Contract

The existing evolution loader reads
<source-generation>/run/all_results_summary.csv and joins rows to
<source-generation>/design_*/parameters.json by design_id.

Each input row needs design_id, final_time, nominal_height_above_ground_m,
and Cl, Cd, CmPitch (or their nonempty *_avg_last_window values).
The standard collector also supplies ground_case, ground_plane_z_m, status,
case_dir, output paths and y+ statistics.

If you retain this aerodynamic scorer, keep its coefficient sign conventions,
reference area and reference length consistent across evaluations. CmPitch is
the moment coefficient consumed as C_m in the joined dataset. If you replace the
scorer, give your new metrics meaningful names and units instead of labelling
unrelated performance measurements as aerodynamic coefficients.

For another tool, define an explicit completion/failure signal and a unique
condition ID. Require the expected set of conditions per individual. A missing,
failed or non-finite evaluation must not silently become a competitive score.
The current full-sweep check uses row counts and validity; an adapter receiving
duplicate rows or arbitrary conditions should validate their identities too.

## Validate a Small Experiment

1. Decode one known genome twice and confirm the same model and metadata.
2. Export it and verify dimensions, units, topology and constraints in the
   downstream tool.
3. Run one authorised evaluation and parse its output. Exercise a failed,
   incomplete and non-finite result as well as a valid result.
4. Use a tiny synthetic population to verify selection direction, parent
   pairing, mutation bounds and child validity. Record that the results are
   synthetic; keep them separate from measured data.
5. Execute one small real generation within the stated budget, inspect the
   quality of its evaluations, and then increase the population or generation
   count if the evidence supports it.

For this wing example, a normal solver End or reaching iteration 300 is a
completion criterion, not a convergence certificate. Mesh quality, layer
coverage, y+, residuals and coefficient histories remain separate checks.
The included STEP exporter is optional; a STEP file existing on disk does not
establish agreement with the evaluated mesh.

## Interpret Evolution Carefully

Keep an unchanged baseline and compare raw objective values under the same
conditions. The current weighted min-max scores depend on the eligible
population, so a larger score in a later generation does not by itself show
improvement. Repeated seeds can help reveal whether an outcome depends on luck.

This implementation selects parents from the immediate source generation,
uses uniform inheritance within configured gene groups, and replaces the
population with children. It does not implement biological dominance,
tournament selection, automatic elitist copying, or an automatic convergence
stopping rule. Those are possible extensions, not existing features.

Keep meaningful biological analogies grounded in the implemented operations:
genotype, phenotype, variation, inheritance, selection pressure and generations.
Document changes to the experiment so that results remain attributable to the
designs and settings that actually produced them.
