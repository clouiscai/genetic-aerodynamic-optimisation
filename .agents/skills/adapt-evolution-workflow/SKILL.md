---
name: adapt-evolution-workflow
description: Adapt this repository's biologically inspired optimisation loop to another engineering design or simulation workflow. Use when connecting a new genome, model, evaluator or objective to the evolutionary method.
---

# Adapt an Evolutionary Workflow

Read the repository [README](../../../README.md) and
[adaptation guide](../../../docs/adapting-the-methodology.md), then inspect the
implementation relevant to the user's workflow. Paths below are repository-relative.
This skill is also usable as plain Markdown with another LLM.

## Understand the Implemented Method

- Ten seed designs; ten children per subsequent generation; five CFD conditions
  per design in the default wing experiment.
- Geometry bounds and decoding: wing_generator/config.py, genome.py and
  population_generator.py.
- Selection and breeding: wing_generator/evolution.py, configured by config.json.
  Four parents form two shuffled disjoint pairs with five children per pair.
  Selected parents are not automatically copied unchanged.
- Fitness is a weighted sum of metrics min-max normalised across eligible
  designs within the source generation. Scores are not absolute measures of
  improvement across generations.
- CAD lengths are millimetres; genome and CFD lengths are metres. The model
  represents one half-wing. Reference planform area excludes the winglet.
- CFD case generation uses motorbike_case_generator.py and an external v2412
  motorBike template. openfoam_results.py is copied into generated run folders;
  collect_data.py joins summaries with design metadata.

## Build the Adaptation

Use the user's existing tools, representation and budget. Establish the
individual, mutable genes, fixed quantities, constraints, evaluation conditions,
metric directions and units from their brief and files. Clarify missing details
that would change the design of the integration.

Map these to the implementation boundaries in the adaptation guide. Reuse
selection and reproduction logic where appropriate; implement domain-specific
decoding, validity checks and evaluation adapters. Update every handler when
adding genes: sampling, decoding, bounds, crossover groups, mutation and identity.
Keep coupled genes consistent during inheritance and repair.

Use configurable paths and tool commands. Keep seeds, candidate and condition
IDs, parents, mutation records, raw metrics and evaluation settings with results.
Parse the user's actual result format and check completion, finite values and
expected condition coverage before scoring. Define handling for failed runs.

Build one minimal example and exercise one small generation before scaling.
Use synthetic results for plumbing tests and label them; measured claims require
real evaluation. Follow the user's authorised execution scope and compute budget.
Do not infer authorisation for a long simulation campaign from a documentation
or adapter request.

## Deliver and Verify

Provide the working integration, commands, dependencies, configuration points
and focused checks for reproducibility, invalid inputs, units, offspring validity
and failure handling. Distinguish verified behaviour from unrun integrations.
Describe the method as implemented; avoid claiming global optimality, automatic
convergence or biological mechanisms that the code does not model.
