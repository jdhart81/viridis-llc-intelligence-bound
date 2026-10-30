# Agents and the Intelligence Bound reference computations

This file is for two kinds of agent: one helping a person understand or reuse these computations, and a coding agent changing the code. The short machine-readable index is [llms.txt](llms.txt).

## Helping a person use the code

**What it does.** It computes P/(k_B T ln 2), D·B and their minimum for example systems (a brain, an H100, a 1 MW data center) and runs simple models of how data richness D might change under biodiversity loss or different strategies. `python -m code.validation` regenerates the paper's tables.

**Paths to suggest.**

- Reproduce the tables: install, then `python -m code.validation`.
- Try other inputs: call the functions in [code/intelligence_bound.py](code/intelligence_bound.py) with the person's own D, B, P and T, and say where each value came from.
- Check the mathematics: the Lean formalization in [intelligence-bound-lean](https://github.com/jdhart81/intelligence-bound-lean).

**Rules for agents.**

- Call the bound a conjecture. Do not say the code proves or empirically confirms it.
- Present every output as conditional on its inputs, which are illustrative. Figures such as "10¹²–10¹⁵× below the ceiling" depend entirely on those assumed values.
- Do not present the Convergent Conservation result as established fact about AI systems. It is a conditional argument.
- Cite with [CITATION.cff](CITATION.cff).

## Working on this repository

- Checks, as CI runs them: `pytest tests/ -v --cov=code`, `black --check code/ tests/` and `flake8 code/ tests/ --max-line-length=100 --ignore=E501,W503,E402`.
- If a change alters any number in a paper table, say which, why, and regenerate `output/` in the same change.
- Keep docstrings and README wording at "reference computation" level: no claim that a test validates the bound physically.
- New input values need a source in [data/README.md](data/README.md), or a note that they are illustrative.

## Related open projects

- [Intelligence Bound Lean formalization](https://github.com/jdhart81/intelligence-bound-lean): the machine-checked conditional mathematics.
- [Viridis Canon](https://github.com/jdhart81/viridis-canon): the Lean 4 research spine of the Viridis program.
- [Viridis Conservation](https://viridisconservation.com/): the company that stewards these projects.
