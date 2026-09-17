# Coordinate Marginal Recovery for Sparse LWE

Implementation of Algorithm 1 (CMR) and reproduction tools for Section 5.
The repository includes the recorded 4,200 trials underlying the paper's
42 experimental batches, so the figures can be regenerated without rerunning
the experiments. The earlier implementation remains in Git history only.

## Install

Use Python 3.10 or later and NumPy 2.x:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

NumPy is the only Python dependency. TeX Live with PGFPlots, subcaption and
float is needed only to compile the generated figure preview, not to run
the attack, tests, or analysis.

## Quick validation (no saved run or log)

From the repository root:

```sh
python -B -m unittest discover -s tests -v
```

These tests use small instances, direct-enumeration checks and isolated
temporary directories. They also check the archived counts and Gamma
calculation. All temporary test outputs are removed; the suite does not
run the 4,200-trial experiment or change the recorded data. Test progress
is printed to the terminal, not written to a log file.

## Reproduce the paper figures from recorded data

```sh
python -B analyze_results.py
python -B make_pgfplots_figures.py
```

The first command aggregates the immutable trial records into
`results/paper/summary_results.csv`. The second writes `figures.tex` and
`preview.tex` in that directory. To view both figures:

```sh
cd results/paper
pdflatex -interaction=nonstopmode -halt-on-error preview.tex
cd ../..
```

Figure 1 shows the five sweeps with Wilson 95% error bars. Figure 2 contains
the same observed rates as points on the Gamma axis, without a fitted curve.
The plotted coordinates and error bars reproduce the paper to its displayed
rounding precision; the standalone preview is not the paper's page layout.

We recompute

```text
theta = (1 - 2/(q-1))/3
Gamma = (m*k/n) * min(1, (q/sigma)*theta**(k-1)) / ln(2*n).
```

The original CSV fields `gamma_corrected` and `gamma_old` use an older
normalization and are retained only as historical data. Neither is used
as the paper's Gamma. See `data/paper/README.md` for provenance and checksums.
Repeated reference configurations remain separate batches, not pooled.

## Optional new experiments

The following command runs the full paper grid; it is not needed to regenerate
the figures and is not run by the test suite:

```sh
python -B run_experiments.py --preset paper --output-dir results/new-paper
```

Defaults match the recorded experiment: base `(n,m,k,q,sigma) =
(200,1600,8,1021,3.19)`, five grids in `data/paper/experiment_config.json`,
100 trials per batch, master seed `820260828`, `B=8`, cached engine,
float32 cache, and smallest-candidate tie order `-1,0,1`.
For a small saved run instead:

```sh
python -B run_experiments.py --preset smoke --output-dir results/smoke
python -B analyze_results.py --trials results/smoke/trial_results.csv --output-dir results/smoke-figures
python -B make_pgfplots_figures.py --summary results/smoke-figures/summary_results.csv --output-dir results/smoke-figures
```

Generated artifacts are allowed only in subdirectories of `results/`.
`data/paper/` is never an output destination. Repeating an identical run
skips completed trials and recomputes the summary from all saved trials.
Different settings, including trial count, master seed, grid, engine,
precision and Python/NumPy versions, require a new output directory.
Do not run two writers against the same output directory concurrently.
A malformed/partial CSV row is reported rather than silently discarded.

Trial RNGs are constructed from
`np.random.SeedSequence([master_seed, point_index, trial])`, with zero-based
indices and sweep order `m,n,k,q,sigma`. Within each sweep the grid order is
the order in the configuration. The CSV `seed` is a generated identifier:
passing that integer alone to `default_rng` does **not** reproduce the trial.
Runtime measurements are machine-dependent and are not expected to match.
The archived run did not record Python/NumPy versions; new runs record both.

## Algorithm and numerical implementation

`sparse_lwe.py` samples exactly k support positions per row and uniform
nonzero coefficients. Errors are drawn from a numerically truncated discrete
Gaussian proportional to `exp(-pi*x*x/sigma**2)`, not a rounded continuous
Gaussian. Masses are wrapped modulo q to form the row-weight table.
The truncation parameter `1e-18` bounds the first omitted unnormalized
point mass; it is not claimed to bound the entire tail probability.

Recovery processes coordinates in fixed order. Each row's marginal weight
is centered, normalized by `max(1, Xi*theta**r)`, capped above at B, and
summed before selecting a candidate and updating the residuals. Xi is
computed by `q*sum(w*w)-1`. No secret or error metadata is used by recovery.
By default the sampler draws a uniform ternary secret as in Section 5;
`sample_instance(..., secret=...)` also supports a prescribed ternary secret.

Two convolution implementations are available:

- `reference`: O(m*q*k^2+n) recovery arithmetic and O(m*k+n+q) storage.
- `cached`: O(m*q*k+n) recovery arithmetic and O(m*q*k+n) storage.

These are Section 3.2's implementations of the same mathematical score,
given the error-mass table. Floating-point results can differ, especially
near ties. The library defaults to a float64 cache; the paper experiment
preset explicitly uses float32. All score aggregation is float64.
Instance generation separately uses O(m*n) temporary random-key storage to
preserve the sampling sequence used for the archived experiments.

The theorem assumes B>4, sigma>=1 and prime q>=max(11,4*sigma). The code
allows some inputs outside that theorem regime (q>=4) for diagnostics;
the generated trial records flag whether the theorem's assumptions hold.
This is an experiment-sized NumPy implementation, not an arbitrary-precision
implementation for cryptographic-size moduli.

## Files

- `sparse_lwe.py`: sampler, reference and cached CMR, theorem scale.
- `run_experiments.py`: seeded experiment execution and checked resume.
- `analyze_results.py`: batch counts, Wilson intervals, current Gamma.
- `make_pgfplots_figures.py`: the two Section 5 figures.
- `artifact_io.py`: output isolation and atomic derived-file writes.
- `data/paper/`: unchanged recorded data and configuration.
- `tests/`: numerical, data-integrity and workflow checks.

This repository covers Gaussian-error CMR and Section 5 experiments.
It does not implement the Appendix's ternary-error comparison or claim a
new recovery threshold based on the observed data.
