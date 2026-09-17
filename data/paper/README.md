# Recorded paper data (read-only inputs)

The following files are byte-for-byte copies of the transition experiment
bundle, not a new run. SHA-256:

| File | SHA-256 |
| --- | --- |
| `trial_results.csv` | `dea77c73b107b40d690689d3a7cf376cfaf09aca23409b80d6884d77c73e2f0d` |
| `summary_results.csv` | `994abc8986b66510e451808abc26d30315f6952f56146a285b5e592f97f71079` |
| `experiment_config.json` | `46a37ee67c4a534f3bf37016570f5df675ee01fa5903949358759d3c46185284` |

There are 4,200 trials: 42 independent batches of 100 trials at 38 distinct
parameter settings. The reference setting was run once in each sweep with
different seeds. Keep all five reference batches separate.

The `full_recovery` field is 0 or 1. `coordinate_accuracy` records the fraction
of correct coordinates; `first_error` is one-based, with 0 for full recovery.
`runtime_seconds` times recovery, not sample generation. `point_index` and
`trial` are zero-based. Recreate RNGs from the configuration's master seed
and these two indices, not from the CSV `seed` identifier alone.

Both archived Gamma columns use ln(3*n): `gamma_corrected` includes the
theta correction and saturation; `gamma_old` uses the older unsaturated
approximation. They are retained verbatim for provenance. Current analysis
recomputes Gamma using ln(2*n) from n,m,k,q,sigma and writes it to a new
summary under results/. The archived summary is never rewritten.

The trial-count and success-count fields are used to reconstruct Wilson
intervals. The paper's repeated reference rates therefore remain independent
observations. There is no pooled logistic fit in the current analysis.
