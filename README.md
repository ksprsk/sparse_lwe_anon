# Greedy Row-Marginal Attack on Sparse-Row LWE

Implementation of the greedy coordinate-recovery attack on sparse-row LWE with ternary secrets.

## Files

- `attack_enum.py` — Full-enumeration implementation (Algorithm 1 in the paper). Complexity: O(m·k·3^k).
- `sweep.py` — Parameter sweep runner over (m, n, k, q, σ).
- `plot.py` — Plot generation from sweep results.

## Usage

```bash
# Single run
python attack_enum.py --n 200 --k 10 --q 1021 --sigma 3.19 --m 1600

# Parameter sweeps (100 seeds per point)
python sweep.py          # all sweeps
python sweep.py m k      # selected sweeps

# Plot results
python plot.py
```

## Requirements

Python 3.8+, NumPy, matplotlib, pandas, tqdm.
