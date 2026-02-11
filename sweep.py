"""
Parameter sweep runner for sparse LWE experiments.

Sweeps each of the 5 parameters (m, n, k, q, sigma) and measures
the empirical threshold of LHS = m*k*q / (n * 3^(k-1) * sigma * ln(3n)).

Usage:
    python sweep.py              # run all 5 sweeps
    python sweep.py m n          # run only m, n sweeps
    python sweep.py --list       # list available sweeps
"""
from attack_dp import run_experiment
import csv
import math
import sys
import numpy as np


# ═══════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════

def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def nearest_prime(n):
    """Return the prime closest to n (ties: pick the smaller one)."""
    if is_prime(n):
        return n
    lo, hi = n - 1, n + 1
    while True:
        if is_prime(lo):
            return lo
        if is_prime(hi):
            return hi
        lo -= 1
        hi += 1


def compute_lhs(m, k, q, n, sigma):
    """LHS = m·k·q / (n · 3^(k-1) · σ · ln(3n))"""
    return (m * k * q) / (n * (3 ** (k - 1)) * sigma * math.log(3 * n))


# ═══════════════════════════════════════════
#  Experiment configs
# ═══════════════════════════════════════════

EXPERIMENTS = [
    {
        'name': 'm',
        'fixed': dict(n=200, k=10, q=1021, sigma=3.19),
        'sweep_param': 'm',
        'sweep_values': list(range(400, 2401, 200)),  # 400..2400 step 200
    },
    {
        'name': 'n',
        'fixed': dict(k=10, q=1021, sigma=3.19, m=1600),
        'sweep_param': 'n',
        'sweep_values': list(range(50, 601, 50)),  # 50..600 step 50
    },
    {
        'name': 'k',
        'fixed': dict(n=200, q=1021, sigma=3.19, m=1600),
        'sweep_param': 'k',
        'sweep_values': list(range(8, 13)),  # 8..12
    },
    {
        'name': 'q',
        'fixed': dict(n=200, k=10, sigma=3.19, m=1600),
        'sweep_param': 'q',
        'sweep_values': [nearest_prime(x) for x in range(100, 1401, 100)],
    },
    {
        'name': 'sigma',
        'fixed': dict(n=200, k=10, q=1021, m=1600),
        'sweep_param': 'sigma',
        'sweep_values': list(range(1, 13)),  # 1..12
    },
]

SEEDS = list(range(100))


# ═══════════════════════════════════════════
#  Sweep runner
# ═══════════════════════════════════════════

def run_one_sweep(exp, seeds=None):
    if seeds is None:
        seeds = SEEDS

    name = exp['name']
    fixed = exp['fixed']
    sweep_param = exp['sweep_param']
    sweep_values = exp['sweep_values']

    detail_file = f"sweep_{name}_detail.csv"
    summary_file = f"sweep_{name}.csv"

    detail_fields = [sweep_param, 'seed', 'success', 'accuracy', 'time_total', 'lhs']
    summary_fields = [sweep_param, 'num_seeds', 'success_rate', 'acc_mean', 'acc_std', 'time_mean', 'lhs']

    import os
    detail_exists = os.path.exists(detail_file) and os.path.getsize(detail_file) > 0

    # Load existing (param, seed) pairs to skip
    existing = set()
    if detail_exists:
        with open(detail_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add((row[sweep_param], row['seed']))

    fd = open(detail_file, 'a', newline='')
    fs = open(summary_file, 'w', newline='')
    wd = csv.DictWriter(fd, fieldnames=detail_fields)
    ws = csv.DictWriter(fs, fieldnames=summary_fields)
    if not detail_exists:
        wd.writeheader()
    ws.writeheader()

    print(f"\n{'='*60}")
    print(f"  Sweep: {name}  ({sweep_param} = {sweep_values})")
    print(f"  Fixed: {fixed}")
    print(f"  Seeds: {seeds}")
    print(f"{'='*60}")

    for val in sweep_values:
        params = {**fixed, sweep_param: val}

        # Compute LHS for this point
        lhs = compute_lhs(
            m=params.get('m', 0),
            k=params.get('k', 0),
            q=params.get('q', 0),
            n=params.get('n', 0),
            sigma=params.get('sigma', 0),
        )

        successes, accs, times = [], [], []

        for seed in seeds:
            if (str(val), str(seed)) in existing:
                continue
            params['seed'] = seed
            r = run_experiment(**params)

            successes.append(r['success'])
            accs.append(r['accuracy'])
            times.append(r['time_total'])

            wd.writerow({
                sweep_param: val, 'seed': seed,
                'success': int(r['success']),
                'accuracy': f"{r['accuracy']:.6f}",
                'time_total': f"{r['time_total']:.3f}",
                'lhs': f"{lhs:.6f}",
            })
            fd.flush()

        sr = np.mean(successes)
        am = np.mean(accs)
        astd = np.std(accs)
        tm = np.mean(times)

        ws.writerow({
            sweep_param: val, 'num_seeds': len(seeds),
            'success_rate': f"{sr:.4f}",
            'acc_mean': f"{am:.6f}",
            'acc_std': f"{astd:.6f}",
            'time_mean': f"{tm:.3f}",
            'lhs': f"{lhs:.6f}",
        })
        fs.flush()

        mark = "OK" if sr == 1.0 else f"{sr*100:.0f}%"
        print(f"  {sweep_param}={val:<8} success={mark:<6} acc={am:.4f}±{astd:.4f}  LHS={lhs:.4f}  t={tm:.1f}s")

    fd.close()
    fs.close()
    print(f"  Detail  -> {detail_file}")
    print(f"  Summary -> {summary_file}")


def main():
    names = {e['name'] for e in EXPERIMENTS}

    if '--list' in sys.argv:
        print("Available sweeps:")
        for e in EXPERIMENTS:
            print(f"  {e['name']:8s}  {e['sweep_param']} = {e['sweep_values']}")
        return

    # Select experiments by CLI args, or run all
    selected = [a for a in sys.argv[1:] if not a.startswith('-')]
    if selected:
        unknown = [s for s in selected if s not in names]
        if unknown:
            print(f"Unknown sweep(s): {unknown}")
            print(f"Available: {sorted(names)}")
            sys.exit(1)
        experiments = [e for e in EXPERIMENTS if e['name'] in selected]
    else:
        experiments = EXPERIMENTS

    for exp in experiments:
        run_one_sweep(exp)

    print(f"\nDone. Ran {len(experiments)} sweep(s).")


if __name__ == "__main__":
    main()
