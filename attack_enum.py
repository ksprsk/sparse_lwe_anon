"""
Greedy row-marginal attack on sparse-row LWE (Algorithm 1).

Full-enumeration version: for each row with current support size ell,
enumerate all 3^ell ternary assignments and accumulate row-marginal
weights W_t^{(i)}(u) for every coordinate i in the support simultaneously.

Complexity per row update: O(ell * 3^ell).
Total complexity: O(m * k * 3^k).
"""
import numpy as np
import itertools
import time
from math import sqrt, pi
from tqdm import tqdm, trange

# ============== Timing ==============
TIMERS = {}
def tick(name):
    TIMERS[name] = TIMERS.get(name, 0)
    TIMERS[f"_{name}_start"] = time.time()
def tock(name):
    TIMERS[name] += time.time() - TIMERS[f"_{name}_start"]
def reset_timers():
    global TIMERS
    TIMERS = {}

# ============== Precomputation ==============
def precompute(q, sigma):
    """Precompute weight function w(r) = exp(-pi * r^2 / sigma^2)."""
    rho_Z = sum(np.exp(-pi * n**2 / sigma**2) for n in range(-100, 101))
    r = np.arange(q)
    ctr = np.where(r > q // 2, r - q, r).astype(np.float64)
    weight = np.exp(-pi * (ctr / sigma) ** 2) / rho_Z
    return weight

# ============== Sampling ==============
def sample_instance(n, m, k, q, sigma, seed=0):
    rng = np.random.default_rng(seed)
    A = np.zeros((m, n), dtype=np.int64)
    supports = []
    for t in range(m):
        S = rng.choice(n, size=k, replace=False)
        supports.append(S.astype(np.int64))
        A[t, S] = rng.integers(1, q, size=k)
    s = rng.integers(-1, 2, size=n, dtype=np.int64)
    sigma_cont = sigma / sqrt(2 * pi)
    e = np.rint(rng.normal(0.0, sigma_cont, size=m)).astype(np.int64)
    b = (A @ s + e) % q
    return A, b, supports, s

# ============== Core: full enumeration (matches Algorithm 1) ==============
def W_to_log_contrib(W):
    """W[3] -> log-contribution[3]"""
    Z = W.sum()
    if Z > 0:
        return np.where(W > 0, np.log(W / Z), -1e10)
    return np.full(3, -1e10)

def compute_row_contributions_enum(t, A, b, supports, q, weight):
    """
    Compute row-marginal weights for all coordinates in row t
    via a single pass over all 3^ell ternary assignments.

    For each assignment x in {-1,0,1}^{S_t}:
      r = (b_t - sum_{j in S_t} a_{t,j} x_j) mod q
      For each i in S_t: accumulate w(r) into W_t^{(i)}(x_i)

    Returns: dict {i: log_contrib[3]} for each i in S_t.
    """
    S = supports[t]
    base = int(b[t])
    ell = len(S)

    if ell == 0:
        return {}

    support_list = [int(j) for j in S]
    a_vals = np.array([int(A[t, j]) for j in support_list], dtype=np.int64)

    # Generate all 3^ell assignments: shape (3^ell, ell)
    assignments = np.array(
        list(itertools.product([-1, 0, 1], repeat=ell)),
        dtype=np.int64
    )

    # Compute all residuals at once: shape (3^ell,)
    residuals = (base - assignments @ a_vals) % q

    # Look up weights for all residuals
    w_vals = weight[residuals]

    # Accumulate W_t^{(i)}(u) for each coordinate and each candidate value
    result = {}
    for idx, i in enumerate(support_list):
        col = assignments[:, idx]
        W = np.array([
            w_vals[col == -1].sum(),  # u = -1
            w_vals[col ==  0].sum(),  # u =  0
            w_vals[col ==  1].sum(),  # u = +1
        ])
        result[i] = W_to_log_contrib(W)

    return result

def softmax(log_L):
    max_val = np.max(log_L)
    exp_vals = np.exp(log_L - max_val)
    return exp_vals / np.sum(exp_vals)

# ============== Main algorithm (Algorithm 1: GreedySpLWE) ==============
def greedy_recover(A, b, supports, q, sigma, verbose=False):
    reset_timers()
    m, n = A.shape

    tick('precompute')
    weight = precompute(q, sigma)
    tock('precompute')

    tick('build_index')
    Ti = [[] for _ in range(n)]
    for t, S in enumerate(supports):
        for i in S:
            Ti[int(i)].append(t)
    tock('build_index')

    remaining = set(range(n))
    s_hat = np.zeros(n, dtype=np.int64)
    b = b.copy()
    supports = [S.copy() for S in supports]

    tick('init_contrib')
    contribution = {}
    all_log_L = {i: np.zeros(3, dtype=np.float64) for i in range(n)}

    for t in (trange(m) if verbose else range(m)):
        contribs = compute_row_contributions_enum(t, A, b, supports, q, weight)
        for i, c in contribs.items():
            contribution[(i, t)] = c
            all_log_L[i] += c
    tock('init_contrib')

    for step in range(n):
        tick('select')
        best_P, best_i, best_u = -1.0, None, None
        for i in remaining:
            P = softmax(all_log_L[i])
            u_idx = np.argmax(P)
            if P[u_idx] > best_P:
                best_P, best_i, best_u = P[u_idx], i, [-1, 0, 1][u_idx]
        tock('select')

        if best_i is None:
            break

        tick('update')
        s_hat[best_i] = best_u

        for t in Ti[best_i]:
            b[t] = (b[t] - A[t, best_i] * best_u) % q

            for j in supports[t]:
                j = int(j)
                if j != best_i and j in remaining:
                    all_log_L[j] -= contribution[(j, t)]

            supports[t] = supports[t][supports[t] != best_i]
            contribution.pop((best_i, t), None)

            if len(supports[t]) > 0:
                new_contribs = compute_row_contributions_enum(
                    t, A, b, supports, q, weight
                )
                for j, c in new_contribs.items():
                    if j in remaining:
                        contribution[(j, t)] = c
                        all_log_L[j] += c

        remaining.remove(best_i)
        del all_log_L[best_i]
        tock('update')

    return s_hat, {k: v for k, v in TIMERS.items() if not k.startswith('_')}

# ============== Experiment runner ==============
def run_experiment(n, k, q, sigma, m=None, seed=0, verbose=False):
    t_start = time.time()
    A, b, supports, s = sample_instance(n, m, k, q, sigma, seed=seed)
    t_sample = time.time() - t_start

    s_hat, timers = greedy_recover(A, b, supports, q, sigma, verbose=verbose)

    match = bool(np.all(s == s_hat))
    accuracy = float(np.mean(s == s_hat))
    total_time = t_sample + sum(timers.values())

    return {
        'success': match,
        'accuracy': accuracy,
        'n': n, 'k': k, 'q': q, 'sigma': sigma, 'm': m, 'seed': seed,
        'time_sample': t_sample,
        'time_total': total_time,
        **{f'time_{k}': v for k, v in timers.items()}
    }

# ============== CLI ==============
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Greedy row-marginal attack on sparse-row LWE (enumeration)"
    )
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--q', type=int, default=1021)
    parser.add_argument('--sigma', type=float, default=3.19)
    parser.add_argument('--m', type=int, default=1600)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    result = run_experiment(args.n, args.k, args.q, args.sigma, args.m, args.seed, args.verbose)

    print(f"\n{'='*40}")
    print(f"Success: {result['success']}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Total time: {result['time_total']*1000:.1f}ms")
    print(f"  sample: {result['time_sample']*1000:.1f}ms")
    print(f"  precompute: {result['time_precompute']*1000:.1f}ms")
    print(f"  build_index: {result['time_build_index']*1000:.1f}ms")
    print(f"  init_contrib: {result['time_init_contrib']*1000:.1f}ms")
    print(f"  select: {result['time_select']*1000:.1f}ms")
    print(f"  update: {result['time_update']*1000:.1f}ms")
