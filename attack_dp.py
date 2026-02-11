import numpy as np
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
def print_timers():
    print("\n=== Timers ===")
    for k, v in sorted(TIMERS.items()):
        if not k.startswith('_'):
            print(f"{k:20s}: {v*1000:8.1f}ms")

# ============== Precomputation ==============
def precompute(q, sigma):
    rho_Z = sum(np.exp(-pi * n**2 / sigma**2) for n in range(-100, 101))
    r = np.arange(q)
    ctr = np.where(r > q//2, r - q, r).astype(np.float64)
    weight = np.exp(-pi * (ctr / sigma)**2) / rho_Z
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
    # Discrete Gaussian approximation: round(N(0, sigma^2))
    sigma_cont = sigma / sqrt(2 * pi)
    e = np.rint(rng.normal(0.0, sigma_cont, size=m)).astype(np.int64)
    b = (A @ s + e) % q
    return A, b, supports, s

# ============== Core computation (No FFT) ==============
def W_to_log_contrib(W):
    """W[3] -> log-contribution[3]"""
    Z = W.sum()
    if Z > 0:
        return np.where(W > 0, np.log(W / Z), -1e10)
    return np.full(3, -1e10)

def dp_forward(cnt, a, q):
    """cnt[r] -> cnt'[r] = cnt[r] + cnt[r-a] + cnt[r+a] (mod q)"""
    return cnt + np.roll(cnt, a) + np.roll(cnt, -a)

def compute_row_contributions_naive(t, A, b, supports, q, weight):
    """Compute contributions for all variables in row t via naive DP."""
    S = supports[t]
    base = int(b[t])
    num_vars = len(S)

    if num_vars == 0:
        return {}

    support_list = [int(j) for j in S]
    a_vals = [int(A[t, j]) for j in support_list]

    # Single variable
    if num_vars == 1:
        i, a = support_list[0], a_vals[0]
        W = np.array([weight[(base + a) % q], weight[base], weight[(base - a) % q]])
        return {i: W_to_log_contrib(W)}

    result = {}

    # For each variable i, DP over the remaining variables
    for idx, i in enumerate(support_list):
        cnt = np.zeros(q, dtype=np.float64)
        cnt[0] = 1.0

        # Forward DP over all variables except i
        for jdx, j in enumerate(support_list):
            if jdx != idx:
                cnt = dp_forward(cnt, a_vals[jdx], q)

        # Compute W: sum of cnt[r] * weight[base - a_i*u - r]
        a = a_vals[idx]
        W = np.array([
            np.dot(cnt, np.roll(weight, base + a)),  # u = -1
            np.dot(cnt, np.roll(weight, base)),       # u = 0
            np.dot(cnt, np.roll(weight, base - a))    # u = +1
        ])
        result[i] = W_to_log_contrib(W)

    return result

def softmax(log_L):
    max_val = np.max(log_L)
    exp_vals = np.exp(log_L - max_val)
    return exp_vals / np.sum(exp_vals)

# ============== Main algorithm ==============
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
        contribs = compute_row_contributions_naive(t, A, b, supports, q, weight)
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
                new_contribs = compute_row_contributions_naive(t, A, b, supports, q, weight)
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
    """Run experiment and return result dict with metadata."""
    
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
    parser = argparse.ArgumentParser()
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
