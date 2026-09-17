"""Algorithm 1: Coordinate Marginal Recovery (CMR), Sections 3.1--3.2.

Residues are stored with the conventional
array indexing 0,...,q-1; this is equivalent to centered representatives for
all table lookups.

Two engines evaluate the same mathematical scores (up to rounding):

* ``engine="reference"`` recomputes the rolling length-q table for every
  incidence pair.  It mirrors the O(m q k^2+n)-time, O(m k+n+q)-storage
  implementation in Lemma 3.1.
* ``engine="cached"`` precomputes every row suffix table.  This reduces the
  repeated work to O(m q k+n) at the cost of O(m k q+n) storage.
  It changes only how convolution tables are reused, not the score definition.

These bounds concern recovery given the error-mass table. The instance
generator separately uses an (m,n) array of random keys to preserve the
sampling sequence used in the published experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np


CANDIDATES = np.asarray([-1, 0, 1], dtype=np.int8)


@dataclass(frozen=True)
class SparseLWEInstance:
    """A sparse-LWE sample matrix in row-compressed form."""

    n: int
    m: int
    q: int
    k: int
    sigma: float
    supports: np.ndarray  # shape (m, k), sorted column indices
    coefficients: np.ndarray  # shape (m, k), entries in {1,...,q-1}
    b: np.ndarray  # shape (m,), residues in {0,...,q-1}
    secret: np.ndarray  # shape (n,), entries in {-1,0,1}
    errors: np.ndarray  # shape (m,), integer discrete-Gaussian errors


@dataclass(frozen=True)
class RecoveryResult:
    estimate: np.ndarray
    scores: np.ndarray | None
    xi: float
    theta: float


def _validate_parameters(n: int, m: int, q: int, k: int, sigma: float) -> None:
    if any(not isinstance(v, (int, np.integer)) for v in (n, m, q, k)):
        raise ValueError("n, m, q, and k must be integers")
    if n < 1 or m < 1 or q < 2 or k < 1:
        raise ValueError("n, m, q, and k must be positive (with q >= 2)")
    if k > n:
        raise ValueError("k must not exceed n")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive")
    if k * (q - 1) > np.iinfo(np.int64).max // 2:
        raise ValueError("parameters exceed safe int64 row arithmetic")


def is_prime(q: int) -> bool:
    """Deterministic primality test adequate for the experiment-sized q."""

    if q < 2:
        return False
    if q % 2 == 0:
        return q == 2
    divisor = 3
    while divisor * divisor <= q:
        if q % divisor == 0:
            return False
        divisor += 2
    return True


def discrete_gaussian_pmf(
    sigma: float, *, tail_tolerance: float = 1e-18
) -> tuple[np.ndarray, np.ndarray]:
    """Return a truncated, renormalized D_{Z,sigma} table.

    The manuscript uses rho_sigma(x)=exp(-pi*x^2/sigma^2).  The truncation
    radius is chosen so that the first omitted unnormalized point mass is at
    most ``tail_tolerance``.  The default is well below double precision at
    the scale of the experiment.
    """

    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive")
    if not 0 < tail_tolerance < 1:
        raise ValueError("tail_tolerance must lie in (0,1)")
    radius = max(
        1,
        int(math.ceil(sigma * math.sqrt(math.log(1.0 / tail_tolerance) / math.pi))),
    )
    support = np.arange(-radius, radius + 1, dtype=np.int64)
    weights = np.exp(-math.pi * (support.astype(np.float64) / sigma) ** 2)
    weights /= weights.sum()
    return support, weights


def modular_error_mass(
    q: int, sigma: float, *, tail_tolerance: float = 1e-18
) -> np.ndarray:
    """Tabulate the modular discrete-Gaussian mass on indices 0,...,q-1."""

    support, weights = discrete_gaussian_pmf(
        sigma, tail_tolerance=tail_tolerance
    )
    w = np.zeros(q, dtype=np.float64)
    np.add.at(w, support % q, weights)
    w /= w.sum()
    return w


def fourier_energy(w: np.ndarray) -> float:
    """Compute Xi=sum_{h != 0} |w_hat(h)|^2 in O(q) using Parseval."""

    q = int(w.size)
    # Under the manuscript's unnormalized DFT, Parseval gives
    # sum_h |w_hat(h)|^2 = q * sum_y |w(y)|^2 and w_hat(0)=1.
    return max(0.0, float(q * np.dot(w, w) - 1.0))


def theorem_predictor(
    n: int, m: int, q: int, k: int, sigma: float
) -> dict[str, float]:
    """Return the theorem LHS and Section 5 Gamma, normalized by ln(2n)."""
    _validate_parameters(n, m, q, k, sigma)
    if q < 4:
        raise ValueError("the signal scale requires q >= 4")
    theta = (1.0 / 3.0) * (1.0 - 2.0 / (q - 1.0))
    attenuation = (q / sigma) * theta ** (k - 1)
    lhs = (m * k / n) * min(1.0, attenuation)
    return {
        "theta": theta,
        "attenuation": attenuation,
        "theorem_lhs": lhs,
        "gamma": lhs / math.log(2.0 * n),
    }


def sample_instance(
    n: int,
    m: int,
    q: int,
    k: int,
    sigma: float,
    rng: np.random.Generator,
    *,
    tail_tolerance: float = 1e-18,
    secret: np.ndarray | None = None,
) -> SparseLWEInstance:
    """Sample spLWE; by default draw the secret uniformly as in Section 5.

An explicit fixed ternary secret is also supported. The default RNG call
order is unchanged from the archived experiment implementation.
"""

    _validate_parameters(n, m, q, k, sigma)
    if secret is None:
        secret = rng.integers(-1, 2, size=n, dtype=np.int8)
    else:
        secret = np.asarray(secret)
        if secret.shape != (n,) or not np.isin(secret, CANDIDATES).all():
            raise ValueError("secret must have n entries in {-1,0,1}")
        secret = secret.astype(np.int8, copy=True)

    # Independent continuous keys make each row's k smallest indices a
    # uniform k-subset of [n].  Sorting gives the natural processing order.
    keys = rng.random((m, n))
    supports = np.argpartition(keys, k - 1, axis=1)[:, :k]
    supports.sort(axis=1)
    supports = supports.astype(np.int32, copy=False)

    coefficients = rng.integers(1, q, size=(m, k), dtype=np.int64)
    error_support, error_pmf = discrete_gaussian_pmf(
        sigma, tail_tolerance=tail_tolerance
    )
    errors = rng.choice(error_support, size=m, p=error_pmf).astype(np.int64)
    if int(error_support[-1]) > np.iinfo(np.int64).max // 2:
        raise ValueError("error support exceeds safe int64 arithmetic")

    row_signal = np.sum(
        coefficients * secret[supports].astype(np.int64), axis=1, dtype=np.int64
    )
    b = (row_signal + errors) % q
    return SparseLWEInstance(
        n=n,
        m=m,
        q=q,
        k=k,
        sigma=sigma,
        supports=supports,
        coefficients=coefficients,
        b=b,
        secret=secret,
        errors=errors,
    )


def _incidence_lists(
    supports: np.ndarray, n: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    row_lists: list[list[int]] = [[] for _ in range(n)]
    pos_lists: list[list[int]] = [[] for _ in range(n)]
    for row in range(supports.shape[0]):
        for pos, column in enumerate(supports[row]):
            row_lists[int(column)].append(row)
            pos_lists[int(column)].append(pos)
    rows = [np.asarray(items, dtype=np.int32) for items in row_lists]
    positions = [np.asarray(items, dtype=np.int32) for items in pos_lists]
    return rows, positions


def _suffix_cache(
    coefficients: np.ndarray, w: np.ndarray, dtype: np.dtype
) -> np.ndarray:
    """Precompute the suffix convolution table at each row/support position."""

    m, k = coefficients.shape
    q = w.size
    cache = np.empty((m, k, q), dtype=dtype)
    initial = w.astype(dtype, copy=False)
    for row in range(m):
        p = initial.copy()
        cache[row, k - 1] = p
        for pos in range(k - 2, -1, -1):
            a = int(coefficients[row, pos + 1])
            p = (np.roll(p, a) + p + np.roll(p, -a)) / 3.0
            cache[row, pos] = p
    return cache


def _recover_cached(
    instance: SparseLWEInstance,
    w: np.ndarray,
    xi: float,
    theta: float,
    B: float,
    *,
    cache_dtype: np.dtype,
    keep_scores: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    supports = instance.supports
    coefficients = instance.coefficients
    q, n, k = instance.q, instance.n, instance.k
    cache = _suffix_cache(coefficients, w, cache_dtype)
    incidence_rows, incidence_pos = _incidence_lists(supports, n)
    residual = instance.b.copy()
    estimate = np.empty(n, dtype=np.int8)
    score_history = np.empty((n, 3), dtype=np.float64) if keep_scores else None
    denominator = np.maximum(
        1.0, xi * theta ** np.arange(k - 1, -1, -1, dtype=np.float64)
    )

    for column in range(n):
        rows = incidence_rows[column]
        positions = incidence_pos[column]
        scores = np.zeros(3, dtype=np.float64)
        if rows.size:
            a = coefficients[rows, positions]
            b = residual[rows]
            denom = denominator[positions]
            for candidate_index, candidate in enumerate(CANDIDATES):
                lookup = (b - a * int(candidate)) % q
                weights = cache[rows, positions, lookup].astype(
                    np.float64, copy=False
                )
                raw = (q * weights - 1.0) / denom
                scores[candidate_index] = np.minimum(raw, B).sum()
        chosen = int(CANDIDATES[int(np.argmax(scores))])
        estimate[column] = chosen
        if score_history is not None:
            score_history[column] = scores
        if rows.size:
            residual[rows] = (residual[rows] - a * chosen) % q
    return estimate, score_history


def _recover_reference(
    instance: SparseLWEInstance,
    w: np.ndarray,
    xi: float,
    theta: float,
    B: float,
    *,
    keep_scores: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    supports = instance.supports
    coefficients = instance.coefficients
    q, n, k = instance.q, instance.n, instance.k
    incidence_rows, incidence_pos = _incidence_lists(supports, n)
    residual = instance.b.copy()
    estimate = np.empty(n, dtype=np.int8)
    score_history = np.empty((n, 3), dtype=np.float64) if keep_scores else None
    denominator = np.maximum(
        1.0, xi * theta ** np.arange(k - 1, -1, -1, dtype=np.float64)
    )

    for column in range(n):
        scores = np.zeros(3, dtype=np.float64)
        for row, pos_value in zip(
            incidence_rows[column], incidence_pos[column], strict=True
        ):
            row = int(row)
            pos = int(pos_value)
            p = w.copy()
            for nuisance_pos in range(pos + 1, k):
                nuisance_a = int(coefficients[row, nuisance_pos])
                p = (np.roll(p, nuisance_a) + p + np.roll(p, -nuisance_a)) / 3.0
            a = int(coefficients[row, pos])
            for candidate_index, candidate in enumerate(CANDIDATES):
                lookup = (int(residual[row]) - a * int(candidate)) % q
                raw = (q * p[lookup] - 1.0) / denominator[pos]
                scores[candidate_index] += min(float(raw), B)
        chosen = int(CANDIDATES[int(np.argmax(scores))])
        estimate[column] = chosen
        if score_history is not None:
            score_history[column] = scores
        for row, pos_value in zip(
            incidence_rows[column], incidence_pos[column], strict=True
        ):
            row = int(row)
            pos = int(pos_value)
            residual[row] = (
                int(residual[row]) - int(coefficients[row, pos]) * chosen
            ) % q
    return estimate, score_history


def recover(
    instance: SparseLWEInstance,
    *,
    B: float = 8.0,
    engine: Literal["cached", "reference"] = "cached",
    cache_dtype: str | np.dtype = "float64",
    keep_scores: bool = False,
    tail_tolerance: float = 1e-18,
) -> RecoveryResult:
    """Run Algorithm 1 with fixed tie order -1 < 0 < 1.

    Theorem 3.2 assumes B>4, sigma>=1, prime q, and
    q>=max(11,4*sigma).  The implementation permits broader inputs for
    diagnostics, but callers should check those hypotheses before interpreting
    the theorem predictor.
    """

    _validate_parameters(instance.n, instance.m, instance.q, instance.k, instance.sigma)
    if instance.q < 4:
        raise ValueError("CMR signal normalization requires q >= 4")
    if not math.isfinite(B) or B <= 4:
        raise ValueError("Algorithm 1 requires B > 4")
    w = modular_error_mass(
        instance.q, instance.sigma, tail_tolerance=tail_tolerance
    )
    xi = fourier_energy(w)
    theta = (1.0 / 3.0) * (1.0 - 2.0 / (instance.q - 1.0))
    if engine == "cached":
        dtype = np.dtype(cache_dtype)
        if dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("cache_dtype must be float32 or float64")
        estimate, scores = _recover_cached(
            instance,
            w,
            xi,
            theta,
            B,
            cache_dtype=dtype,
            keep_scores=keep_scores,
        )
    elif engine == "reference":
        estimate, scores = _recover_reference(
            instance, w, xi, theta, B, keep_scores=keep_scores
        )
    else:
        raise ValueError("engine must be 'cached' or 'reference'")
    return RecoveryResult(estimate=estimate, scores=scores, xi=xi, theta=theta)


def cache_size_bytes(m: int, k: int, q: int, dtype: str = "float64") -> int:
    """Return the main suffix-cache allocation size."""

    return int(m * k * q * np.dtype(dtype).itemsize)
