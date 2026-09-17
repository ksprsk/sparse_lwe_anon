from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import artifact_io
from analyze_results import analyze, read_trials, summarize, wilson_interval
from make_pgfplots_figures import build, generate, normalized_rows, plot_tables
from run_experiments import make_config, points, run, trial_rng
from sparse_lwe import (SparseLWEInstance, discrete_gaussian_pmf, fourier_energy,
                        modular_error_mass, recover, sample_instance, theorem_predictor)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/paper"
HASHES = {
    "trial_results.csv": "dea77c73b107b40d690689d3a7cf376cfaf09aca23409b80d6884d77c73e2f0d",
    "summary_results.csv": "994abc8986b66510e451808abc26d30315f6952f56146a285b5e592f97f71079",
    "experiment_config.json": "46a37ee67c4a534f3bf37016570f5df675ee01fa5903949358759d3c46185284",
}


def enumeration_recover(instance, B=8.):
    """Independent tiny-instance oracle, including actual residual updates."""
    w = modular_error_mass(instance.q, instance.sigma)
    xi = fourier_energy(w)
    theta = (1 - 2 / (instance.q - 1)) / 3
    residual = instance.b.copy()
    estimates, history = [], []
    for i in range(instance.n):
        scores = np.zeros(3)
        incident = []
        for t, support in enumerate(instance.supports):
            if i not in support:
                continue
            pos = int(np.flatnonzero(support == i)[0])
            a = int(instance.coefficients[t, pos])
            nuisance = instance.coefficients[t, pos + 1:]
            incident.append((t, a))
            for index, u in enumerate((-1, 0, 1)):
                total = 0.
                for assignment in itertools.product((-1, 0, 1), repeat=len(nuisance)):
                    y = int(residual[t]) - a * u - sum(int(c) * x for c, x in zip(nuisance, assignment))
                    total += w[y % instance.q]
                weight = total / 3 ** len(nuisance)
                raw = (instance.q * weight - 1) / max(1, xi * theta ** len(nuisance))
                scores[index] += min(raw, B)
        chosen = (-1, 0, 1)[int(np.argmax(scores))]
        for t, a in incident:
            residual[t] = (int(residual[t]) - a * chosen) % instance.q
        estimates.append(chosen)
        history.append(scores)
    return np.array(estimates), np.array(history)


class NumericalTests(unittest.TestCase):
    def test_modular_mass_and_parseval(self):
        for q, sigma in [(101, 2.7), (11, 6.)]:
            support, pmf = discrete_gaussian_pmf(sigma)
            w = modular_error_mass(q, sigma)
            expected = np.array([pmf[support % q == i].sum() for i in range(q)])
            np.testing.assert_allclose(w, expected, atol=2e-16)
            self.assertAlmostEqual(w.sum(), 1.)
            energy = np.sum(np.abs(np.fft.fft(w)) ** 2) - 1
            np.testing.assert_allclose(fourier_energy(w), energy, rtol=2e-14, atol=2e-14)

    def test_engines_and_enumeration(self):
        for k in (1, 3):
            instance = sample_instance(n=7, m=20, q=31, k=k, sigma=1.5,
                                       rng=np.random.default_rng(61 + k))
            expected, scores = enumeration_recover(instance)
            for engine, dtype in [("reference", "float64"), ("cached", "float64"), ("cached", "float32")]:
                result = recover(instance, engine=engine, cache_dtype=dtype, keep_scores=True)
                np.testing.assert_array_equal(result.estimate, expected)
                tolerance = 2e-5 if dtype == "float32" else 3e-13
                np.testing.assert_allclose(result.scores, scores, rtol=tolerance, atol=tolerance)

    def test_empty_incidence_and_ties(self):
        instance = SparseLWEInstance(n=4, m=1, q=31, k=1, sigma=1.5,
                   supports=np.array([[2]]), coefficients=np.array([[7]]),
                   b=np.array([0]), secret=np.zeros(4, dtype=np.int8), errors=np.array([0]))
        for engine in ("cached", "reference"):
            result = recover(instance, engine=engine, keep_scores=True)
            np.testing.assert_array_equal(result.estimate, [-1, -1, 0, -1])
            np.testing.assert_array_equal(result.scores[[0, 1, 3]], np.zeros((3, 3)))

    def test_recovery_does_not_read_secret_or_error(self):
        instance = sample_instance(n=8, m=20, q=31, k=2, sigma=1.5, rng=np.random.default_rng(3))
        unrelated = replace(instance, secret=-instance.secret, errors=instance.errors + 100)
        np.testing.assert_array_equal(recover(instance).estimate, recover(unrelated).estimate)

    def test_upper_truncation_is_active(self):
        # Repeated coefficients concentrate the row marginal enough to clip.
        instance = SparseLWEInstance(n=6, m=1, q=211, k=6, sigma=1.,
                   supports=np.array([[0, 1, 2, 3, 4, 5]]), coefficients=np.ones((1, 6), dtype=int),
                   b=np.array([0]), secret=np.zeros(6, dtype=np.int8), errors=np.array([0]))
        expected, scores = enumeration_recover(instance, B=4.5)
        self.assertTrue(np.any(scores == 4.5))
        for engine in ("reference", "cached"):
            result = recover(instance, B=4.5, engine=engine, keep_scores=True)
            np.testing.assert_array_equal(result.estimate, expected)
            np.testing.assert_allclose(result.scores, scores, atol=2e-13)
            self.assertTrue(np.all(result.scores <= 4.5))

    def test_fixed_secret_and_strong_instance(self):
        secret = np.resize(np.array([-1, 0, 1]), 12)
        instance = sample_instance(n=12, m=1000, q=101, k=3, sigma=2.,
                                   rng=np.random.default_rng(7), secret=secret)
        np.testing.assert_array_equal(instance.secret, secret)
        np.testing.assert_array_equal(recover(instance).estimate, secret)
        self.assertTrue(np.all(np.diff(instance.supports, axis=1) > 0))
        self.assertTrue(np.all((instance.coefficients > 0) & (instance.coefficients < 101)))

    def test_gamma_and_invalid_parameters(self):
        for q in (31, 1021):
            metric = theorem_predictor(n=20, m=100, k=3, q=q, sigma=2.)
            theta = (1 - 2 / (q - 1)) / 3
            self.assertAlmostEqual(metric["gamma"], 15 * min(1, q / 2 * theta**2) / math.log(40))
            self.assertNotIn("gamma_corrected", metric)
        for sigma in (0., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                discrete_gaussian_pmf(sigma)
        instance = sample_instance(n=3, m=2, k=1, q=11, sigma=1., rng=np.random.default_rng(1))
        for B in (4., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                recover(instance, B=B)


class PaperDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = read_trials(DATA / "trial_results.csv")
        cls.summary = summarize(cls.raw)

    def test_immutable_files(self):
        for name, expected in HASHES.items():
            self.assertEqual(hashlib.sha256((DATA / name).read_bytes()).hexdigest(), expected)

    def test_batches_and_recorded_counts(self):
        self.assertEqual(len(self.raw), 4200)
        self.assertEqual(len(self.summary), 42)
        settings = {(r["n"], r["m"], r["k"], r["q"], r["sigma"]) for r in self.summary}
        self.assertEqual(len(settings), 38)
        self.assertEqual(len({r["seed"] for r in self.raw}), 4200)
        with (DATA / "summary_results.csv").open(newline="") as stream:
            archived = {int(r["point_index"]): r for r in csv.DictReader(stream)}
        for row in self.summary:
            self.assertEqual(row["trials"], 100)
            before = archived[row["point_index"]]
            self.assertEqual(row["successes"], int(before["successes"]))
            for name in ("success_rate", "wilson95_low", "wilson95_high"):
                self.assertAlmostEqual(row[name], float(before[name]), places=14)
        self.assertAlmostEqual(max(r["gamma"] for r in self.summary), 4.91878, places=5)

    def test_seed_identifiers_and_paper_grid(self):
        config = make_config()
        grid = points(config)
        self.assertEqual(len(grid), 42)
        for row in self.raw:
            sweep, parameters = grid[int(row["point_index"])]
            self.assertEqual(sweep, row["sweep"])
            for key, value in parameters.items():
                self.assertEqual(value, float(row[key]))
            _, identifier = trial_rng(config["master_seed"], int(row["point_index"]), int(row["trial"]))
            self.assertEqual(identifier, int(row["seed"]))

    def test_gamma_recomputed_and_plot_structure(self):
        bogus = [dict(row, gamma=-500., gamma_corrected=-900.) for row in self.summary]
        normalized = normalized_rows(bogus)
        self.assertTrue(all(row["gamma"] > 0 for row in normalized))
        first, second = plot_tables(normalized)
        self.assertEqual(sum(map(len, first.values())), 42)
        self.assertEqual(sum(map(len, second.values())), 42)
        tex = build(bogus)
        self.assertEqual(tex.count(r"\begin{figure}"), 2)
        self.assertEqual(tex.count("only marks"), 5)
        self.assertNotIn("logistic", tex)
        self.assertNotIn("corrected", tex)
        self.assertIn("1.54194", tex)

    def test_wilson_edges(self):
        self.assertEqual(wilson_interval(0, 100)[0], 0.)
        self.assertEqual(wilson_interval(100, 100)[1], 1.)
        self.assertAlmostEqual(wilson_interval(0, 100)[1], .0369935, places=7)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cmr-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.patch = patch.object(artifact_io, "RESULTS_ROOT", self.root / "results")
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_small_run_resume_analysis_figures(self):
        config = make_config("smoke", trials=1)
        destination = self.root / "results/run"
        self.assertEqual(run(config, destination), 10)
        raw_path = destination / "trial_results.csv"
        original = raw_path.read_bytes()
        self.assertEqual(run(config, destination), 0)
        self.assertEqual(raw_path.read_bytes(), original)
        # Simulate an interrupted run after a complete trial record.
        lines = original.splitlines(keepends=True)
        raw_path.write_bytes(b"".join(lines[:-1]))
        self.assertEqual(run(config, destination), 1)
        replay = read_trials(raw_path)
        self.assertEqual(len(replay), 10)
        original_last = next(csv.DictReader([lines[0].decode(), lines[-1].decode()]))
        for name in ("seed", "full_recovery", "coordinate_accuracy", "first_error"):
            self.assertEqual(replay[-1][name], original_last[name])
        output = self.root / "results/figures"
        analyze(raw_path, output)
        generate(output / "summary_results.csv", output)
        self.assertIn("only marks", (output / "figures.tex").read_text())
        self.assertFalse(list(self.root.rglob("*.log")))

    def test_reject_changed_configuration_before_writes(self):
        config = make_config("smoke", trials=1)
        destination = self.root / "results/run"
        run(config, destination)
        before = {p.name: p.read_bytes() for p in destination.iterdir()}
        for key, value in [("master_seed", 1), ("B", 9.), ("engine", "reference"),
                           ("cache_dtype", "float64"), ("trials_per_point", 2)]:
            with self.assertRaises(ValueError):
                run(dict(config, **{key: value}), destination)
        altered = json.loads(json.dumps(config))
        altered["sweeps"]["m"][0] += 1
        with self.assertRaises(ValueError):
            run(altered, destination)
        self.assertEqual(before, {p.name: p.read_bytes() for p in destination.iterdir()})

    def test_reject_duplicate_or_foreign_rows(self):
        config = make_config("smoke", trials=1)
        destination = self.root / "results/run"
        run(config, destination)
        raw = destination / "trial_results.csv"
        original = raw.read_bytes()
        raw.write_bytes(original + original.splitlines(keepends=True)[1])
        with self.assertRaises(ValueError):
            run(config, destination)
        raw.write_bytes(original)
        rows = read_trials(raw)
        rows[0]["seed"] = "1"
        artifact_io.write_csv(raw, rows)
        with self.assertRaises(ValueError):
            run(config, destination)

    def test_output_protection(self):
        for path in (DATA, DATA / "new", self.root, self.root / "results", self.root / "results/../escape"):
            with self.assertRaises(ValueError):
                artifact_io.output_directory(path)
        results = self.root / "results"
        results.mkdir()
        (results / "escape").symlink_to(DATA, target_is_directory=True)
        with self.assertRaises(ValueError):
            artifact_io.output_directory(results / "escape/new")

    def test_cli_end_to_end_in_temporary_copy(self):
        isolated = self.root / "checkout"
        isolated.mkdir()
        for path in ROOT.glob("*.py"):
            shutil.copy2(path, isolated / path.name)
        (isolated / "data/paper").mkdir(parents=True)
        shutil.copy2(DATA / "experiment_config.json", isolated / "data/paper/experiment_config.json")
        commands = [
            ["run_experiments.py", "--preset", "smoke", "--trials", "1", "--output-dir", "results/smoke"],
            ["analyze_results.py", "--trials", "results/smoke/trial_results.csv", "--output-dir", "results/figures"],
            ["make_pgfplots_figures.py", "--summary", "results/figures/summary_results.csv", "--output-dir", "results/figures"],
        ]
        for command in commands:
            subprocess.run([sys.executable, "-B", *command], cwd=isolated, check=True,
                           capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertTrue((isolated / "results/figures/preview.tex").exists())
        self.assertFalse(list(isolated.rglob("__pycache__")))
        self.assertFalse(list(isolated.rglob("*.log")))


if __name__ == "__main__":
    unittest.main()
