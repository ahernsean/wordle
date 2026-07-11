"""Synthetic-distribution tests for the ERDQueue online cost model.

Node costs per branch size are heavy-tailed (near log-normal), so the model
stores a time-weighted geometric mean (a streaming median under log-normal).
These tests drive the estimator with synthetic samples drawn from known
distributions and check the properties the design depends on:

- geometric size bucketing groups nearby sizes so they accumulate together,
  while sizes more than a ~30% step apart stay separate;
- the collapse is the geometric mean (robust), not the arithmetic mean — a
  single injected tarpit barely moves it;
- exponential time-weighting fades old samples and re-converges to a changed
  regime within ~TAU;
- the second log-moment yields the log-normal sigma (get_cost_spread);
- a pre-summed batch (update_cost_model_logsums) is identical to folding the
  same samples one at a time — the inline-buffer flush path is lossless.

All samples share one fixed BUDGET: these tests exercise size-bucketing,
decay, and spread mechanics, which are independent of the budget value.
"""
import math
import os
import random
import tempfile
import unittest

from erd_queue import (
    ERDQueue, cost_size_bucket, _COST_MODEL_TAU, _COST_MODEL_BUCKET_BASE,
)

POLICY = "erd_all"
BUDGET = 5


def _lognormal_samples(mu, sigma, count, seed):
    """`count` samples of exp(N(mu, sigma)) — geometric mean is exp(mu)."""
    rng = random.Random(seed)
    return [math.exp(rng.gauss(mu, sigma)) for _ in range(count)]


class _TmpQueue(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.q = ERDQueue(os.path.join(self._tmp.name, "q.sqlite3"))
        self.addCleanup(self.q.close)


class TestSizeBucketing(_TmpQueue):
    def test_bucket_is_monotonic_nondecreasing(self):
        prev = -1
        for n in range(1, 500):
            b = cost_size_bucket(n)
            self.assertGreaterEqual(b, prev)
            prev = b

    def test_nearby_sizes_share_a_bucket(self):
        # Two sizes within one geometric step land in the same bucket.
        self.assertEqual(cost_size_bucket(100), cost_size_bucket(105))

    def test_distant_sizes_separate(self):
        # A size > BASE * the other is at least one bucket away.
        self.assertNotEqual(
            cost_size_bucket(100),
            cost_size_bucket(int(100 * _COST_MODEL_BUCKET_BASE) + 5))

    def test_unseen_size_borrows_from_bucket_mate(self):
        # A sample at size 100 warms its whole bucket: an unseen 105 (same
        # bucket) reads warm with the same estimate; a distant 130 stays cold.
        self.assertIsNone(self.q.get_cost_typical(POLICY, 100, budget=BUDGET))
        self.q.update_cost_model(POLICY, 100, 4000, budget=BUDGET)
        self.assertIsNotNone(self.q.get_cost_typical(POLICY, 105, budget=BUDGET))
        self.assertEqual(self.q.get_cost_typical(POLICY, 100, budget=BUDGET),
                         self.q.get_cost_typical(POLICY, 105, budget=BUDGET))
        # 130 is a different bucket — still cold (the whole point of #1: a sample
        # at one exact size does NOT make every other size warm, only its bucket).
        if cost_size_bucket(130) != cost_size_bucket(100):
            self.assertIsNone(self.q.get_cost_typical(POLICY, 130, budget=BUDGET))


class TestGeometricMeanConverges(_TmpQueue):
    def test_converges_to_geometric_mean(self):
        # 2000 log-normal samples; the estimate should sit at exp(mu).
        mu, sigma = math.log(5000), 1.2
        now = 1_000_000
        for x in _lognormal_samples(mu, sigma, 2000, seed=1):
            # Fixed `now` ⇒ no decay between samples ⇒ pure accumulation.
            self.q.update_cost_model(POLICY, 137, x, now=now, budget=BUDGET)
        typical = self.q.get_cost_typical(POLICY, 137, budget=BUDGET)
        self.assertAlmostEqual(math.log(typical), mu, delta=0.08)

    def test_robust_to_single_tarpit(self):
        # 200 samples of 1000, then one 10-million-node tarpit.  Geometric mean
        # barely moves; the arithmetic mean would jump by ~500x.
        now = 1_000_000
        for _ in range(200):
            self.q.update_cost_model(POLICY, 50, 1000, now=now, budget=BUDGET)
        self.q.update_cost_model(POLICY, 50, 10_000_000, now=now, budget=BUDGET)
        typical = self.q.get_cost_typical(POLICY, 50, budget=BUDGET)
        arithmetic = (200 * 1000 + 10_000_000) / 201
        self.assertLess(typical, 1100)          # stayed near 1000
        self.assertGreater(arithmetic, 50_000)  # the mean did not


class TestTimeDecay(_TmpQueue):
    def test_old_regime_fades(self):
        # An expensive old regime, then a long gap and a cheap new regime: the
        # estimate tracks the new regime once the old samples have decayed.
        t0 = 1_000_000
        for _ in range(20):
            self.q.update_cost_model(POLICY, 80, 1_000_000, now=t0, budget=BUDGET)
        # Jump ~10 half-lives forward, then stream the cheap regime.
        t1 = t0 + int(10 * _COST_MODEL_TAU)
        for i in range(50):
            self.q.update_cost_model(POLICY, 80, 100, now=t1 + i, budget=BUDGET)
        typical = self.q.get_cost_typical(POLICY, 80, budget=BUDGET)
        self.assertLess(typical, 200)   # old 1e6 samples have faded out

    def test_no_decay_within_same_instant(self):
        # Samples sharing one timestamp accumulate without decay (elapsed 0).
        now = 5_000
        for _ in range(10):
            self.q.update_cost_model(POLICY, 80, 500, now=now, budget=BUDGET)
        self.assertAlmostEqual(
            self.q.get_cost_typical(POLICY, 80, budget=BUDGET), 500, delta=2)


class TestSpread(_TmpQueue):
    def test_sigma_recovered_from_second_moment(self):
        # Two values a, b: sigma of ln is |ln a - ln b| / 2.
        now = 1_000_000
        a, b = 100, 10_000
        self.q.update_cost_model(POLICY, 60, a, now=now, budget=BUDGET)
        self.q.update_cost_model(POLICY, 60, b, now=now, budget=BUDGET)
        expected = abs(math.log(a) - math.log(b)) / 2
        self.assertAlmostEqual(self.q.get_cost_spread(POLICY, 60, budget=BUDGET),
                               expected, places=6)

    def test_zero_spread_for_identical_samples(self):
        now = 1_000_000
        for _ in range(5):
            self.q.update_cost_model(POLICY, 60, 777, now=now, budget=BUDGET)
        self.assertAlmostEqual(
            self.q.get_cost_spread(POLICY, 60, budget=BUDGET), 0.0, places=6)

    def test_spread_matches_lognormal_sigma(self):
        mu, sigma = math.log(3000), 0.9
        now = 1_000_000
        for x in _lognormal_samples(mu, sigma, 3000, seed=7):
            self.q.update_cost_model(POLICY, 200, x, now=now, budget=BUDGET)
        self.assertAlmostEqual(self.q.get_cost_spread(POLICY, 200, budget=BUDGET),
                               sigma, delta=0.06)


class TestBatchEqualsIndividual(_TmpQueue):
    def test_logsums_batch_matches_one_at_a_time(self):
        # The inline-buffer flush folds (Σln, Σln², count) in one call; it must
        # match folding each sample individually (no lossy collapse).
        samples = _lognormal_samples(math.log(2500), 1.1, 64, seed=3)
        now = 1_000_000
        for x in samples:
            self.q.update_cost_model("indiv", 90, x, now=now, budget=BUDGET)
        log_sum = sum(math.log(x) for x in samples)
        log_sq_sum = sum(math.log(x) ** 2 for x in samples)
        self.q.update_cost_model_logsums(
            "batch", 90, log_sum, log_sq_sum, float(len(samples)), now=now,
            budget=BUDGET)
        self.assertAlmostEqual(
            self.q.get_cost_typical("indiv", 90, budget=BUDGET),
            self.q.get_cost_typical("batch", 90, budget=BUDGET), places=6)
        self.assertAlmostEqual(
            self.q.get_cost_spread("indiv", 90, budget=BUDGET),
            self.q.get_cost_spread("batch", 90, budget=BUDGET), places=6)

    def test_small_geometric_mean_not_floored_to_one(self):
        # Regression for the old int(exp(avg_log)) collapse: a batch whose
        # geometric mean is below 2 nodes used to round to 1 → log(1)=0 → the
        # batch contributed weight with zero log-sum, dragging the estimate to 1.
        # The logsums path preserves the sub-2 geometric mean.
        now = 1_000_000
        samples = [1, 1, 1, 3]   # geometric mean ~1.31
        log_sum = sum(math.log(x) for x in samples)
        log_sq_sum = sum(math.log(x) ** 2 for x in samples)
        self.q.update_cost_model_logsums(
            POLICY, 12, log_sum, log_sq_sum, float(len(samples)), now=now,
            budget=BUDGET)
        typical = self.q.get_cost_typical(POLICY, 12, budget=BUDGET)
        self.assertGreater(typical, 1.0)
        self.assertAlmostEqual(typical, math.exp(log_sum / len(samples)),
                               places=6)


class TestPolicyAndBucketIsolation(_TmpQueue):
    def test_policy_keeps_models_separate(self):
        now = 1_000_000
        self.q.update_cost_model("erd_all", 100, 200, now=now, budget=BUDGET)
        self.q.update_cost_model("erd_answers", 100, 9000, now=now, budget=BUDGET)
        self.assertAlmostEqual(
            self.q.get_cost_typical("erd_all", 100, budget=BUDGET), 200, delta=1)
        self.assertAlmostEqual(
            self.q.get_cost_typical("erd_answers", 100, budget=BUDGET), 9000,
            delta=1)


if __name__ == "__main__":
    unittest.main()
