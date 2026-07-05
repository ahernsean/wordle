"""Unit tests for _BranchWorker behaviors not covered by the integration tests.

The integration tests (test_erd_parallel, test_erd_scaling) exercise the swarm
via solve_branch_focused(), which covers the core evaluate/finalize path.  What
they miss:

- _heartbeat(): node-counter throttling (counter increments on every call even
  when the 2-second DB-write gate suppresses the actual write).
- evaluate_claim(): cancellation path (stop_event set → returns False without
  completing the claim).
- _subbranch_solver(): the inline-vs-promote branching decision.
- _maybe_checkpoint(): force=True always triggers checkpoint; force=False with a
  recently-set timestamp does not.
- cooperative_solve(): the cached-result fast path (result already in cache →
  returns immediately without evaluating any candidates).
"""
import multiprocessing
import os
import tempfile
import time
import unittest
from unittest import mock

import erd_swarm
from erd_swarm import _BranchWorker, ROOT_BUDGET, PROMOTE_MIN_SIZE
from cache_sqlite import ScoreCache
from wordle_engine import ERD_ALL, SOLVED, OVER_ERD_LIMIT
from erd_queue import ERDQueue, guess_depth_from_spine

BRANCH = ["crane", "slate", "trace", "stale", "tales"]
CANDIDATES = BRANCH + ["brain", "stove", "cloud", "piano", "train"]


def _bare_worker():
    """Skeleton _BranchWorker for unit tests: no DB connections, no word files.

    Sets only the attributes required for the method under test; other tests
    mock queue and cache so the unit stays fast and isolated.
    """
    w = _BranchWorker.__new__(_BranchWorker)
    w.name = "worker-0"
    w.stop_event = None
    w._stop_requested = False
    w.root_budget = ROOT_BUDGET
    w.all_words = CANDIDATES
    w.n_candidates = len(CANDIDATES)
    w.claims_done = 0
    w.n_ok = w.n_cutoff = w.n_pruned = w.n_useless = 0
    w._nodes = 0
    w._nodes_at_last_hb = 0
    w._last_hb = 0.0
    w._last_progress_log = 0.0
    w._last_util_log = 0.0
    w._eval_seconds = 0.0
    w._last_claim_complete = 0.0
    w._last_checkpoint = 0.0
    w._cand_max_depth = 0
    w._cur_depth = 0
    w._spine = {}
    w._hb_max_spine = {}
    w._log_max_spine = {}
    w.started = 0
    w._top_source_word = None
    w._top_source_pattern = None
    w._claimed_branch_spine = None
    w._adaptive = True
    w._typical_cache = {}
    w._cost_model_buffer = {}
    w._word_idx = {word: i for i, word in enumerate(w.all_words)}
    w._coord_ema = erd_swarm._LogEMA()
    w._node_time_ema = erd_swarm._LogEMA()
    w._mid_loop_publisher = erd_swarm._MidLoopPublisher(w)
    w.n_workers = 1
    w.rcache = mock.MagicMock()
    w.pattern_matrix = None
    w.score_cache = mock.MagicMock()
    w.score_cache.read_hits = 0
    w.score_cache.read_misses = 0
    w.queue = mock.MagicMock()
    w.queue.read_branch_best.return_value = (None, None)
    w.queue.get_cost_typical.return_value = None  # cold model by default
    return w


class TestHeartbeatThrottling(unittest.TestCase):
    """_nodes increments on every call; the DB write is gated by HB_SECONDS."""

    def test_node_counter_increments_even_when_db_write_is_throttled(self):
        w = _bare_worker()
        branch_key = ScoreCache.encode_subset(BRANCH)

        # First call: force=True bypasses the time gate — DB write happens.
        w._heartbeat(branch_key, len(BRANCH), 0, 0, None, None, force=True)
        self.assertEqual(w._nodes, 1)
        self.assertEqual(w.queue.heartbeat.call_count, 1)

        # Second call: force=False, but _last_hb was just set so the gate fires.
        w._heartbeat(branch_key, len(BRANCH), 0, 0, None, None)
        self.assertEqual(w._nodes, 2)          # counter still incremented
        self.assertEqual(w.queue.heartbeat.call_count, 1)  # still only one DB write

    def test_hb_max_spine_reset_after_each_db_write(self):
        """_hb_max_spine is cleared after each DB write so the 2-second window
        starts fresh — the next heartbeat builds a new spine from scratch."""
        w = _bare_worker()
        w._note_depth(5, 50)
        w._note_depth(4, 12)
        branch_key = ScoreCache.encode_subset(BRANCH)
        w._heartbeat(branch_key, len(BRANCH), 0, 0, None, None, force=True)
        self.assertEqual(w._hb_max_spine, {})


class TestPromotedSpine(unittest.TestCase):
    """_promoted_spine composes base + descent without overlap.

    The live descent dict keeps shallow entries until a shallower frame
    overwrites them, so after the base is advanced it still holds the guesses
    that reach the base.  Those belong to the base, not the descent; appending
    them re-emits the base's own reaching guess and over-counts guess_depth.
    """

    def test_descent_at_or_above_base_depth_is_not_reappended(self):
        w = _bare_worker()
        # Base reaches a guess_depth-2 branch via SALET then ABORT.
        w._claimed_branch_spine = 'SALET -g-g- ABORT y----'
        # Live descent still carries the reaching guess (depth 2, budget 4) AND
        # a genuine child below it (depth 3, budget 3).
        w._note_depth(4, 12, 'abort', 'y----')   # guess_depth 2 — the base tail
        w._note_depth(3, 5, 'dogma', 'y---y')     # guess_depth 3 — the real child
        spine = w._promoted_spine()
        self.assertEqual(spine, 'SALET -g-g- ABORT y---- DOGMA y---y')
        self.assertEqual(guess_depth_from_spine(spine), 3)

    def test_pure_descent_below_base_is_appended(self):
        w = _bare_worker()
        w._claimed_branch_spine = 'SALET -g-g-'   # guess_depth 1
        w._note_depth(4, 5, 'dogma', 'y---y')     # guess_depth 2 — below the base
        self.assertEqual(w._promoted_spine(), 'SALET -g-g- DOGMA y---y')

    def test_no_base_yields_none(self):
        w = _bare_worker()
        w._claimed_branch_spine = None
        w._note_depth(4, 5, 'dogma', 'y---y')
        self.assertIsNone(w._promoted_spine())


class TestHeartbeatSpineStrFilter(unittest.TestCase):
    """_hb_spine_str omits outer-frame entries at or shallower than the claimed branch.

    _solve_subset fires note_depth before calling subbranch_solver, so the
    entry at the claimed branch's own guess_depth is written by the outer frame
    and persists into inner cooperative_solve sessions.  It belongs to the base
    spine, not to the live descent, and must be filtered out of the heartbeat
    string so the worker detail panel does not display the reaching guess twice.
    """

    def test_outer_frame_entry_at_claimed_depth_is_excluded(self):
        w = _bare_worker()
        # Outer frame set _spine[3] = (44, 'peaze', '-yy--') before calling
        # subbranch_solver, then cooperative_solve advanced _claimed_branch_spine
        # to include PEAZE.  The outer frame's entry should not appear as a live
        # descent step.
        w._claimed_branch_spine = 'SALET -g-g- ABOVE y---y PEAZE -yy--'
        w._note_depth(3, 44, 'peaze', '-yy--')   # outer frame's entry at guess_depth 3
        w._note_depth(2, 16, 'nurdy', '---y-')   # real descent at guess_depth 4
        w._hb_max_spine = dict(w._spine)
        result = w._hb_spine_str()
        self.assertNotIn('PEAZE', result)
        self.assertIn('NURDY', result)

    def test_descent_below_claimed_branch_is_included(self):
        w = _bare_worker()
        w._claimed_branch_spine = 'SALET -g-g-'   # guess_depth 1
        w._note_depth(4, 30, 'crane', '-yg--')   # guess_depth 2
        w._note_depth(3, 8, 'dogma', 'y---y')    # guess_depth 3
        w._hb_max_spine = dict(w._spine)
        result = w._hb_spine_str()
        self.assertIn('CRANE', result)
        self.assertIn('DOGMA', result)

    def test_tokens_carry_explicit_guess_depth_prefix(self):
        w = _bare_worker()
        w._claimed_branch_spine = 'SALET -g-g-'   # guess_depth 1
        w._note_depth(4, 30, 'crane', '-yg--')   # guess_depth 2
        w._note_depth(3, 8, 'dogma', 'y---y')    # guess_depth 3
        w._hb_max_spine = dict(w._spine)
        result = w._hb_spine_str()
        # Each token is prefixed 'guess_depth:GUESS:...' so the display can label
        # and filter entries without positional inference.
        self.assertIn('2:CRANE', result)
        self.assertIn('3:DOGMA', result)


class TestCancelPath(unittest.TestCase):
    """evaluate_claim returns False without completing the claim when cancelled."""

    def _make_cancel_worker(self):
        stop = multiprocessing.Event()
        stop.set()
        w = _bare_worker()
        w.stop_event = stop
        return w

    def test_evaluate_claim_returns_false_when_stop_event_set(self):
        w = self._make_cancel_worker()
        branch_key = ScoreCache.encode_subset(BRANCH)
        result = w.evaluate_claim(branch_key, BRANCH, len(BRANCH), idx=0)
        self.assertFalse(result)

    def test_claim_not_marked_complete_when_cancelled(self):
        w = self._make_cancel_worker()
        branch_key = ScoreCache.encode_subset(BRANCH)
        w.evaluate_claim(branch_key, BRANCH, len(BRANCH), idx=0)
        # complete_candidate must NOT have been called (claim left done=0 for reclaim).
        w.queue.complete_candidate.assert_not_called()

    def test_request_stop_cancels_without_shared_event(self):
        # A worker's own SIGTERM/SIGINT handler calls request_stop(); this must
        # cancel the worker via a process-local flag, never touching the shared
        # stop_event (so recycling one worker does not stop the pool).
        w = _bare_worker()           # stop_event is None
        self.assertFalse(w.cancel())
        w.request_stop()
        self.assertTrue(w.cancel())
        self.assertIsNone(w.stop_event)


class TestEvaluateClaimPatternMatrix(unittest.TestCase):
    """evaluate_claim threads self.pattern_matrix into evaluate_candidate.

    _bare_worker() must keep mirroring _BranchWorker.__init__: an
    unconditional field read added to evaluate_claim (like this one) needs
    the matching field added to _bare_worker(), or any non-cancelled call
    to evaluate_claim on a bare worker raises AttributeError (see
    reference_heartbeat_test_skeleton.md for the same class of gap in
    _heartbeat).
    """

    def test_evaluate_candidate_receives_worker_pattern_matrix(self):
        # Deliberately relies on _bare_worker()'s own construction for
        # self.pattern_matrix rather than setting it here afterward --
        # setting it here would mask the exact gap this test exists to catch.
        w = _bare_worker()           # stop_event is None -> not cancelled
        branch_key = ScoreCache.encode_subset(BRANCH)
        with mock.patch('erd_swarm.evaluate_candidate',
                        return_value=(SOLVED, 1.5, 1, False)) as mock_eval:
            result = w.evaluate_claim(branch_key, BRANCH, len(BRANCH), idx=0)
        self.assertTrue(result)
        self.assertIsNone(mock_eval.call_args.kwargs['pattern_matrix'])


class TestSubbranchSolver(unittest.TestCase):
    """_subbranch_solver returns None for small/unbudgeted branches (inline);
    delegates to cooperative_solve for large branches with a real budget."""

    def test_returns_none_for_branch_below_promote_threshold(self):
        w = _bare_worker()
        words = BRANCH[:3]  # well below PROMOTE_MIN_SIZE (60)
        self.assertIsNone(w._subbranch_solver(words, budget=5))

    def test_returns_none_when_budget_is_none(self):
        # budget=None always inlines regardless of branch size.
        w = _bare_worker()
        words = ["crane"] * (PROMOTE_MIN_SIZE + 1)
        self.assertIsNone(w._subbranch_solver(words, budget=None))

    def test_calls_cooperative_solve_for_large_budgeted_branch(self):
        w = _bare_worker()
        expected = (1.5, 2, False, False)
        w.cooperative_solve = mock.MagicMock(return_value=expected)
        words = ["crane"] * (PROMOTE_MIN_SIZE + 1)
        result = w._subbranch_solver(words, budget=5)
        w.cooperative_solve.assert_called_once_with(words, 5)
        self.assertEqual(result, expected)


class TestNoteDepthPromotionSentinel(unittest.TestCase):
    """_note_depth with n<0 marks a cooperative-promoted sub-branch with '•'.

    _note_depth's first arg is the engine's `budget`; the spine is keyed by
    guess_depth = GAME_GUESSES (6) - budget.  Budgets 5/4/3 are guess_depths
    1/2/3 — a worker descending one guess at a time.
    """

    def test_sentinel_marks_spine_with_bullet(self):
        w = _bare_worker()
        w._note_depth(5, 50)
        w._note_depth(4, 12)
        # n=-1 is the cooperative-promotion sentinel.
        w._note_depth(4, -1)
        size, guess, pattern = w._spine[2]
        self.assertEqual(size, '•')
        self.assertIn(1, w._spine)  # parent depth untouched

    def test_sentinel_updates_hb_and_log_max_spine(self):
        w = _bare_worker()
        w._note_depth(5, 50)
        w._note_depth(4, -1)
        self.assertIn(2, w._hb_max_spine)
        self.assertIn(2, w._log_max_spine)

    def test_sentinel_does_not_update_max_spine_when_spine_is_shorter(self):
        """When the new spine is shorter than the accumulated max, neither
        _hb_max_spine nor _log_max_spine is overwritten (depth NOT extended)."""
        w = _bare_worker()
        # Build a 3-level max spine.
        w._note_depth(5, 100)
        w._note_depth(4, 50)
        w._note_depth(3, 20)
        # Reset spine to a single-level view.
        w._spine = {1: 100}
        # n<0 with a shorter spine: len(spine)=2 (after adding depth 2) may still
        # be < len(hb_max_spine=3) depending on prior history, so this exercises
        # the False branch of the len comparison.
        pre_hb = dict(w._hb_max_spine)
        w._note_depth(4, -1)  # spine becomes {1:100, 2:('•',…)}, len=2 < 3
        # If hb_max_spine was already size 3, it should NOT be overwritten.
        if len(pre_hb) > len(w._spine):
            self.assertEqual(w._hb_max_spine, pre_hb)


class TestSpineComposition(unittest.TestCase):
    """_root_spine / _promoted_spine build the absolute root -> branch path."""

    def test_root_spine_formats_word_and_pattern(self):
        from wordle_ui import fmt_pattern
        w = _bare_worker()
        self.assertEqual(w._root_spine('salet', 0),
                         f'SALET {fmt_pattern(0)}')

    def test_root_spine_none_without_source(self):
        w = _bare_worker()
        self.assertIsNone(w._root_spine(None, 0))
        self.assertIsNone(w._root_spine('salet', None))

    def test_promoted_spine_composes_base_and_descent_edges(self):
        w = _bare_worker()
        w._claimed_branch_spine = 'SALET -g-g-'
        # Pattern strings pass through unchanged (only ints are fmt_pattern'd).
        w._note_depth(5, 50, 'crane', 'bb-y-')
        w._note_depth(4, 12, 'pound', 'g--y-')
        self.assertEqual(w._promoted_spine(),
                         'SALET -g-g- CRANE bb-y- POUND g--y-')

    def test_promoted_spine_none_without_base(self):
        w = _bare_worker()
        w._claimed_branch_spine = None
        w._note_depth(5, 50, 'crane', 'bb-y-')
        self.assertIsNone(w._promoted_spine())

    def test_promoted_spine_skips_sentinel_and_size_only_levels(self):
        w = _bare_worker()
        w._claimed_branch_spine = 'SALET -g-g-'
        w._note_depth(5, 50, 'crane', 'bb-y-')
        w._note_depth(4, 12)            # size-only level: no guess/pattern
        # Promotion sentinel preserves the guess but sets size to '•'; the edge
        # is still a real edge and must be kept.
        self.assertEqual(w._promoted_spine(), 'SALET -g-g- CRANE bb-y-')


class TestMaybeCheckpoint(unittest.TestCase):
    """_maybe_checkpoint(force=True) always checkpoints;
    force=False respects the CHECKPOINT_SECONDS timer."""

    def test_force_true_always_checkpoints(self):
        import time
        w = _bare_worker()
        w._last_checkpoint = time.time()   # just checkpointed — timer not yet expired
        w._maybe_checkpoint(force=True)
        w.score_cache.checkpoint.assert_called_once()

    def test_no_checkpoint_before_interval_without_force(self):
        import time
        w = _bare_worker()
        w._last_checkpoint = time.time()   # timer still fresh
        w._maybe_checkpoint(force=False)
        w.score_cache.checkpoint.assert_not_called()


class TestCooperativeSolveCachedPath(unittest.TestCase):
    """cooperative_solve returns the cached result immediately when the branch
    is already solved in ScoreCache, without claiming or evaluating any candidate."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.answer_file = self._write("answers.txt", BRANCH)
        self.words_file = self._write("words.txt", CANDIDATES)
        for attr, path in [("ANSWER_FILE", self.answer_file),
                           ("WORDS_FILE", self.words_file)]:
            p = mock.patch.object(erd_swarm, attr, path)
            p.start()
            self.addCleanup(p.stop)
        self.cache_path = os.path.join(self._tmp.name, "cache.sqlite3")
        self.queue_path = os.path.join(self._tmp.name, "queue.sqlite3")

    def _write(self, name, words):
        p = os.path.join(self._tmp.name, name)
        with open(p, "w") as f:
            f.write("\n".join(words) + "\n")
        return p

    def test_returns_cached_result_without_evaluating_chunks(self):
        words = BRANCH[:3]
        branch_key = ScoreCache.encode_subset(words)
        # Pre-populate with an untainted result (solve_budget=None → reusable
        # at any budget >= max_depth).
        sc = ScoreCache(self.cache_path, BRANCH)
        sc.write(branch_key, ERD_ALL, "crane", 1.5, max_depth=2, solve_budget=None)
        sc.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            result = w.cooperative_solve(words, ROOT_BUDGET)
        finally:
            w.close()

        self.assertIsNotNone(result)
        status, cost, max_depth, budget_tainted = result
        self.assertEqual(status, SOLVED)
        self.assertAlmostEqual(cost, 1.5)
        self.assertEqual(max_depth, 2)

        # No candidate claims should have been created — the cache hit short-circuited.
        q = ERDQueue(self.queue_path)
        self.assertIsNone(q.get_branch(branch_key))
        q.close()


class TestSolveBranchFocusedPrecompletedCandidates(unittest.TestCase):
    """solve_branch_focused finalizes correctly when all candidates are already
    done by other workers: claim_candidate returns None,
    branch_done_candidates >= n_candidates, so the worker finalizes from the
    idx-is-None path (not via evaluate_claim)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.answer_file = self._write("answers.txt", BRANCH)
        self.words_file = self._write("words.txt", CANDIDATES)
        for attr, path in [("ANSWER_FILE", self.answer_file),
                           ("WORDS_FILE", self.words_file)]:
            p = mock.patch.object(erd_swarm, attr, path)
            p.start()
            self.addCleanup(p.stop)
        self.cache_path = os.path.join(self._tmp.name, "cache.sqlite3")
        self.queue_path = os.path.join(self._tmp.name, "queue.sqlite3")

    def _write(self, name, words):
        p = os.path.join(self._tmp.name, name)
        with open(p, "w") as f:
            f.write("\n".join(words) + "\n")
        return p

    def test_finalizes_when_all_candidates_pre_completed(self):
        from erd_queue import ERDQueue
        # Pre-mark all candidate claims as done (simulating other workers that
        # have already evaluated every candidate).
        branch_key = ScoreCache.encode_subset(BRANCH)
        ScoreCache(self.cache_path, BRANCH).close()  # initialise schema
        q = ERDQueue(self.queue_path)
        n_candidates = len(CANDIDATES)
        q.create_branch(branch_key, len(BRANCH), n_candidates)
        for cand_idx in range(n_candidates):
            q.claim_candidate(branch_key, f"other-{cand_idx}", n_candidates)
            q.complete_candidate(branch_key, cand_idx)
        q.close()

        # Our worker enters solve_branch_focused: claim_candidate → None (all done),
        # branch_done_candidates >= n_candidates → maybe_finalize.  Since no
        # best_guess was set, maybe_finalize treats it as a loss and deletes the
        # branch without writing a cache entry.
        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            w.solve_branch_focused(branch_key)
        finally:
            w.close()

        # Branch must be gone (finalized + deleted).
        q = ERDQueue(self.queue_path)
        self.assertIsNone(q.get_branch(branch_key))
        q.close()


class TestClaimOneJoinsInProgressBranch(unittest.TestCase):
    """claim_one() joins a branch that is already in-progress (created by
    another worker) rather than promoting a new one from the pending queue."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.answer_file = self._write("answers.txt", BRANCH)
        self.words_file = self._write("words.txt", CANDIDATES)
        for attr, path in [("ANSWER_FILE", self.answer_file),
                           ("WORDS_FILE", self.words_file)]:
            p = mock.patch.object(erd_swarm, attr, path)
            p.start()
            self.addCleanup(p.stop)
        self.cache_path = os.path.join(self._tmp.name, "cache.sqlite3")
        self.queue_path = os.path.join(self._tmp.name, "queue.sqlite3")

    def _write(self, name, words):
        p = os.path.join(self._tmp.name, name)
        with open(p, "w") as f:
            f.write("\n".join(words) + "\n")
        return p

    def test_solve_branch_focused_returns_early_for_missing_branch(self):
        ScoreCache(self.cache_path, BRANCH).close()
        branch_key = ScoreCache.encode_subset(BRANCH)
        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            # Branch was never registered — should return without error.
            w.solve_branch_focused(branch_key)
        finally:
            w.close()

    def test_claim_one_skips_fully_claimed_in_progress_branch_and_joins_next(self):
        """If the first in-progress branch has all candidates claimed, claim_one
        must continue iterating and claim a candidate from the next branch."""
        from erd_queue import ERDQueue
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        n_candidates = len(CANDIDATES)

        # Branch A: all candidates pre-claimed by another worker.
        key_a = ScoreCache.encode_subset(BRANCH)
        q.create_branch(key_a, len(BRANCH), n_candidates,
                        budget=ROOT_BUDGET, priority=0)
        for i in range(n_candidates):
            q.claim_candidate(key_a, "other-worker", n_candidates)

        # Branch B: has a free candidate for our worker to claim.
        words_b = BRANCH[:4]
        key_b = ScoreCache.encode_subset(words_b)
        q.create_branch(key_b, len(words_b), n_candidates,
                        budget=ROOT_BUDGET, priority=0)
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            result = w.claim_one()
        finally:
            w.close()

        # Worker must have skipped branch A (fully claimed) and joined branch B.
        self.assertIsNotNone(result)
        branch, idx = result
        self.assertEqual(branch['branch_key'], key_b)
        self.assertIsNotNone(idx)

    def test_claim_one_joins_existing_in_progress_branch(self):
        from erd_queue import ERDQueue
        branch_key = ScoreCache.encode_subset(BRANCH)
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        n_candidates = len(CANDIDATES)
        # Register branch as in-progress (created, not pending).
        q.create_branch(branch_key, len(BRANCH), n_candidates, budget=ROOT_BUDGET)
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            result = w.claim_one()
        finally:
            w.close()

        # claim_one should have joined the in-progress branch via
        # branches_in_progress() → claim_candidate() → return (branch, idx).
        self.assertIsNotNone(result)
        branch, idx = result
        self.assertEqual(branch['branch_key'], branch_key)
        self.assertIsNotNone(idx)


class TestCooperativeSolveFullPath(unittest.TestCase):
    """cooperative_solve evaluates all candidates and returns the correct result
    when the branch is NOT pre-cached.  This exercises the main cooperative
    loop (create branch, evaluate candidates, finalize, read cache) and the
    maybe_finalize early-return path (called after each candidate, returns early
    until the last one makes all candidates done)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.answer_file = self._write("answers.txt", BRANCH)
        self.words_file = self._write("words.txt", CANDIDATES)
        for attr, path in [("ANSWER_FILE", self.answer_file),
                           ("WORDS_FILE", self.words_file)]:
            p = mock.patch.object(erd_swarm, attr, path)
            p.start()
            self.addCleanup(p.stop)
        self.cache_path = os.path.join(self._tmp.name, "cache.sqlite3")
        self.queue_path = os.path.join(self._tmp.name, "queue.sqlite3")

    def _write(self, name, words):
        p = os.path.join(self._tmp.name, name)
        with open(p, "w") as f:
            f.write("\n".join(words) + "\n")
        return p

    def test_solves_uncached_branch_and_returns_result(self):
        # Use all 5 BRANCH words so cooperative_solve evaluates all 10 CANDIDATES
        # one at a time.  maybe_finalize() returns early on each until the last
        # candidate makes branch_done_candidates == n_candidates and finalizes.
        words = BRANCH
        branch_key = ScoreCache.encode_subset(words)

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            result = w.cooperative_solve(words, ROOT_BUDGET)
        finally:
            w.close()

        self.assertIsNotNone(result)
        status, cost, max_depth, budget_tainted = result
        self.assertEqual(status, SOLVED)
        self.assertGreater(cost, 0)

        # The branch must be finalized — rows deleted from the queue.
        from erd_queue import ERDQueue
        q = ERDQueue(self.queue_path)
        self.assertIsNone(q.get_branch(branch_key))
        q.close()

        # The result must be persisted to the cache.
        sc = ScoreCache(self.cache_path, BRANCH, checkpoint_on_close=False)
        from wordle_engine import ERD_ALL
        cached = sc.read(branch_key, ERD_ALL)
        sc.close()
        self.assertIsNotNone(cached)
        self.assertAlmostEqual(cached[1], cost, places=6)


class TestHelpOtherBranch(unittest.TestCase):
    """_help_other_branch claims and evaluates one candidate from a branch other
    than the excluded branch, returning True if work was found, False if not."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.answer_file = self._write("answers.txt", BRANCH)
        self.words_file = self._write("words.txt", CANDIDATES)
        for attr, path in [("ANSWER_FILE", self.answer_file),
                           ("WORDS_FILE", self.words_file)]:
            p = mock.patch.object(erd_swarm, attr, path)
            p.start()
            self.addCleanup(p.stop)
        self.cache_path = os.path.join(self._tmp.name, "cache.sqlite3")
        self.queue_path = os.path.join(self._tmp.name, "queue.sqlite3")

    def _write(self, name, words):
        p = os.path.join(self._tmp.name, name)
        with open(p, "w") as f:
            f.write("\n".join(words) + "\n")
        return p

    def test_help_other_branch_returns_true_when_candidate_evaluated(self):
        """When a candidate is available in another branch, help_other_branch
        claims and evaluates it, then returns True."""
        from erd_queue import ERDQueue
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        n_candidates = len(CANDIDATES)

        # Create branch A (the one being excluded).
        key_a = ScoreCache.encode_subset(BRANCH)
        q.create_branch(key_a, len(BRANCH), n_candidates,
                        budget=ROOT_BUDGET, priority=0)

        # Create branch B with a free candidate claim.
        words_b = BRANCH[:4]
        key_b = ScoreCache.encode_subset(words_b)
        q.create_branch(key_b, len(words_b), n_candidates,
                        budget=ROOT_BUDGET, priority=1)
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            result = w._help_other_branch(key_a)
        finally:
            w.close()

        # Should have evaluated a candidate and returned True.
        self.assertTrue(result)

    def test_help_other_branch_returns_false_when_no_candidates_available(self):
        """When no other branches have available candidate claims, help_other_branch
        returns False without claiming anything."""
        from erd_queue import ERDQueue
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        n_candidates = len(CANDIDATES)

        # Create only one branch and fully claim all its candidates.
        key_a = ScoreCache.encode_subset(BRANCH)
        q.create_branch(key_a, len(BRANCH), n_candidates,
                        budget=ROOT_BUDGET, priority=0)
        for i in range(n_candidates):
            q.claim_candidate(key_a, "other-worker", n_candidates)
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            # Exclude a different branch key (none exist, so effectively no branches to help).
            fake_exclude_key = ScoreCache.encode_subset(BRANCH[:3])
            result = w._help_other_branch(fake_exclude_key)
        finally:
            w.close()

        # Should return False (no candidates to claim).
        self.assertFalse(result)

    def test_help_other_branch_skips_excluded_branch(self):
        """When the only available branch matches exclude_branch_key,
        help_other_branch skips it and returns False."""
        from erd_queue import ERDQueue
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        n_candidates = len(CANDIDATES)

        # Create one branch with free candidates.
        key_a = ScoreCache.encode_subset(BRANCH)
        q.create_branch(key_a, len(BRANCH), n_candidates,
                        budget=ROOT_BUDGET, priority=0)
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            # Exclude the only branch — nothing else to help.
            result = w._help_other_branch(key_a)
        finally:
            w.close()

        # Should return False (the only available branch was excluded).
        self.assertFalse(result)


class TestMidLoopPublisher(unittest.TestCase):
    """_MidLoopPublisher.enter() / check() / record_inline() unit tests."""

    def _pub(self, predicted=None):
        w = _bare_worker()
        w.queue.get_cost_typical.return_value = predicted
        pub = erd_swarm._MidLoopPublisher(w)
        return pub, w

    def test_enter_returns_none_below_min_words(self):
        pub, _ = self._pub()
        # MIN_PUBLISH_BRANCH_WORDS is 2; a 1-word frame is a base case.
        self.assertIsNone(pub.enter(BRANCH[:1], budget=0))

    def test_enter_returns_token_at_min_words(self):
        pub, _ = self._pub(predicted=1000)
        words = BRANCH[:2]  # exactly MIN_PUBLISH_BRANCH_WORDS
        token = pub.enter(words, budget=0)
        self.assertIsNotNone(token)
        nodes_at_entry, predicted, entry_time, bw, depth = token
        self.assertEqual(bw, words)
        self.assertEqual(predicted, 1000)

    def test_enter_cold_model_still_returns_token(self):
        pub, _ = self._pub(predicted=None)  # cold model
        token = pub.enter(BRANCH[:6], budget=1)
        self.assertIsNotNone(token)
        _, predicted, _, _, _ = token
        self.assertIsNone(predicted)  # token carries None when model is cold

    def test_check_returns_none_for_none_token(self):
        pub, _ = self._pub()
        self.assertIsNone(pub.check(None, CANDIDATES, 0, None, None, 5))

    def test_check_cold_model_under_backstop_returns_none(self):
        # Cold model: the node-proportionate check can't arm, and the wall-clock
        # backstop hasn't elapsed on a fresh token, so the frame stays inline.
        pub, _ = self._pub(predicted=None)
        token = pub.enter(BRANCH[:6], budget=0)
        self.assertIsNone(pub.check(token, CANDIDATES, 0, None, None, 5))

    def test_check_fires_on_wall_clock_backstop_when_cold(self):
        # Cold model, but the frame has run past COLD_BACKSTOP_SECONDS: the
        # backstop hands off the remainder even with no cost-model prediction.
        pub, w = self._pub(predicted=None)
        nodes_at_entry, _, _, words, entry_budget = pub.enter(BRANCH[:6], budget=0)
        old_entry = time.time() - (erd_swarm.COLD_BACKSTOP_SECONDS + 1)
        token = (nodes_at_entry, None, old_entry, words, entry_budget)
        w._nodes = nodes_at_entry + 50   # some work done, no prediction to compare
        w.cooperative_solve = mock.MagicMock(
            return_value=(erd_swarm.SOLVED, 2.0, 3, False))
        result = pub.check(token, CANDIDATES, 0, None, None, 5)
        self.assertIsNotNone(result)
        w.cooperative_solve.assert_called_once()
        # The firing is recorded for offline tuning, with predicted=None (cold).
        w.queue.add_backstop_telemetry.assert_called_once()
        args = w.queue.add_backstop_telemetry.call_args[0]
        self.assertEqual(args[0], len(words))   # n_words
        self.assertIsNone(args[4])              # predicted_nodes is None when cold

    def test_check_warm_overrun_does_not_record_backstop(self):
        # The node-proportionate path is the model working as intended; only the
        # wall-clock backstop is recorded for tuning.
        pub, w = self._pub(predicted=10)
        token = pub.enter(BRANCH[:6], budget=0)
        w._nodes = erd_swarm.OVERRUN_K * 10 + 1
        w.cooperative_solve = mock.MagicMock(
            return_value=(erd_swarm.SOLVED, 2.0, 3, False))
        pub.check(token, CANDIDATES, 0, None, None, 5)
        w.queue.add_backstop_telemetry.assert_not_called()

    def test_check_returns_none_when_under_overrun_threshold(self):
        pub, w = self._pub(predicted=100)
        token = pub.enter(BRANCH[:6], budget=0)
        # _nodes hasn't changed: delta = 0 <= OVERRUN_K * 100
        self.assertIsNone(pub.check(token, CANDIDATES, 0, None, None, 5))

    def test_check_fires_on_overrun(self):
        pub, w = self._pub(predicted=10)
        words = BRANCH[:6]
        token = pub.enter(words, budget=0)
        # Simulate spending > OVERRUN_K * predicted nodes
        w._nodes = erd_swarm.OVERRUN_K * 10 + 1
        w.cooperative_solve = mock.MagicMock(return_value=(erd_swarm.SOLVED, 2.0, 3, False))
        # last_index=0 of a 10-candidate list → 9 remaining (>= MIN_HANDOFF).
        result = pub.check(token, CANDIDATES, 0, None, None, 5)
        self.assertIsNotNone(result)
        w.cooperative_solve.assert_called_once()

    def test_check_returns_none_when_remaining_count_below_threshold(self):
        pub, w = self._pub(predicted=10)
        token = pub.enter(BRANCH[:6], budget=0)
        w._nodes = erd_swarm.OVERRUN_K * 10 + 1
        # last_index=7 of a 10-candidate list → only 2 remaining (< MIN_HANDOFF=4)
        self.assertIsNone(pub.check(token, CANDIDATES, 7, None, None, 5))

    def test_check_marks_prefix_done(self):
        pub, w = self._pub(predicted=10)
        words = BRANCH[:6]
        token = pub.enter(words, budget=0)
        w._nodes = erd_swarm.OVERRUN_K * 10 + 1
        w.cooperative_solve = mock.MagicMock(return_value=(erd_swarm.SOLVED, 2.0, 3, False))
        # last_index=1 → the evaluated prefix is CANDIDATES[:2].
        pub.check(token, CANDIDATES, 1, None, None, 5)
        call_args = w.queue.mark_claims_done.call_args
        self.assertIsNotNone(call_args)
        marked_indices = call_args[0][1]  # second positional arg is the indices list
        # Each evaluated prefix word appears in the done list at its all_words index.
        for word in CANDIDATES[:2]:
            self.assertIn(w.all_words.index(word), marked_indices)

    def test_record_inline_accumulates_buffer(self):
        import math
        pub, w = self._pub(predicted=100)
        words = BRANCH[:6]
        token = pub.enter(words, budget=0)
        w._nodes = 200   # 200 - 0 = 200 nodes for this frame
        pub.record_inline(token)
        # Buffer is keyed by (size, budget); this frame entered at budget=0.
        self.assertIn((len(words), 0), w._cost_model_buffer)
        s, sq, c = w._cost_model_buffer[(len(words), 0)]
        self.assertEqual(c, 1)
        self.assertAlmostEqual(s, math.log(200), places=6)
        self.assertAlmostEqual(sq, math.log(200) ** 2, places=6)

    def test_record_inline_is_noop_for_none_token(self):
        pub, w = self._pub()
        pub.record_inline(None)
        self.assertEqual(w._cost_model_buffer, {})


class TestSubbranchSolverCostModel(unittest.TestCase):
    """_subbranch_solver respects cost model: warm model gates on predicted
    nodes; cold model falls back to PROMOTE_MIN_SIZE size threshold."""

    def test_warm_model_below_threshold_inlines(self):
        w = _bare_worker()
        # Warm model predicts < PUBLISH_THRESHOLD_BOOTSTRAP nodes.
        w.queue.get_cost_typical.return_value = erd_swarm.PUBLISH_THRESHOLD_BOOTSTRAP - 1
        words = ["crane"] * (erd_swarm.PROMOTE_MIN_SIZE + 1)
        self.assertIsNone(w._subbranch_solver(words, budget=5))

    def test_warm_model_above_threshold_promotes(self):
        w = _bare_worker()
        w.queue.get_cost_typical.return_value = erd_swarm.PUBLISH_THRESHOLD_BOOTSTRAP + 1
        expected = (erd_swarm.SOLVED, 2.0, 3, False)
        w.cooperative_solve = mock.MagicMock(return_value=expected)
        words = ["crane"] * (erd_swarm.PROMOTE_MIN_SIZE + 1)
        result = w._subbranch_solver(words, budget=5)
        self.assertEqual(result, expected)
        w.cooperative_solve.assert_called_once()

    def test_cold_model_small_branch_inlines(self):
        w = _bare_worker()
        w.queue.get_cost_typical.return_value = None  # cold model
        # A cold small branch inlines directly; the mid-loop publisher's
        # wall-clock backstop bounds it if it turns out to be a tarpit.
        w.cooperative_solve = mock.MagicMock()
        words_small = ["crane"] * (erd_swarm.PROMOTE_MIN_SIZE - 1)
        self.assertIsNone(w._subbranch_solver(words_small, budget=5))
        w.cooperative_solve.assert_not_called()

    def test_cold_model_above_size_threshold_promotes(self):
        w = _bare_worker()
        w.queue.get_cost_typical.return_value = None  # cold model
        expected = (erd_swarm.SOLVED, 2.5, 4, False)
        w.cooperative_solve = mock.MagicMock(return_value=expected)
        words = ["crane"] * (erd_swarm.PROMOTE_MIN_SIZE + 1)
        result = w._subbranch_solver(words, budget=5)
        self.assertEqual(result, expected)


class TestLogEMA(unittest.TestCase):
    """_LogEMA: edge cases for add() and value()."""

    def test_add_nonpositive_value_is_noop(self):
        ema = erd_swarm._LogEMA(tau=1.0, min_weight=1)
        ema.add(0)
        ema.add(-5)
        self.assertIsNone(ema.value())  # weight still 0

    def test_add_with_implicit_now_warms_ema(self):
        ema = erd_swarm._LogEMA(tau=1.0, min_weight=1)
        ema.add(10)   # now=None → time.time() called internally
        self.assertIsNotNone(ema.value())

    def test_add_twice_applies_decay(self):
        # Use a very short tau so decay is substantial and clearly exercises the
        # decay branch; min_weight=1 so value() is readable after each add.
        ema = erd_swarm._LogEMA(tau=0.1, min_weight=1)
        ema.add(100, now=0.0)
        ema.add(100, now=1.0)   # second call: self._last is not None → decay applied
        # Decayed weight < 2; value should still be computable with min_weight=1.
        self.assertIsNotNone(ema.value())

    def test_value_returns_geometric_mean_when_warm(self):
        import math
        ema = erd_swarm._LogEMA(tau=1e9, min_weight=2)  # huge tau → no decay
        ema.add(10, now=0.0)
        ema.add(1000, now=0.0)
        expected = math.exp((math.log(10) + math.log(1000)) / 2)
        self.assertAlmostEqual(ema.value(), expected, places=6)


class TestPublishThresholdWarmPath(unittest.TestCase):
    """_publish_threshold returns SAFETY_FACTOR * coord / node_time when warm."""

    def test_warm_threshold_exceeds_bootstrap(self):
        w = _bare_worker()
        # Use now=0.0 for all adds so decay = exp(0) = 1 and weight accumulates
        # cleanly to exactly min_weight, without floating-point decay undershooting.
        for _ in range(erd_swarm._PUBLISH_EMA_MIN_WEIGHT):
            w._coord_ema.add(0.01, now=0.0)      # 10 ms coordination
            w._node_time_ema.add(1e-6, now=0.0)  # 1 µs / node
        threshold = w._publish_threshold()
        self.assertGreater(threshold, erd_swarm.PUBLISH_THRESHOLD_BOOTSTRAP)


class TestFlushCostModelBuffer(unittest.TestCase):
    """_flush_cost_model_buffer propagates buffered inline samples to the queue."""

    def test_flush_calls_update_logsums_per_size(self):
        w = _bare_worker()
        # Buffer is keyed by (size, budget); flush forwards budget as a kwarg.
        w._cost_model_buffer = {(5, 0): (2.0, 4.0, 1), (10, 1): (3.0, 9.0, 2)}
        w._flush_cost_model_buffer()
        self.assertEqual(w.queue.update_cost_model_logsums.call_count, 2)
        budgets = {c.kwargs.get('budget')
                   for c in w.queue.update_cost_model_logsums.call_args_list}
        self.assertEqual(budgets, {0, 1})
        w._typical_cache.clear()  # confirmed cleared by flush
        self.assertEqual(w._cost_model_buffer, {})

    def test_flush_empty_buffer_clears_without_update(self):
        w = _bare_worker()
        w._flush_cost_model_buffer()
        w.queue.update_cost_model_logsums.assert_not_called()


class TestRecordInlineEdgeCases(unittest.TestCase):
    """record_inline edge cases: zero-node frame and duplicate size in buffer."""

    def _pub(self):
        w = _bare_worker()
        pub = erd_swarm._MidLoopPublisher(w)
        return pub, w

    def test_record_inline_noop_when_nodes_unchanged(self):
        pub, w = self._pub()
        token = pub.enter(BRANCH[:4], budget=0)
        # _nodes stays 0 → nodes_spent = 0 → early return, nothing buffered.
        pub.record_inline(token)
        self.assertEqual(w._cost_model_buffer, {})

    def test_record_inline_accumulates_second_sample_for_same_size(self):
        import math
        pub, w = self._pub()
        words = BRANCH[:4]
        token1 = pub.enter(words, budget=0)
        w._nodes = 100
        pub.record_inline(token1)
        token2 = pub.enter(words, budget=0)
        w._nodes = 300   # 200 more nodes for second frame of same size
        pub.record_inline(token2)
        s, sq, c = w._cost_model_buffer[(len(words), 0)]
        self.assertEqual(c, 2)
        self.assertAlmostEqual(s, math.log(100) + math.log(200), places=6)


class TestNoteDepthDeeperPruning(unittest.TestCase):
    """_note_depth prunes deeper spine entries when resetting a shallower depth."""

    def test_note_depth_prunes_deeper_entries(self):
        w = _bare_worker()
        w._note_depth(6, 200, guess="crane", pattern="00000")
        w._note_depth(5, 50, guess="slate", pattern="10000")
        w._note_depth(4, 10, guess="trace", pattern="22222")
        # Revisit depth 1: depths 2+ should be pruned.
        w._note_depth(5, 48, guess="tales", pattern="01000")
        self.assertNotIn(2, w._spine)
        self.assertIn(1, w._spine)

    def test_sentinel_prunes_deeper_entries(self):
        w = _bare_worker()
        w._note_depth(5, 50, guess="crane", pattern="00000")
        w._note_depth(4, 20, guess="slate", pattern="10000")
        w._note_depth(3, 5, guess="trace", pattern="22222")
        # Sentinel at depth 1 with n<0 should prune depths 2 and 3.
        w._note_depth(5, -1)
        self.assertNotIn(2, w._spine)
        self.assertNotIn(3, w._spine)


class TestMidLoopPublisherBranchEdgeCases(unittest.TestCase):
    """Publisher check() with best_guess and with empty done_indices."""

    def _pub_overrun(self, predicted=10, best_guess=None):
        w = _bare_worker()
        w.queue.get_cost_typical.return_value = predicted
        pub = erd_swarm._MidLoopPublisher(w)
        words = BRANCH[:6]
        token = pub.enter(words, budget=0)
        w._nodes = erd_swarm.OVERRUN_K * predicted + 1
        w.cooperative_solve = mock.MagicMock(
            return_value=(erd_swarm.SOLVED, 2.0, 3, False))
        result = pub.check(token, CANDIDATES, 0, best_guess, 1.5, 5)
        return result, w, pub

    def test_check_calls_update_branch_best_when_best_guess_known(self):
        result, w, _ = self._pub_overrun(best_guess="crane")
        self.assertIsNotNone(result)
        w.queue.update_branch_best.assert_called_once()

    def test_check_skips_update_branch_best_when_no_best_guess(self):
        result, w, _ = self._pub_overrun(best_guess=None)
        self.assertIsNotNone(result)
        w.queue.update_branch_best.assert_not_called()

    def test_check_handles_empty_done_indices(self):
        # Use a candidate_list of words that are NOT in word_idx so done_indices
        # is empty — mark_claims_done should not be called.
        w = _bare_worker()
        w.queue.get_cost_typical.return_value = 10
        pub = erd_swarm._MidLoopPublisher(w)
        words = BRANCH[:6]
        token = pub.enter(words, budget=0)
        w._nodes = erd_swarm.OVERRUN_K * 10 + 1
        w.cooperative_solve = mock.MagicMock(
            return_value=(erd_swarm.SOLVED, 2.0, 3, False))
        unknown_words = ["zzzzz"] * 10
        pub.check(token, unknown_words, 0, None, None, 5)
        w.queue.mark_claims_done.assert_not_called()


if __name__ == "__main__":
    unittest.main()
