"""Tests for the read-only hint cache (issue #304).

A clean ERD rebuild runs two databases: an empty writable live cache holding
only results this solver recomputed, and a quarantined historical cache that is
allowed to influence candidate *order* and nothing else.  These tests are
mostly about the second half of that sentence — what a historical row must not
be able to do.

Two habits run through the file.  The poisoning fixtures carry values that are
provably wrong (an ERD of 0.001, a worst case of 1), so an assertion that the
live result is unchanged is only meaningful if the same value *would* change it
when trusted; TestPoisonIsRealWhenTrusted wires each fixture through the
ordinary exact-read path and shows it does.  And the ordering assertions come
in pairs: what the order is with the hint, and what it is without, so a test
that would pass with hint selection removed cannot hide.
"""
import os
import shutil
import sqlite3
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest import mock

import erd_search
import erd_swarm
import export_cache
import import_cache
from cache_sqlite import ScoreCache
from erd_queue import ERDQueue, encode_subset
from hint_cache import HintCache, HintCacheError, open_hint_cache
from report_model import _normalize_worker
from report_terminal import render_report
from tests.test_report_terminal import overview_report
import wordle_engine
from wordle_engine import ERD_ALL, ResponseCache, min_expected_guesses

# Two rhyme families the guess vocabulary can only partly separate, so the root
# is big enough for the ORDER_MIN_N gate and the search recurses through real
# multi-word sub-branches instead of collapsing to singletons.
ANSWERS = ["cater", "dater", "eater", "hater", "later", "rater", "tater",
           "water", "eight", "fight", "light", "might", "night", "right",
           "sight", "tight"]
GUESSES = ANSWERS + ["bumpy", "clomp", "dwarf", "crumb", "flesh"]
ROOT_BUDGET = 6

# The value the search reaches on this fixture with no hints at all.  Every
# poisoning test asserts the run still lands here.
TRUE_ROOT_ERD = 3.125


def _drop_sidecars(db_path):
    """Remove the WAL/SHM sidecars a finished write session leaves behind.

    The hint artifact is defined as a checkpointed, self-contained file.  A
    zero-byte -wal and a stale -shm are exactly what ScoreCache.close leaves,
    and archiving them is what makes "no sidecar exists afterwards" a
    meaningful assertion rather than a comparison against clutter.
    """
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = db_path + suffix
        if os.path.exists(sidecar):
            os.remove(sidecar)


class _HintCacheTest(unittest.TestCase):
    """Two directories: the live cache in one, the hint artifact alone in the
    other, so a directory listing is a usable "nothing was created" assertion.
    """

    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._dir, ignore_errors=True)
        self.hint_dir = os.path.join(self._dir, "hints")
        os.mkdir(self.hint_dir)
        self.live_path = os.path.join(self._dir, "live.sqlite3")
        self.hint_path = os.path.join(self.hint_dir, "history.sqlite3")

    # -- fixtures ---------------------------------------------------------

    def write_history(self, rows, path=None, answer_words=ANSWERS):
        """Build a hint artifact holding `rows` and nothing else.

        rows are (branch_words, best_guess, best_score, max_depth,
        solve_budget) — the same shape ScoreCache.write takes, so a test can
        state a historical fact exactly as the historical solver would have.
        """
        path = path or self.hint_path
        cache = ScoreCache(path, answer_words)
        for branch_words, best_guess, best_score, max_depth, budget in rows:
            cache.write(ScoreCache.encode_subset(branch_words), ERD_ALL,
                        best_guess, best_score, max_depth=max_depth,
                        solve_budget=budget)
        cache.close()
        _drop_sidecars(path)
        return path

    def hints(self, path=None, answer_words=ANSWERS):
        hint = HintCache(path or self.hint_path, answer_words,
                         live_cache_path=self.live_path)
        self.addCleanup(hint.close)
        return hint

    def live(self, name="live.sqlite3"):
        cache = ScoreCache(os.path.join(self._dir, name), ANSWERS,
                           checkpoint_on_close=False)
        self.addCleanup(cache.close)
        return cache

    def solve(self, score_cache, hint_cache=None, budget=ROOT_BUDGET):
        rcache = ResponseCache(ANSWERS, score_cache)
        return min_expected_guesses(
            ANSWERS, rcache, score_cache, guesses=GUESSES, policy=ERD_ALL,
            budget=budget, hint_cache=hint_cache)


class TrueResultTest(_HintCacheTest):
    """The baseline every poisoning test is measured against."""

    def test_the_unhinted_solve_reaches_the_documented_erd(self):
        self.assertAlmostEqual(self.solve(self.live()), TRUE_ROOT_ERD)


class HistoricalValuesAreNotFactsTest(_HintCacheTest):
    """A historical row may name a word.  It may not price one."""

    def _assert_unchanged_by(self, rows):
        hint = self.hints(self.write_history(rows))
        self.assertAlmostEqual(self.solve(self.live(), hint_cache=hint),
                               TRUE_ROOT_ERD)
        return hint

    def test_an_absurdly_low_historical_erd_changes_nothing(self):
        hint = self._assert_unchanged_by(
            [(ANSWERS, "bumpy", 0.001, 1, None)])
        # The word was still used: the guard is that its *price* was ignored,
        # not that the row went unread.
        self.assertEqual(hint.accepted, 1)

    def test_an_absurdly_high_historical_erd_changes_nothing(self):
        self._assert_unchanged_by([(ANSWERS, "bumpy", 99.0, 6, None)])

    def test_a_wrong_historical_max_remaining_depth_changes_nothing(self):
        self._assert_unchanged_by([(ANSWERS, "clomp", 2.0, 1, None)])

    def test_a_wrong_budget_specific_historical_row_changes_nothing(self):
        self._assert_unchanged_by(
            [(ANSWERS, "clomp", 0.5, 2, ROOT_BUDGET)])

    def test_a_historical_loss_row_never_makes_a_branch_unsolvable(self):
        path = self.write_history([(ANSWERS, "bumpy", 0.5, 1, None)])
        # Re-open writable to add the loss, then re-archive: write_loss is not
        # part of the row shape write_history takes because no other test
        # needs it.
        historical = ScoreCache(path, ANSWERS)
        historical.write_loss(ScoreCache.encode_subset(ANSWERS), ERD_ALL,
                              ROOT_BUDGET)
        historical.close()
        _drop_sidecars(path)

        self.assertAlmostEqual(
            self.solve(self.live(), hint_cache=self.hints(path)),
            TRUE_ROOT_ERD)

    def test_a_recursively_reached_branch_is_recomputed_not_returned(self):
        """A sub-branch the live cache has never seen is solved from scratch.

        The historical row for it is deliberately both wrong and cheap: if any
        part of it were reused, the parent's fold would carry the lie upward
        and the root ERD would move.
        """
        live = self.live()
        # A branch the unhinted search genuinely reaches and caches (its own
        # exact result is "flesh" at 2.428…), so the assertions below are
        # about a recomputation that really happened.
        sub_branch = ["eight", "fight", "light", "might", "night", "sight",
                      "tight"]
        hint = self.hints(self.write_history([
            (ANSWERS, "flesh", 2.0, 2, None),
            (sub_branch, "bumpy", 0.001, 1, None),
        ]))
        with mock.patch("wordle_engine.ORDER_MIN_N", 2):
            cost = self.solve(live, hint_cache=hint)
        self.assertAlmostEqual(cost, TRUE_ROOT_ERD)
        # And the live cache holds its own recomputed result for that
        # sub-branch, in whichever scope the search solved it under, rather
        # than the historical one.
        stored = live._conn.execute(
            """SELECT best_guess, best_score FROM branch_best_by_policy
                   WHERE branch_key = ?
               UNION ALL
               SELECT best_guess, best_score
                   FROM branch_best_by_policy_and_budget
                   WHERE branch_key = ?""",
            (ScoreCache.encode_subset(sub_branch),) * 2).fetchall()
        self.assertTrue(stored)
        for row in stored:
            self.assertNotEqual(row["best_guess"], "bumpy")
            self.assertGreater(row["best_score"], 1.0)


class PoisonIsRealWhenTrustedTest(_HintCacheTest):
    """Proof that the fixtures above are actually poisonous.

    Each historical row the previous class calls harmless is handed to the
    solver as its ordinary exact-result cache — the mistake this whole feature
    exists to prevent.  The result must move.  If it did not, every "unchanged"
    assertion above would be passing for the wrong reason.
    """

    def _solve_trusting(self, rows):
        path = os.path.join(self._dir, "trusted.sqlite3")
        self.write_history(rows, path=path)
        trusted = ScoreCache(path, ANSWERS, checkpoint_on_close=False)
        self.addCleanup(trusted.close)
        return self.solve(trusted)

    def test_a_low_historical_erd_is_believed_when_read_as_exact(self):
        self.assertAlmostEqual(
            self._solve_trusting([(ANSWERS, "bumpy", 0.001, 1, None)]), 0.001)

    def test_a_high_historical_erd_is_believed_when_read_as_exact(self):
        self.assertAlmostEqual(
            self._solve_trusting([(ANSWERS, "bumpy", 99.0, 6, None)]), 99.0)

    def test_a_budget_specific_historical_row_is_believed_when_read_as_exact(self):
        self.assertAlmostEqual(
            self._solve_trusting([(ANSWERS, "clomp", 0.5, 2, ROOT_BUDGET)]),
            0.5)

    def test_a_historical_loss_row_is_believed_when_read_as_exact(self):
        path = os.path.join(self._dir, "trusted_loss.sqlite3")
        trusted = ScoreCache(path, ANSWERS, checkpoint_on_close=False)
        self.addCleanup(trusted.close)
        trusted.write_loss(ScoreCache.encode_subset(ANSWERS), ERD_ALL,
                           ROOT_BUDGET)
        self.assertIsNone(self.solve(trusted))


class CandidateOrderTest(_HintCacheTest):
    """Where the hinted word lands, and what happens when it cannot land."""

    def _first_candidates(self, hint_cache, live_name):
        """The order _solve_subset's top-level loop evaluated candidates in.

        live_name gives each call its own empty live cache: a second solve
        against a cache that already holds the root's exact result returns
        from it and evaluates no candidates at all.
        """
        seen = []
        original = wordle_engine.evaluate_candidate

        def recording(branch_words, candidate, *args, **kwargs):
            if len(branch_words) == len(ANSWERS):
                seen.append(candidate)
            return original(branch_words, candidate, *args, **kwargs)

        with mock.patch("wordle_engine.evaluate_candidate", recording):
            self.solve(self.live(live_name), hint_cache=hint_cache)
        self.assertTrue(seen)
        return seen

    def test_the_hinted_word_is_evaluated_first(self):
        hint = self.hints(self.write_history([(ANSWERS, "bumpy", 3.0, 4, None)]))
        hinted_order = self._first_candidates(hint, "hinted.sqlite3")
        unhinted_order = self._first_candidates(None, "plain.sqlite3")

        self.assertEqual(hinted_order[0], "bumpy")
        # Non-vacuous: without the hint the search starts somewhere else, so a
        # build with hint selection removed fails this test rather than
        # coincidentally satisfying it.
        self.assertNotEqual(unhinted_order[0], "bumpy")

    def test_a_word_outside_the_candidate_pool_is_ignored(self):
        hint = self.hints(
            self.write_history([(ANSWERS, "zzzzz", 3.0, 4, None)]))
        self.assertEqual(self._first_candidates(hint, "hinted.sqlite3"),
                         self._first_candidates(None, "plain.sqlite3"))
        self.assertEqual(hint.rejected, 1)
        self.assertEqual(hint.accepted, 0)

    def test_a_branch_the_artifact_does_not_cover_keeps_the_normal_order(self):
        hint = self.hints(self.write_history([]))
        self.assertEqual(self._first_candidates(hint, "hinted.sqlite3"),
                         self._first_candidates(None, "plain.sqlite3"))
        self.assertEqual(hint.hits, 0)
        self.assertGreater(hint.lookups, 0)

    def test_a_live_exact_hit_never_consults_the_hint_cache(self):
        live = self.live()
        live.write(ScoreCache.encode_subset(ANSWERS), ERD_ALL, "flesh",
                   TRUE_ROOT_ERD, max_depth=4, solve_budget=None)
        hint = self.hints(self.write_history([(ANSWERS, "bumpy", 0.001, 1, None)]))

        self.assertAlmostEqual(self.solve(live, hint_cache=hint),
                               TRUE_ROOT_ERD)
        self.assertEqual(hint.lookups, 0)

    def test_the_hint_is_skipped_on_branches_below_the_ordering_gate(self):
        """The ORDER_MIN_N gate applies to the hint for the same reason it
        applies to best-first ordering: below it the scan costs more than the
        subtree it could reorder."""
        hint = self.hints(self.write_history([(ANSWERS, "bumpy", 3.0, 4, None)]))
        with mock.patch("wordle_engine.ORDER_MIN_N", len(ANSWERS) + 1):
            self.solve(self.live(), hint_cache=hint)
        self.assertEqual(hint.lookups, 0)

    def test_a_winning_hint_is_recorded_as_a_winner(self):
        # "flesh" is what the unhinted search picks for the root, so hinting it
        # produces a winner; the count reflects a recomputed victory, since the
        # historical price attached to it here is nonsense.
        hint = self.hints(self.write_history([(ANSWERS, "flesh", 9.0, 6, None)]))
        self.assertAlmostEqual(self.solve(self.live(), hint_cache=hint),
                               TRUE_ROOT_ERD)
        self.assertEqual(hint.inline_wins, 1)

    def test_a_losing_hint_is_not_recorded_as_a_winner(self):
        hint = self.hints(self.write_history([(ANSWERS, "bumpy", 3.0, 4, None)]))
        self.solve(self.live(), hint_cache=hint)
        self.assertEqual(hint.accepted, 1)
        self.assertEqual(hint.inline_wins, 0)


class ScopePreferenceTest(_HintCacheTest):
    """Which historical scope names the candidate when several could."""

    def test_the_exact_budget_row_is_preferred_over_the_unrestricted_one(self):
        hint = self.hints(self.write_history([
            (ANSWERS, "clomp", 3.0, 4, None),
            (ANSWERS, "bumpy", 3.0, 4, ROOT_BUDGET),
        ]))
        key = ScoreCache.encode_subset(ANSWERS)
        self.assertEqual(hint.hint_candidate(key, ERD_ALL, ROOT_BUDGET),
                         "bumpy")
        self.assertEqual(hint.hint_candidate(key, ERD_ALL, ROOT_BUDGET - 1),
                         "clomp")
        self.assertEqual(hint.hint_candidate(key, ERD_ALL, None), "clomp")

    def test_stats_report_every_counter_under_its_heartbeat_name(self):
        hint = self.hints(self.write_history([(ANSWERS, "clomp", 3.0, 4, None)]))
        key = ScoreCache.encode_subset(ANSWERS)
        hint.hint_candidate(key, ERD_ALL, None)
        # One acceptance from each site: the swarm's counts only as an
        # acceptance, the engine's also enters the inline population whose
        # wins this process sees.
        hint.note_accepted_for_branch_order()
        hint.note_accepted_in_frame()
        hint.note_inline_win()
        hint.note_rejected()

        self.assertEqual(hint.stats(), {
            "hint_lookups": 1, "hint_hits": 1, "hint_accepted": 2,
            "hint_rejected": 1, "hint_inline_placements": 1,
            "hint_inline_wins": 1})

    def test_a_non_counting_read_leaves_the_lookup_rate_alone(self):
        hint = self.hints(self.write_history([(ANSWERS, "clomp", 3.0, 4, None)]))
        key = ScoreCache.encode_subset(ANSWERS)
        hint.hint_candidate(key, ERD_ALL, None, count_lookup=False)
        self.assertEqual((hint.lookups, hint.hits), (0, 0))
        hint.hint_candidate(key, ERD_ALL, None)
        self.assertEqual((hint.lookups, hint.hits), (1, 1))


class ArtifactImmutabilityTest(_HintCacheTest):
    """The artifact must come out of a run byte-for-byte as it went in."""

    def _snapshot(self):
        listing = sorted(os.listdir(self.hint_dir))
        stat = os.stat(self.hint_path)
        return listing, (stat.st_size, stat.st_mtime_ns, stat.st_ino)

    def test_a_full_hinted_solve_leaves_the_artifact_untouched(self):
        self.write_history([
            (ANSWERS, "bumpy", 0.001, 1, None),
            (["eight", "fight", "light", "might", "night", "sight",
              "tight"], "clomp", 0.002, 1, None),
        ])
        before = self._snapshot()
        hint = self.hints()
        with mock.patch("wordle_engine.ORDER_MIN_N", 2):
            self.solve(self.live(), hint_cache=hint)
        hint.close()

        self.assertEqual(self._snapshot(), before)
        # Stated separately from the stat comparison because it is the
        # condition Sean measured a plain mode=ro open failing: no sidecar may
        # be created, not merely no page written.
        self.assertEqual(sorted(os.listdir(self.hint_dir)),
                         ["history.sqlite3"])

    def test_the_connection_rejects_a_write(self):
        self.write_history([(ANSWERS, "bumpy", 3.0, 4, None)])
        hint = self.hints()
        with self.assertRaises(sqlite3.OperationalError):
            hint._conn.execute(
                "UPDATE branch_best_by_policy SET best_guess = 'zzzzz'")

    def test_no_migration_marker_is_written_to_the_artifact(self):
        """A pre-split artifact is presented through temp views, never
        migrated: the file must not gain the split's migration row."""
        path = os.path.join(self.hint_dir, "presplit.sqlite3")
        _build_pre_split_cache(path, ANSWERS)
        before = os.stat(path).st_mtime_ns
        hint = HintCache(path, ANSWERS, live_cache_path=self.live_path)
        self.addCleanup(hint.close)
        hint.hint_candidate(ScoreCache.encode_subset(ANSWERS), ERD_ALL, None)
        self.assertEqual(os.stat(path).st_mtime_ns, before)


def _build_pre_split_cache(path, answer_words):
    """A cache in the shape the split's migration has never run on.

    One branch_best_by_policy table carrying both scopes, no
    branch_best_by_policy_and_budget, no schema_migrations row for the split —
    which is exactly what the quarantined historical file looks like.
    """
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE branch_best_by_policy (
            branch_key     BLOB NOT NULL,
            policy         TEXT NOT NULL,
            answer_list_id TEXT NOT NULL,
            solve_budget   INTEGER,
            best_guess     TEXT NOT NULL,
            best_score     REAL NOT NULL,
            updated_at     INTEGER,
            max_depth      INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE answer_list (
            answer_list_id TEXT PRIMARY KEY,
            answer_hash    TEXT NOT NULL,
            answer_count   INTEGER NOT NULL,
            created_at     INTEGER NOT NULL
        )
    """)
    from cache_sqlite import answer_list_id
    list_id = answer_list_id(list(answer_words))
    conn.execute("INSERT INTO answer_list VALUES (?, ?, ?, 0)",
                 (list_id, list_id, len(answer_words)))
    conn.execute(
        "INSERT INTO branch_best_by_policy VALUES (?, ?, ?, NULL, ?, ?, 0, 4)",
        (ScoreCache.encode_subset(answer_words), ERD_ALL, list_id, "clomp",
         3.0))
    conn.execute(
        "INSERT INTO branch_best_by_policy VALUES (?, ?, ?, 6, ?, ?, 0, 4)",
        (ScoreCache.encode_subset(answer_words), ERD_ALL, list_id, "bumpy",
         3.0))
    conn.commit()
    conn.close()
    _drop_sidecars(path)
    return path


class PreSplitArtifactTest(_HintCacheTest):
    """The quarantined file predates the scope split and must still serve."""

    def test_both_scopes_are_readable_from_a_pre_split_artifact(self):
        path = _build_pre_split_cache(
            os.path.join(self.hint_dir, "presplit.sqlite3"), ANSWERS)
        hint = HintCache(path, ANSWERS, live_cache_path=self.live_path)
        self.addCleanup(hint.close)
        key = ScoreCache.encode_subset(ANSWERS)

        self.assertEqual(hint.hint_candidate(key, ERD_ALL, None), "clomp")
        self.assertEqual(hint.hint_candidate(key, ERD_ALL, 6), "bumpy")
        self.assertEqual(hint.namespace_branch_count, 2)


class StartupRefusalTest(_HintCacheTest):
    """Every way an operator can name something that is not a hint artifact."""

    def test_a_missing_path_is_refused(self):
        with self.assertRaises(HintCacheError) as caught:
            HintCache(os.path.join(self.hint_dir, "absent.sqlite3"), ANSWERS,
                      live_cache_path=self.live_path)
        self.assertIn("does not exist", str(caught.exception))

    def test_the_live_cache_path_is_refused(self):
        self.write_history([(ANSWERS, "bumpy", 3.0, 4, None)],
                           path=self.live_path)
        with self.assertRaises(HintCacheError) as caught:
            HintCache(self.live_path, ANSWERS, live_cache_path=self.live_path)
        self.assertIn("live cache", str(caught.exception))

    def test_a_path_resolving_onto_the_live_cache_is_refused(self):
        self.write_history([(ANSWERS, "bumpy", 3.0, 4, None)],
                           path=self.live_path)
        link = os.path.join(self.hint_dir, "alias.sqlite3")
        os.symlink(self.live_path, link)
        with self.assertRaises(HintCacheError) as caught:
            HintCache(link, ANSWERS, live_cache_path=self.live_path)
        self.assertIn("live cache", str(caught.exception))

    def test_a_hard_linked_live_cache_is_refused(self):
        self.write_history([(ANSWERS, "bumpy", 3.0, 4, None)],
                           path=self.live_path)
        link = os.path.join(self.hint_dir, "hardlink.sqlite3")
        os.link(self.live_path, link)
        with self.assertRaises(HintCacheError) as caught:
            HintCache(link, ANSWERS, live_cache_path=self.live_path)
        self.assertIn("same file", str(caught.exception))

    def test_a_nonempty_write_ahead_log_is_refused(self):
        """immutable=1 would read straight past it and serve a stale
        snapshot, so the open refuses rather than quietly losing rows."""
        self.write_history([(ANSWERS, "bumpy", 3.0, 4, None)])
        with open(self.hint_path + "-wal", "wb") as wal:
            wal.write(b"\x00" * 32)
        with self.assertRaises(HintCacheError) as caught:
            HintCache(self.hint_path, ANSWERS,
                      live_cache_path=self.live_path)
        self.assertIn("write-ahead log", str(caught.exception))

    def test_an_empty_write_ahead_log_is_accepted(self):
        self.write_history([(ANSWERS, "bumpy", 3.0, 4, None)])
        open(self.hint_path + "-wal", "wb").close()
        hint = HintCache(self.hint_path, ANSWERS,
                         live_cache_path=self.live_path)
        self.addCleanup(hint.close)
        self.assertEqual(hint.namespace_branch_count, 1)

    def test_a_hot_rollback_journal_is_refused(self):
        self.write_history([(ANSWERS, "bumpy", 3.0, 4, None)])
        with open(self.hint_path + "-journal", "wb") as journal:
            journal.write(b"\x00" * 32)
        with self.assertRaises(HintCacheError) as caught:
            HintCache(self.hint_path, ANSWERS,
                      live_cache_path=self.live_path)
        self.assertIn("rollback journal", str(caught.exception))

    def test_a_file_with_no_branch_results_is_refused(self):
        path = os.path.join(self.hint_dir, "empty.sqlite3")
        sqlite3.connect(path).close()
        with self.assertRaises(HintCacheError) as caught:
            HintCache(path, ANSWERS, live_cache_path=self.live_path)
        self.assertIn("branch results", str(caught.exception))

    def test_a_file_whose_answer_list_cannot_be_inspected_is_refused(self):
        """A branch table exists but the answer-list namespace cannot be
        counted, so "how much of this artifact applies to my run" has no
        answer and the run refuses rather than reporting zero coverage."""
        path = os.path.join(self.hint_dir, "noscope.sqlite3")
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE branch_best_by_policy (branch_key BLOB)")
        conn.execute(
            "CREATE TABLE branch_best_by_policy_and_budget (branch_key BLOB)")
        conn.commit()
        conn.close()
        _drop_sidecars(path)
        with self.assertRaises(HintCacheError) as caught:
            HintCache(path, ANSWERS, live_cache_path=self.live_path)
        self.assertIn("answer-list namespace", str(caught.exception))

    def test_an_artifact_for_another_answer_list_reports_no_coverage(self):
        """Not a refusal: a different word list is a legitimate artifact that
        happens to answer nothing, and a run may proceed hintless in fact."""
        self.write_history([(ANSWERS, "bumpy", 3.0, 4, None)])
        other = HintCache(self.hint_path, ANSWERS + ["zzzzz"],
                          live_cache_path=self.live_path)
        self.addCleanup(other.close)
        self.assertEqual(other.namespace_branch_count, 0)
        self.assertIsNone(other.hint_candidate(
            ScoreCache.encode_subset(ANSWERS), ERD_ALL, None))

    def test_no_path_means_no_hint_cache_and_no_error(self):
        self.assertIsNone(open_hint_cache(None, ANSWERS, self.live_path))

    def test_a_directory_is_refused_as_unopenable(self):
        with self.assertRaises(HintCacheError) as caught:
            HintCache(self.hint_dir, ANSWERS, live_cache_path=self.live_path)
        self.assertIn("could not be opened read-only", str(caught.exception))

    def test_a_file_that_is_not_a_database_is_refused(self):
        path = os.path.join(self.hint_dir, "notes.txt")
        with open(path, "w") as text_file:
            text_file.write("this is not a database\n")
        with self.assertRaises(HintCacheError) as caught:
            HintCache(path, ANSWERS, live_cache_path=self.live_path)
        self.assertIn("does not hold branch results", str(caught.exception))

    def test_opening_without_a_live_cache_path_skips_the_collision_check(self):
        """The check is a courtesy to the CLI, which always has both paths; a
        direct construction (a tool inspecting an artifact) may omit it."""
        self.write_history([(ANSWERS, "bumpy", 3.0, 4, None)])
        hint = HintCache(self.hint_path, ANSWERS)
        self.addCleanup(hint.close)
        self.assertEqual(hint.namespace_branch_count, 1)


class SurfaceTest(_HintCacheTest):
    """The class cannot hand out anything but a word."""

    def test_the_public_surface_has_no_write_or_result_reads(self):
        forbidden = ("write", "delete", "import", "export", "read_for_budget",
                     "read_with_depth", "read_loss", "repair")
        public = [name for name in dir(HintCache)
                  if not name.startswith("_")]
        for name in public:
            for prefix in forbidden:
                self.assertFalse(
                    name.startswith(prefix),
                    f"HintCache.{name} exposes {prefix}-shaped behaviour; the "
                    f"hint surface must return words only")

    def test_the_hint_query_selects_only_the_candidate_word(self):
        """Structural, not behavioural: a query that never names best_score or
        max_depth cannot leak one, whatever a future caller does."""
        import inspect

        import hint_cache
        source = inspect.getsource(hint_cache.HintCache._read_hint)
        for column in ("best_score", "max_depth", "solve_budget FROM"):
            self.assertNotIn(column, source)

    def test_export_and_import_move_only_live_cache_tables(self):
        """No hint-shaped table joins the five that cross to the phone."""
        expected = {"answer_list", "response_decomposition",
                    "branch_best_by_policy",
                    "branch_best_by_policy_and_budget", "candidate_scores"}
        self.assertEqual(set(export_cache.EXPORT_TABLES), expected)
        self.assertEqual(set(import_cache.TABLES), expected)


class SupervisorStartupTest(_HintCacheTest):
    """erd_search run's own refusal, before any worker forks."""

    def _args(self, hint_cache_path):
        return types.SimpleNamespace(cache=self.live_path,
                                     hint_cache=hint_cache_path,
                                     queue=os.path.join(self._dir, "q.sqlite3"),
                                     workers=2)

    def test_no_hint_cache_is_usable(self):
        self.assertTrue(erd_search._hint_cache_is_usable(self._args(None)))

    def test_a_usable_artifact_is_reported_with_its_coverage(self):
        self.write_history([(ANSWERS, "bumpy", 3.0, 4, None)])
        out = StringIO()
        with redirect_stdout(out):
            usable = erd_search._hint_cache_is_usable(
                self._args(self.hint_path))
        self.assertTrue(usable)
        self.assertIn("read-only, candidate order only", out.getvalue())

    def test_an_unusable_artifact_stops_the_run(self):
        err = StringIO()
        with redirect_stderr(err):
            usable = erd_search._hint_cache_is_usable(
                self._args(os.path.join(self.hint_dir, "absent.sqlite3")))
        self.assertFalse(usable)
        self.assertIn("Refusing to start", err.getvalue())

    def test_cmd_run_returns_without_spawning_on_a_refusal(self):
        args = self._args(os.path.join(self.hint_dir, "absent.sqlite3"))
        args.workers = 1
        with mock.patch.object(erd_search, "_spawn_worker") as spawn, \
                redirect_stderr(StringIO()):
            erd_search.cmd_run(args)
        spawn.assert_not_called()

    def test_every_respawned_worker_is_given_the_same_hint_path(self):
        """Worker restart preserves the separation: the path travels as a
        process argument, so a respawn re-opens it read-only exactly as the
        original did."""
        args = self._args(self.hint_path)
        with mock.patch.object(erd_search.multiprocessing, "Process") as proc:
            erd_search._spawn_worker(3, args, stop_event=None)
        self.assertEqual(proc.call_args.kwargs["kwargs"]["hint_cache_path"],
                         self.hint_path)
        self.assertIs(proc.call_args.kwargs["target"], erd_swarm.swarm_worker)


class QueueBoundaryTest(_HintCacheTest):
    """Queue admission answers from the live cache alone."""

    def _add_args(self, cache_path):
        return types.SimpleNamespace(
            word=["cater"], words_file=None, pattern=None, priority=None,
            priority_step=erd_search.DEFAULT_PRIORITY_STEP,
            priority_words=None, max_branch_size=None, delete_erd_cache=False,
            cache=cache_path,
            queue=os.path.join(self._dir, "queue.sqlite3"))

    def _run_queue_add(self, cache_path):
        out = StringIO()
        with redirect_stdout(out), redirect_stderr(StringIO()):
            erd_search.cmd_queue_add(self._add_args(cache_path))
        return out.getvalue()

    def _one_response_group_key(self):
        answers = erd_search.load_word_list(erd_search.ANSWER_FILE)
        probe = ScoreCache(os.path.join(self._dir, "probe.sqlite3"), answers)
        self.addCleanup(probe.close)
        groups = ResponseCache(answers, probe).group_words("cater", answers)
        branch = max(groups.values(), key=len)
        return encode_subset(branch), answers

    def test_hint_only_coverage_does_not_satisfy_queue_admission(self):
        branch_key, answers = self._one_response_group_key()
        # A complete historical result for the branch, in a file queue add has
        # no way to name.
        history = ScoreCache(os.path.join(self.hint_dir, "h.sqlite3"), answers)
        history.write(branch_key, ERD_ALL, "salet", 2.0, max_depth=3)
        history.close()

        self._run_queue_add(os.path.join(self._dir, "empty_live.sqlite3"))

        queue = ERDQueue(os.path.join(self._dir, "queue.sqlite3"))
        self.addCleanup(queue.close)
        self.assertIsNotNone(queue.get_pending_branch(branch_key))

    def test_the_same_row_in_the_live_cache_does_satisfy_it(self):
        """The pairing that makes the test above non-vacuous: the row is one
        queue add would honour, and it is honoured when it is live."""
        branch_key, answers = self._one_response_group_key()
        live_path = os.path.join(self._dir, "seeded_live.sqlite3")
        live = ScoreCache(live_path, answers)
        live.write(branch_key, ERD_ALL, "salet", 2.0, max_depth=3)
        live.close()

        self._run_queue_add(live_path)

        queue = ERDQueue(os.path.join(self._dir, "queue.sqlite3"))
        self.addCleanup(queue.close)
        self.assertIsNone(queue.get_pending_branch(branch_key))

    def test_queue_add_takes_no_hint_cache_option(self):
        """There is no way to hand queue add a hint artifact, so hint-only
        coverage cannot reach admission even by operator error."""
        with mock.patch.object(erd_search, "cmd_queue_add") as handler, \
                mock.patch("sys.argv",
                           ["erd_search.py", "queue", "add", "--word", "cater"]):
            erd_search.main()
        self.assertNotIn("hint_cache", vars(handler.call_args.args[0]))

    def test_run_does_take_a_hint_cache_option(self):
        with mock.patch.object(erd_search, "cmd_run") as handler, \
                mock.patch("sys.argv", ["erd_search.py", "run",
                                        "--hint-cache", self.hint_path]):
            erd_search.main()
        self.assertEqual(handler.call_args.args[0].hint_cache, self.hint_path)


class SwarmCandidateOrderTest(_HintCacheTest):
    """The branch-level packing order the swarm hands out claims in."""

    def _worker(self, hint_cache):
        worker = erd_swarm._BranchWorker.__new__(erd_swarm._BranchWorker)
        worker.hint_cache = hint_cache
        worker.all_words = tuple(GUESSES)
        return worker

    def test_the_hinted_word_leads_the_packing_order(self):
        hint = self.hints(self.write_history([(ANSWERS, "bumpy", 3.0, 4, None)]))
        worker = self._worker(hint)
        natural = list(range(len(GUESSES)))

        reordered = worker._hint_first_in_order(
            ScoreCache.encode_subset(ANSWERS), list(natural), ROOT_BUDGET)

        self.assertEqual(reordered[0], GUESSES.index("bumpy"))
        self.assertNotEqual(natural[0], GUESSES.index("bumpy"))
        # Still a permutation: the hint changes which candidate goes out
        # first, never which candidates the branch owes.
        self.assertEqual(sorted(reordered), natural)

    def test_an_order_already_led_by_the_hint_is_returned_unchanged(self):
        hint = self.hints(self.write_history([(ANSWERS, "cater", 3.0, 4, None)]))
        worker = self._worker(hint)
        order = [GUESSES.index("cater")] + [
            i for i in range(len(GUESSES)) if i != GUESSES.index("cater")]

        self.assertEqual(
            worker._hint_first_in_order(ScoreCache.encode_subset(ANSWERS),
                                        list(order), ROOT_BUDGET),
            order)

    def test_a_word_outside_the_guess_vocabulary_leaves_the_order_alone(self):
        hint = self.hints(self.write_history([(ANSWERS, "zzzzz", 3.0, 4, None)]))
        worker = self._worker(hint)
        natural = list(range(len(GUESSES)))

        self.assertEqual(
            worker._hint_first_in_order(ScoreCache.encode_subset(ANSWERS),
                                        list(natural), ROOT_BUDGET),
            natural)
        self.assertEqual(hint.rejected, 1)

    def test_a_worker_with_no_artifact_leaves_the_order_alone(self):
        worker = self._worker(None)
        natural = list(range(len(GUESSES)))
        self.assertEqual(
            worker._hint_first_in_order(ScoreCache.encode_subset(ANSWERS),
                                        list(natural), ROOT_BUDGET),
            natural)

    def test_the_finalize_outcome_names_the_hint_and_whether_it_won(self):
        hint = self.hints(self.write_history([(ANSWERS, "bumpy", 3.0, 4, None)]))
        worker = self._worker(hint)
        key = ScoreCache.encode_subset(ANSWERS)

        self.assertEqual(worker._hint_outcome(key, "bumpy", ROOT_BUDGET),
                         {"hint_word": "bumpy", "hint_was_winner": True})
        self.assertEqual(worker._hint_outcome(key, "flesh", ROOT_BUDGET),
                         {"hint_word": "bumpy", "hint_was_winner": False})
        # A cut or a loss has no winner, so there is no contest the hint could
        # have lost; the word is still on record.
        self.assertEqual(worker._hint_outcome(key, None, ROOT_BUDGET),
                         {"hint_word": "bumpy", "hint_was_winner": None})
        # Reporting must not inflate the lookup rate the payoff is measured
        # against.
        self.assertEqual(hint.lookups, 0)

    def test_a_branch_the_artifact_does_not_cover_records_a_null_hint(self):
        hint = self.hints(self.write_history([]))
        outcome = self._worker(hint)._hint_outcome(
            ScoreCache.encode_subset(ANSWERS), "flesh", ROOT_BUDGET)
        self.assertEqual(outcome, {"hint_word": None, "hint_was_winner": None})

    def test_a_worker_with_no_artifact_records_nothing_at_all(self):
        """An empty mapping leaves both columns NULL, which is a different
        fact from a measured miss."""
        self.assertEqual(
            self._worker(None)._hint_outcome(
                ScoreCache.encode_subset(ANSWERS), "flesh", ROOT_BUDGET),
            {})


class TelemetryColumnTest(_HintCacheTest):
    """The queue columns the payoff measurement is read out of."""

    def queue(self):
        queue = ERDQueue(os.path.join(self._dir, "queue.sqlite3"))
        self.addCleanup(queue.close)
        return queue

    def test_the_first_incumbent_is_stamped_once_and_never_moved(self):
        queue = self.queue()
        key = encode_subset(ANSWERS)
        queue.create_branch(key, len(ANSWERS), len(GUESSES), budget=ROOT_BUDGET)
        queue.add_nodes_spent(key, 40)

        queue.update_branch_best(key, "flesh", 3.5, max_depth=4)
        first = dict(queue.get_branch(key))
        queue.add_nodes_spent(key, 60)
        queue.update_branch_best(key, "clomp", 3.1, max_depth=4)
        later = dict(queue.get_branch(key))

        self.assertEqual(first["nodes_at_first_best"], 40)
        self.assertIsNotNone(first["first_best_at"])
        self.assertEqual(later["nodes_at_first_best"], 40)
        self.assertEqual(later["first_best_at"], first["first_best_at"])
        # The best itself did move, so the stamp is pinned by intent rather
        # than by nothing having happened.
        self.assertEqual(later["best_guess"], "clomp")

    def test_a_branch_with_no_incumbent_has_no_stamp(self):
        queue = self.queue()
        key = encode_subset(ANSWERS)
        queue.create_branch(key, len(ANSWERS), len(GUESSES), budget=ROOT_BUDGET)
        row = queue.get_branch(key)
        self.assertIsNone(row["first_best_at"])
        self.assertIsNone(row["nodes_at_first_best"])

    def test_the_finalize_log_carries_the_hint_and_first_incumbent(self):
        queue = self.queue()
        key = encode_subset(ANSWERS)
        queue.add_branch_finalize_log(
            key, "cater -g-g-", len(ANSWERS), ROOT_BUDGET, 100, 200, 900, 12,
            hint_word="bumpy", hint_was_winner=False,
            first_best_at=150, nodes_at_first_best=40)
        row = queue._conn.execute(
            "SELECT hint_word, hint_was_winner, first_best_at, "
            "nodes_at_first_best FROM telemetry.branch_finalize_log"
        ).fetchone()
        self.assertEqual(tuple(row), ("bumpy", 0, 150, 40))

    def test_a_run_without_hints_leaves_the_finalize_columns_null(self):
        queue = self.queue()
        queue.add_branch_finalize_log(
            encode_subset(ANSWERS), None, len(ANSWERS), ROOT_BUDGET, 100, 200,
            900, 12)
        row = queue._conn.execute(
            "SELECT hint_word, hint_was_winner FROM "
            "telemetry.branch_finalize_log").fetchone()
        self.assertEqual(tuple(row), (None, None))

    def test_the_heartbeat_carries_hint_counters(self):
        queue = self.queue()
        queue.heartbeat("worker-0", 1234, None, None, 0, 0,
                        hint_lookups=9, hint_hits=7, hint_accepted=6,
                        hint_rejected=1, hint_inline_placements=4,
                        hint_inline_wins=2)
        row = queue._conn.execute(
            "SELECT hint_lookups, hint_hits, hint_accepted, hint_rejected, "
            "hint_inline_placements, hint_inline_wins "
            "FROM worker_heartbeat").fetchone()
        self.assertEqual(tuple(row), (9, 7, 6, 1, 4, 2))

    def test_a_heartbeat_without_hints_leaves_the_counters_null(self):
        queue = self.queue()
        queue.heartbeat("worker-0", 1234, None, None, 0, 0)
        row = queue._conn.execute(
            "SELECT hint_lookups, hint_hits FROM worker_heartbeat").fetchone()
        self.assertEqual(tuple(row), (None, None))


class HintReportingTest(unittest.TestCase):
    """Hint statistics are shown, and shown apart from the live cache's."""

    def _report(self, **hint_totals):
        report = overview_report()
        report["data"]["worker_totals"].update(hint_totals)
        return report

    def test_no_hint_section_when_the_run_has_no_artifact(self):
        output = render_report(self._report(), width=100)
        self.assertNotIn("Hints", output)
        self.assertIn("Cache:", output)

    def test_hint_statistics_render_under_their_own_label(self):
        output = render_report(self._report(
            hint_lookup_count=1000, hint_hit_count=800,
            hint_accepted_count=750, hint_rejected_count=50,
            hint_inline_placement_count=400,
            hint_inline_win_count=300), width=100)

        self.assertIn("Hints (ordering only):", output)
        self.assertIn("lookups 1,000", output)
        # "named" and "legal" are rates over lookups; the win rate is taken
        # only over inline placements, whose wins the same counter sees.
        self.assertIn("named 80.0%", output)
        self.assertIn("legal 93.8%", output)
        self.assertIn("inline won 75.0%", output)
        # The live cache's own counts keep their own line and their own
        # meaning: a hint is not a cache hit.
        cache_line = next(line for line in output.splitlines()
                          if line.startswith("Cache:"))
        self.assertNotIn("hint", cache_line.lower())

    def test_a_worker_row_reports_hint_counters_apart_from_cache_counters(self):
        worker = _normalize_worker(
            {"worker_id": "worker-0", "pid": 7, "updated_at": 100,
             "cache_hits": 5, "cache_misses": 2,
             "hint_lookups": 9, "hint_hits": 7, "hint_accepted": 6,
             "hint_rejected": 1, "hint_inline_placements": 4,
             "hint_inline_wins": 2},
            generated_at=100, answer_set=set(ANSWERS))

        self.assertEqual(worker["cache_hit_count"], 5)
        self.assertEqual(
            (worker["hint_lookup_count"], worker["hint_hit_count"],
             worker["hint_accepted_count"], worker["hint_rejected_count"],
             worker["hint_inline_placement_count"],
             worker["hint_inline_win_count"]),
            (9, 7, 6, 1, 4, 2))

    def test_a_worker_with_no_artifact_reports_no_hint_measurement(self):
        """None, not zero: the worker did not measure zero hints, it had no
        artifact to measure."""
        worker = _normalize_worker(
            {"worker_id": "worker-0", "pid": 7, "updated_at": 100},
            generated_at=100, answer_set=set(ANSWERS))
        self.assertIsNone(worker["hint_lookup_count"])
        self.assertEqual(worker["cache_hit_count"], 0)


class RecursiveHintDepthTest(_HintCacheTest):
    """The hint reaches every frame the search recurses into, not just the root."""

    def _branch_sizes_looked_up(self, hint):
        """Answer-set sizes the search asked the artifact about."""
        sizes = []
        real = hint.hint_candidate

        def recording(branch_key, policy, budget, count_lookup=True):
            if count_lookup:
                sizes.append(len(branch_key) // 5)
            return real(branch_key, policy, budget, count_lookup=count_lookup)

        hint.hint_candidate = recording
        with mock.patch("wordle_engine.ORDER_MIN_N", 2):
            self.solve(self.live(), hint_cache=hint)
        return sizes

    def test_descendant_frames_get_their_own_ordering_hints(self):
        """#304 requires recursively reached branches to be hinted too.

        The failure this guards against is silent: forwarding the hint into
        evaluate_candidate's own recursion but not into the candidate loop
        that calls it leaves the root the only branch ever asked about, and
        every result assertion still passes.
        """
        sizes = self._branch_sizes_looked_up(
            self.hints(self.write_history([])))

        self.assertIn(len(ANSWERS), sizes)
        deeper = sorted({size for size in sizes if size < len(ANSWERS)})
        self.assertTrue(
            deeper, f"only the root was looked up; sizes seen: {sorted(set(sizes))}")
        # More than one level below the root, so a hint that reached children
        # but not grandchildren is caught as well.
        self.assertGreater(len(deeper), 1, f"only one depth below the root: {deeper}")


class BudgetScopedPackingHintTest(_HintCacheTest):
    """A branch the artifact covers only at a budget still gets its hint.

    The quarantined file holds tens of thousands of budget-specific rows, so a
    packing order that asked only the unrestricted table would pass over every
    branch represented by one.
    """

    def _worker(self, hint_cache):
        worker = erd_swarm._BranchWorker.__new__(erd_swarm._BranchWorker)
        worker.hint_cache = hint_cache
        worker.all_words = tuple(GUESSES)
        return worker

    def test_a_budget_only_row_leads_the_packing_order(self):
        hint = self.hints(self.write_history(
            [(ANSWERS, "bumpy", 3.0, 4, ROOT_BUDGET)]))
        worker = self._worker(hint)
        key = ScoreCache.encode_subset(ANSWERS)
        natural = list(range(len(GUESSES)))

        at_budget = worker._hint_first_in_order(key, list(natural), ROOT_BUDGET)
        self.assertEqual(at_budget[0], GUESSES.index("bumpy"))
        self.assertEqual(sorted(at_budget), natural)

        # Non-vacuous: the unrestricted scope — what the packing order used to
        # ask for — has nothing to say about this branch, so a build that
        # dropped the budget leaves the order untouched.
        self.assertEqual(
            worker._hint_first_in_order(key, list(natural), None), natural)

    def test_the_branch_budget_selects_between_two_historical_scopes(self):
        hint = self.hints(self.write_history([
            (ANSWERS, "clomp", 3.0, 4, None),
            (ANSWERS, "bumpy", 3.0, 4, ROOT_BUDGET),
        ]))
        worker = self._worker(hint)
        key = ScoreCache.encode_subset(ANSWERS)
        natural = list(range(len(GUESSES)))

        self.assertEqual(
            worker._hint_first_in_order(key, list(natural), ROOT_BUDGET)[0],
            GUESSES.index("bumpy"))
        self.assertEqual(
            worker._hint_first_in_order(key, list(natural), ROOT_BUDGET - 1)[0],
            GUESSES.index("clomp"))

    def test_a_packing_order_is_cached_per_branch_and_budget(self):
        """A branch re-created at another budget must not inherit the order
        computed for the one it was finalized at."""
        hint = self.hints(self.write_history([
            (ANSWERS, "clomp", 3.0, 4, None),
            (ANSWERS, "bumpy", 3.0, 4, ROOT_BUDGET),
        ]))
        worker = self._worker(hint)
        worker._packing_stats_cache = {}
        worker.n_candidates = len(GUESSES)
        worker.pattern_matrix = _FlatCandidateStats(len(GUESSES))
        key = ScoreCache.encode_subset(ANSWERS)

        at_budget, _ = worker._packing_stats(key, ANSWERS, ROOT_BUDGET)
        unrestricted, _ = worker._packing_stats(key, ANSWERS, ROOT_BUDGET - 1)

        self.assertEqual(at_budget[0], GUESSES.index("bumpy"))
        self.assertEqual(unrestricted[0], GUESSES.index("clomp"))
        self.assertEqual(len(worker._packing_stats_cache), 2)


class _FlatCandidateStats:
    """PatternMatrix stand-in giving every candidate the same ordering key.

    With no Σk² signal the natural index order survives sorting, so a test can
    attribute any change in the packing order to the hint alone.
    """

    def __init__(self, n_candidates):
        self._n = n_candidates

    def answer_indices(self, words):
        return list(range(len(words)))

    def candidate_stats(self, branch_indices):
        return types.SimpleNamespace(
            sum_squared_group_sizes=[0] * self._n,
            cost_lower_bound=[0.0] * self._n)


class HintPopulationTest(_HintCacheTest):
    """The displayed win rate compares like with like."""

    def _worker(self, hint_cache):
        worker = erd_swarm._BranchWorker.__new__(erd_swarm._BranchWorker)
        worker.hint_cache = hint_cache
        worker.all_words = tuple(GUESSES)
        return worker

    def test_a_branch_order_acceptance_stays_out_of_the_inline_population(self):
        """Several workers each place the same cooperative branch's hint, and
        only one of them finalizes it.  Counting those placements in the
        inline denominator would report a win rate diluted by branches whose
        winner this process never sees."""
        hint = self.hints(self.write_history([(ANSWERS, "bumpy", 3.0, 4, None)]))
        key = ScoreCache.encode_subset(ANSWERS)
        natural = list(range(len(GUESSES)))
        for _ in range(3):
            self._worker(hint)._hint_first_in_order(key, list(natural), None)

        self.assertEqual(hint.accepted, 3)
        self.assertEqual(hint.inline_placements, 0)
        self.assertEqual(hint.inline_wins, 0)

    def test_an_inline_placement_counts_in_both(self):
        hint = self.hints(self.write_history([(ANSWERS, "flesh", 9.0, 6, None)]))
        self.solve(self.live(), hint_cache=hint)

        self.assertEqual(hint.accepted, 1)
        self.assertEqual(hint.inline_placements, 1)
        self.assertEqual(hint.inline_wins, 1)


class SupervisorLeavesTheArtifactAloneTest(_HintCacheTest):
    """cmd_run must refuse a same-file configuration before writing anything.

    Every step of startup writes to --cache: a WAL checkpoint, then a
    ScoreCache open that runs the schema migrations.  If the operator names one
    file for both flags, doing any of that first rewrites the quarantined
    artifact — new tables, its budget rows moved, a changed mtime — and the
    refusal that follows can no longer undo it.  So these assert on the file,
    not on the message.
    """

    def _args(self, cache_path, hint_path):
        return types.SimpleNamespace(
            cache=cache_path, hint_cache=hint_path,
            queue=os.path.join(self._dir, "q.sqlite3"), workers=1)

    def _fingerprint(self, path):
        """Everything a startup write would disturb: bytes, timestamp, and
        the schema the migration would have added to."""
        stat = os.stat(path)
        conn = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
        try:
            tables = sorted(
                row[0] for row in
                conn.execute("SELECT name FROM sqlite_master WHERE type='table'"))
            rows = conn.execute(
                "SELECT COUNT(*) FROM branch_best_by_policy "
                "WHERE solve_budget IS NOT NULL").fetchone()[0]
        finally:
            conn.close()
        return (stat.st_size, stat.st_mtime_ns, tables, rows,
                sorted(os.listdir(os.path.dirname(path))))

    def _assert_refused_without_touching(self, artifact, args):
        before = self._fingerprint(artifact)
        with mock.patch.object(erd_search, "_spawn_worker") as spawn, \
                mock.patch.object(erd_search, "_checkpoint_cache_on_start") as ckpt, \
                redirect_stderr(StringIO()) as err:
            erd_search.cmd_run(args)
        spawn.assert_not_called()
        # The checkpoint is the first thing that would have written; proving it
        # was never reached is what makes this a guard on ordering rather than
        # on the artifact happening to survive.
        ckpt.assert_not_called()
        self.assertIn("Refusing to start", err.getvalue())
        self.assertEqual(self._fingerprint(artifact), before)

    def test_the_same_path_for_both_flags_is_refused_before_any_write(self):
        artifact = _build_pre_split_cache(
            os.path.join(self.hint_dir, "presplit.sqlite3"), ANSWERS)
        self._assert_refused_without_touching(
            artifact, self._args(artifact, artifact))

    def test_a_symlinked_live_cache_is_refused_before_any_write(self):
        artifact = _build_pre_split_cache(
            os.path.join(self.hint_dir, "presplit.sqlite3"), ANSWERS)
        alias = os.path.join(self._dir, "live_alias.sqlite3")
        os.symlink(artifact, alias)
        self._assert_refused_without_touching(
            artifact, self._args(alias, artifact))

    def test_a_hard_linked_live_cache_is_refused_before_any_write(self):
        artifact = _build_pre_split_cache(
            os.path.join(self.hint_dir, "presplit.sqlite3"), ANSWERS)
        alias = os.path.join(self._dir, "live_hardlink.sqlite3")
        os.link(artifact, alias)
        self._assert_refused_without_touching(
            artifact, self._args(alias, artifact))

    def test_opening_a_pre_split_cache_writably_is_what_this_prevents(self):
        """The pairing that makes the tests above non-vacuous.

        A writable open of the same fixture migrates it, so "unchanged" is a
        fact about ordering, not about a file that was never at risk.
        """
        artifact = _build_pre_split_cache(
            os.path.join(self.hint_dir, "presplit.sqlite3"), ANSWERS)
        before = self._fingerprint(artifact)

        ScoreCache(artifact, ANSWERS).close()

        after = self._fingerprint(artifact)
        self.assertNotEqual(after, before)
        self.assertIn("branch_best_by_policy_and_budget", after[2])
        self.assertNotIn("branch_best_by_policy_and_budget", before[2])


class LoggedHintIsThePlacedHintTest(_HintCacheTest):
    """What finalize records must be the word that actually led the order.

    The two sites read the artifact independently — packing when the branch is
    claimed, finalize when it is retired — so they can disagree without either
    one looking wrong on its own.  These tests assert the pair, not each half:
    whatever `_hint_first_in_order` puts first is what `_hint_outcome` logs.
    """

    def _worker(self, hint_cache):
        worker = erd_swarm._BranchWorker.__new__(erd_swarm._BranchWorker)
        worker.hint_cache = hint_cache
        worker.all_words = tuple(GUESSES)
        return worker

    def _placed_and_logged(self, worker, budget, winner):
        key = ScoreCache.encode_subset(ANSWERS)
        order = worker._hint_first_in_order(
            key, list(range(len(GUESSES))), budget)
        placed = GUESSES[order[0]]
        return placed, worker._hint_outcome(key, winner, budget)

    def test_a_budget_only_row_is_logged_not_dropped(self):
        """The failure shape: packing leads with the budget-specific word
        while finalize, asking the unrestricted scope, has nothing to record —
        so the payoff telemetry silently omits every branch the artifact
        covers only at a budget."""
        hint = self.hints(self.write_history(
            [(ANSWERS, "bumpy", 3.0, 4, ROOT_BUDGET)]))
        placed, logged = self._placed_and_logged(
            self._worker(hint), ROOT_BUDGET, "bumpy")

        self.assertEqual(placed, "bumpy")
        self.assertEqual(logged, {"hint_word": "bumpy", "hint_was_winner": True})

    def test_conflicting_scopes_log_the_scope_that_ordered(self):
        """The other failure shape: with both scopes present, packing leads
        with the budget-specific word and finalize records the unrestricted
        one — a hint that won is filed as a miss."""
        hint = self.hints(self.write_history([
            (ANSWERS, "clomp", 3.0, 4, None),
            (ANSWERS, "bumpy", 3.0, 4, ROOT_BUDGET),
        ]))
        placed, logged = self._placed_and_logged(
            self._worker(hint), ROOT_BUDGET, "bumpy")

        self.assertEqual(placed, "bumpy")
        self.assertEqual(logged, {"hint_word": "bumpy", "hint_was_winner": True})

    def test_the_pair_agrees_at_every_budget_the_artifact_covers(self):
        """Swept rather than spot-checked: at each budget the two sites must
        name the same word, whichever scope answers there."""
        hint = self.hints(self.write_history([
            (ANSWERS, "clomp", 3.0, 4, None),
            (ANSWERS, "bumpy", 3.0, 4, ROOT_BUDGET),
        ]))
        worker = self._worker(hint)
        for budget in (None, ROOT_BUDGET, ROOT_BUDGET - 1, ROOT_BUDGET - 2):
            with self.subTest(budget=budget):
                placed, logged = self._placed_and_logged(
                    worker, budget, "flesh")
                self.assertEqual(logged["hint_word"], placed)
        # And the two scopes really do differ, so the sweep is not agreeing
        # because there is only one answer to give.
        self.assertEqual(
            self._placed_and_logged(worker, ROOT_BUDGET, "flesh")[0], "bumpy")
        self.assertEqual(
            self._placed_and_logged(worker, None, "flesh")[0], "clomp")


class BranchLevelHintRateQueryTest(_HintCacheTest):
    """The rate SWARM.md documents, run against real finalize rows.

    A cut or a proven loss has a hint_word but a NULL hint_was_winner, because
    no candidate won for the hint to have matched.  Counting those in the
    denominator scores every no-contest branch as a miss.
    """

    DOCUMENTED_QUERY = """
        SELECT AVG(hint_was_winner) FROM telemetry.branch_finalize_log
        WHERE hint_was_winner IS NOT NULL
    """

    def _queue_with_rows(self, rows):
        queue = ERDQueue(os.path.join(self._dir, "queue.sqlite3"))
        self.addCleanup(queue.close)
        for index, (hint_word, was_winner) in enumerate(rows):
            queue.add_branch_finalize_log(
                encode_subset(ANSWERS[index:index + 3]), None, 3, ROOT_BUDGET,
                100, 200, 900, 12,
                hint_word=hint_word, hint_was_winner=was_winner)
        return queue

    def test_no_contest_branches_are_excluded_from_the_denominator(self):
        # Two hinted branches reached a winner and both matched; three more
        # were hinted but finalized as a cut or a loss.
        queue = self._queue_with_rows([
            ("bumpy", True), ("clomp", True),
            ("dwarf", None), ("crumb", None), ("flesh", None),
        ])
        rate = queue._conn.execute(self.DOCUMENTED_QUERY).fetchone()[0]
        self.assertEqual(rate, 1.0)

        # The denominator the prose used to name counts the no-contest rows,
        # turning a perfect record into 40%.  This is what the filter fixes.
        naive = queue._conn.execute("""
            SELECT COUNT(*) FILTER (WHERE hint_was_winner = 1) * 1.0 / COUNT(*)
            FROM telemetry.branch_finalize_log WHERE hint_word IS NOT NULL
        """).fetchone()[0]
        self.assertAlmostEqual(naive, 0.4)

    def test_a_genuine_miss_still_lowers_the_rate(self):
        queue = self._queue_with_rows([
            ("bumpy", True), ("clomp", False), ("dwarf", None),
        ])
        self.assertAlmostEqual(
            queue._conn.execute(self.DOCUMENTED_QUERY).fetchone()[0], 0.5)

    def test_the_documented_query_is_the_one_in_the_guide(self):
        """SWARM.md's example and this test must not drift apart."""
        with open("SWARM.md") as guide:
            text = guide.read()
        for fragment in ("AVG(hint_was_winner)",
                         "WHERE hint_was_winner IS NOT NULL"):
            self.assertIn(fragment, text)


if __name__ == "__main__":
    unittest.main()
