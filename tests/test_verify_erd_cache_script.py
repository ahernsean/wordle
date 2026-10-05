"""verify_erd_cache.py drops the stored opener ERDs a correction may falsify."""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import verify_erd_cache
from cache_sqlite import ScoreCache
from wordle_engine import ERD_ALL

ANSWERS = ["crane", "slate"]
CANDIDATES = ["crane", "slate", "raise"]


class VerifyERDCacheScriptTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache_path = os.path.join(tmp.name, "cache.sqlite3")
        self.log_path = os.path.join(tmp.name, "verify.log")
        self.answer_path = os.path.join(tmp.name, "answers.txt")
        self.candidate_path = os.path.join(tmp.name, "candidates.txt")
        for path, words in ((self.answer_path, ANSWERS),
                            (self.candidate_path, CANDIDATES)):
            with open(path, "w") as handle:
                handle.write("\n".join(words) + "\n")
        for name, path in (("ANSWER_FILE", self.answer_path),
                           ("WORDS_FILE", self.candidate_path)):
            patcher = mock.patch.object(verify_erd_cache, name, path)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _cache(self, stored_guess, stored_score):
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        cache.write(ScoreCache.encode_subset(ANSWERS), ERD_ALL, stored_guess,
                    stored_score, max_depth=2)
        cache.write_opener_erd("crane", ERD_ALL, 1.5, 2, 2)
        cache.close()

    def _stored_openers(self):
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        try:
            return cache.opener_names_with_erd(ERD_ALL)
        finally:
            cache.close()

    def _run(self):
        output = io.StringIO()
        argv = ["verify_erd_cache.py", "--cache", self.cache_path,
                "--log", self.log_path, "--workers", "1"]
        with mock.patch.object(sys, "argv", argv), \
                contextlib.redirect_stdout(output):
            verify_erd_cache.main()
        return output.getvalue()

    def test_a_correction_drops_the_stored_opener_erds(self):
        # CRANE separates the pair for 1.5; a stored 2.0 is too high.
        self._cache("raise", 2.0)

        output = self._run()

        self.assertIn("Score corrected : 1", output)
        self.assertEqual(self._stored_openers(), set())
        self.assertIn("reconcile-opener-erds", output)

    def test_a_run_that_corrects_nothing_leaves_them(self):
        self._cache("crane", 1.5)

        output = self._run()

        self.assertIn("Score corrected : 0", output)
        self.assertEqual(self._stored_openers(), {"crane"})
        self.assertNotIn("reconcile-opener-erds", output)


if __name__ == "__main__":
    unittest.main()
