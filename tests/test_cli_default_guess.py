"""The guess prompt's best-ERD default.

Covers the cache lookup behind the default (_best_erd_guess), the two prompt
forms (gray placeholder on a real terminal, [word] in the prompt text
elsewhere), the raw-mode line editor itself, and cmd_guess playing the
default when the entry is empty.
"""
import io
import os
import pty
import sys
import termios
import tty
import unittest
from contextlib import redirect_stdout
from unittest import mock

import wordle
from wordle import cmd_guess, ScoreCache
from wordle_engine import ERD_ALL
from clitestutil import CliTestCase

GRAY = wordle.ANSI_COLORS['gray']


def _drive_ghost(keys, prompt="Word to guess? ", ghost="crane"):
    """Run _read_line_with_ghost against a pty preloaded with `keys`.

    The pty is put in raw mode before the keys are written so control
    characters reach the reader as bytes, exactly as they do when a player
    types them at an already-raw prompt.  Returns (result, written output),
    with the exception type name as the result if the read raised.
    """
    master, slave = pty.openpty()
    tty.setraw(slave)
    os.write(master, keys)
    stdin = os.fdopen(slave, 'rb', 0)
    saved_stdin, sys.stdin = sys.stdin, stdin
    out = io.StringIO()
    try:
        with redirect_stdout(out):
            try:
                result = wordle._read_line_with_ghost(prompt, ghost)
            except (EOFError, KeyboardInterrupt) as e:
                result = type(e).__name__
    finally:
        sys.stdin = saved_stdin
        stdin.close()
        os.close(master)
    return result, out.getvalue()


class TestReadLineWithGhost(unittest.TestCase):
    def test_enter_takes_the_ghost(self):
        result, out = _drive_ghost(b"\r")
        self.assertEqual(result, "crane")
        # Drawn gray, with the cursor parked back on its first letter.
        self.assertIn(f'{GRAY}crane{wordle.ANSI_RESET}', out)
        self.assertIn('\033[5D', out)

    def test_typing_replaces_the_ghost(self):
        result, out = _drive_ghost(b"slate\r")
        self.assertEqual(result, "slate")
        # The ghost is drawn once, before anything is typed, and never again.
        self.assertEqual(out.count(f'{GRAY}crane'), 1)
        self.assertTrue(out.rstrip('\n').endswith("Word to guess? slate"))

    def test_backspacing_to_empty_restores_the_ghost(self):
        result, out = _drive_ghost(b"sl\x7f\x7f\r")
        self.assertEqual(result, "crane")
        self.assertEqual(out.count(f'{GRAY}crane'), 3)  # initial, empty, final

    def test_ctrl_u_clears_the_entry(self):
        result, _ = _drive_ghost(b"stale\x15\r")
        self.assertEqual(result, "crane")

    def test_arrow_key_is_not_typed_text(self):
        result, _ = _drive_ghost(b"\x1b[Aslate\r")
        self.assertEqual(result, "slate")

    def test_typing_through_a_bare_escape_loses_nothing(self):
        result, _ = _drive_ghost(b"\x1bslate\r")
        self.assertEqual(result, "slate")

    def test_editing_an_empty_entry_leaves_the_ghost_alone(self):
        # Backspace and ctrl-U on an empty line, then an unprintable key:
        # none of them disturb the placeholder.
        result, out = _drive_ghost(b"\x7f\x15\x01\r")
        self.assertEqual(result, "crane")
        self.assertEqual(out.count(f'{GRAY}crane'), 2)  # initial, final

    def test_end_of_input_raises(self):
        result, _ = _drive_ghost(b"\x04")
        self.assertEqual(result, "EOFError")

    def test_closed_stream_raises_eof(self):
        # A genuinely empty os.read -- distinct from ctrl-D -- is what a
        # hung-up stream returns once its writer is gone.  Mocked rather
        # than reproduced by actually closing the pty's master: closing it
        # while a read is already blocked races with the kernel and can
        # surface as EIO instead of a clean b'', which isn't the case this
        # test is pinning.
        master, slave = pty.openpty()
        tty.setraw(slave)
        stdin = os.fdopen(slave, 'rb', 0)
        saved_stdin, sys.stdin = sys.stdin, stdin
        out = io.StringIO()
        try:
            with redirect_stdout(out), \
                    mock.patch('wordle.os.read', return_value=b''):
                with self.assertRaises(EOFError):
                    wordle._read_line_with_ghost("Word to guess? ", "crane")
        finally:
            sys.stdin = saved_stdin
            stdin.close()
            os.close(master)

    def test_dead_fd_on_restore_does_not_mask_the_eof(self):
        # A hangup can leave the fd unable to take a termios restore too;
        # that must not replace the EOFError it caused with a termios.error.
        master, slave = pty.openpty()
        tty.setraw(slave)
        stdin = os.fdopen(slave, 'rb', 0)
        saved_stdin, sys.stdin = sys.stdin, stdin
        out = io.StringIO()
        try:
            with redirect_stdout(out), \
                    mock.patch('wordle.os.read', return_value=b''), \
                    mock.patch('wordle.termios.tcsetattr',
                              side_effect=termios.error(5, 'Input/output error')):
                with self.assertRaises(EOFError):
                    wordle._read_line_with_ghost("Word to guess? ", "crane")
        finally:
            sys.stdin = saved_stdin
            stdin.close()
            os.close(master)

    def test_interrupt_raises_and_keeps_typed_text_visible(self):
        result, out = _drive_ghost(b"sl\x03")
        self.assertEqual(result, "KeyboardInterrupt")
        self.assertTrue(out.rstrip('\n').endswith("Word to guess? sl"))
        self.assertNotIn(f'{GRAY}crane{wordle.ANSI_RESET}\n', out)

    def test_multibyte_character_is_typed_not_treated_as_eof(self):
        # 'é' is UTF-8 b'\xc3\xa9': one os.read(fd, 1) per byte. The first
        # byte alone decodes to '' -- indistinguishable from a closed stream
        # unless the reader tells the two apart.
        result, _ = _drive_ghost("é".encode('utf-8') + b"\r")
        self.assertEqual(result, "é")

    def test_multibyte_character_mid_entry_is_not_dropped(self):
        result, _ = _drive_ghost(b"sl" + "é".encode('utf-8') + b"te\r")
        self.assertEqual(result, "sléte")

    def test_invalid_utf8_byte_is_discarded(self):
        result, _ = _drive_ghost(b"s\xffl\r")
        self.assertEqual(result, "sl")


class TestGhostDefaultSupported(unittest.TestCase):
    """The capability probe, with SUPPORTS_COLOR forced on: what remains is
    the tty question."""

    @staticmethod
    def _probe(stdin, stdout):
        with mock.patch.object(wordle, 'SUPPORTS_COLOR', True), \
                mock.patch.object(wordle, 'IS_PYTHONISTA', False), \
                mock.patch.object(wordle, 'sys') as fake_sys:
            fake_sys.stdin, fake_sys.stdout = stdin, stdout
            return wordle._ghost_default_supported()

    def test_supported_when_both_ends_are_ttys(self):
        tty_stream = mock.Mock(isatty=lambda: True)
        self.assertTrue(self._probe(tty_stream, tty_stream))

    def test_unsupported_when_output_is_redirected(self):
        self.assertFalse(self._probe(mock.Mock(isatty=lambda: True),
                                     mock.Mock(isatty=lambda: False)))

    def test_unsupported_when_the_stream_is_closed(self):
        closed = mock.Mock(isatty=mock.Mock(side_effect=ValueError))
        self.assertFalse(self._probe(closed, closed))

    def test_unsupported_without_ansi_color(self):
        with mock.patch.object(wordle, 'SUPPORTS_COLOR', False):
            self.assertFalse(wordle._ghost_default_supported())


class TestEscapeSequences(unittest.TestCase):
    """_pending_key and _drain_escape against a pty holding known bytes.

    _drain_escape runs with the ESC itself already read, so `_fd` holds only
    what follows it.
    """

    def _fd(self, after_escape):
        master, slave = pty.openpty()
        tty.setraw(slave)
        os.write(master, after_escape)
        self.addCleanup(os.close, master)
        self.addCleanup(os.close, slave)
        return slave

    def test_pending_key_is_empty_when_nothing_is_typed(self):
        self.assertEqual(wordle._pending_key(self._fd(b""), timeout=0.01), '')

    def test_key_after_a_bare_escape_is_handed_back(self):
        fd = self._fd(b"s")  # Escape pressed on its own, then a guess letter
        self.assertEqual(wordle._drain_escape(fd), 's')

    def test_truncated_sequence_ends_at_end_of_input(self):
        fd = self._fd(b"[")
        # Returns rather than blocking for a final byte that never comes.
        self.assertEqual(wordle._drain_escape(fd), '')

    def test_sequence_is_consumed_up_to_its_final_byte(self):
        fd = self._fd(b"[1;2Bs")  # shift-Down, then a guess letter
        self.assertEqual(wordle._drain_escape(fd), '')
        self.assertEqual(wordle._pending_key(fd), 's')


class TestInputWithDefault(unittest.TestCase):
    def _run(self, default, typed, supported):
        out = io.StringIO()
        with redirect_stdout(out), \
                mock.patch.object(wordle, '_ghost_default_supported',
                                  return_value=supported), \
                mock.patch.object(wordle, '_read_line_with_ghost',
                                  return_value='ghosted') as ghost, \
                mock.patch('builtins.input', lambda *a: typed):
            value = wordle.input_with_default("Word to guess? ", default)
        return value, out.getvalue(), ghost

    def test_no_default_prompts_plainly(self):
        value, out, ghost = self._run(None, "slate", supported=True)
        self.assertEqual(value, "slate")
        self.assertEqual(out, "Word to guess? ")
        ghost.assert_not_called()

    def test_plain_terminal_offers_default_in_brackets(self):
        value, out, ghost = self._run("crane", "", supported=False)
        self.assertEqual(value, "crane")
        self.assertIn("[crane]", out)
        ghost.assert_not_called()

    def test_plain_terminal_entry_wins_over_default(self):
        value, _, _ = self._run("crane", "slate", supported=False)
        self.assertEqual(value, "slate")

    def test_capable_terminal_uses_the_ghost_reader(self):
        value, out, ghost = self._run("crane", "slate", supported=True)
        self.assertEqual(value, "ghosted")
        self.assertEqual(out, "")
        ghost.assert_called_once_with("Word to guess? ", "crane")


class TestBestERDGuess(CliTestCase):
    def _cache_erd(self, word, score=3.142, gs=None):
        gs = gs or self.gs
        gs.score_cache.write(
            ScoreCache.encode_subset(self.soln(gs).current_words),
            ERD_ALL, word, score)

    def test_returns_the_cached_hit(self):
        self._cache_erd("trace")
        self.assertEqual(wordle._best_erd_guess(self.gs, self.soln()),
                         ("trace", 3.142))

    def test_none_when_uncached(self):
        self.assertIsNone(wordle._best_erd_guess(self.gs, self.soln()))

    def test_none_once_solved(self):
        self._cache_erd("trace")
        self.soln().current_words = ["crane"]
        self.assertIsNone(wordle._best_erd_guess(self.gs, self.soln()))

    def test_status_line_and_default_agree(self):
        self._cache_erd("trace")
        out = io.StringIO()
        with redirect_stdout(out):
            wordle.print_status(self.gs)
        self.assertIn("3.142 TRACE", out.getvalue())


class TestCmdGuessDefault(CliTestCase):
    def _cache_erd(self, word, score=3.142):
        self.gs.score_cache.write(
            ScoreCache.encode_subset(self.soln().current_words),
            ERD_ALL, word, score)

    def test_empty_entry_plays_the_best_erd_word(self):
        self._cache_erd("trace")
        out = self.run_cmd(cmd_guess, inputs=["", "ggggg"])
        self.assertIn("[trace]", out)
        self.assertEqual(self.soln().guesses[0][0], "trace")

    def test_typed_word_still_wins(self):
        self._cache_erd("trace")
        self.run_cmd(cmd_guess, inputs=["slate", "ggggg"])
        self.assertEqual(self.soln().guesses[0][0], "slate")

    def test_no_default_without_a_cached_erd(self):
        out = self.run_cmd(cmd_guess, inputs=[""])
        self.assertNotIn("[", out.split("Word to guess?")[1])
        self.assertIn("5 letters", out)
        self.assertEqual(self.soln().guesses, [])

    def test_no_default_across_several_solutions(self):
        self._cache_erd("trace")
        self.gs.solutions.append(wordle.Solution(
            self.gs.all_answers, self.gs.all_words,
            self.gs.cache, self.gs.score_cache))
        out = self.run_cmd(cmd_guess, inputs=["a", ""])
        self.assertNotIn("[trace]", out)
        self.assertIn("5 letters", out)


if __name__ == '__main__':
    unittest.main()
