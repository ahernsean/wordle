"""Tests for section-based change detection in the swarm status display.

The live status display (erd_search._redraw_status) highlights characters that
changed since the previous refresh.  Diffing is done per named section so that a
section which grows or shrinks (e.g. a branch added to the Branches table) only
repaints its own rows and does not shift later sections into a spurious
all-changed diff.
"""

import io
import unittest
from contextlib import redirect_stdout

import erd_search

RED = '\033[1;31m'
CLEAR_EOL = '\033[K'


class BranchIdAndSpineHelperTest(unittest.TestCase):
    """Pure display helpers: stable ids, spine rendering, compact fallback."""

    def test_branch_id_is_stable_and_keyed_on_content(self):
        key = b'some-branch-key'
        self.assertEqual(erd_search._branch_id(key), erd_search._branch_id(key))
        self.assertEqual(len(erd_search._branch_id(key)), 4)
        self.assertNotEqual(erd_search._branch_id(key),
                            erd_search._branch_id(b'other-key'))

    def test_fmt_spine_path_pairs_edges(self):
        self.assertEqual(
            erd_search._fmt_spine_path('SALET -g-g- CRANE bb-y-'),
            'SALET -g-g- ▸ CRANE bb-y-')
        self.assertEqual(erd_search._fmt_spine_path(None), '')

    def test_compact_spine_uses_stored_path(self):
        out = erd_search._compact_spine_path(
            'SALET -g-g- CRANE bb-y-', 'salet', 0, lambda p: '-g-g-', 2)
        self.assertEqual(out, 'SALET -g-g- ▸ CRANE bb-y-')

    def test_compact_spine_tail_truncates_long_path(self):
        spine = 'SALET -g-g- CRANE bb-y- POUND g--y- MOUND gg--y QUILT yyyyy'
        out = erd_search._compact_spine_path(
            spine, 'salet', 0, lambda p: '-g-g-', 5, width=20)
        self.assertTrue(out.startswith('…'))
        self.assertLessEqual(len(out), 20)
        # The tail (deepest edges) is what survives truncation.
        self.assertTrue(out.endswith('QUILT yyyyy'))

    def test_compact_spine_falls_back_to_root_with_depth_marker(self):
        out = erd_search._compact_spine_path(
            None, 'salet', 0, lambda p: '-g-g-', 3)
        self.assertEqual(out, 'SALET -g-g- ▸ ?×3')

    def test_compact_spine_fallback_root_only_at_depth_zero(self):
        out = erd_search._compact_spine_path(
            None, 'salet', 0, lambda p: '-g-g-', 0)
        self.assertEqual(out, 'SALET -g-g-')

    def test_sweep_progress_bar_marks_partial_buckets_by_density(self):
        out = erd_search._sweep_progress_bar(
            100, done_indices=[0, 20, 21, 22, 23, 24, 50, 99],
            worker_positions=[], width=10)
        self.assertEqual(out, '▁ ▄  ▁   ▁')

    def test_sweep_progress_bar_marks_full_buckets(self):
        out = erd_search._sweep_progress_bar(
            10, done_indices=[0, 1, 2, 6], worker_positions=[], width=5)
        self.assertEqual(out, '█▄ ▄ ')

    def test_sweep_progress_bar_overlays_workers_on_done_map(self):
        out = erd_search._sweep_progress_bar(
            100, done_indices=[10, 20, 30],
            worker_positions=[(20, '2'), (21, '3')], width=10)
        self.assertEqual(out, ' ▁23      ')


class BranchDisplayOrderTest(unittest.TestCase):
    """The watch display keeps branch rows stable across refreshes."""

    def _branch(self, key, spine=None, source_word='salet',
                source_pattern=0, n_words=0):
        return {
            'branch_key': key,
            'spine': spine,
            'source_word': source_word,
            'source_pattern': source_pattern,
            'n_words': n_words,
        }

    def _bid(self, branch):
        return erd_search._branch_id(branch['branch_key'])

    def test_existing_branches_keep_previous_display_order(self):
        first = self._branch(
            b'first-branch',
            spine='SALET bb--- CRANE bby--',
            n_words=20)
        second = self._branch(
            b'second-branch',
            spine='SALET bb--- PRONE bby--',
            n_words=200)
        new = self._branch(
            b'new-branch',
            spine='SALET bb--- BLEND bbby-',
            n_words=1000)
        previous = [self._bid(second), self._bid(first)]

        ordered, order_state = erd_search._stable_branch_display_order(
            [new, first, second], previous)

        self.assertEqual(ordered, [second, first, new])
        self.assertEqual(order_state,
                         [self._bid(second), self._bid(first), self._bid(new)])

    def test_disappeared_branches_are_pruned_from_order_state(self):
        first = self._branch(b'first-branch')
        second = self._branch(b'second-branch')
        gone = self._branch(b'gone-branch')
        previous = [self._bid(second), self._bid(gone), self._bid(first)]

        ordered, order_state = erd_search._stable_branch_display_order(
            [first, second], previous)

        self.assertEqual(ordered, [second, first])
        self.assertEqual(order_state, [self._bid(second), self._bid(first)])

    def test_new_branches_use_stable_topology_order_not_size_order(self):
        later = self._branch(
            b'later-branch',
            spine='SALET bb--- CRANE yyyyy',
            n_words=1000)
        earlier = self._branch(
            b'earlier-branch',
            spine='SALET bb--- CRANE bbbbb',
            n_words=1)

        ordered, _ = erd_search._stable_branch_display_order(
            [later, earlier], [])

        self.assertEqual(ordered, [earlier, later])


class ParseSpineTest(unittest.TestCase):
    """_parse_spine handles both old-format and new depth-tagged tokens."""

    def test_new_format_returns_depth_guess_pattern_size(self):
        result = erd_search._parse_spine('3:KHAKI:--y--/33→4:NURDY:---y-/17')
        self.assertEqual(result, [
            (3, 'KHAKI', '--y--', '33'),
            (4, 'NURDY', '---y-', '17'),
        ])

    def test_old_format_returns_none_depth(self):
        result = erd_search._parse_spine('KHAKI:--y--/33→NURDY:---y-/17')
        self.assertEqual(result, [
            (None, 'KHAKI', '--y--', '33'),
            (None, 'NURDY', '---y-', '17'),
        ])

    def test_empty_path_returns_empty_list(self):
        self.assertEqual(erd_search._parse_spine(''), [])
        self.assertEqual(erd_search._parse_spine(None), [])

    def test_size_only_sentinel_new_format(self):
        result = erd_search._parse_spine('3:•')
        self.assertEqual(result, [(3, None, None, '•')])

    def test_size_only_sentinel_old_format(self):
        result = erd_search._parse_spine('•')
        self.assertEqual(result, [(None, None, None, '•')])


class WorkerGuessDepthLabelTest(unittest.TestCase):
    """Compact worker rows render only absolute guess_depth labels."""

    def test_explicit_path_depth_is_already_absolute(self):
        label = erd_search._worker_guess_depth_label(
            3, 0, '4:KHAKI:--y--/33→5:NURDY:---y-/17', 'nurdy')
        self.assertEqual(label, 'd5')

    def test_current_max_deeper_than_parent_is_absolute(self):
        label = erd_search._worker_guess_depth_label(
            3, 5, '', 'nurdy')
        self.assertEqual(label, 'd5')

    def test_old_format_path_is_rebased_from_parent(self):
        label = erd_search._worker_guess_depth_label(
            3, 0, 'KHAKI:--y--/33→NURDY:---y-/17', 'nurdy')
        self.assertEqual(label, 'd5')

    def test_current_candidate_is_next_guess_below_parent(self):
        label = erd_search._worker_guess_depth_label(
            3, 0, '', 'tzars')
        self.assertEqual(label, 'd4')

    def test_unknown_descent_keeps_parent_context(self):
        label = erd_search._worker_guess_depth_label(
            3, 0, '', '')
        self.assertEqual(label, 'd3+?')

    def test_non_deeper_current_max_does_not_render_as_absolute(self):
        label = erd_search._worker_guess_depth_label(
            3, 3, '', '')
        self.assertEqual(label, 'd3+?')


class SplitSectionsTest(unittest.TestCase):
    def test_splits_on_markers(self):
        lines = [
            f'{erd_search._SECTION_MARK}header',
            'Title',
            f'{erd_search._SECTION_MARK}branches',
            'Branches:',
            '  row',
        ]
        sections = erd_search._split_sections(lines)
        # Leading empty section before the first marker, then the two named ones.
        self.assertEqual(sections[0], ('', []))
        self.assertEqual(sections[1], ('header', ['Title']))
        self.assertEqual(sections[2], ('branches', ['Branches:', '  row']))

    def test_no_markers_is_one_unnamed_section(self):
        self.assertEqual(
            erd_search._split_sections(['a', 'b']),
            [('', ['a', 'b'])],
        )


class RedrawSectionDiffTest(unittest.TestCase):
    def setUp(self):
        # _redraw_status keeps prev_sections on the function object; clear it so
        # each test starts from a full (un-highlighted) redraw.
        if hasattr(erd_search._redraw_status, 'prev_sections'):
            del erd_search._redraw_status.prev_sections

    def _render(self, sections):
        """Drive _redraw_status with a stubbed _print_status emitting `sections`.

        `sections` is a list of (name, [lines]); the stub prints the same
        section markers _print_status emits in interactive mode.
        """
        def fake_print_status(args, selected_worker=None, selected_branch=None,
                              interactive=False):
            for name, lines in sections:
                print(f'{erd_search._SECTION_MARK}{name}')
                for line in lines:
                    print(line)

        original = erd_search._print_status
        erd_search._print_status = fake_print_status
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                erd_search._redraw_status(None)
            return buf.getvalue()
        finally:
            erd_search._print_status = original

    def test_first_render_has_no_highlights(self):
        out = self._render([('workers', ['Workers:', '  W1  busy'])])
        self.assertNotIn(RED, out)

    def test_adding_branch_does_not_redden_workers(self):
        worker_lines = ['Workers:', '  W1  busy', '  W2  busy']
        first = [
            ('header', ['ERD_ALL Precache', 'Cache: 1', '']),
            ('branches', ['Branches:', '  AROSE gg---', '']),
            ('workers', worker_lines),
        ]
        self._render(first)

        second = [
            ('header', ['ERD_ALL Precache', 'Cache: 1', '']),
            ('branches', ['Branches:', '  AROSE gg---', '  CRANE -y---', '']),
            ('workers', worker_lines),
        ]
        out = self._render(second)

        # Workers section is unchanged and diffed independently, so every worker
        # line is emitted verbatim (text immediately followed by the clear-EOL
        # code), never wrapped in a red highlight run.
        for line in worker_lines:
            self.assertIn(line + CLEAR_EOL, out)

        # The genuinely new branch row is what gets highlighted.
        self.assertIn(RED, out)
        self.assertNotIn('  CRANE -y---' + CLEAR_EOL, out)

    def test_new_section_does_not_desync_existing_sections(self):
        worker_lines = ['Workers:', '  W1  busy']
        first = [
            ('header', ['ERD_ALL Precache']),
            ('workers', worker_lines),
        ]
        self._render(first)

        # A detail section appears (a worker was selected).  Sections match by
        # name, so the unchanged workers section must stay un-highlighted even
        # though a new section now follows it.
        second = [
            ('header', ['ERD_ALL Precache']),
            ('workers', worker_lines),
            ('detail', ['── Worker 1 spine ──', '  d0  CRANE']),
        ]
        out = self._render(second)
        for line in worker_lines:
            self.assertIn(line + CLEAR_EOL, out)
        self.assertNotIn(RED, out)


if __name__ == '__main__':
    unittest.main()
