"""Tests for erd_search.py's start/stop/restart systemd delegation.

These exercise cmd_start/cmd_stop/cmd_restart directly by mocking
erd_search._run_systemctl -- unlike the SWARM.md example-parsing test in
test_report_terminal.py, which mocks the whole command function and never
reaches this logic.
"""

import argparse
import io
import json
import os
import sqlite3
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import ANY, Mock, patch

import erd_search
from erd_queue import ERDQueue


def _args(swarm_only=False, web_only=False):
    return SimpleNamespace(swarm_only=swarm_only, web_only=web_only)


def _services_acted_on(run):
    return {call.args[0] for call in run.call_args_list}


class CmdStartTest(unittest.TestCase):
    def test_starts_both_services_by_default(self):
        with (
            patch.object(erd_search, '_run_systemctl', return_value=0) as run,
            patch.object(erd_search, '_run_journalctl', return_value=0) as journal,
            patch.object(erd_search.time, 'time', return_value=123.5),
        ):
            erd_search.cmd_start(_args())
        run.assert_any_call(erd_search._SYSTEMD_SERVICE, 'start')
        run.assert_any_call(erd_search._REPORT_SERVER_SYSTEMD_SERVICE, 'start')
        run.assert_any_call(
            erd_search._SYSTEMD_SERVICE, 'status', '--no-pager', '--lines=0')
        run.assert_any_call(
            erd_search._REPORT_SERVER_SYSTEMD_SERVICE, 'status', '--no-pager',
            '--lines=0')
        journal.assert_any_call(erd_search._SYSTEMD_SERVICE, 123.5)
        journal.assert_any_call(erd_search._REPORT_SERVER_SYSTEMD_SERVICE, 123.5)

    def test_journal_diagnostics_start_at_command_timestamp(self):
        completed = SimpleNamespace(returncode=0)
        with patch('subprocess.run', return_value=completed) as run:
            self.assertEqual(
                erd_search._run_journalctl('wordle-erd', 123.5), 0)
        run.assert_called_once_with(
            ['journalctl', '--user', '--unit', 'wordle-erd', '--since',
             '@123.5', '--no-pager', '--full'],
            capture_output=False)

    def test_swarm_only_skips_report_server(self):
        with (
            patch.object(erd_search, '_run_systemctl', return_value=0) as run,
            patch.object(erd_search, '_run_journalctl', return_value=0) as journal,
        ):
            erd_search.cmd_start(_args(swarm_only=True))
        self.assertEqual(_services_acted_on(run), {erd_search._SYSTEMD_SERVICE})
        journal.assert_called_once_with(erd_search._SYSTEMD_SERVICE, ANY)

    def test_web_only_skips_supervisor(self):
        with (
            patch.object(erd_search, '_run_systemctl', return_value=0) as run,
            patch.object(erd_search, '_run_journalctl', return_value=0) as journal,
        ):
            erd_search.cmd_start(_args(web_only=True))
        self.assertEqual(
            _services_acted_on(run),
            {erd_search._REPORT_SERVER_SYSTEMD_SERVICE},
        )
        journal.assert_called_once_with(
            erd_search._REPORT_SERVER_SYSTEMD_SERVICE, ANY)

    def test_aborts_before_report_server_when_supervisor_fails(self):
        def fake(service, action, *extra):
            return 1 if service == erd_search._SYSTEMD_SERVICE else 0

        with (
            patch.object(erd_search, '_run_systemctl', side_effect=fake) as run,
            patch('sys.stderr', io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as raised:
                erd_search.cmd_start(_args())
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(_services_acted_on(run), {erd_search._SYSTEMD_SERVICE})

    def test_exits_nonzero_when_report_server_fails(self):
        def fake(service, action, *extra):
            if (service, action) == (
                erd_search._REPORT_SERVER_SYSTEMD_SERVICE, 'start'
            ):
                return 3
            return 0

        with (
            patch.object(erd_search, '_run_systemctl', side_effect=fake) as run,
            patch('sys.stderr', io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as raised:
                erd_search.cmd_start(_args())
        self.assertEqual(raised.exception.code, 3)
        # Status is still printed for both services for diagnostics.
        run.assert_any_call(
            erd_search._SYSTEMD_SERVICE, 'status', '--no-pager', '--lines=0')
        run.assert_any_call(
            erd_search._REPORT_SERVER_SYSTEMD_SERVICE, 'status', '--no-pager',
            '--lines=0')


class CmdStopTest(unittest.TestCase):
    def test_stops_both_services_by_default(self):
        with (
            patch.object(erd_search, '_run_systemctl', return_value=0) as run,
            patch('sys.stdout', io.StringIO()) as out,
        ):
            erd_search.cmd_stop(_args())
        run.assert_any_call(erd_search._SYSTEMD_SERVICE, 'stop')
        run.assert_any_call(erd_search._REPORT_SERVER_SYSTEMD_SERVICE, 'stop')
        self.assertIn('Supervisor and report server stopped.', out.getvalue())

    def test_swarm_only_skips_report_server_and_message(self):
        with (
            patch.object(erd_search, '_run_systemctl', return_value=0) as run,
            patch('sys.stdout', io.StringIO()) as out,
        ):
            erd_search.cmd_stop(_args(swarm_only=True))
        self.assertEqual(_services_acted_on(run), {erd_search._SYSTEMD_SERVICE})
        self.assertIn('Supervisor stopped.', out.getvalue())
        self.assertNotIn('report server', out.getvalue())

    def test_web_only_skips_supervisor_and_message(self):
        with (
            patch.object(erd_search, '_run_systemctl', return_value=0) as run,
            patch('sys.stdout', io.StringIO()) as out,
        ):
            erd_search.cmd_stop(_args(web_only=True))
        self.assertEqual(
            _services_acted_on(run),
            {erd_search._REPORT_SERVER_SYSTEMD_SERVICE},
        )
        self.assertIn('Report server stopped.', out.getvalue())
        self.assertNotIn('Supervisor', out.getvalue())

    def test_attempts_report_server_even_if_supervisor_stop_fails(self):
        def fake(service, action, *extra):
            return 1 if service == erd_search._SYSTEMD_SERVICE else 0

        with (
            patch.object(erd_search, '_run_systemctl', side_effect=fake) as run,
            patch('sys.stderr', io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as raised:
                erd_search.cmd_stop(_args())
        self.assertEqual(raised.exception.code, 1)
        run.assert_any_call(erd_search._REPORT_SERVER_SYSTEMD_SERVICE, 'stop')


class CmdRestartTest(unittest.TestCase):
    def test_restarts_both_services_by_default(self):
        with patch.object(erd_search, '_run_systemctl', return_value=0) as run:
            erd_search.cmd_restart(_args())
        run.assert_any_call(erd_search._SYSTEMD_SERVICE, 'restart')
        run.assert_any_call(erd_search._REPORT_SERVER_SYSTEMD_SERVICE, 'restart')

    def test_swarm_only_skips_report_server(self):
        with patch.object(erd_search, '_run_systemctl', return_value=0) as run:
            erd_search.cmd_restart(_args(swarm_only=True))
        self.assertEqual(_services_acted_on(run), {erd_search._SYSTEMD_SERVICE})

    def test_web_only_skips_supervisor(self):
        with patch.object(erd_search, '_run_systemctl', return_value=0) as run:
            erd_search.cmd_restart(_args(web_only=True))
        self.assertEqual(
            _services_acted_on(run),
            {erd_search._REPORT_SERVER_SYSTEMD_SERVICE},
        )

    def test_aborts_before_report_server_when_supervisor_fails(self):
        def fake(service, action, *extra):
            return 1 if service == erd_search._SYSTEMD_SERVICE else 0

        with (
            patch.object(erd_search, '_run_systemctl', side_effect=fake) as run,
            patch('sys.stderr', io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as raised:
                erd_search.cmd_restart(_args())
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(_services_acted_on(run), {erd_search._SYSTEMD_SERVICE})


class StartStopRestartCliParsingTest(unittest.TestCase):
    """--swarm-only / --web-only must parse on all three lifecycle subcommands
    and be mutually exclusive."""

    def test_scope_flags_parse_and_dispatch(self):
        for subcommand, handler_name in (
            ('start', 'cmd_start'),
            ('stop', 'cmd_stop'),
            ('restart', 'cmd_restart'),
        ):
            for flag, attribute in (
                ('--swarm-only', 'swarm_only'),
                ('--web-only', 'web_only'),
            ):
                with self.subTest(subcommand=subcommand, flag=flag):
                    with (
                        patch('sys.argv',
                              ['erd_search.py', subcommand, flag]),
                        patch.object(erd_search, handler_name) as handler,
                    ):
                        erd_search.main()
                    args = handler.call_args.args[0]
                    self.assertTrue(getattr(args, attribute))

    def test_scope_flags_are_mutually_exclusive(self):
        for subcommand in ('start', 'stop', 'restart'):
            with self.subTest(subcommand=subcommand):
                with (
                    patch('sys.argv', ['erd_search.py', subcommand,
                                       '--swarm-only', '--web-only']),
                    patch('sys.stderr', io.StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    erd_search.main()
                self.assertEqual(raised.exception.code, 2)


class EpochCommandTest(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.queue_path = os.path.join(
            self._temporary_directory.name, 'queue.sqlite3'
        )

    def test_show_prints_active_epoch_metadata(self):
        queue = ERDQueue(self.queue_path)
        queue.set_epoch(8, label='claiming-regime', git_sha='abcdef12',
                        notes='fresh measurements')
        queue.close()

        with (
            patch('sys.stdout', io.StringIO()) as output,
            patch.object(erd_search, 'ERDQueue', wraps=ERDQueue) as open_queue,
        ):
            erd_search.cmd_epoch_show(SimpleNamespace(queue=self.queue_path))

        open_queue.assert_called_once_with(
            self.queue_path, initialize_schema=False)

        self.assertEqual(json.loads(output.getvalue()), {
            'epoch': 8,
            'label': 'claiming-regime',
            'git_sha': 'abcdef12',
            'started_at': ANY,
            'notes': 'fresh measurements',
        })

    def test_set_refuses_when_worker_heartbeat_is_live(self):
        queue = ERDQueue(self.queue_path)
        queue.heartbeat('worker-1', 1, None, 0, int(time.time()), 0)
        queue.close()
        args = SimpleNamespace(
            queue=self.queue_path, epoch=8, label=None, git_sha=None,
            notes=None, force=False,
        )

        with patch('sys.stderr', io.StringIO()) as error:
            self.assertEqual(erd_search.cmd_epoch_set(args), 1)

        self.assertIn('worker-1', error.getvalue())
        queue = ERDQueue(self.queue_path)
        self.assertEqual(queue.epoch, 0)
        queue.close()

    def test_set_force_changes_epoch_and_uses_current_git_sha(self):
        queue = ERDQueue(self.queue_path)
        queue.heartbeat('worker-1', 1, None, 0, int(time.time()), 0)
        queue.close()
        args = SimpleNamespace(
            queue=self.queue_path, epoch=8, label='claiming-regime',
            git_sha=None, notes='fresh measurements', force=True,
        )

        with (
            patch.object(erd_search, '_current_git_sha', return_value='abcdef12'),
            patch('sys.stdout', io.StringIO()) as output,
        ):
            self.assertEqual(erd_search.cmd_epoch_set(args), 0)

        self.assertEqual(json.loads(output.getvalue())['git_sha'], 'abcdef12')

    def test_epoch_cli_parses_nested_queue_path_and_dispatches(self):
        with (
            patch('sys.argv', [
                'erd_search.py', 'epoch', 'set', '8', '--queue', 'custom.sqlite3',
            ]),
            patch.object(erd_search, 'cmd_epoch_set') as handler,
        ):
            erd_search.main()
        self.assertEqual(handler.call_args.args[0].queue, 'custom.sqlite3')


class OperatorHelperTest(unittest.TestCase):
    def test_view_watch_interval_and_filters_reject_invalid_values(self):
        self.assertEqual(erd_search._view_watch_interval('0.2'), 0.2)
        with self.assertRaises(argparse.ArgumentTypeError):
            erd_search._view_watch_interval('0.1')
        status_values = erd_search.BRANCH_STATUSES[:2]
        self.assertEqual(
            erd_search._branch_status_filter(','.join(status_values)),
            status_values,
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            erd_search._branch_worker_status_filter('unknown')

    def test_service_scope_noun_matches_the_selected_service(self):
        self.assertEqual(
            erd_search._service_scope_noun(_args()),
            'Supervisor and report server',
        )
        self.assertEqual(
            erd_search._service_scope_noun(_args(swarm_only=True)),
            'Supervisor',
        )
        self.assertEqual(
            erd_search._service_scope_noun(_args(web_only=True)),
            'Report server',
        )

    def test_system_commands_return_subprocess_exit_codes(self):
        completed_process = SimpleNamespace(returncode=7)
        with patch('subprocess.run', return_value=completed_process) as run:
            self.assertEqual(
                erd_search._run_systemctl('wordle-erd', 'start'), 7)
            self.assertEqual(erd_search._run_journalctl('wordle-erd', 12.5), 7)
        self.assertEqual(run.call_count, 2)

    def test_current_git_sha_handles_missing_git_and_output(self):
        with patch('subprocess.run', side_effect=OSError):
            self.assertIsNone(erd_search._current_git_sha())
        with patch('subprocess.run', return_value=SimpleNamespace(stdout='abc123\n')):
            self.assertEqual(erd_search._current_git_sha(), 'abc123')

    def test_view_and_nested_path_helpers_dispatch_only_in_scope(self):
        arguments = SimpleNamespace(report_kind='queue')
        with patch('report_terminal.run_view') as run_view:
            erd_search.cmd_view(arguments)
        run_view.assert_called_once_with(arguments)

        queue_arguments = SimpleNamespace(cmd='queue', queue_path='queue.sqlite3')
        erd_search._normalize_queue_cli_args(queue_arguments)
        self.assertEqual(queue_arguments.queue, 'queue.sqlite3')

        epoch_arguments = SimpleNamespace(cmd='epoch', queue_path=None)
        erd_search._normalize_epoch_cli_args(epoch_arguments)
        self.assertEqual(epoch_arguments.queue, erd_search.DEFAULT_QUEUE)

    def test_hint_cache_preflight_reports_refusal_and_closes_artifact(self):
        arguments = SimpleNamespace(hint_cache='history.sqlite3', cache='live.sqlite3')
        with (
            patch.object(erd_search, 'load_word_list', return_value=['crane']),
            patch.object(erd_search, 'open_hint_cache',
                         side_effect=erd_search.HintCacheError('bad artifact')),
        ):
            self.assertFalse(erd_search._hint_cache_is_usable(arguments))

        hint = SimpleNamespace(db_path='history.sqlite3', namespace_branch_count=12,
                               close=Mock())
        with (
            patch.object(erd_search, 'load_word_list', return_value=['crane']),
            patch.object(erd_search, 'open_hint_cache', return_value=hint),
        ):
            self.assertTrue(erd_search._hint_cache_is_usable(arguments))
        hint.close.assert_called_once_with()

    def test_checkpoint_cache_reports_success_and_sqlite_failure(self):
        connection = Mock()
        with patch('sqlite3.connect', return_value=connection):
            erd_search._checkpoint_cache_on_start('cache.sqlite3')
        connection.execute.assert_any_call('PRAGMA wal_checkpoint(TRUNCATE)')
        connection.close.assert_called_once_with()

        with patch('sqlite3.connect', side_effect=sqlite3.Error('locked')):
            erd_search._checkpoint_cache_on_start('cache.sqlite3')

    def test_reap_and_stack_dump_handle_live_and_dead_workers(self):
        queue = Mock()
        queue.reclaim_claims_of_worker.return_value = 3
        erd_search._reap_worker(queue, 2)
        queue.reclaim_claims_of_worker.assert_called_once_with('worker-2')
        queue.clear_heartbeat.assert_called_once_with('worker-2')

        live_process = Mock(pid=12)
        live_process.is_alive.return_value = True
        dead_process = Mock(pid=13)
        dead_process.is_alive.return_value = False
        with patch.object(erd_search.os, 'kill') as kill:
            erd_search._dump_worker_stacks({2: (live_process, 0), 3: (dead_process, 0)})
        kill.assert_called_once_with(12, erd_search.signal.SIGUSR1)

    def test_source_work_invariant_check_accepts_empty_and_logs_rows(self):
        queue = Mock()
        queue.check_source_work_invariants.return_value = []
        erd_search._check_source_work_invariants(queue)

        queue.check_source_work_invariants.return_value = ['branch 17 has no owner']
        with self.assertLogs(erd_search.logger, 'WARNING') as logs:
            erd_search._check_source_work_invariants(queue)
        self.assertIn('branch 17 has no owner', logs.output[-1])


class QueueOperatorCommandTest(unittest.TestCase):
    def test_clear_obeys_confirmation_and_closes_queue(self):
        queue = Mock()
        queue.counts_by_status.return_value = {"pending": 2, "done": 3}
        queue.branches_in_progress.return_value = [object()]
        arguments = SimpleNamespace(queue="queue.sqlite3", yes=False)
        with patch.object(erd_search, "ERDQueue", return_value=queue), \
             patch("builtins.input", return_value="n"):
            erd_search.cmd_queue_clear(arguments)
        queue.clear.assert_not_called()
        queue.close.assert_called_once_with()

        queue.reset_mock()
        with patch.object(erd_search, "ERDQueue", return_value=queue):
            erd_search.cmd_queue_clear(SimpleNamespace(queue="queue.sqlite3", yes=True))
        queue.clear.assert_called_once_with()
        queue.close.assert_called_once_with()

    def test_disk_stop_commands_cover_empty_existing_and_warning_latches(self):
        queue = Mock()
        queue.disk_stop.return_value = None
        with patch.object(erd_search, "ERDQueue", return_value=queue):
            erd_search.cmd_queue_clear_disk_stop(SimpleNamespace(queue="queue.sqlite3"))
        queue.clear_disk_stop.assert_not_called()

        queue.reset_mock()
        queue.disk_stop.return_value = {"reason": "full"}
        with patch.object(erd_search, "ERDQueue", return_value=queue), \
             patch.object(erd_search, "disk_stats", return_value={"used_fraction": 1.0}):
            erd_search.cmd_queue_clear_disk_stop(SimpleNamespace(queue="queue.sqlite3"))
        queue.clear_disk_stop.assert_called_once_with()

        queue.reset_mock()
        queue.set_disk_stop_if_unset.return_value = False
        queue.disk_stop.return_value = {"reason": "operator"}
        with patch.object(erd_search, "ERDQueue", return_value=queue):
            erd_search.cmd_queue_set_disk_stop(SimpleNamespace(queue="queue.sqlite3", reason="operator"))
        queue.close.assert_called_once_with()

    def test_priority_opener_path_reports_success_and_validation_error(self):
        queue = Mock()
        queue.set_ownerless_active_priority.return_value = 2
        arguments = SimpleNamespace(queue="queue.sqlite3", opener_word="raise", priority=7)
        with patch.object(erd_search, "ERDQueue", return_value=queue):
            erd_search.cmd_queue_priority(arguments)
        queue.set_ownerless_active_priority.assert_called_once_with("raise", 7)

        queue.reset_mock()
        queue.set_ownerless_active_priority.side_effect = ValueError("no opener")
        with patch.object(erd_search, "ERDQueue", return_value=queue):
            erd_search.cmd_queue_priority(arguments)
        queue.close.assert_called_once_with()

    def test_remove_and_branch_priority_cover_active_and_missing_branch_paths(self):
        score_cache = Mock()
        response_cache = Mock()
        response_cache.group_words.return_value = {0: ["cigar", "rebut"]}
        queue = Mock()
        arguments = SimpleNamespace(
            queue="queue.sqlite3", cache="cache.sqlite3", word="raise",
            pattern="-----", force=False, priority=9, opener_word=None,
        )
        with (
            patch.object(erd_search, "load_word_list", return_value=["cigar", "rebut"]),
            patch.object(erd_search, "ScoreCache", return_value=score_cache),
            patch.object(erd_search, "ResponseCache", return_value=response_cache),
            patch.object(erd_search, "ERDQueue", return_value=queue),
        ):
            queue.get_active_branch.return_value = object()
            erd_search.cmd_queue_remove(arguments)
        queue.remove_pending.assert_not_called()
        queue.close.assert_called_once_with()

        queue.reset_mock()
        queue.get_active_branch.return_value = object()
        arguments.force = True
        with (
            patch.object(erd_search, "load_word_list", return_value=["cigar", "rebut"]),
            patch.object(erd_search, "ScoreCache", return_value=score_cache),
            patch.object(erd_search, "ResponseCache", return_value=response_cache),
            patch.object(erd_search, "ERDQueue", return_value=queue),
        ):
            erd_search.cmd_queue_remove(arguments)
        queue.cancel_active_branch.assert_called_once()

        queue.reset_mock()
        queue.set_priority.return_value = False
        with (
            patch.object(erd_search, "load_word_list", return_value=["cigar", "rebut"]),
            patch.object(erd_search, "ScoreCache", return_value=score_cache),
            patch.object(erd_search, "ResponseCache", return_value=response_cache),
            patch.object(erd_search, "ERDQueue", return_value=queue),
        ):
            erd_search.cmd_queue_priority(arguments)
        queue.set_priority.assert_called_once()

    def test_opener_priority_handles_invalid_complete_ambiguous_and_updated(self):
        arguments = SimpleNamespace(
            queue="queue.sqlite3", word="raise", priority=5, opener_work_id=None,
        )
        queue = Mock()
        with patch.object(erd_search, "check_source_priority_range", side_effect=ValueError("out of range")):
            erd_search.cmd_queue_source_priority(arguments)

        queue.source_work_rows.return_value = []
        queue.source_work_candidates.return_value = []
        with patch.object(erd_search, "ERDQueue", return_value=queue):
            erd_search.cmd_queue_source_priority(arguments)

        rows = [
            {"source_work_id": 2, "source_word": "raise", "requested_at": 0,
             "requested_priority": 1, "state": "queued", "root_count": 1, "branch_count": 2},
            {"source_work_id": 3, "source_word": "raise", "requested_at": 0,
             "requested_priority": 2, "state": "queued", "root_count": 1, "branch_count": 2},
        ]
        queue.reset_mock()
        queue.source_work_rows.return_value = rows
        queue.source_work_candidates.return_value = rows
        with patch.object(erd_search, "ERDQueue", return_value=queue):
            erd_search.cmd_queue_source_priority(arguments)
        queue.set_source_work_priority.assert_not_called()

        queue.reset_mock()
        queue.source_work_rows.return_value = rows[:1]
        queue.source_work_candidates.return_value = rows[:1]
        queue.set_source_work_priority.return_value = True
        with patch.object(erd_search, "ERDQueue", return_value=queue):
            erd_search.cmd_queue_source_priority(arguments)
        queue.set_source_work_priority.assert_called_once_with(2, 5)
if __name__ == '__main__':
    unittest.main()
