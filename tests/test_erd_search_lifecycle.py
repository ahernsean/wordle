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
        self.assertEqual(
            erd_search._branch_status_filter('active,pending'),
            ('active', 'pending'),
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            erd_search._branch_phase_filter('unknown')

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
if __name__ == '__main__':
    unittest.main()
