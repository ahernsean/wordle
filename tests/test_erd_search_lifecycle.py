"""Tests for erd_search.py's start/stop/restart systemd delegation.

These exercise cmd_start/cmd_stop/cmd_restart directly by mocking
erd_search._run_systemctl -- unlike the SWARM.md example-parsing test in
test_report_terminal.py, which mocks the whole command function and never
reaches this logic.
"""

import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import erd_search


def _args(swarm_only=False, web_only=False):
    return SimpleNamespace(swarm_only=swarm_only, web_only=web_only)


def _services_acted_on(run):
    return {call.args[0] for call in run.call_args_list}


class CmdStartTest(unittest.TestCase):
    def test_starts_both_services_by_default(self):
        with patch.object(erd_search, '_run_systemctl', return_value=0) as run:
            erd_search.cmd_start(_args())
        run.assert_any_call(erd_search._SYSTEMD_SERVICE, 'start')
        run.assert_any_call(erd_search._REPORT_SERVER_SYSTEMD_SERVICE, 'start')
        run.assert_any_call(erd_search._SYSTEMD_SERVICE, 'status', '--no-pager')
        run.assert_any_call(
            erd_search._REPORT_SERVER_SYSTEMD_SERVICE, 'status', '--no-pager')

    def test_swarm_only_skips_report_server(self):
        with patch.object(erd_search, '_run_systemctl', return_value=0) as run:
            erd_search.cmd_start(_args(swarm_only=True))
        self.assertEqual(_services_acted_on(run), {erd_search._SYSTEMD_SERVICE})

    def test_web_only_skips_supervisor(self):
        with patch.object(erd_search, '_run_systemctl', return_value=0) as run:
            erd_search.cmd_start(_args(web_only=True))
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
        run.assert_any_call(erd_search._SYSTEMD_SERVICE, 'status', '--no-pager')
        run.assert_any_call(
            erd_search._REPORT_SERVER_SYSTEMD_SERVICE, 'status', '--no-pager')


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


if __name__ == '__main__':
    unittest.main()
