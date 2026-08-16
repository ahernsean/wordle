"""Browser contract tests for the self-contained report client."""

from contextlib import contextmanager
import copy
from http.server import ThreadingHTTPServer
from html.parser import HTMLParser
import os
import re
import shutil
import subprocess
import tempfile
from threading import Event, Thread
import unittest
from urllib.parse import unquote

from report_model import ReportSources
from report_server import ServerConfiguration, load_fixtures, make_handler

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


ROOT = os.path.dirname(os.path.dirname(__file__))
FIXTURE_DIRECTORY = os.path.join(ROOT, "tests", "fixtures", "reports")
# The client's DEFAULT_POLL; a re-render lands on every one of these.
CLIENT_POLL_MILLIS = 2000
CLIENT_PATH = os.path.join(ROOT, "report_client.html")
REQUIRE_PLAYWRIGHT_BROWSER = os.environ.get("REQUIRE_PLAYWRIGHT_BROWSER") == "1"


def _preinstalled_chromium():
    """Path to a Chromium shipped under PLAYWRIGHT_BROWSERS_PATH, if any.

    The managed environment provides a Chromium build whose revision need not
    match the installed playwright package, so `playwright install` is neither
    available nor wanted.  Launching that build by path decouples the browser
    tests from the package's bundled-browser revision.
    """
    import glob
    base = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if not base or not os.path.isdir(base):
        return None
    for pattern in (
        "chromium-*/chrome-linux/chrome",
        "chromium_headless_shell-*/chrome-linux/headless_shell",
        "chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell",
    ):
        matches = sorted(glob.glob(os.path.join(base, pattern)))
        if matches:
            return matches[-1]
    return None


def _launch_chromium(playwright):
    """Launch headless Chromium, falling back to a pre-installed build.

    The default launch uses the revision bundled with the playwright package;
    when that is absent (the common case in the managed environment) it retries
    with the Chromium already present under PLAYWRIGHT_BROWSERS_PATH.
    """
    try:
        return playwright.chromium.launch(headless=True)
    except Exception:
        executable = _preinstalled_chromium()
        if executable is None:
            raise
        return playwright.chromium.launch(
            headless=True, executable_path=executable
        )


class _ResourceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.resources = []

    def handle_starttag(self, _tag, attributes):
        for name, value in attributes:
            if name in ("src", "href"):
                self.resources.append(value)


class ReportClientStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CLIENT_PATH, encoding="utf-8") as client_file:
            cls.html = client_file.read()

    def test_client_is_self_contained_and_exports_test_layers(self):
        parser = _ResourceParser()
        parser.feed(self.html)
        self.assertEqual(parser.resources, [])
        for export in ("window.parsePageState", "window.buildAPIURL", "window.applyReport"):
            self.assertIn(export, self.html)

    def test_inline_javascript_parses(self):
        node_path = shutil.which("node")
        if node_path is None:
            self.skipTest("Node.js is unavailable")
        script = re.search(r"<script>(.*)</script>", self.html, re.DOTALL).group(1)
        result = subprocess.run(
            [node_path, "--check", "-"], input=script, text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

@contextmanager
def fixture_server():
    configuration = ServerConfiguration(
        ReportSources("unused", "unused", "unused", "unused"),
        CLIENT_PATH,
        FIXTURE_DIRECTORY,
        load_fixtures(FIXTURE_DIRECTORY),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(configuration))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@unittest.skipIf(
    sync_playwright is None and not REQUIRE_PLAYWRIGHT_BROWSER,
    "Playwright is not installed",
)
class ReportClientBrowserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sync_playwright is None:
            raise RuntimeError(
                "Playwright is required when REQUIRE_PLAYWRIGHT_BROWSER=1"
            )
        cls.server_context = fixture_server()
        cls.base_url = cls.server_context.__enter__()
        cls.playwright = sync_playwright().start()
        try:
            cls.browser = _launch_chromium(cls.playwright)
        except Exception as error:
            cls.playwright.stop()
            cls.server_context.__exit__(None, None, None)
            if REQUIRE_PLAYWRIGHT_BROWSER:
                raise RuntimeError("Playwright Chromium failed to start") from error
            raise unittest.SkipTest(f"Chromium is unavailable: {error}")

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.server_context.__exit__(None, None, None)

    def answer_notch(self, word):
        """Whether a word carries the answer-set notch.

        Reads only whether the marker is drawn, never what color it is: a
        change to the palette must not fail a test about answer-set membership.
        """
        return word.evaluate("""(node) => {
          const last = node.querySelector('.letter:nth-child(5)');
          const after = getComputedStyle(last, '::after');
          return after.content !== 'none' && parseFloat(after.borderTopWidth) > 0;
        }""")

    def setUp(self):
        self.page = self.browser.new_page(viewport={"width": 1200, "height": 800})
        self.page.goto(self.base_url)
        self.page.wait_for_selector("h1")

    def tearDown(self):
        self.page.close()

    def apply_branch_target(self, branch_target):
        self.page.locator("#branch-target-input").fill(branch_target)
        self.page.locator("#apply").click()

    def test_branch_target_inference_has_no_type_chooser(self):
        self.apply_branch_target("CRANE")
        self.page.wait_for_selector("text=word report")
        self.apply_branch_target("CRANE .y..g")
        self.page.wait_for_selector("text=branch report")
        self.assertEqual(self.page.locator("[data-kind]").count(), 6)
        self.assertEqual(self.page.locator("text=Choose word or branch").count(), 0)

    def test_overview_nav_highlight_tracks_root_not_auto_kind(self):
        overview_button = self.page.locator("[data-overview]")
        self.assertEqual(overview_button.get_attribute("aria-current"), "page")
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("text=branch report")
        self.assertEqual(overview_button.get_attribute("aria-current"), "false")
        self.apply_branch_target("")
        self.page.wait_for_selector("text=overview report")
        self.assertEqual(overview_button.get_attribute("aria-current"), "page")

    def test_positional_cache_queue_and_explicit_navigation_urls(self):
        result = self.page.evaluate("""() => ({
          inferredCache: buildAPIURL(parsePageState({search:'?branch_target=CACHE'})),
          inferredQueue: buildAPIURL(parsePageState({search:'?branch_target=QUEUE'})),
          explicitCache: buildAPIURL(parsePageState({search:'?kind=cache'})),
          explicitQueue: buildAPIURL(parsePageState({search:'?kind=queue'})),
          defaultOverview: buildAPIURL(parsePageState({search:''})),
          allOverview: buildAPIURL(parsePageState({search:'?branch_status=all'}))
        })""")
        self.assertTrue(result["inferredCache"].startswith("/api/view?"))
        self.assertTrue(result["inferredQueue"].startswith("/api/view?"))
        self.assertEqual(result["explicitCache"], "/api/view/cache")
        self.assertEqual(result["explicitQueue"], "/api/view/queue")
        self.assertEqual(result["defaultOverview"], "/api/view")
        self.assertEqual(result["allOverview"], "/api/view?branch_status=all")

    def test_tree_branch_status_filter_and_context_node(self):
        self.apply_branch_target("RAISE .....")
        self.page.locator("#layout-tree").click()
        self.page.wait_for_selector("ul.tree > li")
        self.assertIn("tree=1", self.page.url)
        self.assertIn("branch_status=active", self.page.url)
        self.assertGreater(self.page.locator("text=pending").count(), 0)
        base_words = self.page.locator("ul.tree > li")
        self.assertGreater(base_words.count(), 0)
        self.assertEqual(
            base_words.count(),
            self.page.locator("ul.tree > li.word-group").count(),
        )

    def test_tree_groups_patterns_under_their_word_at_every_level(self):
        self.page.locator("[data-kind=queue]").click()
        self.page.locator("#layout-tree").click()
        self.page.wait_for_selector("ul.tree > li.word-group")
        self.assertEqual(self.page.locator("text=Sources and filters").count(), 0)
        group = self.page.locator("ul.tree > li.word-group")
        self.assertEqual(group.count(), 1)
        self.assertIsNone(group.locator("> details").get_attribute("open"))
        self.assertEqual(
            " ".join(group.locator("> details > summary").inner_text().split()),
            "RAISE 1 branch · 0 workers",
        )
        rows = group.locator("> details > .tree-pattern-page > ul.patterns > li")
        self.assertEqual(rows.count(), 1)
        # The group heads the word unplayed; each row beneath draws that same
        # word wearing its own response, so the two read as one object in two
        # states rather than a word and a detached pattern.
        row_summary = rows.locator("> .clickable")
        self.assertEqual(row_summary.locator(".word").count(), 1)
        self.assertEqual(
            row_summary.locator(".word").get_attribute("data-spine"), "RAISE -----")

    def test_tree_root_response_branches_align_as_siblings(self):
        self.page.evaluate("""async () => {
          const report=await (await fetch('/api/view/queue?tree=1')).json();
          const first=report.data.nodes[0];
          const second={
            ...first,
            node_id:'raise:--g--',
            spine:'RAISE --g--',
            branch_reference:'222222222222',
            step:{...first.step,pattern:'--g--'}
          };
          report.data.nodes=[first,second];
          report.data.paging={parent_spine:'',cursor:null,page_size:10,
            returned_group_count:1,total_group_count:1,next_cursor:null};
          applyReport(report,null,parsePageState({search:'?kind=queue&tree=1'}));
        }""")
        group = self.page.locator("ul.tree > li.word-group")
        group.locator("> details > summary").click()
        rows = group.locator("> details > .tree-pattern-page > ul.patterns > li")
        self.assertEqual(rows.count(), 2)
        first_x = rows.nth(0).bounding_box()["x"]
        second_x = rows.nth(1).bounding_box()["x"]
        self.assertEqual(first_x, second_x)

    def test_tree_pages_response_patterns_inside_a_word_group(self):
        self.page.evaluate("""async () => {
          const report=await (await fetch('/api/view/queue?tree=1')).json();
          const template=report.data.nodes[0];
          report.data.nodes=Array.from({length:30},(_,index)=>({
            ...template,node_id:'raise:pattern-'+index,branch_reference:'1111111111'+String(index).padStart(2,'0')
          }));
          report.data.paging={parent_spine:'',cursor:null,page_size:10,returned_group_count:1,total_group_count:1,next_cursor:null};
          applyReport(report,null,parsePageState({search:'?kind=queue&tree=1&limit=10'}));
        }""")
        group = self.page.locator("ul.tree > li.word-group")
        page_size_select = group.locator(".tree-pager .tree-page-size")
        self.assertEqual(page_size_select.input_value(), "10")
        self.assertEqual(
            page_size_select.locator("option").all_text_contents(),
            ["10", "25", "50", "100"],
        )
        self.assertIsNone(group.locator("> details").get_attribute("open"))
        group.locator("> details > summary").click()
        rows = group.locator("> details > .tree-pattern-page > ul.patterns > li")
        self.page.wait_for_selector(
            "ul.tree > li.word-group > details > .tree-pattern-page > ul.patterns > li"
        )
        self.assertEqual(rows.count(), 10, self.page.locator("#report").inner_html())
        self.assertIn("Showing 1–10 of 30 branches", group.inner_text())
        group.locator(".tree-pager button", has_text="Next").click()
        self.assertEqual(rows.count(), 10)
        self.assertIn("Showing 11–20 of 30 branches", group.inner_text())
        page_size_select.select_option("25")
        self.assertIn("limit=25", self.page.url)
        self.assertEqual(self.page.evaluate("__reportClient.getState().limit"), 25)

    def test_entering_tree_uses_ten_items_per_page(self):
        state = self.page.evaluate("""() => {
          __reportClient.setState({...__reportClient.getState(),kind:'queue',limit:25});
          document.querySelector('#layout-tree').click();
          return __reportClient.getState();
        }""")
        self.assertTrue(state["tree"])
        self.assertEqual(state["limit"], 10)

    def test_tree_row_facts_never_split_a_number_from_its_noun(self):
        self.page.locator("[data-kind=queue]").click()
        self.page.locator("#layout-tree").click()
        group = self.page.locator("ul.tree > li.word-group").first
        group.locator("> details > summary").click()
        self.page.wait_for_selector("ul.tree .inline-facts")
        facts = self.page.locator("ul.tree li:not(.word-group) > .clickable .inline-facts").first
        self.assertEqual(
            [" ".join(text.split()) for text in facts.locator("> span").all_inner_texts()],
            ["d1", "active / evaluating", "8 answers", "2 workers", "20/50", "@22222222"],
        )
        self.assertTrue(all(
            style == "nowrap" for style in facts.locator("> span").evaluate_all(
                "spans => spans.map(span => getComputedStyle(span).whiteSpace)"
            )
        ))
        # Facts kept whole are only half the contract: the container must still
        # break *between* them, or a row cannot fit a narrow screen at all.
        self.page.set_viewport_size({"width": 375, "height": 800})
        self.assertGreater(
            len(set(facts.locator("> span").evaluate_all(
                "spans => spans.map(span => span.getBoundingClientRect().top)"
            ))),
            1,
        )
        self.assertLessEqual(*self.page.evaluate(
            "() => [document.documentElement.scrollWidth,"
            " document.documentElement.clientWidth]"
        ))

    def test_word_group_click_builds_full_branch_spine(self):
        self.apply_branch_target("CACHE")
        self.page.wait_for_selector("text=word report")
        self.page.locator("article.card.clickable").first.click()
        self.assertIn("branch_target=CACHE+-----", self.page.url)

    def test_word_view_shows_pending_erd_while_a_group_is_unsolved(self):
        text = self.page.evaluate("""async () => {
          const report=await (await fetch('/api/view?branch_target=SALET')).json();
          report.data.erd_summary={state:'pending',erd:null,max_remaining_depth:null,
            resolved_group_count:2,infeasible_group_count:0,response_group_count:4};
          applyReport(report,null,{...__reportClient.getState(),branch_target:'SALET'});
          return document.querySelector('#report').innerText;
        }""")
        self.assertIn("2 of 4 response groups solved", text)

    def test_word_view_shows_erd_and_max_depth_when_solved(self):
        text = self.page.evaluate("""async () => {
          const report=await (await fetch('/api/view?branch_target=SALET')).json();
          report.data.erd_summary={state:'complete',erd:3.564102564102564,
            max_remaining_depth:6,resolved_group_count:4,infeasible_group_count:0,
            response_group_count:4};
          applyReport(report,null,{...__reportClient.getState(),branch_target:'SALET'});
          return document.querySelector('#report').innerText;
        }""")
        self.assertIn("3.564", text)
        self.assertNotIn("3.564102564102564", text)
        self.assertIn("max remaining depth", text)

    def test_root_progress_panel_loads_after_the_word_report_renders(self):
        # The rollup behind the panel is a seconds-scale scan, so the word
        # report must render without waiting on it: hold the panel's response
        # open and assert the groups are already on screen behind a
        # progress notice.
        released = Event()

        def hold(route):
            # Long enough that only the release below can end the hold.  A
            # timeout short enough to expire while the page is still loading
            # lets the response through before the assertion runs, which reads
            # as "the notice was never shown" on a loaded machine.
            released.wait(timeout=120)
            route.continue_()

        self.page.route("**/api/view/root-progress**", hold)
        try:
            self.apply_branch_target("SALET")
            self.page.wait_for_selector("text=word report")
            self.page.wait_for_selector("article.card.clickable")
            self.assertIn("Computing root progress",
                          self.page.locator("#report").inner_text())
        finally:
            released.set()
        self.page.wait_for_selector("table.root-progress")
        self.page.unroute("**/api/view/root-progress**")

    def test_root_progress_panel_separates_elapsed_from_worker_time(self):
        self.apply_branch_target("SALET")
        self.page.wait_for_selector("table.root-progress")
        headers = self.page.eval_on_selector_all(
            "table.root-progress th", "cells => cells.map(c => c.textContent)")
        # The branch columns are named for the lifecycle phases the report
        # already shows elsewhere: queued -> evaluating -> finalizing -> done.
        self.assertEqual(
            headers,
            ["Response", "State", "Answers", "Done", "Evaluating", "Nodes",
             "Share", "Elapsed", "Worker-time"])

    def test_root_progress_table_keeps_its_scroll_position_across_polls(self):
        # Every poll rebuilds the word report, so the scroller is a fresh
        # element each cycle.  Without carrying its offsets the table snaps
        # back to the top-left every couple of seconds while being read.
        # Phone width, where the table overflows on both axes: a wide viewport
        # exercises only the vertical one.
        self.page.set_viewport_size({"width": 390, "height": 800})
        self.apply_branch_target("SALET")
        self.page.wait_for_selector("table.root-progress")
        scrolled = self.page.evaluate("""() => {
          const box = document.querySelector('.root-progress-scroll');
          box.scrollLeft = box.scrollWidth - box.clientWidth;
          box.scrollTop = box.scrollHeight - box.clientHeight;
          box.dispatchEvent(new Event('scroll'));
          return [box.scrollLeft, box.scrollTop];
        }""")
        self.assertGreater(scrolled[0], 0, "table must overflow horizontally")
        self.assertGreater(scrolled[1], 0, "table must overflow vertically")
        self.page.wait_for_timeout(2 * CLIENT_POLL_MILLIS)
        after = self.page.evaluate("""() => {
          const box = document.querySelector('.root-progress-scroll');
          return [box.scrollLeft, box.scrollTop];
        }""")
        self.assertEqual(after, scrolled)

    def test_root_progress_table_header_stays_visible_while_scrolling(self):
        # The table is longer than a phone screen; a header that scrolls away
        # leaves the columns unidentifiable exactly where the reading happens.
        self.apply_branch_target("SALET")
        self.page.wait_for_selector("table.root-progress")
        overlap = self.page.evaluate("""() => {
          const box = document.querySelector('.root-progress-scroll');
          box.scrollTop = box.scrollHeight - box.clientHeight;
          const header = box.querySelector('thead th').getBoundingClientRect();
          const frame = box.getBoundingClientRect();
          return {scrolled: box.scrollTop, headerTop: header.top,
                  frameTop: frame.top, frameBottom: frame.bottom};
        }""")
        self.assertGreater(overlap["scrolled"], 0,
                           "fixture table must overflow vertically")
        self.assertLess(overlap["headerTop"], overlap["frameBottom"])
        self.assertAlmostEqual(overlap["headerTop"], overlap["frameTop"],
                               delta=2)

    def test_root_progress_headline_shows_with_the_panel_collapsed(self):
        # The remaining-time estimate is the number worth a glance; burying it
        # behind a disclosure puts it out of reach on a phone.
        self.apply_branch_target("SALET")
        self.page.wait_for_selector("table.root-progress")
        self.page.evaluate(
            "() => document.querySelector('details.root-progress-panel')"
            ".open = false")
        headline = self.page.locator(".root-progress-headline").inner_text()
        self.assertIn("remaining", headline)
        self.assertIn("groups working", headline)
        self.assertIn("began", headline)
        self.assertEqual(
            self.page.locator("table.root-progress").count(), 1)

    def test_root_progress_headline_marks_a_warm_up_estimate_provisional(self):
        headline = self.page.evaluate("""() => renderRootProgressHeadline({
          totals:{state_counts:{}}, work_started_at:1785575213,
          estimate:{estimated_seconds:3600, provisional:true},
        }, null).innerText""")

        self.assertIn("~1.0h remaining (provisional)", headline)

    def test_root_progress_headline_states_no_coverage_percentage(self):
        # Cost concentrates so hard that breadth of coverage reads as
        # percent-complete: 98.8% of answers reached beside ~12d remaining
        # after 11 days of work invites exactly the wrong conclusion.
        self.apply_branch_target("SALET")
        self.page.wait_for_selector("table.root-progress")
        headline = self.page.locator(".root-progress-headline").inner_text()
        self.assertNotIn("%", headline)

    def test_root_progress_headline_counts_groups_by_state(self):
        # The State column says where each pattern sits; the headline says how
        # many are in each live state, and nothing the ERD line already shows.
        text = self.page.evaluate("""() => renderRootProgressHeadline({
          totals:{state_counts:{waiting:34, working:45, solved:38}},
          work_started_at:1785575213, estimate:null,
        }, null).innerText""")
        self.assertIn("45 groups working", text)
        self.assertIn("34 waiting", text)
        self.assertNotIn("38", text)

    def test_root_progress_headline_omits_states_with_no_groups(self):
        # A zero is noise on a phone, and its absence already says the state is
        # empty.  PENIS currently has nothing waiting.
        text = self.page.evaluate("""() => renderRootProgressHeadline({
          totals:{state_counts:{working:79, solved:38}},
          work_started_at:1785575213, estimate:null,
        }, null).innerText""")
        self.assertIn("79 groups working", text)
        self.assertNotIn("waiting", text)

    def test_root_progress_table_states_which_patterns_are_finished(self):
        self.apply_branch_target("SALET")
        self.page.wait_for_selector("table.root-progress")
        states = self.page.eval_on_selector_all(
            "table.root-progress tbody tr td:nth-child(2)",
            "cells => cells.map(c => c.textContent)")
        self.assertEqual(set(states), {"waiting", "working", "solved", "loss"})

    def test_root_progress_table_renders_each_response_as_letter_tiles(self):
        self.apply_branch_target("SALET")
        self.page.wait_for_selector("table.root-progress")
        self.page.wait_for_selector("table.root-progress tbody tr")
        table = self.page.locator("table.root-progress")
        self.assertEqual(
            table.locator("thead th").first.inner_text(), "Response")
        responses = table.locator("tbody tr > td:nth-child(1) .word")
        self.assertGreater(responses.count(), 0)
        self.assertTrue(all(
            re.fullmatch(r"[A-Z]{5} [gy-]{5}",
                         response.get_attribute("data-spine"))
            for response in responses.all()))
        self.assertTrue(all(
            response.locator(".letter").count() == 5
            for response in responses.all()))

    def test_root_progress_response_tiles_open_the_branch_report(self):
        self.apply_branch_target("SALET")
        self.page.wait_for_selector("table.root-progress tbody tr")
        tile = self.page.locator("table.root-progress .tile-button").first
        branch_target = tile.get_attribute("aria-label").removeprefix("Open ").removesuffix(" branch report")
        tile.click()
        self.page.wait_for_selector("text=branch report")
        self.assertEqual(
            self.page.locator("#branch-target-input").input_value(), branch_target)

    def test_root_progress_response_tiles_preserve_a_deeper_spine(self):
        def deeper_progress(route):
            response = route.fetch()
            progress = response.json()
            progress["data"]["spine_prefix"] = "SALET ----- CRANE"
            progress["data"]["word"] = "crane"
            route.fulfill(response=response, json=progress)

        self.page.route("**/api/view/root-progress**", deeper_progress)
        try:
            self.apply_branch_target("SALET ----- CRANE")
            self.page.wait_for_selector("table.root-progress tbody tr")
            tile = self.page.locator("table.root-progress .tile-button").first
            branch_target = (
                tile.get_attribute("aria-label")
                .removeprefix("Open ").removesuffix(" branch report"))
            self.assertEqual(branch_target, "SALET ----- CRANE -y---")
            tile.click()
            self.page.wait_for_selector("text=branch report")
            self.assertEqual(
                self.page.locator("#branch-target-input").input_value(),
                branch_target)
        finally:
            self.page.unroute("**/api/view/root-progress**")

    def test_root_progress_failure_renders_once_and_stops_refiring(self):
        # A failed scan that is not held gets re-requested every poll, and the
        # computing notice and the error wrap to different heights -- so the
        # whole page below the panel shifts on every cycle.
        calls = []
        self.page.route(
            "**/api/view/root-progress**",
            lambda route: (calls.append(1), route.fulfill(
                status=400, content_type="application/json",
                body='{"error": {"message": "bad target"}}'))[-1])
        try:
            self.apply_branch_target("SALET")
            self.page.wait_for_selector(".root-progress-host p.error")
            self.page.wait_for_timeout(3 * CLIENT_POLL_MILLIS)
            # One request for the life of the panel.  Re-firing is what makes
            # the panel alternate between the computing notice and the error,
            # and those wrap to different heights on a phone, so every poll
            # shifts the whole page below it.  The height cannot be asserted
            # here -- this fixture answers in milliseconds, so the notice is
            # never on screen when a sample is taken -- but the re-fire that
            # causes it is deterministic.
            self.assertEqual(len(calls), 1, "a held failure must not re-fire")
            self.assertEqual(
                self.page.locator(".root-progress-host p.error").count(), 1)
        finally:
            self.page.unroute("**/api/view/root-progress**")

    def test_root_progress_dates_are_day_month_year(self):
        self.apply_branch_target("SALET")
        self.page.wait_for_selector("table.root-progress")
        headline = self.page.locator(".root-progress-headline").inner_text()
        self.assertRegex(headline, r"began \d{1,2} [A-Z][a-z]{2} \d{4}")
        self.assertNotRegex(headline, r"[A-Z][a-z]{2} \d{1,2}, \d{4}")

    def test_root_progress_omits_the_cumulative_branch_total(self):
        # Carving work into a sub-branch raises the count without any answer
        # being closer to solved, so the absolute total tracks scheduling as
        # much as progress.  Per-pattern values stay, where the comparison
        # between patterns is the point.
        self.apply_branch_target("SALET")
        self.page.wait_for_selector("table.root-progress")
        metrics = self.page.locator(
            "details.root-progress-panel .metrics").inner_text()
        self.assertNotIn("564,186", metrics)
        cells = self.page.eval_on_selector_all(
            "table.root-progress tbody tr td:nth-child(4)",
            "cells => cells.map(c => c.textContent)")
        self.assertIn("538,391", cells)

    def test_root_progress_counts_carry_thousands_separators(self):
        self.apply_branch_target("SALET")
        self.page.wait_for_selector("table.root-progress")
        text = self.page.locator("details.root-progress-panel").inner_text()
        self.assertIn("538,391", text)
        self.assertNotIn("538391", text)
        self.assertIn("8,671", text)

    def test_root_progress_omits_a_request_time_the_queue_cannot_vouch_for(self):
        # The fixture carries no request time: the queue rebuild that restamped
        # source_work destroyed it.  Rendering a placeholder would read as a
        # measurement.
        self.apply_branch_target("SALET")
        self.page.wait_for_selector("table.root-progress")
        text = self.page.locator("details.root-progress-panel").inner_text()
        self.assertNotIn("requested", text)
        self.assertIn("work began", text)
        self.assertIn("epoch", text)

    def test_root_progress_url_carries_only_parameters_the_report_accepts(self):
        # The word view's own display state — group_by, sort, limit, branch
        # filters — is rejected outright by the root-progress report, not
        # ignored, so copying the whole state query 400s the panel. The
        # fixture server ignores query parameters, so this asserts on the URL
        # the client builds; test_report_server pins the server side.
        url = self.page.evaluate("""() => rootProgressURL({
          branch_target:'PENIS', kind:'auto', tree:false,
          group_by:'worker_presence', sort:'size', limit:25,
          branch_status:['active'], branch_phase:['evaluating'],
          minimum_answer_count:5, maximum_answer_count:500, budget:5,
          priority:998, by:'nodes', since_seconds:900, sample_size:100,
          worker_id:'worker-1', finalization_cursor:'abc', tree_cursor:'def',
          answers:true, claims:true, epoch:null,
        })""")
        self.assertEqual(url, "/api/view/root-progress?branch_target=PENIS")

    def test_root_progress_panel_shows_open_groups_as_started(self):
        # A group whose first branch is still open is being worked right now.
        # Dimming it as unstarted would hide the live state.
        self.apply_branch_target("SALET")
        self.page.wait_for_selector("table.root-progress")
        started_without_finalizations = self.page.eval_on_selector_all(
            "table.root-progress tbody tr:not(.dim)",
            """rows => rows
                 .map(r => [...r.cells].map(c => c.textContent))
                 .filter(cells => cells[3] === '0')""")
        self.assertTrue(started_without_finalizations)
        cells = started_without_finalizations[0]
        self.assertNotEqual(cells[4], "—")   # open count is known
        self.assertEqual(cells[8], "—")      # worker-time only exists at finalize

    def test_root_progress_panel_marks_unstarted_groups_without_zero_costs(self):
        # An unstarted group has no cost to report; showing 0 would read as
        # "measured zero" rather than "not begun".
        self.apply_branch_target("SALET")
        self.page.wait_for_selector("table.root-progress")
        dimmed = self.page.eval_on_selector_all(
            "table.root-progress tbody tr.dim td",
            "cells => cells.map(c => c.textContent)")
        self.assertIn("—", dimmed)
        self.assertNotIn("0.0%", dimmed)

    def test_root_progress_scan_runs_once_across_poll_cycles(self):
        # The word report polls every couple of seconds; re-running a
        # multi-second telemetry scan on each cycle would pin a worker's
        # database.  One fetch per target, refreshed only on request.
        self.page.evaluate("window.__rootProgressCalls = 0;")
        self.page.route("**/api/view/root-progress**", lambda route: (
            self.page.evaluate("window.__rootProgressCalls++;"),
            route.continue_())[-1])
        try:
            self.apply_branch_target("SALET")
            self.page.wait_for_selector("table.root-progress")
            self.page.wait_for_timeout(2 * CLIENT_POLL_MILLIS)
            self.assertEqual(
                self.page.evaluate("window.__rootProgressCalls"), 1)
        finally:
            self.page.unroute("**/api/view/root-progress**")

    def test_word_report_card_omits_redundant_and_empty_phase_chips(self):
        # The fixture's four groups are branch_status done/done/done/unqueued
        # with phase complete/complete/complete/None — phase never adds
        # information beyond status here, so no card should show a second chip.
        self.apply_branch_target("SALET")
        self.page.wait_for_selector("text=word report")
        chip_counts = self.page.eval_on_selector_all(
            "article.card.clickable",
            "cards => cards.map(c => c.querySelectorAll('.chip').length)",
        )
        self.assertEqual(chip_counts, [1, 1, 1, 1])
        self.assertNotIn("complete", self.page.locator("#report").inner_text())

    def _apply_grouped_salet_report(self):
        # A live background poll re-renders the root overview it's still
        # tracking (this helper injects report state directly, bypassing
        # __reportClient.setState, so currentState never points at SALET) and
        # would wipe the injected DOM mid-test — block it, and load the
        # fixture directly rather than fetching, so nothing here depends on
        # the poll being blocked before an in-page fetch resolves.
        self.page.route("**/api/view**", lambda route: route.abort())
        report = copy.deepcopy(load_fixtures(FIXTURE_DIRECTORY)["word.json"])
        response_groups = report["data"]["response_groups"]
        report["data"]["response_group_summary"] = {"branch_count": 4,
            "answer_count": 17, "trivial_count": 1, "exact_count": 1,
            "loss_count": 1, "missing_count": 1}
        report["data"]["response_group_groups"] = [
            {"label": "done", "rollup": {"branch_count": 3, "answer_count": 14,
                "trivial_count": 1, "exact_count": 1, "loss_count": 1, "missing_count": 0},
                "rows": [row for row in response_groups if row["branch_status"] == "done"]},
            {"label": "unqueued", "rollup": {"branch_count": 1, "answer_count": 3,
                "trivial_count": 0, "exact_count": 0, "loss_count": 0, "missing_count": 1},
                "rows": [row for row in response_groups if row["branch_status"] == "unqueued"]},
        ]
        return self.page.evaluate("""(report) => {
          window.__groupedTestReport = report;
          applyReport(report, null,
            {...__reportClient.getState(), branch_target:'SALET', group_by:'status'});
          return document.querySelectorAll('details.response-group').length;
        }""", report)

    def test_word_report_renders_groups_with_rollup(self):
        group_count = self._apply_grouped_salet_report()
        self.assertEqual(group_count, 2)
        text = self.page.locator("#report").inner_text()
        # 14 of 17 answers is 82%, rounded from 82.35...
        self.assertIn("3 branches", text)
        self.assertIn("(82%)", text)

    def test_word_report_group_collapse_survives_poll(self):
        self._apply_grouped_salet_report()
        details = self.page.locator("details.response-group").first
        details.locator("summary").click()
        # <details> dispatches "toggle" as a queued task, not synchronously
        # with the click, so the listener that records the collapse hasn't
        # necessarily run yet the instant the click returns — the tree-view
        # collapse test waits for the same reason.
        self.page.wait_for_timeout(100)
        self.assertIsNone(details.get_attribute("open"))
        self.page.evaluate("""() => {
          applyReport(window.__groupedTestReport, window.__groupedTestReport,
            {...__reportClient.getState(), branch_target:'SALET', group_by:'status'});
        }""")
        self.assertIsNone(
            self.page.locator("details.response-group").first.get_attribute("open")
        )

    def test_leaderboard_tab_ranks_openers_and_rounds_erd(self):
        self.page.locator("[data-kind=leaderboard]").click()
        self.page.wait_for_selector("text=Opener leaderboard")
        text = self.page.locator("#report").inner_text()
        self.assertIn("SALET", text)
        self.assertIn("3.564", text)
        self.assertNotIn("3.5643502648", text)
        self.assertIn("Worst-case solve: 6 guesses", text)
        self.assertNotIn("max remaining depth", text)
        self.assertNotIn("expected guesses remaining", text)
        self.assertIn("5 answer groups (more groups = better)", text)
        self.assertIn("Largest: 31 answers (31.0%)", text)
        self.assertIn("Two largest: 55 answers (55.0%)", text)
        self.assertIn("CRANE", text)
        self.assertTrue(self.answer_notch(
            self.page.locator(".card", has_text="CRANE").first.locator(".word")))
        cards = self.page.locator(".grid.leaderboard > .leaderboard-card")
        self.assertEqual(cards.count(), 2)
        first_box = cards.nth(0).bounding_box()
        second_box = cards.nth(1).bounding_box()
        self.assertGreaterEqual(second_box["y"], first_box["y"] + first_box["height"])
        rank = cards.nth(0).locator(".leaderboard-rank")
        self.assertEqual(rank.inner_text(), "#1")
        self.assertNotIn("chip", rank.get_attribute("class").split())
        answer_strip = cards.nth(0).locator(".answer-strip")
        segments = answer_strip.locator(".answer-segment")
        self.assertEqual(segments.count(), 5)
        first_box = segments.nth(0).bounding_box()
        second_box = segments.nth(1).bounding_box()
        last_box = segments.nth(4).bounding_box()
        self.assertAlmostEqual(first_box["width"] / second_box["width"],
                               31 / 24, delta=0.02)
        self.assertAlmostEqual(last_box["width"] / first_box["width"],
                               10 / 31, delta=0.02)
        self.assertIn("31", segments.nth(0).inner_text())

    def test_leaderboard_poll_renders_changed_data(self):
        self.page.locator("[data-kind=leaderboard]").click()
        self.page.wait_for_selector("text=Opener leaderboard")
        self.page.evaluate("""() => {
          const realFetch = window.fetch.bind(window);
          window.fetch = (url, options) => realFetch(url, options).then(async response => {
            if (!String(url).includes('/leaderboard')) return response;
            const report = await response.json();
            report.data.rows[0].erd = 9.876;
            return new Response(JSON.stringify(report), {
              status: 200,
              headers: {'Content-Type': 'application/json'},
            });
          });
        }""")
        self.page.evaluate("async () => { await window.__reportClient.fetchReport(); }")
        self.assertIn("9.876", self.page.locator("#report").inner_text())

    def test_slow_view_switch_shows_a_computing_notice(self):
        # Delay only the leaderboard fetch on the client so the slow-request
        # timer fires on the view switch (the fold can take several seconds).
        # The fetch still resolves, so nothing is left pending — unlike a hung
        # route, which is cancelled at teardown and logs an asyncio error.
        self.page.evaluate(
            "() => { const real = window.fetch.bind(window);"
            " window.fetch = (url, opts) => String(url).includes('/leaderboard')"
            " ? new Promise(r => setTimeout(() => r(real(url, opts)), 4000))"
            " : real(url, opts); }"
        )
        self.page.locator("[data-kind=leaderboard]").click()
        self.page.wait_for_selector("text=Computing leaderboard…", timeout=5000)

    def test_tree_branch_click_opens_detail(self):
        self.page.locator("[data-kind=queue]").click()
        self.page.locator("#layout-tree").click()
        self.page.locator("ul.tree > li.word-group > details > summary").first.click()
        self.page.wait_for_selector(".tree button")
        self.page.locator(".tree button").first.click()
        self.page.wait_for_selector("text=branch report")

    def test_overview_renders_branch_status_phase_and_workers(self):
        text = self.page.locator("#report").inner_text()
        self.assertIn("filesystem used", text)
        self.assertIn("12.5%", text)
        self.assertIn("queue WAL", text)
        self.assertIn("active", text)
        self.assertIn("evaluating", text)
        self.assertIn("finalizing", text)
        self.assertIn("w0", text)
        self.assertIn("w4", text)
        self.assertNotIn("worker-0", text)
        self.assertGreater(self.page.locator(".card.dead").count(), 0)

    def test_queue_view_draws_its_spine_as_colored_squares(self):
        # The queue report's rows carry a spine as flat "WORD pattern WORD
        # pattern" text (see the queue.json fixture), not the structured
        # {word,pattern} steps the overview/branch reports use. cardForBranch
        # must still render tiles for it, not fall back to dim raw text.
        self.page.locator("[data-kind=queue]").click()
        self.page.wait_for_selector(".card .tiles")
        card = self.page.locator(".card", has_text="RAISE").first
        counts = card.evaluate("""(node) => ({
          words: node.querySelectorAll('.word').length,
          letters: node.querySelectorAll('.word > .letter').length,
          dimmed: [...node.querySelectorAll('.dim')]
            .filter(item => item.textContent.includes('RAISE')).length,
        })""")
        self.assertGreater(counts["words"], 0)
        # Every guess on the flat spine became five tiles, none left as text.
        self.assertEqual(counts["letters"], 5 * counts["words"], counts)
        self.assertEqual(counts["dimmed"], 0, counts)

    def test_spine_words_carry_the_answer_notch(self):
        self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          branch.data.branch.spine=[{word:'raise',pattern:'-----',word_is_answer:true}];
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
        }""")
        reached = self.page.locator("section:has-text('Reached via')").first
        self.assertIn("RAISE", reached.inner_text())
        self.assertTrue(self.answer_notch(reached.locator(".word").first))

    def test_worker_on_removed_branch_renders_as_transitioning(self):
        result = self.page.evaluate("""async () => {
          const state=__reportClient.getState();
          const overview=await (await fetch('/api/view')).json();
          const stray={...overview.data.workers[0],worker_id:'worker-9',worker_number:'9',
            branch_key_hex:'ff',branch_reference:'ffffffffffff',state:'transitioning'};
          const next=structuredClone(overview);next.data.workers=[...overview.data.workers,stray];
          applyReport(next,overview,state);
          const card=document.querySelector('[data-identity="worker-9"]');
          return {className:card.className,text:card.innerText};
        }""")
        self.assertIn("transitioning", result["className"])
        self.assertIn("transitioning", result["text"])
        self.assertNotIn("working", result["text"])

    def test_worker_on_active_branch_without_candidate_renders_as_coordinating(self):
        result = self.page.evaluate("""async () => {
          const state=__reportClient.getState();
          const overview=await (await fetch('/api/view')).json();
          const base=overview.data.workers[0];
          const working={...base,worker_id:'worker-8',worker_number:'8',
            state:'working',current_candidate:'crane',current_candidate_is_answer:false};
          const coordinating={...base,worker_id:'worker-9',worker_number:'9',
            state:'coordinating',current_candidate:null,current_candidate_is_answer:false};
          const next=structuredClone(overview);
          next.data.workers=[working,coordinating];
          applyReport(next,overview,state);
          const read=id=>document.querySelector(`[data-identity="${id}"]`).innerText;
          return {working:read('worker-8'),coordinating:read('worker-9')};
        }""")
        self.assertIn("working", result["working"])
        self.assertIn("CRANE", result["working"])
        self.assertIn("coordinating", result["coordinating"])
        self.assertNotIn("working", result["coordinating"])
        self.assertIn("—", result["coordinating"])

    def test_workers_tab_renders_removed_branch_worker_as_transitioning(self):
        result = self.page.evaluate("""async () => {
          const state={...__reportClient.getState(),kind:'workers'};
          const workers=await (await fetch('/api/view/workers')).json();
          const stray={...workers.data.rows[0],worker_id:'worker-9',worker_number:'9',
            branch_key_hex:'ff',branch_reference:'ffffffffffff',is_live:true,
            on_active_branch:false,state:'transitioning'};
          const next=structuredClone(workers);next.data.rows=[...workers.data.rows,stray];
          applyReport(next,workers,state);
          const card=document.querySelector('[data-identity="worker-9"]');
          return {className:card.className,text:card.innerText};
        }""")
        self.assertIn("transitioning", result["className"])
        self.assertIn("transitioning", result["text"])

    def test_worker_card_opens_detail_route_with_live_heartbeat_fields(self):
        card = self.page.locator('[data-identity="worker-0"]')
        self.assertIn("900 nodes", card.inner_text())

        with self.page.expect_response(
            lambda response: (
                "/api/view/workers" in response.url
                and "worker=worker-0" in response.url
            )
        ) as worker_response:
            card.click()
        self.assertTrue(worker_response.value.ok)
        self.page.wait_for_selector("text=Worker w0")
        self.assertIn("kind=workers", self.page.url)
        self.assertIn("worker=worker-0", self.page.url)
        self.page.evaluate("""async () => {
          const report=await (await fetch('/api/view/workers?worker=worker-0')).json();
          report.data.rows[0].answer_count=8;
          report.data.rows[0].current_candidate_is_answer=true;
          report.data.rows[0].best_guess_is_answer=true;
          applyReport(report,null,{...__reportClient.getState(),kind:'workers',worker_id:'worker-0'});
        }""")

        text = self.page.locator("#report").inner_text()
        for expected in (
            "Current work", "current-claim nodes 900", "nodes/s 46",
            "candidate index 7", "claim started", "Search state",
            "candidate NURDY", "best CRANE/2.250 18/8", "cache hits 50",
            "cache misses 10",
            "Open branch",
        ):
            self.assertIn(expected, text)
        self.assertEqual(
            self.page.locator("#report > .section > h2").all_inner_texts(),
            [
                "Identity", "Current branch", "Current work",
                "Search state", "Cumulative worker counters",
            ],
        )
        # The marker moved from the text into the tile, so it is asserted as
        # drawn rather than spelled.
        for word in ("NURDY", "CRANE"):
            self.assertTrue(
                self.answer_notch(
                    self.page.locator(f'.word[data-spine="{word}"]').first),
                word,
            )
        self.page.get_by_role("button", name="Open branch").click()
        self.assertIn("branch_target=", self.page.url)

    def test_worker_card_title_keeps_answer_marker_on_its_line(self):
        self.page.set_viewport_size({"width": 375, "height": 800})
        result = self.page.evaluate("""async () => {
          const report=await (await fetch('/api/view/workers')).json();
          report.data.rows=report.data.rows.slice(0,2).map((worker,index)=>({
            ...worker, worker_id:'worker-'+index, worker_number:String(index),
            state:'working', current_candidate:'enols',
            current_candidate_is_answer:index===1,
          }));
          applyReport(report,null,{...__reportClient.getState(),kind:'workers'});
          return [...document.querySelectorAll('.worker .card-title')].map(title=>({
            titleHeight:title.getBoundingClientRect().height,
            childHeights:[...title.children].map(child=>child.getBoundingClientRect().height),
            childWhiteSpace:[...title.children].map(child=>getComputedStyle(child).whiteSpace),
          }));
        }""")
        # One line: the title is no taller than its tallest child.  Its
        # children have different heights now, so their tops legitimately
        # differ while still sharing the line.
        self.assertTrue(
            all(
                card["titleHeight"] <= max(card["childHeights"]) + 1
                for card in result
            ),
            result,
        )
        self.assertTrue(
            all(height < 30 for card in result for height in card["childHeights"]),
            result,
        )
        self.assertTrue(
            all(
                white_space == "nowrap"
                for card in result for white_space in card["childWhiteSpace"]
            ),
            result,
        )

    def test_worker_card_title_wraps_a_long_state_instead_of_clipping(self):
        self.page.set_viewport_size({"width": 700, "height": 800})
        result = self.page.evaluate("""async () => {
          const report=await (await fetch('/api/view/workers')).json();
          report.data.rows[0]={...report.data.rows[0],state:'transitioning'};
          applyReport(report,null,{...__reportClient.getState(),kind:'workers'});
          const card=document.querySelector('.worker');
          const title=card.querySelector('.card-title');
          const chip=title.querySelector('.chip');
          const word=title.querySelector('.word');
          const cardRect=card.getBoundingClientRect();
          const wordRect=word.getBoundingClientRect();
          return {
            chipText:chip.textContent,
            chipHeight:chip.getBoundingClientRect().height,
            titleHeight:title.getBoundingClientRect().height,
            letters:[...word.querySelectorAll('.letter')].map(letter=>letter.textContent),
            wordFitsWithinCard:wordRect.right<=cardRect.right+1&&wordRect.left>=cardRect.left-1,
          };
        }""")
        # The state text arrives and renders whole -- overflow-wrap:anywhere
        # on the ancestor .card must not be allowed to slice it mid-word.
        self.assertEqual(result["chipText"], "transitioning")
        self.assertLess(result["chipHeight"], 20, result)
        # Too wide for one line at this width, so the candidate tiles drop to
        # a line of their own rather than being clipped or spilling past the
        # card: all five letters render, and the tiles land inside the card.
        self.assertGreater(result["titleHeight"], result["chipHeight"] + 5, result)
        self.assertEqual(result["letters"], ["N", "U", "R", "D", "Y"])
        self.assertTrue(result["wordFitsWithinCard"], result)

    def test_candidate_detail_is_a_bounded_summary_not_per_candidate_rows(self):
        requested = []
        self.page.on("request", lambda request: requested.append(request.url))
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("text=Candidate detail")
        text = self.page.locator("section:has-text('Candidate detail')").inner_text()
        # A summary of provenance and per-worker contribution, never a row per
        # candidate — the branch holds far more claims than a browser can render.
        self.assertIn("12,819 done", text)
        self.assertIn("11,200 evaluated", text)
        self.assertIn("1,500 one-level ERD prunes", text)
        self.assertIn("119 two-level ERD prunes", text)
        self.assertIn("5 in flight", text)
        self.assertIn("w0 6,484", text)
        # Nothing fetches the raw per-candidate list, and no per-candidate rows
        # are rendered.
        self.assertFalse(any("claims=1" in url for url in requested))
        self.assertLess(self.page.locator("section:has-text('Candidate detail') .card").count(), 1)

    def test_branch_ownership_stays_visible_and_names_off_branch_claim_holders(self):
        text = self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          branch.data.workers=[];
          branch.data.branch_ownership={
            live_workers:[],
            claim_holders_off_branch:[{
              worker_id:'worker-5',
              worker_number:'5',
              branch_reference:'eb81eb81eb81',
              branch_context:[{word:'salet',pattern:'---g-',word_is_answer:true}],
            }],
          };
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
          const context=document.querySelector('.status-line .word');
          return {text:document.querySelector('#report').innerText,
                  contextIsAWord:!!context,
                  contextSpine:context&&context.dataset.spine};
        }""")
        self.assertEqual(text["contextIsAWord"], True)
        self.assertEqual(text["contextSpine"], "SALET ---g-")
        text = text["text"]
        self.assertIn("Branch ownership", text)
        self.assertIn("Live workers", text)
        self.assertIn("None", text)
        self.assertIn("Claim holders off-branch", text)
        self.assertIn("w5", text)
        self.assertIn("SALET", text)

    def test_branch_surfaces_missing_best_and_rounds_bounds(self):
        text = self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          branch.data.queue.best_guess=null;branch.data.queue.best_erd=null;
          branch.data.queue.ceiling=2.793103449275866;
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
          return document.querySelector('#report').innerText;
        }""")
        self.assertIn("none yet", text)
        self.assertIn("2.793", text)
        self.assertNotIn("2.793103449275866", text)

    def test_branch_erd_values_show_unreduced_lattice_rationals(self):
        text = self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          branch.data.queue.best_erd=17/8;
          branch.data.queue.ceiling=18/8+1e-9;
          branch.data.cache.best_erd=17/8;
          branch.data.recent_finalizations[0]={
            ...branch.data.recent_finalizations[0],best_guess:'cigar',best_erd:15/8,
          };
          branch.data.recent_finalizations[1]={
            ...branch.data.recent_finalizations[1],ceiling:18/8+1e-9,
          };
          branch.data.cut_reuse_misses[0]={
            ...branch.data.cut_reuse_misses[0],wanted_ceiling:17/8+1e-9,
            available_bound:18/8+1e-9,
          };
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
          return document.querySelector('#report').innerText;
        }""")
        self.assertIn("best CRANE/2.125 17/8", text)
        self.assertIn("best guess CRANE/2.125 17/8", text)
        self.assertIn("2.250 18/8", text)
        self.assertIn("solved CIGAR/1.875 15/8", text)
        self.assertIn("wanted ERD ceiling 2.125 17/8", text)
        self.assertIn("available ERD bound 2.250 18/8", text)
        self.assertNotIn("3/2", text)

    def test_branch_erd_ceiling_accepts_its_explicit_padding(self):
        text = self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          branch.data.branch.answer_count=3209;
          branch.data.queue.ceiling=3/3209+1e-9;
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
          return document.querySelector('#report').innerText;
        }""")
        self.assertIn("ERD ceiling", text)
        self.assertIn("0.001 3/3209", text)

    def test_cut_reuse_facts_wrap_between_complete_metrics(self):
        self.page.set_viewport_size({"width": 375, "height": 800})
        self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          branch.data.cut_reuse_misses[0]={
            ...branch.data.cut_reuse_misses[0],answer_count:19,
            available_bound:47/19+1e-9,available_budget:5,
          };
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
        }""")
        card = self.page.locator("section:has-text('Cut-reuse misses') article.card")
        facts = card.locator(".stat-line > span")
        self.assertIn("available ERD bound 2.474 47/19", facts.all_inner_texts())
        self.assertIn("at budget 5", facts.all_inner_texts())
        self.assertTrue(all(
            style == "nowrap" for style in facts.evaluate_all(
                "spans => spans.map(span => getComputedStyle(span).whiteSpace)"
            )
        ))
        self.assertLessEqual(*card.evaluate(
            "card => [card.scrollWidth, card.clientWidth]"
        ))

    def test_branch_cache_updated_at_is_human_readable(self):
        self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          branch.data.cache.updated_at=990;
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
        }""")
        text = self.page.locator("section.section").filter(
            has=self.page.locator("h2", has_text="Cache")
        ).inner_text()
        self.assertIn("updated", text)
        self.assertNotIn("updated at", text)
        self.assertIn("10s ago", text)
        self.assertNotIn("990", text)

    def test_cache_view_updated_at_is_human_readable(self):
        self.page.evaluate("""async () => {
          const report=await (await fetch('/api/view/cache')).json();
          report.data.recent_rows[0].updated_at=990;
          applyReport(report,null,{...__reportClient.getState(),kind:'cache'});
        }""")
        text = self.page.locator("#report").inner_text()
        self.assertIn("updated", text)
        self.assertNotIn("updated at", text)
        self.assertIn("10s ago", text)
        self.assertNotIn("990", text)

    def test_finalizations_show_both_erd_prune_metrics(self):
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("text=Recent finalizations")
        text = self.page.locator("section:has-text('Recent finalizations')").inner_text()
        self.assertIn("one-level ERD prunes", text)
        self.assertIn("two-level ERD prunes", text)

    def test_branch_queue_shows_both_erd_prune_metrics(self):
        self.apply_branch_target("RAISE .....")
        facts = self.page.locator(
            "section:has-text('Queue') .labeled-facts").first
        text = facts.inner_text()
        self.assertIn("one-level ERD prunes", text)
        self.assertIn("two-level ERD prunes", text)

    def test_ceiling_proven_loss_explains_its_proof(self):
        text = self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          branch.data.recent_finalizations[0]={
            ...branch.data.recent_finalizations[0],outcome:'loss',loss_proof:'ceiling_above_budget',budget:3,ceiling:3.25,
          };
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
          return document.querySelector('#report').innerText;
        }""")
        self.assertIn("ERD lower bound 3.250 26/8 exceeds budget 3", text)

    def test_exact_finalization_shows_recorded_solution(self):
        text = self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          branch.data.recent_finalizations[0]={
            ...branch.data.recent_finalizations[0],
            best_guess:'cigar',
            best_erd:1.875,
          };
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
          return document.querySelector('#report').innerText;
        }""")
        self.assertIn("solved CIGAR/1.875", text)
        self.assertNotIn("solution not recorded", text)

    def test_finalization_spine_and_solved_word_agree_on_the_answer_notch(self):
        # A structured spine (what collect_branch_report now emits) must mark
        # the same word the "solved" line marks -- not one marked and the other
        # bare, which is the inconsistency a flat-string spine caused.
        self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          branch.data.recent_finalizations[0]={
            ...branch.data.recent_finalizations[0],
            spine:[{word:'slate',pattern:'g----',word_is_answer:false},
                   {word:'crane',pattern:'-----',word_is_answer:true}],
            best_guess:'crane',
            best_guess_is_answer:true,
            best_erd:1.875,
          };
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
        }""")
        card = self.page.locator(
            "section:has-text('Recent finalizations') .card"
        ).first
        self.assertIn("SLATE", card.inner_text())
        self.assertIn("solved CRANE/1.875", card.inner_text())
        # The spine's CRANE and the solved CRANE are both marked; SLATE, which
        # is not in the answer set, is not.
        words = card.locator(".word")
        self.assertEqual(
            [self.answer_notch(words.nth(index)) for index in range(words.count())],
            [False, True, True],
        )

    def test_queue_card_omits_the_spine_block_for_a_root_branch(self):
        # A root-level branch (no source word/pattern, no stored spine) now
        # gets spine: [] from the backend instead of spine: null -- an empty
        # array is truthy, so the card must check length, not presence, or
        # it appends a childless, still-margined .tiles div.
        tiles_count = self.page.evaluate("""async () => {
          const report=await (await fetch('/api/view/queue')).json();
          report.data.rows=[{...report.data.rows[0],spine:[]}];
          applyReport(report,null,{...__reportClient.getState(),kind:'queue'});
          return document.querySelectorAll('.card .tiles').length;
        }""")
        self.assertEqual(tiles_count, 0)

    def test_old_epoch_finalizations_are_marked_historical(self):
        self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          const activeEpoch=branch.sources.queue.epoch;
          branch.data.recent_finalizations=[
            {...branch.data.recent_finalizations[0], epoch:activeEpoch},
            {...branch.data.recent_finalizations[1], epoch:activeEpoch-1},
          ];
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
        }""")
        cards = self.page.locator("section:has-text('Recent finalizations') .card")
        self.assertNotIn("Historical", cards.nth(0).inner_text())
        self.assertNotIn("historical", cards.nth(0).get_attribute("class"))
        self.assertIn("Historical", cards.nth(1).inner_text())
        self.assertIn("historical", cards.nth(1).get_attribute("class"))

    def test_branch_identity_is_content_and_spine_is_reached_via(self):
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("section:has-text('Identity')")
        identity = self.page.locator(
            "section:has-text('Identity')"
        ).first.inner_text()
        # Identity is the content branch: its @-reference and word count.
        # Budget is not part of branch_key, so it does not appear here.
        self.assertIn("@1111", identity)
        self.assertIn("words", identity)
        self.assertIn("8", identity)
        self.assertNotIn("budget", identity)
        # The spine is not the identity: absent from the top meta line, shown
        # only under "Reached via".
        meta_spans = self.page.locator(".report-meta > span").all_inner_texts()
        self.assertFalse(any("RAISE" in text for text in meta_spans))
        reached = self.page.locator(
            "section:has-text('Reached via')"
        ).inner_text()
        self.assertIn("RAISE", reached)
        copy_button = self.page.locator(
            "section:has-text('Reached via') button:has-text('Copy spine')"
        )
        self.assertTrue(copy_button.is_visible())
        self.page.evaluate("""() => {
          Object.defineProperty(navigator, 'clipboard', {
            configurable: true, value: undefined,
          });
          if (navigator.clipboard !== undefined)
            throw new Error('clipboard stub did not take effect');
          window.__copiedText = null;
          document.execCommand = command => {
            if (command !== 'copy') return false;
            window.__copiedText = document.activeElement.value;
            return true;
          };
        }""")
        copy_button.click()
        self.page.wait_for_selector(
            "section:has-text('Reached via') button:has-text('Copied')"
        )
        copied_text = self.page.evaluate("() => window.__copiedText")
        self.assertEqual(copied_text, "RAISE -----")

    def test_copy_spine_falls_back_when_clipboard_rejects(self):
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("section:has-text('Reached via')")
        copy_button = self.page.locator(
            "section:has-text('Reached via') button:has-text('Copy spine')"
        )
        self.page.evaluate("""() => {
          Object.defineProperty(navigator, 'clipboard', {
            configurable: true,
            value: { writeText: () => Promise.reject(new Error('denied')) },
          });
          window.__copiedText = null;
          document.execCommand = command => {
            if (command !== 'copy') return false;
            window.__copiedText = document.activeElement.value;
            return true;
          };
        }""")
        copy_button.click()
        self.page.wait_for_selector(
            "section:has-text('Reached via') button:has-text('Copied')"
        )
        copied_text = self.page.evaluate("() => window.__copiedText")
        self.assertEqual(copied_text, "RAISE -----")

    def test_branch_reference_matches_hide_filters_and_view(self):
        self.page.evaluate("""async () => {
          const report=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          applyReport({
            ...report,
            report_kind:'branch_reference_matches',
            data:{branch_reference:'aaaa',candidates:[
              {branch_reference:'aaaa3c008711',answer_count:3,
               answer_preview:['audio','avoid','among'],spine:null},
            ]},
          },null,{...__reportClient.getState(),branch_target:'@aaaa'});
        }""")
        self.assertTrue(self.page.locator("details.filters").is_hidden())

    def test_branch_target_subtitle_returns_in_tree_layout(self):
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("section:has-text('Identity')")
        # Flat branch view: the Identity section names the branch, so the meta
        # subtitle omits the spine.
        meta_spans = self.page.locator(".report-meta > span").all_inner_texts()
        self.assertFalse(any("RAISE" in text for text in meta_spans))
        # Tree layout routes to renderTree and never renders the Identity /
        # "Reached via" sections, so the meta subtitle must name the branch
        # target again — otherwise nothing on the page identifies the branch.
        self.page.locator("#layout-tree").click()
        self.page.wait_for_selector("ul.tree > li")
        meta_spans = self.page.locator(".report-meta > span").all_inner_texts()
        self.assertTrue(any("RAISE" in text for text in meta_spans))

    def test_finalization_spines_and_pager_range_render(self):
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("text=Recent finalizations")
        section = self.page.locator("section:has-text('Recent finalizations')")
        text = section.inner_text()
        # Each finalization discloses the spine that reached this answer set.
        self.assertIn("CRIMP", text)
        self.assertIn("DUCHY", text)
        self.assertIn("WRUNG", text)
        # 3 shown, fewer than a full (default 10) page -> both Prev and Next
        # are dead ends; the total is an exact COUNT(*) at query time.
        self.assertIn("Showing 3 of 3 total", text)
        self.assertTrue(section.locator("button:has-text('Prev')").is_disabled())
        self.assertTrue(section.locator("button:has-text('Next')").is_disabled())

    def test_finalization_pager_enables_next_when_a_full_page_is_returned(self):
        # Next's enabled state is a "did a full page come back" heuristic,
        # not a comparison against finalization_total_count: that count is
        # itself a live, moving target (see the cursor-stability queue-layer
        # tests), so it can't be trusted to say whether more rows exist.
        self.apply_branch_target("RAISE .....")
        self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          const base=branch.data.recent_finalizations[0];
          branch.data.recent_finalizations=Array.from({length:10},(_,i)=>({
            ...base, finalization_id:100+i, recorded_at:1000-i,
          }));
          branch.data.finalization_total_count=17;
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
        }""")
        section = self.page.locator("section:has-text('Recent finalizations')")
        self.assertIn("Showing 10 of 17 total", section.inner_text())
        self.assertTrue(section.locator("button:has-text('Prev')").is_disabled())
        self.assertFalse(section.locator("button:has-text('Next')").is_disabled())

    def test_finalization_pager_next_advances_cursor_and_url(self):
        self.apply_branch_target("RAISE .....")
        self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          const base=branch.data.recent_finalizations[0];
          branch.data.recent_finalizations=Array.from({length:10},(_,i)=>({
            ...base, finalization_id:100+i, recorded_at:1000-i,
          }));
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
        }""")
        section = self.page.locator("section:has-text('Recent finalizations')")
        section.locator("button:has-text('Next')").click()
        # The cursor is anchored to the last row actually shown (recorded_at
        # 991, finalization_id 109), not a row count -- that's what keeps a
        # page stable while the swarm keeps appending finalizations.
        self.assertIn("finalization_cursor=after:991:109", unquote(self.page.url))
        # The click's own state update lands synchronously (asserted above);
        # the re-render only lands once the follow-up fetch resolves.  The
        # fixture server ignores query params, so its response reverts the
        # patched 10 rows back to the 3-row fixture: wait on the whole
        # post-fetch sentence, not a fragment like "of 3" that the patched
        # "Showing 10 of 3 total" already satisfies before the fetch lands.
        self.page.wait_for_function(
            "() => document.querySelector('#report')"
            ".innerText.includes('Showing 3 of 3 total')"
        )
        self.assertFalse(section.locator("button:has-text('Prev')").is_disabled())

    def test_erd_and_bounds_round_in_cache_and_hotspot_views(self):
        result = self.page.evaluate("""async () => {
          const out={};
          const cache=await (await fetch('/api/view/cache')).json();
          cache.data.recent_rows[0].best_erd=2.793103449275866;cache.data.recent_rows[0].ceiling=2.0000000009999996;
          applyReport(cache,null,{...__reportClient.getState(),kind:'cache'});
          out.cache=document.querySelector('#report').innerText;
          const hot=await (await fetch('/api/view/hotspots')).json();
          hot.data.rows=[{row_id:'r1',branch_reference:'abcd1234ef00',best_erd:2.517241380310347,answer_count:33}];
          applyReport(hot,null,{...__reportClient.getState(),kind:'hotspots'});
          out.hotspot=document.querySelector('#report').innerText;
          return out;
        }""")
        self.assertIn("2.793", result["cache"])
        self.assertNotIn("2.793103449275866", result["cache"])
        self.assertIn("2.000", result["cache"])
        self.assertIn("2.517", result["hotspot"])
        self.assertNotIn("2.517241380310347", result["hotspot"])

    def test_finalization_outcomes_and_cut_reuse_are_distinct(self):
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("text=Recent finalizations")
        self.assertEqual(self.page.locator(".outcome-exact").count(), 1)
        self.assertEqual(self.page.locator(".outcome-cut").count(), 2)
        self.assertEqual(self.page.locator(".outcome-loss").count(), 1)
        self.assertIn("Cut-reuse misses", self.page.locator("#report").inner_text())

    def test_cut_reuse_miss_card_shows_the_full_record(self):
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("text=Cut-reuse misses")
        text = self.page.locator("section:has-text('Cut-reuse misses')").inner_text()
        # The fixture row has budget 5, wanted_ceiling null (an exact-required
        # caller), available_bound 2.2 at budget 4, and epoch 4 -- all of it
        # should render, not just available_bound/available_budget.
        self.assertIn("budget 5", text)
        self.assertIn("wanted exact", text)
        self.assertIn("available ERD bound 2.200", text)
        self.assertIn("at budget 4", text)
        self.assertIn("epoch 4", text)

    def test_focused_page_size_select_survives_a_poll_refresh(self):
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("text=Recent finalizations")
        select = self.page.locator("section:has-text('Recent finalizations') select")
        select.click()
        select.evaluate("(el) => el.dataset.testMarker = 'still-here'")
        # A poll refresh while the select has focus must not tear out and
        # rebuild the branch section -- there is no DOM state that lets a
        # native <select>'s open dropdown survive its element being replaced.
        self.page.evaluate("async () => { await window.__reportClient.fetchReport(); }")
        self.assertEqual(select.get_attribute("data-test-marker"), "still-here")
        self.assertTrue(select.evaluate("(el) => el === document.activeElement"))

    def test_focused_tree_summary_survives_a_poll_refresh(self):
        self.page.locator("[data-kind=queue]").click()
        self.page.locator("#layout-tree").click()
        self.page.wait_for_selector(".tree details summary")
        summary = self.page.locator(".tree details summary").first
        summary.click()
        self.assertTrue(summary.evaluate("(el) => el === document.activeElement"))
        summary.evaluate("(el) => el.dataset.testMarker = 'still-here'")
        self.page.evaluate("async () => { await window.__reportClient.fetchReport(); }")
        self.assertEqual(summary.get_attribute("data-test-marker"), "still-here")
        self.assertTrue(summary.evaluate("(el) => el === document.activeElement"))

    def test_navigation_rerenders_even_with_a_stale_focused_control(self):
        # sameView polls skip their visual apply while focus sits on a
        # select/input/textarea/summary (verified above); a real navigation
        # must always render regardless, since replacing the old view is the
        # point.  Drive it through setState directly rather than a Playwright
        # click, which would itself move focus off the summary and mask the
        # bug this guards against.
        self.page.locator("[data-kind=queue]").click()
        self.page.locator("#layout-tree").click()
        self.page.wait_for_selector(".tree details summary")
        summary = self.page.locator(".tree details summary").first
        summary.click()
        self.assertTrue(summary.evaluate("(el) => el === document.activeElement"))
        self.page.evaluate(
            "() => __reportClient.setState({...__reportClient.getState(), kind:'workers', tree:false})"
        )
        self.page.wait_for_selector("text=workers report")

    def test_active_text_selection_survives_a_poll_refresh(self):
        # A poll refresh must not collapse a selection in progress: on
        # iPhone, a long-press selection has to survive until the Copy tap
        # lands, and replaceChildren silently drops it if the report
        # re-renders underneath the user's finger.
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("section:has-text('Reached via') .word")
        section = self.page.locator("section:has-text('Reached via')").first
        section.evaluate("(el) => el.dataset.testMarker = 'still-here'")
        self.page.evaluate("""() => {
          const word = document.querySelector("section .tiles .word");
          const range = document.createRange();
          range.selectNodeContents(word);
          const selection = getSelection();
          selection.removeAllRanges(); selection.addRange(range);
        }""")
        self.page.evaluate("async () => { await window.__reportClient.fetchReport(); }")
        self.assertEqual(section.get_attribute("data-test-marker"), "still-here")
        self.assertFalse(self.page.evaluate("() => getSelection().isCollapsed"))

    def test_finalization_page_size_selector_changes_limit_and_resets_cursor(self):
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("text=Recent finalizations")
        section = self.page.locator("section:has-text('Recent finalizations')")
        section.locator("select").select_option("25")
        self.assertIn("limit=25", self.page.url)
        self.assertNotIn("finalization_cursor=", self.page.url)

    def test_semantic_change_highlights_matching_identity_only(self):
        classes = self.page.evaluate("""async () => {
          const report=await (await fetch('/api/view/queue')).json();
          applyReport(report,null,{...__reportClient.getState(),kind:'queue'});
          const changed=structuredClone(report);changed.data.rows[1].completed_candidate_count++;
          applyReport(changed,report,{...__reportClient.getState(),kind:'queue'});
          return [...document.querySelectorAll('[data-identity]')].map(n=>[n.dataset.identity,n.className]);
        }""")
        highlighted = [identity for identity, class_name in classes if "flash-improved" in class_name]
        self.assertEqual(highlighted, ["02"])

    def test_semantic_change_highlights_tree_branch_cache_and_hotspot_identities(self):
        classes = self.page.evaluate("""async () => {
          const state=__reportClient.getState(),result={};
          const tree=await (await fetch('/api/view?tree=1')).json(),changedTree=structuredClone(tree);
          changedTree.data.nodes[1].completed_candidate_count++;
          applyReport(changedTree,tree,{...state,tree:true});
          result.tree=document.querySelector('[data-identity="raise:-----/alibi:y----"]').className;

          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          branch.data.workers=[{worker_id:'worker-12',updated_at:990,is_live:true,branch_key_hex:'01',branch_reference:'111111111111',current_candidate:'crane',current_max_guess_depth:2,nodes_per_second:10}];
          const changedBranch=structuredClone(branch);changedBranch.data.workers[0].current_candidate='slate';
          applyReport(changedBranch,branch,{...state,branch_target:'RAISE .....'});
          result.branch=document.querySelector('[data-identity="worker-12"]').className;
          const deadBranch=structuredClone(branch);deadBranch.data.workers[0].is_live=false;
          applyReport(deadBranch,branch,{...state,branch_target:'RAISE .....'});
          result.deadWorker=document.querySelector('[data-identity="worker-12"]').className;
          const heartbeatOnly=structuredClone(branch);heartbeatOnly.data.workers[0].updated_at=995;heartbeatOnly.data.workers[0].nodes_per_second=99;
          applyReport(heartbeatOnly,branch,{...state,branch_target:'RAISE .....'});
          result.heartbeatWorker=document.querySelector('[data-identity="worker-12"]').className;
          const switchedBranch=structuredClone(branch);switchedBranch.data.workers[0].branch_key_hex='02';switchedBranch.data.workers[0].branch_reference='222222222222';
          applyReport(switchedBranch,branch,{...state,branch_target:'RAISE .....'});
          result.switchedWorker=document.querySelector('[data-identity="worker-12"]').className;

          const cache=await (await fetch('/api/view/cache')).json(),changedCache=structuredClone(cache);
          cache.data.recent_rows[0].cache_state='missing';changedCache.data.recent_rows[0].cache_state='exact';
          applyReport(changedCache,cache,{...state,kind:'cache'});
          result.cache=document.querySelector('[data-identity="01"]').className;

          const hotspots=await (await fetch('/api/view/hotspots')).json(),changedHotspots=structuredClone(hotspots);
          changedHotspots.data.rows[0].claim_count++;
          applyReport(changedHotspots,hotspots,{...state,kind:'hotspots'});
          result.hotspot=document.querySelector('[data-identity="coordination:20:4"]').className;
          return result;
        }""")
        self.assertIn("flash-improved", classes["tree"])
        self.assertIn("flash-improved", classes["branch"])
        self.assertIn("flash-changed", classes["deadWorker"])
        self.assertNotIn("flash", classes["heartbeatWorker"])
        self.assertIn("flash-improved", classes["switchedWorker"])
        self.assertNotIn("flash-changed", classes["switchedWorker"])
        self.assertIn("flash-improved", classes["cache"])
        self.assertIn("flash-changed", classes["hotspot"])

    def test_generated_time_tick_does_not_flash(self):
        count = self.page.evaluate("""async () => {
          const report=await (await fetch('/api/view/queue')).json();
          const tick=structuredClone(report);tick.generated_at++;
          applyReport(tick,report,{...__reportClient.getState(),kind:'queue'});
          return document.querySelectorAll('.flash-changed,.flash-improved').length;
        }""")
        self.assertEqual(count, 0)

    def test_sticky_order_survives_reordering_and_finalizing_state(self):
        identities = self.page.evaluate("""async () => {
          const report=await (await fetch('/api/view/queue')).json();
          applyReport(report,null,{...__reportClient.getState(),kind:'queue'});
          const reordered=structuredClone(report);reordered.data.rows.reverse();reordered.data.rows[2].branch_phase='finalizing';
          applyReport(reordered,report,{...__reportClient.getState(),kind:'queue'});
          return [...document.querySelectorAll('[data-identity]')].map(n=>n.dataset.identity);
        }""")
        self.assertEqual(identities, ["01", "02", "03", "04"])

    def test_tree_collapse_and_browser_back_survive_poll(self):
        self.page.locator("[data-kind=queue]").click()
        self.page.locator("#layout-tree").click()
        details = self.page.locator(".tree > li.word-group > details").first
        details.locator("summary").click()
        self.assertIsNotNone(details.get_attribute("open"))
        details.locator("summary").click()
        self.assertIsNone(details.get_attribute("open"))
        self.page.evaluate("__reportClient.fetchReport()")
        self.page.wait_for_timeout(100)
        self.assertIsNone(self.page.locator(".tree > li.word-group > details").first.get_attribute("open"))
        self.page.go_back()
        self.assertIn("kind=queue", self.page.url)

    def test_disconnect_preserves_data_and_recovers(self):
        original = self.page.locator("#report").inner_text()
        self.page.route("**/api/view**", lambda route: route.abort())
        self.page.evaluate("__reportClient.fetchReport()")
        self.page.evaluate("__reportClient.fetchReport()")
        self.page.wait_for_timeout(100)
        chip = self.page.locator("#connection")
        self.assertIn("disconnected", chip.get_attribute("class") or "")
        self.assertIn("offline", chip.inner_text())
        self.assertEqual(self.page.locator("#report").inner_text(), original)
        self.page.unroute("**/api/view**")
        self.page.evaluate("__reportClient.fetchReport()")
        self.page.wait_for_timeout(100)
        self.assertNotIn("disconnected", chip.get_attribute("class") or "")
        self.assertEqual(chip.inner_text(), "●")

    def test_poll_retries_a_stalled_request_after_timeout(self):
        result = self.page.evaluate("""async () => {
          const realFetch = window.fetch;
          const report = await (await realFetch('/api/view')).json();
          const realNow = Date.now;
          const resolutions = [];
          let calls = 0;
          let firstRequestAborted = false;
          window.fetch = (_url, options) => new Promise(resolve => {
            calls += 1;
            resolutions.push(resolve);
            if (calls === 1) {
              options.signal.addEventListener('abort', () => {
                firstRequestAborted = true;
              });
            }
          });
          const first = __reportClient.fetchReport();
          await Promise.resolve();
          Date.now = () => realNow() + 61000;
          const second = __reportClient.fetchReport();
          const callsBeforeResolution = calls;
          for (const resolve of resolutions) {
            resolve(new Response(JSON.stringify(report), {
              status: 200,
              headers: {'Content-Type': 'application/json'},
            }));
          }
          await Promise.all([first, second]);
          window.fetch = realFetch;
          Date.now = realNow;
          return {callsBeforeResolution, firstRequestAborted};
        }""")
        self.assertEqual(result["callsBeforeResolution"], 2)
        self.assertTrue(result["firstRequestAborted"])

    def test_branch_view_pins_branch_target_to_its_spine(self):
        # Navigating by a queue reference resolves once; the client then pins
        # the view to the branch's spine so later polls never depend on the
        # reference (which 404s after finalization).
        self.page.goto(self.base_url + "?branch_target=@0123456789ab")
        self.page.wait_for_selector("text=branch report")
        self.page.wait_for_function(
            "() => __reportClient.getState().branch_target === 'raise -----'"
        )
        self.assertIn("branch_target=raise", self.page.url)
        self.assertNotIn("0123456789ab", self.page.url)
        self.assertEqual(self.page.locator("#branch-target-input").input_value(), "raise -----")

    def test_filter_change_leaves_typed_branch_target_untouched(self):
        # A spine typed but not sent with Go stays uncommitted; toggling a
        # filter applies against the current view, never navigating to the typed
        # text nor erasing it from the box.
        self.page.goto(self.base_url)
        self.page.wait_for_selector("h1")
        self.page.locator("#branch-target-input").fill("CRANE")
        self.page.locator("details.filters").evaluate("node => node.open = true")
        self.page.locator('[data-branch-status][value="pending"]').check()
        self.page.wait_for_function(
            "() => __reportClient.getState().branch_status.includes('pending')"
        )
        self.assertEqual(
            self.page.evaluate("() => __reportClient.getState().branch_target"), ""
        )
        self.assertNotIn("CRANE", self.page.url)
        self.assertEqual(self.page.locator("#branch-target-input").input_value(), "CRANE")

    def test_view_fields_appear_only_where_the_kind_can_use_them(self):
        self.page.locator("[data-kind=hotspots]").click()
        self.page.wait_for_function(
            "() => __reportClient.getState().kind === 'hotspots'"
        )
        self.page.locator("details.filters").evaluate("node => node.open = true")
        self.assertFalse(self.page.locator("#by-field").is_hidden())
        self.assertFalse(self.page.locator("#epoch-field").is_hidden())
        self.assertFalse(self.page.locator("#since-seconds-field").is_hidden())
        self.assertTrue(self.page.locator("#sort-field").is_hidden())
        self.page.locator("[data-kind=cache]").click()
        self.page.wait_for_function(
            "() => __reportClient.getState().kind === 'cache'"
        )
        self.assertTrue(self.page.locator("#filters-group").is_hidden())
        self.assertTrue(self.page.locator("#sort-field").is_hidden())

    def test_word_report_sort_only_offers_options_the_server_accepts(self):
        # A word report's sort is restricted server-side to size/workers/
        # priority (validate_report_request); age/nodes/slowest would
        # silently bounce back to "default" with no explanation if offered,
        # and iOS Safari's native picker ignores "hidden" on <option> — so
        # the invalid ones, and the meaningless blank "default", must be
        # absent from the DOM entirely, not just hidden.
        self.apply_branch_target("SALET")
        self.page.wait_for_selector("text=word report")
        values = self.page.eval_on_selector_all(
            "#sort option", "options => options.map(o => o.value)"
        )
        self.assertEqual(values, ["size", "workers", "priority"])
        self.assertEqual(self.page.evaluate("__reportClient.getState().sort"), "size")
        self.page.locator("[data-kind=queue]").click()
        self.page.wait_for_function("() => __reportClient.getState().kind === 'queue'")
        self.assertEqual(
            self.page.eval_on_selector_all("#sort option", "options => options.map(o => o.value)"),
            ["", "age", "size", "workers", "priority", "nodes", "slowest"],
        )

    def test_refresh_popover_toggles_open_and_closed(self):
        self.assertEqual(self.page.locator(".conn-wrap.open").count(), 0)
        self.page.locator("#connection").click()
        self.assertEqual(self.page.locator(".conn-wrap.open").count(), 1)
        self.page.keyboard.press("Escape")
        self.assertEqual(self.page.locator(".conn-wrap.open").count(), 0)

    def test_sources_and_filters_disclosure_stays_open_across_refresh(self):
        self.page.goto(self.base_url + "?kind=cache&branch_target=SALET")
        self.page.wait_for_selector("text=cache report")
        details = self.page.locator(".report-meta details.source-paths")
        details.locator("summary").click()
        self.assertIsNotNone(details.get_attribute("open"))
        self.page.evaluate("""async () => {
          await __reportClient.fetchReport();
        }""")
        self.page.wait_for_selector("text=cache report")
        self.assertIsNotNone(
            self.page.locator(".report-meta details.source-paths").get_attribute("open")
        )

    def test_unresolvable_reference_reports_error_not_a_fake_report(self):
        self.page.route(
            "**/api/view**",
            lambda route: route.fulfill(
                status=404, content_type="application/json",
                body='{"error":{"kind":"not_found","message":"branch reference @dead not found"}}',
            ),
        )
        self.page.evaluate(
            "__reportClient.setState({...__reportClient.getState(),kind:'auto',branch_target:'@dead'})"
        )
        self.page.wait_for_selector("#report .error")
        self.assertIn("not found", self.page.locator("#report .error").inner_text())
        self.page.unroute("**/api/view**")

    def test_overview_card_departure_moves_only_the_nearest_survivor(self):
        result = self.page.evaluate("""async () => {
          const report=await (await fetch('/api/view')).json();
          applyReport(report,null,__reportClient.getState());
          const before=[...document.querySelectorAll('.grid > [data-identity]')].map(node=>node.dataset.identity);
          const reordered=structuredClone(report);
          reordered.data.branches.splice(1,1);
          applyReport(reordered,report,__reportClient.getState());
          const moved=[...document.querySelectorAll('.grid > [data-identity]')].filter(node=>node.getAnimations().length).map(node=>node.dataset.identity);
          const leaveClones=document.querySelectorAll('.leave-layer > *').length;
          return {before,moved,leaveClones};
        }""")
        self.assertGreater(len(result["before"]), 2)
        self.assertEqual(len(result["moved"]), 1)
        self.assertEqual(result["moved"][0], result["before"][2])
        self.assertEqual(result["leaveClones"], 1)

    def test_overview_card_reorder_animates_all_moved_survivors(self):
        result = self.page.evaluate("""async () => {
          const state={...__reportClient.getState(),kind:'queue',sort:'default'};
          const report=await (await fetch('/api/view/queue')).json();
          applyReport(report,null,state);
          const reordered=structuredClone(report);
          reordered.data.rows.reverse();
          applyReport(reordered,report,{...state,sort:'priority'});
          const grid=document.querySelector('.grid');
          return {
            moved:[...grid.querySelectorAll(':scope > [data-identity]')]
              .filter(node=>node.getAnimations().length)
              .map(node=>node.dataset.identity),
            leaveClones:document.querySelectorAll('.leave-layer > *').length,
          };
        }""")
        self.assertGreater(len(result["moved"]), 1)
        self.assertEqual(result["leaveClones"], 0)

    def test_republished_candidates_render_as_summary_not_raw_list(self):
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("text=Bundle and republish")
        text = self.page.locator("#report").inner_text()
        self.assertIn("re-queued", text)
        self.assertIn("candidates re-queued", text)
        self.assertNotIn("7×2", text)

    def test_url_state_round_trips_branch_filters(self):
        state = self.page.evaluate("""() => parsePageState({search:'?kind=queue&branch_status=pending,done&branch_phase=queued,complete&limit=25&sort=nodes&poll=5000'})""")
        self.assertEqual(state["branch_status"], ["pending", "done"])
        self.assertEqual(state["branch_phase"], ["queued", "complete"])
        self.assertEqual(state["limit"], 25)
        self.assertEqual(state["sort"], "nodes")
        self.assertEqual(state["poll"], 5000)

    def test_state_normalization_removes_incompatible_controls(self):
        states = self.page.evaluate("""() => ({
          overview: parsePageState({search:''}),
          all: parsePageState({search:'?branch_status=all'}),
          historical: parsePageState({search:'?kind=hotspots&by=coordination&branch_status=pending&branch_phase=evaluating'}),
          tree: parsePageState({search:'?branch_target=RAISE%20.....&tree=1&claims=1&answers=1'}),
          word: parsePageState({search:'?branch_target=RAISE&sort=nodes'})
        })""")
        self.assertEqual(states["overview"]["branch_status"], ["active"])
        self.assertEqual(states["all"]["branch_status"], [])
        self.assertEqual(states["historical"]["branch_status"], [])
        self.assertEqual(states["historical"]["branch_phase"], [])
        self.assertFalse(states["tree"]["claims"])
        self.assertFalse(states["tree"]["answers"])
        self.assertEqual(states["word"]["sort"], "size")

    def test_word_summary_keeps_unfiltered_totals(self):
        text = self.page.evaluate("""async () => {
          const report=await (await fetch('/api/view?branch_target=QUEUE')).json();
          const done=structuredClone(report);
          done.data.total_rows=4;done.data.matched_rows=3;
          done.data.response_groups=done.data.response_groups.filter(row=>row.branch_status==='done');
          applyReport(done,null,{...__reportClient.getState(),branch_target:'QUEUE',branch_status:['done']});
          return document.querySelector('#report').innerText;
        }""")
        self.assertIn("Shown 3 of 3 matched · 4 total response groups", text)
        self.assertIn("response groups", text)

    def test_selected_detail_remains_visible_after_leaving_parent_filter(self):
        text = self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          branch.data.branch.branch_status='done';branch.data.branch.branch_phase='complete';
          branch.data.queue.branch_status='done';branch.data.queue.branch_phase='complete';
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....',branch_status:['active'],branch_phase:[]});
          return document.querySelector('#report').innerText;
        }""")
        self.assertIn("status done", text)
        self.assertNotIn("no longer matches the parent filter", text)

    def test_cache_renderer_handles_root_word_and_branch_contracts(self):
        identities = self.page.evaluate("""async () => {
          const root=await (await fetch('/api/view/cache')).json();
          applyReport(root,null,{...__reportClient.getState(),kind:'cache'});
          const rootIdentities=[...document.querySelectorAll('[data-identity]')].map(node=>node.dataset.identity);
          const word=structuredClone(root);word.data={summary:{response_group_count:1},rows:[{branch_key_hex:'word-key',branch_reference:'word-ref',cache_state:'missing'}]};
          applyReport(word,null,{...__reportClient.getState(),kind:'cache',branch_target:'RAISE'});
          const wordIdentities=[...document.querySelectorAll('[data-identity]')].map(node=>node.dataset.identity);
          const branch=structuredClone(root);branch.data={branch_key_hex:'branch-key',branch_reference:'branch-ref',cache:{cache_state:'exact',best_guess:'crane',best_erd:2.1}};
          applyReport(branch,null,{...__reportClient.getState(),kind:'cache',branch_target:'RAISE .....'});
          return {rootIdentities,wordIdentities,branchIdentities:[...document.querySelectorAll('[data-identity]')].map(node=>node.dataset.identity)};
        }""")
        self.assertEqual(identities["rootIdentities"], ["01", "02"])
        self.assertEqual(identities["wordIdentities"], ["word-key"])
        self.assertEqual(identities["branchIdentities"], ["branch-key"])

    def test_stale_request_cannot_replace_newer_navigation(self):
        result = self.page.evaluate("""async () => {
          const originalFetch=window.fetch.bind(window);
          const overview=await (await originalFetch('/api/view')).json();
          const branch=await (await originalFetch('/api/view?branch_target=RAISE%20.....')).json();
          let releaseOverview;
          window.fetch=(url)=>url.includes('branch_target=RAISE')
            ? Promise.resolve(new Response(JSON.stringify(branch),{status:200,headers:{'Content-Type':'application/json'}}))
            : new Promise(resolve=>{releaseOverview=()=>resolve(new Response(JSON.stringify(overview),{status:200,headers:{'Content-Type':'application/json'}}));});
          __reportClient.setState({...__reportClient.getState(),kind:'auto',branch_target:''});
          __reportClient.setState({...__reportClient.getState(),kind:'auto',branch_target:'RAISE .....'});
          await new Promise(resolve=>setTimeout(resolve,20));
          releaseOverview();await new Promise(resolve=>setTimeout(resolve,20));
          window.fetch=originalFetch;
          return {heading:document.querySelector('h1').textContent,branch_target:__reportClient.getState().branch_target};
        }""")
        self.assertEqual(result["heading"], "branch report")
        self.assertEqual(result["branch_target"], "RAISE .....")

    def test_malicious_text_is_literal_and_inert(self):
        result = self.page.evaluate("""async () => {
          const report=await (await fetch('/api/view')).json();
          report.sources.queue.error='<img id="owned" src=x onerror="window.owned=1">';
          applyReport(report,null,__reportClient.getState());
          return {owned:window.owned||0,node:!!document.querySelector('#owned'),text:document.querySelector('#report').textContent};
        }""")
        self.assertEqual(result["owned"], 0)
        self.assertFalse(result["node"])
        self.assertIn("<img id=", result["text"])

    def test_tile_colors_match_wordle_palette(self):
        self.apply_branch_target("CACHE")
        self.page.wait_for_selector(".word > .letter")
        colors = self.page.evaluate("""() => {
          const result={};
          for(const name of ['g','y','']){
            const word=document.createElement('span');word.className='word word-sm';
            const node=document.createElement('span');node.className='letter '+name;
            word.append(node);document.body.append(word);
            result[name||'gray']=getComputedStyle(node).backgroundColor;word.remove();}
          return result;
        }""")
        declared = self.page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--green').trim()")
        self.assertEqual(declared, "#6aaa64")
        self.assertEqual(colors["g"], "rgb(106, 170, 100)")
        self.assertEqual(colors["y"], "rgb(201, 180, 88)")
        self.assertEqual(colors["gray"], "rgb(120, 124, 126)")

    def test_notch_is_drawn_in_the_letter_colour(self):
        # The one test with an opinion about notch colour: it must equal the
        # letter colour of the tile it sits on, which is what currentColor buys
        # -- white over a response colour, dark over the unplayed tile.
        tones = self.page.evaluate("""() => {
          const result = {};
          for (const tone of ['g', 'y', '', 'blank']) {
            const word = document.createElement('span');
            word.className = 'word word-sm is-answer';
            for (let index = 0; index < 5; index++) {
              const letter = document.createElement('span');
              letter.className = 'letter ' + tone;
              letter.textContent = 'A';
              word.append(letter);
            }
            document.body.append(word);
            const last = word.querySelector('.letter:nth-child(5)');
            result[tone || 'gray'] = {
              notch: getComputedStyle(last, '::after').borderTopColor,
              letter: getComputedStyle(last).color,
            };
            word.remove();
          }
          return result;
        }""")
        for tone, colors in tones.items():
            self.assertEqual(colors["notch"], colors["letter"], tone)

    def test_notch_renders_on_every_tone_and_size(self):
        drawn = self.page.evaluate("""() => {
          const result = {};
          for (const size of ['word-sm', 'word-md', 'word-lg']) {
            for (const tone of ['g', 'y', '', 'blank']) {
              const word = document.createElement('span');
              word.className = 'word ' + size + ' is-answer';
              for (let index = 0; index < 5; index++) {
                const letter = document.createElement('span');
                letter.className = 'letter ' + tone;
                letter.textContent = 'A';
                word.append(letter);
              }
              document.body.append(word);
              const after = getComputedStyle(
                word.querySelector('.letter:nth-child(5)'), '::after');
              result[size + '/' + (tone || 'gray')] =
                after.content !== 'none' && parseFloat(after.borderTopWidth) > 0;
              word.remove();
            }
          }
          return result;
        }""")
        self.assertEqual(len(drawn), 12)
        for combination, present in drawn.items():
            self.assertTrue(present, combination)

    def test_a_word_that_is_not_an_answer_carries_no_notch(self):
        absent = self.page.evaluate("""() => {
          const word = document.createElement('span');
          word.className = 'word word-sm';
          for (let index = 0; index < 5; index++) {
            const letter = document.createElement('span');
            letter.className = 'letter';
            letter.textContent = 'A';
            word.append(letter);
          }
          document.body.append(word);
          const after = getComputedStyle(
            word.querySelector('.letter:nth-child(5)'), '::after');
          const result = after.content === 'none'
            || parseFloat(after.borderTopWidth) === 0;
          word.remove();
          return result;
        }""")
        self.assertTrue(absent)

    def test_letters_sit_centred_in_their_tiles_at_every_size(self):
        # The browser rounds font ascent and descent to whole pixels and snaps
        # the baseline it derives from them, which left a capital up to a pixel
        # high in its tile.  Compare the baseline the tile actually uses against
        # the one cap-centring wants, computed from the font's real ink.
        offsets = self.page.evaluate("""() => {
          const inkExtents = (weight, family) => {
            const size = 600, box = 1200;
            const canvas = document.createElement('canvas');
            canvas.width = canvas.height = box;
            const context = canvas.getContext('2d', {willReadFrequently: true});
            context.fillStyle = '#fff'; context.fillRect(0, 0, box, box);
            context.fillStyle = '#000';
            context.font = weight + ' ' + size + 'px ' + family;
            context.textBaseline = 'alphabetic';
            context.fillText('HETIL', 40, 900);
            const pixels = context.getImageData(0, 0, box, box).data;
            let top = null, bottom = null;
            for (let y = 0; y < box; y++) {
              let inked = false;
              for (let x = 0; x < box; x++)
                if (pixels[(y * box + x) * 4] < 128) { inked = true; break; }
              if (inked) { if (top === null) top = y; bottom = y; }
            }
            return {ascent: (900 - top) / size, descent: (bottom + 1 - 900) / size};
          };
          const result = {};
          for (const size of ['word-sm', 'word-md', 'word-lg']) {
            const word = document.createElement('span');
            word.className = 'word ' + size;
            const letter = document.createElement('span');
            letter.className = 'letter'; letter.textContent = 'H';
            word.append(letter); document.body.append(word);
            const style = getComputedStyle(letter);
            const ink = inkExtents(style.fontWeight, style.fontFamily);
            const tile = parseFloat(style.height);
            const fontSize = parseFloat(style.fontSize);
            const marker = document.createElement('span');
            marker.style.cssText = 'display:inline-block;width:0;height:0';
            letter.append(marker);
            const actual = marker.getBoundingClientRect().top
              - letter.getBoundingClientRect().top;
            const wanted = (tile + (ink.ascent - ink.descent) * fontSize) / 2;
            word.remove();
            result[size] = actual - wanted;
          }
          return result;
        }""")
        # A baseline can only land on a whole pixel, so half a pixel is the
        # floor; before the correction the smallest tile was a full pixel out.
        for size, offset in offsets.items():
            self.assertLessEqual(abs(offset), 0.5 + 1e-9, f"{size} off by {offset}")

    def test_tile_letters_share_the_baseline_of_adjacent_text(self):
        delta = self.page.evaluate("""() => {
          const line = document.createElement('div');
          line.style.cssText = 'font:13px/1.4 monospace';
          const before = document.createElement('span');
          before.style.cssText = 'display:inline-block;width:0;height:0';
          const word = document.createElement('span');
          word.className = 'word word-sm';
          for (const character of 'CRANE') {
            const letter = document.createElement('span');
            letter.className = 'letter blank'; letter.textContent = character;
            word.append(letter);
          }
          line.append('best ', before, word);
          document.body.append(line);
          const inside = document.createElement('span');
          inside.style.cssText = 'display:inline-block;width:0;height:0';
          word.querySelector('.letter').append(inside);
          const result = inside.getBoundingClientRect().top
            - before.getBoundingClientRect().top;
          line.remove();
          return result;
        }""")
        self.assertLessEqual(abs(delta), 1.0, f"tile baseline off by {delta}px")

    def copy_selection(self, locator):
        """Text the clipboard would receive for a selection of `locator`."""
        return locator.evaluate("""(host) => {
          const range = document.createRange();
          range.selectNodeContents(host);
          const selection = getSelection();
          selection.removeAllRanges();
          selection.addRange(range);
          const data = new DataTransfer();
          const event = new ClipboardEvent(
            'copy', {clipboardData: data, bubbles: true, cancelable: true});
          document.dispatchEvent(event);
          selection.removeAllRanges();
          return {text: data.getData('text/plain'), handled: event.defaultPrevented};
        }""")

    def test_copying_a_word_yields_text_the_branch_target_box_parses(self):
        # The response lives in the tile colors, which no selection can reach,
        # so a copied word carries its pattern as spine text instead.
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("section:has-text('Reached via') .word")
        copied = self.copy_selection(
            self.page.locator("section:has-text('Reached via') .tiles").first)
        self.assertTrue(copied["handled"])
        self.assertEqual(copied["text"], "RAISE -----")
        self.assertNotIn("\n", copied["text"])
        # Round trip: paste it back and the same branch comes up.
        self.page.fill("#branch-target-input", copied["text"])
        self.page.click("#apply")
        self.page.wait_for_selector("section:has-text('Identity')")
        self.assertIn("@1111", self.page.locator(
            "section:has-text('Identity')").first.inner_text())

    def test_copying_a_multi_guess_spine_stays_on_one_line(self):
        self.page.locator("[data-kind=queue]").click()
        self.page.wait_for_selector(".card .tiles .word")
        copied = self.page.evaluate("""() => {
          const spine = [...document.querySelectorAll('.card .tiles')]
            .find(node => node.querySelectorAll('.word').length > 1);
          const range = document.createRange();
          range.selectNodeContents(spine);
          const selection = getSelection();
          selection.removeAllRanges(); selection.addRange(range);
          const data = new DataTransfer();
          document.dispatchEvent(new ClipboardEvent(
            'copy', {clipboardData: data, bubbles: true, cancelable: true}));
          selection.removeAllRanges();
          return data.getData('text/plain');
        }""")
        self.assertNotIn("\n", copied)
        self.assertEqual(copied, "RAISE ----- ALIBI y----")

    def test_copying_part_of_a_word_still_yields_the_whole_guess(self):
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("section:has-text('Reached via') .word")
        copied = self.page.evaluate("""() => {
          const word = document.querySelector("section .tiles .word");
          const range = document.createRange();
          range.setStart(word.children[1].firstChild, 0);
          range.setEnd(word.children[3].firstChild, 1);
          const selection = getSelection();
          selection.removeAllRanges(); selection.addRange(range);
          const data = new DataTransfer();
          document.dispatchEvent(new ClipboardEvent(
            'copy', {clipboardData: data, bubbles: true, cancelable: true}));
          selection.removeAllRanges();
          return data.getData('text/plain');
        }""")
        self.assertEqual(copied, "RAISE -----")

    def test_copying_prose_is_left_to_the_browser(self):
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("section:has-text('Identity')")
        copied = self.copy_selection(
            self.page.locator("section:has-text('Identity') .metrics").first)
        self.assertFalse(copied["handled"])

    def test_copying_prose_that_contains_a_word_is_left_to_the_browser(self):
        # The case with teeth: a fact row holds both prose and a word.  Rewriting
        # it would flatten the row, because textContent carries neither the line
        # breaks block layout makes nor the middots generated content draws.
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("section:has-text('Queue') .labeled-facts")
        facts = self.page.locator("section:has-text('Queue') .labeled-facts").first
        self.assertGreater(facts.locator(".word").count(), 0)
        copied = self.copy_selection(facts)
        self.assertFalse(copied["handled"], copied["text"])

    def test_copying_a_whole_report_keeps_its_line_structure(self):
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("section:has-text('Queue')")
        result = self.page.evaluate("""() => {
          const range = document.createRange();
          range.selectNodeContents(document.getElementById('report'));
          const selection = getSelection();
          selection.removeAllRanges(); selection.addRange(range);
          const native = selection.toString();
          const data = new DataTransfer();
          const event = new ClipboardEvent(
            'copy', {clipboardData: data, bubbles: true, cancelable: true});
          document.dispatchEvent(event);
          selection.removeAllRanges();
          return {handled: event.defaultPrevented,
                  nativeLines: native.split(String.fromCharCode(10)).length};
        }""")
        self.assertFalse(result["handled"])
        self.assertGreater(result["nativeLines"], 20)

    def test_copy_spine_button_and_a_copied_selection_agree(self):
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("section:has-text('Reached via') .word")
        selected = self.copy_selection(
            self.page.locator("section:has-text('Reached via') .tiles").first)
        self.page.evaluate("""() => {
          Object.defineProperty(navigator, 'clipboard', {
            configurable: true, value: undefined,
          });
          window.__copiedText = null;
          document.execCommand = command => {
            if (command !== 'copy') return false;
            window.__copiedText = document.activeElement.value;
            return true;
          };
        }""")
        self.page.click("section:has-text('Reached via') button:has-text('Copy spine')")
        self.page.wait_for_selector(
            "section:has-text('Reached via') button:has-text('Copied')")
        self.assertEqual(
            self.page.evaluate("() => window.__copiedText"), selected["text"])

    def test_rendered_views_mark_answers_on_colored_and_heading_tiles(self):
        # Guards the coverage hole the fixtures used to have: every answer flag
        # sat on a small unplayed tile, so a notch that failed on a response
        # color or at heading size would have gone unseen.
        self.page.goto(self.base_url + "?branch_target=CACHE")
        self.page.wait_for_selector("h2 .word")
        heading = self.page.locator("h2 .word").first
        self.assertTrue(self.answer_notch(heading))
        self.assertIn("word-lg", heading.get_attribute("class"))
        group = self.page.locator(".grid .card .word").first
        self.assertTrue(self.answer_notch(group))
        self.assertGreater(
            group.locator(".letter.g, .letter.y, .letter:not(.blank)").count(), 0)

        self.page.goto(self.base_url)
        self.page.wait_for_selector(".card .tiles .word")
        colored = self.page.locator(
            ".card .tiles .word:has(.letter.y), .card .tiles .word:has(.letter.g)"
        ).first
        self.assertTrue(self.answer_notch(colored))

    def test_unmeasurable_font_gives_up_centring_once_not_every_repaint(self):
        # centerLetters runs from applyReport, so a failure that does not latch
        # rasterises a 1200x1200 canvas on every poll forever.  A browser whose
        # canvas reads back blank (privacy hardening) is the reachable case.
        page = self.browser.new_page(viewport={"width": 1200, "height": 800})
        try:
            page.add_init_script("""
              window.__readbacks = 0;
              const original = CanvasRenderingContext2D.prototype.getImageData;
              CanvasRenderingContext2D.prototype.getImageData = function (...args) {
                window.__readbacks++;
                const data = original.apply(this, args);
                data.data.fill(255);
                return data;
              };
            """)
            page.goto(self.base_url)
            page.wait_for_selector(".card")
            at_load = page.evaluate("() => window.__readbacks")
            self.assertGreater(at_load, 0)
            page.evaluate("""async () => {
              const report = await (await fetch('/api/view')).json();
              for (let index = 0; index < 5; index++)
                applyReport(report, null, __reportClient.getState());
            }""")
            self.assertEqual(page.evaluate("() => window.__readbacks"), at_load)
        finally:
            page.close()

    def test_notch_geometry_scales_with_the_tile(self):
        # Both the notch and the gap holding it off the corner are tile ratios.
        # The inset is derived from --tile, which only exists on the word, so a
        # declaration in the wrong place computes invalid and silently falls
        # back to auto -- visible only by measuring.
        geometry = self.page.evaluate("""() => {
          const result = {};
          for (const size of ['word-sm', 'word-md', 'word-lg']) {
            const word = document.createElement('span');
            word.className = 'word ' + size + ' is-answer';
            for (let index = 0; index < 5; index++) {
              const letter = document.createElement('span');
              letter.className = 'letter'; letter.textContent = 'A';
              word.append(letter);
            }
            document.body.append(word);
            const last = word.querySelector('.letter:nth-child(5)');
            const after = getComputedStyle(last, '::after');
            const tile = parseFloat(getComputedStyle(last).height);
            result[size] = {
              notch: parseFloat(after.borderTopWidth) / tile,
              inset: parseFloat(after.top) / tile,
            };
            word.remove();
          }
          return result;
        }""")
        ratios = {size: value["inset"] for size, value in geometry.items()}
        self.assertEqual(len(set(round(ratio, 4) for ratio in ratios.values())), 1, ratios)
        for size, value in geometry.items():
            self.assertAlmostEqual(value["inset"], 0.07, places=3, msg=size)
            self.assertAlmostEqual(value["notch"], 0.38, places=1, msg=size)

    def test_a_lattice_rational_wraps_instead_of_leaving_the_card(self):
        # An ERD on the lattice trails a rational whose width grows with the
        # answer count, so "best CLART/3.131 1572/502" is wider than a branch
        # card.  Held rigid it left the card; it may wrap after the decimal,
        # but the word must never split from the decimal it earned.
        for width in (390, 480, 700, 1200):
            self.page.set_viewport_size({"width": width, "height": 900})
            self.page.goto(self.base_url)
            self.page.wait_for_selector(".card")
            measured = self.page.evaluate("""async () => {
              const report = await (await fetch('/api/view')).json();
              report.data.branches = [{...report.data.branches[0],
                answer_count: 502, candidate_count: 14855,
                completed_candidate_count: 6363, best_guess: 'clart',
                best_guess_is_answer: false, best_erd: 1572 / 502,
                worker_count: 1, completed_candidate_indexes: [],
                spine: [{word: 'raise', pattern: '-----'}]}];
              applyReport(report, null, {...__reportClient.getState()});
              const card = document.querySelector('.card');
              const fact = [...document.querySelectorAll('.stat-line > span')]
                .find(item => item.textContent.includes('best'));
              const pair = fact.querySelector('.word-erd-pair');
              return {
                text: fact.textContent,
                overflow: fact.getBoundingClientRect().right
                          - card.getBoundingClientRect().right,
                pairLines: pair.getClientRects().length,
                pairText: pair.textContent,
              };
            }""")
            self.assertIn("1572/502", measured["text"])
            self.assertLessEqual(
                measured["overflow"], 0.5,
                f"the ERD left the card at {width}px: {measured}")
            # The pair is one unbroken run: CLART and /3.131 together.
            self.assertEqual(measured["pairLines"], 1, measured)
            self.assertEqual(measured["pairText"], "CLART/3.131", measured)

    def test_no_horizontal_scroll_at_required_widths(self):
        # Every view, not just whichever one setUp left loaded: the tree view
        # reached phone widths overflowing because it was never measured here.
        for path in (
            "", "?kind=queue", "?kind=workers", "?kind=cache", "?kind=hotspots",
            "?kind=leaderboard", "?kind=queue&tree=1", "?branch_target=RAISE+.....",
            "?branch_target=RAISE+.....&tree=1",
            # The word view carries the root-progress table, which is wider
            # than a phone and must scroll inside its own box.
            "?branch_target=SALET",
        ):
            self.page.goto(self.base_url + path)
            self.page.wait_for_selector("h1")
            if "branch_target=SALET" in path:
                self.page.wait_for_selector("table.root-progress")
            for width in (375, 390, 480, 800, 1200):
                with self.subTest(path=path or "overview", width=width):
                    self.page.set_viewport_size({"width": width, "height": 800})
                    measured = self.page.evaluate(
                        "() => ({scroll: document.documentElement.scrollWidth,"
                        " client: document.documentElement.clientWidth})"
                    )
                    self.assertLessEqual(measured["scroll"], measured["client"])

    def test_touch_layout_does_not_cover_controls_with_sticky_header(self):
        context = self.browser.new_context(
            viewport={"width": 800, "height": 600},
            has_touch=True,
            is_mobile=True,
        )
        page = context.new_page()
        try:
            page.goto(self.base_url)
            page.wait_for_selector("h1")
            self.assertEqual(
                page.locator("header").evaluate(
                    "node => getComputedStyle(node).position"
                ),
                "static",
            )
        finally:
            context.close()

    def test_tablet_leaderboard_keeps_legends_for_unlabeled_bar_segments(self):
        context = self.browser.new_context(
            viewport={"width": 834, "height": 1112},
            has_touch=True,
            is_mobile=True,
        )
        page = context.new_page()
        try:
            page.goto(self.base_url + "?kind=leaderboard")
            page.wait_for_selector("text=Opener leaderboard")
            measured = page.evaluate("""async () => {
              const report = await (await fetch('/api/view/leaderboard')).json();
              report.data.rows[0].response_groups = [
                ...Array.from({length: 24}, (_, index) => ({
                  pattern: String(index), answer_count: 1,
                })),
                ...[2, 5, 9, 17, 33, 65, 129].map((answer_count, index) => ({
                  pattern: 'large-' + String(index), answer_count,
                })),
              ];
              applyReport(report, null,
                {...__reportClient.getState(), kind: 'leaderboard'});
              await new Promise(requestAnimationFrame);
              const segment = [...document.querySelectorAll('.response-count-segment')]
                .find(node => node.title.startsWith('2–4 answers'));
              const legend = [...document.querySelectorAll('.response-bucket-legend > span')]
                .find(node => node.textContent === '2–4 words: 1');
              return {
                touch: matchMedia('(hover:none) and (pointer:coarse)').matches,
                fontSize: getComputedStyle(segment).fontSize,
                segmentText: segment.textContent,
                legendText: legend?.textContent,
              };
            }""")
            self.assertTrue(measured["touch"], measured)
            self.assertNotEqual(measured["fontSize"], "0px", measured)
            self.assertEqual(measured["segmentText"], "", measured)
            self.assertEqual(measured["legendText"], "2–4 words: 1", measured)
        finally:
            context.close()

    def test_branch_report_renders_candidate_sweep_with_worker_marker(self):
        result = self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          branch.data.completed_candidate_indexes=[...Array(50).keys()];
          branch.data.workers=[{worker_id:'worker-3',worker_number:'3',updated_at:999,is_live:true,branch_key_hex:'01',branch_reference:'111111111111',candidate_index:75,current_candidate:'crane',current_candidate_is_answer:true}];
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
          const cells=[...document.querySelectorAll('.sweep-cell')];
          return {cellCount:cells.length,firstFill:cells[0].style.getPropertyValue('--fill'),lastFill:cells[cells.length-1].style.getPropertyValue('--fill'),fills:cells.map(cell=>Number.parseInt(cell.style.getPropertyValue('--fill'),10)),markers:[...document.querySelectorAll('.sweep-marker')].map(marker=>marker.dataset.workerNumber)};
        }""")
        self.assertEqual(result["cellCount"], 50)
        self.assertEqual(result["firstFill"], "100%")
        self.assertEqual(result["lastFill"], "0%")
        self.assertEqual(result["markers"], ["3"])
        self.assertFalse(any(85 < fill < 100 for fill in result["fills"]))

    def test_completed_sweep_cells_are_marked_for_desaturated_color(self):
        result = self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          branch.data.completed_candidate_indexes=[...Array(branch.data.queue.candidate_count).keys()];
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
          const cells=[...document.querySelectorAll('.sweep-cell')];
          const rootStyle=getComputedStyle(document.documentElement);
          const probe=document.createElement('span');
          probe.style.color=rootStyle.getPropertyValue('--green-complete');
          document.body.append(probe);
          const greenCompleteColor=getComputedStyle(probe).color;
          probe.style.color=rootStyle.getPropertyValue('--green');
          const greenColor=getComputedStyle(probe).color;
          probe.remove();
          return {
            completeCount:cells.filter(cell=>cell.classList.contains('complete')).length,
            completeColor:getComputedStyle(cells[0]).backgroundColor,
            greenCompleteColor,
            greenColor,
          };
        }""")
        self.assertEqual(result["completeCount"], 50)
        # A completed cell's flat fill is the --green-complete custom
        # property, distinct from the --green used while filling — not a
        # specific shade, so this survives future tuning of either color.
        self.assertEqual(result["completeColor"], result["greenCompleteColor"])
        self.assertNotEqual(result["completeColor"], result["greenColor"])

    def test_overview_branch_cards_render_sweep_with_worker_markers(self):
        result = self.page.evaluate("""() => {
          const cards=[...document.querySelectorAll('#report .grid article.card.clickable')];
          const first=cards.find(card=>card.dataset.identity==='01');
          const cells=[...first.querySelectorAll('.sweep-cell')];
          return {sweepCount:document.querySelectorAll('#report .sweep').length,cellCount:cells.length,workerNumbers:[...first.querySelectorAll('.sweep-marker')].map(marker=>marker.dataset.workerNumber),fullCells:cells.filter(cell=>cell.style.getPropertyValue('--fill')==='100%').length};
        }""")
        self.assertEqual(result["sweepCount"], 3)
        self.assertEqual(result["cellCount"], 50)
        self.assertEqual(result["workerNumbers"], ["0"])
        self.assertGreater(result["fullCells"], 0)

    def test_worker_marker_slides_between_cells_on_progress(self):
        result = self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          const makeWorker=index=>({worker_id:'worker-3',worker_number:'3',updated_at:999,is_live:true,branch_key_hex:'01',branch_reference:'111111111111',candidate_index:index,current_candidate:'crane',current_candidate_is_answer:true});
          branch.data.workers=[makeWorker(10)];
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
          const moved=structuredClone(branch);moved.data.workers=[makeWorker(80)];
          applyReport(moved,branch,{...__reportClient.getState(),branch_target:'RAISE .....'});
          const marker=document.querySelector('.sweep-marker');
          const during=Number.parseFloat(getComputedStyle(marker).left);
          await new Promise(resolve=>setTimeout(resolve,700));
          const settled=Number.parseFloat(getComputedStyle(marker).left);

          const rebranched=structuredClone(branch);rebranched.data.workers=[{...makeWorker(80),branch_key_hex:'02',branch_reference:'222222222222'}];
          applyReport(rebranched,branch,{...__reportClient.getState(),branch_target:'RAISE .....'});
          const jumped=document.querySelector('.sweep-marker');
          const immediate=Number.parseFloat(getComputedStyle(jumped).left);
          const fadingIn=getComputedStyle(jumped).opacity;
          return {during,settled,immediate,fadingIn};
        }""")
        self.assertLess(result["during"], result["settled"] - 10)
        self.assertAlmostEqual(result["immediate"], result["settled"], delta=2)
        self.assertLess(float(result["fadingIn"]), 1)

    def test_worker_markers_carry_a_pointer_triangle(self):
        result = self.page.evaluate("""() => {
          const marker=document.querySelector('.sweep-marker');
          const after=getComputedStyle(marker,'::after');
          return {content:after.content,bottomColor:after.borderBottomColor,bottomWidth:parseFloat(after.borderBottomWidth),leftColor:after.borderLeftColor,textColor:getComputedStyle(document.documentElement).getPropertyValue('--text').trim(),markerCount:document.querySelectorAll('.sweep-marker').length};
        }""")
        self.assertNotEqual(result["content"], "none")
        self.assertGreater(result["bottomWidth"], 0)
        self.assertEqual(result["bottomColor"], "rgb(26, 26, 27)")
        self.assertIn("rgba(0, 0, 0, 0)", result["leftColor"])

    def test_rows_without_claim_indexes_fall_back_to_progress_bar(self):
        self.page.locator("[data-kind=queue]").click()
        self.page.wait_for_selector("text=queue report")
        self.assertEqual(self.page.locator("#report .sweep").count(), 0)
        self.assertGreater(self.page.locator("#report .progress").count(), 0)

    def test_a_spine_is_one_row_or_one_column_never_a_ragged_mix(self):
        # A guess and its response are one object now, so the old hazard (a word
        # wrapping away from its pattern) cannot arise.  What can is a spine
        # laying some guesses across and wrapping the rest onto a second row;
        # the whole spine drops to a column instead, and never clips.
        for width in (320, 375, 390, 600, 1200):
            self.page.set_viewport_size({"width": width, "height": 900})
            for path in ("", "?branch_target=RAISE+.....", "?kind=queue"):
                self.page.goto(self.base_url + path)
                self.page.wait_for_selector(".tiles")
                spines = self.page.evaluate("""() => [...document.querySelectorAll('.tiles')]
                  .map(node => {
                    const words = [...node.querySelectorAll('.word')];
                    const tops = new Set(words.map(
                      word => Math.round(word.getBoundingClientRect().top)));
                    return {stacked: node.classList.contains('stacked'),
                            clipped: node.scrollWidth > node.clientWidth,
                            words: words.length, rows: tops.size};
                  })""")
                self.assertTrue(spines, f"no spine at {width} on {path!r}")
                for spine in spines:
                    where = f"at {width} on {path!r}: {spine}"
                    self.assertFalse(spine["clipped"], f"spine clipped {where}")
                    # One row carrying every guess, or one guess per row.  Any
                    # count between the two is the ragged mix this forbids.
                    self.assertEqual(
                        spine["rows"],
                        spine["words"] if spine["stacked"] else 1,
                        f"ragged spine {where}")

    def test_integers_use_comma_separators(self):
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("text=branch report")
        text = self.page.locator("#report").inner_text()
        self.assertIn("12,000", text)
        self.assertNotIn("12000", text)

    def test_candidates_are_uppercase_and_answers_are_notched(self):
        text = self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          branch.data.workers=[{worker_id:'worker-3',worker_number:'3',updated_at:999,is_live:true,branch_key_hex:'01',branch_reference:'111111111111',current_candidate:'crane',current_candidate_is_answer:true,current_max_guess_depth:2,nodes_per_second:10}];
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
          return document.querySelector('#report').innerText;
        }""")
        self.assertIn("CRANE", text)
        self.assertNotIn("crane", text)
        self.assertTrue(self.answer_notch(
            self.page.locator(".card.worker .word").first))

    def test_hotspot_population_is_humanized_and_hex_is_hidden(self):
        self.page.locator("[data-kind=hotspots]").click()
        self.page.wait_for_selector("text=hotspots report")
        text = self.page.locator("#report").inner_text()
        self.assertIn("recent claim coordination buckets", text)
        self.assertNotIn("recent_claim_coordination_buckets", text)
        self.page.locator("[data-kind=cache]").click()
        self.page.wait_for_selector("text=cache report")
        cache_text = self.page.locator("#report").inner_text()
        self.assertNotIn("branch key hex", cache_text)
        self.assertNotIn("branch_key_hex", cache_text)

    def test_layout_toggle_is_hidden_where_there_is_no_topology(self):
        toggle = self.page.locator("#layout-toggle")
        flat = self.page.locator("#layout-flat")
        tree = self.page.locator("#layout-tree")
        self.assertTrue(toggle.is_visible())
        # Cache, hotspots, and leaderboard have no branch topology, so the
        # layout switch is hidden entirely rather than shown-but-inert.
        for treeless in ("cache", "hotspots", "leaderboard"):
            self.page.locator(f"[data-kind={treeless}]").click()
            self.page.wait_for_selector(f"text={treeless} report")
            self.assertFalse(toggle.is_visible())
        self.page.locator("[data-kind=queue]").click()
        self.page.wait_for_selector("text=queue report")
        self.assertTrue(toggle.is_visible())
        # Flat is the selected layout by default; Tree is not.
        self.assertEqual(flat.get_attribute("aria-pressed"), "true")
        self.assertEqual(tree.get_attribute("aria-pressed"), "false")
        tree.click()
        self.page.wait_for_selector("ul.tree > li")
        self.assertEqual(tree.get_attribute("aria-pressed"), "true")
        self.assertEqual(flat.get_attribute("aria-pressed"), "false")
        self.assertIn("tree=1", self.page.url)
        flat.click()
        self.page.wait_for_timeout(150)
        self.assertEqual(tree.get_attribute("aria-pressed"), "false")
        self.assertEqual(flat.get_attribute("aria-pressed"), "true")
        self.assertNotIn("tree=1", self.page.url)

    def test_review_screenshots_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            for width in (390, 1200):
                for name, path in (
                    ("overview", ""), ("word", "?branch_target=CACHE"),
                    ("branch", "?branch_target=RAISE+....."),
                    ("tree", "?kind=queue&tree=1"),
                ):
                    self.page.set_viewport_size({"width": width, "height": 900})
                    self.page.goto(self.base_url + path)
                    self.page.wait_for_selector("h1")
                    screenshot = os.path.join(directory, f"{name}-{width}.png")
                    self.page.screenshot(path=screenshot, full_page=True)
                    self.assertGreater(os.path.getsize(screenshot), 0)


if __name__ == "__main__":
    unittest.main()
