"""Browser contract tests for the self-contained report client."""

from contextlib import contextmanager
import copy
import itertools
import json
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
from tests.webkit_container import WebKitContainerUnavailable, start_webkit_server

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
# The WebKit suite additionally needs a container runtime (podman/docker) and
# a network pull of the Microsoft Playwright image, so unlike the Chromium
# suite above it stays off by default even when playwright is installed.
RUN_WEBKIT_CONTAINER_TESTS = os.environ.get("RUN_WEBKIT_CONTAINER_TESTS") == "1"
REQUIRE_WEBKIT_CONTAINER_TESTS = (
    os.environ.get("REQUIRE_WEBKIT_CONTAINER_TESTS") == "1"
)

# Longer than the grid choreography, so a report applied before this has elapsed
# is one the transition has already finished with.
GRID_TRANSITION_MILLIS = 1700

# Helpers for the grid-transition tests, prepended to their page scripts.
#
# parkThePoll matters more than it looks: these tests sample across more than a
# poll interval, and a refresh landing mid-sample re-renders the report
# underneath the measurement.  The result reads as a phantom jump in the numbers.
GRID_SCRIPT_HELPERS = """
  const parkThePoll=()=>{window.fetch=()=>new Promise(()=>{});};
  // Parked before the fixture is fetched, not after: a poll issued while the
  // fixture request is still open would land mid-measurement and re-render the
  // report underneath it.  The fixture is then fetched through the saved real
  // fetch, and kept, because parking took the global one with it.
  const overviewReport=async()=>{
    if(!window.__overviewFixture){
      const realFetch=window.fetch.bind(window);
      parkThePoll();
      window.__overviewFixture=await (await realFetch('/api/view')).json();
    }
    return window.__overviewFixture;
  };
  const settled=()=>new Promise(resolve=>setTimeout(resolve,%d));
  const namedBranches=(base,names)=>{
    const report=structuredClone(base);
    report.data.branches=names.map((name,index)=>{
      const row=structuredClone(base.data.branches[index%%base.data.branches.length]);
      row.branch_key_hex=name;row.branch_reference=name;return row;
    });
    return report;
  };
  // Every card's (row, column) read off the laid-out grid rather than from the
  // DOM order, so the assertions describe what is on screen.
  const gridCells=()=>{
    const grid=document.querySelector('.grid'),box=grid.getBoundingClientRect();
    const cards=[...grid.querySelectorAll(':scope > [data-identity]')];
    const axis=values=>[...new Set(values)].sort((left,right)=>left-right);
    const offset=node=>{
      const rect=node.getBoundingClientRect();
      return [Math.round(rect.top-box.top),Math.round(rect.left-box.left)];
    };
    const tops=axis(cards.map(node=>offset(node)[0])),lefts=axis(cards.map(node=>offset(node)[1]));
    return Object.fromEntries(cards.map(node=>
      [node.dataset.identity,[tops.indexOf(offset(node)[0]),lefts.indexOf(offset(node)[1])]]));
  };
  const workersHeading=()=>[...document.querySelectorAll('#report h2')]
    .find(node=>node.textContent==='Workers');
  const sampleEveryFrame=async collect=>{
    const samples=[],start=performance.now();
    await new Promise(resolve=>{
      const tick=()=>{
        samples.push([Math.round(performance.now()-start),...collect()]);
        if(performance.now()-start<%d)requestAnimationFrame(tick);else resolve();
      };
      requestAnimationFrame(tick);
    });
    return samples;
  };
""" % (GRID_TRANSITION_MILLIS, GRID_TRANSITION_MILLIS)


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
        self.assertEqual(self.page.locator("[data-kind]").count(), 7)
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

    def test_completed_root_progress_replaces_an_inapplicable_estimate(self):
        def completed_progress(route):
            response = route.fetch()
            progress = response.json()
            progress["data"]["estimate"] = None
            progress["data"]["completed_at"] = 1785575213
            route.fulfill(response=response, json=progress)

        self.page.route("**/api/view/root-progress**", completed_progress)
        try:
            self.apply_branch_target("SALET")
            self.page.wait_for_selector("table.root-progress")
            text = self.page.locator(".root-progress-panel").inner_text()
        finally:
            self.page.unroute("**/api/view/root-progress**")

        self.assertIn("Completed", text)
        self.assertNotIn("No completion estimate", text)

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
        # empty.  SCOPE currently has nothing waiting.
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
            self.assertRegex(branch_target, r"^SALET ----- CRANE [gy-]{5}$")
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
        self.assertIn("active epoch", text)
        self.assertIn("no recorded telemetry", text)

    def test_root_progress_url_carries_only_parameters_the_report_accepts(self):
        # The word view's own display state — group_by, sort, limit, branch
        # filters — is rejected outright by the root-progress report, not
        # ignored, so copying the whole state query 400s the panel. The
        # fixture server ignores query parameters, so this asserts on the URL
        # the client builds; test_report_server pins the server side.
        url = self.page.evaluate("""() => rootProgressURL({
          branch_target:'SCOPE', kind:'auto', tree:false,
          group_by:'worker_presence', sort:'size', limit:25,
          branch_status:['active'], branch_phase:['evaluating'],
          minimum_answer_count:5, maximum_answer_count:500, budget:5,
          priority:998, by:'nodes', since_seconds:900, sample_size:100,
          worker_id:'worker-1', finalization_cursor:'abc', tree_cursor:'def',
          answers:true, claims:true, epoch:null,
        })""")
        self.assertEqual(url, "/api/view/root-progress?branch_target=SCOPE")

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

    def test_word_report_draws_the_group_breakdown_above_the_detail(self):
        self.apply_branch_target("SALET")
        self.page.wait_for_selector(".response-group-breakdown")
        breakdown = self.page.locator(".response-group-breakdown")
        self.assertIn("4 answer groups (more groups = better)",
                      breakdown.inner_text())
        self.assertEqual(breakdown.locator(".response-visual-key").count(), 1)
        self.assertEqual(breakdown.locator(".response-count-track").count(), 1)
        segments = breakdown.locator(".answer-segment")
        self.assertEqual(segments.count(), 4)
        widths = [segments.nth(index).bounding_box()["width"]
                  for index in range(4)]
        self.assertAlmostEqual(widths[0] / widths[1], 8 / 5, delta=0.02)
        self.assertAlmostEqual(widths[3] / widths[0], 1 / 8, delta=0.02)
        # High up: above the root-progress panel and the per-group cards.
        top = breakdown.bounding_box()["y"]
        self.assertLess(
            top, self.page.locator(".root-progress-panel").bounding_box()["y"])
        self.assertLess(
            top,
            self.page.locator(".grid > [data-identity]").first.bounding_box()["y"])

    def _menu(self):
        return self.page.locator(".group-menu")

    def test_word_report_breakdown_group_opens_a_menu_naming_the_group(self):
        # Which group a tap landed on is a guess until the menu names it, so
        # the title draws the guess and the response it caught.
        self.apply_branch_target("SALET")
        self.page.wait_for_selector(".response-group-breakdown .answer-segment")
        self.page.locator(
            ".response-group-breakdown .answer-segment").first.click()
        self.page.wait_for_selector(".group-menu")
        tiles = self._menu().locator(".group-menu-title .word")
        self.assertEqual(tiles.count(), 1)
        self.assertEqual(tiles.get_attribute("data-spine"), "QUEUE -----")
        self.assertEqual(tiles.locator(".letter").count(), 5)
        facts = self._menu().locator(".group-menu-facts").inner_text()
        self.assertIn("8 answers", facts)
        self.assertIn("solved", facts)

    def test_word_report_breakdown_menu_opens_the_branch_report(self):
        self.apply_branch_target("SALET")
        self.page.wait_for_selector(".response-group-breakdown .answer-segment")
        self.page.locator(
            ".response-group-breakdown .answer-segment").first.click()
        self.page.wait_for_selector(".group-menu")
        self._menu().locator("button", has_text="Open branch report").click()
        self.page.wait_for_selector("text=branch report")
        self.assertEqual(
            self.page.locator("#branch-target-input").input_value(), "SALET -----")
        self.assertEqual(self._menu().count(), 0)

    def test_word_report_breakdown_menu_dismisses_without_navigating(self):
        # A tap that caught the wrong sliver costs one dismissal, not a trip to
        # a branch the reader never chose.
        self.apply_branch_target("SALET")
        self.page.wait_for_selector(".response-group-breakdown .answer-segment")
        segments = self.page.locator(".response-group-breakdown .answer-segment")
        segments.first.click()
        self.page.wait_for_selector(".group-menu")
        self.page.locator("h2").first.click()
        self.page.wait_for_selector(".group-menu", state="detached")
        self.assertEqual(
            self.page.locator("#branch-target-input").input_value(), "SALET")
        segments.first.click()
        self.page.wait_for_selector(".group-menu")
        self.page.keyboard.press("Escape")
        self.page.wait_for_selector(".group-menu", state="detached")
        self.assertEqual(
            self.page.locator("#branch-target-input").input_value(), "SALET")

    def test_word_report_group_menu_survives_a_poll(self):
        # The menu lives inside the report, so a refresh that re-rendered would
        # delete it out from under the finger reaching for it.  Every other
        # menu test acts within milliseconds and never sees a poll land.
        self.apply_branch_target("SALET")
        self.page.wait_for_selector(".response-group-breakdown .answer-segment")
        self.page.locator(
            ".response-group-breakdown .answer-segment").first.click()
        self.page.wait_for_selector(".group-menu")
        self.page.wait_for_timeout(int(CLIENT_POLL_MILLIS * 2.5))
        self.assertEqual(self._menu().count(), 1,
                         "a poll took the menu away")
        # Still wired to its group, not merely still on screen.
        self._menu().locator("button", has_text="Open branch report").click()
        self.page.wait_for_selector("text=branch report")
        self.assertEqual(
            self.page.locator("#branch-target-input").input_value(), "SALET -----")

    def test_word_report_keeps_refreshing_once_the_menu_closes(self):
        # The guard must lift with the menu: a report frozen by a dismissed
        # menu would look identical to a dead connection.
        self.apply_branch_target("SALET")
        self.page.wait_for_selector(".response-group-breakdown .answer-segment")
        self.page.locator(
            ".response-group-breakdown .answer-segment").first.click()
        self.page.wait_for_selector(".group-menu")
        self.page.keyboard.press("Escape")
        self.page.wait_for_selector(".group-menu", state="detached")
        rendered = self.page.evaluate("""() => new Promise(resolve => {
          const root = document.querySelector('#report');
          const observer = new MutationObserver(() => {
            observer.disconnect(); resolve(true);
          });
          observer.observe(root, {childList: true});
          setTimeout(() => { observer.disconnect(); resolve(false); }, 6000);
        })""")
        self.assertTrue(rendered, "the report stopped refreshing after a dismissal")

    def _apply_many_group_word_report(self):
        # A word that splits its branch 201 ways: every segment but the first
        # is a couple of pixels wide at any viewport this suite uses.
        self.page.route("**/api/view**", lambda route: route.abort())
        report = copy.deepcopy(load_fixtures(FIXTURE_DIRECTORY)["word.json"])
        patterns = ["".join(letters)
                    for letters in itertools.product("gy-", repeat=5)]
        # Every group solved, so the outline threshold is measured against
        # segments that all ask for one.
        report["data"]["response_group_breakdown"] = (
            [{"pattern": patterns[0], "answer_count": 400, "solved": True}]
            + [{"pattern": pattern, "answer_count": 1, "solved": True}
               for pattern in patterns[1:201]]
        )
        report["data"]["total_rows"] = 201
        self.page.evaluate("""(report) => {
          applyReport(report, null,
            {...__reportClient.getState(), branch_target:'SALET'});
        }""", report)
        self.page.wait_for_selector(".response-group-breakdown .answer-segment")

    def test_word_report_breakdown_menu_reaches_a_two_pixel_group(self):
        # The reason every group opens a menu rather than only the wide ones:
        # a sliver is impossible to identify by sight, and a dead tap on one
        # reads as the page being broken.
        self._apply_many_group_word_report()
        segments = self.page.locator(".response-group-breakdown .answer-segment")
        self.assertEqual(segments.count(), 201)
        narrow = segments.last
        self.assertLess(narrow.bounding_box()["width"], 5)
        narrow.click(force=True)
        self.page.wait_for_selector(".group-menu")
        last_pattern = "".join(
            list(itertools.product("gy-", repeat=5))[200])
        self.assertEqual(
            self._menu().locator(".group-menu-title .word").get_attribute(
                "data-spine"),
            "QUEUE " + last_pattern)
        self.assertIn("1 answer", self._menu().locator(
            ".group-menu-facts").inner_text())

    def test_word_report_breakdown_strip_is_one_tab_stop(self):
        # 201 groups would otherwise be 201 stops between a reader and the rest
        # of the page; the arrow keys move within the strip instead.
        self._apply_many_group_word_report()
        stops = self.page.eval_on_selector_all(
            ".response-group-breakdown .answer-segment",
            "nodes => nodes.map(node => node.tabIndex)")
        self.assertEqual(stops.count(0), 1)
        self.assertEqual(stops[0], 0)
        self.page.locator(".response-group-breakdown .answer-segment").first.focus()
        self.page.keyboard.press("ArrowRight")
        moved = self.page.eval_on_selector_all(
            ".response-group-breakdown .answer-segment",
            "nodes => nodes.map(node => node.tabIndex)")
        self.assertEqual(moved.count(0), 1)
        self.assertEqual(moved[1], 0)
        self.assertEqual(
            self.page.evaluate(
                "() => document.activeElement.getAttribute('aria-label')"),
            self.page.locator(
                ".response-group-breakdown .answer-segment").nth(1)
            .get_attribute("aria-label"))

    def test_word_report_group_menu_stays_in_the_viewport_after_a_resize(self):
        # The menu's position is written as pixels against the strip it was
        # opened on.  A menu placed at a desktop width and then carried to a
        # phone would keep that offset and widen the document -- the exact
        # regression test_no_horizontal_scroll_at_required_widths guards
        # against, which never sees it because it opens no menu.
        self._apply_many_group_word_report()
        self.page.locator(
            ".response-group-breakdown .answer-segment").last.click(force=True)
        self.page.wait_for_selector(".group-menu")
        for width in (375, 390, 480, 800, 1200):
            with self.subTest(width=width):
                self.page.set_viewport_size({"width": width, "height": 800})
                self.page.wait_for_timeout(150)
                box = self._menu().bounding_box()
                self.assertGreaterEqual(box["x"], 0)
                self.assertLessEqual(box["x"] + box["width"], width)
                scroll, client = self.page.evaluate(
                    "() => [document.documentElement.scrollWidth,"
                    " document.documentElement.clientWidth]")
                self.assertLessEqual(scroll, client, f"document widened at {width}px")

    def _apply_graded_word_report(self):
        """A solved decomposition whose segments straddle the outline threshold.

        Every segment is a fixed share of the strip, so a viewport change moves
        all of them at once -- the sizes here are graded so that some cross the
        threshold on the way and others stay on their side of it.
        """
        self.page.route("**/api/view**", lambda route: route.abort())
        report = copy.deepcopy(load_fixtures(FIXTURE_DIRECTORY)["word.json"])
        patterns = ["".join(letters)
                    for letters in itertools.product("gy-", repeat=5)]
        counts = [500, 40, 30, 25, 20, 15, 10, 5, 3, 2, 1]
        report["data"]["response_group_breakdown"] = [
            {"pattern": pattern, "answer_count": count, "solved": True}
            for pattern, count in zip(patterns, counts)
        ]
        report["data"]["total_rows"] = len(counts)
        self.page.evaluate("""(report) => {
          applyReport(report, null,
            {...__reportClient.getState(), branch_target:'SALET'});
        }""", report)
        self.page.wait_for_selector(
            ".response-group-breakdown .answer-segment.solved-group")

    def _breakdown_segment_states(self):
        return self.page.eval_on_selector_all(
            ".response-group-breakdown .answer-segment",
            """nodes => nodes.map(node => [node.clientWidth,
                 node.classList.contains('solved-group')])""")

    def test_word_report_breakdown_remeasures_outlines_when_the_window_resizes(self):
        # A segment's width is a share of the strip, so a resize moves it across
        # the outline threshold with the report itself unchanged.  The poll is
        # parked, so only the resize can put this right.
        self._apply_graded_word_report()
        wide = self._breakdown_segment_states()
        self.page.set_viewport_size({"width": 375, "height": 800})
        self.page.wait_for_timeout(200)
        narrow = self._breakdown_segment_states()
        self.assertGreater(
            sum(outlined for _, outlined in wide),
            sum(outlined for _, outlined in narrow),
            "no segment crossed the outline threshold, so this proves nothing")
        for width, outlined in narrow:
            self.assertEqual(outlined, width >= 16, f"{width}px segment")
        # And back: a segment the resize widened regains its outline.
        self.page.set_viewport_size({"width": 1200, "height": 800})
        self.page.wait_for_timeout(200)
        self.assertEqual(self._breakdown_segment_states(), wide)

    def test_word_report_histogram_scales_against_the_branch_best(self):
        # Without a scale the fill spans the track and says nothing about how
        # good the split is; with one it is this word's groups against the most
        # any guess achieves on the same branch.
        self.page.route("**/api/view**", lambda route: route.abort())
        base = copy.deepcopy(load_fixtures(FIXTURE_DIRECTORY)["word.json"])

        def apply(scale):
            payload = copy.deepcopy(base)
            if scale is not None:
                payload["data"]["maximum_response_group_count"] = scale
            self.page.evaluate("""(payload) => {
              applyReport(payload, null,
                {...__reportClient.getState(), branch_target:'SALET'});
            }""", payload)
            self.page.wait_for_selector(".response-group-breakdown")
            return self.page.evaluate(
                """() => {
                  const track = document.querySelector(
                    '.response-group-breakdown .response-count-track');
                  return track.querySelector(
                    '.response-count-fill').getBoundingClientRect().width
                    / track.getBoundingClientRect().width;
                }""")

        # The fixture's word splits its branch four ways.
        self.assertAlmostEqual(apply(None), 1.0, delta=0.02)
        self.assertAlmostEqual(apply(4), 1.0, delta=0.02)
        self.assertAlmostEqual(apply(8), 0.5, delta=0.02)
        self.assertEqual(
            self.page.locator(
                ".response-group-breakdown .response-count-track"
            ).get_attribute("aria-label"),
            "4 response groups for QUEUE out of the 8 the best guess splits"
            " this branch into")

    def test_word_report_breakdown_outlines_only_groups_wide_enough_to_show_one(self):
        # A 2-px outline on each edge of a sliver is all border and no
        # interior, so a run of finished slivers would read as one black block.
        self._apply_many_group_word_report()
        segments = self.page.eval_on_selector_all(
            ".response-group-breakdown .answer-segment",
            """nodes => nodes.map(node => [node.clientWidth,
                 node.classList.contains('solved-group'),
                 node.title.includes('solved')])""")
        self.assertEqual(len(segments), 201)
        for width, outlined, says_solved in segments:
            self.assertEqual(outlined, width >= 16, f"{width}px segment")
            self.assertTrue(says_solved)
        self.assertTrue(any(outlined for _, outlined, _ in segments))
        self.assertTrue(any(not outlined for _, outlined, _ in segments))

    def test_word_report_breakdown_outlines_solved_groups(self):
        self.apply_branch_target("SALET")
        self.page.wait_for_selector(".response-group-breakdown .answer-segment.solved-group")
        outlined = self.page.eval_on_selector_all(
            ".response-group-breakdown .answer-segment",
            """nodes => nodes.map(node => [
                 node.classList.contains('solved-group'),
                 getComputedStyle(node).boxShadow,
                 node.title])""")
        # The fixture's decomposition is solved, unsolved, unsolved, solved.
        self.assertEqual([solved for solved, _, _ in outlined],
                         [True, False, False, True])
        for solved, shadow, title in outlined:
            self.assertEqual(solved, "solved" in title)
            self.assertEqual(solved, "inset" in shadow and "rgb(0, 0, 0)" in shadow)

    def test_leaderboard_breakdown_group_opens_a_menu_then_the_branch(self):
        # The same interaction in both views: a group is never a direct jump in
        # one place and a menu in the other.
        self.page.locator("[data-kind=leaderboard]").click()
        self.page.wait_for_selector(".leaderboard-card .answer-segment")
        self.page.locator(".leaderboard-card .answer-segment").first.click()
        self.page.wait_for_selector(".group-menu")
        self.assertEqual(
            self._menu().locator(".group-menu-title .word").get_attribute(
                "data-spine"),
            "SALET -----")
        self._menu().locator("button", has_text="Open branch report").click()
        self.page.wait_for_selector("text=branch report")
        self.assertEqual(
            self.page.locator("#branch-target-input").input_value(), "SALET -----")

    def test_leaderboard_breakdown_leaves_solved_groups_unoutlined(self):
        # Every ranked opener is complete, so outlining there would black out
        # every segment and say nothing.  The leaderboard sends no solved flag.
        self.page.locator("[data-kind=leaderboard]").click()
        self.page.wait_for_selector(".leaderboard-card .answer-segment")
        self.assertEqual(
            self.page.locator(".leaderboard-card .answer-segment.solved-group").count(),
            0)

    def test_leaderboard_tab_drops_an_opener_only_sort(self):
        self.page.locator("[data-kind=openers]").click()
        self.page.wait_for_selector("text=openers report")
        self.assertEqual(self.page.locator("#sort").input_value(), "completed")
        self.page.locator("[data-kind=leaderboard]").click()
        self.page.wait_for_selector("text=Opener leaderboard")
        self.assertEqual(self.page.evaluate("__reportClient.getState().sort"), "")
        self.assertNotIn("sort=completed", self.page.url)

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

    def test_unchanged_leaderboard_poll_keeps_the_reading_position(self):
        self.page.set_viewport_size({"width": 834, "height": 1112})
        self.page.evaluate("""() => {
          const realFetch = window.fetch.bind(window);
          let leaderboard;
          window.fetch = (url, options) => realFetch(url, options).then(async response => {
            if (!String(url).includes('/leaderboard')) return response;
            if (!leaderboard) {
              leaderboard = await response.json();
              const row = leaderboard.data.rows[0];
              leaderboard.data.rows = Array.from({length: 12}, (_, index) => ({
                ...row, word: 'a' + String(index).padStart(4, '0'), rank: index + 1,
              }));
              leaderboard.data.total_rows = leaderboard.data.rows.length;
              leaderboard.data.counts.complete = leaderboard.data.rows.length;
            }
            return new Response(JSON.stringify(leaderboard), {
              status: 200, headers: {'Content-Type': 'application/json'},
            });
          });
        }""")
        self.page.locator("[data-kind=leaderboard]").click()
        self.page.wait_for_selector(".leaderboard-card")
        cards = self.page.locator(".leaderboard-card")
        self.assertEqual(cards.count(), 12)
        card = cards.nth(8)
        card.scroll_into_view_if_needed()
        before = card.evaluate("(node) => { window.scrollBy(0, 40); return node.getBoundingClientRect().top; }")
        card.evaluate("(node) => node.dataset.testMarker = 'still-here'")
        self.page.evaluate("async () => { await window.__reportClient.fetchReport(); }")
        self.assertEqual(card.get_attribute("data-test-marker"), "still-here")
        self.assertAlmostEqual(
            card.evaluate("(node) => node.getBoundingClientRect().top"), before, delta=1
        )

    def test_leaderboard_count_only_poll_keeps_its_cards(self):
        self.page.locator("[data-kind=leaderboard]").click()
        self.page.wait_for_selector(".leaderboard-card")
        card = self.page.locator(".leaderboard-card").first
        card.evaluate("(node) => node.dataset.testMarker = 'still-here'")
        self.page.evaluate("""() => {
          const realFetch = window.fetch.bind(window);
          window.fetch = (url, options) => realFetch(url, options).then(async response => {
            if (!String(url).includes('/leaderboard')) return response;
            const report = await response.json();
            report.data.counts.pending = 123;
            return new Response(JSON.stringify(report), {
              status: 200, headers: {'Content-Type': 'application/json'},
            });
          });
        }""")
        self.page.evaluate("async () => { await window.__reportClient.fetchReport(); }")
        self.assertEqual(card.get_attribute("data-test-marker"), "still-here")
        self.assertEqual(
            self.page.locator("h2 + .metrics .metric", has_text="pending").locator("strong").inner_text(),
            "123",
        )

    def test_changed_leaderboard_poll_keeps_the_visible_word_in_place(self):
        self.page.set_viewport_size({"width": 834, "height": 1112})
        self.page.evaluate("""() => {
          const realFetch = window.fetch.bind(window);
          let leaderboard, allowChangedReport = false;
          window.__changeLeaderboardReport = () => { allowChangedReport = true; };
          window.fetch = (url, options) => realFetch(url, options).then(async response => {
            if (!String(url).includes('/leaderboard')) return response;
            if (!leaderboard) {
              leaderboard = await response.json();
              const row = leaderboard.data.rows[0];
              leaderboard.data.rows = Array.from({length: 12}, (_, index) => ({
                ...row, word: 'a' + String(index).padStart(4, '0'), rank: index + 1,
              }));
              leaderboard.data.total_rows = leaderboard.data.rows.length;
              leaderboard.data.counts.complete = leaderboard.data.rows.length;
            }
            const report = structuredClone(leaderboard);
            if (allowChangedReport) report.data.rows[0].erd = 9.876;
            return new Response(JSON.stringify(report), {
              status: 200, headers: {'Content-Type': 'application/json'},
            });
          });
        }""")
        self.page.locator("[data-kind=leaderboard]").click()
        self.page.wait_for_selector(".leaderboard-card")
        cards = self.page.locator(".leaderboard-card")
        self.assertEqual(cards.count(), 12)
        card = cards.nth(8)
        card.scroll_into_view_if_needed()
        before = card.evaluate("(node) => { window.scrollBy(0, 40); return node.getBoundingClientRect().top; }")
        self.page.evaluate("async () => { window.__changeLeaderboardReport(); await window.__reportClient.fetchReport(); await new Promise(requestAnimationFrame); }")
        self.assertAlmostEqual(
            card.evaluate("(node) => node.getBoundingClientRect().top"), before, delta=1
        )

    def test_changed_leaderboard_poll_keeps_the_reader_at_the_bottom(self):
        self.page.set_viewport_size({"width": 834, "height": 1112})
        distances = self.page.evaluate("""async () => {
          const base = await (await fetch('/api/view/leaderboard')).json();
          const state = {...__reportClient.getState(), kind: 'leaderboard'};
          const leaderboard = count => {
            const report = structuredClone(base), row = report.data.rows[0];
            report.data.rows = Array.from({length: count}, (_, index) => ({
              ...row, word: 'a' + String(index).padStart(4, '0'), rank: index + 1,
            }));
            report.data.total_rows = count;
            report.data.counts.complete = count;
            return report;
          };
          const before = leaderboard(12), after = leaderboard(15), later = leaderboard(18);
          applyReport(before, null, state);
          await new Promise(requestAnimationFrame);
          scrollTo(0, document.documentElement.scrollHeight);
          applyReport(after, before, state);
          await new Promise(requestAnimationFrame);
          const bottomDistance = document.documentElement.scrollHeight - (scrollY + innerHeight);
          scrollBy(0, -160);
          const nearBottomDistance = document.documentElement.scrollHeight - (scrollY + innerHeight);
          applyReport(later, after, state);
          await new Promise(requestAnimationFrame);
          return {bottomDistance, nearBottomDistance,
                  restoredNearBottomDistance: document.documentElement.scrollHeight - (scrollY + innerHeight)};
        }""")
        self.assertLessEqual(distances["bottomDistance"], 1)
        self.assertAlmostEqual(
            distances["restoredNearBottomDistance"], distances["nearBottomDistance"], delta=1
        )

    def test_leaderboard_selection_survives_a_changed_poll(self):
        self.page.locator("[data-kind=leaderboard]").click()
        self.page.wait_for_selector(".leaderboard-card")
        card = self.page.locator(".leaderboard-card").first
        card.evaluate("(node) => node.dataset.testMarker = 'still-here'")
        self.page.evaluate("""() => {
          const realFetch = window.fetch.bind(window);
          window.fetch = (url, options) => realFetch(url, options).then(async response => {
            if (!String(url).includes('/leaderboard')) return response;
            const report = await response.json();
            report.data.rows[0].erd = 9.876;
            return new Response(JSON.stringify(report), {
              status: 200, headers: {'Content-Type': 'application/json'},
            });
          });
          const range = document.createRange();
          range.selectNodeContents(document.querySelector('.leaderboard-card .word'));
          const selection = getSelection();
          selection.removeAllRanges(); selection.addRange(range);
        }""")
        self.page.evaluate("async () => { await window.__reportClient.fetchReport(); }")
        self.assertEqual(card.get_attribute("data-test-marker"), "still-here")
        self.assertFalse(self.page.evaluate("() => getSelection().isCollapsed"))

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

    def test_candidates_panel_is_a_bounded_summary_not_per_candidate_rows(self):
        requested = []
        self.page.on("request", lambda request: requested.append(request.url))
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("section:has-text('Candidates')")
        text = self.page.locator("section:has-text('Candidates')").first.inner_text()
        # A summary of provenance and per-worker contribution, never a row per
        # candidate — the branch holds far more claims than a browser can render.
        self.assertNotIn("12,819 done", text)
        self.assertNotIn("= + evaluated", text)
        self.assertIn("evaluated 11,200", text)
        self.assertNotIn("1,500 one-level ERD prunes", text)
        self.assertNotIn("119 two-level ERD prunes", text)
        self.assertIn("in flight 5", text)
        self.assertIn("worker evals w0:6,484 w2:6,335", text)
        # Nothing fetches the raw per-candidate list, and no per-candidate rows
        # are rendered.
        self.assertFalse(any("claims=1" in url for url in requested))
        self.assertLess(self.page.locator("section:has-text('Candidates') .card").count(), 1)

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
        self.assertIn("0.001 3/3,209", text)

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
            "section:has-text('Candidates') .labeled-facts").first
        text = facts.inner_text()
        self.assertIn("one-level ERD prunes", text)
        self.assertIn("two-level ERD prunes", text)

    def test_candidate_eta_labels_projected_work_as_remaining(self):
        text = self.page.evaluate("""async () => {
          const report=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          report.data.candidate_eta={state:'rough',sample_duration_seconds:300,
            estimated_seconds:13,remaining_inspection_count:0,
            expected_full_evaluation_count:8393,worker_count_changed:false};
          applyReport(report,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
          const section=[...document.querySelectorAll('section')].find(
            section=>section.querySelector('h2')?.textContent==='Candidates'
          );
          return [...section.querySelectorAll('.labeled-facts')].map(
            facts=>facts.innerText
          );
        }""")
        rough_eta = next(line for line in text if line.startswith("Rough ETA"))
        self.assertNotIn("ETA work", rough_eta)
        eta_work = next(line for line in text if line.startswith("ETA work remaining"))
        self.assertIn("checks 0 · full evals ~8,393", eta_work)

    def test_ceiling_proven_loss_explains_its_proof(self):
        facts = self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          branch.data.recent_finalizations[0]={
            ...branch.data.recent_finalizations[0],outcome:'loss',loss_proof:'ceiling_above_budget',budget:3,ceiling:3.25,wall_millis:374529,
          };
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
          const card=document.querySelector('[data-grid-key="finalizations"] .card');
          return [...card.querySelectorAll('.stat-line > span')].map(span=>span.innerText);
        }""")
        self.assertIn("ERD lower bound 3.250 26/8", facts)
        self.assertIn("exceeds budget 3", facts)
        self.assertIn("6m", facts)
        self.assertNotIn("374,529 ms", facts)
        self.page.set_viewport_size({"width": 375, "height": 800})
        self.assertLessEqual(*self.page.evaluate(
            "() => [document.documentElement.scrollWidth,"
            "document.documentElement.clientWidth]"))

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

    def test_finalization_shows_best_first_scheduling_evidence(self):
        text = self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          branch.data.recent_finalizations[0]={
            ...branch.data.recent_finalizations[0],
            winner_best_first_rank:46,
            winner_republish_count:1,
            candidates_completed_before_winner:4900,
            weakest_best_first_rank_before_winner:7795,
            republished_candidate_count:5305,
            max_candidate_republish_count:3,
          };
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
          return document.querySelector('#report').innerText;
        }""")
        self.assertIn("winner ranked 46", text)
        self.assertIn("4,900 candidates completed first", text)
        self.assertIn("weakest of them ranked 7,795", text)
        self.assertIn("winner republished 1\u00d7", text)
        self.assertIn("5,305 (up to 3\u00d7 each)", text)

    def test_finalization_omits_scheduling_evidence_when_unrecorded(self):
        text = self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          branch.data.recent_finalizations[0]={
            ...branch.data.recent_finalizations[0],
            winner_best_first_rank:null,
            republished_candidate_count:0,
          };
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
          return document.querySelector('#report').innerText;
        }""")
        self.assertNotIn("winner ranked", text)

    def test_finalization_omits_comparisons_without_a_winner_rank(self):
        text = self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          branch.data.recent_finalizations[0]={
            ...branch.data.recent_finalizations[0],
            winner_best_first_rank:null,
            candidates_completed_before_winner:4900,
            weakest_best_first_rank_before_winner:7795,
            republished_candidate_count:12,
            max_candidate_republish_count:2,
          };
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
          return document.querySelector('#report').innerText;
        }""")
        self.assertIn("12 (up to", text)
        self.assertNotIn("candidates completed first", text)
        self.assertNotIn("weakest of them ranked", text)

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
        # The source view keeps the filter group for its own state filter and
        # drops the branch-shaped half of it; which controls it offers is
        # pinned by test_sources_controls_offer_source_axes_not_branch_ones.
        self.page.locator("[data-kind=openers]").click()
        self.page.wait_for_function(
            "() => __reportClient.getState().kind === 'openers'"
        )
        self.assertFalse(self.page.locator("#filters-group").is_hidden())
        self.assertTrue(self.page.locator("#branch-filters").is_hidden())
        self.assertFalse(self.page.locator("#limit-field").is_hidden())
        # The control pages this view rather than capping it, and says so.
        self.assertEqual(self.page.locator("#limit-label").inner_text(),
                         "Words per page")

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

    def grid_script(self, body):
        return "async () => {" + GRID_SCRIPT_HELPERS + body + "}"

    def two_column_overview(self):
        """A viewport where the branch grid lays out in exactly two columns."""
        self.page.set_viewport_size({"width": 500, "height": 1400})
        self.page.wait_for_selector(".grid > [data-identity]")
        columns = self.page.evaluate(
            "() => getComputedStyle(document.querySelector('.grid'))"
            ".gridTemplateColumns.split(' ').filter(Boolean).length"
        )
        self.assertEqual(columns, 2)

    def test_overview_departure_compacts_within_its_own_column(self):
        self.two_column_overview()
        result = self.page.evaluate(self.grid_script("""
          const base=await overviewReport();
          const state=__reportClient.getState();
          const five=namedBranches(base,['P','Q','R','S','T']);
          const withoutP=namedBranches(base,['Q','R','S','T']);
          applyReport(five,null,state);await settled();
          // A second identical report lets the packing settle onto the columns
          // it will actually use before anything departs.
          applyReport(five,five,state);await settled();
          const before=gridCells();
          applyReport(withoutP,five,state);await settled();
          return {before,after:gridCells()};
        """))
        # P sits at the top of the left column, R and T beneath it.
        self.assertEqual(result["before"]["P"], [0, 0])
        self.assertEqual(result["before"]["R"], [1, 0])
        self.assertEqual(result["before"]["T"], [2, 0])
        # Its column closes upward and the right column is untouched.
        self.assertEqual(result["after"]["R"], [0, 0])
        self.assertEqual(result["after"]["T"], [1, 0])
        self.assertEqual(result["after"]["Q"], result["before"]["Q"])
        self.assertEqual(result["after"]["S"], result["before"]["S"])

    def test_overview_crosses_columns_only_to_save_a_row(self):
        self.two_column_overview()
        result = self.page.evaluate(self.grid_script("""
          const base=await overviewReport();
          const state=__reportClient.getState();
          const four=namedBranches(base,['A','B','C','D']);
          const leftColumn=namedBranches(base,['A','C']);
          applyReport(four,null,state);await settled();
          applyReport(four,four,state);await settled();
          const before=gridCells();
          applyReport(leftColumn,four,state);await settled();
          return {before,after:gridCells()};
        """))
        # A over C in the left column, B over D in the right.
        self.assertEqual(result["before"]["A"], [0, 0])
        self.assertEqual(result["before"]["C"], [1, 0])
        # With the right column gone, staying put would cost a second row, so C
        # moves diagonally up into it.
        self.assertEqual(result["after"]["A"], [0, 0])
        self.assertEqual(result["after"]["C"], [0, 1])

    def test_overview_arrivals_fill_in_reading_order(self):
        self.two_column_overview()
        cells = self.page.evaluate(self.grid_script("""
          const base=await overviewReport();
          const state=__reportClient.getState();
          applyReport(namedBranches(base,['seed']),null,state);await settled();
          applyReport(namedBranches(base,['P','Q','R','S','T']),
                      namedBranches(base,['seed']),state);await settled();
          return gridCells();
        """))
        self.assertEqual(
            [cells[name] for name in ("P", "Q", "R", "S", "T")],
            [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]],
        )

    def test_overview_holds_completed_branch_before_its_departure(self):
        """A completed branch stays visible long enough to explain its exit."""
        result = self.page.evaluate(self.grid_script("""
          const base=await overviewReport();
          const state=__reportClient.getState();
          const finalizing=namedBranches(base,['complete']);
          finalizing.data.branches[0].branch_phase='finalizing';
          const empty=namedBranches(base,[]);
          applyReport(finalizing,null,state);await settled();
          applyReport(empty,finalizing,state);
          const card=()=>document.querySelector('[data-identity="complete"]');
          const held={className:card().className,background:getComputedStyle(card()).backgroundColor,text:card().innerText};
          const realNow=Date.now;Date.now=()=>realNow()+5001;
          applyReport(empty,empty,state);Date.now=realNow;
          await settled();
          const ordinary=namedBranches(base,['ordinary']);
          applyReport(ordinary,null,state);await settled();
          applyReport(empty,ordinary,state);
          return {held,remaining:!!card(),ordinaryClass:document.querySelector('[data-identity="ordinary"]').className};
        """))
        self.assertIn("recently-completed", result["held"]["className"])
        self.assertEqual(result["held"]["background"], "rgb(234, 242, 252)")
        self.assertIn("done", result["held"]["text"])
        self.assertFalse(result["remaining"])
        self.assertNotIn("recently-completed", result["ordinaryClass"])

    def test_grid_transition_moves_the_page_below_it_monotonically(self):
        """The content under the grid must never reverse direction mid-flight.

        Every jump reported against this animation has been the page below the
        grid lurching one way and gliding back the other, because its position
        was the sum of separately animated card boxes.  The grid now carries one
        animated height, so a transition may move that content but only ever in
        the direction it is going to end up.
        """
        self.two_column_overview()
        for label, before_names, after_names in (
            ("shrinking", list("abcdefgh"), list("ace")),
            ("growing", list("ace"), list("abcdefgh")),
        ):
            with self.subTest(label):
                samples = self.page.evaluate(self.grid_script("""
                  const [beforeNames,afterNames]=%s;
                  const base=await overviewReport();
                  const state=__reportClient.getState();
                  applyReport(namedBranches(base,beforeNames),null,state);await settled();
                  const start=Math.round(workersHeading().getBoundingClientRect().top);
                  applyReport(namedBranches(base,afterNames),
                              namedBranches(base,beforeNames),state);
                  const samples=await sampleEveryFrame(
                    ()=>[Math.round(workersHeading().getBoundingClientRect().top)]);
                  return [[0,start],...samples];
                """ % json.dumps([before_names, after_names])))
                positions = [top for _, top in samples]
                steps = [
                    second - first
                    for first, second in zip(positions, positions[1:])
                    if second != first
                ]
                self.assertTrue(steps, "the page below the grid never moved")
                self.assertTrue(
                    all(step > 0 for step in steps) or all(step < 0 for step in steps),
                    f"page below the grid reversed direction: {steps}",
                )

    def test_grid_transition_never_widens_the_document(self):
        """Overflow during an animation is invisible to a resting measurement.

        A card translated past the right edge widens the document for as long as
        it is out there, and the phone rescales the whole page to fit and back.
        test_no_horizontal_scroll_at_required_widths only ever measures a settled
        layout, so it cannot see this.
        """
        for width in (390, 500, 620, 880):
            with self.subTest(width=width):
                self.page.set_viewport_size({"width": width, "height": 900})
                self.page.wait_for_selector(".grid > [data-identity]")
                samples = self.page.evaluate(self.grid_script("""
                  const base=await overviewReport();
                  const state=__reportClient.getState();
                  const few=namedBranches(base,['a','b','c']);
                  const many=namedBranches(base,['a','b','c','d','e','f','g','h']);
                  applyReport(few,null,state);await settled();
                  applyReport(many,few,state);
                  const growing=await sampleEveryFrame(()=>[
                    document.documentElement.scrollWidth,document.documentElement.clientWidth]);
                  applyReport(few,many,state);
                  const shrinking=await sampleEveryFrame(()=>[
                    document.documentElement.scrollWidth,document.documentElement.clientWidth]);
                  return [...growing,...shrinking];
                """))
                overflowing = [
                    (millis, scroll, client)
                    for millis, scroll, client in samples
                    if scroll > client
                ]
                self.assertEqual(overflowing, [], f"document widened at {width}px")

    def test_grids_are_paired_by_name_not_by_position(self):
        """A word report's grids come and go, so position is not identity.

        One response group disappearing while another appears leaves the grid
        count unchanged.  Pairing by position would then measure a grid against
        a different grid entirely, and every card in it would be classified as
        having both left and arrived: live cards fading out as ghosts inside a
        neighbour while their real cards wipe in as though newly claimed.
        """
        result = self.page.evaluate(self.grid_script("""
          const realFetch=window.fetch.bind(window);
          parkThePoll();
          const base=await (await realFetch('/api/view?branch_target=CRANE')).json();
          const state={...__reportClient.getState(),branch_target:'CRANE',group_by:'status'};
          const grouped=(labels)=>{
            const report=structuredClone(base);
            report.data.response_group_groups=labels.map(label=>({
              label,
              rollup:{answer_count:2,branch_count:2},
              rows:[1,2].map(index=>{
                const row=structuredClone(base.data.response_groups[0]);
                row.branch_key_hex=label+index;row.branch_reference=label+index;
                return row;
              }),
            }));
            return report;
          };
          const before=grouped(['alpha','beta']),after=grouped(['beta','gamma']);
          applyReport(before,null,state);await settled();
          applyReport(after,before,state);
          const gridsNow=()=>[...document.querySelectorAll('.grid[data-grid-key]')].map(grid=>({
            key:grid.dataset.gridKey,
            cards:[...grid.querySelectorAll(':scope > [data-identity]')].map(node=>node.dataset.identity),
          }));
          const midFlight=gridsNow();
          await settled();
          return {midFlight,settledGrids:gridsNow()};
        """))
        # Mid-flight, no card may appear under two grids: that only happens when
        # a grid is measured against one it is not.
        seen = [
            identity
            for grid in result["midFlight"]
            for identity in grid["cards"]
        ]
        self.assertEqual(sorted(seen), sorted(set(seen)), f"card in two grids: {result['midFlight']}")
        # beta survives untouched; alpha's cards are gone and gamma's are its own.
        settled_grids = {grid["key"]: sorted(grid["cards"]) for grid in result["settledGrids"]}
        self.assertEqual(
            settled_grids,
            {
                "response-groups/beta": ["beta1", "beta2"],
                "response-groups/gamma": ["gamma1", "gamma2"],
            },
        )

    def test_packing_rebuild_preserves_reading_order_across_a_resize(self):
        """The column count is always one refresh behind a viewport change.

        That is only safe because rebuilding the columns for a new count is
        order-preserving, so the cards a reader is looking at stay in the order
        they were in.  Nothing else pins that identity.
        """
        refresh = self.grid_script("""
          const base=await overviewReport();
          const state=__reportClient.getState();
          const eight=namedBranches(base,['a','b','c','d','e','f','g','h']);
          applyReport(eight,window.__applied||null,state);
          window.__applied=eight;
          await settled();
          const branches=document.querySelector('.grid[data-grid-key="branches"]');
          return {
            order:[...branches.querySelectorAll(':scope > [data-identity]')]
              .map(node=>node.dataset.identity),
            columns:getComputedStyle(branches)
              .gridTemplateColumns.split(' ').filter(Boolean).length,
          };
        """)
        self.two_column_overview()
        self.page.evaluate(refresh)
        narrow = self.page.evaluate(refresh)
        self.page.set_viewport_size({"width": 1000, "height": 1400})
        # The first refresh after a resize still packs on the old column count;
        # the second is the one that rebuilds the columns for the new one.
        first_after = self.page.evaluate(refresh)
        rebuilt = self.page.evaluate(refresh)
        self.assertEqual(narrow["columns"], 2)
        self.assertEqual(narrow["order"], list("abcdefgh"))
        self.assertGreater(first_after["columns"], 2)
        self.assertEqual(first_after["order"], list("abcdefgh"))
        self.assertEqual(rebuilt["order"], list("abcdefgh"))

    def test_grid_transition_returns_the_grid_to_normal_layout(self):
        result = self.page.evaluate(self.grid_script("""
          const base=await overviewReport();
          const state=__reportClient.getState();
          const many=namedBranches(base,['a','b','c','d','e','f']);
          for(const row of many.data.branches){
            row.branch_phase='evaluating';
            row.completed_candidate_count=Math.min(row.completed_candidate_count,row.candidate_count-1);
          }
          const few=namedBranches(base,['b','d']);
          applyReport(many,null,state);await settled();
          applyReport(few,many,state);
          const midFlight={
            gridStyle:document.querySelector('.grid').getAttribute('style')||'',
            pinned:[...document.querySelectorAll('.grid > [data-identity]')]
              .filter(node=>node.style.position==='absolute').length,
          };
          await settled();
          const grid=document.querySelector('.grid');
          return {midFlight,
            identities:[...grid.querySelectorAll(':scope > [data-identity]')]
              .map(node=>node.dataset.identity),
            gridStyle:grid.getAttribute('style')||'',
            styled:[...grid.querySelectorAll(':scope > [data-identity]')]
              .filter(node=>node.getAttribute('style')).length};
        """))
        # Mid-flight the grid is a fixed-height canvas of pinned cards.
        self.assertIn("height", result["midFlight"]["gridStyle"])
        self.assertGreater(result["midFlight"]["pinned"], 0)
        # Afterwards nothing of that remains, and the departed cards are gone.
        self.assertEqual(sorted(result["identities"]), ["b", "d"])
        self.assertEqual(result["gridStyle"], "")
        self.assertEqual(result["styled"], 0)

    def test_queue_reorder_animates_every_moved_card(self):
        """Grids whose order carries meaning still reflow, and all of them move.

        Only the overview's branches are packed by column; a sort the user chose
        must reorder as asked.
        """
        result = self.page.evaluate(self.grid_script("""
          const state={...__reportClient.getState(),kind:'queue',sort:'default'};
          const realFetch=window.fetch.bind(window);
          parkThePoll();
          const report=await (await realFetch('/api/view/queue')).json();
          applyReport(report,null,state);
          const identities=()=>[...document.querySelectorAll('.grid > [data-identity]')]
            .map(node=>node.dataset.identity);
          const before=identities();
          const reordered=structuredClone(report);
          reordered.data.rows.reverse();
          applyReport(reordered,report,{...state,sort:'priority'});
          const moved=[...document.querySelectorAll('.grid > [data-identity]')]
            .filter(node=>node.getAnimations().length).map(node=>node.dataset.identity);
          await settled();
          return {moved,before,after:identities()};
        """))
        self.assertGreater(len(result["moved"]), 1)
        self.assertEqual(result["after"], list(reversed(result["before"])))

    def test_republished_candidates_render_as_summary_not_raw_list(self):
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("text=Bundles")
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

    def sample_theme_colors(self):
        """--bg and the unplayed-tile background under the page's current
        color-scheme preference, read live with no reload in between."""
        return self.page.evaluate("""() => {
          const word=document.createElement('span');word.className='word word-sm';
          const node=document.createElement('span');node.className='letter blank';
          word.append(node);document.body.append(word);
          const result={
            bg:getComputedStyle(document.documentElement).getPropertyValue('--bg').trim(),
            blankTile:getComputedStyle(node).backgroundColor,
          };
          word.remove();
          return result;
        }""")

    def test_dark_mode_follows_os_color_scheme_live(self):
        # Default emulation is light; the unplayed tile matches the page
        # background, which is white.
        light = self.sample_theme_colors()
        self.assertEqual(light["bg"], "#ffffff")
        self.assertEqual(light["blankTile"], "rgb(255, 255, 255)")

        # Flipping the OS/browser preference repaints every themed color
        # immediately -- no reload, no in-page toggle -- because every color
        # in report_client.html reads a custom property rather than a literal.
        self.page.emulate_media(color_scheme="dark")
        dark = self.sample_theme_colors()
        self.assertNotEqual(dark["bg"], light["bg"])
        self.assertNotEqual(dark["blankTile"], light["blankTile"])
        # The unplayed tile still tracks the page background in dark mode.
        self.assertEqual(dark["blankTile"], "rgb(18, 18, 19)")

        # Flipping back is just as live.
        self.page.emulate_media(color_scheme="light")
        restored = self.sample_theme_colors()
        self.assertEqual(restored["bg"], light["bg"])
        self.assertEqual(restored["blankTile"], light["blankTile"])

    def render_sweep_marker(self):
        """Render a branch report with one live worker; the marker persists
        in the DOM afterward for a subsequent theme check with no rerender."""
        self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          branch.data.completed_candidate_indexes=[...Array(50).keys()];
          branch.data.workers=[{worker_id:'worker-3',worker_number:'3',updated_at:999,is_live:true,branch_key_hex:'01',branch_reference:'111111111111',candidate_index:75,current_candidate:'crane',current_candidate_is_answer:true}];
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
        }""")

    def sweep_marker_background_image(self):
        """The decoded SVG data URI currently behind the .sweep-marker."""
        return self.page.evaluate("""() => {
          const marker=document.querySelector('.sweep-marker');
          const raw=marker.style.backgroundImage;
          const uri=raw.slice('url("'.length,-'")'.length);
          return decodeURIComponent(uri);
        }""")

    def test_worker_marker_halo_tracks_theme_with_no_rerender(self):
        # The worker-marker digit is baked into an SVG data URI at render
        # time, so it must be actively refreshed on a color-scheme change --
        # reading --panel/--text live only fixes the *next* render.  This
        # never calls applyReport a second time: every other themed color
        # repaints on the OS/browser preference alone, and the marker must
        # keep up the same way, not wait for the next poll to replace it.
        self.render_sweep_marker()
        # The background poll (default every 2s) would otherwise call
        # fetchReport()/applyReport() again on its own schedule and rebuild
        # the marker from scratch with live colors, which would mask a
        # missing live-refresh path.  Blocking it means only the
        # media-query listener under test can update the marker already
        # on screen.
        self.page.route("**/api/view*", lambda route: route.abort())
        light_svg = self.sweep_marker_background_image()
        self.assertIn('stroke="#f8f9fa"', light_svg)
        self.assertIn('fill="#1a1a1b"', light_svg)

        self.page.emulate_media(color_scheme="dark")
        self.page.wait_for_function(
            "document.querySelector('.sweep-marker').style.backgroundImage.includes('%231c1c1e')"
        )
        dark_svg = self.sweep_marker_background_image()
        self.assertIn('stroke="#1c1c1e"', dark_svg)
        self.assertIn('fill="#e4e6eb"', dark_svg)

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
            self.assertIn("1,572/502", measured["text"])
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
            "?kind=leaderboard", "?kind=openers", "?kind=queue&tree=1",
            "?branch_target=RAISE+.....",
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

    def test_leaderboard_legend_labels_never_overlap_after_an_unhandled_resize(self):
        # The bucket legend lays its labels out with plain CSS grid flow, so
        # no two labels can overlap regardless of the card's width or
        # whether a render has ever run at that width. setUp's page is
        # already 1200px wide; each width below drops straight to it with no
        # wait, so nothing gets a chance to re-render first.
        self.page.goto(self.base_url + "?kind=leaderboard")
        self.page.wait_for_selector(".response-bucket-legend > span")
        for width in (375, 480, 600, 700, 800, 801, 900, 1000, 1100, 1199, 1200):
            with self.subTest(width=width):
                self.page.set_viewport_size({"width": width, "height": 800})
                overlaps = self.page.evaluate("""() => {
                  const overlaps = [];
                  for (const legend of document.querySelectorAll('.response-bucket-legend')) {
                    const spans = [...legend.querySelectorAll(':scope > span')]
                      .map(span => span.getBoundingClientRect());
                    for (let i = 0; i < spans.length; i++) {
                      for (let j = i + 1; j < spans.length; j++) {
                        const a = spans[i], b = spans[j];
                        if (a.left < b.right && b.left < a.right &&
                            a.top < b.bottom && b.top < a.bottom) {
                          overlaps.push([i, j]);
                        }
                      }
                    }
                  }
                  return overlaps;
                }""")
                self.assertEqual(overlaps, [], overlaps)

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

    def open_sources(self):
        self.page.locator("[data-kind=openers]").click()
        self.page.wait_for_selector("#report h1:text-is('openers report')")

    def test_sources_view_collapses_every_word_to_one_card(self):
        # The report's unit is the source word: three queued roots owning 1,376
        # branches between them read as three cards, and no branch card is
        # rendered until one of them is named.
        self.open_sources()
        requests = self.page.locator(".card.source-word")
        self.assertEqual(requests.count(), 3)
        self.assertEqual(
            self.page.locator("[data-grid-key=source-memberships]").count(), 0
        )
        # Grouped by state out of the box: the fixture's two queued words and
        # its one complete word read as two groups without touching a control.
        self.assertEqual(
            self.page.evaluate("() => __reportClient.getState().group_by"), "state")
        groups = self.page.locator(".source-word-groups > details")
        self.assertEqual(groups.count(), 2)
        self.assertEqual(
            [" ".join(groups.nth(index).locator("summary strong").inner_text().split())
             for index in range(2)],
            ["queued", "complete"])
        metrics = " ".join(
            self.page.locator("#report .metrics").first.inner_text().split()
        )
        self.assertIn("3 openers", metrics)
        self.assertIn("1,376 branches", metrics)
        self.assertIn("1,211 open", metrics)
        request_text = " ".join(requests.first.inner_text().split())
        self.assertIn("priority 5", request_text)
        self.assertIn("1,240 branches", request_text)
        self.assertIn("1,203 open", request_text)
        self.assertIn("37 done", request_text)
        self.assertIn("12 direct", request_text)
        self.assertIn("3 workers", request_text)
        # A word queued more than once is still one card, and says so rather
        # than splitting into a card per request.
        self.assertIn("2 requests", request_text)
        complete_text = " ".join(
            requests.filter(has_text="CRANE").inner_text().split())
        self.assertIn("completed 8m ago", complete_text)
        self.assertNotIn("priority 1", complete_text)
        self.assertNotIn("0 open", complete_text)

    def test_sources_progress_separates_no_work_from_completed_work(self):
        self.open_sources()
        progress = self.page.locator(".card.source-word .source-group-progress").first
        segments = self.page.eval_on_selector_all(
                ".card.source-word .source-group-progress:first-of-type > span",
                "spans => spans.map(span => ({className: span.className, width: Number.parseFloat(span.style.width)}))",
            )[:2]
        self.assertEqual([segment["className"] for segment in segments],
                         ["no-work work-boundary", ""])
        self.assertEqual(
            progress.locator("span.no-work").evaluate(
                "span => getComputedStyle(span).borderRightWidth"),
            "1px",
        )
        self.assertAlmostEqual(segments[0]["width"], 100 * 136 / 148, places=4)
        self.assertAlmostEqual(segments[1]["width"], 100 * 3 / 148, places=4)
        segment_boxes = self.page.locator(
            ".card.source-word .source-group-progress:first-of-type > span"
        ).evaluate_all("spans => spans.slice(0, 2).map(span => { const box = span.getBoundingClientRect(); return {left: box.left, top: box.top, width: box.width}; })")
        self.assertEqual(segment_boxes[0]["top"], segment_boxes[1]["top"])
        self.assertAlmostEqual(
            segment_boxes[0]["left"] + segment_boxes[0]["width"],
            segment_boxes[1]["left"],
            places=2,
        )
        self.assertEqual(
            progress.get_attribute("title"),
            "136 response groups need no work; 3 of 12 work groups completed",
        )

    def test_sources_progress_marks_no_work_before_unfinished_work(self):
        self.open_sources()
        class_name = self.page.evaluate("""async () => {
          const report = await (await fetch('/api/view/openers')).json();
          report.data.summary[0].direct_done_branch_count = 0;
          applyReport(report, null, parsePageState({search:'?kind=openers'}));
          return document.querySelector('.source-group-progress > span').className;
        }""")
        self.assertEqual(class_name, "no-work work-boundary")

    def test_sources_controls_offer_source_axes_not_branch_ones(self):
        self.open_sources()
        self.page.locator("details.filters").evaluate("node => node.open = true")
        # A source word has its own state; branch status, phase, answer count,
        # budget and priority all describe one branch, so they stay hidden.
        self.assertFalse(self.page.locator("#opener-state-filters").is_hidden())
        self.assertTrue(self.page.locator("#branch-filters").is_hidden())
        self.assertFalse(self.page.locator("#sort-field").is_hidden())
        self.assertFalse(self.page.locator("#group-by-field").is_hidden())
        self.assertEqual(
            self.page.eval_on_selector_all(
                "[data-source-state]", "inputs => inputs.map(i => i.value)"),
            ["queued", "active", "complete"])
        # iOS Safari shows hidden <option>s, so each report's own strategies
        # must be the only ones in the DOM.
        self.assertEqual(
            self.page.eval_on_selector_all(
                "#group-by option", "options => options.map(o => o.value)"),
            ["state", "completed", "elapsed", "worker_time", "requested",
             "worker_presence", "priority", "none"])
        self.assertEqual(
            self.page.eval_on_selector_all(
                "#group-by option", "options => options.map(o => [o.value, o.textContent])"),
            [["state", "state (default)"], ["completed", "completion date"],
             ["elapsed", "elapsed time"], ["worker_time", "total worker time"],
             ["requested", "time since request"], ["worker_presence", "worker"],
             ["priority", "priority"], ["none", "none"]],
        )
        # State is the default, and "none" is an explicit choice rather than
        # the absence of one.
        self.assertEqual(self.page.locator("#group-by").input_value(), "state")
        self.assertEqual(self.page.locator("#sort").input_value(), "completed")
        self.assertEqual(
            self.page.eval_on_selector_all(
                "#sort option", "options => options.map(o => o.value)"),
            ["completed", "", "elapsed", "worker_time", "priority",
             "requested", "age", "word", "branches", "open", "done", "workers"])

    def test_sources_state_filter_and_sort_reach_the_request(self):
        result = self.page.evaluate("""() => ({
          filtered: buildAPIURL(parsePageState({search:'?kind=openers&opener_state=queued,active'})),
          sorted: buildAPIURL(parsePageState({search:'?kind=openers&sort=branches'})),
          grouped: buildAPIURL(parsePageState({search:'?kind=openers&group_by=state'})),
          branchSort: buildAPIURL(parsePageState({search:'?kind=openers&sort=nodes'})),
          branchGroup: buildAPIURL(parsePageState({search:'?kind=openers&group_by=cache_state'})),
          ungrouped: buildAPIURL(parsePageState({search:'?kind=openers&group_by=none'})),
          elsewhere: buildAPIURL(parsePageState({search:'?kind=queue&opener_state=queued'}))
        })""")
        self.assertEqual(result["ungrouped"], "/api/view/openers?group_by=none")
        # URLSearchParams percent-encodes the separator; the server decodes it.
        self.assertEqual(
            result["filtered"],
            "/api/view/openers?opener_state=queued%2Cactive&group_by=state")
        self.assertEqual(result["sorted"],
                         "/api/view/openers?sort=branches&group_by=state")
        self.assertEqual(result["grouped"], "/api/view/openers?group_by=state")
        # A sort or grouping this report cannot serve falls back to the
        # default rather than being sent to be rejected, and the source filter
        # never leaks to a report that would reject it.
        self.assertEqual(result["branchSort"], "/api/view/openers?group_by=state")
        self.assertEqual(result["branchGroup"], "/api/view/openers?group_by=state")
        self.assertEqual(result["elsewhere"], "/api/view/queue")

    def test_sources_grouping_marks_visible_cards_as_stale_while_regrouping(self):
        self.open_sources()
        result = self.page.evaluate("""async () => {
          const originalFetch=window.fetch.bind(window);
          const replacement=await (await originalFetch('/api/view/openers')).json();
          let release;
          window.fetch=(url, options)=>String(url).includes('group_by=elapsed')
            ? new Promise(resolve=>{release=()=>resolve(new Response(JSON.stringify(replacement),{status:200,headers:{'Content-Type':'application/json'}}));})
            : originalFetch(url, options);
          const groupBy=document.querySelector('#group-by');
          groupBy.value='elapsed';
          groupBy.dispatchEvent(new Event('change',{bubbles:true}));
          await new Promise(resolve=>setTimeout(resolve,20));
          const pending={
            status:document.querySelector('#report-status').textContent,
            hidden:document.querySelector('#report-status').hidden,
            groupBy:groupBy.value,
            oldGroups:[...document.querySelectorAll('.source-word-groups > details > summary strong')].map(node=>node.textContent),
          };
          release();
          await new Promise(resolve=>setTimeout(resolve,20));
          const settled=document.querySelector('#report-status').hidden;
          window.fetch=originalFetch;
          return {pending,settled};
        }""")
        self.assertEqual(result["pending"]["groupBy"], "elapsed")
        self.assertFalse(result["pending"]["hidden"])
        self.assertEqual(
            result["pending"]["status"],
            "Showing groups by state (default) while regrouping by elapsed time…",
        )
        self.assertEqual(result["pending"]["oldGroups"], ["queued", "complete"])
        self.assertTrue(result["settled"])

    def test_sources_pager_walks_the_word_list(self):
        # The page size is the "Words per page" control; without a pager a
        # limit would just truncate the list with no way to the rest.  The
        # fixture server ignores query parameters, so the paged payload is
        # applied directly -- the pager is what is under test.
        def apply_page(offset, shown):
            self.page.evaluate("""async ([offset, shown]) => {
              const report = await (await fetch('/api/view/openers')).json();
              report.data.matched_source_word_count = 12;
              report.data.total_source_word_count = 12;
              report.data.source_word_offset = offset;
              report.data.summary = report.data.summary.slice(0, shown);
              applyReport(report, null,
                parsePageState({search:'?kind=openers&limit=3'}));
            }""", [offset, shown])
        apply_page(0, 3)
        pager = self.page.locator(".source-word-pager")
        self.assertIn("Showing 1–3 of 12 words",
                      " ".join(pager.inner_text().split()))
        previous_button = pager.locator("button", has_text="Prev")
        next_button = pager.locator("button", has_text="Next")
        # Nothing before the first page, and nothing after the last.
        self.assertTrue(previous_button.is_disabled())
        self.assertFalse(next_button.is_disabled())
        apply_page(9, 3)
        self.assertIn("Showing 10–12 of 12 words",
                      " ".join(pager.inner_text().split()))
        self.assertFalse(pager.locator("button", has_text="Prev").is_disabled())
        self.assertTrue(pager.locator("button", has_text="Next").is_disabled())

    def test_sources_branch_rows_have_their_own_pager(self):
        # The branch list pages like the word list: a named word can own
        # hundreds of branches, and a page size that truncated them with no
        # way to the rest is the defect the word pager already fixed.
        self.page.evaluate("""async () => {
          const report = await (await fetch('/api/view/openers?branch_target=SALET')).json();
          report.data.matched_rows = 40;
          report.data.branch_row_offset = 4;
          report.data.rows = report.data.rows.slice(0, 4);
          applyReport(report, null,
            parsePageState({search:'?kind=openers&branch_target=SALET&limit=4'}));
        }""")
        pager = self.page.locator(".branch-row-pager")
        self.assertIn("Showing 5–8 of 40 branch rows",
                      " ".join(pager.inner_text().split()))
        self.assertFalse(pager.locator("button", has_text="Prev").is_disabled())
        self.assertFalse(pager.locator("button", has_text="Next").is_disabled())
        # The word pager is a separate control over a separate list.
        self.assertEqual(self.page.locator(".source-word-pager").count(), 0)

    def test_sources_pager_is_absent_without_a_page_size(self):
        self.open_sources()
        self.assertEqual(self.page.locator(".source-word-pager").count(), 0)
        self.assertEqual(
            self.page.evaluate("() => __reportClient.getState().opener_offset"),
            None)
        self.assertEqual(self.page.locator(".branch-row-pager").count(), 0)

    def test_sources_grouping_buckets_cards_under_their_rollup(self):
        # The fixture server ignores query parameters, so the grouped payload
        # is applied directly -- the renderer is what is under test.
        self.page.evaluate("""async () => {
          const report = await (await fetch('/api/view/openers')).json();
          const rows = report.data.summary;
          const bucket = state => rows.filter(row => row.state === state);
          const rollup = group => ({
            source_word_count: group.length,
            branch_count: group.reduce((total, row) => total + row.branch_count, 0),
            open_branch_count: group.reduce((total, row) => total + row.open_branch_count, 0),
            done_branch_count: group.reduce((total, row) => total + row.done_branch_count, 0),
            worker_count: group.reduce((total, row) => total + row.worker_count, 0),
          });
          report.data.summary_groups = ['queued', 'complete'].map(state => ({
            label: state, rows: bucket(state), rollup: rollup(bucket(state)),
          }));
          applyReport(report, null,
            parsePageState({search:'?kind=openers&group_by=state'}));
        }""")
        groups = self.page.locator(".source-word-groups > details")
        self.assertEqual(groups.count(), 2)
        first = " ".join(groups.first.locator("summary").inner_text().split())
        self.assertIn("queued", first)
        self.assertIn("2 words", first)
        self.assertIn("1,336 branches", first)
        self.assertIn("1,211 open", first)
        self.assertEqual(
            groups.first.locator("[data-grid-key='source-words/queued'] > .card")
            .count(), 2)
        self.assertEqual(
            groups.nth(1).locator("[data-grid-key='source-words/complete'] > .card")
            .count(), 1)
        # Every word lands in exactly one group.
        self.assertEqual(
            self.page.locator(".source-word-groups .card").count(), 3)

    def test_sources_card_opens_the_word_report_where_its_erd_lives(self):
        # The card leads to the word report: that is where a word's ERD,
        # response groups and root progress are, and it is what an operator
        # reaches for after seeing a word listed.
        self.open_sources()
        self.page.locator(".card.source-word").first.click()
        self.page.wait_for_selector("text=word report")
        self.assertIn("branch_target=SALET", self.page.url)
        self.assertNotIn("kind=openers", self.page.url)
        # The source-only grouping cannot be served by a word report, so it
        # must not ride along into a request that would be rejected.
        self.assertNotIn("group_by", self.page.url)

    def test_sources_card_shows_the_words_own_erd(self):
        self.open_sources()
        cards = {
            card.locator("[data-spine]").first.get_attribute("data-spine"):
                " ".join(card.inner_text().split())
            for card in self.page.locator(".card.source-word").all()
        }
        # Complete: the exact ERD and the worst-case line it earned.
        self.assertIn("ERD 3.421 · max 5", cards["RAISE"])
        self.assertIn("ERD 3.389 · max 4", cards["CRANE"])
        # Still searching: how much of it is solved, not a number that moves.
        self.assertIn("ERD pending · 96/148 groups solved", cards["SALET"])

    def test_active_source_card_shows_elapsed_work_time(self):
        text = self.page.evaluate("""async () => {
          const report = await (await fetch('/api/view/openers')).json();
          const row = report.data.summary[0];
          row.state = 'active';
          row.elapsed_millis = 30000;
          row.worker_millis = null;
          delete report.data.summary_groups;
          applyReport(report, null, parsePageState({search:'?kind=openers'}));
          return document.querySelector('.card.source-word').innerText;
        }""")

        self.assertIn("elapsed 30s", text)

    def test_sources_branch_cards_appear_only_for_the_named_word(self):
        self.open_sources()
        self.page.locator(".card.source-word").first.locator(
            "button", has_text="Branches").click()
        self.page.wait_for_selector("[data-grid-key=source-memberships] > .card")
        ownership = self.page.locator("[data-grid-key=source-memberships] > .card")
        self.assertEqual(ownership.count(), 4)
        # The shared branch is never reduced to a single owner: each owning
        # request gets its own row, carrying that request's requested priority
        # beside the branch's effective one.
        shared = " ".join(
            self.page.locator('[data-identity="2:02"]').inner_text().split()
        )
        self.assertIn("shared", shared)
        self.assertIn("requested priority 3", shared)
        self.assertIn("effective priority 5", shared)
        self.assertIn("owners 2", shared)
        self.assertIn("parent @11111111", shared)
        sole = " ".join(
            self.page.locator('[data-identity="2:04"]').inner_text().split()
        )
        # Owned by only one request: no "shared" chip -- its absence is what
        # says so, rather than a redundant "sole" chip beside it.
        self.assertNotIn("shared", sole)
        self.assertNotIn("parent", sole)
        ownership_text = " ".join(
            self.page.locator("#report").inner_text().split()
        )
        self.assertIn("Shown 4 of 4 matched", ownership_text)
        self.assertIn("1 shared branch", ownership_text)

    def open_named_source(self):
        self.open_sources()
        self.page.locator(".card.source-word").first.locator(
            "button", has_text="Branches").click()
        self.page.wait_for_selector("[data-grid-key=source-memberships] > .card")

    def test_sources_metrics_survive_a_row_limit_truncating_the_grid(self):
        # matched_rows and shared_branch_count are the model's pre-limit
        # counts; a limit that drops every shared row from the grid must not
        # silently shrink the metric beside it.  The fixture server ignores
        # limit, so the truncated payload is applied directly.
        self.page.evaluate("""async () => {
          const report = await (await fetch('/api/view/openers?branch_target=SALET')).json();
          report.data.rows = report.data.rows.filter(row => !row.is_shared);
          applyReport(report, null,
            parsePageState({search:'?kind=openers&branch_target=SALET'}));
        }""")
        shown = self.page.locator("[data-grid-key=source-memberships] > .card")
        self.assertEqual(shown.count(), 2)
        report_text = " ".join(self.page.locator("#report").inner_text().split())
        self.assertIn("Shown 2 of 4 matched", report_text)
        self.assertIn("1 shared branch", report_text)

    def test_named_word_with_no_live_branches_is_not_the_unpicked_state(self):
        # An empty row list means two different things, and telling a reader
        # to pick a word they have already picked is the wrong one.
        self.page.evaluate("""async () => {
          const report = await (await fetch('/api/view/openers?branch_target=SALET')).json();
          report.data.rows = [];
          report.data.matched_rows = 0;
          applyReport(report, null,
            parsePageState({search:'?kind=openers&branch_target=SALET'}));
        }""")
        text = " ".join(self.page.locator("#report").inner_text().split())
        self.assertIn("SALET owns no live branches", text)
        self.assertNotIn("Pick a word", text)
        # The unfiltered view no longer carries a prompt for an action that is
        # not required to understand the source-word cards.
        self.open_sources()
        self.assertNotIn(
            "Pick a word", self.page.locator("#report").inner_text())

    def test_sources_metrics_count_a_branch_two_words_own_once(self):
        # The totals come from the model, which counts each branch once; the
        # client must not re-derive them by summing the per-word counts.
        metrics = self.page.evaluate("""async () => {
          const report = await (await fetch('/api/view/openers')).json();
          report.data.matched_branch_count = 900;
          report.data.matched_open_branch_count = 700;
          applyReport(report, null, parsePageState({search:'?kind=openers'}));
          return document.querySelector('#report .metrics').innerText;
        }""")
        metrics = " ".join(metrics.split())
        self.assertIn("900 branches", metrics)
        self.assertIn("700 open", metrics)
        # 1,376 is what summing the fixture's per-word counts would give.
        self.assertNotIn("1,376", metrics)

    def test_sources_ownership_row_draws_its_lineage_as_a_spine_step(self):
        self.open_named_source()
        root_step = self.page.locator('[data-identity="1:02"] .word')
        self.assertEqual(root_step.count(), 1)
        self.assertEqual(root_step.get_attribute("data-spine"), "SALET -y---")

    def test_sources_branches_control_narrows_the_report_and_clears_it_again(self):
        self.open_sources()
        card = self.page.locator(".card.source-word").first
        card.locator("button", has_text="Branches").click()
        self.page.wait_for_function(
            "() => __reportClient.getState().branch_target === 'SALET'"
        )
        self.assertIn("kind=openers", self.page.url)
        self.assertIn("branch_target=SALET", self.page.url)
        # The card that set the filter is the one that clears it, and is drawn
        # as the filter in force while it is -- otherwise nothing in the view
        # widens it again.
        marked = ".card.source-word.filtered"
        self.page.wait_for_selector(marked)
        self.page.locator(marked).locator(
            "button", has_text="Hide branches").click()
        self.page.wait_for_function(
            "() => __reportClient.getState().branch_target === ''"
        )
        self.assertNotIn("branch_target", self.page.url)
        self.page.wait_for_selector(marked, state="detached")

    def test_sources_ownership_card_opens_the_branch_it_names(self):
        self.open_named_source()
        self.page.locator('[data-identity="2:04"]').click()
        self.page.wait_for_selector("text=branch report")

    def test_sources_state_keeps_only_the_word_and_limit_the_report_reads(self):
        # The opener report reads a trailing word and a row limit, and rejects
        # a target naming no word rather than ignoring it, so the client must
        # not forward a branch spine or a branch filter it happens to be
        # carrying from the view the operator came from.  A spine that does
        # reach a word is accepted by the report, which then answers for the
        # trailing word alone -- so the prefix is dropped here rather than
        # displayed as though it had narrowed the answer.
        result = self.page.evaluate("""() => ({
          explicit: buildAPIURL(parsePageState({search:'?kind=openers'})),
          word: buildAPIURL(parsePageState({search:'?kind=openers&branch_target=SALET'})),
          spineToWord: buildAPIURL(parsePageState({search:'?kind=openers&branch_target=SALET+-y---+CRANE'})),
          branch: buildAPIURL(parsePageState({search:'?kind=openers&branch_target=RAISE+-----'})),
          reference: buildAPIURL(parsePageState({search:'?kind=openers&branch_target=%40222222222222'})),
          filtered: buildAPIURL(parsePageState({search:'?kind=openers&branch_status=active&priority=3&sort=size&tree=1'})),
          limited: buildAPIURL(parsePageState({search:'?kind=openers&limit=2'}))
        })""")
        # Grouping by state is the default, and the request says so rather
        # than leaving the server to guess: a pasted URL reproduces the view.
        self.assertEqual(result["explicit"], "/api/view/openers?group_by=state")
        self.assertEqual(result["word"],
                         "/api/view/openers?branch_target=SALET&group_by=state")
        self.assertEqual(result["spineToWord"],
                         "/api/view/openers?branch_target=CRANE&group_by=state")
        self.assertEqual(result["branch"], "/api/view/openers?group_by=state")
        self.assertEqual(result["reference"], "/api/view/openers?group_by=state")
        self.assertEqual(result["filtered"], "/api/view/openers?group_by=state")
        self.assertEqual(result["limited"],
                         "/api/view/openers?group_by=state&limit=2")

    def test_worker_cards_name_the_scheduling_role_and_why(self):
        preferred = self.page.locator('.card.worker[data-identity="worker-0"]')
        # The visible text carries the noun: "preferred" alone would not say
        # preferred what, and a tooltip cannot supply it on a touch screen.
        self.assertIn("preferred opener", preferred.inner_text())
        reason = preferred.get_by_title("serving its preferred opener work")
        self.assertEqual(reason.count(), 1)
        fallback = self.page.locator('.card.worker[data-identity="worker-3"]')
        self.assertIn("fallback opener", fallback.inner_text())
        self.assertIn(
            "no claimable bundle",
            fallback.get_by_title(re.compile("fallback")).get_attribute("title"),
        )
        # A worker between claims has no role recorded, and none is invented
        # for it on the card.
        idle = self.page.locator('.card.worker[data-identity="worker-1"]')
        self.assertNotIn("unattributed", idle.inner_text())
        self.assertEqual(idle.get_by_title(re.compile("serving")).count(), 0)

    def test_layout_toggle_is_hidden_where_there_is_no_topology(self):
        toggle = self.page.locator("#layout-toggle")
        flat = self.page.locator("#layout-flat")
        tree = self.page.locator("#layout-tree")
        self.assertTrue(toggle.is_visible())
        # Cache, hotspots, leaderboard, and sources have no branch topology, so
        # the layout switch is hidden entirely rather than shown-but-inert.
        for treeless in ("cache", "hotspots", "leaderboard", "openers"):
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


@unittest.skipUnless(
    RUN_WEBKIT_CONTAINER_TESTS or REQUIRE_WEBKIT_CONTAINER_TESTS,
    "Set RUN_WEBKIT_CONTAINER_TESTS=1 to exercise WebKit via a container "
    "(podman or docker required)",
)
class ReportClientWebKitBrowserTest(ReportClientBrowserTest):
    """The Chromium suite's test bodies, replayed against real WebKit.

    The browser itself runs inside the Microsoft Playwright container (see
    tests/webkit_container.py); only setUpClass/tearDownClass differ from the
    Chromium base class.
    """

    @classmethod
    def setUpClass(cls):
        if sync_playwright is None:
            raise RuntimeError(
                "Playwright is required when REQUIRE_WEBKIT_CONTAINER_TESTS=1"
            )
        cls.server_context = fixture_server()
        cls.base_url = cls.server_context.__enter__()
        cls.playwright = sync_playwright().start()
        cls.webkit_container = None
        try:
            cls.webkit_container = start_webkit_server()
            cls.browser = cls.playwright.webkit.connect(
                cls.webkit_container.ws_endpoint
            )
        except (WebKitContainerUnavailable, Exception) as error:
            if cls.webkit_container is not None:
                cls.webkit_container.stop()
            cls.playwright.stop()
            cls.server_context.__exit__(None, None, None)
            if REQUIRE_WEBKIT_CONTAINER_TESTS:
                raise RuntimeError("WebKit container failed to start") from error
            raise unittest.SkipTest(f"WebKit container is unavailable: {error}")

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.webkit_container.stop()
        cls.server_context.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
