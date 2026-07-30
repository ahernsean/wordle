"""Browser contract tests for the self-contained report client."""

from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from html.parser import HTMLParser
import os
import re
import shutil
import subprocess
import tempfile
from threading import Thread
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

    def test_wordle_palette_and_responsive_breakpoint_are_declared(self):
        for color in (
            "#ffffff", "#f8f9fa", "#1a1a1b", "#787c7e", "#6aaa64",
            "#c9b458", "#d3d6da", "#d14b4b", "#b59f3b",
        ):
            self.assertIn(color, self.html)
        self.assertIn("color-scheme: light", self.html)
        self.assertIn("@media (max-width:600px)", self.html)


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

    def test_answer_word_count_is_shown_before_expansion(self):
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("text=branch report")
        summary = self.page.locator("summary:has-text('Answer words')")
        self.assertEqual(summary.inner_text(), "Answer words (8)")
        self.assertIsNone(summary.locator("xpath=..").get_attribute("open"))

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
        group = self.page.locator("ul.tree > li.word-group")
        self.assertEqual(group.count(), 1)
        self.assertEqual(
            " ".join(group.locator("> details > summary").inner_text().split()),
            "RAISE 1 branch",
        )
        rows = group.locator("> details > ul.patterns > li")
        self.assertEqual(rows.count(), 1)
        # The group names the word, so its rows carry only the response pattern.
        row_summary = rows.locator("> details > summary")
        self.assertNotIn("RAISE", row_summary.inner_text())
        self.assertEqual(row_summary.locator(".step").count(), 1)
        # A row's own children are grouped by word the same way.  The branch
        # tree is awaited by its nested group: a bare top-level group is
        # already on the page from the queue tree, so waiting for one would
        # match the view being navigated away from.
        nested_selector = (
            "ul.tree > li.word-group > details > ul.patterns > li"
            " > details > ul > li.word-group"
        )
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector(nested_selector)
        nested = self.page.locator(nested_selector)
        self.assertEqual(nested.count(), 1)
        self.assertEqual(
            " ".join(nested.locator("> details > summary").inner_text().split()),
            "ALIBI 1 branch",
        )
        nested_rows = nested.locator("> details > ul.patterns > li")
        self.assertEqual(nested_rows.count(), 1)
        self.assertNotIn(
            "ALIBI", nested_rows.locator("> details > summary").inner_text()
        )

    def test_tree_row_facts_never_split_a_number_from_its_noun(self):
        self.page.locator("[data-kind=queue]").click()
        self.page.locator("#layout-tree").click()
        self.page.wait_for_selector("ul.tree .inline-facts")
        facts = self.page.locator("ul.tree li:not(.word-group) > details > summary .inline-facts").first
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

    def test_word_view_shows_infeasible_erd_for_a_proven_loss(self):
        # Served straight from the fixture (loss group present), so this pins
        # the renderer to the model's real erd_summary field names.
        self.apply_branch_target("SALET")
        self.page.wait_for_selector("text=word report")
        text = self.page.locator("#report").inner_text()
        self.assertIn("1 of 4 response groups unsolvable within budget", text)

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

    def test_leaderboard_tab_ranks_openers_and_rounds_erd(self):
        self.page.locator("[data-kind=leaderboard]").click()
        self.page.wait_for_selector("text=Opener leaderboard")
        text = self.page.locator("#report").inner_text()
        self.assertIn("SALET", text)
        self.assertIn("3.564", text)
        self.assertNotIn("3.5643502648", text)
        self.assertIn("CRANE*", text)  # word_is_answer renders the asterisk

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
        self.assertIn("1,619 bulk proofs", text)
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
          return document.querySelector('#report').innerText;
        }""")
        self.assertIn("Branch ownership", text)
        self.assertIn("Live workers", text)
        self.assertIn("None", text)
        self.assertIn("Claim holders off-branch", text)
        self.assertIn("w5", text)
        self.assertIn("SALET", text)
        self.assertIn("\ng\n", text)

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

    def test_finalizations_are_glossed_and_timestamped(self):
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("text=Recent finalizations")
        text = self.page.locator("section:has-text('Recent finalizations')").inner_text()
        self.assertIn("Cut — best line exceeds the budget", text)
        self.assertIn("Exact — solved within budget", text)
        self.assertIn("Loss — unsolvable in the game", text)
        self.assertIn("solution not recorded", text)
        self.assertIn("newest first", text)
        self.assertIn("ago", text)
        self.assertIn("budget", text)

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
        self.assertIn("solved CIGAR / ERD 1.875", text)
        self.assertNotIn("solution not recorded", text)

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
          navigator.clipboard = undefined;
          window.__copiedSpine = false;
          document.execCommand = command => {
            window.__copiedSpine = command === 'copy';
            return window.__copiedSpine;
          };
        }""")
        copy_button.click()
        self.page.wait_for_function(
            "() => window.__copiedSpine === true"
        )

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
        self.assertIn("available 2.200 at budget 4", text)
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
        self.page.wait_for_selector(".tree details")
        details = self.page.locator(".tree details").first
        details.locator("summary").first.click()
        self.assertFalse(details.get_attribute("open") is not None)
        self.page.evaluate("__reportClient.fetchReport()")
        self.page.wait_for_timeout(100)
        self.assertIsNone(self.page.locator(".tree details").first.get_attribute("open"))
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

    def test_overview_cards_animate_moves_and_departures(self):
        result = self.page.evaluate("""async () => {
          const report=await (await fetch('/api/view')).json();
          applyReport(report,null,__reportClient.getState());
          const before=[...document.querySelectorAll('.grid > [data-identity]')].map(node=>node.dataset.identity);
          const reordered=structuredClone(report);
          reordered.data.branches.reverse();
          reordered.data.branches.shift();
          applyReport(reordered,report,__reportClient.getState());
          const moved=[...document.querySelectorAll('.grid > [data-identity]')].filter(node=>node.getAnimations().length).length;
          const leaveClones=document.querySelectorAll('.leave-layer > *').length;
          return {before,moved,leaveClones};
        }""")
        self.assertGreater(len(result["before"]), 1)
        self.assertGreater(result["moved"], 0)
        self.assertEqual(result["leaveClones"], 1)

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
        self.assertEqual(states["word"]["sort"], "")

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
        self.page.wait_for_selector(".tile")
        colors = self.page.evaluate("""() => {
          const result={};
          for(const name of ['g','y','']){const node=document.createElement('span');node.className='tile '+name;document.body.append(node);result[name||'gray']=getComputedStyle(node).backgroundColor;node.remove();}
          return result;
        }""")
        declared = self.page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--green').trim()")
        self.assertEqual(declared, "#6aaa64")
        self.assertEqual(colors["g"], "rgb(106, 170, 100)")
        self.assertEqual(colors["y"], "rgb(201, 180, 88)")
        self.assertEqual(colors["gray"], "rgb(120, 124, 126)")

    def test_no_horizontal_scroll_at_required_widths(self):
        # Every view, not just whichever one setUp left loaded: the tree view
        # reached phone widths overflowing because it was never measured here.
        for path in (
            "", "?kind=queue", "?kind=workers", "?kind=cache", "?kind=hotspots",
            "?kind=leaderboard", "?kind=queue&tree=1", "?branch_target=RAISE+.....",
            "?branch_target=RAISE+.....&tree=1",
        ):
            self.page.goto(self.base_url + path)
            self.page.wait_for_selector("h1")
            for width in (375, 390, 480, 800, 1200):
                with self.subTest(path=path or "overview", width=width):
                    self.page.set_viewport_size({"width": width, "height": 800})
                    measured = self.page.evaluate(
                        "() => ({scroll: document.documentElement.scrollWidth,"
                        " client: document.documentElement.clientWidth})"
                    )
                    self.assertLessEqual(measured["scroll"], measured["client"])

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

    def test_spine_words_never_separate_from_their_patterns(self):
        result = self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
          const groups=[...document.querySelectorAll('.tiles .step-group')];
          const spineGroups=groups.map(group=>({text:group.textContent,hasTiles:!!group.querySelector('.step'),noWrap:getComputedStyle(group).whiteSpace==='nowrap'}));
          const tree=await (await fetch('/api/view?tree=1')).json();
          applyReport(tree,null,{...__reportClient.getState(),branch_target:'',tree:true});
          const treeGroups=[...document.querySelectorAll('summary .step-group')].map(group=>({hasTiles:!!group.querySelector('.step'),noWrap:getComputedStyle(group).whiteSpace==='nowrap'}));
          return {spineGroups,treeGroupCount:treeGroups.length,treeAllNoWrap:treeGroups.every(group=>group.noWrap)};
        }""")
        self.assertTrue(result["spineGroups"])
        for group in result["spineGroups"]:
            self.assertTrue(group["hasTiles"])
            self.assertTrue(group["noWrap"])
        self.assertGreater(result["treeGroupCount"], 0)
        self.assertTrue(result["treeAllNoWrap"])

    def test_integers_use_comma_separators(self):
        self.apply_branch_target("RAISE .....")
        self.page.wait_for_selector("text=branch report")
        text = self.page.locator("#report").inner_text()
        self.assertIn("12,000", text)
        self.assertNotIn("12000", text)

    def test_candidates_are_uppercase_with_answer_asterisk(self):
        text = self.page.evaluate("""async () => {
          const branch=await (await fetch('/api/view?branch_target=RAISE%20.....')).json();
          branch.data.workers=[{worker_id:'worker-3',worker_number:'3',updated_at:999,is_live:true,branch_key_hex:'01',branch_reference:'111111111111',current_candidate:'crane',current_candidate_is_answer:true,current_max_guess_depth:2,nodes_per_second:10}];
          applyReport(branch,null,{...__reportClient.getState(),branch_target:'RAISE .....'});
          return document.querySelector('#report').innerText;
        }""")
        self.assertIn("CRANE*", text)
        self.assertNotIn("crane", text)

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
