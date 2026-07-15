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
        for color in ("#121213", "#d7dadc", "#818384", "#538d4e", "#b59f3b", "#3a3a3c", "#cc4444", "#d0a215"):
            self.assertIn(color, self.html)
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
            cls.browser = cls.playwright.chromium.launch(headless=True)
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

    def apply_selector(self, selector):
        self.page.locator("#selector-input").fill(selector)
        self.page.locator("#apply").click()

    def test_selector_inference_has_no_type_chooser(self):
        self.apply_selector("CRANE")
        self.page.wait_for_selector("text=word report")
        self.apply_selector("CRANE .y..g")
        self.page.wait_for_selector("text=branch report")
        self.assertEqual(self.page.locator("[data-kind]").count(), 5)
        self.assertEqual(self.page.locator("text=Choose word or branch").count(), 0)

    def test_positional_cache_queue_and_explicit_navigation_urls(self):
        result = self.page.evaluate("""() => ({
          inferredCache: buildAPIURL(parsePageState({search:'?selector=CACHE'})),
          inferredQueue: buildAPIURL(parsePageState({search:'?selector=QUEUE'})),
          explicitCache: buildAPIURL(parsePageState({search:'?kind=cache'})),
          explicitQueue: buildAPIURL(parsePageState({search:'?kind=queue'}))
        })""")
        self.assertTrue(result["inferredCache"].startswith("/api/view?"))
        self.assertTrue(result["inferredQueue"].startswith("/api/view?"))
        self.assertEqual(result["explicitCache"], "/api/view/cache")
        self.assertEqual(result["explicitQueue"], "/api/view/queue")

    def test_tree_active_only_and_context_node(self):
        self.apply_selector("RAISE .....")
        self.page.locator("#tree").check()
        self.page.locator("#active-only").check()
        self.page.wait_for_selector("text=Live queue tree")
        self.assertIn("tree=1", self.page.url)
        self.assertIn("active_only=1", self.page.url)
        self.assertGreater(self.page.locator("text=pending").count(), 0)

    def test_word_group_click_builds_full_branch_spine(self):
        self.apply_selector("CACHE")
        self.page.wait_for_selector("text=word report")
        self.page.locator("article.card.clickable").first.click()
        self.assertIn("selector=CACHE+-----", self.page.url)

    def test_tree_branch_click_opens_detail(self):
        self.page.locator("[data-kind=queue]").click()
        self.page.locator("#tree").check()
        self.page.wait_for_selector(".tree button")
        self.page.locator(".tree button").first.click()
        self.page.wait_for_selector("text=branch report")

    def test_overview_renders_branch_and_worker_lifecycles(self):
        text = self.page.locator("#report").inner_text()
        self.assertIn("filesystem used", text)
        self.assertIn("12.5%", text)
        self.assertIn("queue WAL", text)
        self.assertIn("active", text)
        self.assertIn("finalizing", text)
        self.assertIn("worker-0", text)
        self.assertIn("worker-4", text)
        self.assertGreater(self.page.locator(".card.dead").count(), 0)

    def test_candidate_disclosure_requests_claims_and_labels_proof(self):
        requested = []
        self.page.on("request", lambda request: requested.append(request.url))
        self.apply_selector("RAISE .....")
        self.page.wait_for_selector("text=Candidate detail")
        self.page.locator("details:has-text('Candidate detail') summary").click()
        self.page.wait_for_timeout(150)
        self.assertTrue(any("claims=1" in url for url in requested))
        self.assertIn("Proof (bulk eliminated)", self.page.locator("#report").inner_text())
        self.assertNotIn("Worker bulk-elimination", self.page.locator("#report").inner_text())

    def test_finalization_outcomes_and_cut_reuse_are_distinct(self):
        self.apply_selector("RAISE .....")
        self.page.wait_for_selector("text=Recent finalizations")
        self.assertEqual(self.page.locator(".outcome-exact").count(), 1)
        self.assertEqual(self.page.locator(".outcome-cut").count(), 2)
        self.assertEqual(self.page.locator(".outcome-loss").count(), 1)
        self.assertIn("Cut-reuse misses", self.page.locator("#report").inner_text())

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
          changedTree.data.nodes[2].completed_candidate_count++;
          applyReport(changedTree,tree,{...state,tree:true});
          result.tree=document.querySelector('[data-identity="raise:-----/alibi:y----"]').className;

          const branch=await (await fetch('/api/view?selector=RAISE%20.....')).json();
          branch.data.workers=[{worker_id:'worker-12',updated_at:990,is_live:true,branch_key_hex:'01',branch_reference:'111111111111',current_candidate:'crane',current_max_guess_depth:2,nodes_per_second:10}];
          const changedBranch=structuredClone(branch);changedBranch.data.workers[0].current_candidate='slate';
          applyReport(changedBranch,branch,{...state,selector:'RAISE .....'});
          result.branch=document.querySelector('[data-identity="worker-12"]').className;

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
        self.assertIn("flash-changed", classes["branch"])
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
          const reordered=structuredClone(report);reordered.data.rows.reverse();reordered.data.rows[2].lifecycle='finalizing';
          applyReport(reordered,report,{...__reportClient.getState(),kind:'queue'});
          return [...document.querySelectorAll('[data-identity]')].map(n=>n.dataset.identity);
        }""")
        self.assertEqual(identities, ["01", "02", "03", "04"])

    def test_tree_collapse_and_browser_back_survive_poll(self):
        self.page.locator("[data-kind=queue]").click()
        self.page.locator("#tree").check()
        self.page.wait_for_selector(".tree details")
        details = self.page.locator(".tree details").first
        details.locator("summary").click()
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
        self.assertIn("disconnected", self.page.locator("#connection").inner_text())
        self.assertEqual(self.page.locator("#report").inner_text(), original)
        self.page.unroute("**/api/view**")
        self.page.evaluate("__reportClient.fetchReport()")
        self.page.wait_for_timeout(100)
        self.assertEqual(self.page.locator("#connection").inner_text(), "connected")

    def test_url_state_round_trips_statuses_and_filters(self):
        state = self.page.evaluate("""() => parsePageState({search:'?kind=queue&status=pending&status=done&limit=25&sort=nodes&poll=5000'})""")
        self.assertEqual(state["status"], ["pending", "done"])
        self.assertEqual(state["limit"], 25)
        self.assertEqual(state["sort"], "nodes")
        self.assertEqual(state["poll"], 5000)

    def test_state_normalization_removes_incompatible_controls(self):
        states = self.page.evaluate("""() => ({
          active: parsePageState({search:'?kind=queue&active_only=1&status=pending'}),
          tree: parsePageState({search:'?selector=RAISE%20.....&tree=1&claims=1&answers=1'}),
          word: parsePageState({search:'?selector=RAISE&sort=nodes'})
        })""")
        self.assertEqual(states["active"]["status"], [])
        self.assertFalse(states["tree"]["claims"])
        self.assertFalse(states["tree"]["answers"])
        self.assertEqual(states["word"]["sort"], "")

    def test_word_summary_keeps_unfiltered_totals(self):
        text = self.page.evaluate("""async () => {
          const report=await (await fetch('/api/view?selector=QUEUE')).json();
          const active=structuredClone(report);
          active.data.total_rows=4;active.data.matched_rows=1;
          active.data.response_groups=active.data.response_groups.filter(row=>row.lifecycle==='active');
          applyReport(active,null,{...__reportClient.getState(),selector:'QUEUE',active_only:true});
          return document.querySelector('#report').innerText;
        }""")
        self.assertIn("Shown 1 of 1 matched · 4 total response groups", text)
        self.assertIn("response group count", text)

    def test_cache_renderer_handles_root_word_and_branch_contracts(self):
        identities = self.page.evaluate("""async () => {
          const root=await (await fetch('/api/view/cache')).json();
          applyReport(root,null,{...__reportClient.getState(),kind:'cache'});
          const rootIdentities=[...document.querySelectorAll('[data-identity]')].map(node=>node.dataset.identity);
          const word=structuredClone(root);word.data={summary:{response_group_count:1},rows:[{branch_key_hex:'word-key',branch_reference:'word-ref',cache_state:'missing'}]};
          applyReport(word,null,{...__reportClient.getState(),kind:'cache',selector:'RAISE'});
          const wordIdentities=[...document.querySelectorAll('[data-identity]')].map(node=>node.dataset.identity);
          const branch=structuredClone(root);branch.data={branch_key_hex:'branch-key',branch_reference:'branch-ref',cache:{cache_state:'exact',best_guess:'crane',best_erd:2.1}};
          applyReport(branch,null,{...__reportClient.getState(),kind:'cache',selector:'RAISE .....'});
          return {rootIdentities,wordIdentities,branchIdentities:[...document.querySelectorAll('[data-identity]')].map(node=>node.dataset.identity)};
        }""")
        self.assertEqual(identities["rootIdentities"], ["01", "02"])
        self.assertEqual(identities["wordIdentities"], ["word-key"])
        self.assertEqual(identities["branchIdentities"], ["branch-key"])

    def test_stale_request_cannot_replace_newer_navigation(self):
        result = self.page.evaluate("""async () => {
          const originalFetch=window.fetch.bind(window);
          const overview=await (await originalFetch('/api/view')).json();
          const branch=await (await originalFetch('/api/view?selector=RAISE%20.....')).json();
          let releaseOverview;
          window.fetch=(url)=>url.includes('selector=RAISE')
            ? Promise.resolve(new Response(JSON.stringify(branch),{status:200,headers:{'Content-Type':'application/json'}}))
            : new Promise(resolve=>{releaseOverview=()=>resolve(new Response(JSON.stringify(overview),{status:200,headers:{'Content-Type':'application/json'}}));});
          __reportClient.setState({...__reportClient.getState(),kind:'auto',selector:''});
          __reportClient.setState({...__reportClient.getState(),kind:'auto',selector:'RAISE .....'});
          await new Promise(resolve=>setTimeout(resolve,20));
          releaseOverview();await new Promise(resolve=>setTimeout(resolve,20));
          window.fetch=originalFetch;
          return {heading:document.querySelector('h1').textContent,selector:__reportClient.getState().selector};
        }""")
        self.assertEqual(result["heading"], "branch report")
        self.assertEqual(result["selector"], "RAISE .....")

    def test_malicious_text_is_literal_and_inert(self):
        result = self.page.evaluate("""async () => {
          const report=await (await fetch('/api/view')).json();
          report.sources.queue.error='<img id="owned" src=x onerror="window.owned=1">';
          applyReport(report,null,__reportClient.getState());
          return {owned:window.owned||0,node:!!document.querySelector('#owned'),text:document.querySelector('#report').innerText};
        }""")
        self.assertEqual(result["owned"], 0)
        self.assertFalse(result["node"])
        self.assertIn("<img id=", result["text"])

    def test_tile_colors_match_wordle_palette(self):
        self.apply_selector("CACHE")
        self.page.wait_for_selector(".tile")
        colors = self.page.evaluate("""() => {
          const result={};
          for(const name of ['g','y','']){const node=document.createElement('span');node.className='tile '+name;document.body.append(node);result[name||'gray']=getComputedStyle(node).backgroundColor;node.remove();}
          return result;
        }""")
        declared = self.page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--green').trim()")
        self.assertEqual(declared, "#538d4e")
        self.assertEqual(colors["g"], "rgb(83, 141, 78)")
        self.assertEqual(colors["y"], "rgb(181, 159, 59)")
        self.assertEqual(colors["gray"], "rgb(58, 58, 60)")

    def test_no_horizontal_scroll_at_required_widths(self):
        for width in (375, 390, 480, 800, 1200):
            with self.subTest(width=width):
                self.page.set_viewport_size({"width": width, "height": 800})
                overflow = self.page.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth")
                self.assertFalse(overflow)

    def test_review_screenshots_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            for width in (390, 1200):
                for name, path in (
                    ("overview", ""), ("word", "?selector=CACHE"),
                    ("branch", "?selector=RAISE+....."),
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
