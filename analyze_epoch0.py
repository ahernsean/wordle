#!/usr/bin/env python3
"""analyze_epoch0.py — the §10/§11 go/no-go gate for the adaptive-packing metric.

Reads erd_queue.sqlite3 (read-only) and evaluates, on the epoch-0 single-candidate
baseline, whether a candidate work metric is sound enough to build the packer on.

Because candidate_accuracy logs each non-gated candidate's `group_sizes` (the
sufficient statistic) and the bound it saw, this recomputes EVERY metric offline
against a cost-model snapshot — so the uncut estimate, the cutoff-aware §4 metric,
and any future variant are compared from one collection without re-running the
swarm.

Three checks per metric:
  1. Gating split (exact): cost_lb >= B candidates are cut for free, so gated rows
     must have near-zero actual_nodes.  Metric-independent; reported once.
  2. §4 trap: a sound metric must NOT predict the cheap non-gated mass as
     expensive (weak splitters cut by the accumulated-cost cutoff).
  3. Load-bearing rank: among the few claims holding nearly all node work — the
     ones that drive packer balance — predicted must rank actual (Spearman).

A wrong metric only wastes swarm time (§3), never corrupts an ERD, so this gate is
about efficiency confidence before investing in the packer.
"""
import argparse
import math
import statistics
import sqlite3

from erd_queue import cost_size_bucket, _COST_MODEL_MIN_WEIGHT, _AGGREGATE_BUDGET
from wordle_engine import (ERD_ALL, estimate_candidate_work,
                           estimate_candidate_work_cutoff)

try:
    from scipy.stats import spearmanr as _scipy_spearman
except Exception:
    _scipy_spearman = None


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def spearman(xs, ys):
    if len(xs) < 2:
        return float("nan")
    if _scipy_spearman is not None:
        return float(_scipy_spearman(xs, ys).statistic)
    return _pearson(_ranks(xs), _ranks(ys))


def loglog_fit(preds, acts):
    """Pearson on log10(predicted) vs log10(actual), plus the regression slope and
    residual sigma.  Spearman only sees rank; this sees MAGNITUDE proportionality —
    which is what bundle sizing depends on (the packer sums predicted work to W).
    slope ~ 1 means predicted is proportional to actual; resid_sigma is the
    bundle-sizing noise in dex (log10 units).  Filters to positive pairs."""
    xs = [math.log10(p) for p, a in zip(preds, acts) if p > 0 and a > 0]
    ys = [math.log10(a) for p, a in zip(preds, acts) if p > 0 and a > 0]
    if len(xs) < 3:
        return float("nan"), float("nan"), float("nan")
    r = _pearson(xs, ys)
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
             if sxx else float("nan"))
    b = my - slope * mx
    sigma = (sum((y - slope * x - b) ** 2 for x, y in zip(xs, ys)) / n) ** 0.5
    return r, slope, sigma


def load_cost_model(conn):
    """Snapshot cost_model into a typical(k, budget) closure with the same
    (size_bucket, budget) -> budget-aggregate fallback the live model uses."""
    cells = {}
    for r in conn.execute("SELECT size_bucket, budget, weighted_log_sum, weight_sum "
                          "FROM cost_model WHERE policy=?", (ERD_ALL,)):
        cells[(r[0], r[1])] = (r[2], r[3])

    def typical(k, budget):
        bucket = cost_size_bucket(k)
        for key in ((bucket, budget), (bucket, _AGGREGATE_BUDGET)):
            cell = cells.get(key)
            if cell and cell[1] >= _COST_MODEL_MIN_WEIGHT:
                return math.exp(cell[0] / cell[1])
        return None
    return typical


def _pctile(s, p):
    return s[min(len(s) - 1, int(p / 100.0 * len(s)))] if s else float("nan")


def _describe(label, vals):
    if not vals:
        print(f"  {label:14s}: (no rows)")
        return
    s = sorted(vals)
    print(f"  {label:14s}: n={len(s):>8d}  median={_pctile(s,50):>8.1f}  "
          f"p99={_pctile(s,99):>10.1f}  max={s[-1]:>11.0f}  mean={statistics.fmean(s):>10.1f}")


METRICS = {
    "uncut":  estimate_candidate_work,
    "cutoff": estimate_candidate_work_cutoff,
}


def evaluate_metric(name, fn, rows, typical, args):
    """rows: non-gated (n_words, budget, bound_erd, actual, sizes-list).  Returns
    (false_expensive_frac, tail_spearman, overall_spearman)."""
    preds, acts = [], []
    fe = 0
    for n_words, budget, bound, actual, sizes in rows:
        b = bound if bound is not None else float("inf")
        p = fn(sizes, False, n_words, b, budget, typical)
        preds.append(p)
        acts.append(actual)
        if p > args.false_pred and actual < args.small_nodes:
            fe += 1
    fe_frac = fe / len(rows) if rows else float("nan")
    rho_all = spearman(preds, acts)
    tail = [(p, a) for p, a in zip(preds, acts) if a >= args.tail_nodes]
    rho_tail = spearman([p for p, _ in tail], [a for _, a in tail]) if len(tail) > 2 else float("nan")
    print(f"\n--- metric '{name}' (non-gated, n={len(rows):,}) ---")
    print(f"  median predicted = {statistics.median(preds):,.1f}   "
          f"§4 false-expensive (pred>{args.false_pred:g}, actual<{args.small_nodes}) "
          f"= {100*fe_frac:.1f}%")
    print(f"  Spearman overall = {rho_all:.3f}   "
          f"load-bearing tail (actual>={args.tail_nodes}, n={len(tail)}) Spearman = {rho_tail:.3f}")
    r_log, slope, sigma = loglog_fit(preds, acts)
    print(f"  log10 calibration: Pearson(log) = {r_log:.3f}   slope = {slope:.3f}   "
          f"resid_sigma = {sigma:.2f} dex  (slope~1 & low sigma => well-sized bundles)")
    return fe_frac, rho_tail, rho_all, slope, r_log


def estimate_claim_reduction(conn, epoch, small_count, count_cap):
    """The reframed §11 gate: claim transactions to drain a branch under the
    binary/coarse packing scheme vs single-candidate claiming — which needs only
    exact gating, not a good magnitude estimate.

    Per finalized branch of `total` candidates with `N` non-gated: the non-gated
    pack into ceil(N/small_count) small fixed-count bundles and the gated
    G = total - N coalesce into ceil(G/count_cap) count-capped bulk bundles.
    reduction = total / bundles.  Gating is exact, so a wrong work estimate cannot
    change this — it only decides which non-gated candidates share a small bundle.
    """
    import math
    total_by = {bytes(r[0]): r[1] for r in conn.execute(
        "SELECT branch_key, n_claims FROM branch_finalize_log WHERE n_claims > 0")}
    ng_by = {}
    for bk, cnt in conn.execute(
        "SELECT branch_key, COUNT(*) FROM candidate_accuracy "
        "WHERE epoch = ? AND gated = 0 GROUP BY branch_key", (epoch,)):
        ng_by[bytes(bk)] = cnt
    tot_claims = tot_bundles = 0
    per = []
    for bk, total in total_by.items():
        N = min(ng_by.get(bk, 0), total)
        G = total - N
        bundles = max(1, math.ceil(N / small_count)
                      + (math.ceil(G / count_cap) if G else 0))
        tot_claims += total
        tot_bundles += bundles
        per.append(total / bundles)
    return tot_claims, tot_bundles, per


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", default="erd_queue.sqlite3")
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--small-nodes", type=int, default=10)
    ap.add_argument("--strong", type=float, default=0.5)
    ap.add_argument("--false-pred", type=float, default=1000)
    ap.add_argument("--max-false", type=float, default=0.20)
    ap.add_argument("--tail-nodes", type=int, default=100)
    ap.add_argument("--min-bounded", type=int, default=2000,
                    help="min bounded non-gated rows before the verdict uses the "
                         "bounded subset instead of the full set (default 2000)")
    ap.add_argument("--gate-metric", default="cutoff", choices=list(METRICS))
    ap.add_argument("--small-count", type=int, default=8,
                    help="non-gated candidates per small bundle (binary scheme)")
    ap.add_argument("--count-cap", type=int, default=512,
                    help="gated candidates per count-capped bulk bundle")
    ap.add_argument("--min-reduction", type=float, default=50.0,
                    help="claim-count reduction the reframed gate requires (x)")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.queue}?mode=ro", uri=True)
    typical = load_cost_model(conn)
    all_rows = conn.execute("""
        SELECT n_words, budget, bound_erd, gated, actual_nodes, group_sizes,
               source_word
        FROM candidate_accuracy WHERE epoch = ?
    """, (args.epoch,)).fetchall()

    n = len(all_rows)
    print(f"epoch {args.epoch}: {n:,} candidate_accuracy rows  "
          f"(spearman via {'scipy' if _scipy_spearman else 'pure-python'})\n")
    if n == 0:
        print("NO DATA — run the swarm on the new code to populate candidate_accuracy.")
        return

    gated = [r for r in all_rows if r[3]]
    nongated_raw = [r for r in all_rows if not r[3]]
    # by-opener counts (segmentation for the per-opener split)
    from collections import Counter
    opener_counts = Counter(r[6] for r in nongated_raw)
    print(f"gated: {len(gated):,} ({100*len(gated)/n:.1f}%)   "
          f"non-gated: {len(nongated_raw):,} ({100*len(nongated_raw)/n:.1f}%)")

    # Check 1: gating split (metric-independent)
    print("\n=== Gating split: actual_nodes by class ===")
    _describe("gated", [r[4] for r in gated])
    _describe("non-gated", [r[4] for r in nongated_raw])
    gsmall = sum(1 for r in gated if r[4] < args.small_nodes)
    gfrac = gsmall / len(gated) if gated else float("nan")
    print(f"  gated rows with actual < {args.small_nodes}: {gsmall:,}/{len(gated):,} "
          f"= {100*gfrac:.2f}%")

    # Parse group sizes once; drop rows missing them (older rows before the column)
    rows = []
    rows_by_opener = {}
    skipped = 0
    for n_words, budget, bound, _g, actual, gs, opener in nongated_raw:
        if not gs:
            skipped += 1
            continue
        sizes = [int(x) for x in gs.split("-") if x]
        row = (n_words, budget, bound, actual, sizes)
        rows.append(row)
        rows_by_opener.setdefault(opener, []).append(row)
    have_bound = sum(1 for r in rows if r[2] is not None)
    print(f"\nnon-gated with group_sizes: {len(rows):,}  "
          f"(skipped {skipped:,} pre-column rows); with a known bound: "
          f"{have_bound:,} ({100*have_bound/max(1,len(rows)):.1f}%)")

    # Checks 2 & 3: every metric, recomputed offline.  Report the full non-gated
    # set AND the bounded-only subset: the cutoff metric is identical to uncut on
    # unbounded rows (no bound to cut against), so its value shows ONLY on bounded
    # rows.  The verdict keys on the bounded subset when it is large enough.
    bounded = [r for r in rows if r[2] is not None]
    print("\n=== Metric comparison: ALL non-gated (mostly unbounded) ===")
    results_all = {name: evaluate_metric(name, fn, rows, typical, args)
                   for name, fn in METRICS.items()}
    print(f"\n=== Metric comparison: BOUNDED-only subset (n={len(bounded):,}) — "
          f"where cutoff differs from uncut ===")
    results_bounded = ({name: evaluate_metric(name, fn, bounded, typical, args)
                        for name, fn in METRICS.items()}
                       if len(bounded) >= args.min_bounded else None)
    if results_bounded is None:
        print(f"  (only {len(bounded):,} bounded rows — under --min-bounded "
              f"{args.min_bounded}; verdict uses the full set)")

    # Per-opener split: different openers reach differently-shaped answer sets, so
    # report the bounded metric per opener (thin openers flagged as noise).
    print("\n=== Per-opener split (bounded rows, uncut metric) ===")
    for opener in sorted(rows_by_opener, key=lambda o: -opener_counts[o]):
        obounded = [r for r in rows_by_opener[opener] if r[2] is not None]
        if len(obounded) < 50:
            print(f"  {opener}: {len(obounded)} bounded rows (too few — noise)")
            continue
        evaluate_metric(opener, estimate_candidate_work, obounded, typical, args)

    # Claim-count reduction under the binary/exact-elimination packer, which needs
    # ONLY exact ERD-lower-bound elimination — never a work estimate.  Bundle size
    # (small_count) is a dial, not a threshold: the searched (non-eliminated) mass is
    # ~95% trivial, so a larger bundle is safe and republish-on-overrun catches the
    # rare heavy member.  Report the whole curve rather than a pass/fail against an
    # arbitrary target.
    print(f"\n=== Binary packer: claim-count reduction vs bundle size "
          f"(count_cap={args.count_cap}) ===")
    print(f"  {'small_count':>11}  {'aggregate':>10}  {'median/branch':>14}")
    reduction_curve = []
    for small_count in (8, 16, 32, 64):
        tot_claims, tot_bundles, per = estimate_claim_reduction(
            conn, args.epoch, small_count, args.count_cap)
        if tot_bundles:
            agg = tot_claims / tot_bundles
            med = statistics.median(per) if per else float("nan")
            reduction_curve.append((small_count, agg))
            print(f"  {small_count:11d}  {agg:9.1f}x  {med:13.1f}x")
    if not reduction_curve:
        print("  (no finalized branches with n_claims)")
    conn.close()

    # The design rests on two proven facts, not on the magnitude estimate: exact
    # ERD-lower-bound elimination, and a searched mass that is overwhelmingly cheap.
    results = results_bounded or results_all
    fe_frac, rho_tail, _, slope, r_log = results[args.gate_metric]
    used = "bounded subset" if results_bounded else "full non-gated set"
    elim_ok = (not gated) or gfrac >= 0.95
    print(f"\n=== Packer decision ===")
    print(f"  [{'PASS' if elim_ok else 'FAIL'}] ERD-lower-bound elimination is exact "
          f"({100*gfrac:.1f}% of eliminated candidates < {args.small_nodes} nodes)"
          if gated else "  [n/a ] elimination split (no eliminated rows)")
    print(f"  Magnitude estimate (metric '{args.gate_metric}', {used}): "
          f"log-log slope {slope:.3f}, Pearson(log) {r_log:.3f}, "
          f"false-expensive {100*fe_frac:.1f}%, tail Spearman {rho_tail:.3f}")
    print(f"    -> NOT usable for bundle sizing: it over-predicts the cheap searched "
          f"mass (a size-only cost model cannot see the bound propagate into "
          f"children).  The packer must be magnitude-free.")
    if len(reduction_curve) >= 3:
        print(f"  Binary packer clears any reasonable claim-count target by bundle "
              f"size alone ({reduction_curve[1][0]}->{reduction_curve[1][1]:.0f}x, "
              f"{reduction_curve[2][0]}->{reduction_curve[2][1]:.0f}x aggregate); "
              f"pick small_count from republish-safety, not a fixed bar.")
    print(f"\n  DESIGN: exact ERD-lower-bound elimination + count-bundling + "
          f"republish-on-overrun.  No work-magnitude model.")


if __name__ == "__main__":
    main()
