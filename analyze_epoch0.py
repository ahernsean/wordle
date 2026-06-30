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
from wordle_engine import estimate_candidate_work, estimate_candidate_work_cutoff

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


def load_cost_model(conn):
    """Snapshot cost_model into a typical(k, budget) closure with the same
    (size_bucket, budget) -> budget-aggregate fallback the live model uses."""
    cells = {}
    for r in conn.execute("SELECT size_bucket, budget, weighted_log_sum, weight_sum "
                          "FROM cost_model WHERE policy='erd_all'"):
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
    return fe_frac, rho_tail, rho_all


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
    ap.add_argument("--gate-metric", default="cutoff", choices=list(METRICS))
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.queue}?mode=ro", uri=True)
    typical = load_cost_model(conn)
    all_rows = conn.execute("""
        SELECT n_words, budget, bound_erd, gated, actual_nodes, group_sizes
        FROM candidate_accuracy WHERE epoch = ?
    """, (args.epoch,)).fetchall()
    conn.close()

    n = len(all_rows)
    print(f"epoch {args.epoch}: {n:,} candidate_accuracy rows  "
          f"(spearman via {'scipy' if _scipy_spearman else 'pure-python'})\n")
    if n == 0:
        print("NO DATA — run the swarm on the new code to populate candidate_accuracy.")
        return

    gated = [r for r in all_rows if r[3]]
    nongated_raw = [r for r in all_rows if not r[3]]
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
    skipped = 0
    for n_words, budget, bound, _g, actual, gs in nongated_raw:
        if not gs:
            skipped += 1
            continue
        sizes = [int(x) for x in gs.split("-") if x]
        rows.append((n_words, budget, bound, actual, sizes))
    have_bound = sum(1 for r in rows if r[2] is not None)
    print(f"\nnon-gated with group_sizes: {len(rows):,}  "
          f"(skipped {skipped:,} pre-column rows); with a known bound: "
          f"{have_bound:,} ({100*have_bound/max(1,len(rows)):.1f}%)")

    # Checks 2 & 3: every metric, recomputed offline
    print("\n=== Metric comparison (recomputed offline from group_sizes) ===")
    results = {name: evaluate_metric(name, fn, rows, typical, args)
               for name, fn in METRICS.items()}

    # Verdict on the chosen packer metric
    fe_frac, rho_tail, _ = results[args.gate_metric]
    print(f"\n=== §11 go/no-go  (gate metric: '{args.gate_metric}') ===")
    gate_ok = (not gated) or gfrac >= 0.95
    fe_ok = not math.isnan(fe_frac) and fe_frac <= args.max_false
    tail_ok = not math.isnan(rho_tail) and rho_tail >= args.strong
    print(f"  [{'PASS' if gate_ok else 'FAIL'}] gating split exact "
          f"({100*gfrac:.1f}% near-zero)" if gated else "  [n/a ] gating split")
    print(f"  [{'PASS' if fe_ok else 'FAIL'}] §4 trap avoided "
          f"(<= {100*args.max_false:.0f}% false-expensive, got {100*fe_frac:.1f}%)")
    print(f"  [{'PASS' if tail_ok else 'FAIL'}] load-bearing rank "
          f"(tail Spearman >= {args.strong}, got {rho_tail:.3f})")
    go = gate_ok and fe_ok and tail_ok
    print(f"\n  VERDICT: {'GO — build the packer on this metric' if go else 'NO-GO (refine metric before packer)'}")


if __name__ == "__main__":
    main()
