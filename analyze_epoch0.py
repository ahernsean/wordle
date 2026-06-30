#!/usr/bin/env python3
"""analyze_epoch0.py — the §10/§11 go/no-go gate for adaptive claim packing.

Reads erd_queue.sqlite3 (read-only) and evaluates, on the epoch-0 single-candidate
baseline, whether the packer's work metric is sound enough to build the packer on:

  1. Gating split (exact part): a candidate with cost_lb >= B is provably cut for
     free, so gated rows must have near-zero actual_nodes.  We report the actual
     distribution for gated vs non-gated and the gated fraction under a small-node
     threshold.

  2. Predicted-vs-actual correlation (estimate part): on the NON-gated rows (where
     predicted_work is a real estimate, not the exact 0), Spearman rank correlation
     between predicted_work and actual_nodes must be strongly positive.  Spearman,
     not Pearson, because both are heavy-tailed and we only need monotonicity for
     scheduling.

A wrong metric can only waste swarm time (§3), never corrupt an ERD, so this gate is
about *efficiency confidence* before investing in the packer — not correctness.
"""
import argparse
import math
import sqlite3
import statistics


def _ranks(xs):
    """Average-rank vector for xs (ties share the mean of their rank span)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0   # 1-based average rank over the tie block
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
    return _pearson(_ranks(xs), _ranks(ys))


def _pctile(sorted_xs, p):
    if not sorted_xs:
        return float("nan")
    idx = min(len(sorted_xs) - 1, int(p / 100.0 * len(sorted_xs)))
    return sorted_xs[idx]


def _describe(label, vals):
    if not vals:
        print(f"  {label:14s}: (no rows)")
        return
    s = sorted(vals)
    print(f"  {label:14s}: n={len(s):>8d}  "
          f"median={_pctile(s,50):>10.1f}  p90={_pctile(s,90):>11.1f}  "
          f"p99={_pctile(s,99):>12.1f}  max={s[-1]:>12.0f}  mean={statistics.fmean(s):>11.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", default="erd_queue.sqlite3")
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--small-nodes", type=int, default=10,
                    help="actual_nodes below this counts as 'near-zero' (default 10)")
    ap.add_argument("--strong", type=float, default=0.5,
                    help="Spearman at/above this is 'strong' (default 0.5)")
    ap.add_argument("--false-pred", type=float, default=1000,
                    help="predicted_work above this with tiny actual = a §4 "
                         "false-expensive row (default 1000)")
    ap.add_argument("--max-false", type=float, default=0.20,
                    help="max tolerable fraction of false-expensive non-gated "
                         "rows (default 0.20)")
    ap.add_argument("--tail-nodes", type=int, default=100,
                    help="actual_nodes at/above this is the load-bearing tail "
                         "(default 100)")
    args = ap.parse_args()

    c = sqlite3.connect(f"file:{args.queue}?mode=ro", uri=True)
    rows = c.execute("""
        SELECT n_words, budget, predicted_work, bound_erd, cost_lb, gated,
               actual_nodes
        FROM candidate_accuracy WHERE epoch = ?
    """, (args.epoch,)).fetchall()
    c.close()

    n = len(rows)
    print(f"epoch {args.epoch}: {n:,} candidate_accuracy rows\n")
    if n == 0:
        print("NO DATA — run the swarm on the new code to populate candidate_accuracy.")
        return

    gated = [r for r in rows if r[5]]
    nongated = [r for r in rows if not r[5]]
    print(f"gated (cost_lb >= B): {len(gated):,} ({100*len(gated)/n:.1f}%)   "
          f"non-gated: {len(nongated):,} ({100*len(nongated)/n:.1f}%)\n")

    # --- Check 1: gating split is exact (gated ⇒ near-zero nodes) -------------
    print("=== Gating split: actual_nodes by class ===")
    _describe("gated", [r[6] for r in gated])
    _describe("non-gated", [r[6] for r in nongated])
    gated_small = sum(1 for r in gated if r[6] < args.small_nodes)
    gated_small_frac = gated_small / len(gated) if gated else float("nan")
    print(f"\n  gated rows with actual_nodes < {args.small_nodes}: "
          f"{gated_small:,}/{len(gated):,} = {100*gated_small_frac:.2f}%")

    # --- Check 2: predicted vs actual correlation on the non-gated estimate ---
    print("\n=== Predicted-vs-actual on NON-gated rows ===")
    pa_pred = [r[2] for r in nongated if r[2] is not None]
    pa_act = [r[6] for r in nongated if r[2] is not None]
    rho_all = spearman(pa_pred, pa_act)
    print(f"  Spearman(predicted_work, actual_nodes) [all non-gated] = {rho_all:.3f}")

    print("\n  per budget (non-gated):")
    print(f"  {'budget':>6} {'n':>8} {'spearman':>9} {'med_pred':>10} {'med_act':>10}")
    for b in sorted({r[1] for r in nongated if r[1] is not None}):
        sub = [r for r in nongated if r[1] == b and r[2] is not None]
        if len(sub) < 2:
            continue
        ps = [r[2] for r in sub]
        as_ = [r[6] for r in sub]
        print(f"  {b:>6} {len(sub):>8} {spearman(ps, as_):>9.3f} "
              f"{statistics.median(ps):>10.1f} {statistics.median(as_):>10.1f}")

    # Whole-population monotonicity (gated 0s included) — the metric as the packer
    # actually uses it: gated→0 predicted should sit below the non-gated mass.
    rho_full = spearman([r[2] for r in rows if r[2] is not None],
                        [r[6] for r in rows if r[2] is not None])
    print(f"\n  Spearman over ALL rows (gated included) = {rho_full:.3f}")

    # --- Check 3: the §4 trap and the load-bearing tail ----------------------
    # The overall non-gated Spearman is inflated by the gross free-vs-not-free
    # split.  What actually matters is (a) the metric must NOT predict the cheap
    # mass as expensive (the §4 weak-splitter trap), and (b) it must rank the few
    # genuinely expensive claims — which hold nearly all the node work and so
    # drive balance — correctly.
    ng_pred = [r for r in nongated if r[2] is not None]
    false_expensive = [r for r in ng_pred if r[2] > args.false_pred and r[6] < args.small_nodes]
    fe_frac = len(false_expensive) / len(ng_pred) if ng_pred else float("nan")
    print("\n=== §4 trap: cheap candidates predicted expensive ===")
    print(f"  non-gated with predicted > {args.false_pred} but actual < {args.small_nodes}: "
          f"{len(false_expensive):,}/{len(ng_pred):,} = {100*fe_frac:.1f}%")
    tail = [r for r in ng_pred if r[6] >= args.tail_nodes]
    tail_share = (sum(r[6] for r in tail) / sum(r[6] for r in ng_pred)
                  if ng_pred else float("nan"))
    rho_tail = spearman([r[2] for r in tail], [r[6] for r in tail]) if len(tail) > 2 else float("nan")
    print(f"\n=== load-bearing tail (actual >= {args.tail_nodes}) ===")
    print(f"  n={len(tail):,}  holds {100*tail_share:.1f}% of non-gated node work  "
          f"Spearman(pred,act)={rho_tail:.3f}")

    # --- Verdict --------------------------------------------------------------
    print("\n=== §11 go/no-go ===")
    gate_ok = (not gated) or gated_small_frac >= 0.95
    fe_ok = not math.isnan(fe_frac) and fe_frac <= args.max_false
    tail_ok = not math.isnan(rho_tail) and rho_tail >= args.strong
    if gated:
        print(f"  [{'PASS' if gate_ok else 'FAIL'}] gating split exact: "
              f">=95% of gated rows near-zero (got {100*gated_small_frac:.1f}%)")
    else:
        print("  [n/a ] gating split: no gated rows in sample")
    print(f"  [{'PASS' if fe_ok else 'FAIL'}] §4 trap avoided: "
          f"<= {100*args.max_false:.0f}% cheap-but-predicted-expensive "
          f"(got {100*fe_frac:.1f}%)")
    print(f"  [{'PASS' if tail_ok else 'FAIL'}] load-bearing rank: "
          f"tail Spearman >= {args.strong} (got {rho_tail:.3f})")
    go = gate_ok and fe_ok and tail_ok
    verdict = "GO" if go else "NO-GO (re-derive metric per §4 — cutoff-aware, before packer)"
    print(f"\n  NOTE: overall non-gated Spearman ({rho_all:.3f}) is inflated by the "
          f"free-vs-not-free split; the tail rank and §4-trap rate are decisive.")
    print(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
