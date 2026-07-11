"""Deterministic trap-family workloads for the swarm performance guards.

A "trap family" is a set of answer words sharing their last four letters
(batch/catch/hatch/latch/match/patch/watch): no early guess separates the
family, so the search must recurse deep and alpha-beta ceilings do real
pruning.  Branches built from whole families are the workload shape where
the adaptive-decomposition layer earns or wastes its keep — unlike small
random branches, whose wall time is dominated by per-claim coordination.
"""
from erd_queue import ERDQueue
from wordle_engine import ERD_ALL


def trap_families(answers, min_family_size=4):
    """All families of >= min_family_size answers sharing last-four letters,
    largest first (ties by first member, so the order is deterministic)."""
    by_suffix = {}
    for word in answers:
        by_suffix.setdefault(word[1:], []).append(word)
    families = [group for group in by_suffix.values()
                if len(group) >= min_family_size]
    families.sort(key=lambda group: (-len(group), group[0]))
    return families


def build_trap_branches(answers, n_branches, branch_size):
    """n_branches word-disjoint branches of branch_size trap-family words."""
    pool = [word for family in trap_families(answers) for word in family]
    need = n_branches * branch_size
    if len(pool) < need:
        raise ValueError(
            f"answer list yields only {len(pool)} trap-family words, "
            f"need {need}")
    return [pool[i * branch_size:(i + 1) * branch_size]
            for i in range(n_branches)]


def build_candidates(guesses, branches, n_base):
    """Candidate vocabulary: the first n_base guess words plus every branch
    word (a branch word must be guessable or the solve is artificially
    crippled), deterministically ordered."""
    words = set(guesses[:n_base])
    for branch in branches:
        words.update(branch)
    return sorted(words)


def seed_cost_model(queue_path, max_n_words=40, budgets=(3, 4)):
    """Warm the queue's cost model so entry-gate promotion decides from
    predictions rather than the cold PROMOTE_MIN_SIZE fallback (sized for
    production branches, far above anything a CI-scale workload produces).

    Node costs follow the measured trap-branch curve (~45 * n^2 at budget 3
    with a ~200-word candidate vocabulary), so the gate behaves as it would
    mid-production: sub-branches of ~11+ words predict above the bootstrap
    publish threshold and promote; smaller ones inline.
    """
    queue = ERDQueue(queue_path)
    try:
        for budget in budgets:
            for n_words in range(3, max_n_words + 1):
                queue.update_cost_model(ERD_ALL, n_words, 45 * n_words * n_words,
                                        budget=budget)
    finally:
        queue.close()
