"""Reducing a node's response-group results to that node's own ERD.

An **ERD reduction** is one level of the ERD recurrence evaluated over results
that are already computed, rather than by searching: given each of a
candidate's response groups and the exact result stored for it, the candidate's
own ERD is the answer-weighted mean of those groups' ERDs plus the one guess
that playing the candidate spends, and its worst-case line is the deepest group
line plus one.

The reduction lives here rather than in a reporting module because both sides
of the system need it.  The swarm reduces an opener's ERD once the last branch
of its tree resolves, which is what makes the opener done.  A report reduces a
*candidate's* ERD at an arbitrary branch on every read, because that value is
not stored anywhere: it is keyed by (branch, candidate), an unbounded set whose
dependencies cannot be enumerated, so a branch result deleted by a repair or a
requeue would falsify stored reductions that nothing can name.

It depends on nothing but the response-pattern spelling, so a caller holding
group facts can reduce them without a cache, a queue, or a pattern matrix.
"""

from wordle_ui import fmt_pattern

ALL_GREEN_PATTERN_TEXT = fmt_pattern(3 ** 5 - 1)


def response_group_is_solved(group, group_budget):
    """Whether a response group needs no further search.

    The same test `reduce_candidate_erd` applies when it counts a group as
    resolved, so a report's per-group answer and its "N of M response groups
    solved" line always agree.  A cached ERD counts only alongside a proven
    worst-case line; a group of fewer than two answers is solved by playing the
    survivor, and that guess needs a budget to spend unless the guess already
    was the answer.
    """
    if group["best_erd"] is None:
        if group["answer_count"] < 2:
            return (group["pattern"] == ALL_GREEN_PATTERN_TEXT
                    or group_budget >= 1)
        return False
    return group["max_remaining_depth"] is not None


def reduce_candidate_erd(response_groups, group_budget):
    """Reduce a candidate's response groups to its own ERD and worst-case line.

    `response_groups` carry each group's branch fact as
    `ScoreCache.report_branch_states` resolved it at `group_budget`, so a child
    whose only exact result was solved at some other budget arrives here as
    `missing` and leaves the candidate `pending` — the same scope rule the
    solver reuses a child under.

    Playing the candidate spends one guess from this branch's budget; each
    response group is then solved independently.  So the candidate's ERD is the
    answer-weighted mean of the groups' ERDs plus one, and its worst-case line
    is the deepest group line plus one.  A single remaining answer is solved by
    playing it (one more guess) unless the candidate itself was the answer
    (all-green response, zero more guesses) — but that one guess needs a guess
    left, so with `group_budget < 1` a lone survivor is a proven loss, matching
    `wordle_engine.evaluate_candidate`, which checks the budget floor before its
    n == 1 shortcut.

    The reduction reports one of three states.  It is `complete` — an exact ERD
    and worst-case line — only once every group is solved.  A group proven
    unsolvable within budget (a loss, or a lone survivor with no guess left)
    makes the candidate `infeasible`: its ERD is unbounded and no further search
    changes that.  A group still being searched leaves the candidate `pending`.
    """
    total_answers = sum(group["answer_count"] for group in response_groups)
    weighted_remaining_depth = 0.0
    max_group_remaining_depth = 0
    resolved_group_count = 0
    infeasible_group_count = 0
    pending_group_count = 0
    for group in response_groups:
        best_erd = group["best_erd"]
        max_remaining_depth = group["max_remaining_depth"]
        if best_erd is None:
            if group["answer_count"] < 2:
                solved_by_candidate = group["pattern"] == ALL_GREEN_PATTERN_TEXT
                if not solved_by_candidate and group_budget < 1:
                    infeasible_group_count += 1
                    continue
                best_erd = 0.0 if solved_by_candidate else 1.0
                max_remaining_depth = 0 if solved_by_candidate else 1
            elif group["cache_state"] == "loss":
                infeasible_group_count += 1
                continue
            else:
                pending_group_count += 1
                continue
        elif max_remaining_depth is None:
            # An ERD with no proven worst-case line cannot complete the
            # reduction.
            pending_group_count += 1
            continue
        resolved_group_count += 1
        weighted_remaining_depth += group["answer_count"] * best_erd
        max_group_remaining_depth = max(max_group_remaining_depth, max_remaining_depth)
    if infeasible_group_count:
        state = "infeasible"
    elif pending_group_count or total_answers == 0:
        state = "pending"
    else:
        state = "complete"
    return {
        "state": state,
        "erd": (
            1.0 + weighted_remaining_depth / total_answers
            if state == "complete" else None
        ),
        "max_remaining_depth": (
            1 + max_group_remaining_depth if state == "complete" else None
        ),
        "resolved_group_count": resolved_group_count,
        "infeasible_group_count": infeasible_group_count,
        "response_group_count": len(response_groups),
    }


def reduce_opener(opener, all_answers, response_cache, score_cache, policy,
                  group_budget):
    """Reduce one opener's response groups, read from cached branch results.

    `response_cache` partitions the answer list by the opener's response
    patterns and `score_cache` supplies each group's result at `group_budget`;
    both are used through the methods named here, so this module imports
    neither.  An opener spends the first guess, so the caller's budget is the
    root budget less one.  The partition is the same one the word report
    uses, which is what keeps a stored reduction and a report of the same
    opener from disagreeing.
    """
    groups = response_cache.group_words(opener, all_answers)
    rows = []
    for pattern_code, answer_words in sorted(groups.items()):
        if answer_words:
            rows.append((fmt_pattern(pattern_code), len(answer_words),
                         score_cache.encode_subset(answer_words)))
    states = score_cache.report_branch_states(
        [branch_key for _pattern, _count, branch_key in rows], policy,
        group_budget)
    return reduce_candidate_erd(
        [{
            "pattern": pattern,
            "answer_count": answer_count,
            "best_erd": states[bytes(branch_key)]["best_erd"],
            "max_remaining_depth": states[bytes(branch_key)]["max_remaining_depth"],
            "cache_state": states[bytes(branch_key)]["cache_state"],
        } for pattern, answer_count, branch_key in rows],
        group_budget)
