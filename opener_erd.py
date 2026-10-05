"""An opener's own ERD, folded from its top-level response groups.

An opener's exact ERD is what the swarm is for: the mean line length under
optimal play from that first guess.  It is folded once, by the worker that
finishes the opener's last branch, and read back by the reporting layer -- so
it cannot live in `report_model`, which sits above the swarm and is imported by
nothing below it.

The same fold answers a question about one named opener, where it is a single
partition of the answer list and one indexed read per group.  There the answer
is as fresh as the request rather than as fresh as the last write, which is
worth the few milliseconds; a whole ranking is not, which is why the ranking is
read from `opener_erd_by_policy` instead of refolded.
"""

from cache_sqlite import ScoreCache
from wordle_ui import fmt_pattern

ALL_GREEN_PATTERN_CODE = 3 ** 5 - 1
ALL_GREEN_PATTERN_TEXT = fmt_pattern(ALL_GREEN_PATTERN_CODE)


def fold_response_groups(response_groups, group_budget):
    """Fold a candidate's response groups into its own ERD and worst-case line.

    The single place any caller gets a candidate's own ERD.  `response_groups`
    carry each group's branch fact as `ScoreCache.report_branch_states`
    resolved it at `group_budget`, so a child whose only exact result was
    solved at some other budget arrives here as `missing` and leaves the
    candidate `pending` — the same scope rule the solver reuses a child under.

    Playing the candidate spends one guess from this branch's budget; each
    response group is then solved independently.  So the candidate's ERD is the
    answer-weighted mean of the groups' ERDs plus one, and its worst-case line
    is the deepest group line plus one.  A single remaining answer is solved by
    playing it (one more guess) unless the candidate itself was the answer
    (all-green response, zero more guesses) — but that one guess needs a guess
    left, so with `group_budget < 1` a lone survivor is a proven loss, matching
    `wordle_engine.evaluate_candidate`, which checks the budget floor before its
    n == 1 shortcut.

    The fold reports one of three states.  It is `complete` — an exact ERD and
    worst-case line — only once every group is solved.  A group proven
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
            # An ERD with no proven worst-case line cannot complete the fold.
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




def response_groups_from_patterns(words_by_pattern):
    """Group tuples from a {pattern code: answer words} partition.

    Empty patterns are dropped: a response no answer produces is not a group.
    Sorted by pattern so two callers partitioning the same opener by different
    machinery -- a pattern matrix, a response cache -- hand the fold the same
    list.
    """
    return [
        (fmt_pattern(pattern_code), len(words),
         ScoreCache.encode_subset(words))
        for pattern_code, words in sorted(words_by_pattern.items())
        if words
    ]


def opener_response_groups(pattern_matrix, opener, all_answers):
    """One opener's top-level response groups, by the pattern matrix.

    Every opener partitions the whole answer list, so these are a property of
    the vocabulary alone and say nothing about the cache.
    """
    answer_list = list(all_answers)
    return response_groups_from_patterns(pattern_matrix.group_words(
        opener, answer_list, pattern_matrix.answer_indices(answer_list)))


def fold_opener(cache, response_groups, policy, group_budget):
    """This opener's own ERD as the cache stands right now.

    `response_groups` is the opener's split of the answer list, which the
    caller already holds or builds with `opener_response_groups` -- partitions
    come from a pattern matrix in some callers and a response cache in others,
    and neither is this function's business.

    One indexed read per group through the cache's own reusability gate, so a
    child whose only exact result was solved at some other budget arrives as
    `missing` and leaves the opener pending.
    """
    states = cache.report_branch_states(
        [key for _pattern, _count, key in response_groups],
        policy, group_budget)
    return fold_response_groups(
        [
            {
                "pattern": pattern,
                "answer_count": answer_count,
                "best_erd": states[key]["best_erd"],
                "max_remaining_depth": states[key]["max_remaining_depth"],
                "cache_state": states[key]["cache_state"],
            }
            for pattern, answer_count, key in response_groups
        ],
        group_budget,
    )


def store_opener_verdict(cache, opener, response_groups, policy, group_budget):
    """Fold a finished opener over its own groups and store what it came to.

    The one place an opener's verdict is written, so the row always means the
    same thing however it was reached -- a worker finishing the last branch, or
    a backfill over openers that finished before anything stored them.

    A pending fold stores nothing and is returned for the caller to complain
    about.  Reaching here says the opener's tree is finished, so a group with
    no reusable exact result means one is missing from this cache; a stored
    verdict would assert a tree that is not there, and no row reports the
    opener as unfinished, which is what it is.
    """
    summary = fold_opener(cache, response_groups, policy, group_budget)
    if summary["state"] != "pending":
        cache.write_opener_erd(
            opener, policy, summary["state"], summary["erd"],
            summary["max_remaining_depth"], summary["response_group_count"])
    return summary
