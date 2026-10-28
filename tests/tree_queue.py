"""A tree page built from real queue rows, and the reference it must match.

reference_tree_layout is the tree page built in Python from every matched
queue row.  It reads the whole queue, which is why the report no longer
does, but it is the plain statement of what a page holds, so the page the
report builds from SQL is checked against it.
"""

import collections
import os
import tempfile
import time
from dataclasses import replace

import report_model
from erd_queue import ERDQueue
from report_model import (
    DEFAULT_TREE_PAGE_SIZE,
    GAME_GUESSES,
    _branch_target_payload,
    _spine_steps,
    _spineless_tree_node_id,
    _step_payload,
    _steps_from_queue_row,
    _tree_group_sort_key,
    _tree_node_id,
    branch_reference,
)


def write_tree_rows(queue, rows):
    """Write fixture rows as the queue's own branches.

    A row with a spine or an opener becomes an active branch carrying it; a
    row with neither is a branch from before spines were recorded.  A row's
    worker_count becomes that many live workers on it.
    """
    now = int(time.time())
    worker_number = 0
    for row in rows:
        spine = row.get("spine")
        budget = row.get("budget")
        if budget is None and spine:
            budget = GAME_GUESSES - len(_spine_steps(spine))
        queue.create_branch(
            row["branch_key"], row.get("answer_count", 1),
            row.get("candidate_count", 3), priority=row.get("priority", 0),
            opener=row.get("opener"), opener_pattern=row.get("opener_pattern"),
            budget=budget, spine=spine)
        for _ in range(row.get("worker_count", 0)):
            worker_number += 1
            queue.heartbeat(f"worker-{worker_number}", worker_number,
                            row["branch_key"], row.get("answer_count", 1),
                            now, 0)


def tree_layout(rows, request, answer_set=frozenset()):
    """The report's tree page over a queue holding exactly `rows`."""
    with tempfile.TemporaryDirectory() as directory:
        queue = ERDQueue(os.path.join(directory, "queue.sqlite3"))
        try:
            write_tree_rows(queue, rows)
            filters, prefix = report_model._tree_filters(queue, request)
            return report_model._tree_layout(
                queue, filters, request, prefix, set(answer_set))
        finally:
            queue.close()


def reference_layout_of_queue(queue, request, answer_set):
    """reference_tree_layout over every row the request matches."""
    filters, prefix = report_model._tree_filters(queue, request)
    rows = queue.report_queue_rows(filters)["rows"]
    return reference_tree_layout(rows, request, prefix, set(answer_set))


def reference_tree_layout(rows, request, prefix, answer_set):
    if not rows:
        return {
            "root": _branch_target_payload(request.branch_target),
            "topology_source": "queue",
            "tree_available": False,
            "unavailable_reason": "no extant queue topology",
            "nodes": [],
        }

    # A node is a played guess.  The response includes one page of immediate
    # children; their descendants are fetched only when a node is expanded.
    nodes = {}
    for row in rows:
        steps = _steps_from_queue_row(row)
        if steps:
            parent_id = None
            for guess_depth_value in range(1, len(steps) + 1):
                step = steps[guess_depth_value - 1]
                node_id = _tree_node_id(steps[:guess_depth_value])
                nodes.setdefault(node_id, {
                    "node_id": node_id,
                    "parent_node_id": parent_id,
                    "step": _step_payload(step, answer_set),
                    "branch_key_hex": None,
                    "branch_reference": None,
                    "branch_status": None,
                    "branch_worker_status": None,
                    "answer_count": None,
                    "guess_depth": guess_depth_value,
                    "worker_count": 0,
                    "subtree_worker_count": 0,
                    "child_count": 0,
                    "priority": None,
                    "completed_candidate_count": None,
                    "candidate_count": None,
                    "is_context": False,
                    "spine": " ".join(
                        value for spine_step in steps[:guess_depth_value]
                        for value in (spine_step.word.upper(), spine_step.pattern)
                    ),
                })
                parent_id = node_id
            final_node = nodes[parent_id]
        else:
            guess_depth = (
                GAME_GUESSES - row["budget"] if row.get("budget") is not None else 1
            )
            parent_id = None
            for guess_depth_value in range(1, max(guess_depth, 1) + 1):
                node_id = _spineless_tree_node_id(
                    guess_depth_value, row["branch_key_hex"]
                )
                nodes.setdefault(node_id, {
                    "node_id": node_id,
                    "parent_node_id": parent_id,
                    "step": None,
                    "branch_key_hex": None,
                    "branch_reference": None,
                    "branch_status": None,
                    "branch_worker_status": None,
                    "answer_count": None,
                    "guess_depth": guess_depth_value,
                    "worker_count": 0,
                    "subtree_worker_count": 0,
                    "child_count": 0,
                    "priority": None,
                    "completed_candidate_count": None,
                    "candidate_count": None,
                    "is_context": False,
                    "spine": None,
                })
                parent_id = node_id
            final_node = nodes[parent_id]
        final_node.update({
            "branch_key_hex": row["branch_key_hex"],
            "branch_reference": branch_reference(bytes(row["branch_key"])),
            "branch_status": row["branch_status"],
            "branch_worker_status": row["branch_worker_status"],
            "answer_count": row["answer_count"],
            "worker_count": row["worker_count"],
            "priority": row["priority"],
            "completed_candidate_count": row["completed_candidate_count"],
            "candidate_count": row["candidate_count"],
            "is_context": bool(row.get("is_context")),
        })
    # Counted once per parent: a count taken per node by scanning every node
    # is quadratic, and the whole queue's tree has hundreds of thousands.
    child_counts = collections.Counter(
        candidate["parent_node_id"] for candidate in nodes.values()
        if candidate["parent_node_id"] is not None)
    for node in nodes.values():
        node["child_count"] = child_counts[node["node_id"]]
        node["has_children"] = node["child_count"] > 0
    for node in nodes.values():
        worker_count = node["worker_count"]
        current_node = node
        while current_node is not None:
            current_node["subtree_worker_count"] += worker_count
            parent_node_id = current_node["parent_node_id"]
            current_node = nodes.get(parent_node_id)
    parent_spine = request.tree_parent or prefix
    parent_steps = _spine_steps(parent_spine) if parent_spine else ()
    parent_id = _tree_node_id(parent_steps) if parent_steps else None
    direct_nodes = [
        node for node in nodes.values() if node["parent_node_id"] == parent_id
    ]
    grouped_nodes = {}
    for node in sorted(
        direct_nodes,
        key=lambda node: (
            (node["step"] or {}).get("word", ""),
            (node["step"] or {}).get("pattern", ""),
            node["branch_key_hex"] or "",
        ),
    ):
        group_key = (node["step"] or {}).get("word") or node["node_id"]
        grouped_nodes.setdefault(group_key, []).append(node)
    group_keys = list(grouped_nodes)
    if request.tree_cursor is not None:
        cursor_sort_key = _tree_group_sort_key(request.tree_cursor)
        group_keys = [
            key for key in group_keys
            if _tree_group_sort_key(key) > cursor_sort_key
        ]
    page_size = request.filters.limit or DEFAULT_TREE_PAGE_SIZE
    page_keys = group_keys[:page_size]
    page_nodes = [node for key in page_keys for node in grouped_nodes[key]]
    next_cursor = page_keys[-1] if len(group_keys) > page_size else None
    return {
        "root": _branch_target_payload(request.branch_target),
        "topology_source": "queue",
        "tree_available": True,
        "unavailable_reason": None,
        "nodes": page_nodes,
        "paging": {
            "parent_spine": parent_spine,
            "cursor": request.tree_cursor,
            "page_size": page_size,
            "returned_group_count": len(page_keys),
            "total_group_count": len(grouped_nodes),
            "next_cursor": next_cursor,
        },
    }
