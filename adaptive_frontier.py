"""adaptive_frontier.py - Adaptive work scheduler internals for lookahead search.

This module provides a small priority frontier that supports lazy invalidation
through generation stamps and viability checks against a moving global cutoff.

Design notes:
- Heap entries are immutable once pushed.
- If an item's source becomes dirty, it is recomputed and requeued instead of
  mutating the existing heap entry.
- Pop performs revalidation and skips stale/pruned entries.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import heapq
from itertools import count
from typing import Callable, Dict, List, Optional, Tuple


class WorkItemType(Enum):
    """Supported work item categories."""

    ACTIVATE_TOP_WORD = auto()
    REFINE_CHILD = auto()


@dataclass(frozen=True)
class WorkItem:
    """A single queued work item with lazy revalidation metadata."""

    item_type: WorkItemType
    item_key: Tuple
    priority: float
    generation: int
    upper_bound: float


@dataclass(order=True)
class _HeapEntry:
    """Internal heap wrapper for deterministic max-priority behavior."""

    # Min-heap over negative priority => max-priority first.
    sort_priority: float
    seq: int
    item: WorkItem = field(compare=False)


class AdaptiveFrontier:
    """Priority work frontier with generation and viability revalidation.

    Args:
        get_generation: Returns the current generation stamp for an item key.
        get_cutoff: Returns current global viability cutoff (C_N).
        is_dirty: Returns True when queued item metadata should be recomputed.
        recompute_item: Recompute callback used when item is dirty; should
            return (priority, generation, upper_bound) or None to drop.
    """

    def __init__(
        self,
        get_generation: Callable[[Tuple], int],
        get_cutoff: Callable[[], float],
        is_dirty: Optional[Callable[[WorkItem], bool]] = None,
        recompute_item: Optional[
            Callable[[WorkItem], Optional[Tuple[float, int, float]]]
        ] = None,
    ):
        self._get_generation = get_generation
        self._get_cutoff = get_cutoff
        self._is_dirty = is_dirty
        self._recompute_item = recompute_item

        self._heap: List[_HeapEntry] = []
        self._seq = count()

        # Tuning / reporting counters.
        self.skipped_stale = 0
        self.skipped_pruned = 0
        self.skipped_dirty_requeued = 0
        self.popped_valid = 0

    def __len__(self) -> int:
        return len(self._heap)

    def enqueue(
        self,
        item_type: WorkItemType,
        item_key: Tuple,
        priority: float,
        generation: int,
        upper_bound: float,
    ) -> None:
        """Push a new immutable work item onto the frontier."""
        item = WorkItem(
            item_type=item_type,
            item_key=item_key,
            priority=priority,
            generation=generation,
            upper_bound=upper_bound,
        )
        entry = _HeapEntry(-priority, next(self._seq), item)
        heapq.heappush(self._heap, entry)

    def pop(self) -> Optional[WorkItem]:
        """Pop next valid item; skip stale/pruned; requeue dirty items."""
        while self._heap:
            entry = heapq.heappop(self._heap)
            item = entry.item

            current_gen = self._get_generation(item.item_key)
            if item.generation != current_gen:
                self.skipped_stale += 1
                continue

            cutoff = self._get_cutoff()
            if item.upper_bound < cutoff:
                self.skipped_pruned += 1
                continue

            if self._is_dirty and self._is_dirty(item):
                if not self._recompute_item:
                    self.skipped_stale += 1
                    continue

                refreshed = self._recompute_item(item)
                self.skipped_dirty_requeued += 1
                if refreshed is None:
                    continue
                priority, generation, upper_bound = refreshed
                self.enqueue(
                    item.item_type,
                    item.item_key,
                    priority,
                    generation,
                    upper_bound,
                )
                continue

            self.popped_valid += 1
            return item

        return None

    def counters(self) -> Dict[str, int]:
        """Return skip/pop counters for tuning and reporting."""
        return {
            "skipped_stale": self.skipped_stale,
            "skipped_pruned": self.skipped_pruned,
            "skipped_dirty_requeued": self.skipped_dirty_requeued,
            "popped_valid": self.popped_valid,
        }
