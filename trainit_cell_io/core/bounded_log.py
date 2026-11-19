"""A bounded, append-only log with a running total (P0.b, implementation review fix C7: Python review MINOR 4).

The cell's log and the mock plant's trace grew for the whole run. They keep the most recent `maxlen` entries now; a
reader that consumes them incrementally (`cell_io_node`, the Kit-side loop) remembers the running total it has read
and calls `since()`: it still gets every new entry exactly once, oldest first, as long as it reads before `maxlen`
newer entries pushed one out -- and when it did not, `since()` says how many it lost instead of hiding it.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Generic, Iterator, List, Tuple, TypeVar

T = TypeVar('T')

DEFAULT_MAXLEN = 10000


class BoundedLog(Generic[T]):
    """The last `maxlen` entries, and `total`: how many were ever appended (a reader's running index)."""

    def __init__(self, maxlen: int = DEFAULT_MAXLEN):
        if isinstance(maxlen, bool) or not isinstance(maxlen, int) or maxlen < 1:
            raise ValueError(f'maxlen must be an integer >= 1, got {maxlen!r}')
        self._entries: Deque[T] = deque(maxlen=maxlen)
        self._total = 0

    @property
    def maxlen(self) -> int:
        return int(self._entries.maxlen or 0)

    @property
    def total(self) -> int:
        """Every entry ever appended: the running index a reader keeps (never decreases)."""
        return self._total

    def append(self, entry: T) -> None:
        self._entries.append(entry)
        self._total += 1

    def since(self, index: int) -> Tuple[List[T], int, int]:
        """The entries appended at or after the running index `index` that are still held (oldest first), the new
        running index (`total`), and how many entries after `index` were already dropped (0 for a reader that keeps
        up)."""
        unread = self._total - max(0, index)
        if unread <= 0:
            return [], self._total, 0
        held = len(self._entries)
        dropped = max(0, unread - held)
        fresh = unread - dropped
        return [self._entries[i] for i in range(held - fresh, held)], self._total, dropped

    def __iter__(self) -> Iterator[T]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f'BoundedLog(held={len(self._entries)}, total={self._total}, maxlen={self.maxlen})'


__all__ = ['BoundedLog', 'DEFAULT_MAXLEN']
