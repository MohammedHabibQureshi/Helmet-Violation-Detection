"""Duplicate violation event suppressor.

Prevents the same number plate from being reported multiple times within a
sliding time window using both exact matching and fuzzy edit-distance matching.
"""

import sys
from dataclasses import dataclass, field
from typing import Optional

try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    # Pure-Python fallback
    def levenshtein_distance(s1: str, s2: str) -> int:  # type: ignore[misc]
        """Compute Levenshtein edit distance between two strings."""
        if s1 == s2:
            return 0
        len1, len2 = len(s1), len(s2)
        if len1 == 0:
            return len2
        if len2 == 0:
            return len1
        prev = list(range(len2 + 1))
        for i, c1 in enumerate(s1, 1):
            curr = [i] + [0] * len2
            for j, c2 in enumerate(s2, 1):
                curr[j] = min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + (0 if c1 == c2 else 1),
                )
            prev = curr
        return prev[len2]


@dataclass
class SuppressorStats:
    """Counters for the duplicate suppressor."""

    total_seen: int = 0
    duplicates_suppressed: int = 0
    unique_events: int = 0
    exact_matches: int = 0
    fuzzy_matches: int = 0


class DuplicateSuppressor:
    """Suppresses duplicate violation events using a sliding time window.

    Two events are considered duplicates if:
    - They fall within ``window_seconds`` of each other, AND
    - Their plate texts match exactly (after normalization), OR
    - Their plate texts have a Levenshtein distance <= ``fuzzy_threshold``

    Parameters
    ----------
    window_seconds:
        Duration of the sliding window in seconds.
    fuzzy_threshold:
        Maximum edit distance for two plates to be considered the same.
    """

    def __init__(
        self,
        window_seconds: float = 5.0,
        fuzzy_threshold: int = 1,
    ):
        self.window_seconds = window_seconds
        self.fuzzy_threshold = fuzzy_threshold
        self._events: list[dict] = []  # {plate_text, timestamp_s}
        self._stats = SuppressorStats()

    def _normalize(self, text: str) -> str:
        return text.strip().upper()

    def _prune_window(self, current_time: float) -> None:
        """Remove events outside the current time window."""
        cutoff = current_time - self.window_seconds
        self._events = [
            e for e in self._events if e["timestamp_s"] >= cutoff
        ]

    def is_duplicate(self, plate_text: str, timestamp_s: float) -> bool:
        """Check if this event is a duplicate of a recent event.

        Parameters
        ----------
        plate_text:
            The recognized (normalized) plate text.
        timestamp_s:
            Timestamp of this event in seconds.

        Returns
        -------
        bool
            True if this event should be suppressed as a duplicate.
        """
        self._stats.total_seen += 1
        norm = self._normalize(plate_text)
        if not norm:
            # Empty plates are never flagged as duplicates (can't compare)
            self._stats.unique_events += 1
            self._events.append({
                "plate_text": norm,
                "timestamp_s": timestamp_s,
            })
            return False

        self._prune_window(timestamp_s)

        for event in self._events:
            past_text = event["plate_text"]
            # Exact match
            if past_text == norm:
                self._stats.duplicates_suppressed += 1
                self._stats.exact_matches += 1
                return True
            # Fuzzy match
            if levenshtein_distance(past_text, norm) <= self.fuzzy_threshold:
                self._stats.duplicates_suppressed += 1
                self._stats.fuzzy_matches += 1
                return True

        # Not a duplicate: record the event
        self._events.append({
            "plate_text": norm,
            "timestamp_s": timestamp_s,
        })
        self._stats.unique_events += 1
        return False

    def add_event(self, plate_text: str, timestamp_s: float) -> None:
        """Unconditionally record an event (bypasses duplicate check).

        Used to seed the suppressor from external state.
        """
        norm = self._normalize(plate_text)
        self._events.append({"plate_text": norm, "timestamp_s": timestamp_s})
        self._prune_window(timestamp_s)

    def get_stats(self) -> dict:
        """Return current counters as a plain dict."""
        return {
            "total_seen": self._stats.total_seen,
            "duplicates_suppressed": self._stats.duplicates_suppressed,
            "unique_events": self._stats.unique_events,
            "exact_matches": self._stats.exact_matches,
            "fuzzy_matches": self._stats.fuzzy_matches,
            "window_seconds": self.window_seconds,
            "fuzzy_threshold": self.fuzzy_threshold,
            "current_window_size": len(self._events),
        }

    def reset(self) -> None:
        """Clear all state and reset counters."""
        self._events.clear()
        self._stats = SuppressorStats()


def main() -> int:
    """Smoke-test the suppressor."""
    sup = DuplicateSuppressor(window_seconds=5.0, fuzzy_threshold=1)

    tests = [
        ("MH12AB1234", 0.0, False, "first event"),
        ("MH12AB1234", 2.0, True, "exact duplicate within window"),
        ("MH12AB1234", 10.0, False, "outside window"),
        ("MH12AB1235", 11.0, True, "fuzzy match (edit distance 1)"),
        ("DL01AA0001", 12.0, False, "completely different plate"),
    ]

    all_pass = True
    for plate, ts, expected, label in tests:
        result = sup.is_duplicate(plate, ts)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"[{status}] plate={plate!r:14s} ts={ts:5.1f}s "
              f"expected={expected!r} got={result!r} -- {label}")

    print("\nStats:", sup.get_stats())
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
