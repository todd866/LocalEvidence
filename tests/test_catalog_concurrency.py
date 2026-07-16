"""The store write lock must serialize concurrent same-slug writes.

acquire()'s per-pull timeout can leave an orphan worker writing a slug's files
while a later same-slug pull (single-process multi-question runs) writes them
too. Without serialization those non-atomic file writes interleave into a torn
PDF/text. This proves the critical sections never overlap.
"""
from __future__ import annotations

import threading
import time

from localevidence.library import catalog


def test_store_pdf_serializes_concurrent_same_slug_writes(monkeypatch):
    intervals: list[tuple[float, float]] = []
    record_lock = threading.Lock()

    def slow_extract(pdf_path, text_path):
        # Runs inside catalog._write_lock (store_pdf calls extract_text there).
        t0 = time.monotonic()
        time.sleep(0.05)
        text_path.write_text("body")
        t1 = time.monotonic()
        with record_lock:
            intervals.append((t0, t1))
        return True

    monkeypatch.setattr("localevidence.library.extract.extract_text", slow_extract)

    def worker():
        catalog.store_pdf(b"%PDF-1.4 fake bytes", doi="10.1/same-slug")

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(intervals) == 3
    intervals.sort()
    for (a_start, a_end), (b_start, b_end) in zip(intervals, intervals[1:]):
        assert a_end <= b_start + 1e-4, f"critical sections overlapped: {intervals}"
