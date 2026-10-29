from __future__ import annotations

import os
from pathlib import Path
import sys
import time

from koen_syncnet_eval.ledger import LedgerError, write_atomic


def write_progress(path: Path, payload: str) -> None:
    for attempt in range(1, 6):
        try:
            write_atomic(path, payload)
            return
        except LedgerError as error:
            if not isinstance(error.__cause__, PermissionError):
                raise
            time.sleep(0.05 * attempt)
    fallback = path.with_suffix(".jsonl")
    with fallback.open("ab") as stream:
        stream.write(payload.encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        f"warning: progress replace remained unavailable; appended {fallback}",
        file=sys.stderr,
        flush=True,
    )


__all__ = ["write_progress"]
