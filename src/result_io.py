from __future__ import annotations

import json
import os
import pickle
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def atomic_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_pickle(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_success(path: Path) -> bool:
    try:
        return read_json(path).get("status") == "success"
    except (OSError, ValueError, AttributeError):
        return False


def next_attempt_dir(parent: Path) -> Path:
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    attempts = []
    for child in parent.iterdir():
        match = re.fullmatch(r"attempt(\d+)", child.name)
        if child.is_dir() and match:
            attempts.append(int(match.group(1)))
    result = parent / f"attempt{max(attempts, default=0) + 1}"
    result.mkdir(parents=True, exist_ok=False)
    return result


def alpha_key(alpha: float) -> str:
    return format(float(alpha), ".12g").replace("-", "m").replace(".", "p")


@contextmanager
def exclusive_file_lock(path: Path, timeout_seconds: float = 120.0, stale_seconds: float = 3600.0):
    """Portable lock-file guard for concurrent paper-summary refreshes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    descriptor = None
    while descriptor is None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime > stale_seconds:
                    path.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for result lock: {path}")
            time.sleep(0.2)
    try:
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
