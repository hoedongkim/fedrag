"""A parsed-file cache for routes that serve tracked JSON files: re-read when a file changes."""

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any


class VersionedFileCache:
    """One parsed value per file, kept until the file's mtime or size changes."""

    def __init__(self) -> None:
        self._entries: dict[Path, tuple[tuple[int, int], Any]] = {}
        self._lock = threading.Lock()

    def get(self, path: Path, parse: Callable[[Path], Any]) -> Any:
        """The parsed file, or None when it does not exist; a None from `parse` is not kept."""
        try:
            stat = path.stat()
        except OSError:
            return None
        version = (stat.st_mtime_ns, stat.st_size)
        with self._lock:
            entry = self._entries.get(path)
        if entry and entry[0] == version:
            return entry[1]
        value = parse(path)
        if value is not None:
            with self._lock:
                self._entries[path] = (version, value)
        return value
