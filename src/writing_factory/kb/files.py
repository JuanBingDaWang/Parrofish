"""Content-addressed managed copies of user-supplied source documents."""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from pathlib import Path

from writing_factory.kb.models import ManagedDocument


class ManagedFileStore:
    """Copy imports into ignored application storage and deduplicate by SHA-256."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def import_file(self, source_path: Path) -> ManagedDocument:
        """Create or reuse an immutable managed copy of one local file."""

        source = source_path.expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"Not a regular file: {source}")
        digest = self._sha256(source)
        suffix = source.suffix.lower()
        destination = self.root / f"{digest}{suffix}"
        if not destination.exists():
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            try:
                shutil.copyfile(source, temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        return ManagedDocument(
            doc_id=f"doc_{digest[:24]}",
            sha256=digest,
            filename=source.name,
            format=suffix.removeprefix("/").removeprefix("."),
            source_path=source,
            managed_path=destination,
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
