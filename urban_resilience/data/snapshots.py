"""Data snapshot auditing, checksum verification, and immutability management.

Provides tracking for frozen data snapshots ensuring that pre-event observations,
timestamps, cutoffs, and SHA256 checksums are immutably logged for pipeline reproducibility.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SnapshotMetadata(BaseModel):
    """Metadata specification for a frozen historical or baseline data snapshot."""

    snapshot_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    cutoff_timestamp: str = Field(..., description="Temporal cutoff (ISO 8601); observations strictly before this timestamp")
    data_sources: List[str] = Field(default_factory=list)
    sha256_checksum: str
    record_count: int = 0
    spatial_coverage_zones: List[str] = Field(default_factory=list)
    is_verified: bool = False
    provenance_notes: str = ""


class DataSnapshotRegistry:
    """Registry maintaining registered data snapshots and verifying checksum integrity."""

    def __init__(self):
        self._snapshots: Dict[str, SnapshotMetadata] = {}

    def register_snapshot(self, snapshot: SnapshotMetadata) -> None:
        """Registers a snapshot in the in-memory registry."""
        self._snapshots[snapshot.snapshot_id] = snapshot
        logger.info("Registered DataSnapshot '%s' (cutoff: %s, verified: %s)", snapshot.snapshot_id, snapshot.cutoff_timestamp, snapshot.is_verified)

    def get_snapshot(self, snapshot_id: str) -> Optional[SnapshotMetadata]:
        """Retrieves snapshot metadata by ID."""
        return self._snapshots.get(snapshot_id)

    @staticmethod
    def compute_file_checksum(filepath: Union[str, Path]) -> str:
        """Computes SHA256 checksum of a file on disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        sha = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        return sha.hexdigest()

    def verify_snapshot_integrity(self, snapshot_id: str, filepath: Union[str, Path]) -> bool:
        """Verifies that an on-disk dataset file matches the registered snapshot checksum."""
        snap = self.get_snapshot(snapshot_id)
        if not snap:
            raise KeyError(f"Snapshot '{snapshot_id}' is not registered.")

        actual_checksum = self.compute_file_checksum(filepath)
        is_match = (actual_checksum.lower() == snap.sha256_checksum.lower())
        if not is_match:
            logger.warning(
                "Snapshot '%s' checksum mismatch! Expected: %s, Actual: %s",
                snapshot_id,
                snap.sha256_checksum,
                actual_checksum,
            )
        return is_match
