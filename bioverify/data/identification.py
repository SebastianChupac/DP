"""
Dataset loader for closed-set identification protocol CSV files.
"""

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class IdentificationSample:
    """Single sample entry from identification CSV."""

    record_id: str
    sample_id: str
    image_path: str
    identity: str
    modality: str
    dataset_name: str
    split: str
    metadata: Dict[str, Any]


class IdentificationDataset:
    """Loader for identification CSV manifests with gallery/probe split."""

    def __init__(
        self,
        csv_path: str,
        base_path: Optional[str] = None,
        filter_modality: Optional[str] = None,
        filter_dataset: Optional[str] = None,
    ):
        self.csv_path = csv_path
        self.base_path = base_path or os.path.dirname(csv_path)
        self.filter_modality = filter_modality
        self.filter_dataset = filter_dataset

        self.gallery_by_identity: Dict[str, List[IdentificationSample]] = {}
        self.probes: List[IdentificationSample] = []

        self._load()

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return str(Path(self.base_path) / path)

    def _load(self):
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            modality = row.get("modality", "")
            dataset_name = row.get("dataset_name", "")
            if self.filter_modality and modality != self.filter_modality:
                continue
            if self.filter_dataset and dataset_name != self.filter_dataset:
                continue

            metadata: Dict[str, Any] = {}
            raw_metadata = row.get("metadata", "")
            if raw_metadata:
                try:
                    metadata = json.loads(raw_metadata)
                except json.JSONDecodeError:
                    metadata = {}

            sample = IdentificationSample(
                record_id=row["record_id"],
                sample_id=row["sample_id"],
                image_path=self._resolve_path(row["image_path"]),
                identity=row["identity"],
                modality=modality,
                dataset_name=dataset_name,
                split=row["split"].strip().lower(),
                metadata=metadata,
            )

            if sample.split == "gallery":
                self.gallery_by_identity.setdefault(sample.identity, []).append(sample)
            elif sample.split == "probe":
                self.probes.append(sample)

    def validate_closed_set(self) -> bool:
        gallery_ids = set(self.gallery_by_identity.keys())
        probe_ids = set(sample.identity for sample in self.probes)
        return len(probe_ids - gallery_ids) == 0

    def get_gallery(self) -> Dict[str, List[IdentificationSample]]:
        return self.gallery_by_identity

    def get_probes(self) -> List[IdentificationSample]:
        return self.probes

    def get_statistics(self) -> Dict[str, Any]:
        gallery_count = sum(len(v) for v in self.gallery_by_identity.values())
        probe_count = len(self.probes)
        gallery_ids = set(self.gallery_by_identity.keys())
        probe_ids = set(sample.identity for sample in self.probes)

        return {
            "gallery_samples": gallery_count,
            "probe_samples": probe_count,
            "gallery_identities": len(gallery_ids),
            "probe_identities": len(probe_ids),
            "closed_set_valid": self.validate_closed_set(),
        }
