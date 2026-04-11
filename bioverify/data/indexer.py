"""
Dataset indexer for creating CSV manifests from biometric datasets.

Scans dataset directories, parses structures, generates pairs, and creates
CSV files for use in experiments.
"""
import os
import csv
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Any

from .parsers import get_parser, ImageRecord
from .pairs import PairGenerator, ImagePair


class DatasetIndexer:
    """Main class for indexing datasets and generating pair CSV manifests."""
    
    def __init__(self, public_dataset_root: str, random_seed: int = 42):
        """Initialize dataset indexer.
        
        Args:
            public_dataset_root: Path to PublicDataset directory
            random_seed: Random seed for reproducible pair generation
        """
        self.public_dataset_root = Path(public_dataset_root)
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)
        self.pair_generator = PairGenerator(random_seed=random_seed)
        self.indexed_records: Dict[str, List[ImageRecord]] = {}

    @staticmethod
    def _to_sample_id(record: ImageRecord) -> str:
        """Build a stable sample identifier for identification protocols."""
        if record.sample_id:
            raw_sample_id = str(record.sample_id)
        else:
            raw_sample_id = Path(record.image_path).stem

        parts = [record.dataset_name, record.identity, raw_sample_id]
        if record.side:
            parts.append(record.side)
        if record.finger:
            parts.append(record.finger)
        if record.session:
            parts.append(record.session)

        return "::".join(parts)

    @staticmethod
    def _identification_metadata(record: ImageRecord) -> Dict[str, Any]:
        """Build metadata payload for identification CSV rows."""
        metadata: Dict[str, Any] = {
            "side": record.side,
            "finger": record.finger,
            "session": record.session,
            "modality_type": record.modality_type,
            "image_type": record.image_type,
            "angle": record.angle,
            "lighting": record.lighting,
            "expression": record.expression,
            "aspect_of_hand": record.aspect_of_hand,
        }

        # Keep parser-extracted metadata in a dedicated nested field.
        if record.metadata:
            metadata["dataset_metadata"] = record.metadata

        return {k: v for k, v in metadata.items() if v is not None}

    @staticmethod
    def _record_attr_value(record: ImageRecord, key: str) -> Any:
        """Resolve a filter key to a record value.

        Supports:
        - direct ImageRecord fields (e.g., side, finger, angle, lighting)
        - metadata.<key> for parser metadata entries
        """
        if key.startswith("metadata."):
            metadata_key = key.split(".", 1)[1]
            if record.metadata:
                return record.metadata.get(metadata_key)
            return None
        return getattr(record, key, None)

    @staticmethod
    def _record_matches_identification_filters(record: ImageRecord, filters: Dict[str, Any]) -> bool:
        """Check whether a record satisfies all configured identification filters."""
        for key, expected in filters.items():
            value = DatasetIndexer._record_attr_value(record, key)

            if expected is None:
                if value is not None:
                    return False
                continue

            if isinstance(expected, list):
                if value not in expected:
                    return False
                continue

            if value != expected:
                return False

        return True

    def generate_identification_rows_from_records(
        self,
        records: List[ImageRecord],
        gallery_samples_per_identity: int,
        probes_per_identity: int,
        number_of_identities: int = -1,
        require_session_disjoint: bool = False,
        identification_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """Generate closed-set identification rows.

        Args:
            records: Parsed dataset records.
            gallery_samples_per_identity: Number of gallery samples per identity.
            probes_per_identity: Number of probe samples per identity. If set to -1,
                all remaining samples are used as probes.
            number_of_identities: Number of identities to sample from the dataset.
                If set to -1, use all identities.
            require_session_disjoint: If True, gallery and probe are sampled from
                different sessions when session metadata exists; falls back to regular
                disjoint sampling when impossible.
            identification_filters: Optional record-level filters applied before
                split generation. Values can be scalars or lists. Keys can reference
                ImageRecord fields directly or parser metadata via metadata.<key>.

        Returns:
            List of dictionaries in identification CSV schema.
        """
        if gallery_samples_per_identity <= 0:
            raise ValueError("gallery_samples_per_identity must be > 0")
        if probes_per_identity == 0 or probes_per_identity < -1:
            raise ValueError("probes_per_identity must be -1 (all) or > 0")
        if number_of_identities == 0 or number_of_identities < -1:
            raise ValueError("number_of_identities must be -1 (all) or > 0")

        if identification_filters:
            records = [
                record
                for record in records
                if self._record_matches_identification_filters(record, identification_filters)
            ]

        by_identity: Dict[str, List[ImageRecord]] = {}
        for record in records:
            by_identity.setdefault(record.identity, []).append(record)

        min_required_samples = (
            gallery_samples_per_identity + 1
            if probes_per_identity == -1
            else gallery_samples_per_identity + probes_per_identity
        )

        eligible_identity_list = sorted(
            identity
            for identity, identity_records in by_identity.items()
            if len(identity_records) >= min_required_samples
        )

        if number_of_identities != -1 and len(eligible_identity_list) > number_of_identities:
            identity_list = self.rng.sample(eligible_identity_list, number_of_identities)
        else:
            identity_list = eligible_identity_list

        if number_of_identities != -1 and len(identity_list) < number_of_identities:
            print(
                f"⚠ Requested {number_of_identities} identities but only {len(identity_list)} are eligible after filtering and sample-count constraints; using all eligible identities."
            )

        rows: List[Dict[str, str]] = []
        record_idx = 1

        for identity in identity_list:
            identity_records = by_identity[identity]

            sampled_records = list(identity_records)
            self.rng.shuffle(sampled_records)

            gallery_records: List[ImageRecord] = []
            probe_candidates: List[ImageRecord] = []

            if require_session_disjoint:
                sessions: Dict[str, List[ImageRecord]] = {}
                no_session_bucket: List[ImageRecord] = []
                for rec in sampled_records:
                    if rec.session:
                        sessions.setdefault(str(rec.session), []).append(rec)
                    else:
                        no_session_bucket.append(rec)

                # Choose gallery from the largest session when possible.
                if sessions:
                    gallery_session = max(sessions.items(), key=lambda kv: len(kv[1]))[0]
                    gallery_pool = list(sessions.get(gallery_session, []))
                    self.rng.shuffle(gallery_pool)
                    if len(gallery_pool) >= gallery_samples_per_identity:
                        gallery_records = gallery_pool[:gallery_samples_per_identity]
                        probe_candidates = [
                            rec
                            for rec in sampled_records
                            if rec not in gallery_records and rec.session != gallery_session
                        ]

            if not gallery_records:
                gallery_records = sampled_records[:gallery_samples_per_identity]
                probe_candidates = sampled_records[gallery_samples_per_identity:]

            # Defensive disjointness: keep gallery and probe sample sets exclusive.
            gallery_sample_ids = {self._to_sample_id(rec) for rec in gallery_records}
            gallery_image_paths = {rec.image_path for rec in gallery_records}
            probe_candidates = [
                rec
                for rec in probe_candidates
                if self._to_sample_id(rec) not in gallery_sample_ids
                and rec.image_path not in gallery_image_paths
            ]

            if not probe_candidates:
                continue

            if probes_per_identity == -1:
                probe_records = probe_candidates
            else:
                if len(probe_candidates) < probes_per_identity:
                    continue
                probe_records = probe_candidates[:probes_per_identity]

            for split, split_records in (("gallery", gallery_records), ("probe", probe_records)):
                for rec in split_records:
                    rows.append(
                        {
                            "record_id": str(record_idx),
                            "sample_id": self._to_sample_id(rec),
                            "image_path": rec.image_path,
                            "identity": rec.identity,
                            "modality": rec.modality,
                            "dataset_name": rec.dataset_name,
                            "split": split,
                            "metadata": json.dumps(self._identification_metadata(rec), ensure_ascii=True),
                        }
                    )
                    record_idx += 1

        return rows

    def save_identification_to_csv(
        self,
        rows: List[Dict[str, str]],
        output_path: str,
        relative_to: Optional[str] = None,
    ):
        """Save identification rows to CSV file."""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        prepared_rows: List[Dict[str, str]] = []
        for row in rows:
            row_copy = dict(row)
            if relative_to:
                row_copy["image_path"] = os.path.relpath(row_copy["image_path"], relative_to)
            prepared_rows.append(row_copy)

        if prepared_rows:
            fieldnames = [
                "record_id",
                "sample_id",
                "image_path",
                "identity",
                "modality",
                "dataset_name",
                "split",
                "metadata",
            ]
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(prepared_rows)

        print(f"Saved {len(rows)} identification rows to {output_path}")

    def index_and_generate_identification(
        self,
        dataset_configs: List[Dict],
        output_csv: str,
        gallery_samples_per_identity: int,
        probes_per_identity: int,
        number_of_identities: int = -1,
        relative_paths: bool = True,
        require_session_disjoint: bool = False,
        identification_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """Complete workflow: index dataset(s) and generate identification CSV."""
        if not dataset_configs:
            raise ValueError("No dataset configs provided")

        all_rows: List[Dict[str, str]] = []
        for config in dataset_configs:
            records = self.index_dataset(
                dataset_path=config['dataset_path'],
                dataset_name=config['dataset_name'],
                modality=config['modality'],
                image_type=config.get('image_type'),
                modality_type=config.get('modality_type')
            )

            dataset_identification_filters: Dict[str, Any] = dict(identification_filters or {})
            if config.get('identification_filters'):
                dataset_identification_filters.update(config['identification_filters'])

            rows = self.generate_identification_rows_from_records(
                records=records,
                gallery_samples_per_identity=gallery_samples_per_identity,
                probes_per_identity=probes_per_identity,
                number_of_identities=number_of_identities,
                require_session_disjoint=require_session_disjoint,
                identification_filters=dataset_identification_filters,
            )
            all_rows.extend(rows)

            gallery_count = sum(1 for row in rows if row["split"] == "gallery")
            probe_count = len(rows) - gallery_count
            print(f"  Generated {len(rows)} identification rows:")
            print(f"    - {gallery_count} gallery, {probe_count} probe")

        relative_to = str(self.public_dataset_root) if relative_paths else None
        self.save_identification_to_csv(all_rows, output_csv, relative_to=relative_to)
        return all_rows
    
    def index_dataset(
        self,
        dataset_path: str,
        dataset_name: str,
        modality: str,
        parser_name: Optional[str] = None,
        image_type: Optional[str] = None,
        modality_type: Optional[str] = None
    ) -> List[ImageRecord]:
        """Index a single dataset.
        
        Args:
            dataset_path: Relative path from public_dataset_root or absolute path
            dataset_name: Name identifier for the dataset
            modality: Biometric modality (iris, face, hand, fingervein)
            parser_name: Optional specific parser to use
            image_type: Optional image type filter (e.g., "raw", "processed")
            modality_type: Optional modality type filter (e.g., "dorsal", "vein")
        Returns:
            List of ImageRecord objects
        """
        # Resolve path
        if os.path.isabs(dataset_path):
            full_path = dataset_path
        else:
            full_path = str(self.public_dataset_root / dataset_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Dataset path not found: {full_path}")
        
        print(f"Indexing {dataset_name} at {full_path}...")
        
        # Get appropriate parser
        parser = get_parser(full_path, dataset_name, modality, image_type=image_type, modality_type=modality_type)
        print(f"Using parser: {parser.__class__.__name__}")
        
        # Parse dataset
        records = parser.parse()
        
        # Store for later use
        self.indexed_records[dataset_name] = records
        
        # Print summary
        num_identities = len(set(r.identity for r in records))
        print(f"  Found {len(records)} images from {num_identities} identities")
        
        return records
    
    def generate_pairs_from_records(
        self,
        records: List[ImageRecord],
        genuine_per_identity: Optional[int] = None,
        max_genuine_pairs: Optional[int] = None,
        impostor_ratio: float = 1.0,
        match_constraints: Optional[Dict[str, bool]] = None
    ) -> List[ImagePair]:
        """Generate pairs from indexed records.
        
        Args:
            records: List of image records
            genuine_per_identity: Number of genuine pairs per identity. If None, generates all possible.
            max_genuine_pairs: Maximum number of genuine pairs. If set to -1, generate
                all possible genuine pairs up to a safety cap.
            impostor_ratio: Ratio of impostor to genuine pairs (1.0 = equal number)
            match_constraints: Optional constraints for impostor pair matching
            
        Returns:
            List of ImagePair objects
        """
        pairs = self.pair_generator.generate_pairs(
            records=records,
            genuine_per_identity=genuine_per_identity,
            max_genuine_pairs=max_genuine_pairs,
            impostor_ratio=impostor_ratio,
            match_constraints=match_constraints
        )
        
        return pairs
    
    def save_pairs_to_csv(
        self,
        pairs: List[ImagePair],
        output_path: str,
        relative_to: Optional[str] = None
    ):
        """Save pairs to CSV file.
        
        Args:
            pairs: List of ImagePair objects
            output_path: Path to output CSV file
            relative_to: Optional base path to make image paths relative to
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Convert to dictionaries
        pair_dicts = []
        for pair in pairs:
            pair_dict = pair.to_dict()
            
            # Make paths relative if requested
            if relative_to:
                pair_dict['image1_path'] = os.path.relpath(pair_dict['image1_path'], relative_to)
                pair_dict['image2_path'] = os.path.relpath(pair_dict['image2_path'], relative_to)
            
            pair_dicts.append(pair_dict)
        
        # Write CSV
        if pair_dicts:
            fieldnames = pair_dicts[0].keys()
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(pair_dicts)
        
        print(f"Saved {len(pairs)} pairs to {output_path}")
    
    def index_and_generate(
        self,
        dataset_configs: List[Dict],
        output_csv: str,
        genuine_per_identity: Optional[int] = None,
        max_genuine_pairs: Optional[int] = None,
        impostor_ratio: float = 1.0,
        relative_paths: bool = True
    ):
        """Complete workflow: index datasets and generate pairs in one call.
        
        Args:
            dataset_configs: List of dicts with keys: dataset_path, dataset_name, modality
            output_csv: Path to output CSV file
            genuine_per_identity: Number of genuine pairs per identity. If None, generates all possible.
            max_genuine_pairs: Maximum number of genuine pairs. If set to -1, generate
                all possible genuine pairs up to a safety cap.
            impostor_ratio: Ratio of impostor to genuine pairs (1.0 = equal number)
            relative_paths: Whether to save relative paths in CSV
        """
        all_pairs = []
        
        for config in dataset_configs:
            # Index dataset
            records = self.index_dataset(
                dataset_path=config['dataset_path'],
                dataset_name=config['dataset_name'],
                modality=config['modality'],
                image_type=config.get('image_type'),
                modality_type=config.get('modality_type')
            )
            
            # Generate pairs
            pairs = self.generate_pairs_from_records(
                records=records,
                genuine_per_identity=genuine_per_identity,
                max_genuine_pairs=max_genuine_pairs,
                impostor_ratio=impostor_ratio,
                match_constraints=config.get('match_constraints')
            )
            
            all_pairs.extend(pairs)
            
            # Print statistics
            genuine_count = sum(1 for p in pairs if p.ground_truth)
            impostor_count = len(pairs) - genuine_count
            
            print(f"  Generated {len(pairs)} pairs:")
            print(f"    - {genuine_count} genuine, {impostor_count} impostor")
        
        # Save all pairs
        relative_to = str(self.public_dataset_root) if relative_paths else None
        self.save_pairs_to_csv(all_pairs, output_csv, relative_to=relative_to)
        
        return all_pairs
    
    def print_statistics(self, pairs: Optional[List[ImagePair]] = None):
        """Print detailed statistics about indexed datasets and pairs.
        
        Args:
            pairs: Optional list of pairs. If None, prints only dataset stats.
        """
        print("\n=== Dataset Index Statistics ===")
        
        for dataset_name, records in self.indexed_records.items():
            num_identities = len(set(r.identity for r in records))
            print(f"\n{dataset_name}:")
            print(f"  Images: {len(records)}")
            print(f"  Identities: {num_identities}")
            print(f"  Modality: {records[0].modality if records else 'N/A'}")
        
        if pairs:
            print("\n=== Pair Statistics ===")
            print(f"Total pairs: {len(pairs)}")
            
            genuine = sum(1 for p in pairs if p.ground_truth)
            impostor = len(pairs) - genuine
            print(f"  Genuine: {genuine} ({genuine/len(pairs)*100:.1f}%)")
            print(f"  Impostor: {impostor} ({impostor/len(pairs)*100:.1f}%)")
            
            by_modality = {}
            for pair in pairs:
                by_modality[pair.modality] = by_modality.get(pair.modality, 0) + 1
            
            print(f"\nBy modality:")
            for modality, count in by_modality.items():
                print(f"  {modality}: {count}")
