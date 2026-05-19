# pairs.py - Pair generation for biometric verification.
# Author: Sebastian Chupac (xchupa03)
# Date: 19.05.2026

"""
Pair generation for biometric verification.

Generates genuine (same identity) and impostor (different identity) pairs
from parsed image records for inference and evaluation.
"""
import random
import itertools
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import json

from .parsers import ImageRecord


MAX_GENUINE_PAIRS_CAP = 100_000


@dataclass
class ImagePair:
    """Represents a pair of images for verification."""
    pair_id: str
    image1_path: str
    image2_path: str
    modality: str
    ground_truth: bool  # True for genuine (same person), False for impostor (different person)
    identity1: str
    identity2: str
    dataset_name: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV/JSON serialization."""
        return {
            'pair_id': self.pair_id,
            'image1_path': self.image1_path,
            'image2_path': self.image2_path,
            'modality': self.modality,
            'ground_truth': self.ground_truth,
            'identity1': self.identity1,
            'identity2': self.identity2,
            'dataset_name': self.dataset_name,
            'metadata': json.dumps(self.metadata) if self.metadata else '{}'
        }


class PairGenerator:
    """Generates genuine and impostor pairs from image records."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize pair generator.
        
        Args:
            random_seed: Seed for reproducible random pair generation
        """
        self.random_seed = random_seed
        random.seed(random_seed)
    
    def generate_genuine_pairs(
        self, 
        records: List[ImageRecord], 
        num_pairs_per_identity: Optional[int] = None,
        max_pairs: Optional[int] = None,
        match_constraints: Optional[Dict[str, bool]] = None
    ) -> List[Tuple[ImageRecord, ImageRecord]]:
        """Generate genuine pairs (same identity).
        
        Args:
            records: List of image records
            num_pairs_per_identity: Number of pairs to generate per identity.
                If None, generates all possible pairs.
            max_pairs: Maximum total pairs to generate. If specified, sampling
                is done proportionally across all identities to avoid bias.
            match_constraints: Optional dict specifying matching constraints
                (e.g., {'side': True} to match L with L, R with R only)
            
        Returns:
            List of (record1, record2) tuples with same identity
        """
        if match_constraints is None:
            match_constraints = {}
        
        # Group records by identity
        identity_groups = defaultdict(list)
        for record in records:
            identity_groups[record.identity].append(record)
        
        # Generate candidate pairs for each identity
        identity_pairs = {}
        for identity, images in identity_groups.items():
            if len(images) < 2:
                continue  # Need at least 2 images for a pair
            
            # Generate all possible pairs for this identity
            all_pairs = list(itertools.combinations(images, 2))
            
            # Apply constraints: only keep pairs where constrained attributes match
            if match_constraints:
                filtered_pairs = []
                for img1, img2 in all_pairs:
                    # Check if all constrained attributes match
                    matches = True
                    for attr in match_constraints.keys():
                        val1 = getattr(img1, attr, None)
                        val2 = getattr(img2, attr, None)
                        if val1 != val2:
                            matches = False
                            break
                    if matches:
                        filtered_pairs.append((img1, img2))
                all_pairs = filtered_pairs
            
            if not all_pairs:
                continue  # No valid pairs after constraint filtering
            
            if num_pairs_per_identity is not None and len(all_pairs) > num_pairs_per_identity:
                # Randomly sample specified number of pairs
                identity_pairs[identity] = random.sample(all_pairs, num_pairs_per_identity)
            else:
                identity_pairs[identity] = all_pairs
        
        # Collect all pairs
        all_pairs = []
        for pairs_list in identity_pairs.values():
            all_pairs.extend(pairs_list)
        
        # Apply max_pairs limit with random sampling to avoid bias
        if max_pairs is not None and len(all_pairs) > max_pairs:
            all_pairs = random.sample(all_pairs, max_pairs)
        
        return all_pairs
    
    def generate_impostor_pairs(
        self,
        records: List[ImageRecord],
        max_pairs: Optional[int] = None,
        match_constraints: Optional[Dict[str, bool]] = None
    ) -> List[Tuple[ImageRecord, ImageRecord]]:
        """Generate impostor pairs (different identities).
        
        Args:
            records: List of image records
            max_pairs: Maximum number of pairs to generate. Takes precedence over num_pairs.
            match_constraints: Optional dict specifying matching constraints
                (e.g., {'side': True} to match L with L, R with R)
                
        Returns:
            List of (record1, record2) tuples with different identities
        """
        # Determine target number of pairs
        target_pairs = max_pairs
        if target_pairs is None:
            raise ValueError("Must specify max_pairs")
        
        if match_constraints is None:
            match_constraints = {}
        
        pairs = []
        
        # Group records by identity
        identity_groups = defaultdict(list)
        for record in records:
            identity_groups[record.identity].append(record)
        
        identities = list(identity_groups.keys())
        if len(identities) < 2:
            return []
        
        if match_constraints:
            # Group records by constraints for efficient matching
            constraint_groups = defaultdict(list)
            for record in records:
                key = tuple(getattr(record, k, None) for k in match_constraints.keys())
                constraint_groups[key].append(record)
            
            # Generate candidate pairs from each constrained group
            all_candidate_pairs = []
            seen_pairs = set()  # Track to avoid duplicates
            
            for group_records in constraint_groups.values():
                if len(group_records) < 2:
                    continue
                
                # Get identities within this constrained group
                group_identity_map = defaultdict(list)
                for record in group_records:
                    group_identity_map[record.identity].append(record)
                
                group_identities = list(group_identity_map.keys())
                if len(group_identities) < 2:
                    continue
                
                # Generate pairs within this group
                # Generate more candidates than needed to ensure fair sampling across groups
                pairs_per_group = max(target_pairs, 1000)  # Generate enough candidates
                attempts = 0
                max_attempts = pairs_per_group * 10
                group_pairs = []
                
                while len(group_pairs) < pairs_per_group and attempts < max_attempts:
                    attempts += 1
                    id1, id2 = random.sample(group_identities, 2)
                    img1 = random.choice(group_identity_map[id1])
                    img2 = random.choice(group_identity_map[id2])
                    
                    # Use image paths as unique identifier to avoid duplicates
                    pair_key = (img1.image_path, img2.image_path)
                    if pair_key not in seen_pairs and (pair_key[1], pair_key[0]) not in seen_pairs:
                        seen_pairs.add(pair_key)
                        group_pairs.append((img1, img2))
                
                all_candidate_pairs.extend(group_pairs)
            
            # Randomly sample target_pairs from all candidates to ensure fair distribution
            if len(all_candidate_pairs) > target_pairs:
                pairs = random.sample(all_candidate_pairs, target_pairs)
            else:
                pairs = all_candidate_pairs
        else:
            # No constraints - randomly sample any two different identities
            attempts = 0
            max_attempts = target_pairs * 10
            seen_pairs = set()  # Track to avoid duplicates
            
            while len(pairs) < target_pairs and attempts < max_attempts:
                attempts += 1
                id1, id2 = random.sample(identities, 2)
                img1 = random.choice(identity_groups[id1])
                img2 = random.choice(identity_groups[id2])
                
                # Use image paths as unique identifier to avoid duplicates
                pair_key = (img1.image_path, img2.image_path)
                if pair_key not in seen_pairs and (pair_key[1], pair_key[0]) not in seen_pairs:
                    seen_pairs.add(pair_key)
                    pairs.append((img1, img2))
        
        return pairs[:target_pairs]
    
    def generate_pairs(
        self,
        records: List[ImageRecord],
        genuine_per_identity: Optional[int] = None,
        max_genuine_pairs: Optional[int] = None,
        impostor_ratio: float = 1.0,
        match_constraints: Optional[Dict[str, bool]] = None
    ) -> List[ImagePair]:
        """Generate genuine and impostor pairs for evaluation.
        
        Args:
            records: List of image records
            genuine_per_identity: Number of genuine pairs per identity. If None, generates all possible.
            max_genuine_pairs: Maximum number of genuine pairs. If set to -1, generate
                all possible genuine pairs up to MAX_GENUINE_PAIRS_CAP.
            impostor_ratio: Ratio of impostor to genuine pairs (1.0 = equal number)
            match_constraints: Optional matching constraints for impostor pairs
                (e.g., {'side': True} to match L with L, R with R)
            
        Returns:
            List of ImagePair objects
        """
        if not records:
            return []
        
        dataset_name = records[0].dataset_name if records else "unknown"
        modality = records[0].modality if records else "unknown"
        
        if max_genuine_pairs == -1:
            max_genuine_pairs = MAX_GENUINE_PAIRS_CAP
            genuine_per_identity = None

        # Generate genuine pairs
        genuine_pairs = self.generate_genuine_pairs(
            records,
            num_pairs_per_identity=genuine_per_identity,
            max_pairs=max_genuine_pairs,
            match_constraints=match_constraints
        )
        
        # Determine number of impostor pairs to generate
        num_impostors = int(len(genuine_pairs) * impostor_ratio)
        
        # Generate impostor pairs
        impostor_pairs = self.generate_impostor_pairs(
            records,
            max_pairs=num_impostors,
            match_constraints=match_constraints
        )
        
        # Convert to ImagePair objects
        all_pairs = []
        
        for idx, (rec1, rec2) in enumerate(genuine_pairs):
            pair_id = f"{dataset_name}_genuine_{idx}"
            pair = ImagePair(
                pair_id=pair_id,
                image1_path=rec1.image_path,
                image2_path=rec2.image_path,
                modality=modality,
                ground_truth=True,
                identity1=rec1.identity,
                identity2=rec2.identity,
                dataset_name=dataset_name,
                metadata={
                    'record1_meta': rec1.metadata,
                    'record2_meta': rec2.metadata
                }
            )
            all_pairs.append(pair)
        
        for idx, (rec1, rec2) in enumerate(impostor_pairs):
            pair_id = f"{dataset_name}_impostor_{idx}"
            pair = ImagePair(
                pair_id=pair_id,
                image1_path=rec1.image_path,
                image2_path=rec2.image_path,
                modality=modality,
                ground_truth=False,
                identity1=rec1.identity,
                identity2=rec2.identity,
                dataset_name=dataset_name,
                metadata={
                    'record1_meta': rec1.metadata,
                    'record2_meta': rec2.metadata
                }
            )
            all_pairs.append(pair)
        
        return all_pairs
