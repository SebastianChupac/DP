# Identification Experiments Runner - Implementation Plan

## Overview
Implement closed-set identification experiments (1:N ranking) while reusing verification matcher infrastructure. The design mirrors `BatchExperimentRunner` for consistency but is distinct to avoid affecting verification workflows.

## Architecture Summary

### Input Data Format
Identification CSV: `record_id, sample_id, image_path, identity, modality, dataset_name, split, metadata`
- `split` column: "gallery" or "probe"
- `identity`: person/subject identifier (same across gallery and probe)
- Closed-set constraint: probe identities must exist in gallery

### Design Principles
1. **Matcher Reuse**: Use existing matchers (SIFT, LoFTR, SuperGlue, etc.) without modification
2. **Consistent Config Structure**: Mimic verification config for predictability
3. **Strategy Pattern**: Support multiple gallery aggregation methods
4. **Separation**: New classes (`IdentificationExperimentRunner`, `IdentificationExperimentConfig`) keep verification unaffected

---

## 1. Configuration Layer

### IdentificationExperimentConfig (extends ExperimentConfig structure)

**Location**: `bioverify/experiments/config.py` (extend existing)

**New fields**:
```python
@dataclass
class IdentificationExperimentConfig:
    # Same as ExperimentConfig
    experiment: ExperimentMetadata
    identification_dataset: str  # Path to identification CSV
    base_path: Optional[str] = None
    filter_modality: Optional[str] = None
    
    matchers: List[MatcherExperimentConfig]
    output_dir: str = "results"
    batch_size: int = 1
    verbose: bool = True
    device: str = "cuda"
    
    # NEW: Identification-specific fields
    strategy: str = "single"  # Options: "single", "multiple", "average", "fusion"
    top_k_ranks: List[int] = field(default_factory=lambda: [1, 5, 10])  # Ranks to report
    aggregation_method: str = "max"  # For strategy=="multiple": "max", "mean", "weighted"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "IdentificationExperimentConfig": ...
```

**YAML Example**:
```yaml
experiment:
  name: "Iris CASIA Identification - Single Strategy"
  description: "Closed-set identification on iris CASIA dataset"
  purpose: "identification_comparison"

identification_dataset: "bioverify/data/identification/identification_iris_casia.csv"
base_path: "C:/Users/sebas/Documents/VUT_FIT_MIT/DP/PublicDataset"

strategy: "single"  # Use single gallery template per identity
top_k_ranks: [1, 5, 10, 20]

matchers:
  - name: "loftr"
    config_base: "bioverify/config/matching/loftr.yaml"
    config_overrides:
      use_masking: true
  - name: "sift"
    config_base: "bioverify/config/matching/sift.yaml"

output_dir: "bioverify/results/identification_iris_casia_single"
batch_size: 10
verbose: true
device: "cuda"
```

---

## 2. Data Loading Layer

### IdentificationDataset (new class)

**Location**: `bioverify/data/identification.py` (new file)

**Key Components**:
- Load identification CSV
- Group gallery samples by identity
- Keep probe samples separate
- Apply modality/dataset filters if specified
- Support different strategy modes (which samples to use from gallery)

```python
class IdentificationDataset:
    """Loader for identification protocol CSV files."""
    
    def __init__(
        self,
        csv_path: str,
        base_path: Optional[str] = None,
        filter_modality: Optional[str] = None,
        strategy: str = "single",  # "single", "multiple"
    ):
        # Load CSV
        # Filter by modality if requested
        # Group by split (gallery/probe)
        # Group gallery by identity
        # Store as structured data
        
    def get_gallery(self) -> Dict[str, List[Dict]]:
        """Returns {identity_id: [sample_dicts]} for gallery."""
        
    def get_probes(self) -> List[Dict]:
        """Returns list of probe sample dicts."""
        
    def get_unique_identities_in_gallery(self) -> Set[str]:
        """Returns all identities present in gallery."""
        
    def validate_closed_set(self) -> bool:
        """Ensure all probe identities exist in gallery."""
```

---

## 3. Results and Metrics

### IdentificationResult (new class)

**Location**: `bioverify/results.py` (extend existing)

```python
@dataclass
class IdentificationResult:
    """Result of probing one sample against full gallery."""
    method_name: str  # Matcher name
    probe_id: str  # record_id of probe sample
    probe_sample_id: str
    probe_identity: str  # Ground truth identity
    modality: Optional[str] = None
    
    # Similarity scores by gallery identity
    scores_by_identity: Dict[str, float] = field(default_factory=dict)
    # Sorted ranks: [(identity, score), ...]
    ranked_identities: List[Tuple[str, float]] = field(default_factory=list)
    
    # Ground truth performance
    rank_of_true_identity: Optional[int] = None  # 1-indexed (1=top match)
    is_rank_1: Optional[bool] = None  # True if true identity ranked #1
    
    # Metadata
    gallery_size: int = 0
    num_gallery_samples_used: int = 0
    aggregation_method: Optional[str] = None
    matcher_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())
    
    def get_rank_k_hit(self, k: int) -> bool:
        """True if true identity is in top-k."""
        return self.rank_of_true_identity is not None and self.rank_of_true_identity <= k
```

### Identification Metrics (new functions)

**Location**: `bioverify/evaluation/metrics.py` (extend existing)

```python
def compute_cmc_curve(
    results: List[IdentificationResult],
    max_rank: Optional[int] = None
) -> Dict[str, Any]:
    """Compute Cumulative Match Characteristic curve.
    
    CMC shows probability of true identity being in top-k matches.
    Returns cumulative hit rate at each rank.
    """
    # Count hits at each rank
    # Compute cumulative probabilities
    # Return {rank: hit_rate} dict
    
def compute_rank_k_accuracy(
    results: List[IdentificationResult],
    k: int
) -> float:
    """Compute top-k accuracy: % of probes with true identity in top-k."""
    
def compute_mean_average_precision(
    results: List[IdentificationResult]
) -> float:
    """Compute mAP: average of per-probe AP scores."""
    # For each probe, rank is binary (true/false) for each gallery identity
    # AP = sum of (precision@i * relevance@i) / num_relevant
```

---

## 4. Runner Logic

### IdentificationExperimentRunner (new class)

**Location**: `bioverify/experiments/identification_runner.py` (new file)

**Key Algorithm**:

```python
class IdentificationExperimentRunner:
    """Batch closed-set identification runner."""
    
    def __init__(self, config: IdentificationExperimentConfig):
        ...
    
    def run(self) -> Tuple[List[IdentificationResult], Dict]:
        """Run identification experiments.
        
        1. Load identification dataset (gallery + probes)
        2. Instantiate matchers
        3. For each probe:
             a. For each gallery identity:
                  - Aggregate similarity scores from all gallery samples
                    (strategy: single / multiple / average)
             b. Rank gallery identities by score
             c. Record rank of true identity
        4. Compute metrics (CMC, Rank-k, mAP)
        5. Save results
        """
        # Load data
        dataset = IdentificationDataset(
            csv_path=config.identification_dataset,
            base_path=config.base_path,
            filter_modality=config.filter_modality,
            strategy=config.strategy
        )
        
        gallery = dataset.get_gallery()
        probes = dataset.get_probes()
        
        # Instantiate matchers (same as verification)
        matchers = self._instantiate_matchers()
        
        # Process probes
        results = []
        for probe in probes:  # tqdm wrapper
            for matcher_name, matcher in matchers.items():
                result = self._process_probe_against_gallery(
                    probe=probe,
                    gallery=gallery,
                    matcher=matcher,
                    matcher_name=matcher_name,
                    strategy=config.strategy
                )
                results.append(result)
        
        # Compute metrics
        metrics = self._compute_metrics(results, config.top_k_ranks)
        
        # Save results
        self._save_results(results, metrics)
        
        return results, metrics
    
    def _process_probe_against_gallery(
        self,
        probe: Dict,
        gallery: Dict[str, List[Dict]],
        matcher,
        matcher_name: str,
        strategy: str
    ) -> IdentificationResult:
        """Match one probe against entire gallery."""
        
        scores_by_identity = {}
        
        for identity, gallery_samples in gallery.items():
            if strategy == "single":
                # Use only first/best gallery sample for this identity
                gallery_sample = gallery_samples[0]
                score = self._compute_similarity(
                    probe, gallery_sample, matcher
                )
                scores_by_identity[identity] = score
                
            elif strategy == "multiple":
                # Compute score against all gallery samples, aggregate
                scores = [
                    self._compute_similarity(probe, g_sample, matcher)
                    for g_sample in gallery_samples
                ]
                # Aggregate: max, mean, or weighted
                scores_by_identity[identity] = self._aggregate_scores(
                    scores, method=config.aggregation_method
                )
            
            elif strategy == "average":
                # Create virtual gallery template by averaging features
                # (more complex, deferred to later optimization)
                pass
        
        # Rank identities by score
        ranked = sorted(
            scores_by_identity.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Find rank of true identity
        probe_identity = probe['identity']
        true_identity_rank = None
        for rank, (identity, score) in enumerate(ranked, 1):
            if identity == probe_identity:
                true_identity_rank = rank
                break
        
        return IdentificationResult(
            method_name=matcher_name,
            probe_id=probe['record_id'],
            probe_identity=probe_identity,
            scores_by_identity=scores_by_identity,
            ranked_identities=ranked,
            rank_of_true_identity=true_identity_rank,
            is_rank_1=(true_identity_rank == 1),
            gallery_size=len(gallery),
            aggregation_method=strategy,
        )
    
    def _compute_similarity(
        self,
        probe: Dict,
        gallery_sample: Dict,
        matcher
    ) -> float:
        """Compute similarity score between probe and gallery sample.
        
        Reuses matcher.match() from verification.
        Returns matcher confidence score (0-1, higher=more similar).
        """
        result = matcher.match(
            img1_path=probe['image_path'],
            img2_path=gallery_sample['image_path'],
            modality=probe['modality'],
            visualize=False,
            ground_truth=None,
        )
        return result.verification_confidence
    
    def _compute_metrics(
        self,
        results: List[IdentificationResult],
        top_k_ranks: List[int]
    ) -> Dict:
        """Compute identification metrics by matcher."""
        by_matcher = {}
        for result in results:
            if result.method_name not in by_matcher:
                by_matcher[result.method_name] = []
            by_matcher[result.method_name].append(result)
        
        metrics = {}
        for matcher_name, matcher_results in by_matcher.items():
            scores_array = np.array([
                r.rank_of_true_identity for r in matcher_results
                if r.rank_of_true_identity is not None
            ])
            
            metrics[matcher_name] = {
                'num_probes': len(matcher_results),
                'rank_1_accuracy': np.mean([r.is_rank_1 for r in matcher_results]),
                'cmc_curve': compute_cmc_curve(matcher_results),
                'rank_k_accuracy': {
                    k: compute_rank_k_accuracy(matcher_results, k)
                    for k in top_k_ranks
                },
                'mean_average_precision': compute_mean_average_precision(matcher_results),
                'rank_stats': {
                    'mean': float(np.mean(scores_array)),
                    'median': float(np.median(scores_array)),
                    'std': float(np.std(scores_array)),
                }
            }
        
        return metrics
```

---

## 5. Integration into CLI

**Location**: `bioverify/cli/index.py` (extend existing)

Add new command:
```python
def identification_experiment_command(args):
    """Execute identification experiment from YAML config."""
    try:
        config = IdentificationExperimentConfig.from_yaml(args.config)
        runner = IdentificationExperimentRunner(config)
        results, metrics = runner.run()
        print("✅ Identification experiment complete")
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
```

Add to CLI argparse:
```
python -m bioverify.cli.index identification --config config/experiments/identification_iris_casia.yaml
```

---

## 6. Implementation Roadmap

### Phase 1: Core Data & Results (Minimal)
- [ ] IdentificationDataset loader
- [ ] IdentificationResult dataclass
- [ ] Basic metrics (Rank-1, Rank-k accuracy, CMC)

### Phase 2: Basic Runner (Single Strategy)
- [ ] IdentificationExperimentConfig
- [ ] IdentificationExperimentRunner (strategy == "single")
- [ ] Example YAML configs
- [ ] CLI integration

### Phase 3: Multiple Strategies & Optimization (Later)
- [ ] "multiple" strategy (aggregate over gallery samples)
- [ ] "average" strategy (feature averaging)
- [ ] Batched similarity computation
- [ ] GPU optimization for large galleries

---

## 7. Key Differences from Verification

| Aspect | Verification | Identification |
|--------|---|---|
| Task | 1:1 comparison | 1:N ranking |
| CSV Schema | image1, image2 pairs | single image + split label |
| Result | binary decision | ranked list + score |
| Metrics | TAR/FAR/EER/ROC | CMC/Rank-k/mAP |
| Per-pair work | one match per pair | one match per probe-gallery pair |

---

## 8. Reuse Strategy

**What we reuse:**
- Matcher instantiation (`create_matcher`, `_load_matcher_config`)
- Matcher interface (`matcher.match()` returns similarity score)
- Result structure (extend, don't replace VerificationResult)
- Config loading pattern (inherit structure, add new fields)
- CLI pattern (new command, same framework)

**What is new:**
- Data loading (gallery grouping, split handling)
- Result aggregation (ranking instead of binary decision)
- Metrics (CMC vs ROC)
- Runner loop (all-gallery comparison instead of pair iteration)

---

## 9. Future Optimizations (Design, not implement yet)

1. **Batch Similarity Computation**: Compute all probe-vs-gallery matches in one batch
2. **GPU-Cached Gallery**: Load gallery features once, match against multiple probes
3. **Two-Stage Retrieval**: Coarse ranking + reranking with expensive matchers
4. **Identity Representative**: Use single high-quality template per identity (expensive precomputation)
5. **Parameter Tuning**: Confidence threshold sweep for top-k accuracy

