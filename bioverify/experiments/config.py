"""
Experiment configuration dataclass and loading utilities.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any
import yaml
from pathlib import Path


@dataclass
class MatcherExperimentConfig:
    """Configuration for a matcher within an experiment."""
    name: str                                    # e.g., "loftr", "sift"
    config_base: Optional[str] = None            # Path to base matcher config YAML
    config_overrides: Dict[str, Any] = field(default_factory=dict)  # Params that vary in experiment


@dataclass
class ExperimentMetadata:
    """Metadata about the experiment."""
    name: str                                    # Human-readable experiment name
    description: Optional[str] = None            # Description of what this experiment tests
    researcher: Optional[str] = None             # Who ran it
    created: Optional[str] = None                # ISO timestamp
    purpose: Optional[str] = None                # e.g., "threshold_tuning", "param_sweep", "cross_dataset"


@dataclass
class ExperimentConfig:
    """
    Configuration for a batch experiment.
    
    Typical use cases:
    - Single matcher, single dataset, parameter sweep (threshold tuning)
    - Multiple matchers, single dataset (matcher comparison)
    """
    # Experiment metadata
    experiment: ExperimentMetadata
    
    # Data configuration
    dataset: str                                 # Path to pairs CSV manifest
    base_path: Optional[str] = None              # Base path for resolving relative image paths
    filter_modality: Optional[str] = None        # Optional filter: only process specific modality
    filter_dataset: Optional[str] = None         # Optional filter: only process specific dataset
    
    # Matcher configuration
    matchers: List[MatcherExperimentConfig] = field(default_factory=list)
    
    # Output configuration
    output_dir: str = "results"                  # Output directory for results
    save_visualization_results: bool = False     # Whether to save rich VisualizationResults
    save_pair_images: bool = False               # Whether to save matched pair visualizations
    
    # Processing options
    batch_size: int = 1                          # Process pairs in batches
    verbose: bool = True                         # Print progress
    device: str = "cuda"                         # Device for torch models
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Load experiment config from YAML file.
        
        Args:
            yaml_path: Path to experiment YAML file
            
        Returns:
            ExperimentConfig instance
            
        Raises:
            FileNotFoundError: If YAML file not found
            ValueError: If YAML format invalid
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML format: expected dict, got {type(data)}")
        
        # Parse metadata
        experiment_data = data.get('experiment', {})
        if isinstance(experiment_data, dict):
            experiment = ExperimentMetadata(**experiment_data)
        else:
            raise ValueError("experiment field must be a dict")
        
        # Parse matchers
        matchers = []
        matchers_data = data.get('matchers', [])
        if isinstance(matchers_data, list):
            for m in matchers_data:
                matcher = MatcherExperimentConfig(
                    name=m['name'],
                    config_base=m.get('config_base'),
                    config_overrides=m.get('config_overrides', {})
                )
                matchers.append(matcher)
        else:
            raise ValueError("matchers field must be a list")
        
        # Create config
        config = cls(
            experiment=experiment,
            dataset=data.get('dataset'),
            base_path=data.get('base_path'),
            filter_modality=data.get('filter_modality'),
            filter_dataset=data.get('filter_dataset'),
            matchers=matchers,
            output_dir=data.get('output_dir', 'results'),
            save_visualization_results=data.get('save_visualization_results', False),
            save_pair_images=data.get('save_pair_images', False),
            batch_size=data.get('batch_size', 1),
            verbose=data.get('verbose', True),
            device=data.get('device', 'cuda')
        )
        
        return config
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for JSON/YAML serialization."""
        return asdict(self)
    
    def to_yaml(self) -> str:
        """Convert config to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False)


@dataclass
class IdentificationExperimentConfig:
    """Configuration for closed-set identification experiments."""

    # Experiment metadata
    experiment: ExperimentMetadata

    # Data configuration
    identification_dataset: str
    base_path: Optional[str] = None
    filter_modality: Optional[str] = None
    filter_dataset: Optional[str] = None

    # Matcher configuration
    matchers: List[MatcherExperimentConfig] = field(default_factory=list)

    # Identification strategy configuration
    ranking_strategy: str = "bruteforce"                  # bruteforce | cascade
    samples_per_gallery: str = "single"                   # single | multiple - how many gallery samples to use
    aggregation_method: str = "max"                       # max | mean - how to combine scores from multiple samples
    
    # Cascade strategy specific parameters
    shortlist_matcher: MatcherExperimentConfig = field(
        default_factory=lambda: MatcherExperimentConfig(name="sift-fingervein")
    )
    shortlist_k: int = 15                                 # Number of top candidates to keep in shortlist
    
    top_k_ranks: List[int] = field(default_factory=lambda: [1, 5, 10])
    cache_gallery_templates: bool = True

    # Output configuration
    output_dir: str = "results"
    verbose: bool = True
    device: str = "cuda"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "IdentificationExperimentConfig":
        """Load identification experiment config from YAML."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML format: expected dict, got {type(data)}")

        experiment_data = data.get('experiment', {})
        if not isinstance(experiment_data, dict):
            raise ValueError("experiment field must be a dict")
        experiment = ExperimentMetadata(**experiment_data)

        matchers: List[MatcherExperimentConfig] = []
        matchers_data = data.get('matchers', [])
        if not isinstance(matchers_data, list):
            raise ValueError("matchers field must be a list")
        for m in matchers_data:
            matcher = MatcherExperimentConfig(
                name=m['name'],
                config_base=m.get('config_base'),
                config_overrides=m.get('config_overrides', {}),
            )
            matchers.append(matcher)

        # Handle new ranking_strategy field (bruteforce | cascade)
        ranking_strategy = data.get('ranking_strategy', 'bruteforce')
        if ranking_strategy not in ('bruteforce', 'cascade'):
            raise ValueError("ranking_strategy must be one of: bruteforce, cascade")

        # Handle samples_per_gallery (single | multiple)
        samples_per_gallery = data.get('samples_per_gallery', 'single')
        if samples_per_gallery not in ('single', 'multiple'):
            raise ValueError("samples_per_gallery must be one of: single, multiple")

        aggregation_method = data.get('aggregation_method', 'max')
        if aggregation_method not in ('max', 'mean'):
            raise ValueError("aggregation_method must be one of: max, mean")

        top_k_ranks = data.get('top_k_ranks', [1, 5, 10])
        if not isinstance(top_k_ranks, list) or not top_k_ranks:
            raise ValueError("top_k_ranks must be a non-empty list")

        dataset_path = data.get('identification_dataset') or data.get('dataset')
        if not dataset_path:
            raise ValueError("identification_dataset is required")

        shortlist_matcher_data = data.get('shortlist_matcher', {'name': 'sift-fingervein'})
        # Backward compatibility: allow string shorthand for shortlist matcher name.
        if isinstance(shortlist_matcher_data, str):
            shortlist_matcher_data = {'name': shortlist_matcher_data}
        if not isinstance(shortlist_matcher_data, dict):
            raise ValueError("shortlist_matcher must be a dict with keys: name, config_base, config_overrides")
        if 'name' not in shortlist_matcher_data:
            raise ValueError("shortlist_matcher.name is required")
        shortlist_matcher = MatcherExperimentConfig(
            name=shortlist_matcher_data['name'],
            config_base=shortlist_matcher_data.get('config_base'),
            config_overrides=shortlist_matcher_data.get('config_overrides', {}),
        )

        shortlist_k = data.get('shortlist_k', 15)
        if not isinstance(shortlist_k, int) or shortlist_k < 1:
            raise ValueError("shortlist_k must be a positive integer")

        return cls(
            experiment=experiment,
            identification_dataset=dataset_path,
            base_path=data.get('base_path'),
            filter_modality=data.get('filter_modality'),
            filter_dataset=data.get('filter_dataset'),
            matchers=matchers,
            ranking_strategy=ranking_strategy,
            samples_per_gallery=samples_per_gallery,
            aggregation_method=aggregation_method,
            shortlist_matcher=shortlist_matcher,
            shortlist_k=shortlist_k,
            top_k_ranks=top_k_ranks,
            cache_gallery_templates=data.get('cache_gallery_templates', True),
            output_dir=data.get('output_dir', 'results'),
            verbose=data.get('verbose', True),
            device=data.get('device', 'cuda'),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False)
