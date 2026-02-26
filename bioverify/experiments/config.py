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
    - Single matcher, multiple datasets (cross-dataset evaluation)
    - Multiple matchers, single dataset (matcher comparison)
    - Multiple matchers, multiple datasets (comprehensive evaluation)
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
