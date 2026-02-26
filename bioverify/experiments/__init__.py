"""
Experiment running infrastructure for batch processing and evaluation.

Provides tools for running matchers on datasets, parameter tuning, and result analysis.
"""

from .config import ExperimentConfig
from .runner import BatchExperimentRunner, run_experiment

__all__ = ['ExperimentConfig', 'BatchExperimentRunner', 'run_experiment']

