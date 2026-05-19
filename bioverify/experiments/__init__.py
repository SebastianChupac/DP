# __init__.py - Experiment running infrastructure for batch processing and evaluation.
# Author: Sebastian Chupac (xchupa03)
# Date: 19.05.2026


"""
Experiment running infrastructure for batch processing and evaluation.

Provides tools for running matchers on datasets, parameter tuning, and result analysis.
"""

from .config import ExperimentConfig, IdentificationExperimentConfig
from .runner import BatchExperimentRunner, run_experiment
from .identification_runner import IdentificationExperimentRunner, run_identification_experiment

__all__ = [
	'ExperimentConfig',
	'IdentificationExperimentConfig',
	'BatchExperimentRunner',
	'IdentificationExperimentRunner',
	'run_experiment',
	'run_identification_experiment',
]

