# runner.py - Batch experiment runner for biometric verification experiments.
# Author: Sebastian Chupac (xchupa03)
# Date: 19.05.2026

"""
Batch experiment runner for processing pairs and evaluating matchers.
"""

import os
import json
import csv
import yaml
from statistics import mean, median
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict
from datetime import datetime
import traceback

from .config import ExperimentConfig, MatcherExperimentConfig
from ..data.dataset import PairDataset
from ..matchers.registry import create_matcher
from ..matchers.base import MatcherConfig
from ..results import VerificationResult

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, total=None, disable=False):
        """Fallback tqdm if not installed."""
        return iterable


class BatchExperimentRunner:
    """
    Batch experiment runner for processing pairs with matchers.
    
    Handles:
    - Loading pairs from CSV
    - Instantiating matchers with config overrides
    - Processing all pairs through matchers
    - Saving results to JSON and CSV
    - Computing summary metrics
    """
    
    def __init__(self, experiment_config: ExperimentConfig):
        """Initialize batch runner.
        
        Args:
            experiment_config: ExperimentConfig instance
        """
        self.config = experiment_config
        self.results: List[VerificationResult] = []
        self.metrics: Dict = {}
        self.errors: List[Dict] = []
        
        # Create output directory
        Path(experiment_config.output_dir).mkdir(parents=True, exist_ok=True)
        
        if experiment_config.verbose:
            print(f"Experiment: {experiment_config.experiment.name}")
            print(f"Output dir: {experiment_config.output_dir}")
            print(f"Pairs CSV: {experiment_config.dataset}")
            print(f"Matchers: {[m.name for m in experiment_config.matchers]}")
            if self.config.profile_timings:
                print("Profiling enabled: per-stage timings will be recorded in ms")
    
    def _load_matcher_config(
        self,
        matcher_cfg: MatcherExperimentConfig
    ) -> dict:
        """Load and merge matcher config from base + overrides.
        
        Args:
            matcher_cfg: Matcher experiment config with base path and overrides
            
        Returns:
            Configuration dictionary for create_matcher()
        """
        # Load base config if specified
        config_dict = {}
        
        if matcher_cfg.config_base:
            base_path = Path(matcher_cfg.config_base)
            if not base_path.exists():
                raise FileNotFoundError(f"Base config not found: {matcher_cfg.config_base}")
            
            with open(base_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            # Extract the matcher config section
            if 'matcher' in yaml_data and 'config' in yaml_data['matcher']:
                config_dict = yaml_data['matcher']['config'].copy()
        
        # Apply overrides
        if matcher_cfg.config_overrides:
            config_dict.update(matcher_cfg.config_overrides)

        # Pass experiment base path through to matchers for mask/image lookup.
        if self.config.base_path and "public_dataset_root" not in config_dict:
            config_dict["public_dataset_root"] = self.config.base_path
        
        # Ensure device is set
        if 'device' not in config_dict:
            config_dict['device'] = self.config.device

        if 'profile_timings' not in config_dict:
            config_dict['profile_timings'] = self.config.profile_timings
        
        return config_dict

    @staticmethod
    def _normalize_device_name(value) -> Optional[str]:
        if value is None:
            return None
        try:
            text = str(value).strip()
            return text if text else None
        except Exception:
            return None

    def _extract_device_from_object(self, obj) -> Optional[str]:
        """Best-effort extraction of runtime device from matcher internals."""
        if obj is None:
            return None

        direct_device = getattr(obj, "_device", None)
        if direct_device is None:
            direct_device = getattr(obj, "device", None)
        if direct_device is not None and not callable(direct_device):
            normalized = self._normalize_device_name(direct_device)
            if normalized:
                return normalized

        if torch is None:
            return None

        if isinstance(obj, torch.nn.Module):
            for parameter in obj.parameters():
                return self._normalize_device_name(parameter.device)
            for buffer in obj.buffers():
                return self._normalize_device_name(buffer.device)

        nested_model = getattr(obj, "model", None)
        if nested_model is not None and nested_model is not obj:
            nested_device = self._extract_device_from_object(nested_model)
            if nested_device:
                return nested_device

        return None

    def _get_matcher_runtime_device(self, matcher) -> Optional[str]:
        """Return actual matcher runtime device when it can be determined."""
        get_runtime_device = getattr(matcher, "get_runtime_device", None)
        if callable(get_runtime_device):
            detected = get_runtime_device()
            if detected:
                return detected

        for attr in ("_device", "device", "_matching", "_matcher", "model", "_extractor"):
            if not hasattr(matcher, attr):
                continue
            value = getattr(matcher, attr)
            if callable(value):
                continue
            detected = self._extract_device_from_object(value)
            if detected:
                return detected
        return None
    
    def run(self) -> Tuple[List[VerificationResult], Dict]:
        """Run the batch experiment.
        
        Returns:
            Tuple of (results_list, summary_metrics)
        """
        # Load pairs dataset
        if self.config.verbose:
            print(f"\nLoading pairs...")
        
        dataset = PairDataset(
            csv_path=self.config.dataset,
            base_path=self.config.base_path,
            filter_modality=self.config.filter_modality,
            filter_dataset=self.config.filter_dataset,
            load_images=False  # Don't load images yet
        )
        
        if len(dataset) == 0:
            print("Warning: No pairs to process!")
            return [], {}
        
        # Instantiate matchers
        if self.config.verbose:
            print(f"\nInstantiating matchers...")
        
        matchers = {}
        for matcher_cfg in self.config.matchers:
            try:
                matcher_config = self._load_matcher_config(matcher_cfg)
                matcher = create_matcher(matcher_cfg.name, matcher_config)
                matchers[matcher_cfg.name] = matcher
                
                if self.config.verbose:
                    runtime_device = self._get_matcher_runtime_device(matcher)
                    if runtime_device:
                        print(f"   {matcher_cfg.name} (device: {runtime_device})")
                    else:
                        print(f"   {matcher_cfg.name}")
            except Exception as e:
                error_msg = f"Failed to instantiate matcher {matcher_cfg.name}: {str(e)}"
                print(f"   Error: {error_msg}")
                self.errors.append({
                    'type': 'matcher_initialization',
                    'matcher': matcher_cfg.name,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                continue
        
        if not matchers:
            raise RuntimeError("No matchers successfully instantiated!")
        
        # Process pairs
        if self.config.verbose:
            print(f"\nProcessing {len(dataset)} pairs...")
        
        progress_bar = tqdm(
            dataset.pairs,
            desc="Pairs",
            total=len(dataset),
            disable=not self.config.verbose
        )
        
        for pair in progress_bar:
            for matcher_name, matcher in matchers.items():
                try:
                    result = matcher.match(
                        img1_path=pair.image1_path,
                        img2_path=pair.image2_path,
                        modality=pair.modality,
                        visualize=False,
                        ground_truth=pair.ground_truth,
                        matcher_name=matcher_name,  # Pass experiment matcher name (e.g., "sift-v1")
                    )
                    
                    if result is not None:
                        # Add pair metadata to result
                        result.metadata['pair_id'] = pair.pair_id
                        result.metadata['identity1'] = pair.identity1
                        result.metadata['identity2'] = pair.identity2
                        result.metadata['dataset_name'] = pair.dataset_name
                        result.modality = pair.modality
                        
                        self.results.append(result)

                        if self.config.verbose and self.config.profile_timings:
                            timings = result.metadata.get('timings_ms', {})
                            if isinstance(timings, dict) and timings:
                                timing_text = ", ".join(
                                    f"{stage}={float(value):.2f} ms" if isinstance(value, (int, float)) else f"{stage}={value}"
                                    for stage, value in timings.items()
                                )
                                progress_bar.write(
                                    f"Timing {matcher_name} pair {pair.pair_id}: {timing_text}"
                                )
                
                except Exception as e:
                    error_dict = {
                        'type': 'match_execution',
                        'pair_id': pair.pair_id,
                        'matcher': matcher_name,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    }
                    self.errors.append(error_dict)
                    
                    if self.config.verbose:
                        progress_bar.write(
                            f"Warning: Error in pair {pair.pair_id} with {matcher_name}: {str(e)}"
                        )
        
        # Compute summary metrics
        if self.config.verbose:
            print(f"\nComputing metrics...")
        
        self._compute_metrics()
        
        # Save results
        if self.config.verbose:
            print(f"\nSaving results...")
        
        self._save_results()
        
        # Print summary metrics at the end
        if self.config.verbose:
            self._print_summary()
        
        return self.results, self.metrics
    
    def _compute_metrics(self):
        """Compute summary metrics by matcher."""
        # Group results by matcher
        by_matcher = {}
        for result in self.results:
            cls_name = result.method_name
            if cls_name not in by_matcher:
                by_matcher[cls_name] = []
            by_matcher[cls_name].append(result)
        
        # Compute metrics for each matcher
        for matcher_name, results in by_matcher.items():
            if not results:
                continue
            
            # Compute accuracy
            correct = sum(1 for r in results if r.is_same_person_pred == r.ground_truth)
            accuracy = correct / len(results) if results else 0.0
            
            # Compute by ground truth
            genuine = [r for r in results if r.ground_truth == True]
            impostor = [r for r in results if r.ground_truth == False]
            
            genuine_correct = sum(1 for r in genuine if r.is_same_person_pred == True)
            impostor_correct = sum(1 for r in impostor if r.is_same_person_pred == False)
            
            genuine_accuracy = genuine_correct / len(genuine) if genuine else 0.0
            impostor_accuracy = impostor_correct / len(impostor) if impostor else 0.0
            
            # Compute precision and recall
            true_positives = sum(1 for r in results if r.is_same_person_pred == True and r.ground_truth == True)
            false_positives = sum(1 for r in results if r.is_same_person_pred == True and r.ground_truth == False)
            false_negatives = sum(1 for r in results if r.is_same_person_pred == False and r.ground_truth == True)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            
            # Compute biometric verification metrics (at current threshold)
            # TAR (True Acceptance Rate) = % of genuine pairs correctly matched
            tar = genuine_correct / len(genuine) if genuine else 0.0
            # FAR (False Acceptance Rate) = % of impostor pairs incorrectly matched
            far = (len(impostor) - impostor_correct) / len(impostor) if impostor else 0.0
            # FRR (False Rejection Rate) = % of genuine pairs incorrectly rejected
            frr = (len(genuine) - genuine_correct) / len(genuine) if genuine else 0.0
            # TRR (True Rejection Rate) = % of impostor pairs correctly rejected
            trr = impostor_correct / len(impostor) if impostor else 0.0

            timing_summary = self._summarize_timings(results)
            
            self.metrics[matcher_name] = {
                'num_pairs': len(results),
                'num_genuine': len(genuine),
                'num_impostor': len(impostor),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'genuine_accuracy': genuine_accuracy,
                'impostor_accuracy': impostor_accuracy,
                'tar': tar,
                'far': far,
                'frr': frr,
                'trr': trr,
                'avg_inlier_ratio': sum(r.verification_confidence for r in results) / len(results) if results else 0.0,
                'timings_ms': timing_summary,
            }

    @staticmethod
    def _summarize_timings(results: List[VerificationResult]) -> Dict[str, Dict[str, float]]:
        """Aggregate per-stage timing metadata across results."""
        by_stage: Dict[str, List[float]] = {}
        for result in results:
            timings = result.metadata.get('timings_ms')
            if not isinstance(timings, dict):
                continue
            for stage_name, stage_value in timings.items():
                try:
                    by_stage.setdefault(stage_name, []).append(float(stage_value))
                except (TypeError, ValueError):
                    continue

        summary: Dict[str, Dict[str, float]] = {}
        for stage_name, values in by_stage.items():
            if not values:
                continue
            mean_ms = mean(values)
            summary[stage_name] = {
                'mean_ms': mean_ms,
                'mean_fps': (1000.0 / mean_ms) if mean_ms > 0 else 0.0,
                'median_ms': median(values),
                'min_ms': min(values),
                'max_ms': max(values),
            }
        return summary
    
    def _print_summary(self):
        """Print summary metrics in formatted style."""
        print("\nSummary Metrics:")
        for matcher_name, metrics in self.metrics.items():
            print(f"\n  {matcher_name}:")
            print(f"    Pairs: {metrics['num_pairs']} (Genuine: {metrics['num_genuine']}, Impostor: {metrics['num_impostor']})")
            print(f"    Accuracy: {metrics['accuracy']:.2%}")
            print(f"    Precision: {metrics['precision']:.2%}, Recall: {metrics['recall']:.2%}")
            print(f"    Genuine Accuracy: {metrics['genuine_accuracy']:.2%}, Impostor Accuracy: {metrics['impostor_accuracy']:.2%}")
            print(f"    TAR (True Acceptance Rate):  {metrics['tar']:.2%}")
            print(f"    FAR (False Acceptance Rate): {metrics['far']:.2%}")
            print(f"    FRR (False Rejection Rate):  {metrics['frr']:.2%}")
            print(f"    TRR (True Rejection Rate):   {metrics['trr']:.2%}")
            if metrics.get('timings_ms'):
                print("    Timing summary (ms):")
                stage_width = max(5, max(len(stage) for stage in metrics['timings_ms'].keys()))
                header = f"      {'Stage':<{stage_width}} | {'Mean ms':>10} | {'Mean FPS':>10} | {'Median':>10} | {'Min':>10} | {'Max':>10}"
                separator = f"      {'-' * stage_width}-+------------+------------+------------+------------+------------"
                print(header)
                print(separator)
                for stage_name, stage_stats in metrics['timings_ms'].items():
                    print(
                        f"      {stage_name:<{stage_width}} | "
                        f"{stage_stats['mean_ms']:>10.2f} | "
                        f"{stage_stats['mean_fps']:>10.2f} | "
                        f"{stage_stats['median_ms']:>10.2f} | "
                        f"{stage_stats['min_ms']:>10.2f} | "
                        f"{stage_stats['max_ms']:>10.2f}"
                    )
    
    def _save_results(self):
        """Save results to JSON, CSV, and summary JSON."""
        output_dir = Path(self.config.output_dir)
        
        # Save config as YAML
        config_yaml = output_dir / "config.yaml"
        with open(config_yaml, 'w') as f:
            f.write(self.config.to_yaml())
        
        # Collect unique matcher names
        unique_matchers = sorted(set(r.method_name for r in self.results))
        
        # Save full results as JSON
        results_json = output_dir / "results.json"
        results_data = {
            'experiment': asdict(self.config.experiment),
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'total_results': len(self.results),
                'total_errors': len(self.errors),
                'matchers': unique_matchers,
            },
            'results': [
                {
                    'method_name': r.method_name,
                    'modality': r.modality,
                    'is_same_person_pred': r.is_same_person_pred,
                    'verification_confidence': float(r.verification_confidence),
                    'ground_truth': r.ground_truth,
                    'is_correct': r.is_correct,
                    'num_keypoints_image1': r.num_keypoints_image1,
                    'num_keypoints_image2': r.num_keypoints_image2,
                    'num_matches': r.num_matches,
                    'num_inliers': r.num_inliers,
                    'inlier_ratio': r.inlier_ratio,
                    'reprojection_error': r.reprojection_error,
                    'homography_confidence': r.homography_confidence,
                    'matcher_params': r.matcher_params,
                    'timings_ms': r.metadata.get('timings_ms'),
                    'metadata': r.metadata,
                    'timestamp': r.timestamp,
                }
                for r in self.results
            ],
        }
        
        with open(results_json, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save results as CSV
        results_csv = output_dir / "results.csv"
        with open(results_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'matcher', 'modality', 'pair_id', 'identity1', 'identity2',
                'num_keypoints_image1', 'num_keypoints_image2',
                'num_matches', 'num_inliers', 'inlier_ratio',
                'is_same_person_pred', 'confidence', 'ground_truth', 'is_correct',
                'reprojection_error', 'timings_ms'
            ])
            
            for r in self.results:
                writer.writerow([
                    r.method_name,
                    r.modality,
                    r.metadata.get('pair_id', ''),
                    r.metadata.get('identity1', ''),
                    r.metadata.get('identity2', ''),
                    r.num_keypoints_image1,
                    r.num_keypoints_image2,
                    r.num_matches,
                    r.num_inliers,
                    f"{r.inlier_ratio:.4f}" if r.inlier_ratio else '',
                    r.is_same_person_pred,
                    f"{r.verification_confidence:.4f}" if r.verification_confidence else '',
                    r.ground_truth,
                    r.is_correct,
                    f"{r.reprojection_error:.4f}" if r.reprojection_error is not None else '',
                    json.dumps(r.metadata.get('timings_ms', {}), ensure_ascii=True),
                ])
        
        # Save summary metrics
        summary_json = output_dir / "summary.json"
        with open(summary_json, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save errors if any
        if self.errors:
            errors_json = output_dir / "errors.json"
            with open(errors_json, 'w') as f:
                json.dump(self.errors, f, indent=2)
        
        if self.config.verbose:
            print(f"Results saved to {output_dir}")


def run_experiment(config_path: str) -> Tuple[List[VerificationResult], Dict]:
    """Run a batch experiment from a YAML config file.
    
    Convenience function for CLI and other entry points.
    
    Args:
        config_path: Path to experiment YAML config file
        
    Returns:
        Tuple of (results_list, summary_metrics)
        
    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config format invalid
    """
    # Load experiment config
    config = ExperimentConfig.from_yaml(config_path)
    
    # Create and run runner
    runner = BatchExperimentRunner(config)
    results, metrics = runner.run()
    
    return results, metrics

