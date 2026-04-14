"""
Closed-set identification experiment runner.
"""

import csv
import json
import traceback
import yaml
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .config import IdentificationExperimentConfig, MatcherExperimentConfig
from ..data.identification import IdentificationDataset, IdentificationSample
from ..evaluation.metrics import (
    compute_cmc_curve,
    compute_mean_average_precision,
    compute_recall_at_k,
    compute_rank_k_accuracy,
)
from ..evaluation.plotter import plot_cmc_curve
from ..matchers.registry import create_matcher
from ..results import IdentificationResult

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, total=None, disable=False):
        return iterable


class IdentificationExperimentRunner:
    """Batch runner for closed-set identification experiments."""

    def __init__(self, experiment_config: IdentificationExperimentConfig):
        self.config = experiment_config
        self.results: List[IdentificationResult] = []
        self.metrics: Dict = {}
        self.errors: List[Dict] = []

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        if self.config.verbose:
            print(f"✓ Identification experiment: {self.config.experiment.name}")
            print(f"✓ Output dir: {self.config.output_dir}")
            print(f"✓ Identification CSV: {self.config.identification_dataset}")
            print(f"✓ Ranking strategy: {self.config.ranking_strategy}")
            if self.config.ranking_strategy == "cascade":
                print(
                    f"  - Shortlist matcher: {self.config.shortlist_matcher.name} (K={self.config.shortlist_k})"
                )
            print(f"✓ Samples per gallery: {self.config.samples_per_gallery} ({self.config.aggregation_method})")
            print(f"✓ Matchers: {[m.name for m in self.config.matchers]}")

    def _load_matcher_config(self, matcher_cfg: MatcherExperimentConfig) -> dict:
        config_dict = {}

        if matcher_cfg.config_base:
            base_path = Path(matcher_cfg.config_base)
            if not base_path.exists():
                raise FileNotFoundError(f"Base config not found: {matcher_cfg.config_base}")

            with open(base_path, 'r') as f:
                yaml_data = yaml.safe_load(f)

            if 'matcher' in yaml_data and 'config' in yaml_data['matcher']:
                config_dict = yaml_data['matcher']['config'].copy()

        if matcher_cfg.config_overrides:
            config_dict.update(matcher_cfg.config_overrides)

        if 'device' not in config_dict:
            config_dict['device'] = self.config.device

        return config_dict

    def _aggregate_scores(self, scores: List[float]) -> float:
        if not scores:
            return 0.0
        if self.config.aggregation_method == 'mean':
            return float(np.mean(scores))
        return float(np.max(scores))

    def _prepare_gallery_templates(
        self,
        matcher,
        gallery_by_identity: Dict[str, List[IdentificationSample]],
    ) -> Dict[str, List[Dict]]:
        """Prepare cached gallery templates once per matcher."""
        gallery_templates: Dict[str, List[Dict]] = {}
        for identity, gallery_samples in gallery_by_identity.items():
            gallery_templates[identity] = [
                matcher.prepare_identification_template(
                    sample.image_path,
                    modality=sample.modality,
                )
                for sample in gallery_samples
            ]
        return gallery_templates

    def _compute_probe_shortlist(
        self,
        probe: IdentificationSample,
        shortlist_matcher,
        shortlist_matcher_name: str,
        shortlist_gallery_templates_by_identity: Dict[str, List[Dict]],
    ) -> Tuple[Set[str], Optional[int]]:
        """Compute top-K candidate identities for one probe using the shortlist matcher."""
        shortlist_scores: Dict[str, float] = {}
        shortlist_probe_template = shortlist_matcher.prepare_identification_template(
            probe.image_path,
            modality=probe.modality,
        )

        for identity, gallery_templates in shortlist_gallery_templates_by_identity.items():
            try:
                # Shortlist stage intentionally uses one gallery sample per identity for speed.
                gallery_template = gallery_templates[0]
                result = shortlist_matcher.compare_identification_templates(
                    shortlist_probe_template,
                    gallery_template,
                    ground_truth=None,
                    matcher_name=shortlist_matcher_name,
                )
                if result is not None:
                    shortlist_scores[identity] = float(result.verification_confidence)
                else:
                    shortlist_scores[identity] = 0.0
            except Exception as e:
                self.errors.append(
                    {
                        'type': 'identification_shortlist_execution',
                        'probe_record_id': probe.record_id,
                        'gallery_identity': identity,
                        'shortlist_matcher': shortlist_matcher_name,
                        'error': str(e),
                        'traceback': traceback.format_exc(),
                    }
                )
                shortlist_scores[identity] = 0.0

        ranked_shortlist = sorted(shortlist_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_identities = {
            identity for identity, _ in ranked_shortlist[: self.config.shortlist_k]
        }

        shortlist_rank_of_true_identity = None
        for idx, (identity, _score) in enumerate(ranked_shortlist, start=1):
            if identity == probe.identity:
                shortlist_rank_of_true_identity = idx
                break

        return top_k_identities, shortlist_rank_of_true_identity

    def _rank_probe(
        self,
        probe: IdentificationSample,
        gallery_templates_by_identity: Dict[str, List[Dict]],
        matcher,
        matcher_name: str,
    ) -> IdentificationResult:
        probe_template = matcher.prepare_identification_template(
            probe.image_path,
            modality=probe.modality,
        )

        scores_by_identity: Dict[str, float] = {}

        for identity, gallery_templates in gallery_templates_by_identity.items():
            try:
                if self.config.samples_per_gallery == 'single':
                    candidate_templates = [gallery_templates[0]]
                else:
                    candidate_templates = gallery_templates

                identity_scores: List[float] = []
                for gallery_template in candidate_templates:
                    result = matcher.compare_identification_templates(
                        probe_template,
                        gallery_template,
                        ground_truth=None,
                        matcher_name=matcher_name,
                    )
                    if result is not None:
                        identity_scores.append(float(result.verification_confidence))

                if identity_scores:
                    scores_by_identity[identity] = self._aggregate_scores(identity_scores)
                else:
                    scores_by_identity[identity] = 0.0
            except Exception as e:
                self.errors.append(
                    {
                        'type': 'identification_match_execution',
                        'probe_record_id': probe.record_id,
                        'gallery_identity': identity,
                        'matcher': matcher_name,
                        'error': str(e),
                        'traceback': traceback.format_exc(),
                    }
                )
                scores_by_identity[identity] = 0.0

        ranked_identities = sorted(scores_by_identity.items(), key=lambda x: x[1], reverse=True)

        rank_of_true_identity = None
        for idx, (identity, _score) in enumerate(ranked_identities, start=1):
            if identity == probe.identity:
                rank_of_true_identity = idx
                break

        return IdentificationResult(
            method_name=matcher_name,
            probe_record_id=probe.record_id,
            probe_sample_id=probe.sample_id,
            probe_identity=probe.identity,
            modality=probe.modality,
            ranked_identities=ranked_identities,
            rank_of_true_identity=rank_of_true_identity,
            gallery_size=len(gallery_templates_by_identity),
            ranking_strategy=self.config.ranking_strategy,
            samples_per_gallery=self.config.samples_per_gallery,
            aggregation_method=self.config.aggregation_method,
            metadata={
                'dataset_name': probe.dataset_name,
                'probe_metadata': probe.metadata,
            },
        )

    def _rank_probe_cascade(
        self,
        probe: IdentificationSample,
        gallery_templates_by_identity: Dict[str, List[Dict]],
        shortlist_matcher_name: str,
        top_k_identities: Set[str],
        shortlist_rank_of_true_identity: Optional[int],
        main_matcher,
        matcher_name: str,
    ) -> IdentificationResult:
        """Two-stage cascade ranking: rerank shortlist candidates with the main matcher."""
        probe_template = main_matcher.prepare_identification_template(
            probe.image_path,
            modality=probe.modality,
        )

        # Stage 2: Detailed ranking with main matcher on shortlisted identities
        scores_by_identity: Dict[str, float] = {}

        for identity, gallery_templates in gallery_templates_by_identity.items():
            # Skip identities not in shortlist
            if identity not in top_k_identities:
                scores_by_identity[identity] = 0.0
                continue

            try:
                # Use samples_per_gallery setting for main matcher ranking
                if self.config.samples_per_gallery == 'single':
                    candidate_templates = [gallery_templates[0]]
                else:
                    candidate_templates = gallery_templates

                identity_scores: List[float] = []
                for gallery_template in candidate_templates:
                    result = main_matcher.compare_identification_templates(
                        probe_template,
                        gallery_template,
                        ground_truth=None,
                        matcher_name=matcher_name,
                    )
                    if result is not None:
                        identity_scores.append(float(result.verification_confidence))

                if identity_scores:
                    scores_by_identity[identity] = self._aggregate_scores(identity_scores)
                else:
                    scores_by_identity[identity] = 0.0
            except Exception as e:
                self.errors.append(
                    {
                        'type': 'identification_match_execution',
                        'probe_record_id': probe.record_id,
                        'gallery_identity': identity,
                        'matcher': matcher_name,
                        'error': str(e),
                        'traceback': traceback.format_exc(),
                    }
                )
                scores_by_identity[identity] = 0.0

        ranked_identities = sorted(scores_by_identity.items(), key=lambda x: x[1], reverse=True)

        rank_of_true_identity = None
        for idx, (identity, _score) in enumerate(ranked_identities, start=1):
            if identity == probe.identity:
                rank_of_true_identity = idx
                break

        return IdentificationResult(
            method_name=matcher_name,
            probe_record_id=probe.record_id,
            probe_sample_id=probe.sample_id,
            probe_identity=probe.identity,
            modality=probe.modality,
            ranked_identities=ranked_identities,
            rank_of_true_identity=rank_of_true_identity,
            gallery_size=len(gallery_templates_by_identity),
            ranking_strategy=self.config.ranking_strategy,
            samples_per_gallery=self.config.samples_per_gallery,
            aggregation_method=self.config.aggregation_method,
            metadata={
                'dataset_name': probe.dataset_name,
                'probe_metadata': probe.metadata,
                'shortlist_matcher': shortlist_matcher_name,
                'shortlist_k': self.config.shortlist_k,
                'shortlist_rank_of_true_identity': shortlist_rank_of_true_identity,
                'shortlist_hit_at_k': (
                    shortlist_rank_of_true_identity is not None
                    and shortlist_rank_of_true_identity <= self.config.shortlist_k
                ),
            },
        )

    def _compute_metrics(self):
        by_matcher: Dict[str, List[IdentificationResult]] = {}
        for result in self.results:
            by_matcher.setdefault(result.method_name, []).append(result)

        for matcher_name, matcher_results in by_matcher.items():
            ranks = [r.rank_of_true_identity for r in matcher_results]
            valid_ranks = [r for r in ranks if r is not None]

            max_rank = max(self.config.top_k_ranks) if self.config.top_k_ranks else 1
            cmc = compute_cmc_curve(ranks, max_rank=max_rank)

            rank_k = {
                str(k): compute_rank_k_accuracy(ranks, int(k))
                for k in self.config.top_k_ranks
            }

            recall_k = {
                str(k): compute_recall_at_k(ranks, int(k))
                for k in self.config.top_k_ranks
            }

            self.metrics[matcher_name] = {
                'num_probes': len(matcher_results),
                'num_valid_ranks': len(valid_ranks),
                'rank_1_accuracy': compute_rank_k_accuracy(ranks, 1),
                'rank_k_accuracy': rank_k,
                'recall_at_k': recall_k,
                'cmc': cmc,
                'mean_average_precision': compute_mean_average_precision(ranks),
                'mean_rank': float(np.mean(valid_ranks)) if valid_ranks else None,
                'median_rank': float(np.median(valid_ranks)) if valid_ranks else None,
                'std_rank': float(np.std(valid_ranks)) if valid_ranks else None,
                'cache_gallery_templates': self.config.cache_gallery_templates,
            }

    def _save_results(self):
        output_dir = Path(self.config.output_dir)

        config_yaml = output_dir / 'config.yaml'
        with open(config_yaml, 'w') as f:
            f.write(self.config.to_yaml())

        results_json = output_dir / 'identification_results.json'
        results_data = {
            'experiment': asdict(self.config.experiment),
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'total_results': len(self.results),
                'total_errors': len(self.errors),
            },
            'results': [
                {
                    'method_name': r.method_name,
                    'probe_record_id': r.probe_record_id,
                    'probe_sample_id': r.probe_sample_id,
                    'probe_identity': r.probe_identity,
                    'modality': r.modality,
                    'rank_of_true_identity': r.rank_of_true_identity,
                    'gallery_size': r.gallery_size,
                    'ranking_strategy': r.ranking_strategy,
                    'samples_per_gallery': r.samples_per_gallery,
                    'aggregation_method': r.aggregation_method,
                    'ranked_identities': r.ranked_identities,
                    'metadata': r.metadata,
                    'timestamp': r.timestamp,
                }
                for r in self.results
            ],
        }
        with open(results_json, 'w') as f:
            json.dump(results_data, f, indent=2)

        results_csv = output_dir / 'identification_results.csv'
        with open(results_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    'matcher',
                    'probe_record_id',
                    'probe_sample_id',
                    'probe_identity',
                    'rank_of_true_identity',
                    'rank_1_hit',
                    'gallery_size',
                    'ranking_strategy',
                    'samples_per_gallery',
                    'aggregation_method',
                    'top_5_identities',
                ]
            )

            for r in self.results:
                top5 = [identity for identity, _score in r.ranked_identities[:5]]
                writer.writerow(
                    [
                        r.method_name,
                        r.probe_record_id,
                        r.probe_sample_id,
                        r.probe_identity,
                        r.rank_of_true_identity,
                        r.is_rank_1_hit(),
                        r.gallery_size,
                        r.ranking_strategy,
                        r.samples_per_gallery,
                        r.aggregation_method,
                        ';'.join(top5),
                    ]
                )

        summary_json = output_dir / 'summary.json'
        with open(summary_json, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        cmc_plot_path = output_dir / 'cmc_curve.png'
        plot_cmc_curve(self.metrics, output_path=cmc_plot_path, show=False)

        if self.errors:
            errors_json = output_dir / 'errors.json'
            with open(errors_json, 'w') as f:
                json.dump(self.errors, f, indent=2)

        if self.config.verbose:
            print(f"✓ Identification results saved to {output_dir}")

    def _print_summary(self):
        print("\n📈 Identification Summary Metrics:")
        for matcher_name, metrics in self.metrics.items():
            print(f"\n  {matcher_name}:")
            print(f"    Probes: {metrics['num_probes']}")
            print(f"    Rank-1: {metrics['rank_1_accuracy']:.2%}")
            for k, value in metrics['rank_k_accuracy'].items():
                print(f"    Rank-{k}: {value:.2%}")
            for k, value in metrics.get('recall_at_k', {}).items():
                print(f"    Recall@{k}: {value:.2%}")
            print(f"    mAP: {metrics['mean_average_precision']:.4f}")

    def run(self) -> Tuple[List[IdentificationResult], Dict]:
        if self.config.verbose:
            print("\n📂 Loading identification dataset...")

        dataset = IdentificationDataset(
            csv_path=self.config.identification_dataset,
            base_path=self.config.base_path,
            filter_modality=self.config.filter_modality,
            filter_dataset=self.config.filter_dataset,
        )

        stats = dataset.get_statistics()
        if self.config.verbose:
            print(
                f"   Gallery: {stats['gallery_samples']} samples / {stats['gallery_identities']} identities"
            )
            print(
                f"   Probe: {stats['probe_samples']} samples / {stats['probe_identities']} identities"
            )

        if not dataset.validate_closed_set():
            raise ValueError(
                "Identification dataset is not closed-set: probe identities missing in gallery"
            )

        probes = dataset.get_probes()
        gallery = dataset.get_gallery()

        if not probes:
            print("⚠ No probes to process!")
            return [], {}

        if self.config.verbose:
            print("\n🔧 Instantiating matchers...")

        matchers: Dict[str, object] = {}
        for matcher_cfg in self.config.matchers:
            try:
                matcher_config = self._load_matcher_config(matcher_cfg)
                matcher = create_matcher(matcher_cfg.name, matcher_config)
                matchers[matcher_cfg.name] = matcher
                if self.config.verbose:
                    print(f"   ✓ {matcher_cfg.name}")
            except Exception as e:
                self.errors.append(
                    {
                        'type': 'matcher_initialization',
                        'matcher': matcher_cfg.name,
                        'error': str(e),
                        'traceback': traceback.format_exc(),
                    }
                )
                if self.config.verbose:
                    print(f"   ✗ Failed {matcher_cfg.name}: {e}")

        if not matchers:
            raise RuntimeError("No matchers successfully instantiated!")

        # Instantiate shortlist matcher if using cascade strategy
        shortlist_matcher = None
        shortlist_matcher_name = None
        if self.config.ranking_strategy == "cascade":
            shortlist_matcher_cfg = self.config.shortlist_matcher
            shortlist_matcher_name = shortlist_matcher_cfg.name
            if self.config.verbose:
                print(f"\n🔧 Instantiating shortlist matcher: {shortlist_matcher_name}...")
            try:
                shortlist_matcher_config = self._load_matcher_config(shortlist_matcher_cfg)
                shortlist_matcher = create_matcher(shortlist_matcher_name, shortlist_matcher_config)
                if self.config.verbose:
                    print(f"   ✓ {shortlist_matcher_name}")
            except Exception as e:
                self.errors.append(
                    {
                        'type': 'shortlist_matcher_initialization',
                        'matcher': shortlist_matcher_name,
                        'error': str(e),
                        'traceback': traceback.format_exc(),
                    }
                )
                if self.config.verbose:
                    print(f"   ✗ Failed {shortlist_matcher_name}: {e}")
                raise RuntimeError(f"Failed to instantiate shortlist matcher: {e}")

        gallery_templates_by_matcher: Dict[str, Dict[str, List[Dict]]] = {}
        shortlist_gallery_templates: Dict[str, List[Dict]] = None

        if self.config.cache_gallery_templates:
            if self.config.verbose:
                print("\n🗃️  Preparing cached gallery templates...")
            for matcher_name, matcher in matchers.items():
                gallery_templates_by_matcher[matcher_name] = self._prepare_gallery_templates(matcher, gallery)
                if self.config.verbose:
                    total_templates = sum(len(v) for v in gallery_templates_by_matcher[matcher_name].values())
                    print(f"   ✓ {matcher_name}: cached {total_templates} gallery templates")

            # Cache shortlist matcher gallery templates if cascade strategy
            if self.config.ranking_strategy == "cascade" and shortlist_matcher:
                if self.config.verbose:
                    print(f"   Preparing shortlist matcher templates...")
                shortlist_gallery_templates = self._prepare_gallery_templates(shortlist_matcher, gallery)
                if self.config.verbose:
                    total_templates = sum(len(v) for v in shortlist_gallery_templates.values())
                    print(f"   ✓ {shortlist_matcher_name}: cached {total_templates} gallery templates")

        progress_bar = tqdm(
            probes,
            desc='Probes',
            total=len(probes),
            disable=not self.config.verbose,
        )

        for probe in progress_bar:
            probe_top_k_identities: Optional[Set[str]] = None
            probe_shortlist_rank: Optional[int] = None

            if self.config.ranking_strategy == "cascade":
                if shortlist_matcher is None:
                    raise RuntimeError("Cascade strategy requires a shortlist matcher")

                if shortlist_gallery_templates is None:
                    shortlist_gallery_templates = self._prepare_gallery_templates(shortlist_matcher, gallery)

                probe_top_k_identities, probe_shortlist_rank = self._compute_probe_shortlist(
                    probe=probe,
                    shortlist_matcher=shortlist_matcher,
                    shortlist_matcher_name=shortlist_matcher_name or "shortlist",
                    shortlist_gallery_templates_by_identity=shortlist_gallery_templates,
                )

            for matcher_name, matcher in matchers.items():
                if self.config.cache_gallery_templates:
                    gallery_templates = gallery_templates_by_matcher[matcher_name]
                else:
                    gallery_templates = self._prepare_gallery_templates(matcher, gallery)

                if self.config.ranking_strategy == "cascade":
                    result = self._rank_probe_cascade(
                        probe=probe,
                        gallery_templates_by_identity=gallery_templates,
                        shortlist_matcher_name=shortlist_matcher_name or "shortlist",
                        top_k_identities=probe_top_k_identities or set(),
                        shortlist_rank_of_true_identity=probe_shortlist_rank,
                        main_matcher=matcher,
                        matcher_name=matcher_name,
                    )
                else:
                    result = self._rank_probe(
                        probe=probe,
                        gallery_templates_by_identity=gallery_templates,
                        matcher=matcher,
                        matcher_name=matcher_name,
                    )
                self.results.append(result)

        if self.config.verbose:
            print("\n📊 Computing identification metrics...")
        self._compute_metrics()

        if self.config.verbose:
            print("\n💾 Saving identification results...")
        self._save_results()

        if self.config.verbose:
            self._print_summary()

        return self.results, self.metrics


def run_identification_experiment(config_path: str) -> Tuple[List[IdentificationResult], Dict]:
    """Run closed-set identification experiment from YAML config."""
    config = IdentificationExperimentConfig.from_yaml(config_path)
    runner = IdentificationExperimentRunner(config)
    return runner.run()
