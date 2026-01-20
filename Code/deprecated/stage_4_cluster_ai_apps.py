#!/usr/bin/env python3
"""
Stage 4 Extension: Cluster AI Applications using UMAP + HDBSCAN

This module clusters deduplicated AI applications from Stage 4 using:
1. UMAP: Reduce 1024-D BGE embeddings to 50-D
2. HDBSCAN: Conservative clustering (min_cluster_size=5)
3. GPT: Generate cluster labels and quality assessments
4. Analysis: Quality metrics, visualization, noise analysis

Outputs:
- cluster_assignments.parquet: App-to-cluster mapping
- cluster_centroids.pkl: Cluster centroid embeddings
- cluster_labels.json: GPT-generated labels
- cluster_quality_report.json: Quality metrics and statistics
- cluster_visualization.html: Interactive UMAP plot
- cluster_noise_analysis.json: Noise point analysis

Usage:
    python stage_4_cluster_ai_apps.py
    python stage_4_cluster_ai_apps.py --umap-dims 50 --cluster-min-size 10
    python stage_4_cluster_ai_apps.py --skip-cluster-labeling --generate-visualization
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import Counter
import gc

# Required libraries for clustering and visualization
try:
    import umap
    import hdbscan
    from sklearn.metrics import silhouette_score, silhouette_samples
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError as e:
    print(f"ERROR: Missing required library: {e}")
    print("Install with: pip install umap-learn hdbscan scikit-learn plotly")
    exit(1)

# OpenAI setup (from Stage 3 pattern)
try:
    import openai
    from openai import OpenAI
    from dotenv import load_dotenv
except ImportError:
    print("ERROR: openai or python-dotenv not installed")
    print("Install with: pip install openai python-dotenv")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AIApplicationsClusterer:
    """
    Main class for clustering deduplicated AI applications.

    Pipeline:
    1. Load embeddings and dedup apps from Stage 4 outputs
    2. Apply UMAP dimensionality reduction (1024-D → 50-D)
    3. Apply HDBSCAN clustering (min_cluster_size=5)
    4. Compute cluster centroids in both spaces
    5. Analyze noise points
    6. Generate GPT labels for each cluster
    7. Compute quality metrics (silhouette, size distribution, etc.)
    8. Generate visualization (interactive UMAP plot)
    """

    def __init__(self,
                 embeddings_dir: str = "Data/embeddings",
                 data_dir: str = "Data",
                 output_dir: str = "Data/clustering",
                 umap_dims: int = 50,
                 umap_neighbors: int = 15,
                 umap_min_dist: float = 0.1,
                 cluster_min_size: int = 5,
                 cluster_min_samples: int = 5,
                 cluster_metric: str = "euclidean",
                 label_model: str = "gpt-4.1-mini",
                 label_sample_size: int = 15,
                 skip_labeling: bool = False,
                 generate_viz: bool = False):
        """
        Initialize the clusterer.

        Args:
            embeddings_dir: Directory containing embeddings cache
            data_dir: Directory containing dedup_apps.parquet
            output_dir: Where to save cluster outputs
            umap_dims: Target dimensionality for UMAP (default: 50)
            umap_neighbors: UMAP n_neighbors parameter (default: 15)
            umap_min_dist: UMAP min_dist parameter (default: 0.1)
            cluster_min_size: HDBSCAN min_cluster_size (default: 5)
            cluster_min_samples: HDBSCAN min_samples (default: 5)
            cluster_metric: HDBSCAN distance metric (default: euclidean)
            label_model: GPT model for labeling (default: gpt-4.1-mini)
            label_sample_size: Apps per cluster to show GPT (default: 15)
            skip_labeling: Skip GPT labeling if True
            generate_viz: Generate visualization if True
        """
        self.embeddings_dir = embeddings_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.umap_dims = umap_dims
        self.umap_neighbors = umap_neighbors
        self.umap_min_dist = umap_min_dist
        self.cluster_min_size = cluster_min_size
        self.cluster_min_samples = cluster_min_samples
        self.cluster_metric = cluster_metric
        self.label_model = label_model
        self.label_sample_size = label_sample_size
        self.skip_labeling = skip_labeling
        self.generate_viz = generate_viz

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize OpenAI client
        load_dotenv('config.env')
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        logger.info("AIApplicationsClusterer initialized")
        logger.info(f"  Embeddings dir: {embeddings_dir}")
        logger.info(f"  Data dir: {data_dir}")
        logger.info(f"  Output dir: {output_dir}")
        logger.info(f"  UMAP dims: {umap_dims}")
        logger.info(f"  HDBSCAN min_cluster_size: {cluster_min_size}")
        logger.info(f"  Skip labeling: {skip_labeling}")

    def find_embeddings_cache(self) -> Optional[str]:
        """
        Auto-detect most recent embeddings cache file.

        Returns:
            Path to embeddings cache, or None if not found
        """
        cache_dir = Path(self.embeddings_dir)
        if not cache_dir.exists():
            logger.error(f"Embeddings directory not found: {self.embeddings_dir}")
            return None

        # Find all apps_*.pkl files
        cache_files = list(cache_dir.glob("apps_*.pkl"))
        if not cache_files:
            logger.error("No embeddings cache files found (pattern: apps_*.pkl)")
            return None

        # Use most recent by modification time
        latest = max(cache_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Found embeddings cache: {latest.name}")
        return str(latest)

    def load_embeddings_and_apps(self,
                                 embeddings_file: Optional[str] = None,
                                 dedup_apps_file: Optional[str] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Phase 1: Load and validate embeddings cache and deduplicated applications.

        Args:
            embeddings_file: Path to embeddings pickle (auto-detect if None)
            dedup_apps_file: Path to dedup_apps.parquet (auto-detect if None)

        Returns:
            Tuple of (embeddings_array [N x D], dedup_apps_df)
        """
        logger.info("=" * 80)
        logger.info("PHASE 1: Loading and validating data")
        logger.info("=" * 80)

        # Auto-detect embeddings file
        if embeddings_file is None:
            embeddings_file = self.find_embeddings_cache()
            if embeddings_file is None:
                raise FileNotFoundError("Could not find embeddings cache")

        if not os.path.exists(embeddings_file):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

        # Auto-detect dedup_apps file
        if dedup_apps_file is None:
            dedup_apps_file = os.path.join(self.data_dir, "dedup_apps.parquet")

        if not os.path.exists(dedup_apps_file):
            raise FileNotFoundError(f"Dedup apps file not found: {dedup_apps_file}")

        # Load embeddings cache
        logger.info(f"Loading embeddings from: {embeddings_file}")
        with open(embeddings_file, 'rb') as f:
            cache_data = pickle.load(f)

        # Handle two formats: BGE (array + texts list) and OpenAI (dict + metadata)
        if isinstance(cache_data['embeddings'], dict):
            # OpenAI format: embeddings is a dict with text as key, embedding as value
            logger.info("Detected OpenAI embeddings format (dict-based)")
            cache_texts = list(cache_data['embeddings'].keys())
            embeddings = np.array([cache_data['embeddings'][text] for text in cache_texts])
            if 'metadata' in cache_data:
                logger.info(f"  Model: {cache_data['metadata'].get('model', 'unknown')}")
                logger.info(f"  Dimensions: {cache_data['metadata'].get('dimensions', 'unknown')}")
        else:
            # BGE format: embeddings is an array, texts is a separate list
            logger.info("Detected BGE embeddings format (array-based)")
            embeddings = cache_data['embeddings']
            cache_texts = cache_data['texts']

        logger.info(f"Loaded embeddings shape: {embeddings.shape}")
        logger.info(f"  Texts in cache: {len(cache_texts)}")

        # Load dedup apps
        logger.info(f"Loading dedup apps from: {dedup_apps_file}")
        dedup_apps_df = pd.read_parquet(dedup_apps_file)
        logger.info(f"Loaded {len(dedup_apps_df)} dedup app rows")
        logger.info(f"  Unique apps: {dedup_apps_df['app_text'].nunique()}")

        # Validate alignment
        logger.info("Validating embeddings-apps alignment...")
        if embeddings.shape[0] != len(cache_texts):
            raise ValueError(
                f"Embeddings shape mismatch: {embeddings.shape[0]} vs {len(cache_texts)} texts"
            )

        # Note: embeddings correspond to unique app texts, but dedup_apps_df may have duplicates
        # due to different job_uids. We'll work with unique apps.
        unique_apps = dedup_apps_df['app_text'].unique().tolist()
        if len(unique_apps) != len(cache_texts):
            logger.warning(
                f"Unique apps in dedup file ({len(unique_apps)}) != cache texts ({len(cache_texts)})"
            )
            logger.warning("Attempting to align by matching texts...")

            # Create mapping from cache texts
            cache_text_to_idx = {text: idx for idx, text in enumerate(cache_texts)}

            # Check how many unique apps we can match
            matched = sum(1 for text in unique_apps if text in cache_text_to_idx)
            logger.info(f"  Matched {matched}/{len(unique_apps)} unique apps")

            if matched < len(unique_apps) * 0.9:
                raise ValueError("Less than 90% of apps matched to embeddings")

        logger.info("✓ Data validation passed")
        return embeddings, dedup_apps_df

    def apply_umap(self, embeddings: np.ndarray) -> Tuple[np.ndarray, umap.UMAP]:
        """
        Phase 2: Apply UMAP dimensionality reduction.

        Args:
            embeddings: Original embeddings [N x 1024]

        Returns:
            Tuple of (reduced_embeddings [N x 50], umap_model)
        """
        logger.info("=" * 80)
        logger.info("PHASE 2: UMAP dimensionality reduction")
        logger.info("=" * 80)

        logger.info(f"Input shape: {embeddings.shape}")
        logger.info(f"Reducing to {self.umap_dims} dimensions...")
        logger.info(f"  n_neighbors: {self.umap_neighbors}")
        logger.info(f"  min_dist: {self.umap_min_dist}")
        logger.info(f"  metric: cosine")

        # Create and fit UMAP
        umap_model = umap.UMAP(
            n_components=self.umap_dims,
            n_neighbors=self.umap_neighbors,
            min_dist=self.umap_min_dist,
            metric='cosine',
            random_state=42,
            verbose=1
        )

        reduced_embeddings = umap_model.fit_transform(embeddings)
        logger.info(f"UMAP complete. Output shape: {reduced_embeddings.shape}")

        return reduced_embeddings, umap_model

    def apply_hdbscan(self, reduced_embeddings: np.ndarray) -> np.ndarray:
        """
        Phase 3: Apply HDBSCAN clustering.

        Args:
            reduced_embeddings: UMAP-reduced embeddings [N x 50]

        Returns:
            Cluster labels array [N] (-1 for noise)
        """
        logger.info("=" * 80)
        logger.info("PHASE 3: HDBSCAN clustering")
        logger.info("=" * 80)

        logger.info(f"Input shape: {reduced_embeddings.shape}")
        logger.info(f"  min_cluster_size: {self.cluster_min_size}")
        logger.info(f"  min_samples: {self.cluster_min_samples}")
        logger.info(f"  metric: {self.cluster_metric}")

        # Fit HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.cluster_min_size,
            min_samples=self.cluster_min_samples,
            metric=self.cluster_metric,
            cluster_selection_epsilon=0.0
        )

        cluster_labels = clusterer.fit_predict(reduced_embeddings)

        # Statistics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        noise_pct = (n_noise / len(cluster_labels)) * 100 if len(cluster_labels) > 0 else 0

        logger.info(f"✓ Clustering complete")
        logger.info(f"  Number of clusters: {n_clusters}")
        logger.info(f"  Noise points: {n_noise} ({noise_pct:.1f}%)")
        logger.info(f"  Cluster sizes: min={np.min(np.bincount(cluster_labels[cluster_labels >= 0]) if np.any(cluster_labels >= 0) else [0])}, "
                   f"max={np.max(np.bincount(cluster_labels[cluster_labels >= 0]) if np.any(cluster_labels >= 0) else [0])}, "
                   f"mean={np.mean(np.bincount(cluster_labels[cluster_labels >= 0]) if np.any(cluster_labels >= 0) else [0]):.1f}")

        return cluster_labels

    def compute_centroids(self,
                         embeddings: np.ndarray,
                         reduced_embeddings: np.ndarray,
                         cluster_labels: np.ndarray) -> Dict[int, Dict]:
        """
        Phase 4: Compute cluster centroids in both embedding spaces.

        Args:
            embeddings: Original embeddings [N x 1024]
            reduced_embeddings: UMAP embeddings [N x 50]
            cluster_labels: Cluster labels [N]

        Returns:
            Dict mapping cluster_id to centroid data
        """
        logger.info("=" * 80)
        logger.info("PHASE 4: Computing cluster centroids")
        logger.info("=" * 80)

        centroids = {}

        # Get unique clusters (excluding noise=-1)
        unique_clusters = sorted(set(cluster_labels))
        if -1 in unique_clusters:
            unique_clusters.remove(-1)

        logger.info(f"Computing centroids for {len(unique_clusters)} clusters...")

        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            cluster_size = np.sum(mask)

            # Compute centroids in both spaces
            centroid_original = np.mean(embeddings[mask], axis=0)
            centroid_umap = np.mean(reduced_embeddings[mask], axis=0)

            # L2 normalize
            centroid_original = centroid_original / (np.linalg.norm(centroid_original) + 1e-8)
            centroid_umap = centroid_umap / (np.linalg.norm(centroid_umap) + 1e-8)

            centroids[cluster_id] = {
                'centroid_original': centroid_original,
                'centroid_umap': centroid_umap,
                'size': int(cluster_size),
                'label': None,  # Will be filled by GPT labeling
                'quality_score': None  # Will be filled by GPT labeling
            }

        logger.info(f"✓ Computed {len(centroids)} centroids")
        return centroids

    def analyze_noise(self, dedup_apps_df: pd.DataFrame, cluster_labels: np.ndarray) -> Dict:
        """
        Phase 5: Analyze noise points.

        Args:
            dedup_apps_df: Deduplicated apps dataframe
            cluster_labels: Cluster labels [N]

        Returns:
            Noise analysis dictionary
        """
        logger.info("=" * 80)
        logger.info("PHASE 5: Analyzing noise points")
        logger.info("=" * 80)

        # Extract noise points
        unique_apps_list = dedup_apps_df['app_text'].unique().tolist()
        noise_mask = cluster_labels == -1
        noise_indices = np.where(noise_mask)[0]

        if len(noise_indices) == 0:
            logger.info("No noise points found")
            return {
                'noise_summary': {'count': 0, 'percentage': 0.0, 'avg_text_length': 0.0},
                'sample_noise_apps': [],
                'noise_characteristics': {}
            }

        noise_apps = [unique_apps_list[i] for i in noise_indices if i < len(unique_apps_list)]
        noise_pct = (len(noise_apps) / len(unique_apps_list)) * 100

        logger.info(f"Found {len(noise_apps)} noise points ({noise_pct:.1f}%)")

        # Analyze characteristics
        text_lengths = [len(app) for app in noise_apps]
        very_short = sum(1 for length in text_lengths if length < 10)
        very_long = sum(1 for length in text_lengths if length > 500)

        # Sample noise apps
        sample_size = min(20, len(noise_apps))
        sample_indices = np.random.choice(len(noise_apps), sample_size, replace=False)
        sample_apps = [noise_apps[i] for i in sorted(sample_indices)]

        analysis = {
            'noise_summary': {
                'count': len(noise_apps),
                'percentage': noise_pct,
                'avg_text_length': float(np.mean(text_lengths)) if text_lengths else 0.0
            },
            'sample_noise_apps': sample_apps,
            'noise_characteristics': {
                'very_short': very_short,
                'very_long': very_long,
                'potential_extraction_errors': sum(1 for app in noise_apps if len(app) < 5)
            }
        }

        logger.info(f"✓ Noise analysis complete")
        logger.info(f"  Avg text length: {analysis['noise_summary']['avg_text_length']:.1f}")
        logger.info(f"  Very short (< 10 chars): {very_short}")
        logger.info(f"  Very long (> 500 chars): {very_long}")

        return analysis

    def label_clusters_with_gpt(self,
                               dedup_apps_df: pd.DataFrame,
                               cluster_labels: np.ndarray) -> Dict[int, Dict]:
        """
        Phase 6: Generate GPT labels for each cluster.

        Args:
            dedup_apps_df: Deduplicated apps dataframe
            cluster_labels: Cluster labels [N]

        Returns:
            Dict mapping cluster_id to label data
        """
        if self.skip_labeling:
            logger.info("Skipping GPT cluster labeling")
            return {}

        logger.info("=" * 80)
        logger.info("PHASE 6: GPT cluster labeling")
        logger.info("=" * 80)

        unique_apps_list = dedup_apps_df['app_text'].unique().tolist()
        unique_clusters = sorted(set(cluster_labels))
        if -1 in unique_clusters:
            unique_clusters.remove(-1)

        logger.info(f"Labeling {len(unique_clusters)} clusters with {self.label_model}...")

        labels_data = {}
        failed_clusters = []

        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            cluster_indices = np.where(mask)[0]
            cluster_apps = [unique_apps_list[i] for i in cluster_indices if i < len(unique_apps_list)]
            cluster_size = len(cluster_apps)

            # Adaptive sampling
            if cluster_size <= self.label_sample_size:
                sample_apps = cluster_apps
            else:
                # Sample: prioritize apps closer to centroid (future enhancement)
                # For now, just random sample
                sample_indices = np.random.choice(len(cluster_apps), self.label_sample_size, replace=False)
                sample_apps = [cluster_apps[i] for i in sample_indices]

            # Create prompt
            numbered_samples = "\n".join([f"{i+1}. {app}" for i, app in enumerate(sample_apps)])

            user_prompt = f"""Cluster ID: {cluster_id}
Total applications in cluster: {cluster_size}

Sample applications ({len(sample_apps)} of {cluster_size}):
{numbered_samples}

Analyze these applications and provide:
1. A concise label (3-5 words)
2. A description of what they have in common
3. A quality score (1-5) assessing cluster coherence
4. Reasoning for your quality assessment

Respond in JSON format."""

            system_prompt = """You are analyzing clusters of AI applications extracted from job advertisements.
Your task is to generate concise labels and quality assessments for each cluster.

Output valid JSON with this exact structure:
{
    "label": "3-5 word label capturing the cluster theme",
    "description": "1-2 sentence description of common characteristics",
    "quality_score": 1,
    "reasoning": "Brief explanation of quality score"
}"""

            try:
                # Call GPT API
                response = self.openai_client.chat.completions.create(
                    model=self.label_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0,
                    top_p=1,
                    max_tokens=300,
                    seed=42
                )

                # Parse response
                response_text = response.choices[0].message.content
                label_info = json.loads(response_text)

                labels_data[cluster_id] = {
                    'label': label_info.get('label', f'Cluster {cluster_id}'),
                    'description': label_info.get('description', ''),
                    'quality_score': label_info.get('quality_score', 3),
                    'reasoning': label_info.get('reasoning', ''),
                    'sample_apps': sample_apps,
                    'cluster_size': cluster_size
                }

                if (cluster_id + 1) % 10 == 0:
                    logger.info(f"  Labeled {cluster_id + 1}/{len(unique_clusters)} clusters")

            except Exception as e:
                logger.warning(f"Failed to label cluster {cluster_id}: {e}")
                failed_clusters.append(cluster_id)
                labels_data[cluster_id] = {
                    'label': f'Cluster {cluster_id}',
                    'description': 'Label generation failed',
                    'quality_score': 0,
                    'reasoning': str(e),
                    'sample_apps': sample_apps,
                    'cluster_size': cluster_size
                }

        logger.info(f"✓ Labeling complete ({len(unique_clusters) - len(failed_clusters)}/{len(unique_clusters)} successful)")

        if failed_clusters:
            logger.warning(f"Failed to label {len(failed_clusters)} clusters: {failed_clusters}")
            # Save failed clusters for manual review
            with open(os.path.join(self.output_dir, 'failed_cluster_labeling.json'), 'w') as f:
                json.dump({'failed_clusters': failed_clusters}, f, indent=2)

        return labels_data

    def compute_quality_metrics(self,
                               reduced_embeddings: np.ndarray,
                               cluster_labels: np.ndarray,
                               gpt_labels: Dict[int, Dict]) -> Dict:
        """
        Phase 7: Compute cluster quality metrics.

        Args:
            reduced_embeddings: UMAP embeddings [N x 50]
            cluster_labels: Cluster labels [N]
            gpt_labels: GPT-generated labels dict

        Returns:
            Quality report dictionary
        """
        logger.info("=" * 80)
        logger.info("PHASE 7: Computing quality metrics")
        logger.info("=" * 80)

        # Basic statistics
        unique_clusters = sorted(set(cluster_labels))
        if -1 in unique_clusters:
            unique_clusters.remove(-1)

        n_clusters = len(unique_clusters)
        n_noise = list(cluster_labels).count(-1)
        noise_pct = (n_noise / len(cluster_labels)) * 100

        # Cluster sizes
        cluster_sizes = []
        for cluster_id in unique_clusters:
            size = np.sum(cluster_labels == cluster_id)
            cluster_sizes.append(size)

        # Silhouette scores (only if more than 1 cluster)
        silhouette_avg = None
        silhouette_scores_per_cluster = {}

        if n_clusters > 1 and len(cluster_labels) > 0:
            logger.info("Computing silhouette scores...")
            try:
                silhouette_avg = silhouette_score(reduced_embeddings, cluster_labels)
                sample_silhouette_scores = silhouette_samples(reduced_embeddings, cluster_labels)

                for cluster_id in unique_clusters:
                    mask = cluster_labels == cluster_id
                    cluster_silhouette = np.mean(sample_silhouette_scores[mask])
                    silhouette_scores_per_cluster[cluster_id] = float(cluster_silhouette)

                logger.info(f"  Silhouette score (overall): {silhouette_avg:.4f}")
            except Exception as e:
                logger.warning(f"Could not compute silhouette score: {e}")

        # GPT quality scores
        gpt_quality_scores = [info['quality_score'] for info in gpt_labels.values() if info['quality_score'] > 0]
        avg_gpt_quality = np.mean(gpt_quality_scores) if gpt_quality_scores else 0.0

        gpt_quality_dist = Counter(gpt_quality_scores)

        # Top clusters
        top_clusters = []
        for cluster_id in sorted(unique_clusters, key=lambda cid: gpt_labels.get(cid, {}).get('quality_score', 0), reverse=True)[:10]:
            label_info = gpt_labels.get(cluster_id, {})
            top_clusters.append({
                'cluster_id': int(cluster_id),
                'label': label_info.get('label', f'Cluster {cluster_id}'),
                'size': int(np.sum(cluster_labels == cluster_id)),
                'quality_score': label_info.get('quality_score', 0),
                'silhouette': silhouette_scores_per_cluster.get(cluster_id, None)
            })

        quality_report = {
            'clustering_params': {
                'umap_dims': self.umap_dims,
                'umap_neighbors': self.umap_neighbors,
                'umap_min_dist': self.umap_min_dist,
                'hdbscan_min_cluster_size': self.cluster_min_size,
                'hdbscan_min_samples': self.cluster_min_samples
            },
            'summary': {
                'total_apps': len(cluster_labels),
                'num_clusters': n_clusters,
                'num_noise_points': int(n_noise),
                'noise_percentage': float(noise_pct)
            },
            'cluster_size_distribution': {
                'min': int(np.min(cluster_sizes)) if cluster_sizes else 0,
                'max': int(np.max(cluster_sizes)) if cluster_sizes else 0,
                'mean': float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
                'median': float(np.median(cluster_sizes)) if cluster_sizes else 0.0,
                'std': float(np.std(cluster_sizes)) if cluster_sizes else 0.0,
                'q25': float(np.quantile(cluster_sizes, 0.25)) if cluster_sizes else 0.0,
                'q75': float(np.quantile(cluster_sizes, 0.75)) if cluster_sizes else 0.0
            },
            'quality_metrics': {
                'silhouette_score_overall': float(silhouette_avg) if silhouette_avg is not None else None,
                'silhouette_scores_per_cluster': {int(k): float(v) for k, v in silhouette_scores_per_cluster.items()},
                'avg_gpt_quality_score': float(avg_gpt_quality),
                'gpt_quality_distribution': {str(int(k)): int(v) for k, v in gpt_quality_dist.items()}
            },
            'top_clusters': top_clusters
        }

        logger.info(f"✓ Quality metrics computed")
        logger.info(f"  Clusters: {n_clusters}")
        logger.info(f"  Noise: {noise_pct:.1f}%")
        logger.info(f"  Avg cluster size: {quality_report['cluster_size_distribution']['mean']:.1f}")
        logger.info(f"  Silhouette score: {silhouette_avg:.4f}" if silhouette_avg else "  Silhouette score: N/A")
        logger.info(f"  Avg GPT quality: {avg_gpt_quality:.2f}")

        return quality_report

    def generate_visualization(self,
                              reduced_embeddings: np.ndarray,
                              cluster_labels: np.ndarray,
                              dedup_apps_df: pd.DataFrame,
                              gpt_labels: Dict[int, Dict]) -> None:
        """
        Phase 8: Generate interactive UMAP visualization.

        Args:
            reduced_embeddings: UMAP embeddings [N x 50]
            cluster_labels: Cluster labels [N]
            dedup_apps_df: Deduplicated apps dataframe
            gpt_labels: GPT-generated labels dict
        """
        if not self.generate_viz:
            logger.info("Skipping visualization generation")
            return

        logger.info("=" * 80)
        logger.info("PHASE 8: Generating visualization")
        logger.info("=" * 80)

        # Apply UMAP to 2D for visualization (separate from 50D clustering space)
        logger.info("Computing 2D UMAP projection for visualization...")
        umap_2d = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        embedding_2d = umap_2d.fit_transform(reduced_embeddings)

        # Prepare data for Plotly
        unique_apps_list = dedup_apps_df['app_text'].unique().tolist()

        df_viz = pd.DataFrame({
            'x': embedding_2d[:, 0],
            'y': embedding_2d[:, 1],
            'cluster': cluster_labels,
            'app_text': unique_apps_list[:len(cluster_labels)],
            'is_noise': cluster_labels == -1
        })

        # Add cluster labels
        df_viz['cluster_label'] = df_viz['cluster'].map(
            lambda cid: gpt_labels.get(cid, {}).get('label', f'Cluster {cid}') if cid != -1 else 'Noise'
        )

        # Create figure
        fig = go.Figure()

        # Plot noise points first (gray, semi-transparent)
        noise_df = df_viz[df_viz['is_noise']]
        if len(noise_df) > 0:
            fig.add_trace(go.Scatter(
                x=noise_df['x'],
                y=noise_df['y'],
                mode='markers',
                name='Noise',
                marker=dict(size=5, color='lightgray', opacity=0.5),
                text=noise_df['app_text'],
                hovertemplate='<b>Noise</b><br>%{text}<extra></extra>',
                showlegend=True
            ))

        # Plot clusters with distinct colors
        unique_clusters = sorted(set(df_viz['cluster']))
        if -1 in unique_clusters:
            unique_clusters.remove(-1)

        colors = px.colors.qualitative.Plotly
        for idx, cluster_id in enumerate(unique_clusters):
            cluster_df = df_viz[df_viz['cluster'] == cluster_id]
            color = colors[idx % len(colors)]

            cluster_label = gpt_labels.get(cluster_id, {}).get('label', f'Cluster {cluster_id}')
            quality_score = gpt_labels.get(cluster_id, {}).get('quality_score', 0)

            fig.add_trace(go.Scatter(
                x=cluster_df['x'],
                y=cluster_df['y'],
                mode='markers',
                name=f"{cluster_label} (Q:{quality_score})",
                marker=dict(size=6, color=color, opacity=0.7),
                text=cluster_df['app_text'],
                hovertemplate='<b>' + cluster_label + '</b><br>%{text}<extra></extra>',
                showlegend=True
            ))

        # Update layout
        fig.update_layout(
            title=f"AI Applications Clustering (UMAP + HDBSCAN)<br>"
                  f"{len(unique_clusters)} clusters, {list(df_viz['cluster']).count(-1)} noise points",
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            hovermode='closest',
            width=1200,
            height=800,
            font=dict(size=10)
        )

        # Save HTML
        output_path = os.path.join(self.output_dir, 'cluster_visualization.html')
        fig.write_html(output_path)
        logger.info(f"✓ Visualization saved: {output_path}")

    def run_full_pipeline(self,
                         embeddings_file: Optional[str] = None,
                         dedup_apps_file: Optional[str] = None) -> None:
        """
        Run the complete clustering pipeline.

        Args:
            embeddings_file: Path to embeddings cache (auto-detect if None)
            dedup_apps_file: Path to dedup_apps.parquet (auto-detect if None)
        """
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 4 CLUSTERING PIPELINE")
        logger.info("=" * 80 + "\n")

        start_time = datetime.now()

        try:
            # Phase 1: Load data
            embeddings, dedup_apps_df = self.load_embeddings_and_apps(embeddings_file, dedup_apps_file)
            unique_apps_list = dedup_apps_df['app_text'].unique().tolist()

            # Get embeddings for unique apps only
            logger.info(f"Filtering embeddings to {len(unique_apps_list)} unique apps...")
            embeddings = embeddings[:len(unique_apps_list)]

            # Phase 2: UMAP
            reduced_embeddings, umap_model = self.apply_umap(embeddings)

            # Phase 3: HDBSCAN
            cluster_labels = self.apply_hdbscan(reduced_embeddings)

            # Phase 4: Centroids
            centroids = self.compute_centroids(embeddings, reduced_embeddings, cluster_labels)

            # Phase 5: Noise analysis
            noise_analysis = self.analyze_noise(dedup_apps_df, cluster_labels)

            # Phase 6: GPT labeling
            gpt_labels = self.label_clusters_with_gpt(dedup_apps_df, cluster_labels)

            # Update centroids with GPT labels
            for cluster_id, label_info in gpt_labels.items():
                if cluster_id in centroids:
                    centroids[cluster_id]['label'] = label_info.get('label')
                    centroids[cluster_id]['quality_score'] = label_info.get('quality_score')

            # Phase 7: Quality metrics
            quality_report = self.compute_quality_metrics(reduced_embeddings, cluster_labels, gpt_labels)

            # Phase 8: Visualization
            self.generate_visualization(reduced_embeddings, cluster_labels, dedup_apps_df, gpt_labels)

            # Save outputs
            logger.info("=" * 80)
            logger.info("SAVING OUTPUTS")
            logger.info("=" * 80)

            # 1. Cluster assignments
            unique_clusters = sorted(set(cluster_labels))
            assignments_data = []
            for i, app_text in enumerate(unique_apps_list):
                if i >= len(cluster_labels):
                    break
                cluster_id = cluster_labels[i]
                is_noise = cluster_id == -1
                cluster_size = np.sum(cluster_labels == cluster_id) if not is_noise else 0
                cluster_label = gpt_labels.get(cluster_id, {}).get('label') if not is_noise else None

                assignments_data.append({
                    'app_text': app_text,
                    'cluster_id': int(cluster_id),
                    'cluster_size': int(cluster_size),
                    'is_noise': bool(is_noise),
                    'cluster_label': cluster_label
                })

            assignments_df = pd.DataFrame(assignments_data)
            assignments_file = os.path.join(self.output_dir, 'cluster_assignments.parquet')
            assignments_df.to_parquet(assignments_file, index=False)
            logger.info(f"Saved cluster assignments: {assignments_file}")

            # 2. Cluster centroids
            centroids_file = os.path.join(self.output_dir, 'cluster_centroids.pkl')
            with open(centroids_file, 'wb') as f:
                pickle.dump(centroids, f)
            logger.info(f"Saved cluster centroids: {centroids_file}")

            # 3. Cluster labels
            labels_file = os.path.join(self.output_dir, 'cluster_labels.json')
            with open(labels_file, 'w') as f:
                json.dump(gpt_labels, f, indent=2)
            logger.info(f"Saved cluster labels: {labels_file}")

            # 4. Quality report
            quality_file = os.path.join(self.output_dir, 'cluster_quality_report.json')
            with open(quality_file, 'w') as f:
                json.dump(quality_report, f, indent=2)
            logger.info(f"Saved quality report: {quality_file}")

            # 5. Noise analysis
            noise_file = os.path.join(self.output_dir, 'cluster_noise_analysis.json')
            with open(noise_file, 'w') as f:
                json.dump(noise_analysis, f, indent=2)
            logger.info(f"Saved noise analysis: {noise_file}")

            # Summary
            elapsed = datetime.now() - start_time
            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE SUMMARY")
            logger.info("=" * 80)
            logger.info(f"Total applications: {len(unique_apps_list)}")
            logger.info(f"Number of clusters: {quality_report['summary']['num_clusters']}")
            logger.info(f"Noise points: {quality_report['summary']['num_noise_points']} ({quality_report['summary']['noise_percentage']:.1f}%)")
            logger.info(f"Cluster size range: {quality_report['cluster_size_distribution']['min']}-{quality_report['cluster_size_distribution']['max']}")
            logger.info(f"Avg cluster size: {quality_report['cluster_size_distribution']['mean']:.1f}")
            logger.info(f"Avg GPT quality score: {quality_report['quality_metrics']['avg_gpt_quality_score']:.2f}")
            logger.info(f"Silhouette score: {quality_report['quality_metrics']['silhouette_score_overall']:.4f}" if quality_report['quality_metrics']['silhouette_score_overall'] else "Silhouette score: N/A")
            logger.info(f"Time elapsed: {elapsed}")
            logger.info("=" * 80 + "\n")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 4 Extension: Cluster AI Applications using UMAP + HDBSCAN"
    )

    # Input files
    parser.add_argument("--embeddings-file", type=str, default=None,
                       help="Path to embeddings cache (auto-detect if not provided)")
    parser.add_argument("--dedup-apps-file", type=str, default=None,
                       help="Path to dedup_apps.parquet (auto-detect if not provided)")
    parser.add_argument("--embeddings-dir", type=str, default="Data/embeddings",
                       help="Directory containing embeddings cache")
    parser.add_argument("--data-dir", type=str, default="Data",
                       help="Directory containing dedup_apps.parquet")

    # UMAP parameters
    parser.add_argument("--umap-dims", type=int, default=50,
                       help="UMAP target dimensions (default: 50)")
    parser.add_argument("--umap-neighbors", type=int, default=15,
                       help="UMAP n_neighbors parameter (default: 15)")
    parser.add_argument("--umap-min-dist", type=float, default=0.1,
                       help="UMAP min_dist parameter (default: 0.1)")

    # HDBSCAN parameters
    parser.add_argument("--cluster-min-size", type=int, default=5,
                       help="HDBSCAN minimum cluster size (default: 5)")
    parser.add_argument("--cluster-min-samples", type=int, default=5,
                       help="HDBSCAN min_samples parameter (default: 5)")
    parser.add_argument("--cluster-metric", type=str, default="euclidean",
                       help="HDBSCAN distance metric (default: euclidean)")

    # GPT labeling
    parser.add_argument("--label-model", type=str, default="gpt-4.1-mini",
                       help="GPT model for cluster labeling (default: gpt-4.1-mini)")
    parser.add_argument("--label-sample-size", type=int, default=15,
                       help="Apps per cluster to show GPT (default: 15)")
    parser.add_argument("--skip-cluster-labeling", action="store_true",
                       help="Skip GPT labeling (clustering only)")

    # Output options
    parser.add_argument("--cluster-output-dir", type=str, default="Data/clustering",
                       help="Output directory for cluster files (default: Data/clustering)")
    parser.add_argument("--generate-visualization", action="store_true",
                       help="Generate interactive HTML visualization")

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    # Create clusterer
    clusterer = AIApplicationsClusterer(
        embeddings_dir=args.embeddings_dir,
        data_dir=args.data_dir,
        output_dir=args.cluster_output_dir,
        umap_dims=args.umap_dims,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        cluster_min_size=args.cluster_min_size,
        cluster_min_samples=args.cluster_min_samples,
        cluster_metric=args.cluster_metric,
        label_model=args.label_model,
        label_sample_size=args.label_sample_size,
        skip_labeling=args.skip_cluster_labeling,
        generate_viz=args.generate_visualization
    )

    # Run pipeline
    clusterer.run_full_pipeline(args.embeddings_file, args.dedup_apps_file)


if __name__ == "__main__":
    main()
