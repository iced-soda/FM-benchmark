"""Shared helpers for the scGPT-spatial arm of the single-cell/spatial FM benchmark."""

from .paths import DATA_ROOT, RESULTS_ROOT, CHECKPOINT_ROOT, BASE_CHECKPOINT, project_on_path
from .datasets import DATASETS, get_dataset, load_adata, list_datasets
from .clustering import (
    set_all_seeds,
    cluster_singlecell,
    cluster_spatial,
    clustering_metrics,
    save_cluster_bundle,
)

__all__ = [
    "DATA_ROOT",
    "RESULTS_ROOT",
    "CHECKPOINT_ROOT",
    "BASE_CHECKPOINT",
    "project_on_path",
    "DATASETS",
    "get_dataset",
    "load_adata",
    "list_datasets",
    "set_all_seeds",
    "cluster_singlecell",
    "cluster_spatial",
    "clustering_metrics",
    "save_cluster_bundle",
]
