"""Leiden clustering, evaluation metrics and result serialisation.

Two clustering routines are kept deliberately separate because the manuscript
uses different graphs for the two modalities:

  * `cluster_singlecell` -- Leiden directly on the 512-d cell embedding,
    k = 15 neighbours. Resolution swept upward from `start`.
  * `cluster_spatial`    -- embedding first reduced to 20 PCs, k = 50
    neighbours. Resolution swept *downward* from `end`.

Both sweep until the cluster count equals the number of reference labels.
Every stochastic step is pinned to seed 42.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

SEED = 42


def set_all_seeds(seed: int = SEED) -> None:
    """Pin every RNG that the clustering and continual-pretraining paths touch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# --------------------------------------------------------------------------- #
# scRNA-seq: Leiden straight on the embedding
# --------------------------------------------------------------------------- #

def _search_res_singlecell(adata, n_clusters, use_rep, start, end, increment,
                           n_neighbors=15, random_state=SEED) -> Tuple[float, str]:
    step = -abs(increment) if start > end else abs(increment)
    res_list = np.round(np.arange(start, end + (1e-9 if step > 0 else -1e-9), step), 3)

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)
    for res in res_list:
        key_added = f"leiden_{res:.3f}"
        sc.tl.leiden(adata, resolution=float(res), key_added=key_added,
                     random_state=random_state)
        k = int(adata.obs[key_added].nunique())
        print(f"resolution={res:.3f}, cluster number={k}", flush=True)
        if k == n_clusters:
            return float(res), key_added

    raise AssertionError(
        f"No resolution in [{start}, {end}] step {increment} yields {n_clusters} "
        "clusters. Widen the range or shrink the step."
    )


def cluster_singlecell(adata, n_clusters, key="X_scGPT", start=0.9, end=0.5,
                       increment=0.1, n_neighbors=15, key_out="domain",
                       random_state=SEED) -> float:
    """Leiden on `adata.obsm[key]`; writes labels to `adata.obs[key_out]`."""
    res, final_key = _search_res_singlecell(
        adata, n_clusters, use_rep=key, start=start, end=end,
        increment=increment, n_neighbors=n_neighbors, random_state=random_state,
    )
    adata.obs[key_out] = adata.obs[final_key].astype("category")
    adata.uns["chosen_leiden_resolution"] = float(res)
    print(f"[clustering] labels -> adata.obs['{key_out}'] at res={res:.3f} "
          f"(seed={random_state})", flush=True)
    return float(res)


# --------------------------------------------------------------------------- #
# Spatial: PCA(20) -> Leiden, k = 50
# --------------------------------------------------------------------------- #

def _search_res_spatial(adata, n_clusters, use_rep="emb_pca", start=0.1, end=3.0,
                        increment=0.01, n_neighbors=50, random_state=SEED) -> float:
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)
    # Descending sweep, matching the BenchmarkST-style helper used for the
    # published spatial results.
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=random_state, resolution=res)
        count_unique = len(pd.DataFrame(adata.obs["leiden"]).leiden.unique())
        print(f"resolution={res}, cluster number={count_unique}", flush=True)
        if count_unique == n_clusters:
            return res

    raise AssertionError(
        f"No resolution in [{start}, {end}) step {increment} yields {n_clusters} "
        "clusters. Widen the range or shrink the step."
    )


def cluster_spatial(adata, n_clusters, key="X_scGPT", start=0.1, end=3.0,
                    increment=0.01, n_pcs=20, n_neighbors=50, key_out="domain",
                    random_state=SEED) -> float:
    """PCA-reduce `adata.obsm[key]`, then Leiden; labels to `adata.obs[key_out]`."""
    pca = PCA(n_components=n_pcs, random_state=random_state)
    adata.obsm["emb_pca"] = pca.fit_transform(adata.obsm[key].copy())

    res = _search_res_spatial(
        adata, n_clusters, use_rep="emb_pca", start=start, end=end,
        increment=increment, n_neighbors=n_neighbors, random_state=random_state,
    )
    sc.tl.leiden(adata, random_state=random_state, resolution=res)
    adata.obs[key_out] = adata.obs["leiden"]
    adata.uns["chosen_leiden_resolution"] = float(res)
    print(f"[clustering] labels -> adata.obs['{key_out}'] at res={res} "
          f"(seed={random_state})", flush=True)
    return float(res)


def cluster(adata, modality, n_clusters, res_window, key="X_scGPT",
            key_out="domain", random_state=SEED) -> float:
    """Dispatch to the modality-appropriate clustering routine."""
    start, end, increment = res_window
    if modality == "spatial":
        return cluster_spatial(adata, n_clusters, key=key, start=start, end=end,
                               increment=increment, key_out=key_out,
                               random_state=random_state)
    return cluster_singlecell(adata, n_clusters, key=key, start=start, end=end,
                              increment=increment, key_out=key_out,
                              random_state=random_state)


# --------------------------------------------------------------------------- #
# Metrics + serialisation
# --------------------------------------------------------------------------- #

def clustering_metrics(adata, gt_col, pred_col="domain", emb_key="X_scGPT") -> Dict:
    """ARI / NMI against the reference labels, silhouette on the embedding.

    Cells with a missing label on either side are dropped before scoring.
    """
    lab_gt = adata.obs[gt_col].astype("category")
    lab_pred = adata.obs[pred_col].astype("category")
    mask = (~lab_gt.isna()) & (~lab_pred.isna())

    y_true = lab_gt[mask].astype(str).to_numpy()
    y_pred = lab_pred[mask].astype(str).to_numpy()

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    codes = lab_pred[mask].cat.remove_unused_categories().cat.codes.to_numpy()
    if np.unique(codes).size >= 2:
        sil = silhouette_score(adata.obsm[emb_key][mask.values], codes)
    else:
        sil = float("nan")

    print(f"ARI={ari:.4f} | NMI={nmi:.4f} | Silhouette={sil:.4f}", flush=True)
    return {"ARI": float(ari), "NMI": float(nmi), "SIL": float(sil),
            "n_scored": int(mask.sum())}


def save_cluster_bundle(out_path, adata, gt_col, metrics, pred_col="domain",
                        emb_key="X_scGPT") -> Path:
    """Write the compact .npz consumed by `plot_results.py`.

    Contents: predicted labels, cell embeddings, the three metrics, the ground
    truth strings, and spatial coordinates when the dataset has them.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "labels": adata.obs[pred_col].astype(int).to_numpy(),
        "embeddings": adata.obsm[emb_key],
        "ARI": metrics["ARI"],
        "NMI": metrics["NMI"],
        "SIL": metrics["SIL"],
        "ground_truth": adata.obs[gt_col].astype(str).to_numpy(),
    }
    if "spatial" in adata.obsm:
        payload["coords"] = adata.obsm["spatial"]

    np.savez_compressed(out_path, **payload)
    print(f"Saved cluster bundle -> {out_path}", flush=True)
    return out_path


def n_reference_clusters(adata, label_col) -> int:
    """Number of reference labels actually present (unused categories dropped)."""
    lab = adata.obs[label_col]
    if isinstance(lab.dtype, pd.CategoricalDtype):
        lab = lab.cat.remove_unused_categories()
        adata.obs[label_col] = lab
        return len(lab.cat.categories)
    return int(lab.dropna().nunique())
