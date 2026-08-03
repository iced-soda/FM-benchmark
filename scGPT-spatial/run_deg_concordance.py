#!/usr/bin/env python
"""Task 5 -- marker-gene concordance between model clusters and annotated domains.

For each Leiden cluster from the continually-pretrained embedding and each
ground-truth domain, take the top-N Wilcoxon rank-sum marker genes, then compare
the two sets by raw overlap count and by Jaccard index. Writes both matrices as
CSV and as heatmaps.

    python run_deg_concordance.py --dataset Emb
    python run_deg_concordance.py --dataset all --top-n 20

Requires the cluster bundle from `run_continual_pretrain_clustering.py`
(`scGPT_finetune_clusters_<dataset>.npz`) to already exist.

Manuscript: "Marker-gene concordance reveals biological specificity of spatial
clusters" (Figure 5 = overlap counts for MOSTA E9.5; Supplementary Figure S2 =
Jaccard). Both use top-N = 20.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

warnings.filterwarnings("ignore")

from scfm_bench import paths
from scfm_bench.datasets import TASK_DATASETS, get_dataset


def jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


def top_markers(adata, groupby, top_n) -> dict:
    """Top-`top_n` Wilcoxon marker genes per group, as {group: [genes]}."""
    sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon",
                            use_raw=False, n_genes=top_n)
    df = sc.get.rank_genes_groups_df(adata, None)
    return (df.groupby("group")["names"]
              .apply(lambda x: x.head(top_n).tolist())
              .to_dict())


def heatmap(matrix: pd.DataFrame, out_path: Path, title: str, cbar_label: str,
            cmap: str, fmt: str) -> None:
    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap=cmap, linewidths=0.5,
                cbar_kws={"label": cbar_label}, annot_kws={"size": 8})
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel("Predicted cluster", fontsize=12)
    plt.ylabel("Ground-truth domain", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", format="svg")
    plt.close()
    print(f"  -> {out_path}", flush=True)


def run_one(name: str, args) -> None:
    cfg = get_dataset(name, args.dataset_override)
    label_col = args.label_col or cfg["label_col"]
    top_n = args.top_n

    print(f"\n{'=' * 70}\n[DEG] {name}: {cfg.get('long_name', name)} (top-{top_n})\n"
          f"{'=' * 70}", flush=True)

    adata = sc.read_h5ad(cfg["h5ad"])
    adata.obs_names_make_unique()
    if cfg.get("deg", {}).get("var_names_make_unique", True):
        adata.var_names_make_unique()

    labels_path = Path(args.clusters or (
        paths.CLUSTERS_DIR / f"{paths.MODEL_NAME}_finetune_clusters_{name}.npz"))
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{labels_path} not found. Run run_continual_pretrain_clustering.py "
            f"--dataset {name} first."
        )
    bundle = np.load(labels_path, allow_pickle=True)
    cluster_labels = bundle["labels"]
    if len(cluster_labels) != adata.n_obs:
        raise ValueError(
            f"Cluster bundle has {len(cluster_labels)} labels but the h5ad has "
            f"{adata.n_obs} cells. The bundle was produced from a different "
            "version of this dataset."
        )
    adata.obs["leiden"] = pd.Categorical(cluster_labels.astype(str))

    # NOTE: log1p only, no library-size normalisation, matching the published
    # run. Whether X is raw counts or already normalised depends on the source
    # file -- see README "Known inconsistencies".
    if not args.no_log1p:
        sc.pp.log1p(adata)

    top_genes_gt = top_markers(adata, label_col, top_n)
    top_genes_pred = top_markers(adata, "leiden", top_n)

    overlap = pd.DataFrame(
        {cl: {ct: len(set(pred) & set(gt)) for ct, gt in top_genes_gt.items()}
         for cl, pred in top_genes_pred.items()}
    )
    jac = pd.DataFrame(
        index=list(top_genes_gt), columns=list(top_genes_pred), dtype=float
    )
    for ct, gt in top_genes_gt.items():
        for cl, pred in top_genes_pred.items():
            jac.loc[ct, cl] = jaccard(gt, pred)

    out_dir = Path(args.out_dir or paths.FIGURES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    overlap.to_csv(out_dir / f"deg_overlap_{top_n}_{name}.csv")
    jac.to_csv(out_dir / f"deg_jaccard_{top_n}_{name}.csv")
    pd.DataFrame({"cluster": list(top_genes_pred),
                  "top_genes": [", ".join(v) for v in top_genes_pred.values()]}
                 ).to_csv(out_dir / f"deg_cluster_markers_{top_n}_{name}.csv", index=False)

    heatmap(overlap, out_dir / f"similarity_heatmap_{top_n}_{name}.svg",
            "Gene overlap between cluster and domain DEGs",
            "No. of matching genes", "Reds", "d")
    heatmap(jac, out_dir / f"jaccard_index_pairwise_heatmap_{top_n}_{name}.svg",
            "Pairwise Jaccard index, ground truth vs predicted",
            "Jaccard index", "Blues", ".2f")

    print(overlap.to_string(), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   help="registry name, comma-separated list, or 'all'")
    p.add_argument("--dataset-override", default=None)
    p.add_argument("--clusters", default=None,
                   help="explicit path to the cluster .npz (default: derived from dataset)")
    p.add_argument("--label-col", default=None)
    p.add_argument("--top-n", type=int, default=20,
                   help="marker genes per group (20 in the manuscript)")
    p.add_argument("--no-log1p", action="store_true",
                   help="skip log1p, for files already stored log-transformed")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    names = (TASK_DATASETS["deg"] if args.dataset == "all"
             else [s.strip() for s in args.dataset.split(",")])
    for name in names:
        run_one(name, args)


if __name__ == "__main__":
    main()
