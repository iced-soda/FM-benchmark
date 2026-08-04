#!/usr/bin/env python
"""Task 1 -- zero-shot clustering of scGPT-spatial embeddings.

Embeds a dataset with the released scGPT-spatial checkpoint (no adaptation of
any kind), runs Leiden at the resolution that reproduces the reference label
count, and scores ARI / NMI / silhouette.

    python run_zero_shot_clustering.py --dataset ATAA
    python run_zero_shot_clustering.py --dataset all
    python run_zero_shot_clustering.py --dataset BC --res 2.0 3.0 0.01

Manuscript: "Zero-shot embeddings recover biological structure but generalise
unevenly across modalities" (Figure 2, Table 2).
"""

from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

from scfm_bench import paths, clustering
from scfm_bench.datasets import TASK_DATASETS, get_dataset, load_adata

paths.project_on_path()
import scgpt_spatial  # noqa: E402  (needs the repo root on sys.path first)


def run_one(name: str, args) -> dict:
    cfg = get_dataset(name, args.dataset_override)
    modality = cfg.get("modality", "sc")
    label_col = args.label_col or cfg["label_col"]
    gene_col = cfg.get("gene_col", "index")

    print(f"\n{'=' * 70}\n[zero-shot] {name}: {cfg.get('long_name', name)}\n{'=' * 70}",
          flush=True)

    clustering.set_all_seeds(args.seed)

    adata = load_adata(
        cfg,
        hvg=args.hvg if args.hvg is not None else cfg.get("zero_shot_hvg"),
        use_count_layer=cfg.get("zero_shot_use_count_layer",
                                cfg.get("use_count_layer", False)),
    )
    print(adata, flush=True)

    # Cell embeddings from the frozen checkpoint. embed_data aligns genes to the
    # checkpoint vocabulary, drops out-of-vocabulary genes, bins expression into
    # 51 levels and returns the L2-normalised CLS state in obsm["X_scGPT"].
    adata = scgpt_spatial.tasks.embed_data(
        adata,
        str(args.checkpoint),
        gene_col=gene_col,
        obs_to_save=label_col,
        batch_size=args.batch_size,
        return_new_adata=False,
        use_fast_transformer=not args.no_fast_transformer,
    )

    n_clusters = clustering.n_reference_clusters(adata, label_col)
    print(f"reference clusters: {n_clusters}", flush=True)

    res_window = tuple(args.res) if args.res else cfg["zero_shot_res"]
    clustering.cluster(adata, modality, n_clusters, res_window,
                       key=args.emb_key, random_state=args.seed)

    metrics = clustering.clustering_metrics(adata, label_col, emb_key=args.emb_key)

    out = paths.CLUSTERS_DIR / f"{paths.MODEL_NAME}_zeroshot_clusters_{name}.npz"
    clustering.save_cluster_bundle(out, adata, label_col, metrics,
                                   emb_key=args.emb_key)
    return {name: metrics}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   help="registry name, comma-separated list, or 'all'")
    p.add_argument("--dataset-override", default=None,
                   help="JSON file or inline JSON describing a dataset not in the registry")
    p.add_argument("--checkpoint", default=paths.BASE_CHECKPOINT,
                   help="pretrained scGPT-spatial checkpoint directory")
    p.add_argument("--label-col", default=None, help="override the reference label column")
    p.add_argument("--emb-key", default="X_scGPT")
    p.add_argument("--res", nargs=3, type=float, metavar=("START", "END", "STEP"),
                   help="override the Leiden resolution sweep window")
    p.add_argument("--hvg", type=int, default=None,
                   help="subset to N seurat_v3 HVGs before embedding")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--no-fast-transformer", action="store_true",
                   help="disable flash-attention (slower, needed on non-Ampere GPUs)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    names = (TASK_DATASETS["zero_shot"] if args.dataset == "all"
             else [s.strip() for s in args.dataset.split(",")])

    summary = {}
    for name in names:
        summary.update(run_one(name, args))

    print("\n===== zero-shot summary =====")
    for name, m in summary.items():
        print(f"{name:8s} ARI={m['ARI']:.4f}  NMI={m['NMI']:.4f}  SIL={m['SIL']:.4f}")


if __name__ == "__main__":
    main()
