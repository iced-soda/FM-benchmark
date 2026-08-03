#!/usr/bin/env python
"""Task 2 -- continually-pretrained clustering.

Two epochs of self-supervised masked value completion on the target dataset,
then the same Leiden + ARI/NMI/silhouette evaluation as the zero-shot task, so
the two are directly comparable.

    python run_continual_pretrain_clustering.py --dataset HS
    python run_continual_pretrain_clustering.py --dataset all
    # reuse an adapted checkpoint instead of retraining
    python run_continual_pretrain_clustering.py --dataset BC --skip-pretrain

Manuscript: "Continual pretraining provides selective gains rather than
universal improvement" (Figure 3, Table 3).
"""

from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

from scfm_bench import paths, clustering
from scfm_bench.continual import continual_pretrain
from scfm_bench.datasets import TASK_DATASETS, get_dataset, load_adata

paths.project_on_path()
import scgpt_spatial  # noqa: E402


def run_one(name: str, args) -> dict:
    cfg = get_dataset(name, args.dataset_override)
    modality = cfg.get("modality", "sc")
    label_col = args.label_col or cfg["label_col"]
    gene_col = cfg.get("gene_col", "index")

    print(f"\n{'=' * 70}\n[continual] {name}: {cfg.get('long_name', name)}\n{'=' * 70}",
          flush=True)

    clustering.set_all_seeds(args.seed)

    adata = load_adata(cfg, hvg=args.hvg)
    print(adata, flush=True)

    # Registry may pin an adapted-checkpoint name where the published run used
    # one that does not follow the scGPT_spatial_v1_<dataset> convention.
    adapted_dir = args.out_checkpoint or (
        paths.CHECKPOINT_ROOT / cfg.get("continual_checkpoint",
                                        f"scGPT_spatial_v1_{name}")
    )

    if args.skip_pretrain:
        print(f"[continual] reusing existing checkpoint {adapted_dir}", flush=True)
    else:
        adapted_dir = continual_pretrain(
            adata=adata,
            base_model_dir=str(args.checkpoint),
            out_model_dir=str(adapted_dir),
            gene_col=gene_col,
            seed=args.seed,
            mask_ratio=args.mask_ratio,
            n_bins=args.n_bins,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            use_fast_transformer=not args.no_fast_transformer,
        )
        # continual_pretrain works on an internal gene-subset copy, so `adata`
        # here still carries the full gene space -- embed_data does its own
        # vocabulary alignment. This mirrors the original notebooks exactly.

    adata = scgpt_spatial.tasks.embed_data(
        adata,
        str(adapted_dir),
        gene_col=gene_col,
        obs_to_save=label_col,
        batch_size=args.batch_size,
        return_new_adata=False,
        use_fast_transformer=not args.no_fast_transformer,
    )

    n_clusters = clustering.n_reference_clusters(adata, label_col)
    print(f"reference clusters: {n_clusters}", flush=True)

    res_window = tuple(args.res) if args.res else cfg["continual_res"]
    clustering.cluster(adata, modality, n_clusters, res_window,
                       key=args.emb_key, random_state=args.seed)

    metrics = clustering.clustering_metrics(adata, label_col, emb_key=args.emb_key)

    out = paths.CLUSTERS_DIR / f"{paths.MODEL_NAME}_finetune_clusters_{name}.npz"
    clustering.save_cluster_bundle(out, adata, label_col, metrics,
                                   emb_key=args.emb_key)
    return {name: metrics}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   help="registry name, comma-separated list, or 'all'")
    p.add_argument("--dataset-override", default=None)
    p.add_argument("--checkpoint", default=paths.BASE_CHECKPOINT,
                   help="base checkpoint to adapt from")
    p.add_argument("--out-checkpoint", default=None,
                   help="where to write the adapted checkpoint "
                        "(default: <checkpoints>/scGPT_spatial_v1_<dataset>)")
    p.add_argument("--skip-pretrain", action="store_true",
                   help="reuse an already-adapted checkpoint and only re-cluster")
    p.add_argument("--label-col", default=None)
    p.add_argument("--emb-key", default="X_scGPT")
    p.add_argument("--res", nargs=3, type=float, metavar=("START", "END", "STEP"))
    p.add_argument("--hvg", type=int, default=None)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--mask-ratio", type=float, default=0.20)
    p.add_argument("--n-bins", type=int, default=51)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--no-fast-transformer", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    names = (TASK_DATASETS["continual"] if args.dataset == "all"
             else [s.strip() for s in args.dataset.split(",")])

    summary = {}
    for name in names:
        summary.update(run_one(name, args))

    print("\n===== continual-pretraining summary =====")
    for name, m in summary.items():
        print(f"{name:8s} ARI={m['ARI']:.4f}  NMI={m['NMI']:.4f}  SIL={m['SIL']:.4f}")


if __name__ == "__main__":
    main()
