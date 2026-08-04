#!/usr/bin/env python
"""Task 3 -- supervised cell type annotation.

Fine-tunes the scGPT-spatial checkpoint with the cell-type classification
objective, following the authors' annotation tutorial, and scores the held-out
test split with accuracy, macro precision/recall/F1, macro PR-AUC and macro
ROC-AUC.

    python run_cell_annotation.py --dataset ATAA
    python run_cell_annotation.py --dataset SEAAD --epochs 10

Manuscript: "Supervised annotation reveals fine-grained and task-specific
failure modes" (Figure 4, Table 4).

Notes on the split. The 80:20 train/test partition is precomputed and shared
across every model in the benchmark, so it is read from disk rather than drawn
here. Within the training portion a further 10% is held out for model
selection. That inner split intentionally does not pass `random_state`: it
inherits the global NumPy seed set at import, which is how the published run
was seeded. Do not add `random_state` unless you intend to change the split.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import scanpy as sc
import torch
from scipy import sparse
from scipy.sparse import issparse
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    label_ranking_average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
os.environ["KMP_WARNINGS"] = "off"

from scfm_bench import paths
from scfm_bench.annotation_data import load_annotation_splits
from scfm_bench.clustering import set_all_seeds
from scfm_bench.datasets import TASK_DATASETS, get_dataset

paths.project_on_path()
import scgpt_spatial as scg  # noqa: E402
from scgpt_spatial.model import TransformerModel  # noqa: E402
from scgpt_spatial.preprocess import Preprocessor  # noqa: E402
from scgpt_spatial.tokenizer import random_mask_value, tokenize_and_pad_batch  # noqa: E402
from scgpt_spatial.tokenizer.gene_tokenizer import GeneVocab  # noqa: E402
from scgpt_spatial.utils import set_seed  # noqa: E402

# --------------------------------------------------------------------------- #
# Fixed task configuration (mirrors the scGPT annotation tutorial)
# --------------------------------------------------------------------------- #

PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]
MAX_SEQ_LEN = 3001
INPUT_STYLE = "binned"
INPUT_EMB_STYLE = "continuous"
CELL_EMB_STYLE = "cls"
MVC_DECODER_STYLE = "inner product"

# Objectives. Only the classification head is trained; MLM/MVC/ECS/DAB are off,
# so masking is disabled (mask_ratio = 0).
MLM = False
CLS = True
MVC = False
ECS = False
DAB = False
INPUT_BATCH_LABELS = False

MASK_VALUE = -1
PAD_VALUE = -2


class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def _num_workers(batch_size: int) -> int:
    try:
        return min(len(os.sched_getaffinity(0)), max(1, batch_size // 2))
    except (AttributeError, OSError):
        return max(1, batch_size // 2)


def prepare_dataloader(data_pt, batch_size, shuffle=False, drop_last=False):
    return DataLoader(
        dataset=SeqDataset(data_pt), batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, num_workers=_num_workers(batch_size), pin_memory=True,
    )


@torch.no_grad()
def collect_probs(model, loader, pad_token_id, device):
    """Class probabilities for the whole loader (autocast keeps flash-attn happy)."""
    model.eval()
    logits_all = []
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    for batch in loader:
        gene_ids = batch["gene_ids"].to(device)
        values = batch["values"].to(device)
        src_kpm = gene_ids == pad_token_id

        kwargs = {"CLS": True}
        if getattr(model, "use_batch_labels", False):
            kwargs["batch_labels"] = batch["batch_labels"].to(device)

        with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
            out = model(src=gene_ids, values=values,
                        src_key_padding_mask=src_kpm, **kwargs)
            if "cls_output" not in out:
                raise RuntimeError("Model returned no 'cls_output'; need n_cls>1 and CLS=True.")
            logits = out["cls_output"]

        logits_all.append(
            torch.nan_to_num(logits.detach().to(torch.float32),
                             nan=0.0, posinf=1e4, neginf=-1e4).cpu()
        )

    if not logits_all:
        raise RuntimeError("Empty loader produced no batches.")

    probs = F.softmax(torch.cat(logits_all, dim=0), dim=1).cpu().numpy()
    if not np.isfinite(probs).all():
        raise RuntimeError("Non-finite probabilities after softmax.")
    return probs


def macro_ovr(y_true, probs, score_fn):
    """Macro one-vs-rest score, skipping degenerate classes.

    Falls back to label-ranking average precision only if *every* class is
    degenerate (single-class test set), which should not happen in practice.
    """
    classes_present = np.unique(y_true)
    scores = []
    for c in classes_present:
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == y_bin.size:
            continue
        val = score_fn(y_bin, probs[:, c])
        if np.isfinite(val):
            scores.append(float(val))

    if scores:
        return float(np.mean(scores))
    Y = label_binarize(y_true, classes=classes_present)
    return float(label_ranking_average_precision_score(Y, probs[:, classes_present]))


def run_one(name: str, args) -> Dict:
    cfg = get_dataset(name, args.dataset_override)
    print(f"\n{'=' * 70}\n[annotation] {name}: {cfg.get('long_name', name)}\n{'=' * 70}",
          flush=True)

    set_all_seeds(args.seed)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir) / f"dev_{name}-{time.strftime('%b%d-%H-%M')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    print(f"run artefacts -> {save_dir}", flush=True)

    # ---------------- data ---------------- #
    adata, adata_test_raw, id2type, num_types = load_annotation_splits(cfg)

    # ---------------- vocabulary + model config ---------------- #
    model_dir = Path(args.checkpoint)
    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    import shutil
    shutil.copy(model_dir / "vocab.json", save_dir / "vocab.json")
    for s in SPECIAL_TOKENS:
        if s not in vocab:
            vocab.append_token(s)

    adata.var["id_in_vocab"] = [1 if g in vocab else -1 for g in adata.var["gene_name"]]
    logger.info(f"match {int((adata.var['id_in_vocab'] >= 0).sum())}/{adata.n_vars} "
                f"genes in vocabulary of size {len(vocab)}.")
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    with open(model_dir / "args.json") as f:
        model_configs = json.load(f)
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

    # ---------------- preprocessing ---------------- #
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=True,                 # inputs are raw UMI counts
        result_log1p_key="X_log1p",
        subset_hvg=False,
        hvg_flavor="seurat_v3",
        binning=args.n_bins,
        result_binned_key="X_binned",
    )

    adata_test = adata[adata.obs["str_batch"] == "1"]
    adata = adata[adata.obs["str_batch"] == "0"]
    preprocessor(adata, batch_key=None)
    preprocessor(adata_test, batch_key=None)

    input_layer_key = {"normed_raw": "X_normed", "log1p": "X_normed",
                       "binned": "X_binned"}[INPUT_STYLE]

    layer = adata.layers[input_layer_key]
    all_counts = layer.toarray() if issparse(layer) else layer
    genes = adata.var["gene_name"].tolist()
    celltypes_labels = np.array(adata.obs["celltype_id"].tolist())
    batch_ids = np.array(adata.obs["batch_id"].tolist())
    num_batch_types = len(set(batch_ids.tolist()))

    # Inner train/valid split -- see module docstring on the missing random_state.
    (train_data, valid_data, train_celltype_labels, valid_celltype_labels,
     train_batch_labels, valid_batch_labels) = train_test_split(
        all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
    )

    vocab.set_default_index(vocab[PAD_TOKEN])
    gene_ids = np.array(vocab(genes), dtype=int)

    def _tok(data):
        return tokenize_and_pad_batch(
            data, gene_ids, max_len=MAX_SEQ_LEN, vocab=vocab, pad_token=PAD_TOKEN,
            pad_value=PAD_VALUE, append_cls=True, include_zero_gene=False,
        )

    tokenized_train, tokenized_valid = _tok(train_data), _tok(valid_data)
    logger.info(f"train samples: {tokenized_train['genes'].shape[0]}, "
                f"feature length: {tokenized_train['genes'].shape[1]}")
    logger.info(f"valid samples: {tokenized_valid['genes'].shape[0]}, "
                f"feature length: {tokenized_valid['genes'].shape[1]}")

    def prepare_data():
        def pack(tok, ct_labels, b_labels):
            masked = random_mask_value(tok["values"], mask_ratio=args.mask_ratio,
                                       mask_value=MASK_VALUE, pad_value=PAD_VALUE)
            return {
                "gene_ids": tok["genes"],
                "values": masked,
                "target_values": tok["values"],
                "batch_labels": torch.from_numpy(b_labels).long(),
                "celltype_labels": torch.from_numpy(ct_labels).long(),
            }
        return (pack(tokenized_train, train_celltype_labels, train_batch_labels),
                pack(tokenized_valid, valid_celltype_labels, valid_batch_labels))

    # ---------------- model ---------------- #
    model = TransformerModel(
        len(vocab), embsize, nhead, d_hid, nlayers,
        nlayers_cls=3,
        n_cls=num_types if CLS else 1,
        vocab=vocab,
        dropout=args.dropout,
        pad_token=PAD_TOKEN,
        pad_value=PAD_VALUE,
        do_mvc=MVC,
        do_dab=DAB,
        use_batch_labels=INPUT_BATCH_LABELS,
        num_batch_labels=num_batch_types,
        input_emb_style=INPUT_EMB_STYLE,
        n_input_bins=args.n_bins,
        cell_emb_style=CELL_EMB_STYLE,
        mvc_decoder_style=MVC_DECODER_STYLE,
        ecs_threshold=0.0,
        explicit_zero_prob=False,
        use_fast_transformer=not args.no_fast_transformer,
        fast_transformer_backend="flash",
        pre_norm=False,
    )

    model_file = model_dir / "best_model.pt"
    try:
        model.load_state_dict(torch.load(model_file))
        logger.info(f"Loaded all model params from {model_file}")
    except (RuntimeError, KeyError):
        model_dict = model.state_dict()
        pretrained = torch.load(model_file)
        pretrained = {k: v for k, v in pretrained.items()
                      if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained)
        model.load_state_dict(model_dict)
        logger.info(f"Loaded {len(pretrained)} shape-compatible params from {model_file}")

    if args.freeze_encoder:
        for pname, para in model.named_parameters():
            if "encoder" in pname and "transformer_encoder" not in pname:
                para.requires_grad = False

    model.to(device)

    criterion_cls = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 eps=1e-4 if args.amp else 1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=args.schedule_ratio)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    def train_epoch(loader, epoch):
        model.train()
        total_loss = total_error = 0.0
        start_time = time.time()
        for batch, batch_data in enumerate(loader):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[PAD_TOKEN])

            with torch.cuda.amp.autocast(enabled=args.amp):
                output_dict = model(
                    input_gene_ids, input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None, CLS=CLS, MVC=MVC, ECS=ECS, do_sample=False,
                )
                loss = criterion_cls(output_dict["cls_output"], celltype_labels)
                error_rate = 1 - (
                    (output_dict["cls_output"].argmax(1) == celltype_labels).sum().item()
                ) / celltype_labels.size(0)

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_error += error_rate
            if batch % args.log_interval == 0 and batch > 0:
                ms = (time.time() - start_time) * 1000 / args.log_interval
                logger.info(f"| epoch {epoch:3d} | {batch:3d}/{len(loader):3d} batches | "
                            f"lr {scheduler.get_last_lr()[0]:05.4f} | ms/batch {ms:5.2f} | "
                            f"cls {total_loss / args.log_interval:5.2f} | "
                            f"err {total_error / args.log_interval:5.2f} |")
                total_loss = total_error = 0.0
                start_time = time.time()

    @torch.no_grad()
    def evaluate(loader, return_raw=False):
        model.eval()
        total_loss = total_error = 0.0
        total_num = 0
        predictions, embeddings = [], []
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[PAD_TOKEN])

            with torch.cuda.amp.autocast(enabled=args.amp):
                output_dict = model(
                    input_gene_ids, input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None, CLS=CLS, MVC=False, ECS=False, do_sample=False,
                )
                output_values = output_dict["cls_output"]
                loss = criterion_cls(output_values, celltype_labels)
                embeddings.append(output_dict["cell_emb"].cpu().numpy())

            n = len(input_gene_ids)
            total_loss += loss.item() * n
            accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
            total_error += (1 - accuracy / n) * n
            total_num += n
            predictions.append(output_values.argmax(1).cpu().numpy())

        if return_raw:
            return np.concatenate(predictions, axis=0), np.concatenate(embeddings, axis=0)
        return total_loss / total_num, total_error / total_num

    # ---------------- training loop ---------------- #
    best_val_loss, best_model = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_data_pt, valid_data_pt = prepare_data()
        train_loader = prepare_dataloader(train_data_pt, args.batch_size, shuffle=False)
        valid_loader = prepare_dataloader(valid_data_pt, args.batch_size, shuffle=False)

        train_epoch(train_loader, epoch)
        val_loss, val_err = evaluate(valid_loader)
        logger.info("-" * 89)
        logger.info(f"| end of epoch {epoch:3d} | time: {time.time() - epoch_start:5.2f}s | "
                    f"valid loss {val_loss:5.4f} | err {val_err:5.4f}")
        logger.info("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            logger.info(f"Best model with score {best_val_loss:5.4f}")

        scheduler.step()

    model = best_model if best_model is not None else model

    # ---------------- test ---------------- #
    layer = adata_test.layers[input_layer_key]
    test_counts = layer.toarray() if issparse(layer) else layer
    y_true = np.asarray(adata_test.obs["celltype_id"].tolist(), dtype=int)
    test_batch_ids = np.asarray(adata_test.obs["batch_id"].tolist(), dtype=int)

    tokenized_test = _tok(test_counts)
    test_data_pt = {
        "gene_ids": tokenized_test["genes"],
        "values": random_mask_value(tokenized_test["values"], mask_ratio=args.mask_ratio,
                                    mask_value=MASK_VALUE, pad_value=PAD_VALUE),
        "target_values": tokenized_test["values"],
        "batch_labels": torch.from_numpy(test_batch_ids).long(),
        "celltype_labels": torch.from_numpy(y_true).long(),
    }
    test_loader = prepare_dataloader(test_data_pt, args.batch_size, shuffle=False)

    y_pred, _ = evaluate(test_loader, return_raw=True)
    probs = collect_probs(model, test_loader, vocab[PAD_TOKEN], device)

    results = {
        "test/accuracy": float(accuracy_score(y_true, y_pred)),
        "test/precision": float(precision_score(y_true, y_pred, average="macro",
                                                zero_division=0)),
        "test/recall": float(recall_score(y_true, y_pred, average="macro",
                                          zero_division=0)),
        "test/macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                        zero_division=0)),
        "test/roc_auc": macro_ovr(y_true, probs, roc_auc_score),
        "test/pr_auc": macro_ovr(y_true, probs, average_precision_score),
    }
    print(json.dumps(results, indent=2), flush=True)

    # ---------------- visualisation bundle ---------------- #
    adata_test_raw.obs["predictions"] = [id2type[p] for p in y_pred]
    for col in ("celltype", "predictions"):
        adata_test_raw.obs[col] = adata_test_raw.obs[col].astype(str).astype("category")

    adata_vis = adata_test_raw.copy()
    sc.pp.normalize_total(adata_vis, target_sum=1e4)
    sc.pp.log1p(adata_vis)
    sc.pp.filter_genes(adata_vis, min_cells=3)
    sc.pp.highly_variable_genes(adata_vis, n_top_genes=2000, flavor="cell_ranger")
    adata_vis = adata_vis[:, adata_vis.var["highly_variable"]].copy()
    sc.pp.scale(adata_vis)
    sc.tl.pca(adata_vis, svd_solver="arpack", random_state=args.seed)
    sc.pp.neighbors(adata_vis, use_rep="X_pca", random_state=args.seed)
    sc.tl.umap(adata_vis, random_state=args.seed)

    payload = {
        "Model_name": np.array(paths.MODEL_NAME),
        "step": np.array("finetune"),
        "dataset": np.array(name),
        "embedding_name": np.array("X_pca"),
        "embeddings": adata_vis.obsm["X_pca"],
        "umap": adata_vis.obsm["X_umap"],
        "predictions_str": adata_test_raw.obs["predictions"].to_numpy(),
        "labels_num": np.asarray(y_true),
        "celltype": adata_test_raw.obs["celltype"].to_numpy(),
        "test_accuracy": results["test/accuracy"],
        "test_precision": results["test/precision"],
        "test_recall": results["test/recall"],
        "test_macro_f1": results["test/macro_f1"],
        "test_roc_auc": results["test/roc_auc"],
        "test_pr_auc": results["test/pr_auc"],
        "results_json": np.array(json.dumps(results)),
    }

    save_raw = cfg.get("annotation", {}).get("save_raw_payload", True)
    if save_raw and not args.no_raw_payload:
        X_csr = (adata_test_raw.X.tocsr() if sparse.issparse(adata_test_raw.X)
                 else sparse.csr_matrix(adata_test_raw.X))
        payload.update({
            "raw_X_data": X_csr.data,
            "raw_X_indices": X_csr.indices,
            "raw_X_indptr": X_csr.indptr,
            "raw_X_shape": np.array(X_csr.shape, dtype=np.int64),
            "raw_obs_json": np.array(json.dumps(adata_test_raw.obs.to_dict(orient="list")),
                                     dtype=object),
            "raw_var_json": np.array(json.dumps(adata_test_raw.var.to_dict(orient="list")),
                                     dtype=object),
            "raw_obs_names": adata_test_raw.obs_names.to_numpy().astype("U"),
            "raw_var_names": adata_test_raw.var_names.to_numpy().astype("U"),
        })

    paths.CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = (paths.CLUSTERS_DIR /
                f"{paths.MODEL_NAME}_finetune_cell_annotation_{name}.npz")
    np.savez_compressed(npz_path, **payload)
    print(f"Saved annotation bundle -> {npz_path}", flush=True)

    return {name: results}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   help="registry name, comma-separated list, or 'all'")
    p.add_argument("--dataset-override", default=None)
    p.add_argument("--checkpoint", default=paths.BASE_CHECKPOINT)
    p.add_argument("--save-dir", default="./save")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-bins", type=int, default=51)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--mask-ratio", type=float, default=0.0,
                   help="0.0 for the classification-only objective used in the paper")
    p.add_argument("--schedule-ratio", type=float, default=0.9)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--no-fast-transformer", action="store_true")
    p.add_argument("--no-raw-payload", action="store_true",
                   help="skip embedding the raw test matrix in the .npz (smaller output)")
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    names = (TASK_DATASETS["annotation"] if args.dataset == "all"
             else [s.strip() for s in args.dataset.split(",")])

    summary = {}
    for name in names:
        summary.update(run_one(name, args))

    print("\n===== cell annotation summary =====")
    for name, r in summary.items():
        print(f"{name:8s} acc={r['test/accuracy']:.4f}  macroF1={r['test/macro_f1']:.4f}  "
              f"PR-AUC={r['test/pr_auc']:.4f}  ROC-AUC={r['test/roc_auc']:.4f}")


if __name__ == "__main__":
    main()
