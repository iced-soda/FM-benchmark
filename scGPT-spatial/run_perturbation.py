#!/usr/bin/env python
"""Task 4 -- Perturb-seq perturbation prediction.

Fine-tunes scGPT-spatial as a GEARS-style perturbation generator and evaluates
post-perturbation expression prediction on the held-out test split.

    # train, then evaluate, on one dataset
    python run_perturbation.py --dataset adamson --mode train
    # re-evaluate a saved checkpoint without retraining
    python run_perturbation.py --dataset adamson --mode eval
    python run_perturbation.py --dataset all --mode eval

Datasets (all K562 Perturb-seq): adamson, norman, replogle.

Reported metrics. Global MSE and Pearson are reconstruction sanity checks --
most genes barely move after a perturbation, so they saturate near 1.0. The
biological readout is Pearson Delta DE, supported by Pearson Delta and MSE DE,
which score the change relative to control over the genes the perturbation
actually affects.

Manuscript: "Perturbation prediction separates expression reconstruction from
functional response modelling" (Figure 6, Table 5).
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import torch

warnings.filterwarnings("ignore")

from gears import PertData
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis

from scfm_bench import paths
from scfm_bench.datasets import PERTURB_DATASETS, TASK_DATASETS

paths.project_on_path()
from scgpt_spatial.loss import masked_mse_loss  # noqa: E402
from scgpt_spatial.model import TransformerGenerator  # noqa: E402
from scgpt_spatial.tokenizer.gene_tokenizer import GeneVocab  # noqa: E402
from scgpt_spatial.utils import (  # noqa: E402
    compute_perturbation_metrics,
    map_raw_id_to_vocab_id,
    set_seed,
)

PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]
PAD_VALUE = 0
PERT_PAD_ID = 0
INCLUDE_ZERO_GENE = "all"
MAX_SEQ_LEN = 1536

# Objectives: masked expression reconstruction only.
CLS = CCE = MVC = ECS = False

# Only the encoder stack is transferred from the pretrained checkpoint; the
# perturbation decoder is trained from scratch.
TRANSFER_PREFIXES = ["encoder", "value_encoder", "transformer_encoder"]


def load_pert_data(gears_name: str, batch_size: int, eval_batch_size: int,
                   split: str = "simulation", seed: int = 42) -> PertData:
    """Load a pre-downloaded GEARS dataset and its baked simulation split."""
    pert_data = PertData(str(paths.PERTURB_ROOT))
    # Pass data_path (not dataset_name) so GEARS never attempts a download.
    pert_data.load(data_path=str(paths.PERTURB_ROOT / gears_name))
    pert_data.prepare_split(split=split, seed=seed)
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)
    return pert_data


def build_model(pert_data, checkpoint_dir: Path, device, transfer_only: bool,
                use_fast_transformer: bool = True):
    """Instantiate TransformerGenerator and load weights from `checkpoint_dir`."""
    vocab = GeneVocab.from_file(checkpoint_dir / "vocab.json")
    for s in SPECIAL_TOKENS:
        if s not in vocab:
            vocab.append_token(s)

    genes = pert_data.adata.var["gene_name"].tolist()
    n_matched = sum(1 for g in genes if g in vocab)
    print(f"match {n_matched}/{len(genes)} genes in vocabulary of size {len(vocab)}",
          flush=True)

    with open(checkpoint_dir / "args.json") as f:
        cfg = json.load(f)

    vocab.set_default_index(vocab[PAD_TOKEN])
    gene_ids = np.array([vocab[g] if g in vocab else vocab[PAD_TOKEN] for g in genes],
                        dtype=int)

    model = TransformerGenerator(
        len(vocab),
        cfg["embsize"],
        cfg["nheads"],
        cfg["d_hid"],
        cfg["nlayers"],
        nlayers_cls=cfg["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=0,
        pad_token=PAD_TOKEN,
        pad_value=PAD_VALUE,
        pert_pad_id=PERT_PAD_ID,
        use_fast_transformer=use_fast_transformer,
    )

    ckpt = torch.load(checkpoint_dir / "best_model.pt", map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]

    if transfer_only:
        model_dict = model.state_dict()
        filtered = {k: v for k, v in ckpt.items()
                    if any(k.startswith(pref) for pref in TRANSFER_PREFIXES)}
        print(f"transferring {len(filtered)} pretrained tensors "
              f"({', '.join(TRANSFER_PREFIXES)})", flush=True)
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(ckpt, strict=True)

    return model.to(device), vocab, gene_ids


def train_one_epoch(model, loader, optimizer, scaler, gene_ids, device, epoch,
                    amp=True, log_interval=100):
    criterion = masked_mse_loss
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for batch, batch_data in enumerate(loader):
        batch_data.to(device)
        batch_size = batch_data.y.shape[0]
        n_genes = batch_data.y.shape[1]

        x = batch_data.x.to(device)
        ori_gene_values = x[:, 0].view(batch_size, n_genes)

        # Perturbation flags come from pert_idx rather than x[:, 1]: some GEARS
        # versions no longer populate the second feature column.
        pert_flags = torch.zeros(batch_size, n_genes, dtype=torch.long, device=x.device)
        for i, idx in enumerate(batch_data.pert_idx):
            if idx is None:
                continue
            idx_tensor = torch.as_tensor(idx, device=x.device).long().view(-1)
            idx_tensor = idx_tensor[(idx_tensor >= 0) & (idx_tensor < n_genes)]
            if idx_tensor.numel() > 0:
                pert_flags[i, idx_tensor] = 1

        target_gene_values = batch_data.y.to(device)

        if INCLUDE_ZERO_GENE == "all":
            input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
        else:
            input_gene_ids = ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
        if len(input_gene_ids) > MAX_SEQ_LEN:
            input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[:MAX_SEQ_LEN]

        input_values = ori_gene_values[:, input_gene_ids]
        input_pert_flags = pert_flags[:, input_gene_ids]
        target_values = target_gene_values[:, input_gene_ids]

        mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
        mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)
        src_key_padding_mask = torch.zeros_like(input_values, dtype=torch.bool,
                                                device=device)

        with torch.cuda.amp.autocast(enabled=amp):
            output_dict = model(
                mapped_input_gene_ids, input_values, input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=CLS, CCE=CCE, MVC=MVC, ECS=ECS,
            )
            masked_positions = torch.ones_like(input_values, dtype=torch.bool)
            loss = criterion(output_dict["mlm_output"], target_values, masked_positions)

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
        if batch % log_interval == 0 and batch > 0:
            ms = (time.time() - start_time) * 1000 / log_interval
            print(f"| epoch {epoch:3d} | {batch:3d}/{len(loader):3d} batches | "
                  f"ms/batch {ms:5.2f} | mse {total_loss / log_interval:5.4f} |",
                  flush=True)
            total_loss = 0.0
            start_time = time.time()


@torch.no_grad()
def eval_perturb(loader, model, gene_ids, device) -> Dict:
    """Predicted vs observed post-perturbation expression, all genes and DE genes."""
    model.eval()
    model.to(device)
    pert_cat, pred, truth, pred_de, truth_de = [], [], [], [], []

    for batch in loader:
        batch.to(device)
        pert_cat.extend(batch.pert)
        p = model.pred_perturb(batch, include_zero_gene=INCLUDE_ZERO_GENE,
                               gene_ids=gene_ids)
        t = batch.y
        pred.extend(p.cpu())
        truth.extend(t.cpu())
        for i, de_idx in enumerate(batch.de_idx):
            pred_de.append(p[i, de_idx])
            truth_de.append(t[i, de_idx])

    return {
        "pert_cat": np.array(pert_cat),
        "pred": torch.stack(pred).detach().cpu().numpy().astype(float),
        "truth": torch.stack(truth).detach().cpu().numpy().astype(float),
        "pred_de": torch.stack(pred_de).detach().cpu().numpy().astype(float),
        "truth_de": torch.stack(truth_de).detach().cpu().numpy().astype(float),
    }


def subgroup_report(pert_data, test_res) -> Dict:
    """GEARS per-subgroup deeper / non-dropout analysis."""
    deeper_res = deeper_analysis(pert_data.adata, test_res)
    non_dropout_res = non_dropout_analysis(pert_data.adata, test_res)

    metrics = ["pearson_delta", "pearson_delta_de"]
    metrics_non_dropout = ["pearson_delta_top20_de_non_dropout",
                           "pearson_top20_de_non_dropout"]

    out = {}
    for name, pert_list in pert_data.subgroup["test_subgroup"].items():
        out[name] = {}
        for m in metrics:
            out[name][m] = float(np.mean([deeper_res[p][m] for p in pert_list]))
        for m in metrics_non_dropout:
            out[name][m] = float(np.mean([non_dropout_res[p][m] for p in pert_list]))

    for name, result in out.items():
        for m, val in result.items():
            print(f"test_{name}_{m}: {val}", flush=True)
    return out


def run_one(name: str, args) -> Dict:
    if name not in PERTURB_DATASETS:
        raise KeyError(f"Unknown perturbation dataset {name!r}. "
                       f"Known: {', '.join(PERTURB_DATASETS)}")
    spec = PERTURB_DATASETS[name]
    print(f"\n{'=' * 70}\n[perturbation:{args.mode}] {name}: {spec['long_name']}\n"
          f"{'=' * 70}", flush=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pert_data = load_pert_data(spec["gears_name"], args.batch_size,
                               args.eval_batch_size, args.split, args.seed)

    finetuned_dir = Path(args.finetuned_dir or (paths.CHECKPOINT_ROOT / name))

    if args.mode == "train":
        model, vocab, gene_ids = build_model(
            pert_data, Path(args.checkpoint), device, transfer_only=True,
            use_fast_transformer=not args.no_fast_transformer,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

        ctrl_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
        best_val_corr, best_model, patience = 0.0, None, 0

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            train_one_epoch(model, pert_data.dataloader["train_loader"], optimizer,
                            scaler, gene_ids, device, epoch, amp=args.amp,
                            log_interval=args.log_interval)

            val_res = eval_perturb(pert_data.dataloader["val_loader"], model,
                                   gene_ids, device)
            val_metrics = compute_perturbation_metrics(val_res, ctrl_adata)
            print(f"val_metrics at epoch {epoch}: {val_metrics}", flush=True)
            print(f"| end of epoch {epoch:3d} | time: {time.time() - epoch_start:5.2f}s |",
                  flush=True)

            if val_metrics["pearson"] > best_val_corr:
                best_val_corr = val_metrics["pearson"]
                best_model = copy.deepcopy(model)
                patience = 0
                print(f"Best model with score {best_val_corr:5.4f}", flush=True)
            else:
                patience += 1
                if patience >= args.early_stop:
                    print(f"Early stop at epoch {epoch}", flush=True)
                    break
            scheduler.step()

        model = best_model if best_model is not None else model

        finetuned_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), finetuned_dir / "best_model.pt")
        with open(Path(args.checkpoint) / "args.json") as f:
            base_cfg = json.load(f)
        with open(finetuned_dir / "args.json", "w") as f:
            json.dump(base_cfg, f, indent=2)
        vocab.save_json(finetuned_dir / "vocab.json")
        print(f"Saved fine-tuned model -> {finetuned_dir}", flush=True)
    else:
        # Evaluation reloads every weight, not just the transferred prefixes.
        model, vocab, gene_ids = build_model(
            pert_data, finetuned_dir, device, transfer_only=False,
            use_fast_transformer=not args.no_fast_transformer,
        )
        model.eval()
        print(f"Loaded fine-tuned model from {finetuned_dir}", flush=True)

    # ---------------- test ---------------- #
    test_res = eval_perturb(pert_data.dataloader["test_loader"], model, gene_ids, device)
    ctrl_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]

    scgpt_metrics = compute_perturbation_metrics(test_res, ctrl_adata)
    print(f"[scGPT metrics] {scgpt_metrics}", flush=True)

    gears_metrics, _ = compute_metrics(test_res)
    print(f"[GEARS metrics] {gears_metrics}", flush=True)

    subgroups = subgroup_report(pert_data, test_res)

    out_path = paths.CLUSTERS_DIR / f"{paths.MODEL_NAME}_perturbation_{name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"dataset": name, "gears_name": spec["gears_name"], "mode": args.mode,
         "scgpt_metrics": {k: float(v) for k, v in scgpt_metrics.items()},
         "gears_metrics": {k: float(v) for k, v in gears_metrics.items()},
         "subgroups": subgroups},
        indent=2))
    print(f"Saved perturbation metrics -> {out_path}", flush=True)

    return {name: scgpt_metrics}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   help="adamson | norman | replogle, comma-separated list, or 'all'")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--checkpoint", default=paths.BASE_CHECKPOINT,
                   help="pretrained checkpoint to transfer the encoder from (train mode)")
    p.add_argument("--finetuned-dir", default=None,
                   help="where the fine-tuned model is written/read "
                        "(default: <checkpoints>/<dataset>)")
    p.add_argument("--split", default="simulation")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--early-stop", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--no-fast-transformer", action="store_true")
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    names = (TASK_DATASETS["perturbation"] if args.dataset == "all"
             else [s.strip() for s in args.dataset.split(",")])

    summary = {}
    for name in names:
        summary.update(run_one(name, args))

    print("\n===== perturbation summary (scGPT metrics) =====")
    for name, m in summary.items():
        print(f"{name:10s} " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))


if __name__ == "__main__":
    main()
