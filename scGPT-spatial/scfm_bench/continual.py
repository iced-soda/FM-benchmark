"""Short self-supervised continual pretraining of the scGPT-spatial checkpoint.

Two epochs of masked value completion (the MVC head) on the target dataset,
starting from the released checkpoint. Nothing about the architecture changes:
`args.json` is read from the base checkpoint and reused verbatim, and the gene
statistics file is copied across so `embed_data` behaves identically afterwards.

Original scripts had two near-identical copies of this routine, one for
scRNA-seq (which took `gene_col`) and one for spatial (which hard-coded
`var_names`). They are merged here -- passing `gene_col="index"` reproduces the
spatial variant, since that also uppercased `var_names`.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .clustering import set_all_seeds

# Continuous-input convention. These must match between the tokenizer and the
# model, otherwise padding is silently treated as signal.
PAD_VALUE = -2
MASK_VALUE = -1
MAX_LEN = 3001


class _SeqDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def _make_loader(data_pt, batch_size, shuffle=False, drop_last=False, num_workers=0):
    if num_workers == 0:
        try:
            num_workers = min(len(os.sched_getaffinity(0)), max(1, batch_size // 2))
        except (AttributeError, OSError):
            num_workers = max(1, batch_size // 2)
    return DataLoader(_SeqDataset(data_pt), batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=num_workers, pin_memory=True)


def continual_pretrain(
    adata,
    base_model_dir,
    out_model_dir,
    gene_col: str = "index",
    seed: int = 42,
    mask_ratio: float = 0.20,
    n_bins: int = 51,
    batch_size: int = 64,
    lr: float = 1e-4,
    epochs: int = 2,
    amp: bool = True,
    use_fast_transformer: bool = True,
) -> str:
    """Adapt the checkpoint to `adata` for `epochs` epochs; return the new model dir.

    The best checkpoint is selected on held-out masked-token MSE (20% of cells).
    """
    from scgpt_spatial.model import TransformerModel
    from scgpt_spatial.preprocess import Preprocessor
    from scgpt_spatial.tokenizer import random_mask_value, tokenize_and_pad_batch
    from scgpt_spatial.tokenizer.gene_tokenizer import GeneVocab

    set_all_seeds(seed)

    base = Path(base_model_dir)
    model_file = base / "best_model.pt"
    vocab_file = base / "vocab.json"
    args_file = base / "args.json"
    stats_file = base / "all_dict_mean_std.csv"  # needed by embed_data later

    out = Path(out_model_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(vocab_file, out / "vocab.json")

    # ---------------- vocabulary + gene alignment ---------------- #
    vocab = GeneVocab.from_file(vocab_file)
    pad_token = "<pad>"
    for tok in [pad_token, "<cls>", "<eoc>"]:
        if tok not in vocab:
            vocab.append_token(tok)

    adata.obs_names_make_unique()
    if gene_col != "index" and gene_col in adata.var.columns:
        adata.var["gene_name"] = adata.var[gene_col].astype(str).str.upper()
    else:
        adata.var["gene_name"] = adata.var.index.astype(str).str.upper()

    if "count" in adata.layers:
        adata.X = adata.layers["count"].copy()

    adata.var["id_in_vocab"] = [1 if g in vocab else -1 for g in adata.var["gene_name"]]
    n_matched = int((adata.var["id_in_vocab"] >= 0).sum())
    print(f"[CP] {n_matched}/{adata.n_vars} genes matched the checkpoint vocabulary "
          f"(size {len(vocab)})", flush=True)
    adata = adata[:, adata.var["id_in_vocab"] >= 0].copy()

    genes = adata.var["gene_name"].tolist()
    gene_ids = np.array([vocab[g] for g in genes], dtype=int)

    # ---------------- preprocessing: 1e4 norm -> log1p -> 51 bins ---------------- #
    pre = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=True,
        result_log1p_key="X_log1p",
        subset_hvg=False,
        hvg_flavor="seurat_v3",
        binning=n_bins,
        result_binned_key="X_binned",
    )
    pre(adata, batch_key=None)

    binned = adata.layers["X_binned"]
    all_counts = binned.toarray() if issparse(binned) else binned

    train_data, valid_data = train_test_split(
        all_counts, test_size=0.2, random_state=seed, shuffle=True
    )

    def _tokenize(data):
        return tokenize_and_pad_batch(
            data, gene_ids, max_len=MAX_LEN, vocab=vocab, pad_token=pad_token,
            pad_value=PAD_VALUE, append_cls=True, include_zero_gene=False,
        )

    def _mask(tokenized):
        # NOTE: the mask is drawn once and reused for both epochs, matching the
        # run that produced the published numbers. Resampling per epoch would be
        # the more standard choice.
        inp_vals = random_mask_value(
            tokenized["values"], mask_ratio=mask_ratio,
            mask_value=MASK_VALUE, pad_value=PAD_VALUE,
        )
        return {"gene_ids": tokenized["genes"], "values": inp_vals,
                "target_values": tokenized["values"]}

    train_loader = _make_loader(_mask(_tokenize(train_data)), batch_size, shuffle=True)
    valid_loader = _make_loader(_mask(_tokenize(valid_data)), batch_size, shuffle=False)

    # ---------------- model ---------------- #
    with open(args_file) as f:
        cfg = json.load(f)

    model = TransformerModel(
        len(vocab),
        cfg["embsize"],
        cfg.get("nheads", cfg.get("nhead")),
        cfg["d_hid"],
        cfg["nlayers"],
        nlayers_cls=0,          # no classifier head during continual pretraining
        n_cls=1,
        vocab=vocab,
        dropout=cfg.get("dropout", 0.2),
        pad_token=pad_token,
        pad_value=PAD_VALUE,
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        num_batch_labels=1,
        input_emb_style="continuous",
        n_input_bins=n_bins,    # required by the API even for continuous inputs
        cell_emb_style="cls",
        mvc_decoder_style="inner product",
        ecs_threshold=0.0,
        explicit_zero_prob=False,
        use_fast_transformer=use_fast_transformer,
        fast_transformer_backend="flash",
        pre_norm=cfg.get("pre_norm", False),
    )

    try:
        model.load_state_dict(torch.load(model_file, map_location="cpu"))
    except (RuntimeError, KeyError):
        # Shape-compatible subset load, for checkpoints that carry extra heads.
        model_dict = model.state_dict()
        pretrained = torch.load(model_file, map_location="cpu")
        pretrained = {k: v for k, v in pretrained.items()
                      if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained)
        model.load_state_dict(model_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    def run_epoch(loader, train=True):
        model.train(train)
        total, n = 0.0, 0
        for batch in loader:
            gene_ids_b = batch["gene_ids"].to(device)
            inp_vals = batch["values"].to(device)
            tgt_vals = batch["target_values"].to(device)
            src_key_padding_mask = gene_ids_b.eq(vocab[pad_token])

            with torch.cuda.amp.autocast(enabled=amp):
                out_dict = model(
                    gene_ids_b, inp_vals,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=False, MVC=True, ECS=False, do_sample=False,
                )
                masked_pos = inp_vals.eq(MASK_VALUE)
                loss_mvc = ((out_dict["mvc_output"][masked_pos]
                             - tgt_vals[masked_pos]) ** 2).mean()

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss_mvc).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            total += loss_mvc.item() * gene_ids_b.size(0)
            n += gene_ids_b.size(0)
        return total / max(1, n)

    best_val, best_state = float("inf"), None
    for ep in range(1, epochs + 1):
        tr = run_epoch(train_loader, train=True)
        vl = run_epoch(valid_loader, train=False)
        print(f"[CP] epoch {ep}/{epochs}  train_mvc={tr:.4f}  valid_mvc={vl:.4f}",
              flush=True)
        if vl < best_val:
            best_val = vl
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        best_state = model.state_dict()

    torch.save(best_state, out / "best_model.pt")
    with open(out / "args.json", "w") as f:
        json.dump(cfg, f, indent=2)
    if stats_file.exists():
        shutil.copyfile(stats_file, out / "all_dict_mean_std.csv")

    print(f"[CP] adapted checkpoint -> {out} (best valid MVC MSE {best_val:.4f})",
          flush=True)
    return str(out)
