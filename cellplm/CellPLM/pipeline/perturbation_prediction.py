import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Union, Tuple

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.sparse import issparse

from ..utils.data import XDict
from . import Pipeline, load_pretrain  # <- matches your pipeline/__init__.py snippet

# GEARS
from gears import PertData
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis


PerturbationPredictionDefaultModelConfig = {
    # keep consistent with other pipelines
    "drop_node_rate": 0.0,
    "dec_layers": 1,
    "model_dropout": 0.1,
    "mask_node_rate": 0.0,
    "mask_feature_rate": 0.0,
    "dec_mod": "mlp",
    "latent_mod": "ae",
    "head_type": "perturbation_prediction",
    "max_batch_size": 4096,
    # NOTE: out_dim will be set dynamically to number of genes after preprocess
}

PerturbationPredictionDefaultPipelineConfig = {
    "es": 200,
    "lr": 5e-4,
    "wd": 1e-7,
    "scheduler": "plat",
    "epochs": 2000,
    "max_eval_batch_size": 100000,
    "hvg": 0,  # usually keep all overlapping genes for GEARS
    "patience": 25,
    "workers": 0,

    "perturb_value": -100.0,  # paper: set perturbed genes to -100

    # fallback split field if adata_train/val/test not available
    "split_field": "split",
    "train_split": "train",
    "valid_split": "val",
    "test_split": "test",
}


def _to_dense(X):
    return X.toarray() if issparse(X) else np.asarray(X)


def _parse_condition_to_genes(cond: str) -> List[str]:
    cond = str(cond)
    if cond == "ctrl":
        return []
    parts = cond.split("+")
    parts = [p for p in parts if p not in ("ctrl", "control", "")]
    return parts


def _apply_perturb_value(
    x: np.ndarray,
    conds: np.ndarray,
    gene_names: List[str],
    perturb_value: float,
) -> np.ndarray:
    gene2idx = {g: i for i, g in enumerate(gene_names)}
    x_in = x.copy()
    for i, c in enumerate(conds):
        for g in _parse_condition_to_genes(c):
            j = gene2idx.get(g, None)
            if j is not None:
                x_in[i, j] = float(perturb_value)
    return x_in


class _PerturbChunkDataset(torch.utils.data.IterableDataset):
    """
    Iterable dataset yielding chunks (CellPLM style: DataLoader(batch_size=None)).

    Outputs per-chunk dict with:
      - x_seq: [B,G] float32
      - input_mask: [B,G] float32 ones
      - label: sparse [B,G] float32
      - condition: np.array[str] length B
      - (optional) split: pandas Series slice if provided
      - gene_list: list[str] length G
    """
    def __init__(
        self,
        adata: ad.AnnData,
        max_batch_size: int,
        perturb_value: float,
        split_field: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.adata = adata
        self.max_batch_size = int(max_batch_size)
        self.perturb_value = float(perturb_value)
        self.split_field = split_field
        self.seed = seed

        assert "condition" in self.adata.obs, "GEARS AnnData must have obs['condition']."

        self.gene_list = self.adata.var.index.astype(str).tolist()

        self._X = _to_dense(self.adata.X).astype(np.float32)
        conds = self.adata.obs["condition"].astype(str).values
        self._X_in = _apply_perturb_value(self._X, conds, self.gene_list, self.perturb_value).astype(np.float32)

        self._n = self.adata.n_obs
        self._split = None
        if split_field is not None:
            if split_field not in self.adata.obs:
                raise KeyError(f"split_field='{split_field}' not found in adata.obs")
            self._split = self.adata.obs[split_field]

    def __iter__(self):
        # IMPORTANT: IterableDataset cannot be shuffled by DataLoader; shuffle here.
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
            idx = rng.permutation(self._n)
        else:
            idx = np.random.permutation(self._n)

        for s in range(0, self._n, self.max_batch_size):
            cur = idx[s:s + self.max_batch_size]
            x_seq = torch.from_numpy(self._X_in[cur])     # [B,G]
            y = torch.from_numpy(self._X[cur])            # [B,G]
            input_mask = torch.ones_like(x_seq)

            out = {
                "x_seq": x_seq,
                "input_mask": input_mask,
                "label": y.to_sparse(),
                "gene_list": self.gene_list,
                "condition": self.adata.obs["condition"].iloc[cur].astype(str).values,
            }
            if self._split is not None:
                out["split"] = self._split.iloc[cur]
            yield out


def _make_loss_mask(data_split, split_value: Optional[str], B: int) -> torch.Tensor:
    if split_value is None:
        return torch.ones(B, dtype=torch.bool)
    if data_split is None:
        raise ValueError("split_value is set but data_split is None (no split field provided).")
    mask_np = (data_split.to_numpy() == split_value)
    return torch.as_tensor(mask_np, dtype=torch.bool)


@torch.no_grad()
def _inference_regression(
    model,
    dataloader,
    device,
    batch_size: int,
    split_value: Optional[str] = None,
):
    model.eval()
    epoch_loss = []
    pred = []
    truth = []
    pert_cat = []

    for _, data_dict in enumerate(dataloader):
        B = data_dict["x_seq"].shape[0]
        if split_value is not None:
            if "split" not in data_dict:
                raise KeyError("Requested split inference but dataset did not provide 'split'.")
            if np.sum(data_dict["split"] == split_value) == 0:
                continue

        idx = torch.arange(B)
        split_series = data_dict.get("split", None)
        loss_mask_full = _make_loss_mask(split_series, split_value, B)

        for j in range(0, B, batch_size):
            cur = idx[j:] if (B - j < batch_size) else idx[j:j + batch_size]

            input_dict = {
                "x_seq": data_dict["x_seq"].index_select(0, cur).to(device),
                "label": data_dict["label"].index_select(0, cur).to(device),
                "input_mask": data_dict["input_mask"].index_select(0, cur).to(device),
                "loss_mask": loss_mask_full.index_select(0, cur).to(device),
            }

            x_dict = XDict(input_dict)
            out_dict, loss = model(x_dict, data_dict["gene_list"])

            pred.append(out_dict["pred"].detach().cpu())
            truth.append(input_dict["label"].to_dense().detach().cpu())
            pert_cat.extend(list(np.array(data_dict["condition"])[cur.cpu().numpy()]))

            epoch_loss.append(float(loss.item()))

    pred = torch.cat(pred, dim=0) if len(pred) else torch.empty((0, 0))
    truth = torch.cat(truth, dim=0) if len(truth) else torch.empty((0, 0))
    return {
        "pred": pred,
        "truth": truth,
        "loss": float(np.mean(epoch_loss)) if len(epoch_loss) else None,
        "pert_cat": np.array(pert_cat, dtype=object),
    }


def _build_test_res_for_gears(
    pert_data: PertData,
    pred: np.ndarray,
    truth: np.ndarray,
    pert_cat: np.ndarray,
) -> Dict:
    """
    Build a GEARS-compatible test_res dict.

    IMPORTANT: pred/truth columns match pert_data.adata.var.index (after preprocess),
    so we index DE genes using var.index (not var['gene_name']).
    """
    adata = pert_data.adata

    if "condition_name" in adata.obs.columns:
        cond2name = dict(adata.obs[["condition", "condition_name"]].astype(str).values)
    else:
        uniq = adata.obs["condition"].astype(str).unique()
        cond2name = {c: c for c in uniq}

    gene_names = adata.var.index.astype(str).tolist()
    gene2idx = {g: i for i, g in enumerate(gene_names)}

    pred_de = []
    truth_de = []
    top_de = adata.uns.get("top_non_dropout_de_20", None)

    if top_de is not None:
        for i, cond in enumerate(pert_cat):
            cname = cond2name.get(str(cond), None)
            if cname is None:
                continue
            de_genes = top_de.get(cname, None)
            if de_genes is None:
                continue
            de_idx = [gene2idx[g] for g in de_genes if g in gene2idx]
            if len(de_idx) == 0:
                continue
            pred_de.append(pred[i, de_idx])
            truth_de.append(truth[i, de_idx])

    pred_de = np.vstack(pred_de).astype(float) if len(pred_de) else np.zeros((0, 0), dtype=float)
    truth_de = np.vstack(truth_de).astype(float) if len(truth_de) else np.zeros((0, 0), dtype=float)

    return {
        "pert_cat": np.array(pert_cat, dtype=object),
        "pred": pred.astype(float),
        "truth": truth.astype(float),
        "pred_de": pred_de.astype(float),
        "truth_de": truth_de.astype(float),
    }


class PerturbationPredictionPipeline(Pipeline):
    """
    CellPLM-style perturbation prediction pipeline, but instantiates the model *after*
    we know the processed gene list length so `out_dim` matches G exactly.
    """

    def __init__(
        self,
        pretrain_prefix: str,
        overwrite_config: dict = None,
        pretrain_directory: str = "./ckpt",
    ):
        # We call parent init to keep structure, but we will re-load model with correct out_dim in fit().
        cfg = PerturbationPredictionDefaultModelConfig.copy()
        if overwrite_config:
            cfg.update(overwrite_config)
        super().__init__(pretrain_prefix, cfg, pretrain_directory)
        self._pretrain_prefix = pretrain_prefix
        self._pretrain_directory = pretrain_directory
        self._base_overwrite_config = cfg

    def _reload_model_with_out_dim(self, out_dim: int):
        cfg = dict(self._base_overwrite_config)
        cfg["out_dim"] = int(out_dim)
        cfg["head_type"] = "perturbation_prediction"
        self.model = load_pretrain(self._pretrain_prefix, cfg, self._pretrain_directory)

    def _get_split_adatas(self, pert_data: PertData, config: dict) -> Tuple[ad.AnnData, ad.AnnData, ad.AnnData, bool]:
        """
        Prefer GEARS unseen-pert splits if available: adata_train/adata_val/adata_test.
        Otherwise fall back to pert_data.adata with obs[split_field].
        Returns (train, val, test, using_split_field_fallback).
        """
        if hasattr(pert_data, "adata_train") and pert_data.adata_train is not None \
           and hasattr(pert_data, "adata_val") and pert_data.adata_val is not None \
           and hasattr(pert_data, "adata_test") and pert_data.adata_test is not None:
            return pert_data.adata_train.copy(), pert_data.adata_val.copy(), pert_data.adata_test.copy(), False

        # fallback: single adata with split field
        adata = pert_data.adata.copy()
        sf = config["split_field"]
        if sf is None or sf not in adata.obs:
            raise ValueError(
                "PertData does not provide adata_train/val/test and no valid split_field found in pert_data.adata.obs."
            )
        return adata, adata, adata, True

    def fit(
        self,
        pert_data: PertData,
        train_config: dict = None,
        covariate_fields: List[str] = None,
        ensembl_auto_conversion: bool = True,
        device: Union[str, torch.device] = "cpu",
    ):
        config = PerturbationPredictionDefaultPipelineConfig.copy()
        if train_config:
            config.update(train_config)

        assert not self.fitted, (
            "Current pipeline is already fitted and does not support continual training. "
            "Please initialize a new pipeline."
        )

        adata_train, adata_val, adata_test, split_fallback = self._get_split_adatas(pert_data, config)

        # preprocess train first (this sets self.gene_list in Pipeline.common_preprocess)
        adata_train = self.common_preprocess(adata_train, config["hvg"], covariate_fields, ensembl_auto_conversion)
        adata_val = self.common_preprocess(adata_val, config["hvg"], covariate_fields, ensembl_auto_conversion)
        adata_test = self.common_preprocess(adata_test, config["hvg"], covariate_fields, ensembl_auto_conversion)

        # set pert_data.adata to processed version for evaluation functions
        pert_data.adata = adata_test if not split_fallback else adata_train

        G = adata_train.shape[1]
        print(f"After filtering: train genes={G} | val genes={adata_val.shape[1]} | test genes={adata_test.shape[1]}")

        # reload pretrained model with correct out_dim (=G) and correct head_type
        self._reload_model_with_out_dim(out_dim=G)
        self.model.to(device)

        max_bs = getattr(self.model, "max_batch_size", self._base_overwrite_config.get("max_batch_size", 4096))

        train_ds = _PerturbChunkDataset(
            adata=adata_train,
            max_batch_size=max_bs,
            perturb_value=config["perturb_value"],
            split_field=(config["split_field"] if split_fallback else None),
        )
        val_ds = _PerturbChunkDataset(
            adata=adata_val,
            max_batch_size=max_bs,
            perturb_value=config["perturb_value"],
            split_field=(config["split_field"] if split_fallback else None),
        )

        # IterableDataset -> shuffle MUST be False
        train_loader = DataLoader(train_ds, batch_size=None, shuffle=False, num_workers=config["workers"])
        val_loader = DataLoader(val_ds, batch_size=None, shuffle=False, num_workers=config["workers"])

        optim = torch.optim.AdamW(
            [
                {"params": list(self.model.embedder.parameters()), "lr": config["lr"] * 0.1, "weight_decay": 1e-10},
                {
                    "params": list(self.model.encoder.parameters())
                              + list(self.model.head.parameters())
                              + list(self.model.latent.parameters()),
                    "lr": config["lr"],
                    "weight_decay": config["wd"],
                },
            ]
        )
        scheduler = (
            ReduceLROnPlateau(optim, "min", patience=config["patience"], factor=0.95)
            if config["scheduler"] == "plat"
            else None
        )

        best_dict = None
        best_valid = float("inf")
        valid_curve = []
        final_epoch = -1

        for epoch in tqdm(range(config["epochs"])):
            self.model.train()
            epoch_loss = []

            # warmup like other pipelines
            if epoch < 30:
                for pg in optim.param_groups[1:]:
                    pg["lr"] = config["lr"] * (epoch + 1) / 30

            for _, data_dict in enumerate(train_loader):
                input_dict = {
                    "x_seq": data_dict["x_seq"],
                    "label": data_dict["label"],
                    "input_mask": data_dict["input_mask"],
                }

                if split_fallback:
                    train_mask = _make_loss_mask(data_dict.get("split", None), config["train_split"], input_dict["x_seq"].shape[0])
                else:
                    train_mask = torch.ones(input_dict["x_seq"].shape[0], dtype=torch.bool)
                input_dict["loss_mask"] = train_mask

                # move tensors to device
                for k in input_dict:
                    if torch.is_tensor(input_dict[k]):
                        input_dict[k] = input_dict[k].to(device)

                x_dict = XDict(input_dict)
                _, loss = self.model(x_dict, data_dict["gene_list"])

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                optim.step()

                if scheduler is not None:
                    scheduler.step(loss.item())

                epoch_loss.append(float(loss.item()))

            # validation
            val_split = config["valid_split"] if split_fallback else None
            valid_out = _inference_regression(
                self.model,
                val_loader,
                device=device,
                batch_size=config["max_eval_batch_size"],
                split_value=val_split,
            )
            valid_loss = valid_out["loss"]
            valid_curve.append(valid_loss)

            print(f"Epoch {epoch} | Train loss: {float(np.mean(epoch_loss)):.6f} | Valid loss: {valid_loss:.6f}")

            if valid_loss < best_valid:
                best_valid = valid_loss
                best_dict = deepcopy(self.model.state_dict())
                final_epoch = epoch

            if min(valid_curve) != min(valid_curve[-config["es"]:]):
                print(f"Early stopped. Best validation loss achieved at epoch {final_epoch}.")
                break

        assert best_dict is not None, "Best state dict was not stored."
        self.model.load_state_dict(best_dict)
        self.fitted = True
        return self

    @torch.no_grad()
    def predict(
        self,
        adata: ad.AnnData,
        inference_config: dict = None,
        covariate_fields: List[str] = None,
        ensembl_auto_conversion: bool = True,
        device: Union[str, torch.device] = "cpu",
        split_field: Optional[str] = None,
        target_split: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Predict on a provided AnnData (already GEARS split or any compatible adata).

        If split_field + target_split provided, only that split contributes loss_mask (and filtering in inference).
        """
        config = PerturbationPredictionDefaultPipelineConfig.copy()
        if inference_config:
            config.update(inference_config)

        assert self.fitted, "Perturbation prediction does not support zero-shot; fine-tune first."
        self.model.to(device)

        adata = self.common_preprocess(adata.copy(), config["hvg"], covariate_fields, ensembl_auto_conversion)

        max_bs = getattr(self.model, "max_batch_size", self._base_overwrite_config.get("max_batch_size", 4096))
        ds = _PerturbChunkDataset(
            adata=adata,
            max_batch_size=max_bs,
            perturb_value=config["perturb_value"],
            split_field=split_field,
        )
        loader = DataLoader(ds, batch_size=None, shuffle=False, num_workers=config["workers"])

        out = _inference_regression(
            self.model,
            loader,
            device=device,
            batch_size=config["max_eval_batch_size"],
            split_value=target_split,
        )

        return {
            "pred": out["pred"].cpu().numpy(),
            "truth": out["truth"].cpu().numpy(),
            "pert_cat": out["pert_cat"],
        }

    def score(
        self,
        pert_data: PertData,
        evaluation_config: dict = None,
        covariate_fields: List[str] = None,
        ensembl_auto_conversion: bool = True,
        device: Union[str, torch.device] = "cpu",
    ) -> Dict:
        """
        Evaluate on GEARS test set (preferred) or split_field fallback.
        Returns:
          - compute_metrics output
          - deeper_analysis output (or None if DE lists missing)
          - non_dropout_analysis output (or None if DE lists missing)
        """
        config = PerturbationPredictionDefaultPipelineConfig.copy()
        if evaluation_config:
            config.update(evaluation_config)

        adata_train, adata_val, adata_test, split_fallback = self._get_split_adatas(pert_data, config)

        # preprocess test to match model genes
        adata_test = self.common_preprocess(adata_test, config["hvg"], covariate_fields, ensembl_auto_conversion)
        pert_data.adata = adata_test  # ensure GEARS analysis sees same gene axis

        if split_fallback:
            pred_pack = self.predict(
                adata=adata_test,
                inference_config=config,
                covariate_fields=covariate_fields,
                ensembl_auto_conversion=ensembl_auto_conversion,
                device=device,
                split_field=config["split_field"],
                target_split=config["test_split"],
            )
        else:
            pred_pack = self.predict(
                adata=adata_test,
                inference_config=config,
                covariate_fields=covariate_fields,
                ensembl_auto_conversion=ensembl_auto_conversion,
                device=device,
                split_field=None,
                target_split=None,
            )

        test_res = _build_test_res_for_gears(
            pert_data=pert_data,
            pred=pred_pack["pred"],
            truth=pred_pack["truth"],
            pert_cat=pred_pack["pert_cat"],
        )

        test_metrics, test_pert_res = compute_metrics(test_res)

        if "top_non_dropout_de_20" in pert_data.adata.uns:
            deeper_res = deeper_analysis(pert_data.adata, test_res)
            non_dropout_res = non_dropout_analysis(pert_data.adata, test_res)
        else:
            deeper_res = None
            non_dropout_res = None

        return {
            "test_metrics": test_metrics,
            "test_pert_res": test_pert_res,
            "deeper_analysis": deeper_res,
            "non_dropout_analysis": non_dropout_res,
        }
