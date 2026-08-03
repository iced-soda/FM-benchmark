"""Dataset registry for the scGPT-spatial benchmark.

One entry per dataset. Everything that used to be copy-pasted per dataset --
file path, label column, how gene symbols are matched to the checkpoint
vocabulary, and the hand-tuned Leiden resolution window -- lives here, so the
task scripts are dataset-agnostic.

Registry keys are the short names used throughout the manuscript:

    ATAA   ascending thoracic aortic aneurysm scRNA-seq   (GSE155468)
    HS     human splenic dendritic cells, scRNA-seq
    BMMC   NeurIPS 2021 bone marrow mononuclear CITE-seq  (GSE194122)
    BC     HBCA1, human breast cancer Visium              (spatial)
    MHS    MHPC, mouse hippocampus Slide-seqV2            (spatial)
    Emb    MOSTA E9.5 E1S1 mouse embryo Stereo-seq        (spatial)
    SEAAD  Seattle Alzheimer's MTG MERFISH                (spatial, annotation only)

`gene_col` semantics, kept exactly as in the original scripts:
    "index"      -> pass adata.var index straight to embed_data, no case change
    "gene_name"  -> write an UPPERCASED copy of var_names into var["gene_name"]
                    first (case-insensitive vocabulary match)
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import scanpy as sc

from .paths import DATA_ROOT

# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

DATASETS: Dict[str, Dict[str, Any]] = {
    # ---------------- scRNA-seq ---------------- #
    "ATAA": {
        "long_name": "Ascending thoracic aortic aneurysm (GSE155468)",
        "source": "GEO GSE155468",
        "modality": "sc",
        "h5ad": "ATAA/gse155468.h5ad",
        "label_col": "celltype",
        "gene_col": "index",
        "obs_names_make_unique": True,
        # Leiden sweeps: (start, end, increment). Swept until n_clusters == n_labels.
        "zero_shot_res": (0.1, 1.0, 0.01),
        "continual_res": (0.07, 1.0, 0.001),
        "annotation": {
            "train": "train-test_splits/ATAA/ATAA_train.h5ad",
            "test": "train-test_splits/ATAA/ATAA_test.h5ad",
            "label_aliases": ["cell_type"],
        },
        "deg": {"var_names_make_unique": True},
    },
    "HS": {
        "long_name": "Human splenic dendritic cells",
        "source": "Human splenic DC study (see manuscript Methods)",
        "modality": "sc",
        "h5ad": "HS/human_spleen_updated.h5ad",
        "label_col": "cell_type",
        "gene_col": "index",
        "obs_names_make_unique": True,
        "zero_shot_res": (0.1, 1.0, 0.01),
        "continual_res": (0.1, 1.0, 0.01),
        "annotation": {
            "train": "train-test_splits/HS/HS_train.h5ad",
            "test": "train-test_splits/HS/HS_test.h5ad",
            "label_aliases": ["cell_type"],
        },
        "deg": {"var_names_make_unique": False},
    },
    "BMMC": {
        "long_name": "NeurIPS 2021 bone marrow mononuclear cells, CITE-seq (GSE194122)",
        "source": "GEO GSE194122, CITE-seq subset",
        "modality": "sc",
        "h5ad": "SRT/GSE194122_cite_BMMC_processed.h5ad",
        "label_col": "cell_type",
        "gene_col": "gene_name",
        # X holds normalised values; raw counts are in layers["count"].
        "use_count_layer": True,
        "zero_shot_use_count_layer": True,
        "zero_shot_res": (2.1, 2.5, 0.01),
        "continual_res": (1.5, 2.5, 0.01),
        "annotation": {
            "train": "train-test_splits/BMMC/BMMC_train.h5ad",
            "test": "train-test_splits/BMMC/BMMC_test.h5ad",
            "label_aliases": ["cell_type"],
        },
        "deg": {"var_names_make_unique": True},
    },
    # ---------------- spatial transcriptomics ---------------- #
    "BC": {
        "long_name": "HBCA1 human breast cancer, 10x Visium",
        "source": "BenchmarkST data resource",
        "modality": "spatial",
        "h5ad": "SRT/BC/breast_cancer.h5ad",
        "label_col": "fine_annot_type",
        "gene_col": "gene_name",
        # Zero-shot embedded the top-3000 HVGs only; continual pretraining used all genes.
        "zero_shot_hvg": 3000,
        "zero_shot_res": (2.0, 3.0, 0.01),
        "continual_res": (1.6, 1.9, 0.01),
        # The adapted checkpoint from the published run is named ..._BS, not ..._BC.
        "continual_checkpoint": "scGPT_spatial_v1_BS",
        "deg": {"var_names_make_unique": True},
    },
    "MHS": {
        "long_name": "MHPC mouse hippocampus, Slide-seqV2",
        "source": "BenchmarkST data resource (Broad Institute Slide-seqV2)",
        "modality": "spatial",
        "h5ad": "SRT/mouse_hyppocampus_slideseqv2/sshippo.h5ad",
        "label_col": "cluster",
        "gene_col": "gene_name",
        "zero_shot_res": (0.1, 1.0, 0.01),
        "continual_res": (0.2, 1.0, 0.01),
        "deg": {"var_names_make_unique": True},
    },
    "Emb": {
        "long_name": "MOSTA E9.5 E1S1 mouse embryo, Stereo-seq",
        "source": "BenchmarkST data resource (MOSTA)",
        "modality": "spatial",
        "h5ad": "SRT/Embryo/E9.5_E1S1.MOSTA.h5ad",
        "label_col": "annotation",
        "gene_col": "gene_name",
        "use_count_layer": True,
        # The original zero-shot run embedded adata.X as stored and did NOT swap in
        # layers["count"], whereas continual pretraining did. Kept as-is so the
        # published zero-shot numbers reproduce; see README "Known inconsistencies".
        "zero_shot_use_count_layer": False,
        "zero_shot_res": (0.9, 1.3, 0.01),
        "continual_res": (0.1, 1.0, 0.01),
        "deg": {"var_names_make_unique": True},
    },
    "SEAAD": {
        "long_name": "Seattle Alzheimer's Disease Brain Cell Atlas, MTG MERFISH",
        "source": "SEA-AD, MTG MERFISH, 140-gene panel",
        "modality": "spatial",
        # Single file carrying a pre-computed section-wise split in obs["split"],
        # so that no tissue section appears in both train and test.
        "h5ad": "adata_with_split_SEAAD.h5ad",
        "label_col": "celltype",
        "gene_col": "gene_name",
        "annotation": {
            "split_h5ad": "adata_with_split_SEAAD.h5ad",
            "split_key": "split",
            "label_aliases": ["Subclass"],
            # SEAAD test sets are large; skipping the raw-matrix snapshot keeps the
            # result bundle to a sane size.
            "save_raw_payload": False,
        },
    },
}

#: Datasets used by each task in the manuscript.
TASK_DATASETS = {
    "zero_shot": ["ATAA", "HS", "BMMC", "BC", "MHS", "Emb"],
    "continual": ["ATAA", "HS", "BMMC", "BC", "MHS", "Emb"],
    "annotation": ["ATAA", "HS", "BMMC", "SEAAD"],
    "deg": ["ATAA", "HS", "BMMC", "BC", "Emb"],
    "perturbation": ["adamson", "norman", "replogle"],
}

#: GEARS Perturb-seq datasets. `gears_name` is the directory under SCFM_PERTURB_ROOT.
PERTURB_DATASETS = {
    "adamson": {
        "gears_name": "adamson",
        "long_name": "Adamson 2016 UPR Perturb-seq, K562",
        "example_perts": ["KCTD16+ctrl", "DAD1+ctrl"],
    },
    "norman": {
        "gears_name": "norman",
        "long_name": "Norman 2019 single + combinatorial Perturb-seq, K562",
        "example_perts": ["SAMD1+ZBTB1"],
    },
    "replogle": {
        # NOTE: the essential-gene subset, not the full genome-scale screen.
        "gears_name": "replogle_k562_essential",
        "long_name": "Replogle 2022 CRISPRi Perturb-seq, K562 essential subset",
        "example_perts": ["DAD1+ctrl"],
    },
}


# --------------------------------------------------------------------------- #
# Accessors
# --------------------------------------------------------------------------- #

def list_datasets() -> list:
    return sorted(DATASETS)


def get_dataset(name: str, overrides: Optional[str] = None) -> Dict[str, Any]:
    """Return the registry entry for `name`, with absolute paths resolved.

    `overrides` is an optional path to a JSON file, or an inline JSON string,
    merged on top of the registry entry. This is how you point the scripts at a
    dataset that is not in the registry, e.g.

        --dataset-override '{"h5ad": "/path/new.h5ad", "label_col": "ct",
                             "modality": "sc", "zero_shot_res": [0.1, 2.0, 0.01]}'
    """
    cfg = deepcopy(DATASETS.get(name, {}))
    cfg.setdefault("name", name)

    if overrides:
        text = overrides.strip()
        if not text.startswith("{"):
            text = Path(text).read_text()
        cfg.update(json.loads(text))

    if not cfg.get("h5ad") and not cfg.get("annotation"):
        raise KeyError(
            f"Unknown dataset {name!r}. Known: {', '.join(list_datasets())}. "
            "Pass --dataset-override to describe a new one."
        )

    # Resolve relative paths against DATA_ROOT.
    for key in ("h5ad",):
        if cfg.get(key) and not Path(cfg[key]).is_absolute():
            cfg[key] = str(DATA_ROOT / cfg[key])
    if "annotation" in cfg:
        for key in ("train", "test", "split_h5ad"):
            val = cfg["annotation"].get(key)
            if val and not Path(val).is_absolute():
                cfg["annotation"][key] = str(DATA_ROOT / val)

    # Resolution windows may arrive from JSON as lists.
    for key in ("zero_shot_res", "continual_res"):
        if key in cfg:
            cfg[key] = tuple(float(x) for x in cfg[key])

    return cfg


def load_adata(cfg: Dict[str, Any], hvg: Optional[int] = None,
               use_count_layer: Optional[bool] = None):
    """Load and harmonise one dataset exactly as the original per-dataset scripts did.

    Steps, in the original order:
      1. read h5ad
      2. optional HVG subset (seurat_v3) -- zero-shot BC only
      3. obs/var name de-duplication
      4. build var["gene_name"] as uppercase var_names when gene_col == "gene_name"
      5. move layers["count"] into X when the file stores raw counts in a layer

    `use_count_layer=None` falls back to the registry default; pass an explicit
    bool where a task deviates from it.
    """
    h5ad = Path(cfg["h5ad"])
    if not h5ad.exists():
        raise FileNotFoundError(
            f"{h5ad} not found.\n"
            f"Dataset {cfg.get('name')} comes from: "
            f"{cfg.get('source', 'see manuscript Methods')}.\n"
            "Download it, then either place it at that path, set SCFM_DATA_ROOT, "
            "or pass --dataset-override with the real location."
        )
    adata = sc.read_h5ad(h5ad)

    if hvg:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=hvg)
        adata = adata[:, adata.var["highly_variable"]].copy()

    if adata.is_view:
        adata = adata.copy()

    if cfg.get("obs_names_make_unique"):
        adata.obs_names_make_unique()

    if cfg.get("gene_col", "index") == "gene_name":
        adata.var_names_make_unique()
        adata.var["gene_name"] = adata.var_names.astype(str).str.upper()

    if use_count_layer is None:
        use_count_layer = cfg.get("use_count_layer", False)
    if use_count_layer and "count" in adata.layers:
        adata.X = adata.layers["count"].copy()

    return adata
