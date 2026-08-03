"""Loading and harmonisation for the supervised cell type annotation task.

Two dataset layouts are supported:

  * pre-split files -- ``train``/``test`` h5ad written by the shared 80:20
    stratified split, so every model in the benchmark sees the same test cells
    (ATAA, HS, BMMC);
  * single file with a split column -- SEAAD, where the split is section-wise
    rather than cell-wise, so no tissue section straddles train and test.

Both end at the same place: one concatenated AnnData carrying ``str_batch``
("0" = train, "1" = test), integer ``celltype_id`` codes, and an untouched copy
of the test split for downstream visualisation.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import scanpy as sc


def load_annotation_splits(cfg: Dict[str, Any]) -> Tuple:
    """Return ``(adata_concat, adata_test_raw, id2type, num_types)``."""
    ann = cfg.get("annotation")
    if not ann:
        raise KeyError(
            f"Dataset {cfg.get('name')!r} has no 'annotation' block. Add train/test "
            "paths (or split_h5ad + split_key) to the registry, or pass "
            "--dataset-override."
        )

    if ann.get("split_h5ad"):
        master = sc.read(ann["split_h5ad"])
        split_key = ann.get("split_key", "split")
        adata = master[master.obs[split_key] == "train"].copy()
        adata_test = master[master.obs[split_key] == "test"].copy()
    else:
        adata = sc.read(ann["train"])
        adata_test = sc.read(ann["test"])

    # Harmonise the label column name to 'celltype'.
    for aliases_src in (adata, adata_test):
        if "celltype" not in aliases_src.obs:
            for alias in ann.get("label_aliases", []):
                if alias in aliases_src.obs:
                    aliases_src.obs.rename(columns={alias: "celltype"}, inplace=True)
                    break
    for obj in (adata, adata_test):
        if "celltype" not in obj.obs:
            raise KeyError(
                "No 'celltype' column after alias resolution; tried "
                f"{ann.get('label_aliases', [])}"
            )
        obj.obs["celltype"] = obj.obs["celltype"].astype("category")

    # Uppercased gene symbols for case-insensitive vocabulary matching.
    for obj in (adata, adata_test):
        obj.var_names_make_unique()
        obj.var["gene_name"] = obj.var_names.astype(str).str.upper()

    # Tag batches before concatenation.
    adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
    adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"

    adata_test_raw = adata_test.copy()
    adata = adata.concatenate(adata_test, batch_key="str_batch")

    adata.obs["batch_id"] = adata.obs["str_batch"].astype("category").cat.codes.values
    celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
    adata.obs["celltype_id"] = celltype_id_labels
    num_types = len(np.unique(celltype_id_labels))
    id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))

    print(f"[annotation] train={int((adata.obs['str_batch'] == '0').sum())} "
          f"test={int((adata.obs['str_batch'] == '1').sum())} "
          f"classes={num_types}", flush=True)

    return adata, adata_test_raw, id2type, num_types
