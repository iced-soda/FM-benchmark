# GenePT-w Downstream Tasks

This folder contains code adapted from [GenePT](https://github.com/yiqunchen/GenePT) for three downstream single-cell analysis tasks built on **GPT-3.5-derived gene embeddings**: zero-shot clustering, cell type annotation, and perturbation response prediction.

GenePT represents each gene by embedding its textual description (e.g., from NCBI) with GPT-3.5, then represents each cell (`GenePT-w`) as the expression-weighted average of its genes' embeddings. These cell embeddings are used here as a drop-in representation for standard scRNA-seq analysis tasks.

## Contents

| Notebook | Task |
|---|---|
| `zero_shot_clustering.ipynb` | Cluster cells using GenePT-w embeddings with no training (KMeans and Leiden) |
| `cell_type_annotation.ipynb` | Annotate cell types via kNN classification on GenePT-w embeddings |
| `perturbation.ipynb` | Predict transcriptional response to genetic perturbation using [GEARS](https://github.com/snap-stanford/GEARS) initialized with GenePT gene embeddings |

## Requirements

- `scanpy`, `anndata`
- `numpy`, `pandas`, `scikit-learn`
- `torch`
- `matplotlib`, `seaborn`
- `hnswlib` (optional, recommended for fast approximate kNN search — falls back to a user-supplied `get_similar_vectors` if not installed)
- `gears`, `pytorch_lightning` (perturbation notebook only)

Random seeds are fixed (`fix_seed(42)` / `pytorch_lightning.seed_everything(202310)`) for reproducibility, including deterministic CUDA/cuDNN settings where available.

## Data Layout

Only the GenePT-w gene embedding pickles are specific to this repo — they live in `genept/data/` (i.e. the `data/` folder the notebooks read from is local to the `genept/` folder they sit in):

```
genept/
├── run_zero_shot_clustering.ipynb
├── run_cell_annotation.ipynb
├── run_perturbation.ipynb
└── data/
    ├── GPT_3_5_gene_embeddings.pickle          # gene -> GPT-3.5 embedding dict (used in annotation)
```

Gene embeddings are 1536-dimensional (OpenAI `text-embedding` / GPT-3.5 embedding dimension).

## Notebook Details

### `zero_shot_clustering.ipynb`
Loads an `.h5ad` dataset and a gene embedding pickle, computes the GenePT-w cell embedding (`X @ gene_embeddings / n_genes`), then runs:
- **KMeans** with `k` set to the number of ground-truth labels in `label_variable` (dataset-dependent: `cell_type`, `cluster`, `annotation`, or `fine_annot_type`)
- **Leiden** clustering (via `scanpy` neighbors graph on the GenePT-w embedding)


### `cell_type_annotation.ipynb`
Loads pre-split train/test `.h5ad` files, restricts both to genes shared with the embedding table, and computes GenePT-w embeddings for train and test cells. Cell types are predicted by **majority vote over the k=7 nearest neighbors** (cosine distance) in embedding space, found via `hnswlib` when available.

Reported metrics: accuracy, macro precision/recall/F1, macro PR-AUC, and macro ROC-AUC (using per-cell neighbor-vote class probabilities as scores).

### `perturbation.ipynb`
Wraps [GEARS](https://github.com/snap-stanford/GEARS) to predict post-perturbation gene expression, replacing its default gene embedding with the GPT-3.5 gene embedding table (`gene_emb=lookup_embed`, zero-padded for genes without an embedding). Trains a GEARS model on a perturb-seq dataset (default: `adamson`) and evaluates it with:
- Pearson correlation of predicted vs. true mean expression per condition, on all genes and on top differentially-expressed (DE) genes
- Same correlations computed on the perturbation-induced **delta** (change from control), on all genes and on DE genes

## Acknowledgements

This code adapts and builds on:
- **GenePT** — Chen & Zou, *"GenePT: A Simple But Effective Foundation Model for Genes and Cells Built From ChatGPT"* — [github.com/yiqunchen/GenePT](https://github.com/yiqunchen/GenePT)
- **GEARS** — Roohani, Huang & Leskovec, *"Predicting transcriptional outcomes of novel multigene perturbations with GEARS"* — [github.com/snap-stanford/GEARS](https://github.com/snap-stanford/GEARS)

## License

MIT — see [LICENSE](./LICENSE).
