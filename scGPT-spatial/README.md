# scGPT-spatial benchmark scripts

Code for the scGPT-spatial arm of *Harmonised benchmarking of foundation models
for single-cell and spatial transcriptomics reveals context-dependent
generalisation*.

Five task scripts, one per benchmark task. Each one takes `--dataset` and works
on any dataset in the registry, so there is a single copy of every routine
rather than one script per dataset.

| Script | Task | Datasets | Manuscript |
|---|---|---|---|
| `run_zero_shot_clustering.py` | Zero-shot clustering | ATAA, HS, BMMC, BC, MHS, Emb | Fig 2, Table 2 |
| `run_continual_pretrain_clustering.py` | Continually-pretrained clustering | ATAA, HS, BMMC, BC, MHS, Emb | Fig 3, Table 3 |
| `run_cell_annotation.py` | Supervised cell type annotation | ATAA, HS, BMMC, SEAAD | Fig 4, Table 4 |
| `run_perturbation.py` | Perturbation prediction | adamson, norman, replogle | Fig 6, Table 5 |
| `run_deg_concordance.py` | Marker-gene concordance | ATAA, HS, BMMC, BC, Emb | Fig 5, Fig S2 |


## Layout

```
benchmark/
  scfm_bench/
    paths.py             filesystem locations, all env-overridable
    datasets.py          dataset registry: paths, label columns, Leiden windows
    clustering.py        Leiden sweep, ARI/NMI/silhouette, result serialisation
    continual.py         2-epoch masked-value-completion adaptation
    annotation_data.py   train/test split loading and harmonisation
  run_*.py               one script per task         
```

## Datasets

Short names are the ones used throughout the manuscript.

| Key | Dataset | Modality | Reference labels |
|---|---|---|---|
| `ATAA` | Ascending thoracic aortic aneurysm, GSE155468 | scRNA-seq | `celltype`, 11 types |
| `HS` | Human splenic dendritic cells | scRNA-seq | `cell_type`, 7 subtypes |
| `BMMC` | NeurIPS 2021 bone marrow CITE-seq, GSE194122 | scRNA-seq | `cell_type`, 45 types |
| `BC` | HBCA1 human breast cancer, 10x Visium | spatial | `fine_annot_type`, 20 domains |
| `MHS` | MHPC mouse hippocampus, Slide-seqV2 | spatial | `cluster`, 14 domains |
| `Emb` | MOSTA E9.5 E1S1 mouse embryo, Stereo-seq | spatial | `annotation`, 12 regions |
| `SEAAD` | Seattle Alzheimer's MTG MERFISH | spatial | `Subclass`, 24 types |
| `adamson` / `norman` / `replogle` | K562 Perturb-seq | GEARS | — |

## Running

Point the scripts at your data with environment variables (all optional; the
defaults are the NCI Gadi paths used for the manuscript):

```bash
export SCFM_PROJECT_ROOT=/path/to/scGPT-spatial   # holds scgpt_spatial/
export SCFM_DATA_ROOT=/path/to/data
export SCFM_RESULTS_ROOT=/path/to/results         # clusters/ and figures/ go here
export SCFM_CHECKPOINT_ROOT=/path/to/checkpoints
export SCFM_PERTURB_ROOT=/path/to/perturb_datasets
```

Then, from `benchmark/`:

```bash
python run_zero_shot_clustering.py --dataset all
python run_continual_pretrain_clustering.py --dataset all
python run_cell_annotation.py --dataset ATAA
python run_perturbation.py --dataset adamson --mode train
python run_deg_concordance.py --dataset Emb --top-n 20
python plot_results.py clustering --dataset Emb --step finetune
```

`gene_col` is either `"index"` (pass `var_names` straight through) or
`"gene_name"` (write an uppercased copy of `var_names` first, for
case-insensitive matching against the checkpoint vocabulary).

## Method summary

**Zero-shot clustering.** Embeddings come from `scgpt_spatial.tasks.embed_data`
with the released checkpoint: genes aligned to the checkpoint vocabulary,
out-of-vocabulary genes dropped, non-zero genes tokenised, two-step
normalisation, expression binned into 51 levels, `<cls>` prepended, and the
L2-normalised CLS hidden state taken as the cell embedding. scRNA-seq datasets
are clustered with Leiden directly on the embedding (k = 15); spatial datasets
are reduced to 20 PCs first and use k = 50. The resolution is swept until the
cluster count equals the number of reference labels. Seed 42 everywhere.

**Continual pretraining.** Two epochs of masked value completion (mask ratio
0.20, Adam, lr 1e-4, 51 bins) starting from the released checkpoint, best epoch
selected on held-out masked-token MSE, then the same clustering and metrics.

**Cell type annotation.** Supervised fine-tuning with the classification
objective for 10 epochs, raw UMI inputs (`data_is_raw=True`), argmax over the
classifier output at test time, softmax logits for the AUC metrics. The 80:20
train/test partition is precomputed and shared across all models in the
benchmark; SEAAD uses a section-wise split so no tissue section straddles the
two sides.

**Perturbation prediction.** GEARS-style setup: only the encoder stack is
transferred from the pretrained checkpoint, the perturbation decoder is trained
from scratch for up to 15 epochs with early stopping on validation Pearson.
Reported metrics are the scGPT set (`compute_perturbation_metrics`), the GEARS
set (`compute_metrics`) and per-subgroup deeper/non-dropout analyses.

**Marker-gene concordance.** Top-20 Wilcoxon rank-sum markers per Leiden cluster
and per ground-truth domain, compared by raw overlap count and Jaccard index.
