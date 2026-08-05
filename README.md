# FM-benchmark

Code accompanying **"Harmonised benchmarking of foundation models for single-cell and spatial transcriptomics reveals context-dependent generalisation."**

[Preprint (arXiv)](https://arxiv.org/abs/2607.17227)

## Overview

Single-cell and spatial foundation models promise transferable biological representations, yet their generality remains largely untested across modalities, biological domains and analytical tasks. This repository provides a harmonised benchmarking framework spanning scRNA-seq, spatial transcriptomics and Perturb-seq, evaluating zero-shot and continually pretrained clustering, supervised cell type annotation, marker-gene concordance and perturbation prediction.

The repository is organised by model:

- [`cellplm/`](cellplm) — [CellPLM](https://openreview.net/forum?id=BKXvPDekud)
- [`nicheformer/`](nicheformer) — [Nicheformer](https://www.nature.com/articles/s41592-025-02814-z)
- [`scGPT-spatial/`](scGPT-spatial) — [scGPT-spatial](https://www.biorxiv.org/content/10.1101/2025.02.05.636714v1)
- [`GenePT/`](GenePT) — [GenePT](https://www.nature.com/articles/s41551-024-01284-6)
- [`scELMo/`](scELMo) — [scELMo](https://www.cell.com/patterns/fulltext/S2666-3899(25)00279-X)
- [`Novae/`](Novae) — [Novae](https://www.nature.com/articles/s41592-025-02899-6)

Each directory contains the model codebase plus a set of notebooks or scripts that run the shared benchmarking tasks.

## Citation

If you use this benchmark, please cite:

```
Chen, S., Zahedi, R., Chhuo, L., Nguyen, R., BaghGolshani, M., Beheshti, A., Grosser, M., Yang, M.,
Farbehi, N., Lovell, N., Argha, A., Vafaee, F., Ye, Y., Alinejad-Rokny, H. (2026).
Harmonised benchmarking of foundation models for single-cell and spatial transcriptomics
reveals context-dependent generalisation. arXiv:2607.17227.
```

## License

See [LICENSE](LICENSE) for this repository's own code. Each model's original license is included in its own subdirectory.
