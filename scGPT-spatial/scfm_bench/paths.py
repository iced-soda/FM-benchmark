"""Filesystem locations.

Every path can be overridden with an environment variable so the same scripts run
on a different machine without editing code:

    SCFM_PROJECT_ROOT   checkout of this repository (holds scgpt_spatial/)
    SCFM_DATA_ROOT      directory holding the benchmark h5ad files
    SCFM_RESULTS_ROOT   where result bundles (.npz) and figures are written
    SCFM_CHECKPOINT_ROOT  where model checkpoints live
    SCFM_PERTURB_ROOT   GEARS PertData parent directory
"""

import os
import sys
from pathlib import Path

# Repository root: the directory that contains `scgpt_spatial/`.
PROJECT_ROOT = Path(
    os.environ.get(
        "SCFM_PROJECT_ROOT",
        Path(__file__).resolve().parents[2],
    )
).resolve()

DATA_ROOT = Path(
    os.environ.get(
        "SCFM_DATA_ROOT",
        "/g/data/yr31/rn8079/scfm_benchmark/scGPT-spatial/data",
    )
)

RESULTS_ROOT = Path(
    os.environ.get(
        "SCFM_RESULTS_ROOT",
        "/g/data/yr31/rn8079/scfm_benchmark/scGPT-spatial",
    )
)

CHECKPOINT_ROOT = Path(
    os.environ.get(
        "SCFM_CHECKPOINT_ROOT",
        str(PROJECT_ROOT / "checkpoints"),
    )
)

PERTURB_ROOT = Path(
    os.environ.get(
        "SCFM_PERTURB_ROOT",
        "/g/data/yr31/rn8079/scfm_benchmark/perturb_datasets",
    )
)

#: Released scGPT-spatial checkpoint that every task starts from.
BASE_CHECKPOINT = CHECKPOINT_ROOT / "scGPT_spatial_v1"

#: Result bundles (one .npz per model x step x dataset).
CLUSTERS_DIR = RESULTS_ROOT / "clusters"
FIGURES_DIR = RESULTS_ROOT / "figures"

MODEL_NAME = "scGPT"  # prefix used in every output filename


def project_on_path() -> Path:
    """Put the repository root on ``sys.path`` so ``import scgpt_spatial`` works."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PROJECT_ROOT


def ensure_dirs() -> None:
    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
