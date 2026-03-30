#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128GB
#SBATCH -t 12:00:00
#SBATCH -q inferno
#SBATCH -o jobreports/original_pca_pipeline/Report-%A.out
#SBATCH -e jobreports/original_pca_pipeline/Report-%A.err
#SBATCH --mail-user=eayesh3@gatech.edu
#SBATCH --mail-type=FAIL,END

# ── Usage ─────────────────────────────────────────────────────────────────────
# Default:
#   sbatch batch_scripts/original_pca_pipeline.bash
#
# Override config at submit time:
#   sbatch --export=CONFIG=data/configs/pca/my_config.json \
#       batch_scripts/original_pca_pipeline.bash
#
# Add --noperiod (set NOPERIOD=1):
#   sbatch --export=ALL,NOPERIOD=1 batch_scripts/original_pca_pipeline.bash
# ─────────────────────────────────────────────────────────────────────────────

module load anaconda3
conda activate EM_env

export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

cd /storage/project/r-aivanova7-0/shared/eyas/geometry_of_truth_replication

source .env
export HF_TOKEN

CONFIG=${CONFIG:-"data/configs/pca/original_pca_config.json"}
NOPERIOD=${NOPERIOD:-0}
N_PCS=${N_PCS:-20}

ACTS_DIR="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/acts_original"
PCA_DIR="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/pca_original"
FIGURES_DIR="figures/pca_original"

mkdir -p jobreports/original_pca_pipeline "$FIGURES_DIR"

EXTRA_FLAGS=""
[[ "$NOPERIOD" == "1" ]] && EXTRA_FLAGS="--noperiod"

python -u scripts/original_pca_pipeline.py \
    --config      "$CONFIG" \
    --acts_dir    "$ACTS_DIR" \
    --pca_dir     "$PCA_DIR" \
    --figures_dir "$FIGURES_DIR" \
    --n_pcs       "$N_PCS" \
    $EXTRA_FLAGS
