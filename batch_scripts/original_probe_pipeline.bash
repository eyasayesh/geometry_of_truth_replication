#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH -t 8:00:00
#SBATCH -q inferno
#SBATCH -o jobreports/original_probe_pipeline/Report-%A.out
#SBATCH -e jobreports/original_probe_pipeline/Report-%A.err
#SBATCH --mail-user=eayesh3@gatech.edu
#SBATCH --mail-type=FAIL,END

# ── Usage ─────────────────────────────────────────────────────────────────────
# Basic (uses defaults below):
#   sbatch batch_scripts/original_probe_pipeline.bash
#
# Override at submit time with --export, e.g.:
#   sbatch --export=MODEL_NAME="llama-2-13b-hf",LAYERS="12 16 20" \
#       batch_scripts/original_probe_pipeline.bash
# ─────────────────────────────────────────────────────────────────────────────

module load anaconda3
conda activate EM_env

export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

cd /storage/project/r-aivanova7-0/shared/eyas/geometry_of_truth_replication

source .env
export HF_TOKEN

# ── Defaults (override via --export) ─────────────────────────────────────────
MODEL_NAME=${MODEL_NAME:-"llama-2-13b-hf"}
LAYERS=${LAYERS:-"8 9 10 11 12 13 14 15 16 17 18 19 20"}
TRAIN_GROUPS=${TRAIN_GROUPS:-"cities neg_cities sp_en_trans neg_sp_en_trans larger_than smaller_than"}
TEST_DATASETS=${TEST_DATASETS:-"cities neg_cities sp_en_trans neg_sp_en_trans larger_than smaller_than"}
TEST_SPLIT=${TEST_SPLIT:-0.2}
SEED=${SEED:-42}
LR_EPOCHS=${LR_EPOCHS:-1000}

ACTS_DIR="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/acts_original"
OUTPUT_DIR="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/probes_original"

mkdir -p jobreports/original_probe_pipeline "$OUTPUT_DIR"

python -u scripts/original_probe_pipeline.py \
    --model_name    "$MODEL_NAME" \
    --layers        $LAYERS \
    --train_groups  $TRAIN_GROUPS \
    --test_datasets $TEST_DATASETS \
    --acts_dir      "$ACTS_DIR" \
    --output_dir    "$OUTPUT_DIR" \
    --test_split    "$TEST_SPLIT" \
    --seed          "$SEED" \
    --lr_epochs     "$LR_EPOCHS"
