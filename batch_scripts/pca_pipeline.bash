#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128GB
#SBATCH -t 12:00:00
#SBATCH -q inferno
#SBATCH -o jobreports/pca_pipeline/Report-%A.out
#SBATCH -e jobreports/pca_pipeline/Report-%A.err
#SBATCH --mail-user=eayesh3@gatech.edu
#SBATCH --mail-type=FAIL,END

module load anaconda3
conda activate EM_env

cd /storage/project/r-aivanova7-0/shared/eyas/geometry_of_truth_replication

source .env
export HF_TOKEN

# Override at submit time via --export, e.g.:
#   sbatch --export=CONFIG=data/pca_config.json batch_scripts/pca_pipeline.bash
CONFIG=${CONFIG:-"data/configs/pca/pca_config.json"}
ACTS_DIR="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/acts"
PCA_DIR="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/pca"
FIGURES_DIR="figures/pca"

mkdir -p jobreports/pca_pipeline "$FIGURES_DIR"


python -u scripts/pca_pipeline.py \
    --config      "$CONFIG" \
    --acts_dir    "$ACTS_DIR" \
    --pca_dir     "$PCA_DIR" \
    --figures_dir "$FIGURES_DIR" \
    --no_scale
