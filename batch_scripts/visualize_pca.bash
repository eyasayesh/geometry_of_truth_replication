#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH -t 0:30:00
#SBATCH -q inferno
#SBATCH -o jobreports/visualize_pca/Report-%A-%a.out
#SBATCH -e jobreports/visualize_pca/Report-%A-%a.err
#SBATCH --mail-user=eayesh3@gatech.edu
#SBATCH --mail-type=FAIL,END

module load anaconda3
conda activate EM_env

cd /storage/project/r-aivanova7-0/shared/eyas/geometry_of_truth_replication

source .env
export HF_TOKEN

# Override at submit time via --export, e.g.:
#   sbatch --export=MODEL=llama-3.2-1b,DATASET=cities,LAYER=8 batch_scripts/visualize_pca.bash
MODEL=${MODEL:-"llama-3.1-8b"}
DATASET=${DATASET:-"cities"}
LAYERS=${LAYERS:-"9 10 11 12"}
ACTS_DIR="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/acts"
OUTPUT_DIR="figures/pca"
PCA_OUTPUT_DIR="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/pca"

mkdir -p jobreports/visualize_pca "$OUTPUT_DIR" "$PCA_OUTPUT_DIR"

python -u scripts/visualize_pca.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --layers $LAYERS \
    --acts_dir "$ACTS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --pca_output_dir "$PCA_OUTPUT_DIR"
