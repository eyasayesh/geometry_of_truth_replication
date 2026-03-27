#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128GB
#SBATCH -t 12:00:00
#SBATCH -q inferno
#SBATCH -o jobreports/patching_pipeline/Report-%A.out
#SBATCH -e jobreports/patching_pipeline/Report-%A.err
#SBATCH --mail-user=eayesh3@gatech.edu
#SBATCH --mail-type=FAIL,END

module load anaconda3
conda activate EM_env

cd /storage/project/r-aivanova7-0/shared/eyas/geometry_of_truth_replication

source .env
export HF_TOKEN

# Override at submit time via --export, e.g.:
#   sbatch --export=MODELS_FILE=data/patching_models.json \
#       batch_scripts/patching_pipeline.bash
MODELS_FILE=${MODELS_FILE:-"data/patching_models.json"}
PROMPTS_DIR=${PROMPTS_DIR:-"data/patching_prompts"}
OUTPUT_DIR=${OUTPUT_DIR:-"experimental_outputs"}
FIGURES_DIR=${FIGURES_DIR:-"figures/patching"}

mkdir -p jobreports/patching_pipeline "$OUTPUT_DIR" "$FIGURES_DIR"

python -u scripts/patching_pipeline.py \
    --models_file "$MODELS_FILE" \
    --prompts_dir "$PROMPTS_DIR" \
    --output_dir  "$OUTPUT_DIR" \
    --figures_dir "$FIGURES_DIR"
