#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64GB
#SBATCH -t 3:00:00
#SBATCH -q inferno
#SBATCH -o jobreports/patching_experiment/Report-%A-%a.out
#SBATCH -e jobreports/patching_experiment/Report-%A-%a.err
#SBATCH --mail-user=eayesh3@gatech.edu
#SBATCH --mail-type=FAIL,END

module load anaconda3
conda activate EM_env

cd /storage/project/r-aivanova7-0/shared/eyas/geometry_of_truth_replication

source .env
export HF_TOKEN

# Override at submit time via --export, e.g.:
#   sbatch --export=MODEL=llama-3.2-1b,PROMPTS_FILE=data/patching_prompts/sp_en_trans.json \
#       batch_scripts/patching_experiment.bash
MODEL=${MODEL:-"llama-3.1-8b"}
PROMPTS_FILE=${PROMPTS_FILE:-"data/patching_prompts/cities.json"}
DATASET=$(basename "$PROMPTS_FILE" .json)
OUTPUT_FILE=${OUTPUT_FILE:-"experimental_outputs/patching_${DATASET}.json"}

mkdir -p jobreports/patching_experiment experimental_outputs

python -u scripts/patching_experiment.py \
    --model "$MODEL" \
    --prompts_file "$PROMPTS_FILE" \
    --output_file "$OUTPUT_FILE"
