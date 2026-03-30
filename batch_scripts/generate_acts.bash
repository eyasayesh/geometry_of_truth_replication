#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64GB
#SBATCH -t 3:00:00
#SBATCH -q embers
#SBATCH -o jobreports/generate_acts/Report-%A-%a.out
#SBATCH -e jobreports/generate_acts/Report-%A-%a.err
#SBATCH --mail-user=eayesh3@gatech.edu
#SBATCH --mail-type=FAIL,END

module load anaconda3
conda activate EM_env

cd /storage/project/r-aivanova7-0/shared/eyas/geometry_of_truth_replication

source .env
export HF_TOKEN

# Override these at submit time via --export, e.g.:
#   sbatch --export=MODEL=llama-3.2-1b,LAYERS="0 4 8 12 15" batch_scripts/generate_acts.bash
MODEL=${MODEL:-"llama-3.1-8b"}
DATASETS=${DATASETS:-"cities"}
LAYERS=${LAYERS:-"9 10 11 12"}
OUTPUT_DIR="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/acts"

mkdir -p "$OUTPUT_DIR"

LAYER_ARG=""
if [ -n "$LAYERS" ]; then
    LAYER_ARG="--layers $LAYERS"
fi

python -u scripts/generate_acts.py \
    --model "$MODEL" \
    --datasets $DATASETS \
    --output_dir "$OUTPUT_DIR" \
    $LAYER_ARG
