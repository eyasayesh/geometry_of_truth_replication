#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32GB
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
MODEL=${MODEL:-"llama-3.2-1b"}
DATASETS=${DATASETS:-"cities neg_cities sp_en_trans neg_sp_en_trans larger_than smaller_than cities_cities_conj cities_cities_disj companies_true_false common_claim_true_false counterfact_true_false likely"}
LAYERS=${LAYERS:-""}

LAYER_ARG=""
if [ -n "$LAYERS" ]; then
    LAYER_ARG="--layers $LAYERS"
fi

python -u scripts/generate_acts.py \
    --model "$MODEL" \
    --datasets $DATASETS \
    $LAYER_ARG
