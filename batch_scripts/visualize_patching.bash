#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH -t 0:30:00
#SBATCH -q inferno
#SBATCH -o jobreports/visualize_patching/Report-%A-%a.out
#SBATCH -e jobreports/visualize_patching/Report-%A-%a.err
#SBATCH --mail-user=eayesh3@gatech.edu
#SBATCH --mail-type=FAIL,END

module load anaconda3
conda activate EM_env

cd /storage/project/r-aivanova7-0/shared/eyas/geometry_of_truth_replication

source .env
export HF_TOKEN

# Override at submit time via --export, e.g.:
#   sbatch --export=RESULTS_FILE=experimental_outputs/patching_sp_en_trans.json,MODEL=llama-3.1-8b \
#       batch_scripts/visualize_patching.bash
RESULTS_FILE=${RESULTS_FILE:-"experimental_outputs/patching_cities.json"}
MODEL=${MODEL:-"llama-3.1-8b"}
OUTPUT_DIR=${OUTPUT_DIR:-"figures/patching"}

mkdir -p jobreports/visualize_patching "$OUTPUT_DIR"

python -u scripts/visualize_patching.py \
    --results_file "$RESULTS_FILE" \
    --model "$MODEL" \
    --output_dir "$OUTPUT_DIR"
