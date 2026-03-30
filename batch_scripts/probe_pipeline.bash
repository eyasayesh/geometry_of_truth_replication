#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH -t 8:00:00
#SBATCH -q inferno
#SBATCH -o jobreports/probe_pipeline/Report-%A.out
#SBATCH -e jobreports/probe_pipeline/Report-%A.err
#SBATCH --mail-user=eayesh3@gatech.edu
#SBATCH --mail-type=FAIL,END

module load anaconda3
conda activate EM_env

cd /storage/project/r-aivanova7-0/shared/eyas/geometry_of_truth_replication

# Override at submit time via --export, e.g.:
#   sbatch --export=CONFIG=data/configs/probes/finetuned_lora.json batch_scripts/probe_pipeline.bash
CONFIG=${CONFIG:-"data/configs/probes/finetuned_lora.json"}
ACTS_DIR="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/acts"
OUTPUT_DIR="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/probes"

mkdir -p jobreports/probe_pipeline "$OUTPUT_DIR"

python -u scripts/probe_pipeline.py \
    --config     "$CONFIG" \
    --acts_dir   "$ACTS_DIR" \
    --output_dir "$OUTPUT_DIR"
