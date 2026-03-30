#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32GB
#SBATCH -t 3:00:00
#SBATCH -q embers
#SBATCH -o jobreports/Llama_EM_eval/Report-%A-%a.out
#SBATCH -e jobreports/Llama_EM_eval/Report-%A-%a.err
#SBATCH --mail-user=eayesh3@gatech.edu 
#SBATCH --mail-type=FAIL,END

module load anaconda3
conda activate EM_env

cd /storage/project/r-aivanova7-0/shared/eyas/EM_probes

python -u replications/model_organisms_of_EM/Llama_EM_eval.py