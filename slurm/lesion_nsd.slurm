#!/bin/bash
#SBATCH --account=2026_029
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=6:00:00
#SBATCH --job-name=lesion_nsd
#SBATCH --output=${PROJECT_ROOT}/lesion_nsd_%j.out
module load Miniconda3/25.7.0-2
R=${PROJECT_ROOT}
$R/envs/fracture/bin/python $R/eval_lesion_nsd.py --model $1 --tau 2 5 --folds 0 1 2 3 4
