#!/bin/bash
#SBATCH --account=2026_029
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=8:00:00
#SBATCH --job-name=eval_model
#SBATCH --output=${PROJECT_ROOT}/test_eval/eval_%j.out
module load Miniconda3/25.7.0-2
ROOT=${PROJECT_ROOT}
$ROOT/envs/fracture/bin/python $ROOT/eval_model_allfolds.py --model $1 ${@:2}
