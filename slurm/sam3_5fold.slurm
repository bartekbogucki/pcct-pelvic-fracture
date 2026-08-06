#!/bin/bash
#SBATCH --account=2026_029
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --job-name=sam3_5fold
#SBATCH --output=${PROJECT_ROOT}/sam3_5fold_%j.out
module load Miniconda3/25.7.0-2
R=${PROJECT_ROOT}
export HF_HOME=$R/.hf_cache
PY=$R/envs/fracture/bin/python
FOLD=$1
D=$R/samed_data_fold$FOLD
[ -d "$D/train_npz" ] || $PY $R/gen_samed_fold.py --fold $FOLD
cd $R/MedSAM3
$PY $R/train_sam3.py \
  --root_path $D/train_npz --val_path $D/val_npz --list_dir $D/lists \
  --output $R/sam3_lora_results/fold${FOLD}_final \
  --num_classes 1 --img_size 512 --rank 4 --base_lr 0.001 \
  --max_epochs 50 --stop_epoch 50 --batch_size 8 --dice_param 0.8
