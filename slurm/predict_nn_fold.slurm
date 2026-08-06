#!/bin/bash
#SBATCH --account=2026_029
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --job-name=nn_pred_f
#SBATCH --output=${PROJECT_ROOT}/test_eval/nn_pred_f%j.out
module load Miniconda3/25.7.0-2
ENVBIN=${PROJECT_ROOT}/envs/fracture/bin
export PATH=$ENVBIN:$PATH
export nnUNet_raw=${PROJECT_ROOT}/nnsam_workdir/raw
export nnUNet_preprocessed=${PROJECT_ROOT}/nnsam_workdir/preprocessed
export nnUNet_results=${PROJECT_ROOT}/nnsam_workdir/results
F=$1
$ENVBIN/nnUNetv2_predict \
  -i ${PROJECT_ROOT}/test_eval/fold$F/imagesTs_nnunet \
  -o ${PROJECT_ROOT}/test_eval/fold$F/pred_nnunet_s10 \
  -d 2 -c 3d_fullres -f $F -tr nnUNetTrainer_ES \
  -chk checkpoint_best_valloss.pth -step_size 1.0
