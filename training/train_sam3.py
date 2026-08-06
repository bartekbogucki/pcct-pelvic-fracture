#!/usr/bin/env python3
"""
train_sam3.py — train SAMed-on-SAM3 using SAMed's EXACT trainer (trainer_v2.py).

Only difference vs SAMed: the model is SAM3_LoRA_Seg (SAM3 backbone) instead of LoRA_Sam
(SAM1 backbone). Same data, same Dice+CE (softmax, 2-class), same LR schedule, same
best-val-loss checkpointing. This isolates the backbone (SAM1 vs SAM3) as the only variable.

Run from inside SAMed/ (so `from trainer_v2 import trainer_synapse` and datasets resolve),
with the repo root on sys.path for sam3_lora_seg.
"""
import argparse, os, sys, random
import numpy as np
import torch

ROOT = "${PROJECT_ROOT}"
sys.path.insert(0, ROOT)                 # for sam3_lora_seg
sys.path.insert(0, f"{ROOT}/SAMed")      # for trainer_v2, datasets, utils
sys.path.insert(0, f"{ROOT}/MedSAM3")    # for sam3 package

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_path", required=True)
    ap.add_argument("--val_path", required=True)
    ap.add_argument("--list_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num_classes", type=int, default=1)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--base_lr", type=float, default=0.005)
    ap.add_argument("--max_epochs", type=int, default=50)
    ap.add_argument("--stop_epoch", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--es_patience", type=int, default=100)
    ap.add_argument("--warmup", action="store_true", default=True)
    ap.add_argument("--warmup_period", type=int, default=250)
    ap.add_argument("--AdamW", action="store_true", default=True)
    ap.add_argument("--dice_param", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n_gpu", type=int, default=1)
    ap.add_argument("--dataset", default="Synapse")
    a = ap.parse_args()

    # SAMed's trainer references a few args attributes; set them
    a.is_pretrain = True
    a.exp = "SAM3_LoRA"
    a.deterministic = 1
    a.vit_name = "sam3"
    a.module = "sam3_lora_seg"

    random.seed(a.seed); np.random.seed(a.seed)
    torch.manual_seed(a.seed); torch.cuda.manual_seed(a.seed)

    os.makedirs(a.output, exist_ok=True)

    # build the SAM3 model (SAMed interface)
    from sam3_lora_seg import build_sam3_lora_seg
    net = build_sam3_lora_seg(rank=a.rank).cuda()

    # SAM3 native embedding size 72 -> low_res = 72*4 = 288? No: SAMed uses img_size/16*4.
    # We keep SAMed's convention: low_res based on img_size (512/16*4 = 128) so labels match.
    low_res = a.img_size // 16 * 4   # 128, same as SAMed vit_b

    # reuse SAMed's trainer exactly
    os.chdir(f"{ROOT}/SAMed")
    from trainer_v2 import trainer_synapse
    trainer_synapse(a, net, a.output, multimask_output=False, low_res=low_res)


if __name__ == "__main__":
    main()
