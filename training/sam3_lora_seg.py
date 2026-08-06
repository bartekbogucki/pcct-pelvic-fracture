#!/usr/bin/env python3
"""
sam3_lora_seg.py — SAMed-on-SAM3.

Replicates SAMed's exact recipe (LoRA rank 4, q+v only, on the image encoder) but on the
SAM3 backbone instead of SAM1. Everything else — automatic empty-prompt segmentation via the
SAM prompt-encoder + mask-decoder — mirrors SAMed, so the ONLY variable vs SAMed is SAM1 -> SAM3.

Architecture (confirmed by probing sam3.pt):
  m.backbone.forward_image(x) -> {
      "vision_features": (B,256,72,72),                         # image embedding
      "backbone_fpn":    [(B,256,288,288),(B,256,144,144),(B,256,72,72)]  # high-res feats
  }
  m.inst_interactive_predictor.model  (Sam3TrackerPredictor) holds:
      .sam_prompt_encoder, .sam_mask_decoder, ._forward_sam_heads
  image_size=1008, backbone_stride=14, emb=72, hidden_dim=256.

LoRA: SAM3 trunk blocks use a FUSED qkv (like SAM1), so SAMed's _LoRA_qkv (q+v only) applies.

Forward (automatic, no prompt):
  feats = backbone.forward_image(x)
  emb   = feats["vision_features"]                 # (B,256,72,72)
  hi    = [conv_s0(fpn[0]), conv_s1(fpn[1])]       # done inside _forward_sam_heads
  low_res, high_res = tracker._forward_sam_heads(backbone_features=emb, high_res_features=hi, ...)
  -> single fracture logit map, upsampled to img_size.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- SAMed's LoRA on a fused qkv linear: inject into q and v slices only (k untouched) ----
class _LoRA_qkv(nn.Module):
    def __init__(self, qkv, a_q, b_q, a_v, b_v):
        super().__init__()
        self.qkv = qkv
        self.a_q, self.b_q = a_q, b_q
        self.a_v, self.b_v = a_v, b_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)                       # (..., 3*dim)
        new_q = self.b_q(self.a_q(x))
        new_v = self.b_v(self.a_v(x))
        qkv[..., : self.dim] += new_q           # q slice
        qkv[..., -self.dim:] += new_v           # v slice (k, the middle, untouched)
        return qkv


class SAM3_LoRA_Seg(nn.Module):
    """SAMed-style automatic segmenter on the SAM3 backbone, LoRA rank-r on trunk q,v."""

    def __init__(self, sam3_model, rank: int = 4):
        super().__init__()
        self.sam3 = sam3_model
        self.backbone = sam3_model.backbone
        self.tracker = sam3_model.inst_interactive_predictor.model  # Sam3TrackerPredictor
        self.img_size = getattr(self.tracker, "image_size", 1008)

        trunk = self.backbone.vision_backbone.trunk
        assert hasattr(trunk, "blocks"), "trunk has no .blocks"

        # freeze EVERYTHING first (SAMed freezes the whole SAM; only LoRA + decoder train)
        for p in self.sam3.parameters():
            p.requires_grad = False

        # inject LoRA into every trunk block's fused qkv (q,v only)
        self.lora_layers = nn.ModuleList()
        n_inj = 0
        for blk in trunk.blocks:
            if not hasattr(blk, "attn") or not hasattr(blk.attn, "qkv"):
                continue
            qkv = blk.attn.qkv
            dim = qkv.in_features
            a_q = nn.Linear(dim, rank, bias=False)
            b_q = nn.Linear(rank, dim, bias=False)
            a_v = nn.Linear(dim, rank, bias=False)
            b_v = nn.Linear(rank, dim, bias=False)
            nn.init.zeros_(b_q.weight); nn.init.zeros_(b_v.weight)  # start as identity
            blk.attn.qkv = _LoRA_qkv(qkv, a_q, b_q, a_v, b_v)
            self.lora_layers.extend([a_q, b_q, a_v, b_v])
            n_inj += 1
        print(f"[SAM3-LoRA] injected LoRA(rank={rank}, q+v) into {n_inj} trunk blocks")

        # SAMed also trains the mask decoder — unfreeze the SAM mask decoder
        for p in self.tracker.sam_mask_decoder.parameters():
            p.requires_grad = True
        # prompt encoder stays frozen (SAMed keeps it fixed; empty prompt is a learned constant)

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[SAM3-LoRA] trainable {n_train:,} / {n_total:,} ({100*n_train/n_total:.2f}%)")

    def forward(self, x, multimask_output=False, image_size=None, low_res=128):
        """SAMed-compatible interface. Returns dict with 'masks' (B,2,H,W) and
        'low_res_logits' (B,2,low_res,low_res) — 2 channels (bg,fg) for softmax Dice+CE,
        matching SAMed's calc_loss exactly. The 2 channels are formed as [-m, m] from the
        single fracture logit m, so softmax([-m,m]) == [1-sigmoid(m), sigmoid(m)]."""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        in_hw = x.shape[-1]
        if x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)

        # SAM3 expects input normalized with img_mean=img_std=0.5, i.e. [0,1] -> [-1,1]
        # (see sam3/model/io_utils.py). SAMed feeds raw [0,1] because it builds SAM with
        # pixel_mean=0/pixel_std=1, so we must apply SAM3's normalization here ourselves.
        x = (x - 0.5) / 0.5

        feats = self.backbone.forward_image(x)
        emb = feats["vision_features"]
        fpn = feats["backbone_fpn"]
        md = self.tracker.sam_mask_decoder
        pe = self.tracker.sam_prompt_encoder
        high_res = [md.conv_s0(fpn[0]), md.conv_s1(fpn[1])]

        sparse, dense = pe(points=None, boxes=None, masks=None)
        image_pe = pe.get_dense_pe()
        # empty prompt is batch-1; expand to the image batch size B so the decoder's
        # per-image token batch matches image_embeddings (assert at mask_decoder line 200).
        B = emb.shape[0]
        if sparse.shape[0] == 1 and B > 1:
            sparse = sparse.expand(B, -1, -1)
        if dense.shape[0] == 1 and B > 1:
            dense = dense.expand(B, -1, -1, -1)
        dec_out = md(
            image_embeddings=emb, image_pe=image_pe,
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False, repeat_image=False, high_res_features=high_res,
        )
        m = dec_out[0]                                        # (B,1,h,w) single fracture logit

        # form 2-channel [bg, fg] logits: softmax([-m, m]) == [1-sig(m), sig(m)]
        two = torch.cat([-m, m], dim=1)                      # (B,2,h,w)

        out_size = image_size or in_hw
        low = F.interpolate(two, size=(low_res, low_res), mode="bilinear", align_corners=False)
        masks = F.interpolate(two, size=(out_size, out_size), mode="bilinear", align_corners=False)
        return {"masks": masks, "low_res_logits": low}



    # SAMed-compatible save/load: only LoRA + mask-decoder deltas
    def save_lora_parameters(self, path):
        sd = {f"lora_{i}": l.weight for i, l in enumerate(self.lora_layers)}
        md = {f"md_{k}": v for k, v in self.tracker.sam_mask_decoder.state_dict().items()}
        torch.save({**sd, **md}, path)

    def load_lora_parameters(self, path):
        st = torch.load(path, map_location="cpu")
        for i, l in enumerate(self.lora_layers):
            l.weight.data.copy_(st[f"lora_{i}"])
        md = {k[3:]: v for k, v in st.items() if k.startswith("md_")}
        self.tracker.sam_mask_decoder.load_state_dict(md, strict=False)


def build_sam3_lora_seg(rank=4):
    """Build SAM3 (with SAM path) + wrap with LoRA-seg."""
    import sys
    sys.path.insert(0, "${PROJECT_ROOT}/MedSAM3")
    from sam3.model_builder import build_sam3_image_model
    m = build_sam3_image_model(enable_inst_interactivity=True)
    return SAM3_LoRA_Seg(m, rank=rank)


if __name__ == "__main__":
    # smoke test: build + one forward + one backward
    net = build_sam3_lora_seg(rank=4).cuda().train()
    x = torch.rand(1, 3, 512, 512, device="cuda")
    y = (torch.rand(1, 1, 512, 512, device="cuda") > 0.99).float()
    out = net(x, multimask_output=False, image_size=512)
    logits = out["masks"]
    print("masks:", tuple(logits.shape), "| low_res:", tuple(out["low_res_logits"].shape))
    loss = F.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    g = sum(p.grad.abs().sum().item() for p in net.parameters() if p.requires_grad and p.grad is not None)
    print(f"loss {loss.item():.4f} | grad-sum {g:.4f} (should be >0 if LoRA gets gradient)")
    print("SMOKE TEST PASSED" if g > 0 else "WARNING: zero gradient — LoRA not in graph")
