"""
lora_layers.py — Clean LoRA implementation for SAM3 fine-tuning.

Based on:
- Hu et al. 2021 "LoRA: Low-Rank Adaptation of Large Language Models"
- Microsoft's loralib reference implementation
- Sompote/SAM3_LoRA architecture
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation.

    Default targets match SAM3's actual module structure:
      - SAM3 uses fused 'qkv' projection (not separate q/k/v)
      - Attention output: 'proj' (vision) or 'out_proj' (text/decoder)
      - MLP: 'fc1','fc2' (vision) or 'linear1','linear2' (transformer blocks)
    """
    rank: int = 8                          # rank of decomposition
    alpha: int = 16                        # scaling factor (typically 2*rank)
    dropout: float = 0.0                   # dropout on LoRA path
    target_modules: List[str] = field(
        default_factory=lambda: ["qkv", "proj", "fc1", "fc2"]
    )

    # Which model components to adapt (matching against full module path)
    # SAM3 paths: backbone.vision_backbone.*, backbone.text_*, head.*, geometry_encoder.*
    apply_to_vision_backbone: bool = True
    apply_to_text:            bool = False
    apply_to_geometry:        bool = False
    apply_to_head:            bool = False  # DETR-like head with decoder
    apply_to_all:             bool = False  # ignore component filter


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with LoRA adapters.

    Forward: y = W·x + (alpha/rank) * B·A·x
        - W is frozen original weight
        - A is (rank, in_features) — initialized Kaiming uniform
        - B is (out_features, rank) — initialized to zero
        - Initial state: BA = 0 → identical to original model
    """

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.base = base_linear
        in_features  = base_linear.in_features
        out_features = base_linear.out_features

        # Freeze base
        for p in self.base.parameters():
            p.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Init: A ~ Kaiming, B = 0 → BA = 0 (preserves original behaviour at init)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        out = self.base(x)
        # LoRA path
        lora_x = self.dropout(x) @ self.lora_A.t()        # (..., rank)
        lora_x = lora_x @ self.lora_B.t()                  # (..., out_features)
        return out + lora_x * self.scaling


def apply_lora_to_model(model: nn.Module, config: LoRAConfig, verbose: bool = True) -> nn.Module:
    """
    Walk through model and replace nn.Linear modules matching config.target_modules
    with LoRALinear wrappers, only in components enabled in config.

    Component matching is done by checking if the parent path contains one of:
      - 'vision_encoder' (when apply_to_vision_encoder=True)
      - 'text_encoder'
      - 'geometry_encoder'
      - 'detr_encoder'
      - 'detr_decoder'
      - 'mask_decoder'

    Returns the modified model and prints stats.
    """
    # First, freeze ALL parameters
    for p in model.parameters():
        p.requires_grad = False

    # Build the list of enabled component substrings (matched against full module path)
    enabled_components = []
    if config.apply_to_vision_backbone:  enabled_components.append("vision_backbone")
    if config.apply_to_text:             enabled_components.append("text")
    if config.apply_to_geometry:         enabled_components.append("geometry")
    if config.apply_to_head:             enabled_components.append("head")

    # Walk all named modules, find Linear layers to wrap
    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # Check if module name suffix matches target_modules
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in config.target_modules:
            continue

        # Check if module is inside an enabled component
        if config.apply_to_all:
            replacements.append(name)
            continue

        for comp in enabled_components:
            if comp in name.lower():
                replacements.append(name)
                break

    # Apply replacements
    n_replaced = 0
    for full_name in replacements:
        parts   = full_name.split(".")
        parent  = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        leaf_name = parts[-1]
        old_lin   = getattr(parent, leaf_name)
        new_lora  = LoRALinear(old_lin, config.rank, config.alpha, config.dropout)
        setattr(parent, leaf_name, new_lora)
        n_replaced += 1

    if verbose:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [LoRA] Wrapped {n_replaced} Linear modules")
        print(f"  [LoRA] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")
        if n_replaced == 0:
            print("  [LoRA] ⚠️  No modules wrapped! Check target_modules and component names.")

    return model


def count_parameters(model: nn.Module) -> dict:
    """Count trainable vs total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters":      total,
        "trainable_parameters":  trainable,
        "trainable_percentage":  100 * trainable / total if total > 0 else 0,
    }


def save_lora_weights(model: nn.Module, path: str) -> None:
    """Save only LoRA parameters (small file)."""
    lora_state = {n: p.data for n, p in model.named_parameters()
                  if p.requires_grad and ("lora_A" in n or "lora_B" in n)}
    torch.save(lora_state, path)
    print(f"  Saved {len(lora_state)} LoRA params → {path}")


def load_lora_weights(model: nn.Module, path: str, strict: bool = False) -> None:
    """Load LoRA weights into a model that already has LoRA layers applied."""
    state = torch.load(path, map_location="cpu")
    model_state = dict(model.named_parameters())
    loaded = 0
    missing = 0
    for k, v in state.items():
        if k in model_state:
            model_state[k].data.copy_(v.to(model_state[k].device))
            loaded += 1
        else:
            missing += 1
    if strict and missing > 0:
        raise RuntimeError(f"Strict load failed: {missing} keys missing")
    print(f"  Loaded {loaded}/{len(state)} LoRA params (missing: {missing})")


def discover_lora_targets(model: nn.Module, max_show: int = 30) -> None:
    """
    Diagnostic helper: print all nn.Linear modules in the model with their
    full names. Useful for figuring out what target_modules to specify.
    """
    print("\n[LoRA Discovery] All nn.Linear modules in model:")
    count = 0
    components = set()
    leaf_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            count += 1
            leaf = name.rsplit(".", 1)[-1]
            leaf_names.add(leaf)
            # Try to identify component
            for comp_keyword in ["vision_encoder", "text_encoder", "geometry_encoder",
                                  "detr_encoder", "detr_decoder", "mask_decoder",
                                  "backbone", "head"]:
                if comp_keyword in name.lower():
                    components.add(comp_keyword)
                    break
            if count <= max_show:
                print(f"  {name}  ({module.in_features} → {module.out_features})")
    if count > max_show:
        print(f"  ... and {count - max_show} more")
    print(f"\nTotal Linear modules: {count}")
    print(f"Unique leaf names: {sorted(leaf_names)}")
    print(f"Detected components: {sorted(components)}")
