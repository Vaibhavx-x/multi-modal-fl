"""
model.py — Multimodal skin lesion classifier for PAD-UFES-20.

Architecture
------------
    image   (B, 3, H, W)  →  VisionBranch  (ResNet-18)  →  (B, 512) ─┐
                                                                        ├→ concat (B, 576) → ClassificationHead → logits (B, 6)
    tabular (B, D)         →  TabularBranch (MLP)         →  (B,  64) ─┘

Both branches are registered nn.Module children, so model.to(device) moves
everything automatically. No .to() calls are made inside forward().
"""

# pyrefly: ignore [missing-import]
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


# Dimension constants — single source of truth
VISION_OUT_DIM  = 512   # ResNet-18 avgpool output
TABULAR_OUT_DIM = 64    # MLP output
FUSION_DIM      = VISION_OUT_DIM + TABULAR_OUT_DIM  # 576
NUM_CLASSES     = 6     # ACK, BCC, MEL, NEV, SCC, SEK


class VisionBranch(nn.Module):
    """
    Pre-trained ResNet-18 used as a feature extractor.

    The classification head is replaced with nn.Identity so the network
    outputs the 512-dim global average-pooled representation.

    Args:
        freeze_backbone: If True, all ResNet weights are frozen.
    """

    def __init__(self, freeze_backbone: bool = False):
        super().__init__()
        backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()  # remove 1000-class head, output (B, 512)
        self.backbone = backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # (B, 3, H, W) → (B, 512)


class TabularBranch(nn.Module):
    """
    Two-layer MLP that maps clinical feature vectors to a 64-dim embedding.

    Architecture: Linear → BN → ReLU → Dropout → Linear → BN → ReLU

    BatchNorm is used because batch statistics are stable with fixed
    preprocessing, and it regularises well on small per-client batches.

    Args:
        in_dim: Dimensionality of the input tabular vector.
        hidden_dim: Width of the intermediate projection layer.
        out_dim: Output embedding size.
        dropout: Dropout probability after the first BN+ReLU.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = TABULAR_OUT_DIM,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),  # bias=False: BN absorbs it
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # (B, in_dim) → (B, 64)


class ClassificationHead(nn.Module):
    """
    Shallow linear head with dropout that maps the fused vector to class logits.

    Kept deliberately shallow — the two branches already carry enough
    representational power; a deep head would overfit on small per-client batches.
    """

    def __init__(self, in_dim: int = FUSION_DIM, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)  # (B, FUSION_DIM) → (B, num_classes)


class MultiModalClassifier(nn.Module):
    """
    Multimodal skin lesion classifier for PAD-UFES-20.

    Args:
        tabular_in_dim: Width of the tabular feature tensor (see PADUFES20Dataset.feature_dim).
        num_classes: Number of output classes (default 6).
        tabular_hidden: Hidden width of the tabular MLP.
        dropout: Shared dropout rate for the MLP and classification head.
        freeze_backbone: Whether to freeze ResNet-18 weights.

    Forward:
        image   (B, 3, H, W)        — normalised image batch
        tabular (B, tabular_in_dim) — encoded clinical features
        Returns logits (B, num_classes) — pass directly to nn.CrossEntropyLoss.
    """

    def __init__(
        self,
        tabular_in_dim: int,
        num_classes: int = NUM_CLASSES,
        tabular_hidden: int = 128,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.vision_branch  = VisionBranch(freeze_backbone=freeze_backbone)
        self.tabular_branch = TabularBranch(in_dim=tabular_in_dim, hidden_dim=tabular_hidden,
                                            out_dim=TABULAR_OUT_DIM, dropout=dropout)
        self.classifier = ClassificationHead(in_dim=FUSION_DIM, num_classes=num_classes, dropout=dropout)

        # Expose dims for downstream inspection
        self.vision_out_dim  = VISION_OUT_DIM
        self.tabular_out_dim = TABULAR_OUT_DIM
        self.fusion_dim      = FUSION_DIM

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        vision_feat  = self.vision_branch(image)      # (B, 512)
        tabular_feat = self.tabular_branch(tabular)   # (B,  64)
        fused        = torch.cat([vision_feat, tabular_feat], dim=1)  # (B, 576)
        return self.classifier(fused)                 # (B, num_classes)


def build_model(
    tabular_in_dim: int,
    num_classes: int = NUM_CLASSES,
    tabular_hidden: int = 128,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
    device: str = "cpu",
) -> MultiModalClassifier:
    """
    Convenience factory: instantiate MultiModalClassifier and move to device.

    Example:
        >>> from model import build_model
        >>> model = build_model(tabular_in_dim=61, device="cuda")
    """
    return MultiModalClassifier(
        tabular_in_dim=tabular_in_dim,
        num_classes=num_classes,
        tabular_hidden=tabular_hidden,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    ).to(device)


def count_parameters(model: nn.Module) -> dict:
    """Return a dict with total, trainable, and frozen parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


# Smoke-test: python model.py
if __name__ == "__main__":
    TABULAR_DIM = 61  # confirmed by dataset.py
    BATCH       = 4
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = build_model(tabular_in_dim=TABULAR_DIM, device=device)

    dummy_image   = torch.randn(BATCH, 3, 224, 224, device=device)
    dummy_tabular = torch.randn(BATCH, TABULAR_DIM, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(dummy_image, dummy_tabular)

    print(f"Input  image   : {dummy_image.shape}")
    print(f"Input  tabular : {dummy_tabular.shape}")
    print(f"Output logits  : {logits.shape}  (expected: [{BATCH}, {NUM_CLASSES}])")

    params = count_parameters(model)
    print(f"\nParameters — total: {params['total']:,} | trainable: {params['trainable']:,} | frozen: {params['frozen']:,}")

    assert logits.shape == (BATCH, NUM_CLASSES), "Shape mismatch!"
    print("OK -- forward pass shape check passed.")
