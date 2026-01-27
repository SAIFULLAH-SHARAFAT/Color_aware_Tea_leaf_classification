import torch
import torch.nn as nn
import timm

class SwinRGBHSV(nn.Module):
    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = False,
        drop_path_rate: float = 0.0,
        num_classes: int = 7,
        use_hsv_branch: bool = False,
        hsv_embed_dim: int = 128,
        hsv_dropout: float = 0.1,
        fuse_dropout: float = 0.2,
        gate_hidden: int = 256,
        gate_vector: bool = False,
        img_mean = (0.485, 0.456, 0.406),
        img_std = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.use_hsv_branch = use_hsv_branch
        self.gate_vector = gate_vector

        self.register_buffer("img_mean", torch.tensor(img_mean).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(img_std).view(1, 3, 1, 1))

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        
        feat_dim = getattr(self.backbone, "num_features", 768)
        self.feat_dim = int(feat_dim)

        if self.use_hsv_branch:
             pass 
        else:
            self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x_norm: torch.Tensor):
        feat = self.backbone(x_norm)
        logits = self.classifier(feat)
        return logits