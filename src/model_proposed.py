import torch.nn as nn
from src.encoders import VisualEncoder, TextEncoder

class ProposedModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.visual = VisualEncoder(cfg["hidden_dim"])
        self.text = TextEncoder(cfg["hidden_dim"])

        self.attn = nn.MultiheadAttention(
            embed_dim=cfg["hidden_dim"],
            num_heads=4,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(cfg["hidden_dim"], 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, texts):
        v = self.visual(images).unsqueeze(1)   # (B,1,D)
        t = self.text(texts).unsqueeze(1)       # (B,1,D)

        attn_out, _ = self.attn(t, v, v)
        return self.classifier(attn_out.squeeze(1))
