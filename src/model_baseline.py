import torch.nn as nn
from src.encoders import VisualEncoder, TextEncoder
import torch        

class BaselineModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.visual = VisualEncoder(cfg["hidden_dim"])
        self.text = TextEncoder(cfg["hidden_dim"])

        self.classifier = nn.Sequential(
            nn.Linear(cfg["hidden_dim"] * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, texts):
        v = self.visual(images)
        t = self.text(texts)
        fused = torch.cat([v, t], dim=1)
        return self.classifier(fused)
