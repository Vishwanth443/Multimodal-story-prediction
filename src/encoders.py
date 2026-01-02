import torch
import torch.nn as nn
from torchvision.models import resnet18
from transformers import BertTokenizer, BertModel

class VisualEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        backbone = resnet18(weights="IMAGENET1K_V1")
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.fc = nn.Linear(512, embed_dim)

    def forward(self, images):
        B, T, C, H, W = images.shape
        images = images.view(B*T, C, H, W)

        with torch.no_grad():
            feats = self.backbone(images).squeeze()

        feats = self.fc(feats)
        return feats.view(B, T, -1).mean(dim=1)


class TextEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        for p in self.bert.parameters():
            p.requires_grad = False

        self.fc = nn.Linear(768, embed_dim)

    def forward(self, texts):
        device = next(self.parameters()).device

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = self.bert(**enc).pooler_output

        return self.fc(out)
