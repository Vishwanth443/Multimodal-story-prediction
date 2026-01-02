import torch
from datasets import load_dataset
from torchvision import transforms

class StoryReasoningDataset(torch.utils.data.Dataset):
    def __init__(self, split="train", max_frames=5):
        self.ds = load_dataset(
            "daniel3303/StoryReasoning",
            split=split
        )
        self.max_frames = max_frames

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]

        # Images are ALREADY PIL Images
        images = sample["images"][:self.max_frames]
        story = sample["story"]

        # Transform images correctly
        img_tensor = torch.stack([
            self.transform(img.convert("RGB"))
            for img in images
        ])

        # Text sequence (K -> K+1)
        input_text = story[:-1]
        target_text = story[-1]

        return {
            "images": img_tensor,        # (T, 3, 224, 224)
            "text": input_text,          # list[str]
            "target_text": target_text   # str
        }
