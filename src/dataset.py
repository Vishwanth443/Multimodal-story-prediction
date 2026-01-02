import re
from torch.utils.data import Dataset

class WrappedStoryDataset(Dataset):
    def __init__(self, split="train", max_frames=2):
        from data.storyreasoning import StoryReasoningDataset
        self.base = StoryReasoningDataset(split=split, max_frames=max_frames)

        self.actions = [
            "enter", "stand", "look", "walk", "talk",
            "sit", "aim", "open", "close", "move"
        ]
        self.label_map = {a: i for i, a in enumerate(self.actions)}

    def clean(self, text):
        return re.sub(r"<[^>]+>", "", text).strip().lower()

    def extract_action(self, text):
        for a in self.actions:
            if a in text:
                return self.label_map[a]
        return self.label_map["stand"]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        images = sample["images"]
        text = self.clean(sample["text"])
        label = self.extract_action(text)
        return images, text, label
