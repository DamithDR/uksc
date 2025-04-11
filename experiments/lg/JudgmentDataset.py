from typing import List

from torch.utils.data import Dataset


class JudgmentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[str]):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]