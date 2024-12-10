from torch.utils.data import Dataset
import torch
import numpy as np
class GraphDataset(Dataset):
    def __init__(self, data, labels, word_to_idx, transform = None) -> None:
        super(GraphDataset, self).__init__()
        self.data = data
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            text = self.transform(
                text, self.word_to_idx
            )
        text = torch.tensor(text)

        return text, label