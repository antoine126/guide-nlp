"""A Dataset of variable-length documents and its padding collate function (chapter 16)."""

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PAD = 0


class TextDataset(Dataset):
    def __init__(self, documents: list[list[int]], labels: list[int]) -> None:
        self.documents = [torch.tensor(d) for d in documents]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.documents[index], self.labels[index]


def collate(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    documents, labels = zip(*batch, strict=True)
    padded = pad_sequence(list(documents), batch_first=True, padding_value=PAD)
    return padded, torch.tensor(labels)


def make_loader(
    documents: list[list[int]], labels: list[int], batch_size: int = 32, shuffle: bool = True
) -> DataLoader:
    dataset = TextDataset(documents, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
