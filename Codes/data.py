from torch.utils.data import Dataset

class MFCCDataset(Dataset):
    def __init__(self, mfccs, labels, indices):
        self.mfccs = mfccs
        self.labels = labels
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        x = self.mfccs[real_idx]
        y = self.labels[real_idx]

        return x, y
