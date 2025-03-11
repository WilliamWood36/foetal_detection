import torch
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (list of tuples): Each tuple contains a 2D array of shape (4,) as input (mECG)
                                  and a single value as the target output (fECG).
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mecg, fecg = self.data[idx]
        mecg = torch.tensor(mecg, dtype=torch.float32)  # Convert to tensor
        fecg = torch.tensor(fecg, dtype=torch.float32)  # Convert target to tensor
        return mecg, fecg

# Example usage
# Assuming `ecg_data` is a list of tuples where each tuple is (mECG array, fECG value)
# dataset = ECGDataset(ecg_data)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
