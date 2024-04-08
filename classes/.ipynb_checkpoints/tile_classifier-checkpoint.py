import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class CustomDataset(Dataset):
    def __init__(self):
        self.X = torch.tensor(Dataset['X'], dtype=torch.float32)
        self.Y = torch.tensor(Dataset['Y'], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
