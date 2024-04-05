import os
import torch
from torch.utils.data import Dataset

class FlatTileDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        # List all files in the data_dir that are files (not directories)
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    def __len__(self):
        # Return the total number of files
        return len(self.files)

    def __getitem__(self, idx):
        # Get the file path for the given index
        file_path = self.files[idx]
        # Load the data from the file
        data = torch.load(file_path)
        # Assuming the data file is a dictionary with 'tile_data' and 'file_data' keys
        tile_data = torch.from_numpy(data['tile_data'][0])
        file_data = data['file_data']
        # Return the tile data and file data
        return tile_data, file_data
