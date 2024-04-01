import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
import argparse
from classes.genomic_plip_model import GenomicPLIPModel
from transformers import CLIPVisionModel

class PatientTileDataset(Dataset):
    def __init__(self, data_dir, model, save_dir):
        super().__init__()
        self.data_dir = data_dir
        self.model = model
        self.save_dir = Path(save_dir)
        self.files = []
        for patient_id in os.listdir(data_dir):
            patient_dir = os.path.join(data_dir, patient_id)
            if os.path.isdir(patient_dir):
                for f in os.listdir(patient_dir):
                    if f.endswith('.pt'):
                        self.files.append((os.path.join(patient_dir, f), patient_id))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, patient_id = self.files[idx]
        data = torch.load(file_path)
        tile_data = torch.from_numpy(data['tile_data'][0]).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            vision_features, _ = self.model(pixel_values=tile_data, score_vector=torch.zeros(1, 4))
        feature_path = self.save_dir / patient_id / os.path.basename(file_path)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vision_features, feature_path)
        return feature_path

def extract_features(data_dir, save_dir, model_path):
    original_model = CLIPVisionModel.from_pretrained("./plip/")
    custom_model = GenomicPLIPModel(original_model)
    custom_model.load_state_dict(torch.load(model_path))
    custom_model.eval()

    dataset = PatientTileDataset(data_dir=data_dir, model=custom_model, save_dir=save_dir)
    for _ in dataset:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from genomic aligned tiles.")
    parser.add_argument('--data_dir', type=str, default='plip_preprocess/', help='Directory containing the patient data.')
    parser.add_argument('--save_dir', type=str, default='kaal_extract/', help='Directory to save the extracted features.')
    parser.add_argument('--model_path', type=str, default='genomic_plip.pth', help='Path to the trained model file.')

    args = parser.parse_args()

    extract_features(args.data_dir, args.save_dir, args.model_path)
