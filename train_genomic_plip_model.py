import torch
import argparse
from torch import optim
from torch.utils.data import DataLoader
from classes.genomic_plip_model import GenomicPLIPModel
from classes.tile_file_dataloader import FlatTileDataset
from transformers import CLIPVisionModel

def train_model(data_dir, model_save_path, pretrained_model_path, lr, num_epochs, train_batch_size, validation_batch_size, num_workers):

    # Load datasets   
    train_dataset = FlatTileDataset(data_dir=f'{data_dir}/train')
    train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)

    validation_dataset = FlatTileDataset(data_dir=f'{data_dir}/validate')
    validation_data_loader = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=False, num_workers=num_workers)

    # Initialize the model
    base_model = CLIPVisionModel.from_pretrained(pretrained_model_path)
    custom_model = GenomicPLIPModel(base_model)

    criterion = torch.nn.CosineSimilarity(dim=1)
    optimizer = optim.Adam(custom_model.parameters(), lr=lr)


    for epoch in range(num_epochs):
        # Training loop
        custom_model.train()
        train_loss = 0.0

        for batch_images, batch_scores in train_data_loader:
            optimizer.zero_grad()

            batch_loss = 0
            for img, score in zip(batch_images, batch_scores):
                vision_features, score_features = custom_model(img.unsqueeze(0), score.unsqueeze(0))
                cos_sim = criterion(score_features, vision_features)
                loss = -cos_sim.mean()

                batch_loss += loss.item()
                loss.backward()

            optimizer.step()
            train_loss += batch_loss
            print(f"Batch part loss {batch_loss:.4f}")

        avg_train_loss = train_loss / len(train_data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        # Validation loop
        custom_model.eval()
        validation_loss = 0.0

        with torch.no_grad():
            for batch_images, batch_scores in validation_data_loader:
                batch_loss = 0
                for img, score in zip(batch_images, batch_scores):
                    vision_features, score_features = custom_model(img.unsqueeze(0), score.unsqueeze(0))
                    cos_sim = criterion(score_features, vision_features)
                    loss = -cos_sim.mean()

                    batch_loss += loss.item()

                validation_loss += batch_loss
                print(f"Validation Batch part loss {batch_loss:.4f}")

            avg_validation_loss = validation_loss / len(validation_data_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_validation_loss:.4f}")

    # Save the trained model
    torch.save(custom_model.state_dict(), model_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the Genomic PLIP Model')
    parser.add_argument('--data_dir', type=str, default='Datasets/train_03', help='Directory containing the train, validate, and test datasets.')
    parser.add_argument('--model_save_path', type=str, default='genomic_plip.pth', help='Path to save the trained model.')
    parser.add_argument('--pretrained_model_path', type=str, default='./plip', help='Path to the pretrained CLIP model.')
    
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs to train for.')
    parser.add_argument('--train_batch_size', type=int, default=128, help='Batch size for the training data loader.')
    parser.add_argument('--validation_batch_size', type=int, default=128, help='Batch size for the validation data loader.')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of worker threads for data loading.')


    args = parser.parse_args()

    train_model(args.data_dir, args.model_save_path, args.pretrained_model_path, args.lr, args.num_epochs, args.train_batch_size, args.validation_batch_size, args.num_workers)

