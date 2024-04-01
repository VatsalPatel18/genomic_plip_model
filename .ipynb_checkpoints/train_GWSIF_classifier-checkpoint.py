import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import argparse
from classes.tile_classifier import SimpleNN, CustomDataset

# Setting up command line arguments
parser = argparse.ArgumentParser(description='Train a model on WSI data')
parser.add_argument('--data_dir', type=str, default='Datasets/', help='Base directory for dataset')
parser.add_argument('--batch_size', type=int, default=8000, help='Batch size for training')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate for optimizer')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train for')
parser.add_argument('--model_dir', type=str, default='models/', help='Directory to save the model')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading data
with open(os.path.join(args.data_dir, 'data.pkl'), 'rb') as f:
    data = pickle.load(f)

with open(os.path.join(args.data_dir, 'wsi_data.pkl'), 'rb') as f:
    wsi_data = pickle.load(f)

with open(os.path.join(args.data_dir, 'data_info.pkl'),'rb') as f:
    data_info = pickle.load(f)
    X_train = data_info['train']
    X_test = data_info['test']
    X_validate = data_info['validate']

train_data = CustomDataset(data['train'])
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

model = SimpleNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

wsi_metrics = {"true": [], "pred": []}

for epoch in range(args.epochs):
    wsi_metrics = {"true": [], "pred": []}

    for i, (x_batch, y_batch) in enumerate(train_dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Tile-based Loss at iteration {i + 1}: {loss.item()}")

    for pID in X_train:
        wsi_tile_data = torch.tensor(wsi_data[pID]['tiles'], dtype=torch.float32).to(device)
        wsi_outputs = model(wsi_tile_data)

        # Get the mean value
        wsi_mean_score = torch.mean(wsi_outputs).item()

        wsi_true_label = wsi_data[pID]['class']

        wsi_metrics["true"].append(wsi_true_label)
        wsi_metrics["pred"].append(wsi_mean_score)

    wsi_epoch_auc = roc_auc_score(wsi_metrics["true"], wsi_metrics["pred"])
    wsi_epoch_acc = accuracy_score(wsi_metrics["true"], [round(p) for p in wsi_metrics["pred"]])
    wsi_epoch_prec = precision_score(wsi_metrics["true"], [round(p) for p in wsi_metrics["pred"]])
    wsi_epoch_recall = recall_score(wsi_metrics["true"], [round(p) for p in wsi_metrics["pred"]])
    wsi_epoch_f1 = f1_score(wsi_metrics["true"], [round(p) for p in wsi_metrics["pred"]])

    print(f"WSI-specific metrics at epoch {epoch + 1}:")
    print(f"AUC: {wsi_epoch_auc}, Accuracy: {wsi_epoch_acc}, Precision: {wsi_epoch_prec}, Recall: {wsi_epoch_recall}, F1 Score: {wsi_epoch_f1}")
    
    
    for pID in X_validate:
        wsi_tile_data = torch.tensor(wsi_data[pID]['tiles'], dtype=torch.float32).to(device)
        wsi_outputs = model(wsi_tile_data)
        wsi_mean_score = torch.mean(wsi_outputs).item()
        wsi_true_label = wsi_data[pID]['class']

        wsi_metrics["true"].append(wsi_true_label)
        wsi_metrics["pred"].append(wsi_mean_score)


    wsi_epoch_auc = roc_auc_score(wsi_metrics["true"], wsi_metrics["pred"])
    wsi_epoch_acc = accuracy_score(wsi_metrics["true"], [round(p) for p in wsi_metrics["pred"]])
    wsi_epoch_prec = precision_score(wsi_metrics["true"], [round(p) for p in wsi_metrics["pred"]])
    wsi_epoch_recall = recall_score(wsi_metrics["true"], [round(p) for p in wsi_metrics["pred"]])
    wsi_epoch_f1 = f1_score(wsi_metrics["true"], [round(p) for p in wsi_metrics["pred"]])

    print(f"VALIDATE AUC: {wsi_epoch_auc}, Accuracy: {wsi_epoch_acc}, Precision: {wsi_epoch_prec}, Recall: {wsi_epoch_recall}, F1 Score: {wsi_epoch_f1}")


# Save the trained model
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
model_path = os.path.join(args.model_dir, 'pytorch_classifier4.pth')
torch.save(model.state_dict(), model_path)

for pID in X_test:
    wsi_tile_data = torch.tensor(wsi_data[pID]['tiles'], dtype=torch.float32).to(device)
    wsi_outputs = model(wsi_tile_data)
    wsi_mean_score = torch.mean(wsi_outputs).item()
    wsi_true_label = wsi_data[pID]['class']

    wsi_metrics["true"].append(wsi_true_label)
    wsi_metrics["pred"].append(wsi_mean_score)

wsi_epoch_auc = roc_auc_score(wsi_metrics["true"], wsi_metrics["pred"])
wsi_epoch_acc = accuracy_score(wsi_metrics["true"], [round(p) for p in wsi_metrics["pred"]])
wsi_epoch_prec = precision_score(wsi_metrics["true"], [round(p) for p in wsi_metrics["pred"]])
wsi_epoch_recall = recall_score(wsi_metrics["true"], [round(p) for p in wsi_metrics["pred"]])
wsi_epoch_f1 = f1_score(wsi_metrics["true"], [round(p) for p in wsi_metrics["pred"]])

print(f"TEST AUC: {wsi_epoch_auc}, Accuracy: {wsi_epoch_acc}, Precision: {wsi_epoch_prec}, Recall: {wsi_epoch_recall}, F1 Score: {wsi_epoch_f1}")

# Save wsi_metrics
metrics_path = os.path.join(args.model_dir, 'wsi_metrics.pkl')
with open(metrics_path, 'wb') as f:
    pickle.dump(wsi_metrics, f)

print(f"Saved model to {model_path}")
print(f"Saved metrics to {metrics_path}")
