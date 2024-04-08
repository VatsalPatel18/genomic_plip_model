import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import argparse
from classes.tile_classifier import SimpleNN, CustomDataset

def calculate_metrics(true_labels, predictions):
    auc = roc_auc_score(true_labels, predictions)
    acc = accuracy_score(true_labels, [round(p) for p in predictions])
    prec = precision_score(true_labels, [round(p) for p in predictions])
    recall = recall_score(true_labels, [round(p) for p in predictions])
    f1 = f1_score(true_labels, [round(p) for p in predictions])
    return auc, acc, prec, recall, f1

def evaluate_model(model, dataset, wsi_data, device):
    true_labels = []
    predictions = []
    for pID in dataset:
        wsi_tile_data = torch.tensor(wsi_data[pID]['tiles'], dtype=torch.float32).to(device)
        wsi_outputs = model(wsi_tile_data)
        wsi_mean_score = torch.mean(wsi_outputs).item()
        wsi_true_label = wsi_data[pID]['class']

        true_labels.append(wsi_true_label)
        predictions.append(wsi_mean_score)
    return true_labels, predictions

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

for epoch in range(args.epochs):
    model.train()
    for i, (x_batch, y_batch) in enumerate(train_dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Tile-based Loss at iteration {i + 1}: {loss.item()}")

    model.eval()
    with torch.no_grad():
        train_metrics = calculate_metrics(*evaluate_model(model, X_train, wsi_data, device))
        validate_metrics = calculate_metrics(*evaluate_model(model, X_validate, wsi_data, device))

    print(f"Epoch {epoch + 1} Training Metrics: AUC: {train_metrics[0]}, Accuracy: {train_metrics[1]}, Precision: {train_metrics[2]}, Recall: {train_metrics[3]}, F1 Score: {train_metrics[4]}")
    print(f"Epoch {epoch + 1} Validation Metrics: AUC: {validate_metrics[0]}, Accuracy: {validate_metrics[1]}, Precision: {validate_metrics[2]}, Recall: {validate_metrics[3]}, F1 Score: {validate_metrics[4]}")

# Save the trained model
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
model_path = os.path.join(args.model_dir, 'pytorch_classifier.pth')
torch.save(model.state_dict(), model_path)
print(f"Saved model to {modelpath}")

# Evaluate the model on the test set after all epochs
model.eval()
with torch.no_grad():
    test_metrics = calculate_metrics(*evaluate_model(model, X_test, wsi_data, device))

print(f"Test Metrics: AUC: {test_metrics[0]}, Accuracy: {test_metrics[1]}, Precision: {test_metrics[2]}, Recall: {test_metrics[3]}, F1 Score: {test_metrics[4]}")

# Save test metrics
metrics_path = os.path.join(args.model_dir, 'wsi_metrics.pkl')
with open(metrics_path, 'wb') as f:
    pickle.dump(test_metrics, f)
print(f"Saved metrics to {metrics_path}")
