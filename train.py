import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# ---- CNN Model Definition ----
class SudokuSolverCNN(nn.Module):
    def __init__(self, num_layers=16):
        super(SudokuSolverCNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(1, 512, kernel_size=3, padding=1))
        self.layers.append(nn.BatchNorm2d(512))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            self.layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
            self.layers.append(nn.BatchNorm2d(512))
            self.layers.append(nn.ReLU())
        self.final_conv = nn.Conv2d(512, 9, kernel_size=1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_conv(x)


def fit_scaler_encoder(df, sample_size=10000):
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    puzzles = np.array([list(p) for p in sample_df['puzzle']], dtype=np.int8).reshape(-1, 9, 9)
    solutions = np.array([list(s) for s in sample_df['solution']], dtype=np.int8).reshape(-1, 9, 9)
    puzzles_flat = puzzles.reshape(puzzles.shape[0], -1)
    solutions_flat = solutions.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(puzzles_flat)
    encoder = OneHotEncoder(categories=[range(1, 10)], sparse_output=False).fit(solutions_flat)
    return scaler, encoder

# ---- Convert Chunk to Tensors with pre-fitted scaler and encoder ----
def process_chunk(chunk, scaler, encoder):
    puzzles = np.array([list(p) for p in chunk['puzzle']], dtype=np.int8).reshape(-1, 9, 9)
    solutions = np.array([list(s) for s in chunk['solution']], dtype=np.int8).reshape(-1, 9, 9)
    puzzles_flat = puzzles.reshape(puzzles.shape[0], -1)
    solutions_flat = solutions.reshape(-1, 1)
    puzzles_scaled = scaler.transform(puzzles_flat).reshape(-1, 9, 9)
    solutions_encoded = encoder.transform(solutions_flat).reshape(-1, 9, 9, 9)
    X = torch.tensor(puzzles_scaled, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(solutions_encoded, dtype=torch.float32).permute(0, 3, 1, 2)
    return TensorDataset(X, y)


def train_model(args):
    model = SudokuSolverCNN(num_layers=16).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.resume_from:
        print(f"Resuming training from checkpoint: {args.resume_from}")
        model.load_state_dict(torch.load(args.resume_from, map_location=args.device))

    csv_path = os.path.join(args.data_dir, 'sudoku_sampled_1M.csv')
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset shape: {df.shape}")

    # Split into train/val
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)
    print(f"Train: {train_df.shape}, Validation: {val_df.shape}")

    # Fit scaler/encoder on train only
    scaler, encoder = fit_scaler_encoder(train_df)

    # Prepare validation set DataLoader (process all at once)
    val_dataset = process_chunk(val_df, scaler, encoder)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    # Chunked training for train set
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        chunk_iterator = np.array_split(train_df, np.ceil(len(train_df) / args.chunk_size))
        all_predictions, all_labels = [], []

        for i, chunk in enumerate(chunk_iterator):
            print(f"Processing chunk {i+1}...")
            dataset = process_chunk(chunk, scaler, encoder)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

            model.train()
            for puzzles, solutions in dataloader:
                puzzles, solutions = puzzles.to(args.device), solutions.to(args.device)
                outputs = model(puzzles)
                loss = criterion(outputs, torch.argmax(solutions, dim=1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Accumulate predictions and labels for accuracy after epoch
            model.eval()
            with torch.no_grad():
                for puzzles, solutions in dataloader:
                    puzzles, solutions = puzzles.to(args.device), solutions.to(args.device)
                    outputs = model(puzzles)
                    _, predicted = torch.max(outputs, 1)
                    _, labels = torch.max(solutions, 1)
                    all_predictions.append(predicted.cpu())
                    all_labels.append(labels.cpu())

        # Compute training accuracy after epoch
        if all_predictions:
            all_preds = torch.cat(all_predictions)
            all_labs = torch.cat(all_labels)
            total_final = all_labs.numel()
            correct_final = (all_preds == all_labs).sum().item()
            final_accuracy = 100 * correct_final / total_final
            print(f"\n✅ Epoch {epoch+1} Training Accuracy: {final_accuracy:.2f}%")

        # ---- Validation after each epoch ----
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for puzzles, solutions in val_loader:
                puzzles, solutions = puzzles.to(args.device), solutions.to(args.device)
                outputs = model(puzzles)
                _, predicted = torch.max(outputs, 1)
                _, labels = torch.max(solutions, 1)
                val_total += labels.numel()
                val_correct += (predicted == labels).sum().item()
        val_accuracy = 100 * val_correct / val_total
        print(f"🧪 Epoch {epoch+1} Validation Accuracy: {val_accuracy:.2f}%")

        # Save checkpoint once per epoch
        checkpoint_dir = os.environ.get('SM_CHECKPOINT_DIR', args.model_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch{epoch}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    print("Final model saved to model.pth")

# ---- CLI Entry Point ----
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume-from', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--chunk-size', type=int, default=20000, help='Number of rows per chunk')
    args = parser.parse_args()
    train_model(args)

