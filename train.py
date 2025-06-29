import torch
import torch.nn as nn
import torch.optim as optim
from langchain_community.embeddings.dashscope import BATCH_SIZE
from torch.utils.data import DataLoader, TensorDataset, random_split
from nnue import HalfKANNUE
from dataset import HalfKA_Dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Initialize Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = HalfKANNUE()
model.to(device)

# Load Dataset
dataset = HalfKA_Dataset("dataset.pt")

# Split dataset
num_samples = len(dataset)
train_size = int(0.10 * num_samples)
val_size = int(0.10 * num_samples)
test_size = num_samples - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
)

BATCH_SIZE = 512
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

loss_fn = nn.MSELoss()
deviation_fn = nn.L1Loss()

# TensorBoard Writer
writer = SummaryWriter()


def train_nnue():
    EPOCHS = 20
    LR = 0.002

    # Loss and Optimizer
    best_val_loss = 0
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Training", unit="batch")

        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).squeeze()

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).squeeze()
                outputs = model(X_batch).squeeze()
                loss = loss_fn(outputs, y_batch)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader.dataset)

        if val_loss > best_val_loss:
            best_val_loss = val_loss
            model.save()

        writer.add_scalars('Loss', {'train': avg_train_loss, 'val': avg_val_loss}, epoch + 1)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")


def test_nnue():
    model.load()
    model.eval()
    test_loss = 0.0
    test_deviation = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).squeeze()
            outputs = model(X_batch).squeeze()
            loss = loss_fn(outputs, y_batch)
            deviation = deviation_fn(outputs, y_batch)
            test_loss += loss.item()
            test_deviation += deviation / 100

    n_batches = len(test_loader.dataset) / BATCH_SIZE
    avg_test_loss = test_loss / n_batches
    avg_test_deviation = test_deviation / n_batches
    print(f"Test Loss: {avg_test_loss:.3f}")
    print(f"Test Deviation: {avg_test_deviation:.3f}")
    writer.add_scalar('Test Loss', avg_test_loss)
    writer.add_scalar('Test Deviation', avg_test_deviation)


if __name__ == "__main__":
    train_nnue()
    test_nnue()
    writer.close()
