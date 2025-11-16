import os
import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler

# === 1️⃣ Load and preprocess dataset ===
df = pd.read_csv("LUAD_multimodal_dataset_with_paths.csv")
feature_cols = [c for c in df.columns if c not in ["PatientID", "survival_label", "Report", "ImagePath"]]

# Standardize input features
scaler = StandardScaler()
X = scaler.fit_transform(df[feature_cols].fillna(0).values)

# Binary encode survival label
y = (df["survival_label"] == "Long").astype(int).values.reshape(-1, 1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# === 2️⃣ Define model architecture ===
class DeepSurv(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# === 3️⃣ Initialize model, optimizer, and loss ===
model = DeepSurv(input_dim=X.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# === 4️⃣ Create DataLoaders (train/val split) ===
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

# === 5️⃣ Load checkpoint if available ===
if os.path.exists("deep_surv_demo.pth"):
    print("🔁 Found existing checkpoint — loading weights...")
    model.load_state_dict(torch.load("deep_surv_demo.pth", map_location=device))
else:
    print("🆕 No checkpoint found — starting fresh training.")

# === 6️⃣ Train model ===
num_epochs = 100
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# === 7️⃣ Save trained model ===
torch.save(model.state_dict(), "deep_surv_demo.pth")
print("✅ Model training complete. Saved as deep_surv_demo.pth")

# === 8️⃣ Plot training vs validation loss ===
plt.figure(figsize=(6, 4))
plt.plot(train_losses, label="Train Loss", marker="o")
plt.plot(val_losses, label="Validation Loss", marker="x")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("DeepSurv Training Curve (GPU)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("deep_surv_loss_curve.png")
plt.show()

# === 9️⃣ Quick sanity check ===
with torch.no_grad():
    preds = model(torch.tensor(X[:5], dtype=torch.float32).to(device))
    print("\nSample predictions:", preds.cpu().numpy().round(3).flatten())
    print("Sample true labels:", y[:5].flatten())
