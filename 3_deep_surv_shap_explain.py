import os
import shap
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

# === Load dataset ===
df = pd.read_csv("LUAD_multimodal_dataset_with_paths.csv")
feature_cols = [c for c in df.columns if c not in ["PatientID", "survival_label", "Report", "ImagePath"]]
X = df[feature_cols].fillna(0).values
y = (df["survival_label"] == "Long").astype(int).values

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# === Model definition (same as training) ===
class DeepSurv(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# === Load model weights ===
model = DeepSurv(input_dim=X.shape[1]).to(device)
state_dict = torch.load("deep_surv_demo.pth", map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

# === Predict probabilities ===
with torch.no_grad():
    preds = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy().flatten()

df["Predicted_Survival_Prob"] = preds
df.to_csv("LUAD_predictions.csv", index=False)
print("✅ Predictions saved → LUAD_predictions.csv")

# === Compute SHAP values ===
print("💡 Computing SHAP feature importance ... (this may take 1–2 min)")
X_sample = torch.tensor(X[:min(100, len(X))], dtype=torch.float32).to(device)
explainer = shap.DeepExplainer(model, X_sample)
shap_values = explainer.shap_values(X_sample)

# Convert to numpy for plotting
# Convert SHAP values to 2-D array
if isinstance(shap_values, list):
    shap_values_np = shap_values[0].cpu().numpy()
else:
    shap_values_np = shap_values

# Remove the last singleton dimension (n_samples, n_features, 1 → n_samples, n_features)
if shap_values_np.ndim == 3:
    shap_values_np = shap_values_np.squeeze(-1)

# Save files
np.save("deep_surv_shap_values.npy", shap_values_np)
pd.DataFrame(shap_values_np, columns=feature_cols).to_csv("deep_surv_shap_values.csv", index=False)
print("✅ SHAP values saved → deep_surv_shap_values.csv")

print("✅ SHAP values saved → deep_surv_shap_values.csv")

# === SHAP summary plot ===
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_np, X_sample.cpu().numpy(), feature_names=feature_cols, show=False)
plt.title("Feature Importance – DeepSurv (SHAP Summary)")
plt.tight_layout()
plt.savefig("deep_surv_shap_summary.png", dpi=300)
print("✅ Summary plot saved → deep_surv_shap_summary.png")

print("🎯 Explainability complete! Files generated:")
print("  • LUAD_predictions.csv")
print("  • deep_surv_shap_values.csv")
print("  • deep_surv_shap_summary.png")
