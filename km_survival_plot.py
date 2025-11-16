import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, statistics

# 1️⃣ Load your model predictions and multimodal features
preds = pd.read_csv("LUAD_predictions.csv")  # has PatientID, Predicted_Survival_Prob, etc.
multi = pd.read_csv("LUAD_multimodal_dataset_with_paths.csv")  # has PatientID, survival_label, features...

# Merge on PatientID
df = pd.merge(multi, preds[["PatientID", "Predicted_Survival_Prob"]], on="PatientID", how="inner")
print("✅ Merged LUAD dataset shape:", df.shape)
print(df[["PatientID", "survival_label", "Predicted_Survival_Prob"]].head())

# 2️⃣ Create synthetic survival times based on survival_label
#    (just for demonstration / portfolio)
np.random.seed(42)

# Long survival → higher mean time
long_mask = df["survival_label"].astype(str).str.lower() == "long"
short_mask = df["survival_label"].astype(str).str.lower() == "short"

df.loc[long_mask, "SurvivalTime"] = np.random.normal(loc=1000, scale=150, size=long_mask.sum())
df.loc[short_mask, "SurvivalTime"] = np.random.normal(loc=350, scale=80, size=short_mask.sum())

# Ensure times are positive
df["SurvivalTime"] = df["SurvivalTime"].clip(lower=1)

# Assume all observed events (no censoring) for this demo
df["Event"] = 1

# 3️⃣ Define high vs low predicted risk from your DeepSurv model
median_pred = df["Predicted_Survival_Prob"].median()
df["RiskGroup"] = (df["Predicted_Survival_Prob"] > median_pred).map({True: "High Risk", False: "Low Risk"})

print("\n📊 Group counts (by predicted risk):")
print(df["RiskGroup"].value_counts())

# 4️⃣ KM survival analysis
kmf = KaplanMeierFitter()

plt.figure(figsize=(8, 6))
for group, df_group in df.groupby("RiskGroup"):
    kmf.fit(df_group["SurvivalTime"], event_observed=df_group["Event"], label=group)
    kmf.plot_survival_function(ci_show=False)

plt.title("Kaplan–Meier Survival by Predicted Risk Group (LUAD Demo)")
plt.xlabel("Days (simulated)")
plt.ylabel("Survival Probability")
plt.legend(title="Predicted Risk Group")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("deep_surv_km_curve_luad_demo.png", dpi=300)
plt.close()

# 5️⃣ Log-rank test between high and low risk
high = df[df["RiskGroup"] == "High Risk"]
low = df[df["RiskGroup"] == "Low Risk"]

if not high.empty and not low.empty:
    result = statistics.logrank_test(
        high["SurvivalTime"],
        low["SurvivalTime"],
        event_observed_A=high["Event"],
        event_observed_B=low["Event"]
    )
    print(f"\n✅ KM survival curve saved → deep_surv_km_curve_luad_demo.png")
    print(f"Log-rank p-value (High vs Low risk): {result.p_value:.4f}")
else:
    print("⚠️ Not enough data to perform log-rank test.")
