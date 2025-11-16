import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, statistics

# === 1️⃣ Load model predictions ===
preds = pd.read_csv("LUAD_predictions.csv")

# Check variation in predicted survival probabilities
print("\n📈 Predicted Survival Probability Stats:")
print(preds["Predicted_Survival_Prob"].describe())
print("Unique values:", preds["Predicted_Survival_Prob"].unique()[:10])

# Ensure variation for visualization demo
if preds["Predicted_Survival_Prob"].nunique() <= 2:
    print("⚠️ Limited variation detected — adding small random noise for demo visualization.")
    np.random.seed(42)
    preds["Predicted_Survival_Prob"] = np.clip(
        preds["Predicted_Survival_Prob"].astype(float) +
        np.random.normal(0, 0.15, size=len(preds)), 0, 1
    )

# === 2️⃣ Load multimodal dataset ===
multi = pd.read_csv("LUAD_multimodal_dataset_with_paths.csv")

# Merge predictions with multimodal data
df = pd.merge(multi, preds[["PatientID", "Predicted_Survival_Prob"]], on="PatientID", how="inner")
print(f"\n✅ Merged LUAD dataset shape: {df.shape}")

# === 3️⃣ Simulate survival times based on survival_label (for demo) ===
np.random.seed(42)
long_mask = df["survival_label"].astype(str).str.lower() == "long"
short_mask = df["survival_label"].astype(str).str.lower() == "short"

df.loc[long_mask, "SurvivalTime"] = np.random.normal(1000, 150, long_mask.sum())
df.loc[short_mask, "SurvivalTime"] = np.random.normal(400, 80, short_mask.sum())
df["SurvivalTime"] = df["SurvivalTime"].clip(lower=1)
df["Event"] = 1  # All events observed for this synthetic example

# === 4️⃣ Define risk groups (median split, include equals in High Risk) ===
median_pred = df["Predicted_Survival_Prob"].median()
df["RiskGroup"] = np.where(df["Predicted_Survival_Prob"] >= median_pred, "High Risk", "Low Risk")

print("\n📊 Group counts (Predicted Risk):")
print(df["RiskGroup"].value_counts())

# === 5️⃣ Plot Kaplan–Meier Survival Curves ===
kmf = KaplanMeierFitter()
plt.figure(figsize=(8, 6))

for group, subdf in df.groupby("RiskGroup"):
    kmf.fit(subdf["SurvivalTime"], event_observed=subdf["Event"], label=group)
    kmf.plot_survival_function(ci_show=False)

plt.title("Kaplan–Meier Survival by Predicted Risk Group (LUAD Demo)")
plt.xlabel("Days (Simulated)")
plt.ylabel("Survival Probability")
plt.legend(title="Predicted Risk Group")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("deep_surv_km_curve_luad_demo.png", dpi=300)
plt.close()

# === 6️⃣ Log-Rank Test ===
high = df[df["RiskGroup"] == "High Risk"]
low = df[df["RiskGroup"] == "Low Risk"]

if not high.empty and not low.empty:
    result = statistics.logrank_test(
        high["SurvivalTime"], low["SurvivalTime"],
        event_observed_A=high["Event"], event_observed_B=low["Event"]
    )
    print(f"\n✅ KM survival curve saved → deep_surv_km_curve_luad_demo.png")
    print(f"Log-rank p-value (High vs Low): {result.p_value:.4f}")
else:
    print("⚠️ Not enough data to perform log-rank test.")

# === 7️⃣ (Optional) Save group summary ===
summary = df.groupby("RiskGroup")["SurvivalTime"].agg(["count", "mean", "median", "std"])
summary.to_csv("KM_summary_stats.csv", index=True)
print("\n📄 Summary stats saved → KM_summary_stats.csv")


# === 8️⃣ Histogram of Predicted Probabilities by Risk Group ===
plt.figure(figsize=(8, 5))
plt.hist(df[df["RiskGroup"]=="Low Risk"]["Predicted_Survival_Prob"], bins=10, alpha=0.7, label="Low Risk")
plt.hist(df[df["RiskGroup"]=="High Risk"]["Predicted_Survival_Prob"], bins=10, alpha=0.7, label="High Risk")
plt.title("Distribution of Predicted Survival Probabilities")
plt.xlabel("Predicted_Survival_Prob")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("deep_surv_prob_histogram.png", dpi=300)
plt.close()
print("📊 Histogram saved → deep_surv_prob_histogram.png")





