import os
import pandas as pd

# === Step 1: Load structured + report data ===
df_features = pd.read_csv("FeaturesWithLabels_1.csv")
df_reports = pd.read_csv("radiology_reports.csv")

# Rename to consistent PatientID column
df_features.rename(columns={"RID": "PatientID"}, inplace=True)

# Merge on PatientID
df = pd.merge(df_features, df_reports, on="PatientID", how="left")

# === Step 2: Add image paths ===
def find_image_path(pid):
    import re
    base_dir = os.getcwd()
    pid = str(pid).strip().upper()
    nii_files = [f for f in os.listdir(base_dir) if f.endswith(".nii")]

    # Try direct substring match
    for f in nii_files:
        if pid in f.upper():
            return os.path.join(base_dir, f)

    # Handle patterns like R0004 ↔ R_004
    if pid.startswith("R") and pid[1:].isdigit():
        pid_num = int(pid[1:])
        patterns = [f"R_{pid_num:03d}.nii", f"R{pid_num:04d}.nii", f"R{pid_num:03d}.nii"]
        for pattern in patterns:
            for f in nii_files:
                if re.fullmatch(pattern.replace(".nii", "") + r"\.nii", f, re.IGNORECASE):
                    return os.path.join(base_dir, f)

    if pid.startswith("QIN-LSC"):
        for f in nii_files:
            if pid in f.upper():
                return os.path.join(base_dir, f)

    return None

df["ImagePath"] = df["PatientID"].apply(find_image_path)

# === Step 3: Save merged data ===
df.to_csv("LUAD_multimodal_dataset.csv", index=False)
df.to_csv("LUAD_multimodal_dataset_with_paths.csv", index=False)

print("\n✅ Merged dataset saved:")
print(" - LUAD_multimodal_dataset.csv")
print(" - LUAD_multimodal_dataset_with_paths.csv")
print(f"📊 Shape: {df.shape}\n")
print("Sample preview:\n", df.head())

matched = df["ImagePath"].notna().sum()
print(f"\n🧠 Matched images: {matched} out of {len(df)}")
if matched == len(df):
    print("🎉 All image paths successfully matched!")
else:
    missing = df.loc[df['ImagePath'].isna(), 'PatientID'].tolist()
    print("⚠️ Missing matches for:", missing[:10], "..." if len(missing) > 10 else "")
