import torch, time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running GPU test on: {device} ({torch.cuda.get_device_name(0)})")

# Create two large matrices
a = torch.randn((10000, 10000), device=device)
b = torch.randn((10000, 10000), device=device)

# Warm-up
_ = torch.mm(a, b)

# Timed matrix multiplication
torch.cuda.synchronize()
t0 = time.time()
_ = torch.mm(a, b)
torch.cuda.synchronize()
t1 = time.time()

print(f"✅ Matrix multiply done in {t1 - t0:.3f} seconds on GPU.")

x = torch.randn(5000, 5000, device=device)
torch.matmul(x, x)
print("✅ GPU test: Matrix multiplication succeeded")


import pandas as pd

# Read TSV file
df = pd.read_csv("clinical.tsv", sep="\t")

# Save as CSV
df.to_csv("TCGA_LUAD_clinical.csv", index=False)


import pandas as pd

df = pd.read_csv("TCGA_LUAD_clinical.csv")
print(df.columns.tolist())
