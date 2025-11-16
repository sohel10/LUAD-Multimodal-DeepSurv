import pandas as pd, random

# Load the LUAD-CT-Survival feature file
df = pd.read_csv("FeaturesWithLabels_1.csv")

# Rename RID → PatientID
df.rename(columns={"RID": "PatientID"}, inplace=True)

templates = [
    "CT scan shows {location} {size} mm mass, Stage {stage}, {response} to treatment.",
    "{location} lesion ({size} mm) consistent with Stage {stage} adenocarcinoma. {response}.",
    "Stage {stage} {location} tumor measuring {size} mm. {response} observed."
]

locations = ["left upper lobe", "right lower lobe", "left hilum", "right upper lobe"]
responses = ["partial response", "stable disease", "disease progression", "complete remission"]

def make_report(row):
    size = row.get("Longest.Diameter..mm.", random.randint(20, 50))
    # Randomly assign stage since dataset has no explicit stage column
    stage = random.choice(["II", "III", "IV"])
    location = random.choice(locations)
    response = random.choice(responses)
    return random.choice(templates).format(size=size, stage=stage, location=location, response=response)

df["Report"] = df.apply(make_report, axis=1)

# Save the synthetic reports
df[["PatientID", "Report"]].to_csv("radiology_reports.csv", index=False)

print("✅ File saved → radiology_reports.csv")
