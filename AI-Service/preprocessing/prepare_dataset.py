import os
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================================
# Load Dataset
# ==========================================================

DATASET_PATH = "datasets/complaints.csv"

df = pd.read_csv(DATASET_PATH)

print("Dataset Loaded Successfully")
print(f"Total Records : {len(df)}")

# ==========================================================
# Remove ID Column
# ==========================================================

df = df.drop(columns=["ID"])

print("ID column removed")

# ==========================================================
# Clean Complaint Text
# ==========================================================

df["Complaint"] = (
    df["Complaint"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

print("Complaint text cleaned")

# ==========================================================
# Encode Department Labels
# ==========================================================

label_encoder = LabelEncoder()

df["Label"] = label_encoder.fit_transform(df["Department"])

print("Departments encoded")

# ==========================================================
# Save Label Mapping
# ==========================================================

label_mapping = {
    department: int(label)
    for department, label in zip(
        label_encoder.classes_,
        label_encoder.transform(label_encoder.classes_)
    )
}

os.makedirs("datasets", exist_ok=True)

with open("datasets/label_mapping.json", "w") as f:
    json.dump(label_mapping, f, indent=4)

print("Label mapping saved")

# ==========================================================
# Split Dataset
# ==========================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["Label"]
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["Label"]
)

print("Dataset split completed")

# ==========================================================
# Save Files
# ==========================================================

train_df.to_csv("datasets/train.csv", index=False)
validation_df.to_csv("datasets/validation.csv", index=False)
test_df.to_csv("datasets/test.csv", index=False)

print("Files Saved Successfully")

print("\nSummary")
print("-" * 40)

print(f"Training Records   : {len(train_df)}")
print(f"Validation Records : {len(validation_df)}")
print(f"Testing Records    : {len(test_df)}")

print("\nDataset Preparation Completed Successfully!")