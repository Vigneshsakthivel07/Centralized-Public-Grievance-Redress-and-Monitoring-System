import pandas as pd

# Load dataset
df = pd.read_csv("datasets/complaints.csv")

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"\nTotal Records: {len(df)}")
print(f"Total Columns: {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nFirst 5 Rows:")
print(df.head())

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())

print("\n" + "=" * 60)
print("DUPLICATE RECORDS")
print("=" * 60)

print(df.duplicated().sum())

print("\n" + "=" * 60)
print("DEPARTMENT DISTRIBUTION")
print("=" * 60)

print(df["Department"].value_counts())

print("\n" + "=" * 60)
print("NUMBER OF DEPARTMENTS")
print("=" * 60)

print(df["Department"].nunique())

print("\nDepartment Names:")

for dept in sorted(df["Department"].unique()):
    print("-", dept)

print("\n" + "=" * 60)
print("COMPLAINT LENGTH")
print("=" * 60)

df["Complaint_Length"] = df["Complaint"].str.len()

print(df["Complaint_Length"].describe())
