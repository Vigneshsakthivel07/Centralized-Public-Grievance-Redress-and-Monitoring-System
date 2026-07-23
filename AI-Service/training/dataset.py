from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "models/distilbert_grievance"
)

# Load datasets
train_df = pd.read_csv("datasets/train.csv")
validation_df = pd.read_csv("datasets/validation.csv")

# Convert pandas → Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)

# Tokenization function
def tokenize(batch):
    return tokenizer(
        batch["Complaint"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Apply tokenizer
train_dataset = train_dataset.map(tokenize, batched=True)
validation_dataset = validation_dataset.map(tokenize, batched=True)

print(train_dataset)

print("\nFirst Sample:\n")
print(train_dataset[0])