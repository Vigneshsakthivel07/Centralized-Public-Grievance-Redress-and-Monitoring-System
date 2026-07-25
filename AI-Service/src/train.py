import os
import json
import random

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm.auto import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

DATA_PATH = "datasets/complaints.csv"
MODEL_OUTPUT = "models/complaint_classifier"

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
RANDOM_STATE = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("DEVICE:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\nOriginal dataset:")
print(df.shape)
print(df.columns.tolist())


# ============================================================
# CLEAN DATA
# ============================================================

required_columns = ["Complaint", "Department"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )

df = df[required_columns].copy()

df["Complaint"] = (
    df["Complaint"]
    .astype(str)
    .str.strip()
)

df["Department"] = (
    df["Department"]
    .astype(str)
    .str.strip()
)

# Remove empty complaints
df = df[
    (df["Complaint"] != "") &
    (df["Department"] != "")
]

# Remove exact duplicate complaint texts
df = df.drop_duplicates(
    subset=["Complaint"],
    keep="first"
).reset_index(drop=True)


print("\nAfter cleaning:")
print("Rows:", len(df))
print("Unique complaints:", df["Complaint"].nunique())
print("Departments:", df["Department"].nunique())


# ============================================================
# LABEL MAPPING
# ============================================================

departments = sorted(
    df["Department"].unique()
)

label2id = {
    department: index
    for index, department in enumerate(departments)
}

id2label = {
    index: department
    for department, index in label2id.items()
}

df["labels"] = df["Department"].map(label2id)


print("\nLabel Mapping:")

for index, department in id2label.items():
    print(index, "->", department)


# ============================================================
# SAVE LABEL MAPPING
# ============================================================

os.makedirs(MODEL_OUTPUT, exist_ok=True)

with open(
    f"{MODEL_OUTPUT}/label_mapping.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            "label2id": label2id,
            "id2label": {
                str(k): v
                for k, v in id2label.items()
            }
        },
        file,
        indent=4
    )


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["labels"],
    random_state=RANDOM_STATE
)

valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["labels"],
    random_state=RANDOM_STATE
)


print("\nDataset Split:")
print("Train:", len(train_df))
print("Validation:", len(valid_df))
print("Test:", len(test_df))


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# DATASET CLASS
# ============================================================

class ComplaintDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length
    ):

        self.dataframe = (
            dataframe
            .reset_index(drop=True)
        )

        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):

        return len(self.dataframe)


    def __getitem__(self, index):

        complaint = self.dataframe.loc[
            index,
            "Complaint"
        ]

        label = int(
            self.dataframe.loc[
                index,
                "labels"
            ]
        )

        encoding = self.tokenizer(
            complaint,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding[
                "input_ids"
            ].squeeze(0),

            "attention_mask": encoding[
                "attention_mask"
            ].squeeze(0),

            "labels": torch.tensor(
                label,
                dtype=torch.long
            )
        }


# ============================================================
# DATASET OBJECTS
# ============================================================

train_dataset = ComplaintDataset(
    train_df,
    tokenizer,
    MAX_LENGTH
)

valid_dataset = ComplaintDataset(
    valid_df,
    tokenizer,
    MAX_LENGTH
)

test_dataset = ComplaintDataset(
    test_df,
    tokenizer,
    MAX_LENGTH
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

num_labels = len(label2id)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

model.to(device)


print("\nModel:")
print(model.config.architectures)
print("Number of labels:", model.config.num_labels)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# SCHEDULER
# ============================================================

total_steps = (
    len(train_loader) * EPOCHS
)

warmup_steps = int(
    total_steps * 0.1
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(model, dataloader):

    model.eval()

    total_loss = 0

    predictions = []
    actual_labels = []

    with torch.no_grad():

        for batch in tqdm(
            dataloader,
            desc="Evaluating"
        ):

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += (
                outputs.loss.item()
            )

            preds = torch.argmax(
                outputs.logits,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            actual_labels.extend(
                labels.cpu().numpy()
            )

    accuracy = accuracy_score(
        actual_labels,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            actual_labels,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": predictions,
        "labels": actual_labels
    }


# ============================================================
# TRAINING
# ============================================================

best_f1 = -1

for epoch in range(EPOCHS):

    print("\n")
    print("=" * 70)
    print(
        f"EPOCH {epoch + 1}/{EPOCHS}"
    )
    print("=" * 70)

    model.train()

    total_loss = 0

    progress = tqdm(
        train_loader,
        desc="Training"
    )

    for batch in progress:

        optimizer.zero_grad()

        input_ids = batch[
            "input_ids"
        ].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        scheduler.step()

        total_loss += loss.item()

        progress.set_postfix(
            loss=loss.item()
        )


    # ========================================================
    # VALIDATION
    # ========================================================

    validation = evaluate(
        model,
        valid_loader
    )

    print(
        f"\nValidation Loss: "
        f"{validation['loss']:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{validation['accuracy']:.4f}"
    )

    print(
        f"Validation F1: "
        f"{validation['f1']:.4f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if validation["f1"] > best_f1:

        best_f1 = validation["f1"]

        model.save_pretrained(
            MODEL_OUTPUT
        )

        tokenizer.save_pretrained(
            MODEL_OUTPUT
        )

        print(
            "\n✅ BEST MODEL SAVED"
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_OUTPUT
)

model.to(device)


# ============================================================
# FINAL TEST
# ============================================================

test_results = evaluate(
    model,
    test_loader
)

print("\n")
print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"Accuracy : "
    f"{test_results['accuracy']:.4f}"
)

print(
    f"Precision: "
    f"{test_results['precision']:.4f}"
)

print(
    f"Recall   : "
    f"{test_results['recall']:.4f}"
)

print(
    f"F1 Score : "
    f"{test_results['f1']:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        test_results["labels"],
        test_results["predictions"],
        target_names=[
            id2label[i]
            for i in range(num_labels)
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

matrix = confusion_matrix(
    test_results["labels"],
    test_results["predictions"]
)

print("\nConfusion Matrix:")
print(matrix)


print("\n✅ TRAINING COMPLETE")
print(
    f"Model saved at: {MODEL_OUTPUT}"
)