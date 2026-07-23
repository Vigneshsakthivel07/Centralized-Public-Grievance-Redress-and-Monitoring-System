import json
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

train_df = pd.read_csv("datasets/train.csv")
validation_df = pd.read_csv("datasets/validation.csv")

train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)

# -------------------------------------------------
# Load Tokenizer
# -------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    "models/distilbert_grievance"
)

def tokenize(batch):
    return tokenizer(
        batch["Complaint"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_token_type_ids=False
    )

train_dataset = train_dataset.map(tokenize, batched=True)
validation_dataset = validation_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.rename_column("Label", "labels")
validation_dataset = validation_dataset.rename_column("Label", "labels")

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

validation_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# -------------------------------------------------
# Label Mapping
# -------------------------------------------------

with open("datasets/label_mapping.json") as f:
    label2id = json.load(f)

id2label = {v: k for k, v in label2id.items()}

# -------------------------------------------------
# Load Model
# -------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    "models/distilbert_grievance",
    num_labels=12,
    label2id=label2id,
    id2label=id2label
)

# -------------------------------------------------
# Metrics
# -------------------------------------------------

def compute_metrics(pred):
    predictions = np.argmax(pred.predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        pred.label_ids,
        predictions,
        average="weighted"
    )

    accuracy = accuracy_score(
        pred.label_ids,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# -------------------------------------------------
# Training Arguments
# -------------------------------------------------

training_args = TrainingArguments(
    output_dir="models/trained_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_steps=50,
    save_total_limit=2,
    report_to="none"
)

# -------------------------------------------------
# Trainer
# -------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics
)

# -------------------------------------------------
# Train
# -------------------------------------------------

trainer.train()

# -------------------------------------------------
# Save Model
# -------------------------------------------------

trainer.save_model("models/trained_model")
tokenizer.save_pretrained("models/trained_model")

print("\n✅ Model Training Completed Successfully!")