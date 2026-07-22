from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=16
)

model.save_pretrained("./models/distilbert_grievance")
tokenizer.save_pretrained("./models/distilbert_grievance")

print("Model saved successfully!")