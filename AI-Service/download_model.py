from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "distilbert-base-uncased"
SAVE_PATH = "models/distilbert_grievance"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Downloading model...")
model = AutoModel.from_pretrained(MODEL_NAME)

print("Saving tokenizer...")
tokenizer.save_pretrained(SAVE_PATH)

print("Saving model...")
model.save_pretrained(SAVE_PATH)

print("\nModel downloaded successfully!")
print(f"Saved to: {SAVE_PATH}")