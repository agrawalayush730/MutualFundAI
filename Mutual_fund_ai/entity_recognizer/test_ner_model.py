
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
import torch.nn.functional as F

# Load the fine-tuned model and tokenizer
model_path = "C:/Projects/Mutual_fund_ai/entity_recognizer/models/Final_Ner_model"  # adjust checkpoint if needed

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Load id2entity mapping
with open("C:/Projects/Mutual_fund_ai/entity_recognizer/id2entity.json", "r") as f:
    id2entity = json.load(f)

# Set model to evaluation mode
model.eval()

# ----- Write your query here -----
query = "I want to start an Sip of 5000 rupees in Axis Bank Mid-Cap fund for every month upto a year starting from july"  # <-- CHANGE THIS LINE for different tests
# ----------------------------------

# Tokenize input
inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)

# Get model predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Predict entity for each token
predictions = torch.argmax(F.softmax(logits, dim=2), dim=2)
predicted_labels = predictions[0].tolist()

# Decode tokens properly
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# Display extracted entities
print(f"\nQuery: {query}\n")
print(f"{'Token':20} Predicted Entity")
print(f"{'-'*40}")

for token, label_id in zip(tokens, predicted_labels):
    if token.startswith("##") or token in tokenizer.all_special_tokens:
        continue
    label_name = id2entity.get(str(label_id), "O")
    if label_name != "O":  # Only show entities, skip plain 'O' tokens
        print(f"{token:20} {label_name}")
