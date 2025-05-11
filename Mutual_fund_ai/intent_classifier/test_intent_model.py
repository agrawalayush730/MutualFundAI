import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import torch.nn.functional as F

# Load the fine-tuned model and tokenizer
model_path = "C:/Projects/Mutual_fund_ai/intent_classifier/model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Load id2intent mapping
with open("C:/Projects/Mutual_fund_ai/intent_classifier/id2intent.json", "r") as f:
    id2intent = json.load(f)

# Put model into evaluation mode
model.eval()

# ----- Write your query here -----
query = "I want to start sip of 500 on Parg Parikh Flexi cap "  # <-- CHANGE THIS LINE for different tests
# ----------------------------------

# Tokenize the query
inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)

# Get model outputs
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get predicted intent
probs = F.softmax(logits, dim=1)
predicted_label_id = torch.argmax(probs, dim=1).item()

predicted_intent = id2intent[str(predicted_label_id)]
confidence = probs[0][predicted_label_id].item() * 100

# Display result
print(f"User Query: {query}")
print(f"Predicted Intent: {predicted_intent} (Confidence: {confidence:.2f}%)")
