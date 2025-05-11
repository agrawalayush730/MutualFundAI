
import json
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Load data
with open('C:/Projects/Mutual_fund_ai/datasets/train_data_intent.json', 'r') as f:
    data = json.load(f)

texts = [d["text"] for d in data]
labels = [d["intent"] for d in data]

# Encode labels
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

# Save label mappings
id2intent = {i: label for i, label in enumerate(le.classes_)}
intent2id = {label: i for i, label in enumerate(le.classes_)}

with open('C:/Projects/Mutual_fund_ai/intent_classifier/intent2id.json', 'w') as f:
    json.dump(intent2id, f)

with open('C:/Projects/Mutual_fund_ai/intent_classifier/id2intent.json', 'w') as f:
    json.dump(id2intent, f)

# Load tokenizer and tokenize data
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

encodings = tokenizer(texts, truncation=True, padding=True)
dataset = Dataset.from_dict({
    'input_ids': encodings['input_ids'],
    'attention_mask': encodings['attention_mask'],
    'labels': encoded_labels
})

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(le.classes_))

# Define metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'f1': f1
    }

# Define training arguments with 10 epochs as in user script
training_args = TrainingArguments(
    output_dir="C:/Projects/Mutual_fund_ai/intent_classifier/model",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="C:/Projects/Mutual_fund_ai/intent_classifier/logs",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-5
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    compute_metrics=compute_metrics
)

# Train
trainer.train()
# Save model and tokenizer
model.save_pretrained("C:/Projects/Mutual_fund_ai/intent_classifier/model")
tokenizer.save_pretrained("C:/Projects/Mutual_fund_ai/intent_classifier/model")
