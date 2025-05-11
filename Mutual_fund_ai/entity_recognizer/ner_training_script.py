
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from transformers import TrainingArguments, DataCollatorForTokenClassification
from sklearn.metrics import precision_recall_fscore_support
from datasets import Dataset
import torch
import json
import os

# Load your training JSON file
with open('C:/Projects/Mutual_fund_ai/datasets/train_data_ner.json', 'r') as f:
    data = json.load(f)

# Extract all unique entity tags
unique_tags = sorted(list({tag for sample in data for tag in sample['ner_tags']}))

# Create mappings
entity2id = {tag: idx for idx, tag in enumerate(unique_tags)}
id2entity = {idx: tag for tag, idx in entity2id.items()}

# Save mappings
with open('C:/Projects/Mutual_fund_ai/entity_recognizer/entity2id.json', 'w') as f:
    json.dump(entity2id, f, indent=4)

with open('C:/Projects/Mutual_fund_ai/entity_recognizer/id2entity.json', 'w') as f:
    json.dump(id2entity, f, indent=4)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

# Function to align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True, padding='max_length', max_length=128)

    labels = []
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)  # special tokens
        elif word_idx != previous_word_idx:
            labels.append(entity2id[examples['ner_tags'][word_idx]])
        else:
            # For subwords, use the same label
            labels.append(entity2id[examples['ner_tags'][word_idx]])
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(data)

# Apply preprocessing
encoded_dataset = dataset.map(tokenize_and_align_labels)

# Load model
model = AutoModelForTokenClassification.from_pretrained(
    "dslim/bert-base-NER",
    num_labels=len(entity2id),
    ignore_mismatched_sizes=True
)

# Training arguments
training_args = TrainingArguments(
    output_dir="C:/Projects/Mutual_fund_ai/entity_recognizer/models",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-5,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="C:/Projects/Mutual_fund_ai/entity_recognizer/logs",
)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Define compute metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = torch.argmax(torch.tensor(predictions), dim=-1)
    true_labels = torch.tensor(labels)

    true_predictions = []
    true_labels_flat = []

    for pred, label in zip(predictions, true_labels):
        for p_i, l_i in zip(pred, label):
            if l_i != -100:
                true_predictions.append(p_i.item())
                true_labels_flat.append(l_i.item())

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels_flat, true_predictions, average='weighted')
    acc = (torch.tensor(true_predictions) == torch.tensor(true_labels_flat)).float().mean().item()

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    eval_dataset=encoded_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
