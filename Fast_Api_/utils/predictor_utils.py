from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import torch
import torch.nn.functional as F
import json

# ✅ Correct paths for Hugging Face
intent_model_path = Path("C:/Projects/Mutual_fund_ai/intent_classifier/model").resolve().as_posix()
ner_model_path = Path("C:/Projects/Mutual_fund_ai/entity_recognizer/models/Final_Ner_model").resolve().as_posix()

# Load tokenizers and models using POSIX-style absolute paths
intent_tokenizer =  AutoTokenizer.from_pretrained(intent_model_path)
intent_model = AutoModelForSequenceClassification.from_pretrained(intent_model_path)
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_path)

# Load mappings
with open("C:/Projects/Mutual_fund_ai/intent_classifier/id2intent.json") as f:
    id2intent = json.load(f)

with open("C:/Projects/Mutual_fund_ai/entity_recognizer/id2entity.json") as f:
    id2entity = json.load(f)

intent_model.eval()
ner_model.eval()

def get_intent(text: str):
    inputs = intent_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = intent_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_id].item()

    label = id2intent.get(str(pred_id), None)

    if label is None:
        print(f"⚠️ Unknown intent ID predicted: {pred_id}")
        return "unknown_intent", confidence

    return label, confidence


def get_entities(text: str):
    tokenized = ner_tokenizer(text, return_tensors="pt", truncation=True, padding=True, return_offsets_mapping=True)
    offset_mapping = tokenized.pop("offset_mapping")
    with torch.no_grad():
        outputs = ner_model(**tokenized)
        logits = outputs.logits
        predictions = torch.argmax(F.softmax(logits, dim=2), dim=2)[0].tolist()

    tokens = ner_tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])
    offset_mapping = offset_mapping[0].tolist()

    entities = []
    for i, (token, label_id, offsets) in enumerate(zip(tokens, predictions, offset_mapping)):
        label = id2entity.get(str(label_id), "O")
        if label != "O" and not token.startswith("["):
            start, end = offsets
            entity_text = text[start:end]
            entities.append({"text": entity_text, "entity": label})
    return entities
