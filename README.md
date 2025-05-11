
# 🧠 Mutual Fund NLP Bot

A microservice-based Mutual Fund assistant powered by BERT for **Intent Classification** and **Named Entity Recognition (NER)**. Users can input natural language queries like _"Start an SIP of ₹5000 for one year in Parag Parikh Flexi Cap Fund"_ and get actionable insights or directly trigger SIP creation workflows.

---

## 🚀 Features

- 🤖 NLP with BERT: Fine-tuned models for intent classification and entity extraction.
- 🔍 SIP Detection: Automatically identifies SIP-related queries and parses fund name, amount, frequency, start date, etc.
- 🧩 Microservices: SIP creation handled via a separate service (port 8002).
- 🔐 JWT Auth: Secure access to endpoints using access and refresh tokens.
- 🧾 Logging: Robust logging to track user queries, intents, and SIP creation.
- 📊 MySQL: Structured database storage for users and SIPs.

---

## 🗂️ Project Structure

```
MutualFundBot/
│
├── Fast_Api_/               # Main backend server (port 8001)
│   ├── main.py              # Analyzes text, handles routing
│   ├── predictor_utils.py   # Loads and runs intent + entity models
│   └── database/            # SQLAlchemy DB setup
│
├── sip_service/             # Microservice for SIP creation (port 8002)
│   └── sip_main.py          # Handles SIP database commits
│
├── Mutual_fund_intent_entity/
│   ├── intent_training_script.py
│   ├── ner_training_script.py
│   ├── test_intent_model.py
│   ├── test_ner_model.py
│   └── models/              # Trained Hugging Face models
│
├── frontend/
│   └── analyze.html         # User interface (port 8000)
│
└── datasets/                # Labeled JSON data and tag mappings
```

---

## 🔧 Setup Instructions

### 📦 Requirements

- Python 3.10+
- `transformers`, `torch`, `httpx`, `sqlalchemy`, `fastapi`, `uvicorn`, `mysql-connector-python`

Install dependencies:
```bash
pip install -r requirements.txt
```

### 🛠️ Database Setup

Ensure MySQL is running and create databases/tables using:
```sql
-- users table
source /database/auth_user_schema.sql;

-- sip_details table
source /database/sip_schema.sql;
```

---

## 🧪 Running the Project

### 🔐 Backend (port 8001)
```bash
cd Fast_Api_
uvicorn main:app --reload --port 8001
```

### 🔄 SIP Microservice (port 8002)
```bash
cd sip_service
uvicorn sip_main:app --reload --port 8002
```

### 🌐 Frontend (port 8000)
```bash
cd frontend
python3 -m http.server 8000
```

---

## 📡 API Endpoints

### `/analyze-text` `[POST]`
Analyzes user input and:
- Returns `intent` + `entities`, or
- Routes to `/sip-creation` if intent = `create_sip`

### `/sip-creation` `[POST]` (Microservice)
Creates SIP in database if all required entities are present.

---

## 📓 Training & Testing

- Train models:
```bash
python intent_training_script.py
python ner_training_script.py
```

- Test manually:
```bash
python test_intent_model.py
python test_ner_model.py
```

---

## ✅ Future Improvements

- [ ] Add SIP dashboard (frontend view)
- [ ] Entity validation for formats (₹, %, date parsing)
- [ ] Dockerize all 3 services
- [ ] Deploy using reverse proxy (e.g. Nginx)

---

## 🧑‍💻 Author

Built by Ayush Agrawal  
Inspired by real-world financial planning use cases.

