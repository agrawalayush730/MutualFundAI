# Standard Library
import traceback
import yaml
import logging.config

# FastAPI & DB
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware

# Local Modules
from database.db_init import get_db, SessionLocal
from database.sip_service.models import SIPRequest
from entities import SIPCreateRequest

# Logging setup
with open("logging_config.yaml", "r") as f:
    config = yaml.safe_load(f)
    logging.config.dictConfig(config)

logger = logging.getLogger("app_logger")
logger.info("Logging configured for SIP microservice.")

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to backend IP if needed
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "sip-service running"}

@app.post("/sip-creation")
def sip_creation(payload: SIPCreateRequest, db: Session = Depends(get_db)):
    try:
        entities = payload.entities
        user_id = payload.user_id

        entity_map = {ent["entity"]: ent["text"] for ent in entities}

        amount = entity_map.get("B-AMOUNT")
        fund_name = entity_map.get("B-ORG")
        start_date = entity_map.get("B-START_DATE")
        duration = entity_map.get("B-DURATION")
        frequency = entity_map.get("B-FREQUENCY")

        missing_fields = []
        for field, name in zip(
            [amount, fund_name, start_date, duration, frequency],
            ["amount", "fund_name", "start_date", "duration", "frequency"]
        ):
            if not field:
                missing_fields.append(name)

        if missing_fields:
            logger.warning(f"Missing fields: {missing_fields}")
            return {"error": f"Missing required fields: {', '.join(missing_fields)}"}

        sip_request = SIPRequest(
            user_id=user_id,
            amount=amount,
            fund_name=fund_name,
            duration=duration,
            start_date=start_date,
            frequency=frequency
        )
        db.add(sip_request)
        db.commit()
        db.refresh(sip_request)

        logger.info(f"SIP created successfully for user_id={user_id}")
        return {
            "status": "success",
            "message": f"SIP created successfully for user_id={user_id}"
        }

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"SIP creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"SIP creation failed: {str(e)}\n{tb}")
