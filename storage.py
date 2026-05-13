import json
import uuid
import logging
from datetime import datetime
import os
from google.cloud import storage
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BUCKET_NAME = os.environ.get("BUCKET_NAME", "resume-api-storage")
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

def save_to_gcs(resume_text: str, extracted: dict, folder="extractions") -> str:
    record_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    
    payload = {
        "id": record_id,
        "timestamp": timestamp,
        "resume_input": resume_text,
        "extracted": extracted
    }

    # Use the folder passed from app.py
    key = f"{folder}/{timestamp[:10]}/{record_id}.json"
    blob = bucket.blob(key)

    try:
        blob.upload_from_string(
            data=json.dumps(payload, ensure_ascii=False),
            content_type="application/json"
        )
        logger.info(f"[GCS] Saved to {folder} → gs://{BUCKET_NAME}/{key}")
        return record_id
    except Exception as e:
        logger.error(f"[GCS] Upload failed: {e}")
        return None