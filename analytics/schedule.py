# backend/schedule_router.py
from fastapi import APIRouter, HTTPException
from db.init_db import schedule_collection

schedule_router = APIRouter()

@schedule_router.post("/schedule")
async def create_schedule(payload: dict):
    username = payload.get("username")
    sched     = payload.get("schedule")
    if not username or not isinstance(sched, list) or not sched:
        raise HTTPException(status_code=400, detail="username and schedule required")
    # Remove old entries
    schedule_collection.delete_many({"username": username})
    # Insert new docs
    docs = [{
        "username": username,
        "day": entry["day"],
        "exercise_type": entry["exercise"],
        "target": entry["target"]
    } for entry in sched]
    schedule_collection.insert_many(docs)
    return {"status": "ok"}

@schedule_router.get("/schedule/{username}")
async def get_schedule(username: str):
    docs = list(schedule_collection.find({"username": username}, {"_id": 0}))
    # Ensure consistent shape
    return docs
