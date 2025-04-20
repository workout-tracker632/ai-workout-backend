# backend/analytics_router.py
from fastapi import APIRouter, HTTPException
from db.init_db import analytics_collection, schedule_collection

analytics_router = APIRouter()

@analytics_router.get("/{username}")
async def get_analytics(username: str):
    # Fetch all analytics records for user
    raw_counts = list(analytics_collection.find({"username": username}, {"_id": 0}))
    if not raw_counts:
        raise HTTPException(404, "No analytics found")
    
    # Fetch full schedule
    sched = list(schedule_collection.find({"username": username}, {"_id": 0}))
    
    return {"counts": raw_counts, "schedule": sched}
