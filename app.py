import logging
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from auth.auth import auth_router
from analytics.analytics import analytics_router
from analytics.schedule import schedule_router
from video.video import video_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Workout AI Tracker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app.include_router(auth_router, prefix="", tags=["Authentication"])
app.include_router(analytics_router, prefix="/analytics", tags=["Analytics"])
app.include_router(video_router, prefix="", tags=["Video"])
app.include_router(schedule_router, prefix="", tags=["Video"])


@app.get("/")
def root():
    return {"message": "Welcome to the Workout AI Tracker API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
