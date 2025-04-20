import os
import logging
from pymongo import MongoClient

logger = logging.getLogger(__name__)

# Connection settings for MongoDB
# MONGO_URI = "mongodb://localhost:27017"
# MONGO_URI = "mongodb+srv://grad_project_632:RwaujPy1vrnGeHSQ@cluster0.wo9llcy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGO_URI = "mongodb+srv://grad_project_632:RwaujPy1vrnGeHSQ@cluster0.wo9llcy.mongodb.net/?retryWrites=true&w=majority&tls=true&appName=Cluster0"

MONGO_DB_NAME = "workout_tracker"

# Global references
client = None
db = None
users_collection = None
analytics_collection = None
schedule_collection = None
def init_db():
    """
    Initialize MongoDB connection and collections.
    Call this once in main.py or automatically on import.
    """
    global client, db, users_collection, analytics_collection, schedule_collection

    if client is None:
        logger.info("Initializing MongoDB client...")
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB_NAME]

        users_collection = db["users"]
        analytics_collection = db["analytics"]  # For user workout data
        schedule_collection = db["schedule"] # for user schedule
    logger.info("MongoDB initialized successfully.")

# Automatically initialize the DB on import.
init_db()
