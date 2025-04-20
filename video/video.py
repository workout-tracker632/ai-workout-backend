# backend/video_router.py

import os
import uuid
import cv2
import logging
from datetime import datetime
from fastapi import APIRouter, UploadFile, Form, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, JSONResponse
from fastapi.concurrency import run_in_threadpool

import ExerciseAiTrainer
from .gym_live import GymLiveProcessor
from db.init_db import analytics_collection

logger = logging.getLogger("uvicorn")
video_router = APIRouter()

VIDEO_STORAGE_DIR = "video_storage"
os.makedirs(VIDEO_STORAGE_DIR, exist_ok=True)


def cleanup_files(*paths: str):
    """
    Delete any files whose paths are provided.
    """
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.unlink(p)
                logger.info(f"Deleted file: {p}")
            except Exception as e:
                logger.warning(f"Could not delete {p}: {e}")


@video_router.post("/analyze-video")
async def analyze_video(
    file: UploadFile,
    exercise: str = Form(...),
    username: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    # 1. Validate exercise type
    valid_ex = ['Bicept Curl', 'Push Up', 'Squat', 'Shoulder Press', 'general']
    if exercise not in valid_ex:
        raise HTTPException(status_code=400, detail="Invalid exercise type")

    # 2. Save uploaded file to disk
    uid = str(uuid.uuid4())
    in_path = os.path.join(VIDEO_STORAGE_DIR, f"in_{uid}.mp4")
    content = await file.read()
    with open(in_path, "wb") as f:
        f.write(content)
    logger.info(f"Saved upload to {in_path}")

    # 3. GENERAL mode: just count reps, no output video
    if exercise.lower() == "general":
        counts = {}
        total = 0
        for ex in valid_ex[:-1]:
            cap = cv2.VideoCapture(in_path)
            if not cap.isOpened():
                raise HTTPException(status_code=500, detail="Cannot open video")
            exer_obj = ExerciseAiTrainer.Exercise()
            fn = getattr(exer_obj, ex.lower().replace(" ", "_"))
            c = fn(cap, out=None, is_video=False)
            counts[ex] = c
            total += c
            cap.release()

        # persist analytics if any reps counted
        if total > 0:
            day_name = datetime.utcnow().strftime("%A")  # e.g. "Monday", "Tuesday", etc.
            analytics_collection.insert_one({
                "username": username,
                "exercise_type": "general",
                "counts": counts,
                "total_count": total,
                "day": day_name,
                "date": datetime.utcnow()
            })


        # schedule deletion of the input file
        if background_tasks:
            background_tasks.add_task(cleanup_files, in_path)

        return JSONResponse(
            {"exercise_counts": counts, "total_count": total},
            background=background_tasks
        )

    # 4. SPECIFIC mode: process & write out WebM via OpenCV
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Cannot open video")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

    tmp_path   = os.path.join(VIDEO_STORAGE_DIR, f"tmp_{uid}.webm")
    final_path = os.path.join(VIDEO_STORAGE_DIR, f"out_{uid}.webm")

    # VP8 codec for WebM
    fourcc = cv2.VideoWriter_fourcc(*"VP80")
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise HTTPException(status_code=500, detail="Cannot initialize video writer")

    # Run the exercise-specific processing
    exer_obj = ExerciseAiTrainer.Exercise()
    fn = getattr(exer_obj, exercise.lower().replace(" ", "_"))
    count = fn(cap, out, is_video=True)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Rename temp to final
    os.replace(tmp_path, final_path)

    # persist analytics if any reps counted
    if count > 0:
        day_name = datetime.utcnow().strftime("%A")  # e.g. "Monday", "Tuesday", etc.
        analytics_collection.insert_one({
            "username": username,
            "exercise_type": exercise,
            "count": count,
            "day": day_name,
            "date": datetime.utcnow()
        })

    # Read back final video bytes
    with open(final_path, "rb") as f:
        data = f.read()

    # schedule deletion of both input + output after response is sent
    if background_tasks:
        background_tasks.add_task(cleanup_files, in_path, final_path)

    headers = {
        "Content-Disposition": 'attachment; filename="processed.webm"',
        "X-Repetition-Count": str(count),
        "Access-Control-Expose-Headers": "X-Repetition-Count"
    }
    return Response(
        content=data,
        media_type="video/webm",
        headers=headers,
        background=background_tasks
    )


@video_router.websocket("/ws/auto-classify")
async def auto_classify_ws(websocket: WebSocket):
    await websocket.accept()
    processor = GymLiveProcessor()
    gen = processor.run()
    try:
        while True:
            update = await run_in_threadpool(gen.__next__)
            await websocket.send_json(update)
    except WebSocketDisconnect:
        pass
    finally:
        processor.release()
