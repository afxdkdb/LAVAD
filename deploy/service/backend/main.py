import os
import sys
import uuid
import json
import shutil
import asyncio
from pathlib import Path
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import aiofiles

sys.path.append(str(Path(__file__).parent.parent.parent))

TEMP_DIR = Path("/tmp/lavad_uploads")
OUTPUT_DIR = Path("/tmp/lavad_outputs")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

QWEN_MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "/mnt/data/libs/Qwen-7B-Chat")
BLIP2_MODEL_PATH = "Salesforce/blip2-opt-6.7b-coco"

ANOMALY_KEYWORDS = {
    "abuse": ["abuse", "abused", "abusing", "victim", "humiliate", "torment"],
    "arrest": ["arrest", "arrested", "handcuff", "handcuffed", "police", "cop", "detain", "custody"],
    "arson": ["arson", "arsonist", "fire", "burning building", "engulfed in flame", "ignite"],
    "assault": ["assault", "assaulted", "attack", "attacked", "attacking", "beat", "beating", "brawl"],
    "burglary": ["burglar", "break-in", "intruder", "trespass", "forced entry", "burgle"],
    "explosion": ["explosion", "explode", "exploding", "blast", "bomb", "detonate"],
    "fighting": ["fight", "fighting", "punch", "punching", "kick", "kicking", "brawl", "scuffle", "combat", "struggle"],
    "road_accident": ["crash", "collision", "collided", "wreckage", "overturned", "flipped", "rolled", "smash", "t-bone", "rear-end"],
    "robbery": ["robbery", "robber", "armed robbery", "stick-up", "hold-up", "threaten", "weapon"],
    "shooting": ["shooting", "shooter", "gunfire", "gunfire", "firearm", "rifle", "pistol", "bullet"],
    "shoplifting": ["shoplift", "shoplifter", "conceal", "security tag"],
    "stealing": ["steal", "theft", "thief", "stolen", "purse", "snatch", "pickpocket"],
    "vandalism": ["vandal", "vandalism", "graffiti", "spray paint", "smash", "destroy property"]
}

ANOMALY_WEIGHTS = {
    "abuse": 2.0,
    "arrest": 1.0,
    "arson": 3.0,
    "assault": 2.5,
    "burglary": 2.0,
    "explosion": 3.5,
    "fighting": 2.5,
    "road_accident": 2.0,
    "robbery": 3.0,
    "shooting": 4.0,
    "shoplifting": 1.0,
    "stealing": 1.5,
    "vandalism": 1.5
}

task_progress = {}
task_lock = asyncio.Lock()
executor = ThreadPoolExecutor(max_workers=2)

captioner = None


def update_progress(task_id: str, stage: str, percent: int, message: str = ""):
    task_progress[task_id] = {
        "stage": stage,
        "percent": percent,
        "message": message
    }


def detect_anomalies_from_captions(captions: Dict[str, str], frame_interval: int = 16) -> Dict:
    anomaly_scores = {}
    anomaly_frames = []
    anomaly_details = []

    caption_items = sorted(captions.items(), key=lambda x: int(x[0]))
    frame_indices = [int(frame_idx_str) for frame_idx_str, _ in caption_items]

    for idx, (frame_idx_str, caption) in enumerate(caption_items):
        frame_idx = int(frame_idx_str)
        caption_lower = caption.lower()
        caption_length = len(caption.split())

        base_score = 0.0
        detected_types = []
        type_keyword_counts = {}

        for anomaly_type, keywords in ANOMALY_KEYWORDS.items():
            keyword_count = 0
            for keyword in keywords:
                if keyword in caption_lower:
                    keyword_count += 1

            if keyword_count > 0:
                detected_types.append(anomaly_type)
                type_keyword_counts[anomaly_type] = keyword_count
                base_score += ANOMALY_WEIGHTS.get(anomaly_type, 1.0) * min(keyword_count, 3)

        if base_score > 0:
            temporal_bonus = 0.0
            context_frames = 3
            consecutive_count = 1

            for offset in range(1, context_frames + 1):
                prev_idx = idx - offset
                if prev_idx >= 0:
                    prev_frame_idx_str = str(frame_indices[prev_idx])
                    if prev_frame_idx_str in captions:
                        prev_caption_lower = captions[prev_frame_idx_str].lower()
                        for anomaly_type in detected_types:
                            if any(kw in prev_caption_lower for kw in ANOMALY_KEYWORDS[anomaly_type]):
                                consecutive_count += 1

            if consecutive_count > 1:
                temporal_bonus = 0.5 * (consecutive_count - 1)
                temporal_bonus = min(temporal_bonus, 2.0)

            keyword_density_bonus = 0.0
            for anomaly_type, count in type_keyword_counts.items():
                if count >= 3:
                    keyword_density_bonus += 0.5
                elif count >= 2:
                    keyword_density_bonus += 0.25

            final_score = base_score + temporal_bonus + keyword_density_bonus
            final_score = min(final_score, 10.0)

            anomaly_scores[frame_idx_str] = final_score
            anomaly_frames.append(frame_idx)
            anomaly_details.append({
                "frame": frame_idx,
                "time": round(frame_idx * frame_interval / 25, 2),
                "caption": caption,
                "anomaly_types": detected_types,
                "score": round(final_score, 2)
            })

    threshold = 1.0
    is_anomaly = any(score >= threshold for score in anomaly_scores.values())

    has_anomaly = is_anomaly
    anomalous_frames = sorted(list(set(anomaly_frames)))

    return {
        "has_anomaly": has_anomaly,
        "anomaly_count": len(anomalous_frames),
        "anomalous_frames": anomalous_frames,
        "anomaly_details": anomaly_details,
        "all_scores": anomaly_scores,
        "summary": {
            "total_frames": len(captions),
            "anomalous_frames_count": len(anomalous_frames),
            "anomaly_ratio": round(len(anomalous_frames) / len(captions) * 100, 2) if captions else 0
        }
    }


def load_captioner():
    global captioner
    if captioner is None:
        print("Loading BLIP-2 model to GPU (this may take a few minutes)...")
        from src.models.image_captioner import ImageCaptioner

        captioner = ImageCaptioner(
            batch_size=8,
            frame_interval=16,
            imagefile_template="{:06d}.jpg",
            pretrained_model_name=BLIP2_MODEL_PATH,
            dtype_str="float16",
            output_dir=str(OUTPUT_DIR / "captions")
        )
        print("BLIP-2 model loaded successfully!")
    return captioner


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 50)
    print("Starting LAVAD API...")
    print(f"Device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("Pre-loading BLIP-2 model...")
    load_captioner()
    print("=" * 50)
    yield
    print("Shutting down LAVAD API...")
    executor.shutdown(wait=True)


app = FastAPI(
    title="LAVAD API",
    description="Training-free Video Anomaly Detection API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "LAVAD Video Anomaly Detection API", "status": "running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": captioner is not None
    }


@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    if not video.filename.endswith((".mp4", ".avi")):
        raise HTTPException(status_code=400, detail="Only MP4 or AVI videos are supported")

    if captioner is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait.")

    task_id = str(uuid.uuid4())
    video_path = TEMP_DIR / f"{task_id}_{video.filename}"
    output_dir = OUTPUT_DIR / task_id
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        async with aiofiles.open(video_path, 'wb') as f:
            content = await video.read()
            await f.write(content)

        update_progress(task_id, "uploading", 5, "视频上传完成")

        loop = asyncio.get_event_loop()
        loop.run_in_executor(executor, run_pipeline, video_path, output_dir, task_id)

        return JSONResponse(content={
            "task_id": task_id,
            "status": "processing",
            "message": "任务已提交，请使用 /progress/{task_id} 查询进度"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_pipeline(video_path: Path, output_dir: Path, task_id: str):
    try:
        update_progress(task_id, "extracting_frames", 10, "正在提取视频帧...")

        from src.preprocessing.extract_frames import extract_frames
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        annotations_file = output_dir / "annotations.txt"

        video_name, num_frames = extract_frames(str(video_path), str(frames_dir))

        with open(annotations_file, "w") as f:
            f.write(f"{video_name} 0 {num_frames - 1} 0\n")

        update_progress(task_id, "extracting_frames", 25, "帧提取完成")

        frames = sorted(list(frames_dir.glob(f"{video_name}/*.jpg")))
        if not frames:
            raise Exception("No frames extracted")

        update_progress(task_id, "generating_caption", 30, "正在生成Caption (BLIP-2)...")

        from src.data.video_record import VideoRecord

        global captioner
        captioner_to_use = load_captioner()

        video_record = VideoRecord([video_name, "0", str(num_frames - 1), "0"], str(frames_dir))
        captioner_to_use.process_video(video_record)

        update_progress(task_id, "analyzing_anomaly", 65, "正在分析异常...")

        caption_file = OUTPUT_DIR / "captions" / f"{video_name}.json"
        if caption_file.exists():
            with open(caption_file) as f:
                captions = json.load(f)

            anomaly_result = detect_anomalies_from_captions(captions)

            with open(output_dir / "anomaly_result.json", "w") as f:
                json.dump(anomaly_result, f, indent=2, ensure_ascii=False)

            update_progress(task_id, "analyzing_anomaly", 80, "异常分析完成")
        else:
            raise FileNotFoundError(f"Caption file not found: {caption_file}")

        update_progress(task_id, "completed", 100, "处理完成")
        task_progress[task_id]["status"] = "completed"
        task_progress[task_id]["message"] = "处理完成"

    except Exception as e:
        import traceback
        traceback.print_exc()
        update_progress(task_id, "error", 0, f"处理失败: {str(e)}")
        task_progress[task_id]["status"] = "error"
        task_progress[task_id]["error"] = str(e)


@app.get("/progress/{task_id}")
async def get_progress(task_id: str):
    if task_id not in task_progress:
        return JSONResponse(content={
            "task_id": task_id,
            "status": "not_found",
            "message": "任务不存在或已过期"
        })

    return JSONResponse(content={
        "task_id": task_id,
        **task_progress[task_id]
    })


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    output_dir = OUTPUT_DIR / task_id
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="Task not found")

    anomaly_file = output_dir / "anomaly_result.json"
    if anomaly_file.exists():
        with open(anomaly_file) as f:
            anomaly_result = json.load(f)
        return JSONResponse(content={
            "task_id": task_id,
            "status": "completed",
            "result": anomaly_result
        })

    progress = task_progress.get(task_id, {})
    if progress.get("status") == "completed":
        return JSONResponse(content={
            "task_id": task_id,
            "status": "completed",
            "message": "处理完成，请使用 /result/{task_id} 获取结果"
        })

    return JSONResponse(content={
        "task_id": task_id,
        "status": progress.get("status", "processing"),
        "stage": progress.get("stage", "unknown"),
        "percent": progress.get("percent", 0),
        "message": progress.get("message", "处理中...")
    })


@app.get("/visualization/{task_id}")
async def get_visualization(task_id: str):
    output_dir = OUTPUT_DIR / task_id
    video_file = output_dir / "visualization.mp4"

    if not video_file.exists():
        raise HTTPException(status_code=404, detail="Visualization not available")

    return FileResponse(
        video_file,
        media_type="video/mp4",
        filename=f"{task_id}_visualization.mp4"
    )


@app.get("/frame/{task_id}/{frame_number}")
async def get_frame(task_id: str, frame_number: int):
    output_dir = OUTPUT_DIR / task_id
    frames_dir = output_dir / "frames"

    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail="Frames not found")

    video_name = None
    for d in frames_dir.iterdir():
        if d.is_dir():
            video_name = d.name
            break

    if not video_name:
        raise HTTPException(status_code=404, detail="Video folder not found")

    frame_path = frames_dir / video_name / f"{frame_number:06d}.jpg"

    if not frame_path.exists():
        raise HTTPException(status_code=404, detail=f"Frame {frame_number} not found")

    return FileResponse(
        frame_path,
        media_type="image/jpeg",
        filename=f"frame_{frame_number}.jpg"
    )


@app.get("/frames/{task_id}")
async def get_frames_list(task_id: str):
    output_dir = OUTPUT_DIR / task_id
    frames_dir = output_dir / "frames"

    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail="Frames not found")

    video_name = None
    for d in frames_dir.iterdir():
        if d.is_dir():
            video_name = d.name
            break

    if not video_name:
        raise HTTPException(status_code=404, detail="Video folder not found")

    frames_folder = frames_dir / video_name
    frame_files = sorted(list(frames_folder.glob("*.jpg")))

    return JSONResponse(content={
        "task_id": task_id,
        "video_name": video_name,
        "frames_base_url": f"/frame/{task_id}",
        "total_frames": len(frame_files),
        "frames": [f.stem for f in frame_files]
    })


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        reload=False
    )