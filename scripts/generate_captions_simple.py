#!/usr/bin/env python3
import os
import json
from pathlib import Path
from PIL import Image
import torch
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from tqdm import tqdm

MODEL_NAME = "Salesforce/blip2-opt-6.7b-coco"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGEFILE_TEMPLATE = "frame_{:06d}.jpg"

def load_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        return img
    except:
        return None

def process_video_captions(video_path, video_name, processor, model, frame_interval=16, batch_size=8):
    frame_files = sorted(video_path.glob("*.jpg"))
    if not frame_files:
        return None

    frame_count = len(frame_files)
    print(f"  {video_name}: {frame_count} frames")

    video_captions = {}
    sampled_frames = list(range(0, frame_count, frame_interval))

    for batch_start in tqdm(range(0, len(sampled_frames), batch_size), desc=f"{video_name}", leave=False):
        batch_end = min(batch_start + batch_size, len(sampled_frames))
        batch_frame_idxs = sampled_frames[batch_start:batch_end]
        batch_frame_paths = [frame_files[idx] for idx in batch_frame_idxs]

        batch_images = [load_image(p) for p in batch_frame_paths]
        batch_images = [img for img in batch_images if img is not None]

        if not batch_images:
            continue

        try:
            batch_inputs = processor(images=batch_images, return_tensors="pt").to(DEVICE)
            generated_ids = model.generate(**batch_inputs)
            batch_generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for idx, text in zip(batch_frame_idxs, batch_generated_text):
                video_captions[str(idx * frame_interval)] = text.strip()
        except Exception as e:
            print(f"    Error: {e}")
            continue

    return video_captions

def main():
    frames_dir = Path("/mnt/data/frames")
    captions_dir = Path("/mnt/data/captions")
    annotation_file = Path("/mnt/data/test.txt")

    captions_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print("Loading BLIP-2 model...")
    processor = Blip2Processor.from_pretrained(MODEL_NAME, use_fast=False)
    model = Blip2ForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model.to(DEVICE)
    print("Model loaded!")

    with open(annotation_file) as f:
        video_list = [line.strip().split()[0] for line in f]

    print(f"\nTotal videos: {len(video_list)}")
    print(f"Output: {captions_dir}")

    for i, video_name in enumerate(video_list):
        caption_file = captions_dir / f"{video_name}.json"

        if caption_file.exists():
            print(f"[{i+1}/{len(video_list)}] {video_name} exists, skip")
            continue

        video_path = frames_dir / video_name
        if not video_path.exists():
            print(f"[{i+1}/{len(video_list)}] {video_name} not found, skip")
            continue

        video_captions = process_video_captions(video_path, video_name, processor, model)

        if video_captions:
            with open(caption_file, "w") as f:
                json.dump(video_captions, f, indent=4)
            print(f"[{i+1}/{len(video_list)}] {video_name}: {len(video_captions)} captions saved")
        else:
            print(f"[{i+1}/{len(video_list)}] {video_name}: failed")

    print("\nDone!")

if __name__ == "__main__":
    main()