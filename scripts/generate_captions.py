#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import Blip2ForConditionalGeneration, Blip2Processor

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.video_record import VideoRecord
from src.utils.image_utils import load_images_from_paths

MODEL_NAME = "Salesforce/blip2-opt-6.7b-coco"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGEFILE_TEMPLATE = "frame_{:06d}.jpg"

def process_video_captions(video_path, video_name, processor, model, frame_interval=16, batch_size=8):
    video = VideoRecord(str(video_path), 0, 0)
    video_captions = {}

    print(f"  Processing {video_name} ({video.num_frames} frames)...")

    for batch_start_frame in tqdm(range(0, video.num_frames, batch_size * frame_interval), desc=f"{video_name}", leave=False):
        batch_end_frame = min(batch_start_frame + (batch_size * frame_interval), video.num_frames)
        batch_frame_idxs = range(batch_start_frame, batch_end_frame, frame_interval)
        batch_frame_paths = [
            video_path / IMAGEFILE_TEMPLATE.format(frame_idx)
            for frame_idx in batch_frame_idxs
        ]

        batch_raw_images = load_images_from_paths(batch_frame_paths)
        if not batch_raw_images:
            continue

        batch_inputs = processor(images=batch_raw_images, return_tensors="pt").to(DEVICE)
        generated_ids = model.generate(**batch_inputs)
        batch_generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for frame_idx, generated_text in zip(batch_frame_idxs, batch_generated_text):
            video_captions[str(frame_idx)] = generated_text.strip()

    return video_captions

def main():
    frames_dir = Path("/mnt/data/frames")
    captions_dir = Path("/mnt/data/captions")
    annotation_file = Path("/mnt/data/test.txt")

    captions_dir.mkdir(parents=True, exist_ok=True)

    print("Loading BLIP-2 model...")
    processor = Blip2Processor.from_pretrained(MODEL_NAME, use_fast=False)
    model = Blip2ForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model.to(DEVICE)
    print(f"Model loaded on {DEVICE}")

    with open(annotation_file) as f:
        video_list = [line.strip().split()[0] for line in f]

    print(f"\nTotal videos to caption: {len(video_list)}")
    print(f"Output directory: {captions_dir}")

    for i, video_name in enumerate(tqdm(video_list, desc="Generating captions")):
        caption_file = captions_dir / f"{video_name}.json"

        if caption_file.exists():
            print(f"  [{i+1}/{len(video_list)}] {video_name} already exists, skipping")
            continue

        video_path = frames_dir / video_name
        if not video_path.exists():
            print(f"  [{i+1}/{len(video_list)}] {video_name} not found, skipping")
            continue

        try:
            video_captions = process_video_captions(video_path, video_name, processor, model)

            with open(caption_file, "w") as f:
                json.dump(video_captions, f, indent=4)

            print(f"  [{i+1}/{len(video_list)}] Saved {len(video_captions)} captions for {video_name}")
        except Exception as e:
            print(f"  [{i+1}/{len(video_list)}] Error processing {video_name}: {e}")

    print("\n=== Caption Generation Complete ===")
    print(f"Captions saved to: {captions_dir}")

if __name__ == "__main__":
    main()