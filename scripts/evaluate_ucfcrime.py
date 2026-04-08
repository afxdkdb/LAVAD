#!/usr/bin/env python3
import json
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse

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
    "shooting": ["shooting", "shooter", "gunfire", "firearm", "rifle", "pistol", "bullet"],
    "shoplifting": ["shoplift", "shoplifter", "conceal", "security tag"],
    "stealing": ["steal", "theft", "thief", "stolen", "purse", "snatch", "pickpocket"],
    "vandalism": ["vandal", "vandalism", "graffiti", "spray paint", "smash", "destroy property"]
}

ANOMALY_WEIGHTS = {
    "abuse": 2.0, "arrest": 1.0, "arson": 3.0, "assault": 2.5, "burglary": 2.0,
    "explosion": 3.5, "fighting": 2.5, "road_accident": 2.0, "robbery": 3.0,
    "shooting": 4.0, "shoplifting": 1.0, "stealing": 1.5, "vandalism": 1.5
}

def calculate_anomaly_score(caption, frame_idx, prev_caption_lower=None, prev_detected_types=None):
    caption_lower = caption.lower()
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

    if base_score <= 0:
        return 0.0, [], detected_types

    temporal_bonus = 0.0
    if prev_caption_lower and prev_detected_types:
        consecutive_count = 1
        for offset in range(1, 4):
            for anomaly_type in detected_types:
                if any(kw in prev_caption_lower for kw in ANOMALY_KEYWORDS[anomaly_type]):
                    consecutive_count += 1
        if consecutive_count > 1:
            temporal_bonus = min(0.5 * (consecutive_count - 1), 2.0)

    keyword_density_bonus = 0.0
    for anomaly_type, count in type_keyword_counts.items():
        if count >= 3:
            keyword_density_bonus += 0.5
        elif count >= 2:
            keyword_density_bonus += 0.25

    final_score = base_score + temporal_bonus + keyword_density_bonus
    return min(final_score, 10.0), detected_types, detected_types

def load_captions(captions_dir, video_name):
    caption_file = captions_dir / f"{video_name}.json"
    if caption_file.exists():
        with open(caption_file) as f:
            return json.load(f)
    return None

def get_video_labels(annotation_line, num_frames):
    parts = annotation_line.strip().split()
    if len(parts) < 4:
        return [0] * num_frames

    labels = []
    start_idx = int(parts[1])
    end_idx = int(parts[2])
    label = int(parts[3])

    for i in range(num_frames):
        if start_idx <= i <= end_idx:
            labels.append(label)
        else:
            labels.append(0)
    return labels

def evaluate_video(video_name, captions, labels, frame_interval=16):
    if not captions:
        return None

    frame_scores = []
    frame_labels = []
    prev_caption_lower = None
    prev_detected_types = None

    sorted_frames = sorted(captions.items(), key=lambda x: int(x[0]))

    for frame_idx_str, caption in sorted_frames:
        frame_idx = int(frame_idx_str)
        score, _, detected_types = calculate_anomaly_score(caption, frame_idx, prev_caption_lower, prev_detected_types)
        frame_scores.append(score)

        if frame_idx < len(labels):
            frame_labels.append(labels[frame_idx])
        else:
            frame_labels.append(0)

        prev_caption_lower = caption.lower()
        prev_detected_types = detected_types

    frame_scores = np.repeat(frame_scores, frame_interval)
    frame_scores = frame_scores[:len(frame_labels)]

    return {
        "video_name": video_name,
        "frame_scores": frame_scores.tolist(),
        "frame_labels": frame_labels,
        "has_anomaly": any(s > 0 for s in frame_scores),
        "max_score": max(frame_scores) if frame_scores else 0
    }

def calculate_auc(all_scores, all_labels):
    from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve

    flat_scores = []
    flat_labels = []

    for video_scores, video_labels in zip(all_scores, all_labels):
        flat_scores.extend(video_scores)
        flat_labels.extend(video_labels)

    flat_scores = np.array(flat_scores)
    flat_labels = np.array(flat_labels)

    if len(set(flat_labels)) < 2:
        print("Warning: Only one class present in labels")
        return None, None, None

    fpr, tpr, roc_thresholds = roc_curve(flat_labels, flat_scores)
    roc_auc = auc(fpr, tpr)

    precision, recall, pr_thresholds = precision_recall_curve(flat_labels, flat_scores)
    pr_auc = auc(recall, precision)

    return roc_auc, pr_auc, (fpr, tpr, roc_thresholds)

def main():
    parser = argparse.ArgumentParser(description="UCF-Crime Dataset Evaluation")
    parser.add_argument("--frames_dir", type=str, default="/mnt/data/frames",
                        help="Path to extracted frames directory")
    parser.add_argument("--captions_dir", type=str, default="/mnt/data/captions",
                        help="Path to captions directory")
    parser.add_argument("--annotation_file", type=str, default="/mnt/data/test.txt",
                        help="Path to test annotation file")
    parser.add_argument("--output_dir", type=str, default="/mnt/data/eval_results",
                        help="Output directory for results")
    parser.add_argument("--frame_interval", type=int, default=16,
                        help="Frame sampling interval")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    captions_dir = Path(args.captions_dir)
    annotation_file = Path(args.annotation_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== UCF-Crime Dataset Evaluation ===")
    print(f"Frames dir: {frames_dir}")
    print(f"Captions dir: {captions_dir}")
    print(f"Annotation file: {annotation_file}")
    print()

    with open(annotation_file) as f:
        video_list = [line.strip() for line in f]

    print(f"Total videos to evaluate: {len(video_list)}")
    print()

    all_video_results = []
    all_scores = []
    all_labels = []

    for i, line in enumerate(tqdm(video_list, desc="Evaluating videos")):
        parts = line.split()
        video_name = parts[0]
        num_frames = int(parts[2]) + 1

        captions = load_captions(captions_dir, video_name)
        labels = get_video_labels(line, num_frames)

        result = evaluate_video(video_name, captions, labels, args.frame_interval)

        if result:
            all_video_results.append(result)
            all_scores.append(result["frame_scores"])
            all_labels.append(result["frame_labels"])

            result_file = output_dir / f"{video_name}_result.json"
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)

    print("\n=== Calculating AUC ===")
    roc_auc, pr_auc, roc_data = calculate_auc(all_scores, all_labels)

    if roc_auc is not None:
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")

        results_summary = {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "total_videos": len(all_video_results),
            "anomaly_videos": sum(1 for r in all_video_results if r["has_anomaly"])
        }

        with open(output_dir / "auc_results.json", "w") as f:
            json.dump(results_summary, f, indent=2)

        fpr, tpr, thresholds = roc_data
        roc_curve_data = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist()
        }
        with open(output_dir / "roc_curve.json", "w") as f:
            json.dump(roc_curve_data, f, indent=2)

        print(f"\nResults saved to: {output_dir}")
    else:
        print("Failed to calculate AUC")

if __name__ == "__main__":
    main()