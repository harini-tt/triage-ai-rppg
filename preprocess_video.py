#!/usr/bin/env python3
"""
Minimal PhysFormer Video Preprocessing Script

This script converts a raw MP4 video to the exact format PhysFormer expects:
- Extracts 220 frames (from frame 61 to 280)
- Resizes to 128x128 pixels (with proper cropping)
- Creates proper directory structure
- Generates data file in correct format

Usage: python preprocess_video.py input_video.mp4
"""

import cv2
import os
import sys
from pathlib import Path
import hashlib

def preprocess_video_for_physformer(video_path, output_base="./physformer_data"):
    """
    Convert video to PhysFormer format:
    - 220 frames (61-280)
    - 128x128 RGB frames
    - Proper directory structure
    """
    print(f"🎬 Processing: {video_path}")

    # Generate unique subject ID from video hash
    with open(video_path, 'rb') as f:
        video_hash = hashlib.md5(f.read()).hexdigest()[:8]
    subject_id = f"sub_{video_hash}"

    # Create directory structure
    frames_dir = f"{output_base}/VIPL_frames/{subject_id}/v1/source1"
    Path(frames_dir).mkdir(parents=True, exist_ok=True)

    print(f"📁 Subject ID: {subject_id}")
    print(f"📂 Output: {frames_dir}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📊 Video: {total_frames} frames at {fps:.1f} FPS")

    # Extract frames 61-280 (220 frames total)
    saved_frames = 0
    frame_idx = 0

    while saved_frames < 220 and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        physformer_frame_idx = frame_idx + 61

        if physformer_frame_idx >= 61 and physformer_frame_idx <= 280:
            # PhysFormer preprocessing: resize to 132x132, crop to 128x128
            frame = cv2.resize(frame, (132, 132), interpolation=cv2.INTER_CUBIC)
            frame = frame[2:130, 2:130, :]  # Crop to 128x128

            frame_filename = f"image_{physformer_frame_idx:05d}.png"
            frame_path = os.path.join(frames_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_frames += 1

            if saved_frames % 50 == 0:
                print(f"   Saved {saved_frames}/220 frames")

        frame_idx += 1

    cap.release()

    print(f"✅ Extracted {saved_frames} frames")

    # Create data file
    data_file = f"{output_base}/VIPL_fold1_test1.txt"
    data_content = f"{subject_id}/v1/source1 61 29.824965 70.0\n"

    with open(data_file, 'w') as f:
        f.write(data_content)

    print(f"📝 Created data file: {data_file}")
    print("   Format: subject/video/source start_frame framerate heart_rate")
    print(f"   Content: {data_content.strip()}")

    return subject_id, saved_frames

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess_video.py input_video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    try:
        subject_id, frame_count = preprocess_video_for_physformer(video_path)
        print("\n✅ Preprocessing complete!")
        print(f"🏷️ Subject ID: {subject_id}")
        print(f"📊 Frames: {frame_count}")
        print("🚀 Ready for PhysFormer inference!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
