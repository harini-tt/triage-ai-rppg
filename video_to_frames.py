import cv2
import os
import argparse
import sys
from pathlib import Path

def extract_frames_from_video(video_path, output_base_dir="preprocessed_data", subject_id="custom", video_id="v1", source_id="source1"):
    """
    Extract frames from MP4 video and save them in the same format as existing preprocessed data.

    Args:
        video_path (str): Path to the input MP4 video
        output_base_dir (str): Base directory for output (default: "preprocessed_data")
        subject_id (str): Subject identifier (default: "custom")
        video_id (str): Video identifier (default: "v1")
        source_id (str): Source identifier (default: "source1")
    """

    # Validate input video exists
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        return False

    # Create output directory structure matching existing format
    output_dir = os.path.join(
        output_base_dir,
        "scratch",
        "project_2003204",
        "VIPL_frames",
        subject_id,
        video_id,
        source_id
    )

    # Create directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {video_path}")
        return False

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(f"🎥 Video Info:")
    print(f"   - FPS: {fps:.2f}")
    print(f"   - Total frames: {frame_count}")
    print(f"   - Duration: {duration:.2f} seconds")
    print(f"📁 Output directory: {output_dir}")

    # Extract frames
    frame_idx = 0
    saved_frames = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Save frame with same naming convention: image_XXXXX.png (5-digit zero-padded)
        frame_filename = f"image_{frame_idx:05d}.png"
        frame_path = os.path.join(output_dir, frame_filename)

        # Save frame as PNG
        cv2.imwrite(frame_path, frame)
        saved_frames += 1

        # Print progress every 50 frames
        if saved_frames % 50 == 0:
            print(f"   Extracted {saved_frames} frames...")

        frame_idx += 1

    # Cleanup
    cap.release()

    print(f"✅ Successfully extracted {saved_frames} frames to {output_dir}")
    print(f"   Frame naming: image_00000.png to image_{saved_frames-1:05d}.png")

    return True

def main():
    """Command line interface for video to frames extraction"""
    parser = argparse.ArgumentParser(description="Extract frames from MP4 video to match preprocessed data format")
    parser.add_argument("video_path", help="Path to input MP4 video file")
    parser.add_argument("--output_dir", default="preprocessed_data", help="Base output directory (default: preprocessed_data)")
    parser.add_argument("--subject_id", default="custom", help="Subject identifier (default: custom)")
    parser.add_argument("--video_id", default="v1", help="Video identifier (default: v1)")
    parser.add_argument("--source_id", default="source1", help="Source identifier (default: source1)")

    args = parser.parse_args()

    print("🔄 Video to Frames Extractor")
    print("=" * 40)

    success = extract_frames_from_video(
        video_path=args.video_path,
        output_base_dir=args.output_dir,
        subject_id=args.subject_id,
        video_id=args.video_id,
        source_id=args.source_id
    )

    if success:
        print("🎉 Frame extraction completed successfully!")
        return 0
    else:
        print("❌ Frame extraction failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())