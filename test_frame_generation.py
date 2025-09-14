#!/usr/bin/env python3
"""
Test the frame generation logic to ensure it creates all required frames
"""
import os
import sys
sys.path.append('.')
from main import extract_frames_from_video

def test_frame_generation(video_path="test_videos/test_vid.mp4"):
    """Test frame generation and verify all required indices are created"""

    # Extract frames
    print("🎬 Testing frame extraction...")
    try:
        subject_id = extract_frames_from_video(video_path, output_base_dir="./test_frames")
        frame_dir = f"./test_frames/{subject_id}/v1/source1"

        if not os.path.exists(frame_dir):
            print(f"❌ Frame directory not created: {frame_dir}")
            return False

        # Check what frames were created
        frames = [f for f in os.listdir(frame_dir) if f.startswith('image_') and f.endswith('.png')]
        frame_numbers = [int(f[6:11]) for f in frames]  # Extract numbers from image_00061.png
        frame_numbers.sort()

        print(f"📁 Created {len(frames)} frames")
        print(f"📊 Frame range: {min(frame_numbers)} to {max(frame_numbers)}")

        # Check required indices for PhysFormer
        required_indices = set()
        clip_frames = 220

        for clip in range(4):
            start_idx = clip * 160 + 61
            clip_indices = list(range(start_idx, start_idx + clip_frames))
            required_indices.update(clip_indices)
            print(f"🎞️  Clip {clip}: frames {start_idx} to {start_idx + clip_frames - 1}")

        print(f"🔢 Total required indices: {len(required_indices)}")

        # Check if all required frames exist
        missing_frames = []
        for idx in required_indices:
            frame_path = os.path.join(frame_dir, f"image_{idx:05d}.png")
            if not os.path.exists(frame_path):
                missing_frames.append(idx)

        if missing_frames:
            print(f"❌ Missing {len(missing_frames)} required frames:")
            print(f"   First few missing: {missing_frames[:10]}")
            return False
        else:
            print("✅ All required frames are present!")
            return True

    except Exception as e:
        print(f"❌ Error during frame extraction: {e}")
        return False

if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "test_videos/test_vid.mp4"

    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        print("Please provide a valid video path or ensure test_videos/test_vid.mp4 exists")
        sys.exit(1)

    success = test_frame_generation(video_path)
    if success:
        print("\n🎉 Frame generation test passed!")
    else:
        print("\n💥 Frame generation test failed!")
        sys.exit(1)