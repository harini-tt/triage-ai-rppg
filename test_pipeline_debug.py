#!/usr/bin/env python3
"""
Debug tests for PhysFormer pipeline to identify why it returns default 70 BPM.
"""
import os
import tempfile
import shutil
from pathlib import Path
import cv2
import subprocess
import sys

def test_1_check_physformer_inference_script():
    """Test 1: Check if the PhysFormer inference script exists"""
    print("=" * 60)
    print("TEST 1: Checking PhysFormer inference script availability")
    print("=" * 60)

    script_name = "inference_OneSample_VIPL_PhysFormer.py"

    # Check if script exists in current directory
    if os.path.exists(script_name):
        print(f"✅ Found {script_name} in current directory")
        with open(script_name, 'r') as f:
            content = f.read()
            print(f"   Script length: {len(content)} characters")
            print(f"   First 200 chars: {content[:200]}")
    else:
        print(f"❌ {script_name} NOT FOUND in current directory")
        print("   This is why the pipeline returns 70 BPM placeholder!")

    # Check if we can find it anywhere
    try:
        result = subprocess.run(['find', '.', '-name', script_name],
                               capture_output=True, text=True)
        if result.stdout.strip():
            print(f"   Found at: {result.stdout.strip()}")
        else:
            print(f"   Script not found anywhere in project")
    except:
        pass

    return os.path.exists(script_name)

def test_2_check_model_file():
    """Test 2: Check if the model file exists and is valid"""
    print("\n" + "=" * 60)
    print("TEST 2: Checking PhysFormer model file")
    print("=" * 60)

    model_file = "Physformer_VIPL_fold1.pkl"

    if os.path.exists(model_file):
        size = os.path.getsize(model_file)
        print(f"✅ Model file found: {model_file}")
        print(f"   Size: {size:,} bytes ({size / (1024*1024):.1f} MB)")

        # Try to load it
        try:
            import pickle
            with open(model_file, 'rb') as f:
                # Just peek at the first few bytes to validate it's a pickle
                header = f.read(10)
                print(f"   Pickle header: {header[:5]}")
        except Exception as e:
            print(f"   ⚠️  Error reading model: {e}")
    else:
        print(f"❌ Model file NOT FOUND: {model_file}")

    return os.path.exists(model_file)

def test_3_simulate_frame_extraction(test_video_path=None):
    """Test 3: Test the frame extraction function"""
    print("\n" + "=" * 60)
    print("TEST 3: Testing frame extraction")
    print("=" * 60)

    # Import the function from main.py
    sys.path.append('.')
    from main import extract_frames_from_video

    if test_video_path and os.path.exists(test_video_path):
        print(f"Using test video: {test_video_path}")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_base = temp_dir

            try:
                subject_id = extract_frames_from_video(
                    video_path=test_video_path,
                    output_base_dir=output_base
                )

                frame_dir = f"{output_base}/{subject_id}/v1/source1"
                if os.path.exists(frame_dir):
                    frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
                    print(f"✅ Extracted {len(frames)} frames to {frame_dir}")

                    if frames:
                        # Check first and last frame
                        first_frame = frames[0]
                        last_frame = frames[-1]
                        print(f"   Frame range: {first_frame} to {last_frame}")

                        # Check frame dimensions
                        first_frame_path = os.path.join(frame_dir, first_frame)
                        img = cv2.imread(first_frame_path)
                        if img is not None:
                            h, w = img.shape[:2]
                            print(f"   Frame dimensions: {w}x{h}")
                            print(f"   Expected: 128x128")
                            if w == 128 and h == 128:
                                print("   ✅ Correct frame size")
                            else:
                                print("   ❌ Wrong frame size!")

                        return True, subject_id, len(frames)
                else:
                    print(f"❌ No frames extracted to {frame_dir}")

            except Exception as e:
                print(f"❌ Frame extraction failed: {e}")
                return False, None, 0
    else:
        print("⚠️  No test video provided - skipping actual extraction")
        print("   Provide video path with: python test_pipeline_debug.py <video_path>")
        return None, None, 0

def test_4_check_data_file_format():
    """Test 4: Check the data file format matches expectations"""
    print("\n" + "=" * 60)
    print("TEST 4: Checking data file format")
    print("=" * 60)

    # Check current results
    results_file = "results/VIPL_fold1_test1.txt"
    physformer_file = "physformer_data/VIPL_fold1_test1.txt"

    files_to_check = [results_file, physformer_file]

    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"📁 Checking: {file_path}")
            with open(file_path, 'r') as f:
                content = f.read().strip()
                print(f"   Content: {content}")

                parts = content.split()
                if len(parts) >= 4:
                    path, clips, duration, hr = parts[0], parts[1], parts[2], parts[3]
                    print(f"   Path: {path}")
                    print(f"   Clips: {clips}")
                    print(f"   Duration: {duration}")
                    print(f"   Heart Rate: {hr}")

                    if hr == "70.0":
                        print("   ❌ This is the placeholder value!")
                    else:
                        print("   ✅ Heart rate is not placeholder")
        else:
            print(f"❌ File not found: {file_path}")

def test_5_check_physformer_repo_structure():
    """Test 5: Check what PhysFormer files we actually have"""
    print("\n" + "=" * 60)
    print("TEST 5: Checking PhysFormer repository structure")
    print("=" * 60)

    # Look for any PhysFormer-related files
    physformer_files = []

    for root, dirs, files in os.walk('.'):
        for file in files:
            if 'physformer' in file.lower() or 'PhysFormer' in file:
                physformer_files.append(os.path.join(root, file))

    if physformer_files:
        print("📁 Found PhysFormer-related files:")
        for file in physformer_files:
            size = os.path.getsize(file)
            print(f"   {file} ({size:,} bytes)")
    else:
        print("❌ No PhysFormer files found locally")
        print("   The Modal container clones the repo, but it's not available locally")

def main():
    print("🔍 PhysFormer Pipeline Debug Tests")
    print("This will help identify why your pipeline returns 70 BPM")
    print()

    # Get test video path from command line if provided
    test_video = sys.argv[1] if len(sys.argv) > 1 else None

    # Run all tests
    script_exists = test_1_check_physformer_inference_script()
    model_exists = test_2_check_model_file()
    frame_test = test_3_simulate_frame_extraction(test_video)
    test_4_check_data_file_format()
    test_5_check_physformer_repo_structure()

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)

    if not script_exists:
        print("🚨 PRIMARY ISSUE: inference_OneSample_VIPL_PhysFormer.py is missing!")
        print("   Your Modal pipeline tries to run this script, but it doesn't exist locally.")
        print("   The script is only available inside the Modal container after cloning the repo.")
        print("   Solution: Either run the inference locally or copy the script from the repo.")

    if not model_exists:
        print("🚨 SECONDARY ISSUE: Model file is missing!")
        print("   The PhysFormer model needs to be present for inference.")

    print("\n🔧 RECOMMENDED FIXES:")
    print("1. Download the PhysFormer inference script from the GitHub repo")
    print("2. Ensure the model file path is correct in the inference script")
    print("3. Verify the data file format matches what PhysFormer expects")
    print("4. The 70.0 BPM is a hardcoded placeholder - the real inference isn't running")

if __name__ == "__main__":
    main()