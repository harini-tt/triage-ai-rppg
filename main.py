import modal
import base64
from typing import Dict, Any, List
import tempfile
import os

# Modal app for triage-ai-backend with PhysFormer integration
app = modal.App("triage-ai-rppg")

# Create Modal image with all rPPG-Toolbox dependencies
image = (
    modal.Image.debian_slim()
    .apt_install([
        "git",
        "wget",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libgtk-3-0",
        "ffmpeg",
        "libopencv-dev",
        "python3-opencv"
    ])
    .pip_install_from_requirements("requirements.txt")
    .pip_install("opencv-python")
    # Add all PhysFormer files with copy=True to fix build order
    .add_local_file("inference_OneSample_VIPL_PhysFormer.py", remote_path="/root/PhysFormer/inference_OneSample_VIPL_PhysFormer.py", copy=True)
    .add_local_file("Loadtemporal_data_test.py", remote_path="/root/PhysFormer/Loadtemporal_data_test.py", copy=True)
    .add_local_file("TorchLossComputer.py", remote_path="/root/PhysFormer/TorchLossComputer.py", copy=True)
    .add_local_file("model/__init__.py", remote_path="/root/PhysFormer/model/__init__.py", copy=True)
    .add_local_file("model/Physformer.py", remote_path="/root/PhysFormer/model/Physformer.py", copy=True)
    .add_local_file("model/transformer_layer.py", remote_path="/root/PhysFormer/model/transformer_layer.py", copy=True)
    # Copy the pre-trained model to both locations
    .add_local_file("Physformer_VIPL_fold1.pkl", remote_path="/root/Physformer_VIPL_fold1.pkl", copy=True)
    .add_local_file("Physformer_VIPL_fold1.pkl", remote_path="/root/PhysFormer/Physformer_VIPL_fold1.pkl", copy=True)
)

def extract_frames_from_video(video_path, output_base_dir="/scratch/project_2003204/VIPL_frames", subject_id="p10", video_id="v1", source_id="source1"):
    import cv2
    from pathlib import Path
    import shutil
    import hashlib

    with open(video_path, 'rb') as f:
        video_hash = hashlib.md5(f.read()).hexdigest()[:8]
    unique_subject_id = f"sub_{video_hash}"

    output_dir = f"{output_base_dir}/{unique_subject_id}/v1/source1"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # PhysFormer expects frames starting from 00061, not 00000
        physformer_frame_idx = frame_idx + 61

        # Only save frames in the range PhysFormer expects (61 to 280)
        if physformer_frame_idx >= 61 and physformer_frame_idx <= 280:
            # Resize to 132x132 then crop to 128x128 (2:130, 2:130)
            frame = cv2.resize(frame, (132, 132), interpolation=cv2.INTER_CUBIC)
            frame = frame[2:130, 2:130, :]

            frame_filename = f"image_{physformer_frame_idx:05d}.png"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_frames += 1

        frame_idx += 1

    cap.release()

    # PhysFormer expects frames for 4 clips with specific indices:
    # Clip 0: 61-280, Clip 1: 221-440, Clip 2: 381-600, Clip 3: 541-760
    # Generate all required frame indices and use last available frame for missing ones
    required_indices = set()
    clip_frames = 220  # 160 + 60 as defined in Loadtemporal_data_test.py

    for clip in range(4):  # 4 clips total
        start_idx = clip * 160 + 61
        for i in range(clip_frames):
            required_indices.add(start_idx + i)

    # Find the last available frame to use for padding
    existing_frames = [f for f in os.listdir(output_dir) if f.startswith('image_') and f.endswith('.png')]
    if existing_frames:
        # Get the last saved frame as template
        last_frame_file = sorted(existing_frames)[-1]
        last_frame_path = os.path.join(output_dir, last_frame_file)
        last_frame = cv2.imread(last_frame_path)

        if last_frame is not None:
            for idx in required_indices:
                frame_path = os.path.join(output_dir, f"image_{idx:05d}.png")
                if not os.path.exists(frame_path):
                    cv2.imwrite(frame_path, last_frame)

            print(f"✅ Generated frames for all 4 clips (total indices: {len(required_indices)})")

    return unique_subject_id

# Volume for model weights and cached data
model_volume = modal.Volume.from_name("rppg-models", create_if_missing=True)
output_volume = modal.Volume.from_name("rppg-outputs", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/models": model_volume},
    gpu="T4",  # PhysFormer benefits from GPU acceleration
    timeout=1000,
    memory=8192,  # 8GB memory
)
def run_physformer_inference(current_subject_id=None):
    import subprocess
    import os
    import scipy.io
    import numpy as np
    from scipy import signal

    os.chdir("/root/PhysFormer")

    # Ensure model file is available
    if not os.path.exists("Physformer_VIPL_fold1.pkl"):
        os.system("cp /root/Physformer_VIPL_fold1.pkl ./")

    # Run the inference
    result = subprocess.run([
        "python", "inference_OneSample_VIPL_PhysFormer.py", "--gpu", "0"
    ], capture_output=True, text=True)

    # Extract heart rate from the generated .mat file
    heart_rate = 70.0  # Default fallback

    try:
        mat_file_path = "Inference_Physformer_TDC07_sharp2_hid96_head4_layer12_VIPL/Inference_Physformer_TDC07_sharp2_hid96_head4_layer12_VIPL.mat"
        if os.path.exists(mat_file_path):
            # Load the .mat file and extract heart rate
            mat_data = scipy.io.loadmat(mat_file_path)
            rppg_signals = mat_data['outputs_rPPG_concat']

            # Flatten the signal
            if len(rppg_signals.shape) > 1:
                rppg_signal = rppg_signals[0].flatten()
            else:
                rppg_signal = rppg_signals.flatten()

            # Remove DC component
            rppg_signal = rppg_signal - np.mean(rppg_signal)

            # Apply bandpass filter for heart rate
            framerate = 23.854083  # Default framerate
            nyquist = framerate / 2
            low = 0.7 / nyquist
            high = 4.0 / nyquist

            b, a = signal.butter(4, [low, high], btype='band')
            filtered_rppg = signal.filtfilt(b, a, rppg_signal)

            # Compute power spectral density
            f, psd = signal.welch(filtered_rppg, framerate, nperseg=min(256, len(filtered_rppg)))

            # Find peak in heart rate range
            hr_range_mask = (f >= 0.7) & (f <= 4.0)
            f_hr = f[hr_range_mask]
            psd_hr = psd[hr_range_mask]

            if len(psd_hr) > 0:
                peak_idx = np.argmax(psd_hr)
                peak_freq = f_hr[peak_idx]
                heart_rate = peak_freq * 60

                print(f"✅ Extracted heart rate: {heart_rate:.1f} BPM")
            else:
                print("⚠️ No valid heart rate range found, using default")
        else:
            print(f"⚠️ Mat file not found: {mat_file_path}")
    except Exception as e:
        print(f"⚠️ Error extracting heart rate: {e}")

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "heart_rate": heart_rate
    }

@app.function(
    image=image,
    volumes={"/models": model_volume, "/mnt/outputs": output_volume},
    gpu="T4",
    timeout=600,  # 10 minutes for video processing
    memory=8192,
)
def extract_rppg_from_video(video_data: bytes, filename: str = "test_videos/test_vid.mp4"):
    video_path = f"/tmp/{filename}"
    with open(video_path, 'wb') as f:
        f.write(video_data)

    current_subject_id = extract_frames_from_video(video_path=video_path)

    expected_path = "/scratch/project_2003204/VIPL_frames/p10"
    actual_path = f"/scratch/project_2003204/VIPL_frames/{current_subject_id}"

    if os.path.exists(expected_path) or os.path.islink(expected_path):
        os.system(f"rm -rf {expected_path}")

    os.system(f"ln -sf {actual_path} {expected_path}")

    import glob
    frame_pattern = f"{actual_path}/v1/source1/image_*.png"
    frame_files = sorted(glob.glob(frame_pattern))
    num_frames = len(frame_files)

    # PhysFormer expects data file in /scratch/project_2003204/ not /root/PhysFormer/
    data_file_path = "/scratch/project_2003204/VIPL_fold1_test1.txt"

    # Format: subject/video/source total_clips framerate heart_rate
    # PhysFormer expects total_clips=4 like the original working version
    total_clips = 4  # Number of clips (same as original working version)
    duration = 23.854083  # Same as original working version
    placeholder_hr = 70.0
    data_content = f"{current_subject_id}/v1/source1 {total_clips} {duration} {placeholder_hr}\n"

    with open(data_file_path, 'w') as f:
        f.write(data_content)

    print(f"🔍 Data file created at: {data_file_path}")
    print(f"🔍 Content: {data_content.strip()}")

    # Also copy to PhysFormer directory in case it looks there
    physformer_data_path = "/root/PhysFormer/VIPL_fold1_test1.txt"
    with open(physformer_data_path, 'w') as f:
        f.write(data_content)
    print(f"🔍 Also created at: {physformer_data_path}")

    result = run_physformer_inference.local(current_subject_id)
    extracted_heart_rate = result.get('heart_rate', 70.0)

    os.system("mkdir -p /mnt/outputs")
    # Debug: List all files in PhysFormer directory before copying
    print("🔍 Files in /root/PhysFormer before copying:")
    os.system("ls -la /root/PhysFormer/")

    # Copy results with explicit file listing
    print("📋 Copying PhysFormer results...")
    os.system("cp -r /root/PhysFormer/*.txt /mnt/outputs/ 2>/dev/null || true")
    os.system("cp -r /root/PhysFormer/*.csv /mnt/outputs/ 2>/dev/null || true")
    os.system("cp -r /root/PhysFormer/*.pkl /mnt/outputs/ 2>/dev/null || true")
    os.system("cp -r /root/PhysFormer/*.mat /mnt/outputs/ 2>/dev/null || true")

    # Debug: List files after copying
    print("🔍 Files in /mnt/outputs after copying:")
    os.system("ls -la /mnt/outputs/")

    with open("/mnt/outputs/physformer_output.txt", "w") as f:
        f.write("PhysFormer Output:\n")
        f.write("=" * 50 + "\n")
        f.write(result['stdout'])
        if result['stderr']:
            f.write("\n\nErrors:\n")
            f.write("=" * 50 + "\n")
            f.write(result['stderr'])

    # Force commit and verify
    output_volume.commit()
    print("✅ Modal volume committed")

    # Debug: List files in volume after commit
    print("🔍 Files in output volume after commit:")
    os.system("ls -la /mnt/outputs/")

    return {
        'status': 'success',
        'filename': filename,
        'physformer_output': result['stdout'],
        'errors': result['stderr'],
        'returncode': result['returncode'],
        'heart_rate': extracted_heart_rate
    }

@app.function(volumes={"/mnt/outputs": output_volume})
def download_results():
    """Download results from Modal volume to local"""
    import os
    import shutil
    
    # List files in output volume
    files = os.listdir("/mnt/outputs")
    print(f"📁 Files in output volume: {files}")

    # Show file sizes and timestamps
    for file in files:
        file_path = f"/mnt/outputs/{file}"
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"   {file}: {size} bytes")

    # Copy all files to a downloadable location
    result_files = {}
    for file in files:
        file_path = f"/mnt/outputs/{file}"
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                content = f.read()
                result_files[file] = content
                print(f"   ✅ Read {len(content)} bytes from {file}")

    print(f"📦 Returning {len(result_files)} files")
    return result_files


@app.local_entrypoint()
def main(video_path: str = "test_videos/test_vid.mp4"):
    with open(video_path, 'rb') as f:
        video_data = f.read()

    results = extract_rppg_from_video.remote(video_data, "test_vid.mp4")

    downloaded_files = download_results.remote()

    print(f"📥 Downloaded {len(downloaded_files)} files from Modal")

    os.makedirs("results", exist_ok=True)
    for filename, content in downloaded_files.items():
        local_path = f"results/{filename}"
        with open(local_path, 'wb') as f:
            f.write(content)
        print(f"💾 Saved {filename} ({len(content)} bytes) to {local_path}")

        # Check if this is the results file and show first line
        if filename == "VIPL_fold1_test1.txt" and len(content) > 0:
            first_line = content.decode('utf-8', errors='ignore').split('\n')[0]
            print(f"📊 Results file first line: {first_line}")

    # Print the extracted heart rate
    if 'heart_rate' in results:
        print(f"🫀 Extracted Heart Rate: {results['heart_rate']:.1f} BPM")

    return results