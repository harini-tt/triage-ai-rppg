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
        "ffmpeg"
    ])
    .pip_install_from_requirements("requirements.txt")
    # Clone original PhysFormer repo and fix numpy compatibility
    .run_commands([
        "cd /root && git clone https://github.com/ZitongYu/PhysFormer.git",
        "cd /root/PhysFormer && sed -i 's/np\\.float/np.float64/g' *.py",
        "cd /root/PhysFormer && sed -i 's/np\\.int/np.int64/g' *.py"
    ])
    # Copy their p10 data structure and model to container  
    .add_local_dir("preprocessed_data", remote_path="/", copy=True)
    .add_local_file("Physformer_VIPL_fold1.pkl", remote_path="/root/PhysFormer/Physformer_VIPL_fold1.pkl", copy=True)
)

def extract_frames_from_video(video_path, output_base_dir="preprocessed_data", subject_id="custom", video_id="v1", source_id="source1"):
    """
    Extract frames from MP4 video and save them in the same format as existing preprocessed data.
    """
    import cv2
    from pathlib import Path

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
def run_physformer_inference():
    """Run PhysFormer inference using the original script"""
    import subprocess
    import os
    
    print("🚀 Running PhysFormer inference...")
    
    # Change to PhysFormer directory  
    os.chdir("/root/PhysFormer")
    
    # Run the original inference script
    result = subprocess.run([
        "python", "inference_OneSample_VIPL_PhysFormer.py"
    ], capture_output=True, text=True)
    
    print("📋 PhysFormer output:")
    print(result.stdout)
    
    if result.stderr:
        print("⚠️ Errors:")
        print(result.stderr)
    
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }

@app.function(
    image=image,
    volumes={"/models": model_volume, "/mnt/outputs": output_volume},
    gpu="T4",
    timeout=600,  # 10 minutes for video processing
    memory=8192,
)
def extract_rppg_from_video(video_data: bytes, filename: str = "video.mp4") -> Dict[str, Any]:
    """
    Extract rPPG signal using original PhysFormer inference script
    
    Args:
        video_data: Video file as bytes
        filename: Original filename (for format detection)
        
    Returns:
        Dictionary containing PhysFormer results
    """
    try:
        # Save video to PhysFormer expected location
        video_path = f"/root/PhysFormer/{filename}"
        with open(video_path, 'wb') as f:
            f.write(video_data)
        
        # Run original PhysFormer inference
        result = run_physformer_inference.local()
        
        # Save results to output volume
        print("\n=== Extracting Generated Files ===")
        os.system("mkdir -p /mnt/outputs")
        os.system("cp -r /root/PhysFormer/*.txt /mnt/outputs/ 2>/dev/null || true")
        os.system("cp -r /root/PhysFormer/*.csv /mnt/outputs/ 2>/dev/null || true") 
        os.system("cp -r /root/PhysFormer/*.pkl /mnt/outputs/ 2>/dev/null || true")
        os.system("cp -r /root/PhysFormer/*.mat /mnt/outputs/ 2>/dev/null || true")
        
        # Save the output logs
        with open("/mnt/outputs/physformer_output.txt", "w") as f:
            f.write("PhysFormer Output:\n")
            f.write("=" * 50 + "\n")
            f.write(result['stdout'])
            if result['stderr']:
                f.write("\n\nErrors:\n")
                f.write("=" * 50 + "\n") 
                f.write(result['stderr'])
        
        output_volume.commit()
        print("✅ Files copied to rppg-outputs volume")
        
        return {
            'status': 'success' if result['returncode'] == 0 else 'error',
            'filename': filename,
            'physformer_output': result['stdout'],
            'errors': result['stderr'],
            'returncode': result['returncode']
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'filename': filename
        }

@app.function(volumes={"/mnt/outputs": output_volume})
def download_results():
    """Download results from Modal volume to local"""
    import os
    import shutil
    
    # List files in output volume
    files = os.listdir("/mnt/outputs")
    print(f"📁 Files in output volume: {files}")
    
    # Copy all files to a downloadable location
    result_files = {}
    for file in files:
        file_path = f"/mnt/outputs/{file}"
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                result_files[file] = f.read()
    
    return result_files

@app.function()
def preprocess_video_to_frames(video_path: str, subject_id: str = "custom", video_id: str = "v1", source_id: str = "source1"):
    """
    Convert MP4 video to preprocessed frames format

    Args:
        video_path: Path to input MP4 video
        subject_id: Subject identifier (default: "custom")
        video_id: Video identifier (default: "v1")
        source_id: Source identifier (default: "source1")
    """
    return extract_frames_from_video(
        video_path=video_path,
        output_base_dir="preprocessed_data",
        subject_id=subject_id,
        video_id=video_id,
        source_id=source_id
    )

@app.local_entrypoint()
def main(action: str = "test", video_path: str = "test_videos/test_vid.mp4", subject_id: str = "custom"):
    """
    Local entrypoint - Test PhysFormer or convert video to frames

    Args:
        action: Either "test" to run PhysFormer inference or "preprocess" to convert video to frames
        video_path: Path to input video file
        subject_id: Subject ID for preprocessing (default: "custom")
    """

    if action == "preprocess":
        print("🔄 Converting video to preprocessed frames...")
        print("=" * 50)

        success = extract_frames_from_video(
            video_path=video_path,
            subject_id=subject_id
        )

        if success:
            print("✅ Video preprocessing completed!")
            print("🚀 Now you can run: python main.py test")
        else:
            print("❌ Video preprocessing failed!")

        return {"action": "preprocess", "success": success}

    elif action == "test":
        print("🎥 Testing PhysFormer with preprocessed data")
        print("=" * 50)
    
    # Load video
    try:
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        file_size_mb = len(video_data) / (1024 * 1024)
        print(f"📁 Loaded video: {file_size_mb:.1f} MB")
        
        # Process with PhysFormer using original script
        print("🧠 Processing with original PhysFormer script...")
        results = extract_rppg_from_video.remote(video_data, "test_vid.mp4")
        
        print("\n📋 Results:")
        if results.get('status') == 'success':
            print("✅ Success!")
            print("📄 PhysFormer Output:")
            print(results.get('physformer_output', 'No output'))
        else:
            print("❌ Failed!")
            print(f"Error: {results.get('error', 'Unknown')}")
            if results.get('errors'):
                print(f"PhysFormer Errors: {results.get('errors')}")
        
        # Download results to local directory
        print("\n📥 Downloading results from Modal volume...")
        downloaded_files = download_results.remote()
        
        # Save files locally
        os.makedirs("results", exist_ok=True)
        for filename, content in downloaded_files.items():
            local_path = f"results/{filename}"
            with open(local_path, 'wb') as f:
                f.write(content)
            print(f"✅ Downloaded: {local_path}")
        
        print(f"\n📁 All results saved to: ./results/")
        print("🎉 Check the 'results' folder for PhysFormer outputs!")
            
        return results
        
    except FileNotFoundError:
        print(f"❌ Video not found: {video_path}")
        return {"error": "Video file not found"}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}