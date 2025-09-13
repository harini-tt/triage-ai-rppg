import modal
import base64
import numpy as np
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
    # Clone original PhysFormer repo
    .run_commands([
        "cd /root && git clone https://github.com/ZitongYu/PhysFormer.git"
    ])
    # Copy preprocessed frames and model to container
    .add_local_dir("preprocessed_data", remote_path="/", copy=True)
    .add_local_file("Physformer_VIPL_fold1.pkl", remote_path="/root/PhysFormer/Physformer_VIPL_fold1.pkl", copy=True)
)

# Volume for model weights and cached data
model_volume = modal.Volume.from_name("rppg-models", create_if_missing=True)

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
    volumes={"/models": model_volume},
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

@app.local_entrypoint()
def main():
    """Local entrypoint - Test PhysFormer with test_vid.mp4"""
    video_path = "test_videos/test_vid.mp4"
    
    print("🎥 Testing PhysFormer with test_vid.mp4")
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
            
        return results
        
    except FileNotFoundError:
        print(f"❌ Video not found: {video_path}")
        return {"error": "Video file not found"}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}