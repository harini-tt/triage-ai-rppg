# Heart Rate Analysis from Video - rPPG

A simple, lightweight remote photoplethysmography (rPPG) pipeline for extracting heart rate from video using facial analysis.

## Features

- **CPU-based processing** - No GPU required
- **MediaPipe face detection** - Robust face and ROI detection
- **CHROM method** - Chrominance-based rPPG signal extraction
- **Command-line interface** - Simple usage from terminal
- **Fast processing** - Optimized for real-time analysis

## Installation

```bash
# Create virtual environment with Python 3.9
pyenv install 3.9.19
python3.9 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install opencv-python mediapipe numpy scipy
```

## Usage

```bash
# Activate environment
source .venv/bin/activate

# Extract heart rate from video
python simple_rppg.py test_videos/test_vid.mp4
```

## Example Output

```
Processing video: test_videos/test_vid.mp4
Video FPS: 30.01, Total frames: 257
Processed 30/257 frames
Processed 60/257 frames
...
Extracting color signals from 257 frames...
Applying CHROM method...
Estimating heart rate...
Estimated Heart Rate: 77.06 BPM

Processing complete!
Final heart rate estimate: 77.06 BPM
```

## How It Works

1. **Face Detection**: Uses MediaPipe to detect faces in video frames
2. **ROI Extraction**: Extracts forehead region (upper 1/3 of detected face)
3. **Signal Processing**: Computes mean RGB values for each frame
4. **CHROM Method**: Applies chrominance-based transformation to isolate pulse signal
5. **Heart Rate Estimation**: Uses FFT to find dominant frequency in valid HR range (42-240 BPM)

## Requirements

- Python 3.9+
- OpenCV
- MediaPipe
- NumPy
- SciPy

## Modal GPU Usage

For deployment on Modal with GPU acceleration, the pipeline can be easily adapted to use GPU-accelerated OpenCV operations.