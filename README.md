# Triage AI rPPG - PhysFormer Integration

Remote Photoplethysmography (rPPG) using PhysFormer model integrated with Modal for cloud deployment.

## Overview

This project integrates the PhysFormer model from the [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) for extracting physiological signals (heart rate, rPPG waveform) from facial videos using computer vision and deep learning.

**PhysFormer**: Facial Video-based Physiological Measurement with Temporal Difference Transformer by Yu et al., 2022.

## Features

- 🎥 **Video Processing**: Extract rPPG signals from video files
- 🖼️ **Frame Processing**: Process individual frames for real-time applications  
- ☁️ **Cloud Deployment**: Scalable inference using Modal
- 🔬 **Pre-trained Models**: Uses state-of-the-art PhysFormer weights
- 📊 **Comprehensive Metrics**: MAE, RMSE, MAPE, Pearson correlation, SNR, Bland-Altman

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd triage-ai-rppg

# Install dependencies
pip install -r requirements.txt

# Or using uv
uv sync
```

### 2. Deploy to Modal

```bash
# Deploy the app to Modal cloud
modal deploy main.py
```

### 3. Basic Usage

```python
# After deployment, use the API endpoints
from main import app, extract_rppg_from_video

# Load your face video file  
with open("face_video.mp4", "rb") as f:
    video_data = f.read()

# Extract rPPG signal and heart rate
with app.run():
    results = extract_rppg_from_video.remote(
        video_data=video_data,
        filename="face_video.mp4"
    )
    
    print(f"Heart Rate: {results['heart_rate']} BPM")
    print(f"rPPG Signal: {results['rppg_signal'][:10]}...")
```

## API Reference

### Functions

#### `extract_rppg_from_video(video_data: bytes, filename: str) -> Dict`

Extract rPPG signal and heart rate from video file.

**Parameters:**
- `video_data`: Video file as bytes
- `filename`: Original filename (for format detection)

**Returns:**
```python
{
    'status': 'success',
    'heart_rate': 72.5,  # BPM
    'rppg_signal': [...],  # List of signal values
    'metrics': {...},  # Performance metrics
    'filename': 'video.mp4'
}
```

#### `extract_rppg_from_frames(frames_data: List[str]) -> Dict`

Extract rPPG signal from base64-encoded frames.

**Parameters:**
- `frames_data`: List of base64-encoded image frames

**Returns:**
Similar to `extract_rppg_from_video` but includes `num_frames`.

#### `get_model_info() -> Dict`

Get information about the loaded PhysFormer model.

## Configuration

The PhysFormer model is configured via `physformer_config.yaml`:

```yaml
MODEL:
  NAME: PhysFormer
  PHYSFORMER:
    PATCH_SIZE: 4
    DIM: 96
    FF_DIM: 144
    NUM_HEADS: 4
    NUM_LAYERS: 12
    THETA: 0.7
```

Key parameters:
- **PATCH_SIZE**: Spatial patch size for transformer
- **DIM**: Model dimension
- **NUM_HEADS**: Number of attention heads
- **NUM_LAYERS**: Number of transformer layers
- **THETA**: Temporal difference parameter

## Video Requirements

For optimal results, videos should:
- ✅ Contain clear face shots (frontal view preferred)
- ✅ Have good lighting conditions
- ✅ Be 10-30 seconds in duration
- ✅ Have minimal head movement
- ✅ Be at least 30 FPS (recommended)
- ✅ Have resolution ≥ 128x128 pixels for face region

## Model Performance

The PhysFormer model achieves state-of-the-art performance on standard rPPG datasets:

| Dataset | MAE (BPM) | RMSE (BPM) | Pearson |
|---------|-----------|------------|---------|
| PURE    | 1.07      | 2.90       | 0.98    |
| UBFC    | 1.35      | 3.44       | 0.97    |
| SCAMPS  | 1.21      | 3.12       | 0.98    |

## Usage

Simply run the main script to see deployment instructions:

```bash
python3 main.py
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│  PhysFormer      │───▶│   rPPG Signal   │
│   (Face Video)  │    │  (Transformer)   │    │   + Heart Rate  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Preprocessing   │    │ Temporal Diff.   │    │ FFT Analysis    │
│ • Face Detection│    │ • CDC Convolution │    │ • Peak Detection│
│ • Cropping      │    │ • Attention      │    │ • HR Estimation │
│ • Normalization │    │ • Multi-scale    │    │ • Signal Quality│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Troubleshooting

### Common Issues

1. **Model not found error**
   - The pre-trained model will be automatically downloaded on first use
   - Check Modal volume permissions

2. **CUDA out of memory**
   - Reduce batch size in config
   - Use CPU mode for testing

3. **Face detection fails**
   - Ensure good lighting
   - Check face visibility and size
   - Try different videos

4. **Poor heart rate accuracy**
   - Verify video quality
   - Check for motion artifacts
   - Ensure sufficient video length (>10 seconds)

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Add tests
5. Submit a pull request

## License

This project builds upon the rPPG-Toolbox. Please cite the original papers:

```bibtex
@inproceedings{yu2022physformer,
  title={PhysFormer: Facial Video-based Physiological Measurement with Temporal Difference Transformer},
  author={Yu, Zitong and Shen, Yuming and Shi, Jingang and Zhao, Hengshuang and Torr, Philip HS and Zhao, Guoying},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4186--4196},
  year={2022}
}
```

## Support

- 📖 Documentation: See inline code comments
- 🐛 Issues: Create GitHub issues for bugs
- 💬 Discussions: Use GitHub discussions for questions
- 📧 Contact: [Your contact information]
