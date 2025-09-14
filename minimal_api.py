#!/usr/bin/env python3
"""
Minimal FastAPI server for rPPG vital signs processing
Simple POST/GET endpoints for external UI integration
"""
import base64
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time

# Import our rPPG processor
from live_rppg import LiveRPPGProcessor

app = FastAPI(title="rPPG Vital Signs API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global processor instance
processor = LiveRPPGProcessor(buffer_duration=10, fps=30)

# Request/Response models
class FrameRequest(BaseModel):
    frame_data: str  # base64 encoded image

class VitalsResponse(BaseModel):
    heart_rate: float
    respiratory_rate: float
    timestamp: float
    signal_quality: str
    buffer_status: int
    buffer_seconds: float

class ProcessResponse(BaseModel):
    success: bool
    message: str
    vitals: VitalsResponse

@app.post("/process_frame", response_model=ProcessResponse)
async def process_frame(request: FrameRequest):
    """
    Process a single video frame and return updated vital signs
    """
    try:
        # Decode base64 image
        img_data = base64.b64decode(request.frame_data)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Process frame through rPPG pipeline
        roi, roi_coords = processor.detect_face_roi(frame)

        if processor.add_frame_data(roi):
            # Update heart rate estimate
            processor.update_heart_rate()

            vitals = VitalsResponse(
                heart_rate=float(processor.current_hr),
                respiratory_rate=float(processor.current_rr),
                timestamp=time.time(),
                signal_quality="good" if processor.current_hr > 0 else "poor",
                buffer_status=int((len(processor.r_buffer) / processor.max_buffer_size) * 100),
                buffer_seconds=len(processor.r_buffer) / processor.fps
            )

            return ProcessResponse(
                success=True,
                message="Frame processed successfully",
                vitals=vitals
            )
        else:
            return ProcessResponse(
                success=False,
                message="No face detected in frame",
                vitals=VitalsResponse(
                    heart_rate=0.0,
                    respiratory_rate=0.0,
                    timestamp=time.time(),
                    signal_quality="no_face",
                    buffer_status=0,
                    buffer_seconds=0.0
                )
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/vitals", response_model=VitalsResponse)
async def get_vitals():
    """
    Get current vital signs without processing a frame
    """
    return VitalsResponse(
        heart_rate=float(processor.current_hr),
        respiratory_rate=float(processor.current_rr),
        timestamp=time.time(),
        signal_quality="good" if processor.current_hr > 0 else "poor",
        buffer_status=int((len(processor.r_buffer) / processor.max_buffer_size) * 100),
        buffer_seconds=len(processor.r_buffer) / processor.fps
    )

@app.post("/reset")
async def reset_buffers():
    """
    Reset all processing buffers
    """
    processor.reset_buffers()
    return {
        "success": True,
        "message": "Buffers reset successfully",
        "timestamp": time.time()
    }

@app.get("/status")
async def get_status():
    """
    Get processor status and buffer info
    """
    return {
        "buffer_size": len(processor.r_buffer),
        "max_buffer_size": processor.max_buffer_size,
        "buffer_percentage": int((len(processor.r_buffer) / processor.max_buffer_size) * 100),
        "buffer_seconds": len(processor.r_buffer) / processor.fps,
        "fps": processor.fps,
        "is_ready_for_hr": len(processor.r_buffer) >= processor.fps * 3,
        "is_ready_for_rr": len(processor.rppg_buffer) >= processor.fps * 10,
        "timestamp": time.time()
    }

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    print("Starting minimal rPPG API server...")
    print("Endpoints:")
    print("  POST /process_frame - Process video frame")
    print("  GET  /vitals        - Get current vitals")
    print("  POST /reset         - Reset buffers")
    print("  GET  /status        - Get buffer status")
    print("  GET  /health        - Health check")
    print("\nServer running at http://localhost:8000")
    print("API docs at http://localhost:8000/docs")

    uvicorn.run(
        "minimal_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )