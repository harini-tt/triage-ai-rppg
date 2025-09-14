#!/usr/bin/env python3
"""
Simple rPPG heart rate extraction using OpenCV, MediaPipe, and signal processing
Replaces physformer with a lightweight CPU-based approach using CHROM method
Usage: python simple_rppg.py <video_file>
"""
import sys
import os
import cv2
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import mediapipe as mp

def detect_face_roi(frame):
    """Detect face region using MediaPipe"""
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)

            # Extract forehead region (upper 1/3 of face)
            x, y, w, h = bbox
            forehead_roi = frame[y:y+h//3, x:x+w]
            return forehead_roi, True

    return None, False

def extract_green_channel_signal(frames):
    """Extract green channel signal from ROI frames"""
    green_signals = []

    for frame in frames:
        if frame is not None and frame.size > 0:
            # Extract green channel and compute mean
            green_mean = np.mean(frame[:, :, 1])  # Green channel
            green_signals.append(green_mean)
        else:
            green_signals.append(0)

    return np.array(green_signals)

def apply_chrom_method(r_signal, g_signal, b_signal):
    """Apply CHROM method for rPPG signal extraction"""
    # Normalize signals
    r_norm = (r_signal - np.mean(r_signal)) / np.std(r_signal)
    g_norm = (g_signal - np.mean(g_signal)) / np.std(g_signal)
    b_norm = (b_signal - np.mean(b_signal)) / np.std(b_signal)

    # CHROM transformation
    chrom_x = 3 * r_norm - 2 * g_norm
    chrom_y = 1.5 * r_norm + g_norm - 1.5 * b_norm

    # Combine signals
    alpha = np.std(chrom_x) / np.std(chrom_y)
    rppg_signal = chrom_x - alpha * chrom_y

    return rppg_signal

def estimate_heart_rate(rppg_signal, fps, freq_range=(0.7, 4.0)):
    """Estimate heart rate from rPPG signal using FFT"""
    if len(rppg_signal) < fps * 5:  # Need at least 5 seconds
        return 0

    # Apply bandpass filter
    nyquist = fps / 2
    low = freq_range[0] / nyquist
    high = freq_range[1] / nyquist

    b, a = signal.butter(4, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, rppg_signal)

    # Apply FFT
    fft_vals = fft(filtered_signal)
    freqs = fftfreq(len(filtered_signal), 1/fps)

    # Find dominant frequency in valid HR range
    valid_freqs = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    valid_fft = np.abs(fft_vals[valid_freqs])
    valid_freq_range = freqs[valid_freqs]

    if len(valid_fft) == 0:
        return 0

    peak_freq = valid_freq_range[np.argmax(valid_fft)]
    heart_rate = peak_freq * 60  # Convert to BPM

    return heart_rate

def extract_heart_rate(video_file):
    """Extract heart rate from video"""
    if not os.path.exists(video_file):
        print(f"Error: Video file {video_file} not found")
        return None

    print(f"Processing video: {video_file}")

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video FPS: {fps}, Total frames: {total_frames}")

    roi_frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi, detected = detect_face_roi(frame)
        if detected and roi is not None:
            roi_frames.append(roi)
        else:
            roi_frames.append(None)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    cap.release()

    if len(roi_frames) == 0:
        print("Error: No face detected in video")
        return None

    print(f"Extracting color signals from {len(roi_frames)} frames...")

    # Extract RGB signals
    r_signals, g_signals, b_signals = [], [], []

    for frame in roi_frames:
        if frame is not None and frame.size > 0:
            r_mean = np.mean(frame[:, :, 2])  # Red channel
            g_mean = np.mean(frame[:, :, 1])  # Green channel
            b_mean = np.mean(frame[:, :, 0])  # Blue channel

            r_signals.append(r_mean)
            g_signals.append(g_mean)
            b_signals.append(b_mean)
        else:
            # Interpolate missing frames
            if len(r_signals) > 0:
                r_signals.append(r_signals[-1])
                g_signals.append(g_signals[-1])
                b_signals.append(b_signals[-1])
            else:
                r_signals.append(0)
                g_signals.append(0)
                b_signals.append(0)

    r_signals = np.array(r_signals)
    g_signals = np.array(g_signals)
    b_signals = np.array(b_signals)

    # Apply CHROM method
    print("Applying CHROM method...")
    rppg_signal = apply_chrom_method(r_signals, g_signals, b_signals)

    # Estimate heart rate
    print("Estimating heart rate...")
    heart_rate = estimate_heart_rate(rppg_signal, fps)

    print(f"Estimated Heart Rate: {heart_rate:.2f} BPM")

    return heart_rate

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python simple_rppg.py <video_file>")
        sys.exit(1)

    video_file = sys.argv[1]
    heart_rate = extract_heart_rate(video_file)

    if heart_rate:
        print(f"\nProcessing complete!")
        print(f"Final heart rate estimate: {heart_rate:.2f} BPM")
    else:
        print("Failed to extract heart rate from video")