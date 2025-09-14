#!/usr/bin/env python3
"""
Real-time rPPG heart rate extraction from live camera feed
Usage: python live_rppg.py
Press 'q' to quit, 'r' to reset buffer
"""
import cv2
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import mediapipe as mp
from collections import deque
import time
import heartpy as hp  # NEW: Import HeartPy for RR estimation

class LiveRPPGProcessor:
    def __init__(self, buffer_duration=10, fps=30):
        self.buffer_duration = buffer_duration  # seconds
        self.fps = fps
        self.max_buffer_size = int(buffer_duration * fps)

        # Signal buffers
        self.r_buffer = deque(maxlen=self.max_buffer_size)
        self.g_buffer = deque(maxlen=self.max_buffer_size)
        self.b_buffer = deque(maxlen=self.max_buffer_size)
        self.timestamps = deque(maxlen=self.max_buffer_size)
        self.rppg_buffer = deque(maxlen=self.max_buffer_size)  # NEW: Buffer for rPPG signal

        # MediaPipe setup
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )

        # Heart rate tracking
        self.current_hr = 0
        self.hr_history = deque(maxlen=10)
        self.last_hr_update = time.time()

        # NEW: Respiratory rate tracking
        self.current_rr = 0
        self.rr_history = deque(maxlen=10)
        self.last_rr_update = time.time()

    def detect_face_roi(self, frame):
        """Detect face region and extract forehead ROI"""
        results = self.face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape

            # Calculate face bounding box
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            # Extract forehead region (upper 1/3 of face)
            forehead_roi = frame[y:y+h//3, x:x+w]

            return forehead_roi, (x, y, w, h//3)

        return None, None

    def add_frame_data(self, roi):
        """Add RGB data from current frame to buffers"""
        if roi is not None and roi.size > 0:
            r_mean = np.mean(roi[:, :, 2])  # Red channel
            g_mean = np.mean(roi[:, :, 1])  # Green channel
            b_mean = np.mean(roi[:, :, 0])  # Blue channel

            self.r_buffer.append(r_mean)
            self.g_buffer.append(g_mean)
            self.b_buffer.append(b_mean)
            self.timestamps.append(time.time())

            # NEW: Compute rPPG signal and add to buffer
            rppg_signal = self.apply_chrom_method(
                np.array(list(self.r_buffer)),
                np.array(list(self.g_buffer)),
                np.array(list(self.b_buffer))
            )
            if len(rppg_signal) > 0:
                self.rppg_buffer.append(rppg_signal[-1])  # Add latest rPPG sample

            return True
        return False

    def apply_chrom_method(self, r_signal, g_signal, b_signal):
        """Apply CHROM method for rPPG signal extraction"""
        if len(r_signal) < 30:  # Need minimum samples
            return np.array([])

        # Normalize signals
        r_norm = (r_signal - np.mean(r_signal)) / (np.std(r_signal) + 1e-8)
        g_norm = (g_signal - np.mean(g_signal)) / (np.std(g_signal) + 1e-8)
        b_norm = (b_signal - np.mean(b_signal)) / (np.std(b_signal) + 1e-8)

        # CHROM transformation
        chrom_x = 3 * r_norm - 2 * g_norm
        chrom_y = 1.5 * r_norm + g_norm - 1.5 * b_norm

        # Combine signals
        alpha = np.std(chrom_x) / (np.std(chrom_y) + 1e-8)
        rppg_signal = chrom_x - alpha * chrom_y

        return rppg_signal

    def estimate_heart_rate(self):
        """Estimate heart rate from current buffer"""
        if len(self.r_buffer) < self.fps * 3:  # Need at least 3 seconds
            return 0

        # Convert buffers to arrays
        r_signal = np.array(list(self.r_buffer))
        g_signal = np.array(list(self.g_buffer))
        b_signal = np.array(list(self.b_buffer))

        # Apply CHROM method
        rppg_signal = self.apply_chrom_method(r_signal, g_signal, b_signal)

        if len(rppg_signal) < self.fps * 3:
            return 0

        # Apply bandpass filter (0.7-4.0 Hz = 42-240 BPM)
        nyquist = self.fps / 2
        low = 0.7 / nyquist
        high = 4.0 / nyquist

        try:
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_signal = signal.filtfilt(b, a, rppg_signal)
        except:
            return 0

        # Apply FFT
        fft_vals = fft(filtered_signal)
        freqs = fftfreq(len(filtered_signal), 1/self.fps)

        # Find dominant frequency in valid HR range
        valid_freqs = (freqs >= 0.7) & (freqs <= 4.0)
        if not np.any(valid_freqs):
            return 0

        valid_fft = np.abs(fft_vals[valid_freqs])
        valid_freq_range = freqs[valid_freqs]

        if len(valid_fft) == 0:
            return 0

        # Find peak frequency
        peak_freq = valid_freq_range[np.argmax(valid_fft)]
        heart_rate = peak_freq * 60  # Convert to BPM

        return heart_rate

    def estimate_respiratory_rate(self):  # NEW: Estimate respiratory rate using FFT
        """Estimate respiratory rate from rPPG buffer using FFT analysis"""
        if len(self.rppg_buffer) < self.fps * 10:  # Need at least 10 seconds
            return 0

        try:
            rppg_signal = np.array(list(self.rppg_buffer))

            # Remove DC component and normalize
            rppg_signal = rppg_signal - np.mean(rppg_signal)
            if np.std(rppg_signal) > 0:
                rppg_signal = rppg_signal / np.std(rppg_signal)
            else:
                return 0

            # Apply low-pass filter to emphasize respiratory modulation
            from scipy import signal as sp_signal
            nyquist = self.fps / 2

            # Low-pass filter at 1 Hz to remove heart rate components
            low_cutoff = 1.0 / nyquist
            b_low, a_low = sp_signal.butter(4, low_cutoff, btype='low')
            low_pass_signal = sp_signal.filtfilt(b_low, a_low, rppg_signal)

            # Bandpass filter for respiratory frequencies (0.1-0.5 Hz = 6-30 breaths/min)
            low = 0.1 / nyquist   # 6 breaths/min
            high = 0.5 / nyquist  # 30 breaths/min
            b, a = sp_signal.butter(4, [low, high], btype='band')
            filtered_signal = sp_signal.filtfilt(b, a, low_pass_signal)

            # Apply FFT to find dominant respiratory frequency
            fft_vals = fft(filtered_signal)
            freqs = fftfreq(len(filtered_signal), 1/self.fps)

            # Focus on respiratory frequency range
            valid_freqs = (freqs >= 0.1) & (freqs <= 0.5)  # 6-30 breaths/min
            if not np.any(valid_freqs):
                return 0

            valid_fft = np.abs(fft_vals[valid_freqs])
            valid_freq_range = freqs[valid_freqs]

            if len(valid_fft) == 0:
                return 0

            # Find peak frequency
            peak_freq = valid_freq_range[np.argmax(valid_fft)]
            respiratory_rate = peak_freq * 60  # Convert to breaths/min

            # Validate result
            if 6 <= respiratory_rate <= 30:
                return respiratory_rate
            else:
                return 0

        except Exception as e:
            print(f"DEBUG: RR estimation failed: {e}")
            return 0

    def update_heart_rate(self):
        """Update heart rate estimate and smooth it"""
        current_time = time.time()

        # Update every 1 second
        if current_time - self.last_hr_update > 1.0:
            hr = self.estimate_heart_rate()

            if 40 <= hr <= 200:  # Reasonable HR range
                self.hr_history.append(hr)
                # Use median of recent estimates for stability
                self.current_hr = np.median(list(self.hr_history))

            # NEW: Update respiratory rate every 10 seconds
            if current_time - self.last_rr_update > 10.0:
                rr = self.estimate_respiratory_rate()
                if rr > 0:
                    self.rr_history.append(rr)
                    self.current_rr = np.median(list(self.rr_history))
                self.last_rr_update = current_time

            self.last_hr_update = current_time

    def reset_buffers(self):
        """Reset all buffers"""
        self.r_buffer.clear()
        self.g_buffer.clear()
        self.b_buffer.clear()
        self.timestamps.clear()
        self.rppg_buffer.clear()  # NEW: Reset rPPG buffer
        self.hr_history.clear()
        self.rr_history.clear()  # NEW: Reset RR history
        self.current_hr = 0
        self.current_rr = 0  # NEW: Reset current RR

    def draw_ui(self, frame, roi_coords):
        """Draw UI elements on frame"""
        # Draw ROI rectangle if face detected
        if roi_coords:
            x, y, w, h = roi_coords
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Forehead ROI', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw heart rate
        hr_text = f"Heart Rate: {self.current_hr:.1f} BPM"
        cv2.putText(frame, hr_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # NEW: Draw respiratory rate
        rr_text = f"Resp Rate: {self.current_rr:.1f} breaths/min"
        cv2.putText(frame, rr_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Draw buffer status
        buffer_pct = (len(self.r_buffer) / self.max_buffer_size) * 100
        buffer_text = f"Buffer: {buffer_pct:.0f}% ({len(self.r_buffer)}/{self.max_buffer_size})"
        cv2.putText(frame, buffer_text, (10, 90),  # Adjusted y-position
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw instructions
        cv2.putText(frame, "Press 'q' to quit, 'r' to reset", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw signal quality indicator
        if len(self.r_buffer) >= self.fps * 3:
            quality_color = (0, 255, 0) if self.current_hr > 0 else (0, 0, 255)
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, quality_color, -1)

        return frame

def main():
    print("Starting live rPPG heart rate monitoring...")
    print("Make sure you have good lighting and your face is visible")
    print("Press 'q' to quit, 'r' to reset buffer")

    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use default camera (0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize processor
    processor = LiveRPPGProcessor(buffer_duration=10, fps=30)

    print("Camera initialized. Starting processing...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Detect face ROI
            roi, roi_coords = processor.detect_face_roi(frame)

            # Add frame data to buffers
            if processor.add_frame_data(roi):
                # Update heart rate estimate
                processor.update_heart_rate()

            # Draw UI
            frame = processor.draw_ui(frame, roi_coords)

            # Display frame
            cv2.imshow('Live rPPG Heart Rate Monitor', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                processor.reset_buffers()
                print("Buffers reset")

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        processor.face_detection.close()
        print("Cleanup complete")

if __name__ == "__main__":
    main()