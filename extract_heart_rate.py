#!/usr/bin/env python3
"""
Extract heart rate from PhysFormer rPPG output .mat file
"""
import scipy.io
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def extract_heart_rate_from_mat(mat_file_path, framerate=23.854083, show_plot=False):
    """
    Extract heart rate from PhysFormer .mat output file

    Args:
        mat_file_path: Path to the .mat file containing rPPG signals
        framerate: Frame rate of the video (frames per second)
        show_plot: Whether to show the rPPG signal and spectrum plot

    Returns:
        heart_rate: Estimated heart rate in BPM
    """
    # Load the .mat file
    mat_data = scipy.io.loadmat(mat_file_path)

    # Extract the rPPG signal
    # The key is typically 'outputs_rPPG_concat'
    rppg_signals = mat_data['outputs_rPPG_concat']

    print(f"rPPG data shape: {rppg_signals.shape}")
    print(f"rPPG data type: {type(rppg_signals)}")

    # Flatten the signal if it's not already 1D
    if len(rppg_signals.shape) > 1:
        # Take the first signal if multiple signals
        rppg_signal = rppg_signals[0].flatten()
    else:
        rppg_signal = rppg_signals.flatten()

    print(f"rPPG signal length: {len(rppg_signal)}")
    print(f"rPPG signal range: [{rppg_signal.min():.6f}, {rppg_signal.max():.6f}]")

    # Remove DC component (mean)
    rppg_signal = rppg_signal - np.mean(rppg_signal)

    # Apply bandpass filter (0.7-4.0 Hz for heart rate 42-240 BPM)
    nyquist = framerate / 2
    low = 0.7 / nyquist
    high = 4.0 / nyquist

    # Design Butterworth bandpass filter
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_rppg = signal.filtfilt(b, a, rppg_signal)

    # Compute power spectral density using Welch's method
    f, psd = signal.welch(filtered_rppg, framerate, nperseg=min(256, len(filtered_rppg)))

    # Find the frequency with maximum power in the heart rate range
    hr_range_mask = (f >= 0.7) & (f <= 4.0)  # 42-240 BPM
    f_hr = f[hr_range_mask]
    psd_hr = psd[hr_range_mask]

    # Find peak frequency
    peak_idx = np.argmax(psd_hr)
    peak_freq = f_hr[peak_idx]

    # Convert frequency to BPM
    heart_rate = peak_freq * 60

    print(f"Peak frequency: {peak_freq:.3f} Hz")
    print(f"Estimated heart rate: {heart_rate:.1f} BPM")

    if show_plot:
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(rppg_signal)
        plt.title('Raw rPPG Signal')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

        plt.subplot(2, 2, 2)
        plt.plot(filtered_rppg)
        plt.title('Filtered rPPG Signal')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

        plt.subplot(2, 2, 3)
        plt.plot(f, psd)
        plt.axvline(peak_freq, color='r', linestyle='--', label=f'Peak: {peak_freq:.3f} Hz')
        plt.xlim(0, 5)
        plt.title('Power Spectral Density')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(f_hr * 60, psd_hr)  # Convert to BPM for x-axis
        plt.axvline(heart_rate, color='r', linestyle='--', label=f'HR: {heart_rate:.1f} BPM')
        plt.title('Heart Rate Range Spectrum')
        plt.xlabel('Heart Rate (BPM)')
        plt.ylabel('PSD')
        plt.legend()

        plt.tight_layout()
        plt.savefig('rppg_analysis.png', dpi=150)
        plt.show()

    return heart_rate

if __name__ == "__main__":
    import sys

    mat_file = "Inference_Physformer_TDC07_sharp2_hid96_head4_layer12_VIPL/Inference_Physformer_TDC07_sharp2_hid96_head4_layer12_VIPL.mat"

    if len(sys.argv) > 1:
        mat_file = sys.argv[1]

    try:
        hr = extract_heart_rate_from_mat(mat_file, show_plot=True)
        print(f"\n🫀 Final estimated heart rate: {hr:.1f} BPM")
    except Exception as e:
        print(f"Error processing {mat_file}: {e}")