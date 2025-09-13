import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import pandas as pd
import argparse
import os
from pathlib import Path

class rPPGAnalyzer:
    """
    Analyze rPPG signals from PhysFormer output and extract vital signs like heart rate
    """

    def __init__(self, sampling_rate=30):
        """
        Initialize the analyzer

        Args:
            sampling_rate (int): Video frame rate (FPS) - default 30 for most videos
        """
        self.sampling_rate = sampling_rate
        self.hr_freq_range = (0.7, 4.0)  # Heart rate frequency range (0.7-4 Hz = 42-240 BPM)

    def parse_physformer_output(self, file_path):
        """
        Parse PhysFormer output file to extract rPPG signals

        Args:
            file_path (str): Path to PhysFormer output text file

        Returns:
            dict: Parsed data containing subject info and rPPG signals
        """
        results = []

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('p') == False:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                # Parse subject info
                subject_path = parts[0]  # e.g., "p10/v1/source1"
                segment_count = int(parts[1])  # Number of signal segments
                ground_truth_hr = float(parts[2])  # Ground truth heart rate
                estimated_hr = float(parts[3])  # PhysFormer's HR estimate

                # Extract rPPG signal values (skip filter coefficients at the end)
                rppg_values = []
                for i in range(4, len(parts)):
                    val = float(parts[i])
                    # Stop when we hit the filter coefficients (they start with 0.000000)
                    if val == 0.0 and i > 100:  # Assume signal has more than 100 samples
                        break
                    rppg_values.append(val)

                results.append({
                    'subject_path': subject_path,
                    'segment_count': segment_count,
                    'ground_truth_hr': ground_truth_hr,
                    'physformer_hr': estimated_hr,
                    'rppg_signal': np.array(rppg_values)
                })

        return results

    def bandpass_filter(self, signal_data, lowcut=0.7, highcut=4.0, order=4):
        """
        Apply bandpass filter to isolate heart rate frequencies

        Args:
            signal_data (array): Input rPPG signal
            lowcut (float): Low frequency cutoff (Hz)
            highcut (float): High frequency cutoff (Hz)
            order (int): Filter order

        Returns:
            array: Filtered signal
        """
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist

        b, a = signal.butter(order, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, signal_data)

        return filtered_signal

    def estimate_heart_rate_welch(self, rppg_signal, segment_overlap=0.5):
        """
        Estimate heart rate using Welch's method for Power Spectral Density

        Args:
            rppg_signal (array): rPPG signal
            segment_overlap (float): Overlap between segments for Welch's method

        Returns:
            dict: Heart rate analysis results
        """
        # Normalize signal
        rppg_normalized = (rppg_signal - np.mean(rppg_signal)) / np.std(rppg_signal)

        # Apply bandpass filter
        rppg_filtered = self.bandpass_filter(rppg_normalized)

        # Calculate power spectral density using Welch's method
        nperseg = min(len(rppg_filtered) // 3, 256)  # Segment length
        freqs, psd = signal.welch(rppg_filtered,
                                fs=self.sampling_rate,
                                nperseg=nperseg,
                                noverlap=int(nperseg * segment_overlap))

        # Find frequencies in heart rate range
        hr_mask = (freqs >= self.hr_freq_range[0]) & (freqs <= self.hr_freq_range[1])
        hr_freqs = freqs[hr_mask]
        hr_psd = psd[hr_mask]

        if len(hr_freqs) == 0:
            return {
                'estimated_hr': 0,
                'dominant_freq': 0,
                'confidence': 0,
                'freqs': freqs,
                'psd': psd,
                'signal_quality': 'poor'
            }

        # Find dominant frequency
        dominant_idx = np.argmax(hr_psd)
        dominant_freq = hr_freqs[dominant_idx]

        # Convert to BPM
        estimated_hr = dominant_freq * 60

        # Calculate confidence based on spectral power concentration
        total_power = np.sum(hr_psd)
        peak_power = hr_psd[dominant_idx]
        confidence = peak_power / total_power if total_power > 0 else 0

        # Assess signal quality
        snr = np.max(hr_psd) / np.mean(hr_psd) if np.mean(hr_psd) > 0 else 0
        if snr > 3:
            quality = 'good'
        elif snr > 2:
            quality = 'fair'
        else:
            quality = 'poor'

        return {
            'estimated_hr': estimated_hr,
            'dominant_freq': dominant_freq,
            'confidence': confidence,
            'snr': snr,
            'freqs': freqs,
            'psd': psd,
            'signal_quality': quality,
            'filtered_signal': rppg_filtered
        }

    def calculate_hrv_metrics(self, rppg_signal):
        """
        Calculate Heart Rate Variability (HRV) metrics

        Args:
            rppg_signal (array): rPPG signal

        Returns:
            dict: HRV metrics
        """
        # Find peaks in the signal (R-peaks equivalent)
        filtered_signal = self.bandpass_filter(rppg_signal)
        peaks, _ = signal.find_peaks(filtered_signal, distance=int(self.sampling_rate * 0.4))

        if len(peaks) < 3:
            return {
                'rmssd': 0,
                'sdnn': 0,
                'mean_rr': 0,
                'hr_variability': 0
            }

        # Calculate RR intervals (time between peaks)
        rr_intervals = np.diff(peaks) / self.sampling_rate * 1000  # Convert to milliseconds

        # Calculate HRV metrics
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))  # Root mean square of successive differences
        sdnn = np.std(rr_intervals)  # Standard deviation of RR intervals
        mean_rr = np.mean(rr_intervals)

        return {
            'rmssd': rmssd,
            'sdnn': sdnn,
            'mean_rr': mean_rr,
            'hr_variability': sdnn / mean_rr if mean_rr > 0 else 0,
            'peak_count': len(peaks)
        }

    def analyze_file(self, file_path, output_dir="analysis_results"):
        """
        Analyze a PhysFormer output file and generate comprehensive results

        Args:
            file_path (str): Path to PhysFormer output file
            output_dir (str): Directory to save analysis results

        Returns:
            list: Analysis results for all subjects
        """
        # Parse the file
        parsed_data = self.parse_physformer_output(file_path)

        if not parsed_data:
            print(f"❌ No valid data found in {file_path}")
            return []

        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)

        all_results = []

        for i, data in enumerate(parsed_data):
            print(f"\n🔍 Analyzing subject: {data['subject_path']}")

            # Perform heart rate analysis
            hr_analysis = self.estimate_heart_rate_welch(data['rppg_signal'])

            # Calculate HRV metrics
            hrv_metrics = self.calculate_hrv_metrics(data['rppg_signal'])

            # Compile results
            results = {
                'subject_path': data['subject_path'],
                'signal_length_samples': len(data['rppg_signal']),
                'signal_duration_seconds': len(data['rppg_signal']) / self.sampling_rate,

                # Heart Rate Results
                'ground_truth_hr': data['ground_truth_hr'],
                'physformer_hr': data['physformer_hr'],
                'welch_estimated_hr': hr_analysis['estimated_hr'],
                'dominant_frequency_hz': hr_analysis['dominant_freq'],

                # Signal Quality
                'signal_quality': hr_analysis['signal_quality'],
                'confidence_score': hr_analysis['confidence'],
                'snr': hr_analysis['snr'],

                # HRV Metrics
                'rmssd_ms': hrv_metrics['rmssd'],
                'sdnn_ms': hrv_metrics['sdnn'],
                'mean_rr_ms': hrv_metrics['mean_rr'],
                'hr_variability': hrv_metrics['hr_variability'],

                # Error Analysis
                'physformer_error_bpm': abs(data['physformer_hr'] - data['ground_truth_hr']),
                'welch_error_bpm': abs(hr_analysis['estimated_hr'] - data['ground_truth_hr']),
            }

            all_results.append(results)

            # Generate plots
            self.create_analysis_plots(data, hr_analysis, results, output_dir, i)

        # Save summary CSV
        self.save_summary_csv(all_results, output_dir)

        # Print summary
        self.print_analysis_summary(all_results)

        return all_results

    def create_analysis_plots(self, data, hr_analysis, results, output_dir, subject_idx):
        """Create visualization plots for the analysis"""

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Time vector
        time = np.arange(len(data['rppg_signal'])) / self.sampling_rate

        # Plot 1: Raw and Filtered Signal
        axes[0].plot(time, data['rppg_signal'], 'b-', alpha=0.7, label='Raw rPPG Signal')
        axes[0].plot(time, hr_analysis['filtered_signal'], 'r-', linewidth=2, label='Filtered Signal')
        axes[0].set_title(f"rPPG Signal Analysis - {data['subject_path']}")
        axes[0].set_xlabel('Time (seconds)')
        axes[0].set_ylabel('Amplitude')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Power Spectral Density
        axes[1].plot(hr_analysis['freqs'], hr_analysis['psd'])
        axes[1].axvline(hr_analysis['dominant_freq'], color='r', linestyle='--',
                       label=f'Dominant Freq: {hr_analysis["dominant_freq"]:.2f} Hz')
        axes[1].axvspan(self.hr_freq_range[0], self.hr_freq_range[1], alpha=0.2, color='green',
                       label='HR Range')
        axes[1].set_title('Power Spectral Density')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Power')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, 5)

        # Plot 3: Heart Rate Comparison
        hr_methods = ['Ground Truth', 'PhysFormer', 'Welch Method']
        hr_values = [data['ground_truth_hr'], data['physformer_hr'], hr_analysis['estimated_hr']]
        colors = ['green', 'blue', 'red']

        bars = axes[2].bar(hr_methods, hr_values, color=colors, alpha=0.7)
        axes[2].set_title('Heart Rate Comparison')
        axes[2].set_ylabel('Heart Rate (BPM)')
        axes[2].grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, hr_values):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.1f}', ha='center', va='bottom')

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(output_dir, f'subject_{subject_idx:02d}_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Plot saved: {plot_path}")

    def save_summary_csv(self, results, output_dir):
        """Save analysis results to CSV file"""

        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, 'heart_rate_analysis_summary.csv')
        df.to_csv(csv_path, index=False)

        print(f"💾 Summary saved: {csv_path}")

    def print_analysis_summary(self, results):
        """Print a summary of the analysis results"""

        print("\n" + "="*60)
        print("📊 HEART RATE ANALYSIS SUMMARY")
        print("="*60)

        for i, result in enumerate(results):
            print(f"\n👤 Subject {i+1}: {result['subject_path']}")
            print(f"   📏 Signal Duration: {result['signal_duration_seconds']:.1f} seconds")
            print(f"   💗 Ground Truth HR: {result['ground_truth_hr']:.1f} BPM")
            print(f"   🤖 PhysFormer HR: {result['physformer_hr']:.1f} BPM (Error: ±{result['physformer_error_bpm']:.1f})")
            print(f"   📈 Welch Method HR: {result['welch_estimated_hr']:.1f} BPM (Error: ±{result['welch_error_bpm']:.1f})")
            print(f"   📊 Signal Quality: {result['signal_quality'].upper()}")
            print(f"   🎯 Confidence: {result['confidence_score']:.3f}")

            if result['rmssd_ms'] > 0:
                print(f"   💓 HRV RMSSD: {result['rmssd_ms']:.1f} ms")
                print(f"   📏 HRV SDNN: {result['sdnn_ms']:.1f} ms")

        # Overall statistics
        if len(results) > 1:
            physformer_errors = [r['physformer_error_bpm'] for r in results]
            welch_errors = [r['welch_error_bpm'] for r in results]

            print(f"\n📈 OVERALL PERFORMANCE:")
            print(f"   PhysFormer MAE: {np.mean(physformer_errors):.2f} ± {np.std(physformer_errors):.2f} BPM")
            print(f"   Welch Method MAE: {np.mean(welch_errors):.2f} ± {np.std(welch_errors):.2f} BPM")

def main():
    """Command line interface for rPPG analysis"""
    parser = argparse.ArgumentParser(description="Analyze PhysFormer rPPG output and extract heart rate metrics")
    parser.add_argument("input_file", help="Path to PhysFormer output text file")
    parser.add_argument("--sampling_rate", type=int, default=30, help="Video sampling rate (FPS)")
    parser.add_argument("--output_dir", default="analysis_results", help="Output directory for results")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"❌ Error: File not found: {args.input_file}")
        return 1

    print("🔬 rPPG Signal Analyzer")
    print("="*40)
    print(f"📁 Input file: {args.input_file}")
    print(f"🎥 Sampling rate: {args.sampling_rate} FPS")
    print(f"📂 Output directory: {args.output_dir}")

    # Create analyzer and run analysis
    analyzer = rPPGAnalyzer(sampling_rate=args.sampling_rate)
    results = analyzer.analyze_file(args.input_file, args.output_dir)

    if results:
        print(f"\n✅ Analysis completed! Check '{args.output_dir}' for detailed results.")
        return 0
    else:
        print("\n❌ Analysis failed!")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())