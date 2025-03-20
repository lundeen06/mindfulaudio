import librosa
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import os
import argparse
import sys
from pathlib import Path
import json


def analyze_audio_simple(file_path, output_dir=None):
    """
    Perform basic audio analysis on an audio file with minimal computation.
    
    Parameters:
    - file_path: Path to the audio file
    - output_dir: Directory to save figures (optional)
    
    Returns:
    - Dictionary containing analysis results
    """
    print(f"Loading audio file: {file_path}")
    
    # Load the audio file with a lower sample rate for faster processing
    y, sr = librosa.load(file_path, sr=22050)  # Downsample to 22.05kHz
    duration = librosa.get_duration(y=y, sr=sr)
    
    print(f"Audio loaded: {duration:.2f} seconds, {sr} Hz sampling rate")
    
    # Basic audio properties
    results = {
        "duration": duration,
        "sample_rate": sr,
        "n_samples": len(y)
    }
    
    # Create time array for plotting
    time = np.linspace(0, duration, len(y))
    
    # -------------------- AMPLITUDE ANALYSIS --------------------
    
    # Parameters
    hop_length = 1024
    frame_length = 2048
    
    # Compute RMS energy over time (amplitude envelope)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_time = librosa.times_like(rms, sr=sr, hop_length=hop_length)
    
    # Calculate zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr_time = librosa.times_like(zcr, sr=sr, hop_length=hop_length)
    
    # Create amplitude analysis plot
    fig_amplitude = sp.make_subplots(
        rows=2, cols=1, 
        subplot_titles=("Waveform & RMS Energy", "Zero Crossing Rate"),
        vertical_spacing=0.15
    )
    
    # Plot waveform (downsample for faster plotting)
    downsample_factor = max(1, len(y) // 10000)  # Limit to ~10K points for plotting
    fig_amplitude.add_trace(
        go.Scatter(
            x=time[::downsample_factor], 
            y=y[::downsample_factor], 
            name="Waveform", 
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    # Add RMS overlay
    fig_amplitude.add_trace(
        go.Scatter(
            x=rms_time, 
            y=rms, 
            name="RMS Energy", 
            line=dict(color='red', width=1.5)
        ),
        row=1, col=1
    )
    
    # Zero Crossing Rate
    fig_amplitude.add_trace(
        go.Scatter(
            x=zcr_time, 
            y=zcr, 
            name="Zero Crossing Rate", 
            line=dict(color='green')
        ),
        row=2, col=1
    )
    
    fig_amplitude.update_layout(
        height=600, 
        title_text="Amplitude Analysis",
        showlegend=True
    )
    
    # Store amplitude results
    results["amplitude"] = {
        "rms_mean": float(rms.mean()),
        "rms_std": float(np.std(rms)),
        "peak_amplitude": float(np.max(np.abs(y))),
        "zero_crossing_rate_mean": float(zcr.mean())
    }
    
    # -------------------- SPECTRAL ANALYSIS --------------------
    
    # Compute spectrogram
    n_fft = 1024  # Smaller FFT size for faster processing
    spec_hop_length = 512
    
    # Calculate spectrogram
    D = librosa.stft(y, n_fft=n_fft, hop_length=spec_hop_length)
    spectrogram = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Compute spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=spec_hop_length)[0]
    centroid_times = librosa.times_like(spectral_centroid, sr=sr, hop_length=spec_hop_length)
    
    # Get dominant frequencies
    avg_spectrum = np.mean(np.abs(D), axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    dominant_freq_idx = np.argmax(avg_spectrum)
    
    # Create frequency plots
    fig_freq = sp.make_subplots(
        rows=2, cols=1, 
        subplot_titles=("Spectrogram", "Spectral Centroid"),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )
    
    # Spectrogram
    fig_freq.add_trace(
        go.Heatmap(
            z=spectrogram, 
            x=librosa.times_like(spectrogram[0], sr=sr, hop_length=spec_hop_length),
            y=freqs,
            colorscale='Viridis'
        ),
        row=1, col=1
    )
    
    # Spectral Centroid
    fig_freq.add_trace(
        go.Scatter(
            x=centroid_times, 
            y=spectral_centroid, 
            name="Spectral Centroid", 
            line=dict(color='red')
        ),
        row=2, col=1
    )
    
    fig_freq.update_layout(
        height=700, 
        title_text="Frequency Analysis"
    )
    
    # Create average spectrum plot
    fig_avg_spectrum = go.Figure()
    fig_avg_spectrum.add_trace(
        go.Scatter(
            x=freqs, 
            y=librosa.amplitude_to_db(avg_spectrum), 
            name="Average Spectrum"
        )
    )
    
    fig_avg_spectrum.update_layout(
        title="Average Frequency Spectrum",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude (dB)",
        xaxis_type="log"  # Log scale for frequency
    )
    
    # Store frequency analysis results
    results["frequency"] = {
        "spectral_centroid_mean": float(spectral_centroid.mean()),
        "dominant_frequency": float(freqs[dominant_freq_idx]),
        "dominant_frequency_magnitude": float(avg_spectrum[dominant_freq_idx]),
    }
    
    # Save figures if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Get filename without extension for use in output filenames
        base_filename = Path(file_path).stem
        
        # Save figures as HTML
        fig_amplitude.write_html(os.path.join(output_dir, f"{base_filename}_amplitude.html"))
        fig_freq.write_html(os.path.join(output_dir, f"{base_filename}_frequency.html"))
        fig_avg_spectrum.write_html(os.path.join(output_dir, f"{base_filename}_spectrum.html"))
        
        # Save results as JSON
        with open(os.path.join(output_dir, f"{base_filename}_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_dir}/")
    
    return {
        "results": results,
        "figures": {
            "amplitude": fig_amplitude,
            "frequency": fig_freq,
            "avg_spectrum": fig_avg_spectrum
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Basic audio analysis with focus on spectral features')
    parser.add_argument('input_file', help='Path to the input audio file')
    parser.add_argument('-o', '--output', default='output', help='Directory to save output files (default: output)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)
    
    # Analyze the audio file
    analysis = analyze_audio_simple(args.input_file, args.output)
    
    # Display the results
    print("\nAnalysis Results:")
    for category, metrics in analysis["results"].items():
        if isinstance(metrics, dict):
            print(f"\n{category.upper()}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
        else:
            print(f"{category}: {metrics}")
    
    print(f"\nVisualization files saved to {os.path.abspath(args.output)}/")
    print("You can open these HTML files in any web browser to view the interactive plots.")


if __name__ == "__main__":
    main()