# Audio Analysis Tools

This directory contains tools for analyzing audio recordings as part of the mindful audio research project. The main script provides spectral and temporal analysis of audio files with rich visualizations.

## Overview

The `audio-analysis.py` script performs comprehensive audio analysis including:

- Amplitude envelope (RMS energy) calculation
- Zero crossing rate analysis
- Spectral centroid measurement
- Dominant frequency detection
- Spectrogram generation
- Average spectrum analysis

## Requirements

The audio analysis tools require the following Python packages:

```bash
pip install librosa numpy plotly pandas matplotlib
```

## Usage

### Basic Analysis

```bash
python audio-analysis.py path/to/your/audio_file.mp3
```

By default, the script will analyze the audio file and save the results to the `output` directory.

### Custom Output Directory

```bash
python audio-analysis.py path/to/your/audio_file.mp3 -o custom_output_directory
```

## Output

The script generates several interactive HTML visualizations and a JSON file with analysis results.

### Amplitude Analysis

![Amplitude Analysis](output/pilot1.1_amplitude.html)

The amplitude analysis visualization shows:
- The audio waveform (blue)
- RMS energy envelope (red)
- Zero crossing rate (green)

This visualization helps identify changes in audio intensity and tonal characteristics over time.

### Frequency Analysis

![Frequency Analysis](output/pilot1.1_frequency.html)

The frequency analysis visualization includes:
- A spectrogram showing frequency content over time (heat map)
- The spectral centroid track (red line), indicating the "center of mass" of the spectrum

This visualization helps identify the dominant frequencies and their changes over time.

### Average Spectrum

![Average Spectrum](output/pilot1.1_spectrum.html)

The average spectrum visualization shows the frequency distribution averaged over the entire audio file, helping identify the most prominent frequencies in the recording.

### JSON Results

The script also generates a JSON file (`*_results.json`) containing numerical results:

```json
{
  "duration": 120.5,
  "sample_rate": 22050,
  "n_samples": 2657025,
  "amplitude": {
    "rms_mean": 0.123,
    "rms_std": 0.045,
    "peak_amplitude": 0.891,
    "zero_crossing_rate_mean": 0.134
  },
  "frequency": {
    "spectral_centroid_mean": 2345.67,
    "dominant_frequency": 440.0,
    "dominant_frequency_magnitude": 0.567
  }
}
```

## Technical Details

### Feature Calculation

The script calculates audio features using the following methods:

- **RMS Energy**: Root Mean Square calculation of amplitude on windowed frames
- **Zero Crossing Rate**: Rate at which the signal changes from positive to negative or vice versa
- **Spectral Centroid**: Weighted mean of the frequencies present in the signal
- **Dominant Frequency**: Frequency bin with maximum magnitude in the average spectrum

### Visualization

All visualizations are created using Plotly, a Python library that enables interactive graphs viewable in web browsers. This allows for zooming, panning, and hover-based inspection of the data.

## Example Analysis

When analyzing recordings of mindful breathing or vocal practices, look for:

- Changes in RMS energy that correspond to breathing patterns
- Shifts in spectral centroid that might indicate changes in vocal quality
- Stability of dominant frequencies during sustained sounds
- Correlations between audio features and movement data from the camera tracking

## Troubleshooting

- **File format issues**: The script supports most audio formats through Librosa. If you encounter issues, try converting to WAV format.
- **Memory errors**: For very long recordings, the script may use significant memory. Try using shorter segments if needed.
- **Visualization rendering**: The HTML visualizations work best in Chrome or Firefox browsers.