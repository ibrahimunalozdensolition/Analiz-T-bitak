# Erytroscope

## Overview

Professional erythrocyte (red blood cell) velocity analysis application developed with Python, OpenCV, and PySide6. Uses STD (Space-Time Diagram) + Hough Transform method in full compliance with protocol.pdf.

## Analysis Method

### STD (Space-Time Diagram) + Hough Transform

Advanced method compliant with protocol, replacing traditional Optical Flow:

1. **Video Stabilization** - Camera shake correction using feature-based tracking
2. **Hybrid Background Cleaning** - Automatic mode selection (Mean+CLAHE or Square-CLAHE)
3. **Adaptive Optical Density** - OD transformation for low-contrast videos
4. **Vessel Centerline** - Automatic vessel detection and centerline extraction
5. **STD Generation** - Intensity profiles extracted along centerline from all frames
6. **Hough Transform** - Automatic detection of erythrocyte lines in STD
7. **Slope-Based Velocity** - Velocity calculated from each line's slope
8. **Aliasing Control** - Measurement reliability tested according to Nyquist limit

## Features

### Preprocessing Modules

**Video Stabilization:**
- Feature point detection using goodFeaturesToTrack
- Tracking with calcOpticalFlowPyrLK
- Transform calculation with estimateAffinePartial2D
- Frame correction with warpAffine

**Hybrid CLAHE System:**
- Automatic video quality analysis (SNR calculation)
- High quality: Mean+CLAHE mode
- Low quality: Square-based CLAHE mode
- Adaptive histogram equalization

**Adaptive Optical Density:**
- Automatic contrast measurement
- Low contrast: OD = -log(I/I0) transformation
- High contrast: Standard grayscale

### Vessel Processing

**Automatic Vessel Detection:**
- Adaptive thresholding
- Morphological operations (opening, closing)
- Skeleton extraction (thinning)
- Centerline smoothing
- Resampling (equally spaced points)

**Visualization:**
- Centerline shown in blue after ROI selection
- Vessel geometry automatically detected
- Manual mode: Vertical centerline used if detection fails

### Velocity Analysis

**STD (Space-Time Diagram):**
- Intensity profile extracted along centerline for each frame
- Profiles arranged side by side (space x time) forming 2D matrix
- Erythrocyte movement appears as lines

**Hough Transform:**
- Pre-filtering with Gaussian blur
- Canny edge detection
- Line detection with HoughLinesP
- Min line length and max line gap parameters

**Velocity Calculation:**
```
Delta_t = (x2 - x1) / FPS
Delta_s = (y2 - y1) × 1.832 um/pixel
Velocity = Delta_s / Delta_t
```

**Aliasing Control:**
```
Nyquist_limit = (ROI_height / 2) × 1.832 × FPS
Valid: Velocity < Nyquist_limit
Aliasing: Velocity > Nyquist_limit
```

### Statistical Analysis

**Basic Statistics:**
- Average velocity
- Median velocity
- Standard deviation
- Minimum velocity
- Maximum velocity

**Percentiles:**
- P25 (25th percentile)
- P50 (50th percentile / median)
- P75 (75th percentile)
- P90 (90th percentile)
- P95 (95th percentile)

**Measurement Counts:**
- n_valid: Valid (reliable) measurement count
- n_alias: Aliasing suspected measurement count
- n_total: Total detected line count

### Outputs

**Show STD:**
- Space-Time Diagram display
- Hough Transform lines in red
- Visual analysis of erythrocyte movements

**Show Histogram:**
- Velocity distribution with Matplotlib
- Average and median lines
- P25 and P75 markers
- n and SD information

**Save CSV:**
- All statistics
- Percentile values
- Measurement counts
- Analysis information (background mode, OD usage)
- Each velocity measurement

## Installation

### Requirements

```bash
python >= 3.8
opencv-python >= 4.8.0
opencv-contrib-python >= 4.8.0
PySide6 >= 6.5.0
numpy >= 1.24.0
matplotlib >= 3.7.0
```

### Steps

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Running

```bash
source venv/bin/activate
python main.py
```

### Workflow

1. **Load Video** - Select video file (.avi, .mp4, .mov)
2. **FPS Check** - Automatically detected, manually enter if needed using Edit
3. **Select Region** - Select vessel region (ROI)
4. **Centerline** - Automatically detected and shown in blue
5. **Full Analysis** - STD generation and velocity analysis started
6. **Results** - Statistics displayed in cards
7. **Show STD** - Examine Space-Time Diagram
8. **Show Histogram** - View velocity distribution
9. **Save CSV** - Export all results to file

## File Structure

```
main.py                  Main application and GUI
analysis_engine.py       STD generation, Hough Transform, velocity calculation
preprocessing.py         Video stabilization, CLAHE, OD transformation
vessel_processing.py     Vessel detection, centerline extraction
utils.py                 CSV export, histogram, helper functions
requirements.txt         Package dependency list
main_backup.py           Old Optical Flow version (backup)
```

## Technical Details

### Spatial Scale

```
546 pixel = 1000 um
1 pixel = 1.832 um
```

### Video Parameters

- FPS: Automatic detection (CAP_PROP_FPS from video)
- Manual editing support
- Default for invalid FPS: 30 FPS

### ROI (Region of Interest)

- Mouse selection
- Minimum 10x10 pixels
- Vessel centerline automatically extracted
- Vertical line used if centerline < 10 points

### STD Dimensions

```
Height: Centerline point count
Width: Video frame count
Format: Grayscale (0-255)
```

### Hough Transform Parameters

```python
rho = 1
theta = pi/180
threshold = 30
minLineLength = 20
maxLineGap = 5
```

## Interface

### Main Window (1500x950)

**Top Panel:**
- Title: "Erythrocyte Velocity Analysis"
- Method label: "STD + Hough Transform"
- About button (shows developer information)
- Close button

**Left Panel (Video):**
- Video display area (1200x800)
- Load Video, Select Region, Full Analysis buttons
- Live Preview button
- Frame navigation (Previous, Slider, Next)
- Progress bar

**Right Panel (Information and Results):**
- Video information (FPS, scale, frame count) - Font size: +3
- Analysis result cards (average, median, SD, min, max) - Font size: +3
- Valid/Aliasing counts - Font size: +3
- Output buttons (Show STD, Show Histogram, Save CSV)

**Bottom Panel:**
- Status bar

### Colors

- Background: #F5F5F5 (light gray)
- Cards: #FFFFFF (white)
- Primary color: #2196F3 (blue)
- Success: #4CAF50 (green)
- Warning: #FF9800 (orange)
- Error: #f44336 (red)
- ROI frame: Green
- Centerline: Blue
- Hough lines: Red

### Fonts

- All text: Bold (font-weight: 600-700)
- Title: 20px
- Sub-headings: 14-16px (right panel: +3)
- Value cards: 20px (right panel: 23px)
- Buttons: 12px

## Development

### Modular Architecture

Each module can be tested independently:

```python
from preprocessing import VideoPreprocessor
from vessel_processing import VesselProcessor
from analysis_engine import SpaceTimeDiagramAnalyzer
from utils import ResultsManager, HistogramGenerator
```

### Thread Management

Analysis process runs in QThread:
- Main GUI thread doesn't freeze
- Progress tracking with callback
- Error handling (error signal)

### Error Handling

- Video reading errors
- ROI selection cancellation
- Centerline extraction error (fallback: vertical line)
- Hough Transform empty result
- CSV saving errors

## Protocol Compliance

This application meets all basic requirements in protocol.pdf:

- [x] Video stabilization
- [x] Hybrid background cleaning
- [x] Adaptive OD transformation
- [x] Automatic vessel correction
- [x] STD generation
- [x] Hough Transform
- [x] Slope-based velocity calculation
- [x] Aliasing control
- [x] Histogram output
- [x] Statistical analysis
- [x] CSV export

## About

**Application Name:** Erytroscope

**Developer:** Ibrahim UNAL

**Supervisor:** Prof. Dr. Ugur AKSU

**Project:** TUBITAK Erythrocyte Velocity Analysis

**Method:** STD + Hough Transform

**Framework:** PySide6

**License:** -


Video → ROI Çıkarma → ECC Stabilizasyon → Arka Plan Çıkarma → STD → Hough → Hız