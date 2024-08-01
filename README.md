
# Line Stabilization and Detection in Video Frames

This repository contains a Python script for line stabilization and detection in video frames using OpenCV and ORB (Oriented FAST and Rotated BRIEF) feature detection. The script processes a video to stabilize lines and remove detected bounding boxes based on inferences from a pre-trained model. 

## Features

- **Line Stabilization**: The script uses ORB feature detection and homography transformation to stabilize lines in video frames.
- **Bounding Box Detection and Removal**: Bounding boxes are inferred from a pre-trained model and removed from the frames.
- **Frame Shrinking**: Frames are resized to reduce processing time, with coordinates adjusted accordingly.
- **Feature Matching with FLANN**: Feature points are matched using the FLANN (Fast Library for Approximate Nearest Neighbors) algorithm.

## Requirements

- Python 3.x
- OpenCV
- Pandas
- NumPy

Install the required packages using:
```bash
pip install opencv-python pandas numpy
```

## Usage

1. Place your video file and inference parquet file in the same directory as the script.
2. Update the `parquet_file_path` and `video_path` variables if needed.
3. Run the script:
    ```bash
    python line_stabilization.py
    ```

## Script Description

The script is organized into several functions:

### `inference_from_file(frame_index, df=df)`

Fetches bounding box coordinates for a given frame index from a parquet file.

### `create_blocks(frame, line_coordinates, margin=50)`

Creates black blocks around specified line coordinates to mask those areas in the frame.

### `shrink(frame, lines, scale_factor=scale_factor)`

Resizes the frame and adjusts line coordinates based on the given scale factor.

### `unshrink(lines, scale_factor)`

Resizes the line coordinates back to their original scale.

### `line_stabilize(frame, prev_line, kp1, des1, reduced_matches=1, bf=flann)`

Stabilizes lines in the current frame using ORB feature matching and homography transformation.

### `feature_detector(frame, prev_line, orb)`

Detects features around specified lines in the frame using ORB.

### `video_line_stabelize(video_path, show=True, save=True, coordinates=coordinates, calibration_frame_period=3)`

Main function to read the video, process each frame, stabilize lines, and remove bounding boxes. Optionally displays and/or saves the processed video.

## Customization

- **Scale Factor**: Adjust `scale_factor` to balance processing time and accuracy.
- **Calibration Frame Period**: Modify `calibration_frame_period` to control how often the lines are recalibrated using previous frames.
- **Coordinates**: Update the `coordinates` dictionary with the initial coordinates of lines for each video.

## Example

The example below demonstrates how to process a video file named `video_1.mp4` and display the stabilized lines:

```python
video_line_stabelize('video_1.mp4', show=True)
```

## Output

The processed video will be saved as `output_video_1.avi` in the same directory as the input video file.
