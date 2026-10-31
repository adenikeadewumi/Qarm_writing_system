# Handwritten Digit Recognition and Robotic Writing Using the Quanser QArm

**Contributors:** Adenike Adewumi | Oyesanmi Iseoluwa

## Overview

This project combines computer vision, machine learning, and robotic manipulation to enable a Quanser QArm robot to recognise handwritten digits and physically reproduce them on paper.

A webcam captures a handwritten digit, an existing MNIST-trained ONNX model classifies it, and the predicted digit is passed through a trajectory-planning system that generates the corresponding writing motion for the QArm.

The project demonstrates an end-to-end pipeline connecting visual perception, recognition, trajectory planning, and physical robot execution.

## System Workflow

1. **Image Capture:** A webcam captures a handwritten digit within a defined region of interest.
2. **Image Preprocessing:** The image is converted to grayscale, filtered, thresholded, cropped, resized, centred, and normalised to match the MNIST input format.
3. **Digit Recognition:** An existing ONNX model classifies the processed image into one of ten digit classes (0–9).
4. **User Confirmation:** The predicted digit and confidence score are displayed. The user confirms the prediction before proceeding.
5. **Trajectory Planning:** The writing system generates the required Cartesian and joint-space trajectory.
6. **Safety Validation:** The planned trajectory is checked before physical execution.
7. **Robotic Writing:** The QArm executes the validated trajectory and writes the recognised digit.

## Key Features

- Real-time handwritten digit capture and recognition.
- Image preprocessing for compatibility with the MNIST model.
- ONNX inference using OpenCV DNN.
- Confidence-based prediction confirmation.
- Cartesian trajectory generation for robotic writing.
- Joint-space planning using precomputed reach lookup tables in the Cornell-style writer.
- Joint-limit and trajectory validation.
- Dry-run mode before physical execution.
- Human confirmation before the robot is authorised to write.

## Project Structure

| File | Description |
|---|---|
| `mnist_digit_camera.py` | Standalone handwritten digit recognition from a camera or image. |
| `mnist_qarm_integration.py` | Main integration program connecting digit recognition to the QArm writing system. |
| `qarm_cornell_writer.py` | Cornell-style trajectory planner and physical writing implementation using reach lookup tables. |
| `qarm_write_text.py` | Separate text-writing implementation containing glyph definitions and shared safety utilities. |
| `paddle_camera_ocr.py` | Separate camera-based OCR experiment, not integrated into the main digit-writing pipeline. |
| `mnist-8.onnx` | Existing ONNX model used for handwritten digit classification. |

## Technologies Used

- Python
- OpenCV
- NumPy
- ONNX
- OpenCV DNN
- Quanser QArm
- Quanser QArm Utilities
- Computer Vision
- Robotic Kinematics and Trajectory Planning

## Safety Measures

The robot does not begin writing immediately after making a prediction.

The system requires:
- A prediction confidence of at least 50%.
- Explicit user confirmation of the recognised digit.
- Generation and validation of the complete trajectory.
- An explicit `WRITE` command and calibrated gripper input before physical execution.

## Experimental Observations

The integrated system successfully recognised and physically reproduced several different handwritten digits using the Quanser QArm.

Marker-written digits were generally recognised more reliably than thin-tip pen inputs. This highlights the influence of writing thickness and image quality on recognition performance.

Physical writing experiments were conducted with a pen protrusion setting of 0 m.

## Scope and Limitations

The main integrated workflow focuses on handwritten digit recognition and robotic reproduction of digits from 0 to 9. It is not a general-purpose handwriting recognition or arbitrary text-to-writing system.

A separate PaddleOCR experiment and a text-writing implementation also exist in the project, but they are not the same as the main MNIST-to-QArm integration.

## Contributors

- **Adenike Adewumi**
- **Oyesanmi Iseoluwa**
