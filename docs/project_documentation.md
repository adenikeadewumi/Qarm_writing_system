# PROJECT DOCUMENTATION

# Handwritten Digit Recognition and Robotic Writing Using the Quanser QArm

**Contributors:** Oyesanmi Iseoluwa and Adenike Adewumi

---

## 1. Introduction

This project explores the integration of computer vision, machine learning, and robotic manipulation through the development of a system that enables a Quanser QArm robot to recognise handwritten digits and physically reproduce them.

The system captures handwritten digits using a webcam, processes the captured image, and uses a pre-trained MNIST-based ONNX model to identify the digit. Following successful recognition and user confirmation, the system generates a trajectory that allows the QArm to reproduce the digit on a writing surface.

The project brings together three major stages:

1. Visual perception and image processing.
2. Machine learning-based digit recognition.
3. Robotic trajectory planning and physical execution.

The overall objective is to demonstrate how information obtained from a camera can be translated into a meaningful physical action by a robotic manipulator.

## 2. Project Objectives

The main objectives were to:

- Develop a camera-based handwritten digit recognition system.
- Preprocess captured images to match the input requirements of an MNIST-based model.
- Perform digit classification using an ONNX model.
- Integrate the recognition system with the Quanser QArm.
- Convert recognised digits into writing paths.
- Generate and validate robot trajectories before execution.
- Enable the QArm to physically reproduce recognised digits.
- Introduce user confirmation and safety checks into the execution process.

## 3. System Architecture

The system follows a sequential perception-to-action architecture.

### 3.1 Overall Workflow

Webcam Capture
      |
      v
Region of Interest Extraction
      |
      v
Image Preprocessing
      |
      v
ONNX Digit Classification
      |
      v
Confidence Evaluation
      |
      v
User Confirmation
      |
      v
Trajectory Generation
      |
      v
Trajectory Validation
      |
      v
User Authorisation
      |
      v
Quanser QArm Execution
      |
      v
Physical Digit Writing

Each stage prepares information required by the next stage. The recognition module determines what should be written, while the trajectory-planning module determines how the robot should perform the writing motion.

## 4. Software Components

The project contains several Python programs, each responsible for a different part of the system.

### 4.1 `mnist_digit_camera.py`

This program handles standalone handwritten digit recognition.

It supports capturing a digit from a camera or processing an existing image. It performs image preprocessing, passes the processed image to the ONNX model, and returns the predicted digit and its confidence.

It does not establish a connection to the QArm.

### 4.2 `mnist_qarm_integration.py`

This is the main integration program.

It connects the digit recognition process to the robotic writing implementation. Once a digit has been recognised and confirmed, the program passes the recognised character to the writing component.

This establishes the connection between the computer vision system and the physical robot.

### 4.3 `qarm_writer_main.py`

This program implements the Cornell-style trajectory planner and writing system.

It converts the required writing strokes into Cartesian waypoints, determines corresponding joint configurations using precomputed reach lookup tables, validates the resulting trajectory, and executes the motion on the QArm.

The implementation uses the QArm's available geometry and kinematic utilities.

### 4.4 `qarm_write_text.py`

This is a separate text-writing implementation.

It contains hand-designed glyph definitions for uppercase letters, digits, spaces, and hyphens. It also includes shared safety and writing utilities.

Unlike the Cornell-style writer, this implementation uses inverse kinematics for individual Cartesian waypoints, using the previous solution as the initial seed for the next waypoint.

It should be distinguished from the main Cornell-style lookup-table approach.

### 4.5 `paddle_camera_ocr.py`

This program represents a separate camera-based OCR experiment.

It is not part of the main MNIST digit recognition and QArm writing pipeline.

### 4.6 `mnist-8.onnx`

This is the existing ONNX model used for handwritten digit classification.

The project uses the model for inference; the model itself was not trained as part of this project.

---

## 5. Image Acquisition and Preprocessing

The first stage of the system is capturing a handwritten digit using a webcam.

A square region of interest (ROI), approximately 280 pixels in size, is used to isolate the digit from the camera frame.

The captured image must be transformed into a format compatible with the MNIST model.

### 5.1 Preprocessing Pipeline

The preprocessing operations include:

1. **Grayscale Conversion**

   The captured image is converted from colour to grayscale. This removes unnecessary colour information and simplifies subsequent processing.

2. **Filtering**

   A blur/filtering operation is applied to reduce noise and improve the quality of the image before thresholding.

3. **Otsu Thresholding**

   Otsu's thresholding method is used to separate the handwritten digit from the background.

   The thresholded image is inverted so that the dark handwritten strokes become white foreground pixels.

4. **Connected-Component Noise Removal**

   Connected-component analysis is used to remove unwanted small components that may interfere with digit recognition.

5. **Digit Cropping**

   The digit is tightly cropped to remove unnecessary surrounding background.

6. **Resizing**

   The cropped digit is resized while preserving its aspect ratio so that its longer dimension fits within a 20-pixel range.

7. **Centre Placement**

   The resized digit is placed at the centre of a black 28 × 28 image.

8. **Normalisation**

   Pixel values are normalised to the range 0 to 1.

9. **Tensor Reshaping**

   The final image is reshaped into a four-dimensional tensor with dimensions:

   `1 × 1 × 28 × 28`

This format is used as the input to the ONNX model.

### 5.2 Why Preprocessing Is Important

The MNIST model expects images with a specific size, orientation, foreground-background arrangement, and numerical range.

A webcam image does not naturally satisfy these requirements. Preprocessing therefore helps reduce differences between the captured image and the type of input expected by the model.

---

## 6. Handwritten Digit Recognition

After preprocessing, the image is passed to the existing `mnist-8.onnx` model.

### 6.1 Model Inference

The model is loaded using OpenCV's Deep Neural Network (DNN) module.

The processed image is supplied as the model input, and the network produces scores corresponding to the ten possible digit classes:

`0, 1, 2, 3, 4, 5, 6, 7, 8, 9`

### 6.2 Prediction and Confidence

The output scores are passed through a softmax operation to obtain probability-like confidence values.

The class with the highest score is selected as the predicted digit.

The system then displays the predicted digit and its confidence.

### 6.3 Confidence Threshold

A confidence threshold of 50% is used.

If the prediction does not meet the required threshold, it is not automatically passed to the writing stage.

When the prediction is acceptable, the user is asked to confirm it by entering:

`YES`

This prevents the robot from proceeding based solely on an uncertain prediction.

---

## 7. Robotic Writing and Trajectory Planning

Following recognition and confirmation, the system must convert the digit into a sequence of robot movements.

The QArm does not understand a digit as a visual character. It requires a sequence of positions and corresponding joint configurations that define the movement of the pen.

The writing system therefore represents digits as collections of strokes.

### 7.1 Stroke Representation

Each digit is represented using a set of connected points or polylines.

The points define the path that the pen must follow to reproduce the digit.

The writing planner converts these strokes into Cartesian waypoints.

The trajectory includes:

- Movement towards the writing position.
- Pen lowering.
- Movement along the writing stroke.
- Pen lifting.
- Travel between separate strokes.
- Movement away from the writing surface.

This allows the robot to reproduce digits consisting of multiple disconnected strokes.

### 7.2 Cartesian Waypoint Generation

The writing path is first represented in Cartesian space.

The planner generates intermediate points along each stroke rather than commanding the robot to move directly from the beginning to the end of a stroke.

The recorded planning parameters include:

| Parameter | Value |
|---|---:|
| Letter height | 50 mm |
| Letter width | 20 mm |
| Letter spacing | 7 mm |
| Line spacing | 30 mm |
| Stroke waypoint spacing | 5 mm |
| Travel waypoint spacing | 15 mm |
| Approach height | 25 mm |
| Pen lift levels | 0, 5, 10, 15 mm |

These parameters define the scale and spacing of the writing paths.

---

## 8. Cornell-Style Lookup-Table Approach

The main robotic writing implementation uses a lookup-table approach to determine joint configurations for the generated Cartesian waypoints.

### 8.1 Kinematic Model

The QArm's geometry and kinematic utilities are used to calculate the relationship between joint configurations and the resulting pen-tip position.

The implementation accounts for the position of the pen tip rather than treating the robot's end-effector reference point as the writing point.

The pen-tip position is calculated using forward kinematics, with the pen protrusion accounted for along the tool axis.

The writing orientation uses the fixed configuration:

`PHI_WRITE = [-10, 65, -40, -10] degrees`

This represents the configured pen-contact pose used by the planner.

### 8.2 Generating Reach Lookup Tables

The planner generates lookup tables by sweeping through valid shoulder configurations and calculating the corresponding reachable positions.

The tables are generated for different pen heights.

During table generation, the implementation:

- Uses the robot's geometric model.
- Retains the required downward-pointing tool configuration.
- Enforces joint limits.
- Checks the monotonic reach range.
- Stores the relationship between reach and the corresponding shoulder and elbow configurations.

The recorded lookup-table reach resolution is 1 mm.

### 8.3 Determining Joint Configurations

For a required Cartesian position, the planner calculates:

- The base angle using the position's angular direction.
- The horizontal reach using the distance from the robot base.
- The required shoulder and elbow configurations using interpolation within the lookup table.

The base angle is calculated from:

`base_angle = atan2(y, x)`

The horizontal reach is calculated from:

`reach = hypot(x, y)`

The lookup-table method avoids repeatedly solving an iterative inverse-kinematics problem for every waypoint.

It also helps maintain consistency between neighbouring points in the writing trajectory.

### 8.4 Joint Limits

The planner enforces the following recorded joint limits:

| Joint | Minimum | Maximum |
|---|---:|---:|
| Base | -170° | +170° |
| Shoulder | -80° | +80° |
| Elbow | -95° | +75° |
| Wrist | -160° | +160° |

The maximum permitted adjacent joint change is 10 degrees.

These checks help prevent invalid or excessively abrupt joint movements.

### 8.5 Forward-Kinematics Verification

After obtaining joint configurations from the lookup tables, the planner uses forward kinematics to verify the resulting positions.

The recorded position verification tolerance is:

`0.0005 m (0.5 mm)`

The planner reports the maximum horizontal-position and height errors.

This provides a way to check whether the calculated joint configurations reproduce the intended Cartesian waypoints.

### 8.6 QArm Kinematic Constraint

The Cornell-style arm configuration and the QArm do not have identical pitch-joint arrangements.

The QArm has two pitch joints where the Cornell arm has three.

Consequently, the QArm can control the pen's reach and height but cannot independently maintain the same pen tilt at every writing position.

The pen tilt can therefore vary with reach, and the implementation reports this behaviour.

---

## 9. Separate Inverse-Kinematics Text Writer

The project also contains an earlier or separate text-writing implementation in `qarm_write_text.py`.

This implementation uses a different approach from the Cornell-style lookup-table planner.

### 9.1 Glyph Database

The program contains hand-designed polyline representations for:

- Uppercase letters A–Z.
- Digits 0–9.
- Spaces.
- Hyphens.

The glyphs are represented using points within unit boxes and scaled to the required writing dimensions.

Curved portions are approximated using connected line segments.

### 9.2 Inverse Kinematics

For each generated Cartesian waypoint, the implementation calculates the corresponding joint configuration using inverse kinematics.

The previous waypoint's joint solution is used as the initial seed for the next calculation.

This helps maintain continuity between successive solutions.

### 9.3 Validation

The implementation checks:

- Whether the calculated joint values are finite.
- Whether the joint configurations satisfy the robot's limits.
- Whether the forward-kinematics position agrees sufficiently with the requested Cartesian position.
- Whether neighbouring joint configurations contain excessively large jumps.

The recorded position-error tolerance for this implementation is 0.005 m (5 mm), and adjacent joint jumps greater than 20 degrees are checked.

The complete trajectory is planned and validated before physical robot movement.

This implementation should not be confused with the Cornell-style writer, which uses precomputed reach lookup tables.

---

## 10. Safety and Physical Execution

Physical robot movement introduces risks that are not present during image processing or offline trajectory generation.

The system therefore uses several checks before allowing the QArm to write.

### 10.1 Human-in-the-Loop Confirmation

The user must confirm the recognised digit by entering:

`YES`

This ensures that the recognised digit is the intended character before trajectory execution proceeds.

### 10.2 Dry-Run Validation

The writing planner generates the complete trajectory before physical execution.

The trajectory is checked for joint limits, joint discontinuities, and position accuracy.

A dry run allows the planned movement to be examined without immediately commanding the physical robot to execute it.

### 10.3 Explicit Writing Authorisation

Physical execution requires an explicit command:

`WRITE`

The user also supplies the calibrated gripper value required by the writing setup.

This provides an additional authorisation stage before the robot begins moving.

### 10.4 Hardware Monitoring

The implementation includes hardware fault monitoring and checks associated with the robot's connection and execution state.

The arm is also configured to return to its HOME position as part of the execution workflow.

The LED indicates the connection state, with green used when active and red when disconnected.

---

## 11. Experimental Results and Observations

The integrated system was tested using handwritten digits captured through a webcam and reproduced using the physical Quanser QArm.

### 11.1 Successful Recognition and Writing

The complete physical workflow was demonstrated:

1. A handwritten digit was presented to the webcam.
2. The system captured and preprocessed the image.
3. The ONNX model classified the digit.
4. The prediction was presented for user confirmation.
5. The writing planner generated the robot trajectory.
6. The QArm physically reproduced the digit.

Several different digits were successfully written by the robot.

### 11.2 Effect of Writing Thickness

An important observation was the difference in recognition reliability between marker-written digits and digits written with a thin-tip pen.

Marker-written digits were generally recognised more reliably.

Thin-tip pen inputs were more likely to be misclassified.

This suggests that stroke thickness and the resulting foreground pixel distribution influence the recognition process.

It also demonstrates the importance of preprocessing and input-image quality when deploying a trained model outside its original dataset environment.

### 11.3 Pen Protrusion

The physical writing experiments used a pen protrusion setting of:

`0 m`

This was the experimental setting used during the physical tests.

---

## 12. Limitations

The project has some limitations that should be considered when interpreting its results.

### 12.1 Recognition Scope

The main integrated pipeline recognises handwritten digits from 0 to 9.

It does not provide general-purpose handwriting recognition or arbitrary text recognition.

### 12.2 Model Training

The ONNX model was an existing model used for inference. Training a new model was not part of the project.

### 12.3 Input Sensitivity

Recognition performance depends on the quality and appearance of the handwritten input.

Differences in stroke thickness can affect the prediction.

### 12.4 Writing Constraints

The physical writing process is constrained by the QArm's geometry, joint limits, reachable workspace, pen configuration, and the assumptions used in trajectory planning.

### 12.5 Pen Orientation

The QArm's available pitch joints limit its ability to maintain an independently controlled pen orientation throughout the writing path.

### 12.6 Separate OCR and Text-Writing Experiments

The PaddleOCR camera experiment and the separate glyph-based text writer are additional project components. They should not be interpreted as features already integrated into the main MNIST digit-to-QArm workflow.

---

## 13. Conclusion

This project demonstrates an integrated application of computer vision, machine learning, and robotic manipulation.

A webcam-based perception system captures handwritten digits, an existing ONNX model performs classification, and a trajectory-planning system converts the recognised digit into movements that can be executed by a Quanser QArm.

The project goes beyond digit classification by connecting the output of a machine learning model to a physical robotic action.

The implementation also demonstrates the importance of image preprocessing, kinematic modelling, trajectory validation, and human-in-the-loop safety when deploying intelligent systems on physical robotic platforms.

The successful physical reproduction of several recognised digits demonstrates the feasibility of the perception-to-action pipeline.

---

## 14. Project Files

| File | Purpose |
|---|---|
| `mnist_digit_camera.py` | Camera-based and image-based digit recognition. |
| `mnist_qarm_integration.py` | Integration of digit recognition with robotic writing. |
| `qarm_writer_main.py` | Cornell-style trajectory planning and QArm execution. |
| `qarm_write_text.py` | Separate glyph-based text-writing implementation and shared utilities. |
| `paddle_camera_ocr.py` | Separate OCR experiment. |
| `mnist-8.onnx` | Existing ONNX digit-classification model. |

## 15. Contributors

**Oyesanmi Iseoluwa**

**Adenike Adewumi**
