"""
MNIST Camera -> QArm Cornell Writer integration.

1. Show a handwritten digit to the camera.
2. Press SPACE to recognize it.
3. Review the prediction.
4. Run the writer's dry-run planner.
5. Only if the plan succeeds, optionally start physical writing.

Keep mnist-8.onnx and the writer files in this repository folder.
"""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mnist-8.onnx"

# Use the writer committed in this repository, not a machine-specific path.
WRITER_SCRIPT = BASE_DIR / "qarm_writer_main.py"
WRITER_DIR = BASE_DIR

CAMERA_INDEX = 0
ROI_SIZE = 280
MIN_CONFIDENCE = 0.50


# ============================================================
# MNIST MODEL
# ============================================================

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
    print(f"Loaded model: {MODEL_PATH}")
    return net


def preprocess_digit(roi):
    """Convert a camera ROI into an MNIST-style 28x28 input."""

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # Remove small connected components/noise.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    cleaned = np.zeros_like(binary)

    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 5:
            cleaned[labels == i] = 255

    ys, xs = np.where(cleaned > 0)

    if len(xs) == 0:
        raise ValueError(
            "No digit detected. Write a darker, clearer digit."
        )

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    digit = cleaned[y0:y1, x0:x1]

    h, w = digit.shape
    scale = 20.0 / max(h, w)

    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    resized = cv2.resize(
        digit, (new_w, new_h), interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    blob = canvas.astype(np.float32).reshape(1, 1, 28, 28)
    blob /= 255.0

    return canvas, blob


def predict_digit(net, blob):
    net.setInput(blob)
    scores = net.forward().reshape(-1)

    if scores.size != 10:
        raise ValueError(
            f"Expected 10 digit scores, got {scores.size}."
        )

    # Convert model scores to a probability distribution.
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    probabilities = exp_scores / exp_scores.sum()

    digit = int(np.argmax(probabilities))
    confidence = float(probabilities[digit])

    return digit, confidence


# ============================================================
# WRITER INTEGRATION
# ============================================================

def run_writer(digit):
    text = str(digit)

    if not WRITER_SCRIPT.exists():
        raise FileNotFoundError(
            f"Writer script not found: {WRITER_SCRIPT}"
        )

    glyph_file = WRITER_DIR / "qarm_write_text.py"

    if not glyph_file.exists():
        raise FileNotFoundError(
            "qarm_write_text.py must be in the repository folder:\n"
            f"{WRITER_DIR}"
        )

    print(f"\nRecognized text: {text!r}")
    print("First, checking the writing plan without moving the arm.")

    dry_run_command = [
        sys.executable,
        str(WRITER_SCRIPT),
        text,
        "--dry-run",
    ]

    result = subprocess.run(
        dry_run_command,
        cwd=str(WRITER_DIR),
        check=False,
    )

    if result.returncode != 0:
        print(
            "\nDry run failed. The arm will NOT be started.\n"
            "Check the writer output above. The digit must have "
            "a corresponding glyph in qarm_write_text.py."
        )
        return

    print("\nDry run completed successfully.")
    print("Review the planned paper coordinates and joint ranges.")

    answer = input(
        "\nDo you want to proceed to physical writing? "
        "Type WRITE to continue: "
    ).strip()

    if answer != "WRITE":
        print("Cancelled. No physical movement requested.")
        return

    grip = input(
        "Enter your previously calibrated gripper value "
        "(0.1 to 0.9): "
    ).strip()

    try:
        grip_value = float(grip)
    except ValueError:
        print("Invalid gripper value. Cancelled.")
        return

    if not 0.1 <= grip_value <= 0.9:
        print("Gripper value must be between 0.1 and 0.9.")
        return

    # The writer will perform its own safety checks and prompt
    # before connecting to/moving the physical QArm.
    command = [
        sys.executable,
        str(WRITER_SCRIPT),
        text,
        "--grip",
        str(grip_value),
    ]

    print(f"\nLaunching writer for digit {text}...")
    subprocess.run(command, cwd=str(WRITER_DIR), check=False)


# ============================================================
# CAMERA
# ============================================================

def main():
    net = load_model()

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera index {CAMERA_INDEX}."
        )

    print("\nCamera ready.")
    print("Place ONE handwritten digit inside the square.")
    print("SPACE = recognize and review")
    print("Q = quit")
    print("The arm will not move unless you explicitly confirm.")

    last_prediction = None

    try:
        while True:
            ok, frame = camera.read()

            if not ok:
                print("Could not read a camera frame.")
                break

            # Crop from the unannotated camera frame.
            h, w = frame.shape[:2]
            size = min(ROI_SIZE, h - 20, w - 20)

            x1 = (w - size) // 2
            y1 = (h - size) // 2
            x2, y2 = x1 + size, y1 + size

            roi = frame[y1:y2, x1:x2].copy()

            display = frame.copy()
            cv2.rectangle(
                display, (x1, y1), (x2, y2), (0, 255, 0), 2
            )

            cv2.putText(
                display,
                "SPACE: recognize | Q: quit",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            if last_prediction is not None:
                digit, confidence = last_prediction
                label = f"Prediction: {digit} ({confidence:.1%})"
                cv2.putText(
                    display, label, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2
                )

            cv2.imshow("MNIST to QArm", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == 32:  # SPACE
                try:
                    preview, blob = preprocess_digit(roi)
                    digit, confidence = predict_digit(net, blob)
                    last_prediction = (digit, confidence)

                    enlarged = cv2.resize(
                        preview,
                        (280, 280),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    cv2.imshow("28x28 MNIST input", enlarged)

                    print(
                        f"\nPrediction: {digit} | "
                        f"confidence: {confidence:.1%}"
                    )

                    if confidence < MIN_CONFIDENCE:
                        print(
                            "Confidence is low. Try again with a "
                            "clearer digit; writer not launched."
                        )
                        continue

                    confirm = input(
                        f"Is the recognized digit {digit} correct? "
                        "Type YES to pass it to the writer: "
                    ).strip().upper()

                    if confirm == "YES":
                        run_writer(digit)
                    else:
                        print("Prediction rejected. Try again.")

                except Exception as error:
                    print(f"Recognition/integration error: {error}")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
