
import argparse
from pathlib import Path

import cv2
import numpy as np


MODEL_PATH = Path(__file__).resolve().parent / "mnist-8.onnx"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Place mnist-8.onnx beside this script."
        )

    print("Loading MNIST ONNX model...")
    net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
    print("Model loaded successfully.")

    # Print the model's output layer information where available.
    print("OpenCV version:", cv2.__version__)
    return net


def preprocess_digit(image):
    """
    Prepare one dark digit on light paper as a 28x28 MNIST-style image.
    Returns the processed image and normalized NCHW model input.
    """
    if image is None or image.size == 0:
        raise ValueError("Empty image.")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Improve local contrast and reduce sensor noise.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Assumes dark ink on light paper.
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Ignore tiny isolated noise components.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    cleaned = np.zeros_like(binary)
    min_area = max(4, int(binary.size * 0.00005))

    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            cleaned[labels == i] = 255

    ys, xs = np.where(cleaned > 0)

    if len(xs) == 0:
        raise ValueError(
            "No digit found. Use dark ink, white paper, and better lighting."
        )

    # Tight crop around all remaining digit pixels.
    x_min, x_max = int(xs.min()), int(xs.max()) + 1
    y_min, y_max = int(ys.min()), int(ys.max()) + 1
    crop = cleaned[y_min:y_max, x_min:x_max]

    h, w = crop.shape
    if h < 2 or w < 2:
        raise ValueError("The detected digit is too small.")

    # MNIST digits typically occupy roughly a 20x20 area in a 28x28 image.
    scale = 20.0 / max(h, w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    resized = cv2.resize(
        crop, (new_w, new_h), interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    # MNIST input convention: one grayscale channel, NCHW, values 0..1.
    blob = canvas.astype(np.float32).reshape(1, 1, 28, 28) / 255.0

    return canvas, blob


def predict_digit(net, image):
    processed, blob = preprocess_digit(image)

    net.setInput(blob)
    output = net.forward().reshape(-1)

    if output.size != 10:
        raise RuntimeError(
            f"Expected 10 digit outputs, but model returned {output.size}. "
            "We need to inspect this model's output format."
        )

    # MNIST ONNX classifiers commonly return class scores or probabilities.
    # Softmax converts class scores to probabilities; if outputs already
    # resemble probabilities, retain their normalized values.
    if (
        np.all(output >= 0)
        and np.all(output <= 1)
        and abs(float(output.sum()) - 1.0) < 0.05
    ):
        probabilities = output / output.sum()
    else:
        scores = output - np.max(output)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum()

    order = np.argsort(probabilities)[::-1]
    digit = int(order[0])
    confidence = float(probabilities[digit])

    top_predictions = [
        (int(i), float(probabilities[i])) for i in order[:3]
    ]

    return digit, confidence, processed, top_predictions


def show_result(image, processed, digit, confidence, top_predictions):
    print("\n" + "=" * 42)
    print(f"Predicted digit: {digit}")
    print(f"Confidence: {confidence:.1%}")
    print("Top predictions:")

    for number, probability in top_predictions:
        print(f"  {number}: {probability:.1%}")

    print("=" * 42)

    # Enlarge the 28x28 image so the input can be inspected.
    enlarged = cv2.resize(
        processed, (280, 280), interpolation=cv2.INTER_NEAREST
    )

    cv2.imshow("Original input", image)
    cv2.imshow("MNIST input (28x28 enlarged)", enlarged)


def recognize_image(net, image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    digit, confidence, processed, top = predict_digit(net, image)
    show_result(image, processed, digit, confidence, top)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_camera(net, camera_index):
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        raise RuntimeError(
            f"Cannot open camera {camera_index}. Try --camera 1."
        )

    message = "Place one dark digit in the box. SPACE = predict, Q = quit"

    print("\nCamera opened.")
    print("SPACE = recognize the digit")
    print("Q = quit")
    print("This script does not move the QArm.")

    try:
        while True:
            ok, raw_frame = camera.read()

            if not ok:
                print("Failed to read camera frame.")
                break

            # IMPORTANT: preserve the clean frame before drawing overlays.
            display_frame = raw_frame.copy()

            height, width = raw_frame.shape[:2]
            box_size = min(300, height - 20, width - 20)

            if box_size <= 0:
                raise RuntimeError("Camera frame is too small.")

            x1 = (width - box_size) // 2
            y1 = (height - box_size) // 2
            x2 = x1 + box_size
            y2 = y1 + box_size

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                message[:75],
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            cv2.imshow("MNIST Digit Camera", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                # Crop from the original frame, not the green-box overlay.
                roi = raw_frame[y1:y2, x1:x2].copy()

                try:
                    digit, confidence, processed, top = predict_digit(
                        net, roi
                    )

                    message = f"Digit: {digit} | Confidence: {confidence:.1%}"

                    print("\nPrediction:", digit)
                    print(f"Confidence: {confidence:.1%}")
                    print("Top 3:", ", ".join(
                        f"{n} ({p:.1%})" for n, p in top
                    ))

                    enlarged = cv2.resize(
                        processed,
                        (280, 280),
                        interpolation=cv2.INTER_NEAREST
                    )
                    cv2.imshow("MNIST input (28x28 enlarged)", enlarged)

                except ValueError as error:
                    message = str(error)
                    print("Preprocessing error:", error)

    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Recognize one handwritten digit using an MNIST ONNX model."
    )

    parser.add_argument(
        "--image",
        type=Path,
        help="Path to a saved image of one handwritten digit."
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index; default is 0."
    )

    args = parser.parse_args()
    net = load_model()

    if args.image:
        recognize_image(net, args.image)
    else:
        run_camera(net, args.camera)


if __name__ == "__main__":
    main()
