"""
detect.py — Real-Time Face Mask Detection
==========================================
Uses:
    • OpenCV Haar Cascade → detect faces in frame
    • Trained CNN (mask_cnn.pth) → classify each face: Mask / No Mask

Usage:
    python detect.py                     # webcam (default)
    python detect.py --source image.jpg  # single image
    python detect.py --source video.mp4  # video file
"""

import argparse
import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from model import MaskCNN

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH    = "mask_cnn.pth"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES   = {0: "Mask", 1: "No Mask"}
CLASS_COLORS  = {0: (0, 200, 0), 1: (0, 0, 230)}   # BGR: green / red
IMG_SIZE      = 64

# Haar Cascade for frontal face detection (ships with OpenCV)
FACE_CASCADE  = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ── Preprocessing for a single face crop ─────────────────────────────────────
_preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])


def load_model(path: str) -> MaskCNN:
    """Load trained weights into the CNN."""
    model = MaskCNN().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    print(f"Model loaded from '{path}'  (device: {DEVICE})")
    return model


def predict_face(model: MaskCNN, face_bgr: np.ndarray):
    """
    Given a BGR face crop (numpy array), return (class_id, confidence).

    Steps:
        1. BGR → RGB  (OpenCV uses BGR, PyTorch expects RGB)
        2. numpy → PIL Image
        3. Apply transforms
        4. Add batch dimension: (C,H,W) → (1,C,H,W)
        5. Forward pass → logits
        6. Softmax → probabilities
    """
    face_rgb  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(face_rgb)
    tensor    = _preprocess(pil_img).unsqueeze(0).to(DEVICE)  # (1,3,64,64)

    with torch.no_grad():
        logits  = model(tensor)                     # (1, 2)
        probs   = torch.softmax(logits, dim=1)      # (1, 2) probabilities
        conf, pred = torch.max(probs, dim=1)

    return pred.item(), conf.item()


def draw_label(frame, x, y, w, h, label: str, color: tuple, conf: float):
    """Draw bounding box + label on frame."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    text = f"{label}: {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

    # Filled rectangle behind text for readability
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 4, y), color, -1)
    cv2.putText(frame, text, (x + 2, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def process_frame(frame: np.ndarray, model: MaskCNN) -> np.ndarray:
    """
    Pipeline for a single frame:
        1. Convert to grayscale for faster face detection
        2. Detect faces with Haar Cascade
        3. For each detected face → crop → CNN predict → draw result
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,      # how much to shrink image at each scale
        minNeighbors=5,        # min detections needed to keep a rectangle
        minSize=(60, 60)       # ignore tiny detections
    )

    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]
        if face_crop.size == 0:
            continue

        class_id, confidence = predict_face(model, face_crop)
        label  = CLASS_NAMES[class_id]
        color  = CLASS_COLORS[class_id]
        draw_label(frame, x, y, w, h, label, color, confidence)

    # Show face count
    cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return frame


# ── Main entry points ─────────────────────────────────────────────────────────
def run_webcam(model: MaskCNN):
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = process_frame(frame, model)
        cv2.imshow("Face Mask Detection — Press Q to quit", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image(model: MaskCNN, path: str):
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] Cannot load image: {path}")
        return

    result = process_frame(frame, model)
    cv2.imshow(f"Result — {path}", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    out_path = "result_" + path.split("/")[-1]
    cv2.imwrite(out_path, result)
    print(f"Saved to {out_path}")


def run_video(model: MaskCNN, path: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter("result_output.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    print("Processing video... press 'q' to stop.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = process_frame(frame, model)
        out.write(result)
        cv2.imshow("Face Mask Detection", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Saved result_output.mp4")


def main():
    parser = argparse.ArgumentParser(description="Face Mask Detector")
    parser.add_argument("--source", type=str, default="webcam",
                        help="'webcam', path to image, or path to video")
    args = parser.parse_args()

    model = load_model(MODEL_PATH)

    if args.source == "webcam":
        run_webcam(model)
    elif args.source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        run_image(model, args.source)
    else:
        run_video(model, args.source)


if __name__ == "__main__":
    main()
