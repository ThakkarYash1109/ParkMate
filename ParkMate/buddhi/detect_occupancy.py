"""
BUDDHI - ParkMate Occupancy Detection Module
---------------------------------------------
Detects vehicles in a parking lot image/frame using YOLOv8 (Ultralytics)
and reports which slots are occupied vs empty.

Requirements (already installed by you):
    pip install ultralytics opencv-python

Usage:
    python detect_occupancy.py --source parking.jpg
    python detect_occupancy.py --source 0            # webcam
    python detect_occupancy.py --source video.mp4    # video file
"""

import argparse
import cv2
from ultralytics import YOLO

# COCO class IDs for vehicles we care about
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

CONF_THRESHOLD = 0.35


def load_model(model_path: str = "yolov8n.pt") -> YOLO:
    """Load a YOLOv8 model. yolov8n.pt is small/fast - good for a prototype."""
    return YOLO(model_path)


def detect_vehicles(model: YOLO, frame):
    """
    Run YOLO on a single frame and return:
      - annotated frame (with boxes drawn)
      - list of detections: (x1, y1, x2, y2, class_name, confidence)
    """
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in VEHICLE_CLASSES:
            continue  # skip non-vehicle objects (person, sign, etc.)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        label = VEHICLE_CLASSES[cls_id]
        detections.append((x1, y1, x2, y2, label, conf))

        # Draw box + label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(frame, text, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, detections


def process_image(model: YOLO, source_path: str, output_path: str = "output_annotated.jpg"):
    frame = cv2.imread(source_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {source_path}")

    annotated, detections = detect_vehicles(model, frame)

    occupied_count = len(detections)
    summary = f"Vehicles detected (occupied slots): {occupied_count}"
    cv2.putText(annotated, summary, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imwrite(output_path, annotated)

    print(summary)
    for i, (x1, y1, x2, y2, label, conf) in enumerate(detections, start=1):
        print(f"  {i}. {label} (conf={conf:.2f}) at [{x1},{y1},{x2},{y2}]")
    print(f"Annotated image saved to: {output_path}")

    return occupied_count, detections


def process_video(model: YOLO, source, output_path: str = "output_annotated.mp4"):
    """source can be a video file path or an integer (webcam index)."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, detections = detect_vehicles(model, frame)
        cv2.putText(annotated, f"Occupied: {len(detections)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        writer.write(annotated)
        cv2.imshow("Buddhi - Occupancy Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Annotated video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Buddhi - ParkMate occupancy detector")
    parser.add_argument("--source", required=True,
                         help="Image path, video path, or webcam index (e.g. 0)")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model weights")
    args = parser.parse_args()

    model = load_model(args.model)

    # Decide image vs video/webcam
    if args.source.isdigit():
        process_video(model, int(args.source))
    elif args.source.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        process_video(model, args.source)
    else:
        process_image(model, args.source)


if __name__ == "__main__":
    main()
