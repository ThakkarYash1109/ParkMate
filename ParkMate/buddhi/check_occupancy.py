"""
BUDDHI - Slot Occupancy Checker
----------------------------------
Works with predefined slot rectangles (from select_slots.py) and decides
occupied vs empty per slot. This approach is camera-angle independent
(works for top-down/drone shots, angled CCTV, or the miniature model camera)
because it doesn't depend on YOLO recognizing "car" shapes - it just measures
how much "stuff" (edges/texture) is inside each marked rectangle compared to
bare, empty pavement.

Optional: pass --model to additionally run YOLO vehicle detection and
cross-check/label which vehicle class occupies each slot.

Usage:
    python check_occupancy.py --source parking.jpg --slots slots.json
    python check_occupancy.py --source parking.jpg --slots slots.json --model yolov8n.pt
"""

import argparse
import json
import cv2
import numpy as np

# Tune this after testing on your own empty vs occupied slots.
# Higher = only strong texture/edges counted as "occupied".
EDGE_DENSITY_THRESHOLD = 0.01


def load_slots(path):
    with open(path) as f:
        data = json.load(f)
    return [(d["id"], tuple(d["box"])) for d in data]


def slot_edge_density(gray_frame, box):
    x1, y1, x2, y2 = box
    roi = gray_frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    edges = cv2.Canny(roi, 50, 150)
    return float(np.count_nonzero(edges)) / edges.size


def check_frame(frame, slots, model=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Optional YOLO detections, used only to confirm/label vehicle type
    yolo_boxes = []
    if model is not None:
        results = model.predict(frame, conf=0.15, verbose=False)[0]
        vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in vehicle_classes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                yolo_boxes.append((bx1, by1, bx2, by2, vehicle_classes[cls_id]))

    occupied = 0
    for slot_id, box in slots:
        density = slot_edge_density(gray, box)
        is_occupied = density > EDGE_DENSITY_THRESHOLD

        # if YOLO available, confirm using IoU overlap with any vehicle box
        label = ""
        if model is not None:
            for (bx1, by1, bx2, by2, vclass) in yolo_boxes:
                if boxes_overlap(box, (bx1, by1, bx2, by2)):
                    is_occupied = True
                    label = vclass
                    break

        x1, y1, x2, y2 = box
        color = (0, 0, 255) if is_occupied else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        text = f"{slot_id}:{'OCC' if is_occupied else 'FREE'} {label}"
        cv2.putText(frame, text, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

        if is_occupied:
            occupied += 1

    summary = f"Occupied: {occupied}/{len(slots)}  Free: {len(slots) - occupied}"
    cv2.putText(frame, summary, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)
    return frame, occupied


def boxes_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    inter = (ix2 - ix1) * (iy2 - iy1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    return inter / a_area > 0.2  # >20% of slot covered by a detected vehicle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Image path")
    parser.add_argument("--slots", required=True, help="slots.json from select_slots.py")
    parser.add_argument("--model", default=None, help="Optional YOLO weights, e.g. yolov8n.pt")
    parser.add_argument("--out", default="occupancy_result.jpg")
    args = parser.parse_args()

    frame = cv2.imread(args.source)
    if frame is None:
        raise FileNotFoundError(args.source)

    slot_list = load_slots(args.slots)

    model = None
    if args.model:
        from ultralytics import YOLO
        model = YOLO(args.model)

    annotated, occupied = check_frame(frame, slot_list, model)
    cv2.imwrite(args.out, annotated)
    print(f"Occupied: {occupied}/{len(slot_list)}")
    print(f"Result saved to {args.out}")


if __name__ == "__main__":
    main()
