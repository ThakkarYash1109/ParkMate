import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Allow running this file directly from the project root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from buddhi.vehicle_detection import VehicleDetector

BASE_DIR = Path(__file__).resolve().parent
SLOTS_FILE = BASE_DIR / "slots.json"
STATUS_FILE = BASE_DIR / "status.json"


def point_in_polygon(point, polygon):
    polygon_np = np.asarray(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(polygon_np, point, False) >= 0


def load_slots():
    if not SLOTS_FILE.exists():
        return []
    with open(SLOTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_status(slots, detections):
    occupied = sum(1 for s in slots if s["occupied"])
    total = len(slots)
    available = total - occupied
    utilization = round((occupied / total) * 100, 1) if total else 0.0

    data = {
        "timestamp": time.time(),
        "total_slots": total,
        "occupied": occupied,
        "available": available,
        "utilization": utilization,
        "slots": slots,
        "vehicles": detections,
    }

    tmp = STATUS_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, STATUS_FILE)


def draw_results(frame, slots, detections):
    """
    Draw clean parking-slot occupancy results.

    Green  = FREE
    Red    = OCCUPIED

    YOLO bounding boxes are intentionally hidden
    for a cleaner ParkMate presentation.
    """

    # -------------------------------------------------
    # 1. Reset slot status
    # -------------------------------------------------

    for slot in slots:
        slot["occupied"] = False

    # -------------------------------------------------
    # 2. Match detected vehicles with parking slots
    # -------------------------------------------------

    for det in detections:

        center = tuple(det["center"])

        for slot in slots:

            if point_in_polygon(center, slot["points"]):
                slot["occupied"] = True
                break

    # -------------------------------------------------
    # 3. Create transparent overlay
    # -------------------------------------------------

    overlay = frame.copy()

    for slot in slots:

        pts = np.asarray(
            slot["points"],
            dtype=np.int32
        )

        occupied = slot["occupied"]

        # Red = occupied
        # Green = free
        if occupied:
            color = (0, 0, 255)
        else:
            color = (0, 200, 0)

        # Slight transparent fill
        cv2.fillPoly(
            overlay,
            [pts],
            color
        )

    # Blend overlay with original image
    frame = cv2.addWeighted(
        overlay,
        0.18,
        frame,
        0.82,
        0
    )

    # -------------------------------------------------
    # 4. Draw parking slot boundaries + labels
    # -------------------------------------------------

    for slot in slots:

        pts = np.asarray(
            slot["points"],
            dtype=np.int32
        )

        occupied = slot["occupied"]

        if occupied:
            color = (0, 0, 255)
            status = "OCCUPIED"
        else:
            color = (0, 200, 0)
            status = "FREE"

        # Slot outline
        cv2.polylines(
            frame,
            [pts],
            True,
            color,
            3,
            cv2.LINE_AA
        )

        # -------------------------------------------------
        # Find center of parking slot
        # -------------------------------------------------

        moments = cv2.moments(pts)

        if moments["m00"] != 0:

            cx = int(
                moments["m10"] /
                moments["m00"]
            )

            cy = int(
                moments["m01"] /
                moments["m00"]
            )

        else:

            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))

        # -------------------------------------------------
        # Slot label
        # -------------------------------------------------

        label = f'{slot["id"]}'

        # Dark background behind label
        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2
        )

        cv2.rectangle(
            frame,
            (cx - tw // 2 - 5, cy - th - 8),
            (cx + tw // 2 + 5, cy + 5),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            frame,
            label,
            (
                cx - tw // 2,
                cy
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # -------------------------------------------------
    # 5. Calculate parking statistics
    # -------------------------------------------------

    total = len(slots)

    occupied = sum(
        1 for slot in slots
        if slot["occupied"]
    )

    free = total - occupied

    occupancy = (
        (occupied / total) * 100
        if total > 0
        else 0
    )

    # -------------------------------------------------
    # 6. Professional dashboard
    # -------------------------------------------------

    panel_width = 390
    panel_height = 155

    # Semi-transparent dashboard
    dashboard = frame.copy()

    cv2.rectangle(
        dashboard,
        (15, 15),
        (
            panel_width,
            panel_height
        ),
        (15, 15, 15),
        -1
    )

    frame = cv2.addWeighted(
        dashboard,
        0.88,
        frame,
        0.12,
        0
    )

    # -------------------------------------------------
    # ParkMate title
    # -------------------------------------------------

    cv2.putText(
        frame,
        "PARKMATE",
        (35, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "AI PARKING MONITOR",
        (37, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )

    # -------------------------------------------------
    # Statistics
    # -------------------------------------------------

    cv2.putText(
        frame,
        f"TOTAL     {total}",
        (35, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"OCCUPIED  {occupied}",
        (35, 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"FREE      {free}",
        (190, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 200, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"USAGE     {occupancy:.1f}%",
        (190, 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # -------------------------------------------------
    # 7. Small legend
    # -------------------------------------------------

    cv2.circle(
        frame,
        (35, 158),
        6,
        (0, 200, 0),
        -1
    )

    cv2.putText(
        frame,
        "FREE",
        (48, 163),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.circle(
        frame,
        (105, 158),
        6,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        frame,
        "OCCUPIED",
        (118, 163),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return frame
def setup_slots(image_path=None):
    image_path = image_path or str(BASE_DIR / "parking.jpg")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot open parking image: {image_path}")

    original = image.copy()
    points = []
    slots = []
    window = "ParkMate - Slot Setup"

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, mouse_callback)

    print("\nSLOT SETUP")
    print("Click 4 corners of one parking slot clockwise.")
    print("Press N to save that slot. Press R to reset current points.")
    print("Press Q when all slots are completed.\n")

    while True:
        canvas = original.copy()

        for i, slot in enumerate(slots):
            pts = np.asarray(slot["points"], dtype=np.int32)
            cv2.polylines(canvas, [pts], True, (0, 180, 0), 2)
            x, y = pts[0]
            cv2.putText(canvas, slot["id"], (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)

        if points:
            for p in points:
                cv2.circle(canvas, tuple(p), 5, (0, 0, 255), -1)
            if len(points) > 1:
                cv2.polylines(canvas, [np.asarray(points)], False, (255, 180, 0), 2)

        cv2.putText(canvas, f"Current points: {len(points)}/4 | N=save R=reset Q=finish",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            points.clear()
        elif key == ord("n"):
            if len(points) == 4:
                slots.append({
                    "id": f"A{len(slots) + 1}",
                    "points": points.copy(),
                    "occupied": False
                })
                points.clear()
                print(f"Saved slot A{len(slots)}")
            else:
                print("Need exactly 4 points before pressing N.")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

    with open(SLOTS_FILE, "w", encoding="utf-8") as f:
        json.dump(slots, f, indent=2)

    print(f"Saved {len(slots)} slots -> {SLOTS_FILE}")


class ParkingDetector:
    def __init__(self, model_name="yolo11s.pt", confidence=0.15):
        self.vehicle_detector = VehicleDetector(model_name, confidence)
        self.slots = load_slots()

    def process(self, frame):
        detections = self.vehicle_detector.detect(frame)

        # Reset and assign occupancy.
        for slot in self.slots:
            slot["occupied"] = False

        for det in detections:
            center = tuple(det["center"])
            for slot in self.slots:
                if point_in_polygon(center, slot["points"]):
                    slot["occupied"] = True
                    break

        output = draw_results(frame.copy(), self.slots, detections)
        save_status(self.slots, detections)
        return output


def run(source):
    slots = load_slots()
    if not slots:
        print("No parking slots found.")
        print("Run: python buddhi/parking_detection.py --setup")
        return

    detector = ParkingDetector()
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    while True:
        ok, frame = cap.read()
        if not ok:
            # For an image or an ended video, reopen the image once.
            if isinstance(source, str) and Path(source).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                frame = cv2.imread(source)
                if frame is None:
                    break
                output = detector.process(frame)
                cv2.imshow("Buddhi - Parking Detection", output)
                cv2.waitKey(0)
                break
            break

        output = detector.process(frame)
        cv2.imshow("Buddhi - Parking Detection", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--source", default=str(BASE_DIR / "parking.jpg"))
    parser.add_argument("--model", default="yolo11s.pt")
    args = parser.parse_args()

    if args.setup:
        setup_slots(args.source)
    else:
        run(args.source)
