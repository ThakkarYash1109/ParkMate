import cv2
import numpy as np
from ultralytics import YOLO


# COCO vehicle classes
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


class VehicleDetector:

    def __init__(
        self,
        model_name="yolo11s.pt",
        confidence=0.10,
        tile_size=640,
        overlap=0.20
    ):

        print(f"Loading YOLO model: {model_name}")

        self.model = YOLO(model_name)
        self.confidence = confidence
        self.tile_size = tile_size
        self.overlap = overlap

        print("YOLO model loaded successfully.")

    def detect_tile(self, tile):

        results = self.model.predict(
            source=tile,
            conf=self.confidence,
            imgsz=640,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False
        )

        result = results[0]

        detections = []

        if result.boxes is None:
            return detections

        for box in result.boxes:

            cls_id = int(box.cls[0].item())

            if cls_id not in VEHICLE_CLASSES:
                continue

            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            detections.append({
                "class_id": cls_id,
                "class_name": VEHICLE_CLASSES[cls_id],
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            })

        return detections

    def detect(self, frame):

        height, width = frame.shape[:2]

        step = int(self.tile_size * (1 - self.overlap))

        all_detections = []

        print(
            f"Running tiled detection on "
            f"{width}x{height} image..."
        )

        # ----------------------------------------
        # Divide image into overlapping tiles
        # ----------------------------------------

        for y in range(0, height, step):

            for x in range(0, width, step):

                x2 = min(x + self.tile_size, width)
                y2 = min(y + self.tile_size, height)

                tile = frame[y:y2, x:x2]

                if tile.size == 0:
                    continue

                tile_detections = self.detect_tile(tile)

                # Convert tile coordinates
                # back to original image coordinates
                for det in tile_detections:

                    bx1, by1, bx2, by2 = det["bbox"]

                    bx1 += x
                    bx2 += x
                    by1 += y
                    by2 += y

                    all_detections.append({
                        "class_id": det["class_id"],
                        "class_name": det["class_name"],
                        "confidence": round(
                            det["confidence"], 3
                        ),
                        "bbox": [
                            bx1,
                            by1,
                            bx2,
                            by2
                        ]
                    })

        # ----------------------------------------
        # Remove duplicate detections
        # caused by overlapping tiles
        # ----------------------------------------

        final_detections = self.remove_duplicates(
            all_detections
        )

        # ----------------------------------------
        # Calculate centers
        # ----------------------------------------

        for det in final_detections:

            x1, y1, x2, y2 = det["bbox"]

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            det["center"] = [
                center_x,
                center_y
            ]

        print(
            f"Vehicles detected: "
            f"{len(final_detections)}"
        )

        for det in final_detections:

            print(
                f"  {det['class_name']} "
                f"confidence={det['confidence']} "
                f"center={det['center']}"
            )

        return final_detections

    def remove_duplicates(self, detections):

        if not detections:
            return []

        boxes = []
        scores = []

        for det in detections:

            x1, y1, x2, y2 = det["bbox"]

            boxes.append([
                x1,
                y1,
                x2 - x1,
                y2 - y1
            ])

            scores.append(
                float(det["confidence"])
            )

        # OpenCV Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.confidence,
            0.40
        )

        if len(indices) == 0:
            return []

        indices = np.array(indices).flatten()

        final = [
            detections[int(i)]
            for i in indices
        ]

        return final
    
    if __name__ == "__main__":
    print("Vehicle detection module loaded successfully!")
    detector = VehicleDetector()
    print("Detector initialized successfully!")