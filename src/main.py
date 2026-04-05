from ultralytics import YOLO
import cv2
import math
import os
from datetime import datetime
import time

# models
model = YOLO("yolov8n.pt")
helmet_model = YOLO("models/helmet.pt")

# output folder
os.makedirs("outputs", exist_ok=True)

# webcam
cap = cv2.VideoCapture(0)

last_helmet_time = 0
last_triple_time = 0
cooldown = 5  

while True:
    ret, frame = cap.read()
    
    start = time.time()

    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)
    helmet_results = helmet_model(frame)

    annotated_frame = frame.copy()

    persons = []
    bikes = []

    # person & bike
    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            name = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if name == "person":
                color = (255, 0, 0)
            elif name == "motorcycle":
                color = (0, 255, 255)
            else:
                continue

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if name == "person":
                persons.append((cx, cy))
            elif name == "motorcycle":
                bikes.append((cx, cy))

    # triple ride
    for bike in bikes:
        rider_count = 0

        for person in persons:
            dist = math.sqrt((bike[0] - person[0])**2 + (bike[1] - person[1])**2)

            if dist < 150:
                rider_count += 1

        if rider_count > 2:
            cv2.putText(annotated_frame, "Triple Riding!",
                        (bike[0], bike[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

            current_time = datetime.now().timestamp()

            if current_time - last_triple_time > cooldown:
                cv2.imwrite(f"outputs/triple_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", annotated_frame)
                last_triple_time = current_time

    # helmet
    for r in helmet_results:
        for box in r.boxes:
            cls = int(box.cls)
            name = helmet_model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            hx = int((x1 + x2) / 2)
            hy = int((y1 + y2) / 2)

            is_rider = False
            for bike in bikes:
                dist = math.sqrt((hx - bike[0])**2 + (hy - bike[1])**2)
                if dist < 150:
                    is_rider = True
                    break

            if is_rider and "without" in name.lower():
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2),
                              (0, 0, 255), 2)

                cv2.putText(annotated_frame, "No Helmet!",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255), 2)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                current_time = datetime.now().timestamp()

                if current_time - last_helmet_time > cooldown:
                    cv2.imwrite(f"outputs/nohelmet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", annotated_frame)
                    last_helmet_time = current_time

    
    end = time.time()
    fps = 1 / (end - start)

    cv2.putText(annotated_frame, f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)
    cv2.putText(annotated_frame, "Traffic Violation Detection System",
            (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2)
    
    cv2.imshow("Real-Time Detection", annotated_frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()