import cv2
from ultralytics import YOLO


model = YOLO("yolov8m.pt")

video_path = "cars.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video not loading")
    exit()

car_count = 0
tracked_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

   
    results = model.track(frame, persist=True, verbose=False)[0]

    if results.boxes is None:
        continue

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

       
        if label == "car" and box.conf[0] > 0.5:
            track_id = int(box.id[0])

       
            if track_id not in tracked_ids:
                tracked_ids.add(track_id)
                car_count += 1

         
            x1, y1, x2, y2 = map(int, box.xyxy[0])

          
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

           
            anchor_x = (x1 + x2) // 2
            anchor_y = y2

            
            cv2.circle(frame, (anchor_x, anchor_y), 5, (0, 0, 255), -1)

      
            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

    
    cv2.putText(
        frame,
        f"Cars Counted: {car_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow("Car Detection & Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Total number of cars detected: {car_count}")
