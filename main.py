from ultralytics import YOLO
import cv2
import time
import torch

model = YOLO("models/yolo26n.pt").to("cuda")

cap = cv2.VideoCapture("videos/video2.mp4")

print(torch.cuda.is_available())

WIDTH = 1080
HEIGHT = 720
LINE_Y = 500

entered = 0
exited = 0
positions = {}

last_seen = {}
frame_id = 0
TTL = 15   # x frame görünmezse sil


fps_list = []

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 codec
video_fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "output.mp4",
    fourcc,
    10,
    (WIDTH, HEIGHT)
)

while cap.isOpened():

    start_time = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    results = model.track(
        frame,
        classes=[2],
        tracker="botsort.yaml",
        persist=True,
        verbose=False
    )

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        print(positions)
        for box, track_id in zip(boxes, ids):

            x1, y1, x2, y2 = box
            center_y = int((y1 + y2) / 2)
            track_id = int(track_id)

            if track_id in positions:

                prev_y = positions[track_id]
                last_seen[track_id] = frame_id

                if prev_y < LINE_Y and center_y >= LINE_Y:
                    entered += 1

                elif prev_y > LINE_Y and center_y <= LINE_Y:
                    exited += 1

            positions[track_id] = center_y

            cv2.rectangle(frame,
                          (int(x1), int(y1)),
                          (int(x2), int(y2)),
                          (0,255,0), 2)
            cv2.putText(frame, f"Id : {track_id}", (int(x1), int(y1 -10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # FPS hesapla
    end_time = time.perf_counter()
    fps = 1 / (end_time - start_time)


    # Çizgiler
    cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (255,0,255), 3)

    cv2.putText(frame, f"Car Count: {entered}", (20,40),
                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.putText(frame, f"FPS: {fps}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.putText(frame, f"Running on CPU", (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    out.write(frame)

    cv2.imshow("Vehicle Counter", frame)

    dead_ids = [
        tid for tid, last in last_seen.items()
        if frame_id - last > TTL
    ]

    for tid in dead_ids:
        positions.pop(tid, None)
        last_seen.pop(tid, None)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
