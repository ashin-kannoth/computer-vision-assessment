import cv2
import threading
import queue
import time
import random
import numpy as np

# Shared queue between threads
frame_queue = queue.Queue(maxsize=10)
stop_event = threading.Event()


def capture_frames(video_source=0):
    """Continuously capture frames from webcam or RTSP stream"""
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Cannot open video source")
        stop_event.set()
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue  # skip if frame not read
        if not frame_queue.full():
            frame_queue.put(frame)

    cap.release()


def fake_ai_inference(frame):
    """Simulated AI model inference (adds delay + fake detections)"""
    time.sleep(0.03)  # simulate model processing delay

    # Draw random rectangles (fake detections)
    h, w, _ = frame.shape
    for _ in range(random.randint(2, 4)):
        x1, y1 = random.randint(0, w // 2), random.randint(0, h // 2)
        x2, y2 = x1 + random.randint(50, 150), y1 + random.randint(50, 150)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame


def process_frames():
    """Processes frames from queue, simulating AI inference and calculating FPS"""
    prev_time = time.time()
    fps = 0
    frame_count = 0
    prev_frame = None

    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Skip empty or duplicate frames to improve performance
            if frame is None or frame.size == 0:
                continue
            if prev_frame is not None and np.array_equal(frame, prev_frame):
                continue
            prev_frame = frame

            # Simulated AI inference step
            frame = fake_ai_inference(frame)

            # FPS calculation
            frame_count += 1
            if frame_count >= 10:
                curr_time = time.time()
                fps = frame_count / (curr_time - prev_time)
                prev_time = curr_time
                frame_count = 0

            # Display FPS on frame
            cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Video Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Start threads
    t1 = threading.Thread(target=capture_frames)
    t2 = threading.Thread(target=process_frames)

    t1.start()
    t2.start()

    # Wait for both threads to complete
    t1.join()
    t2.join()

    print("Program closed cleanly.")
