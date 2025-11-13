import cv2
import threading
import queue
import time
import numpy as np
from ultralytics import YOLO

# Event to stop all threads safely
stop_event = threading.Event()

# Load YOLO model ONCE (very important)
# yolov8n = fastest model
print("Loading YOLO model...")

try:
    model = YOLO("yolov8n.pt")
    model.to("cuda")   # Use GPU if available
    print("YOLO model loaded on GPU.")
except:
    model = YOLO("yolov8n.pt")
    print("⚠ GPU not available. Running YOLO on CPU.")


def capture_frames(video_source, frame_queue):
    """Capture frames from each camera independently."""
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"❌ Error: Cannot open video source {video_source}")
        stop_event.set()
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        if not frame_queue.full():
            frame_queue.put(frame)

    cap.release()


def yolo_inference(frame):
    """Run YOLO inference and return annotated frame."""
    results = model(frame, verbose=False)
    annotated = results[0].plot()  # Draw boxes + labels
    return annotated


def process_frames(name, frame_queue):
    """Process frames with YOLO + FPS measurement."""
    prev_time = time.time()
    fps = 0
    frame_count = 0
    prev_frame = None

    while not stop_event.is_set():
        if frame_queue.empty():
            continue

        frame = frame_queue.get()

        # Skip empty or duplicate frames
        if frame is None or frame.size == 0:
            continue
        if prev_frame is not None and np.array_equal(frame, prev_frame):
            continue

        prev_frame = frame

        # Run YOLO inference
        frame = yolo_inference(frame)

        # FPS calculation
        frame_count += 1
        if frame_count >= 10:
            curr_time = time.time()
            fps = frame_count / (curr_time - prev_time)
            prev_time = curr_time
            frame_count = 0

        # Display FPS
        cv2.putText(frame, f"{name} | FPS: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    """
    Add all your video streams here.
    0 = laptop webcam
    Add RTSP streams like:
    rtsp://username:password@ip_address:554/stream
    """

    sources = [0]  
    # Example with multiple cameras:
    # sources = [0, "rtsp://192.168.1.10/stream1"]

    threads = []

    # Create thread pairs per camera
    for i, src in enumerate(sources):
        q = queue.Queue(maxsize=10)

        t_capture = threading.Thread(
            target=capture_frames, args=(src, q)
        )
        t_process = threading.Thread(
            target=process_frames, args=(f"Camera {i+1}", q)
        )

        threads.extend([t_capture, t_process])

    # Start all threads
    for t in threads:
        t.start()

    # Wait for threads to close
    for t in threads:
        t.join()

    print("🎉 All streams closed cleanly.")
