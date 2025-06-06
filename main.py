import time
import json
import threading
from datetime import datetime
import cv2
from ultralytics import YOLO

with open("./info.json", "r") as f:
    info = json.load(f)

RTSP_LINK = info['RTSP_LINK']
MODEL_NAME = info['MODEL_NAME']
DEVICE = info['DEVICE']
INTERVAL = info['INTERVAL']

IMG = None

def rtsp_buf_flush(rtsp_stream: str, num: int, stop_event: threading.Event):
    global IMG
    try:
        print(f"{num} Thread Start")
        while not stop_event.is_set():
            _, IMG = rtsp_stream.read()
    except Exception as e:
        print(f" Error : {e}")
    finally:
        rtsp_stream.release()

cam_stream = cv2.VideoCapture(RTSP_LINK)

stop_event = threading.Event()
thread_0 = threading.Thread(target=rtsp_buf_flush, args=(cam_stream, 0, stop_event,))
thread_0.start()

# Load the YOLO11 model
model = YOLO(MODEL_NAME, verbose=False).to(DEVICE)

# Loop through the video frames
try:
    while not stop_event.is_set():
        if IMG is None:
            time.sleep(0.01)
            continue
        
        result = model.predict(IMG, verbose=False)[0]
        h, w, c = IMG.shape

        annotated_frame = result.plot()
        result_dict_list = json.loads(result.to_json())
        cv2.imshow("Inference Result", annotated_frame)
        time.sleep(INTERVAL)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("KeyboardInterrupt received. Stopping gracefully...")

finally:
    stop_event.set()
    thread_0.join()
    cv2.destroyAllWindows()
