from ultralytics import YOLO
import cv2
import numpy as np
import json
from datetime import datetime

# 加载模型
model = YOLO("yolov8n.pt")

# 轨迹存储字典：key为track_id，value是轨迹点列表
track_history = dict()
# 轨迹最大点数，防止内存溢出
max_track_len = 80
# 输出轨迹json文件
trajectory_save_path = "track_result.json"

# 视频输入，0代表摄像头，也可以写"test.mp4"读取本地视频
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)
# 视频输出
out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_track.mp4", fourcc, fps, (out_w, out_h))

frame_count = 0
# 用于保存全部轨迹数据
all_tracks_data = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # 跟踪推理，bytetrack跟踪器，只检测行人class=0，保持跟踪persist=True
    results = model.track(frame, tracker="bytetrack.yaml", persist=True, classes=[0], conf=0.3)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, tid in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # 记录轨迹
            if tid not in track_history:
                track_history[tid] = []
                all_tracks_data[tid] = []
            track_history[tid].append((center_x, center_y))
            all_tracks_data[tid].append({
                "frame": frame_count,
                "x": float(center_x),
                "y": float(center_y),
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            # 限制轨迹长度
            if len(track_history[tid]) > max_track_len:
                track_history[tid].pop(0)

            # 绘制轨迹线
            pts = np.array(track_history[tid], np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 255), thickness=2)

            # 绘制框和ID
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f"ID:{tid}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 画面展示
    cv2.imshow("YOLO ByteTrack Track", frame)
    out.write(frame)

    # 按q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

# 保存轨迹数据到json
with open(trajectory_save_path, "w", encoding="utf-8") as f:
    json.dump(all_tracks_data, f, ensure_ascii=False, indent=2)
print(f"轨迹数据已经保存至 {trajectory_save_path}")
print("视频输出：output_track.mp4")
