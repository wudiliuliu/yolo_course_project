from ultralytics import YOLO
import cv2
import json

# 加载模型
model = YOLO("yolov8n.pt")

# 视频路径
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 输出视频
out = cv2.VideoWriter("output_track.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# -------------------------- 计数线设置 --------------------------
# 线的两个端点，你可以自己改坐标
line_start = (100, h // 2)
line_end = (w - 100, h // 2)

track_history = dict()
cross_state = dict()  # 记录每个id上一帧在线的哪一侧
in_count = 0
out_count = 0
all_tracks_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0])

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # 保存轨迹
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((int(cx), int(cy)))
            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)

            # 跨线判断
            # 线是水平中线，用y坐标判断上下
            now_side = 1 if cy > line_start[1] else -1
            if track_id not in cross_state:
                cross_state[track_id] = now_side
            else:
                old_side = cross_state[track_id]
                if old_side != now_side:
                    # 发生穿越
                    if now_side == 1:
                        out_count += 1
                    else:
                        in_count += 1
                    cross_state[track_id] = now_side

            # 画框+id
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)

    # 绘制轨迹
    for tid, pts in track_history.items():
        for i in range(len(pts)-1):
            cv2.line(frame, pts[i], pts[i+1], (255,0,255), 2)

    # 绘制计数线
    cv2.line(frame, line_start, line_end, (0,0,255), 3)
    cv2.putText(frame, f"In:{in_count}  Out:{out_count}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

    out.write(frame)
    cv2.imshow("track", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 保存轨迹json
all_tracks_data = [{"track_id":k, "points":v} for k,v in track_history.items()]
with open("track_result.json","w",encoding="utf-8") as f:
    json.dump(all_tracks_data, f, ensure_ascii=False, indent=2)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"统计完成：进入 {in_count} 人，离开 {out_count} 人")
