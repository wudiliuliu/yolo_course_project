from sympy.codegen.ast import continue_
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolo11n.pt")
track_history = {}
track_max_len = 60

# ===== 1. 越线计数的横线（沿用之前）
line_y = 320
in_count = 0
out_count = 0
# 记录每个ID上一次在线的位置，防止重复计数
id_last_side = dict()

# ===== 🆕新增：多边形警戒区域坐标，你可以自己改点
alert_poly = np.array([
    [200, 120],
    [480, 120],
    [520, 380],
    [160, 380]
], dtype=np.int32)

cap = cv2.VideoCapture("test.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    if results[0].boxes.id is None:
        # 🆕新增：没有目标的时候也要绘制警戒区
        cv2.polylines(frame, [alert_poly], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.putText(frame, "ALERT ZONE", (alert_poly[0][0], alert_poly[0][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2,cv2.LINE_AA)
        cv2.imshow("YOLO Track", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    boxes = results[0].boxes.xyxy.cpu().numpy()
    track_ids = results[0].boxes.id.cpu().numpy().astype(int)

    for i,(box, track_id) in enumerate(zip(boxes, track_ids)):
        cls = int(results[0].boxes.cls[i].cpu().numpy())
        if cls != 0:
            continue
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # 轨迹维护
        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append((cx, cy))
        if len(track_history[track_id]) > track_max_len:
            track_history[track_id].pop(0)

        # 绘制轨迹
        pts = np.array(track_history[track_id], dtype=np.int32)
        cv2.polylines(frame, [pts], False, (255, 0, 255), 2)

        # -------- 越线计数逻辑 --------
        current_side = 1 if cy < line_y else -1
        if track_id in id_last_side:
            prev_side = id_last_side[track_id]
            if prev_side != current_side:
                if current_side == -1:
                    in_count += 1
                else:
                    out_count += 1
        id_last_side[track_id] = current_side

        # -------- 🆕新增：入侵检测判断 --------
        in_alert = cv2.pointPolygonTest(alert_poly, (cx, cy), False)
        if in_alert >= 0:
            box_color = (0, 0, 255)
            tip_text = "INTRUSION!"
        else:
            box_color = (0, 255, 0)
            tip_text = ""

        # 画框+ID（这里颜色变量box_color是配合上面新增逻辑改动的）
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
        cv2.putText(frame, f"ID:{track_id} {tip_text}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)

    # 绘制计数分界线
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
    cv2.putText(frame, f"In:{in_count} | Out:{out_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)

    # 🆕新增：画面上绘制警戒多边形和文字
    cv2.polylines(frame, [alert_poly], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.putText(frame, "ALERT ZONE", (alert_poly[0][0], alert_poly[0][1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2,cv2.LINE_AA)

    cv2.imshow("YOLO Track", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
