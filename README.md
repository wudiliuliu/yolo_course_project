# yolo_course_project YOLO目标检测课程项目


# YOLOv8行人跟踪项目
功能：行人跟踪、轨迹绘制、越线计数、区域入侵告警，过滤非机动车

## 运行前准备
1. 电脑安装 Python 3.10
2. 解压代码压缩包，存放路径**不能有中文**

## 运行步骤
1. 使用终端进入项目文件夹
2. 【可选】创建虚拟环境
python -m venv venv
venv\Scripts\activate
3. 安装依赖
pip install -r requirements.txt
4. 运行程序
python main.py

> 默认调用摄像头实时检测
> 按键盘 q 关闭程序

## 说明
- 项目内自带 yolov8n.pt 模型，无需联网下载
- 无NVIDIA显卡也可以运行，CPU推理速度较慢
- 摄像头打不开：检查摄像头权限，关掉占用摄像头的软件（微信等）

## 常见问题
- 提示模块找不到：重新执行 pip install -r requirements.txt
- 程序闪退：检查文件夹路径有没有中文
