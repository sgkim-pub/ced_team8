import cv2
from ultralytics import YOLO

# 1. USB 카메라 설정
# v4l2-ctl 정보 기반: /dev/video0 사용 및 MJPG 코덱 설정 (MJPG 사용 시 30/60/90 FPS 전송 가능)
device_id = 0
width = 1280
height = 720
fps = 30

cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2) # CAPture via V4L2

if not cap.isOpened():
    print(f"USB 카메라(/dev/video{device_id})를 열 수 없습니다. 케이블 연결을 확인하세요.")
    exit()

# MJPG 포맷 지정 (MJPG를 지정해야 60/90 FPS 고속 캡처 가능)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))   # FOURCC: Four-Character Code
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps)

# 실제 설정된 카메라 해상도 및 프레임 레이트 확인
actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"USB 카메라 연결 성공: {int(actual_w)}x{int(actual_h)} @ {actual_fps} FPS")

# 2. YOLO 모델 로드 (가장 가벼운 nano 모델 또는 TensorRT .engine)
# model = YOLO("yolov8s.engine")  # 또는 yolov8n.pt / yolo11n.pt
# model = YOLO("yolo11n.pt")
# model = YOLO("yolo11n_polyhedra.pt")
# model = YOLO("yolo11n_polyhedra2.pt")
# model = YOLO("yolo11n_polyhedra3.pt")
# model = YOLO("yolo11n_polyhedra4.pt")
model = YOLO("yolo11n_statue.pt")

print("YOLO 실시간 인식 시작... (종료: OpenCV 창 선택 후 'q' 키 또는 터미널에서 Ctrl+C)")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라에서 프레임을 읽어올 수 없습니다.")
            break

        # 3. YOLO 추론 프로세스 진행 (verbose=False: 매 프레임 터미널 출력 방지)
        results = model(frame, stream=True, device='cuda', verbose=False)
        # Attribute, Type, Shape, description
        # result.boxes, Boxes, (N), Detection boxes.
        # result.boxes.data, torch.float32, (N,6|7), Raw [x1,y1,x2,y2,conf,cls] plus optional track ID.
        # result.boxes.xyxy, torch.float32, (N,4), xyxy pixel boxes.
        # result.boxes.conf, torch.float32, (N,), Confidence scores.
        # result.boxes.cls, torch.float32, (N,), Class IDs; cast to int for names.

        # 4. 인식된 결과를 화면에 그리기
        annotated_frame = frame
        for r in results:
            annotated_frame = r.plot()
            print(f'r.boxes.cls.int(): {r.boxes.cls.int()}')
            # r.boxes.cls.int(): tensor([ 0,  0, 56, 60], device='cuda:0', dtype=torch.int32)

            items = [cls.item() for cls in r.boxes.cls.int()]
            print(f'items: {items}')
            # items: [0, 0, 56, 60]

            names = [r.names[cls.item()] for cls in r.boxes.cls.int()]
            print(f'names: {names}')
            # names: ['person', 'person', 'chair', 'dining table']

            all_names = r.names
            print(f'all_names: {all_names}')
            # all_names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', \
            # 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', \
            # 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', \
            # 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', \
            # 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', \
            # 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', \
            # 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', \
            # 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', \
            # 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', \
            # 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', \
            # 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', \
            # 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', \
            # 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', \
            # 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', \
            # 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

        # 화면 표시
        cv2.imshow("Jetson Orin Nano - USB Camera YOLO", annotated_frame)

        # OpenCV 팝업 창이 활성화된 상태에서 'q' 키 또는 ESC(27) 키를 누르면 종료
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("사용자 요청으로 프로그램을 종료합니다.")
            break
except KeyboardInterrupt:
    print("\n터미널 인터럽트(Ctrl+C)로 프로그램을 종료합니다.")

cap.release()
cv2.destroyAllWindows()

