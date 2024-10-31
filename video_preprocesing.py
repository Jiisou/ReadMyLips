import cv2
import numpy as np
# from ultralytics.YOLO import YOLOv8  # YOLOv8 모델 사용 가정
from ultralytics import YOLO

# Load a YOLOv8 model (you can specify a pretrained model or your custom one)
model = YOLO('yolov8n.pt')  # 'yolov8n.pt' is a lightweight version of YOLOv8

def detect_scene_changes(video_path, threshold=30):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    scene_changes = []
    frame_count = 0
    clips = []
    
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 간 차이 계산
        diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 
                           cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
        mean_diff = np.mean(diff)
        
        if mean_diff > threshold:
            scene_changes.append(frame_count)
            clips.append(prev_frame)
        
        prev_frame = frame
        frame_count += 1
        
    cap.release()
    return scene_changes, clips

def detect_human_in_frame(frame, model):
    # results = model(frame)
    # return any('person' in res['label'] for res in results)
    
    # YOLO 모델을 통해 예측
    results = model(frame)
    
    # boxes 필드가 존재하는지 확인하고 처리
    for result in results:
        if result.boxes:  # boxes가 존재할 경우
            for box in result.boxes:
                # 클래스 레이블을 확인 (YOLO 클래스 0번이 person일 경우)
                if int(box.cls[0]) == 0:  # 0이 'person'을 의미하는 클래스 번호라고 가정
                    return True
                print(f"Detected class: {int(box.cls[0])}")
    return False


def process_video(video_path, output_dir):
    scene_changes, clips = detect_scene_changes(video_path)
    
    model = YOLO('yolov8n.pt')  # YOLOv8 모델 가정
    # model = YOLOv8("yolov8_weights.pt")  # YOLOv8 모델 가정
    human_clips = []
    
    for clip in clips:
        if detect_human_in_frame(clip, model):
            human_clips.append(clip)
    
    # 존재하는 클립 저장
    for i, clip in enumerate(human_clips):
        output_path = f"{output_dir}/clip_{i}.mp4"
        cv2.imwrite(output_path, clip)
        print(f"Clip saved: {output_path}")

# process_video('input_video.mp4', 'output_clips')
process_video('/Users/jisu/Desktop/dev/prometheus/ReadMyLips/TalkMedia_talkv_high.mp4.mp4', '/Users/jisu/Desktop/dev/prometheus/ReadMyLips/output_clips')