import cv2
import numpy as np
from ultralytics import YOLO

class VideoPreprocessor:
    def __init__(self, video_path, output_dir, scene_threshold=30, resize_size=(224, 224)):
        self.video_path = video_path
        self.output_dir = output_dir
        self.scene_threshold = scene_threshold
        self.resize_size = resize_size
        self.model = YOLO('yolov8n.pt')  # YOLOv8 모델 초기화
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_scene_changes(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: 비디오 파일을 열 수 없습니다.")
            return [], []  # 빈 리스트를 반환하여 종료 방지
        ret, prev_frame = cap.read()
        scene_changes = []
        clips = []
        frame_count = 0
        clip_frames = []
        
        while ret:
            ret, frame = cap.read()
            if not ret:
                break

            diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
            mean_diff = np.mean(diff)

            # 장면 전환 감지 (역치 이상)
            if mean_diff > self.scene_threshold:
                # scene_changes.append(frame_count)
                # clips.append(prev_frame)
                if len(clip_frames) > 0:
                    clips.append(clip_frames)  # 클립을 리스트에 추가
                    clip_frames = []  # 새로운 클립을 위해 초기화

            clip_frames.append(frame)
            prev_frame = frame
            frame_count += 1
        
        cap.release()
        return scene_changes, clips

    def detect_faces(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        return faces

    def resize_frame_to_face(self, frame, faces):
        if len(faces) > 0:
            x, y, w, h = faces[0]  # 첫 번째 얼굴을 기준으로 리사이징
            face_roi = frame[y:y+h, x:x+w]
            resized_frame = cv2.resize(face_roi, self.resize_size)
            return resized_frame
        return frame  # 얼굴이 없으면 원본 프레임 반환

    def detect_human_in_frame(self, frame):
        # results = self.model(frame)
        # return any('person' in res['label'] for res in results)
            # YOLO 모델을 통해 예측
        results = self.model(frame)
        
        # boxes 필드가 존재하는지 확인하고 처리
        for result in results:
            if result.boxes:  # boxes가 존재할 경우
                for box in result.boxes:
                    # 클래스 레이블을 확인 (YOLO 클래스 0번이 person일 경우)
                    if int(box.cls[0]) == 0:  # 0이 'person'을 의미하는 클래스 번호라고 가정
                        return True
                    print(f"Detected class: {int(box.cls[0])}")

    def save_clip_as_video(self, frames, output_path, fps=30):
        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 포맷 설정
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        print(f"Clip saved at: {output_path}")

    def process_video(self):
        scene_changes, clips = self.detect_scene_changes()
        clip_count = 0
        
        for clip_frames in clips:
            human_detected = any(self.detect_human_in_frame(frame) for frame in clip_frames)

            if human_detected:
                # 얼굴이 있을 경우 리사이징 처리 후 저장
                resized_frames = [self.resize_frame_to_face(frame, self.detect_faces(frame)) for frame in clip_frames]
                
                # .mp4로 저장
                output_path = f"{self.output_dir}/clip_{clip_count}.mp4"
                self.save_clip_as_video(resized_frames, output_path)
                clip_count += 1


if __name__ == "__main__":


    video_preprocessor = VideoPreprocessor(video_path='output_video.mp4', output_dir='extract_result_2')
    video_preprocessor.process_video()
