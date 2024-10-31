import cv2
import numpy as np
from ultralytics import YOLO
import ffmpeg

class VideoPreprocessor:
    def __init__(self, video_path, output_dir, scene_threshold=30, resize_size=(224, 224), fps=30):
        self.video_path = video_path
        self.output_dir = output_dir
        self.scene_threshold = scene_threshold
        self.resize_size = resize_size
        self.model = YOLO('yolov8n.pt')  # YOLOv8 모델 초기화
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.fps = fps  # 비디오의 프레임 속도 설정

    def detect_scene_changes(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: 비디오 파일을 열 수 없습니다.")
            return [], []  # 빈 리스트 반환
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

            # 장면 전환 감지 및 간격 확인
            if mean_diff > self.scene_threshold:
                if len(clip_frames) >= self.fps * 2:  # 2초 이상 간격인 경우만
                    clips.append(clip_frames)
                    scene_changes.append((frame_count - len(clip_frames), frame_count))  # 시작 및 종료 프레임 기록

                clip_frames = []  # 새로운 클립 시작
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
            x, y, w, h = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            resized_frame = cv2.resize(face_roi, self.resize_size)
            return resized_frame
        return None  # 얼굴이 없으면 None 반환

    def detect_human_in_frame(self, frame):
        results = self.model(frame)
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    if int(box.cls[0]) == 0:
                        return True
        return False

    # 자른 구간에 대한 오디오만 나와야 되는데 전체 오디오가 나옴
    # def save_clip_as_video_with_audio(self, frames, output_path):
    #     temp_video_path = output_path.replace(".mp4", "_temp.mp4")
    #     height, width, _ = frames[0].shape
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     out = cv2.VideoWriter(temp_video_path, fourcc, self.fps, (width, height))
        
    #     for frame in frames:
    #         out.write(frame)
        
    #     out.release()

    #     # ffmpeg-python으로 오디오와 함께 최종 비디오 파일 생성
    #     video_input = ffmpeg.input(temp_video_path)  # 비디오 입력
    #     audio_input = ffmpeg.input(self.video_path)  # 오디오 입력
        
    #     (
    #         ffmpeg
    #         .output(video_input, audio_input, output_path, vcodec='copy', acodec='aac', **{'map': '0:v', 'map': '1:a'})  # 비디오와 오디오 병합
    #         .run(overwrite_output=True)
    #     )
        
    #     print(f"Clip with audio saved at: {output_path}")


    def save_clip_as_video_with_audio(self, frames, output_path, start_time, end_time):
        temp_video_path = output_path.replace(".mp4", "_temp.mp4")
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, self.fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()

        # ffmpeg로 지정된 구간의 오디오와 함께 최종 비디오 파일 생성
        video_stream = ffmpeg.input(temp_video_path)
        audio_stream = ffmpeg.input(self.video_path, ss=start_time, to=end_time).audio  # 오디오 스트림만 추출

        # 비디오와 오디오 스트림을 결합하여 최종 출력
        ffmpeg.output(video_stream, audio_stream, output_path, vcodec='copy', acodec='aac').run(overwrite_output=True)
        
        print(f"Clip with audio saved at: {output_path}")

    
    def process_video(self):
        scene_changes, clips = self.detect_scene_changes()
        clip_count = 0
        
        for (start_frame, end_frame), clip_frames in zip(scene_changes, clips):

            start_time = start_frame / self.fps
            end_time = end_frame / self.fps

            # 사람 얼굴이 있는 프레임이 있는지 확인
            face_detected = any(len(self.detect_faces(frame)) >0 for frame in clip_frames if frame is not None)
            human_detected = any(self.detect_human_in_frame(frame) for frame in clip_frames)

            # # face_detected = any(self.detect_faces(frame) for frame in clip_frames if frame is not None)
            # face_detected = any(len(self.detect_faces(frame)) > 0 for frame in clip_frames if frame is not None)
            # human_detected = any(self.detect_human_in_frame(frame) for frame in clip_frames)

            if human_detected and face_detected:
                # 얼굴이 감지된 프레임만 리사이즈하여 저장
                resized_frames = [self.resize_frame_to_face(frame, self.detect_faces(frame)) for frame in clip_frames]
                resized_frames = [frame for frame in resized_frames if frame is not None]  # None 제외

                # 최종 .mp4로 저장
                output_path = f"{self.output_dir}/clip_{clip_count}.mp4"
                self.save_clip_as_video_with_audio(resized_frames, output_path, start_time, end_time)
                clip_count += 1


if __name__ == "__main__":
    video_preprocessor = VideoPreprocessor(video_path='TalkMedia_talkv_high.mp4.mp4', output_dir='extracted_new')
    video_preprocessor.process_video()
