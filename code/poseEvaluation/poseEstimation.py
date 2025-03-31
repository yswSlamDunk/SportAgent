import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

def extract_frames(video_path, output_folder, fps=10):
    # 비디오 파일 존재 여부 확인
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
    
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
    
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(frame_rate / fps)
    
    current_frame = 0
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if current_frame % frame_interval == 0:
                frame_output_path = f"{output_folder}/frame_{str(frame_count).zfill(5)}.jpg"
                success = cv2.imwrite(frame_output_path, frame)
                if not success:
                    print(f"경고: 프레임 저장 실패 - {frame_output_path}")
                frame_count += 1
            
            current_frame += 1
            
            # 진행 상황 출력 (선택사항)
            if current_frame % 100 == 0:
                progress = (current_frame / total_frames) * 100
                print(f"진행률: {progress:.1f}%")
    
    finally:
        cap.release()
    
    print(f"총 {frame_count}개의 프레임이 추출되었습니다.")
    return frame_count

def extract_pose(frame_path, model_path="model/pose_landmarker_heavy.task"):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        num_poses=1
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    image = cv2.imread(frame_path)
    if image is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {frame_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    
    detection_result = detector.detect(mp_image)
    
    if detection_result.pose_landmarks:
        print(f"감지된 포즈의 수: {len(detection_result.pose_landmarks)}")
        print(f"첫 번째 포즈의 랜드마크 수: {len(detection_result.pose_landmarks[0])}")
    else:
        print("포즈가 감지되지 않았습니다.")
    
    return detection_result

if __name__ == "__main__":
    try:
        # 현재 작업 디렉토리 확인
        current_dir = os.getcwd()
        print(f"현재 작업 디렉토리: {current_dir}")
        
        video_path = "../../data/크로스핏/Video/elbow_high_1.avi"
        output_folder = "../../data/크로스핏/images/elbow_high_1"
        
        # 절대 경로로 변환
        video_path = os.path.abspath(video_path)
        output_folder = os.path.abspath(output_folder)
        
        print(f"비디오 파일 경로: {video_path}")
        print(f"출력 폴더 경로: {output_folder}")
        
        # 비디오 파일 존재 여부 확인
        if not os.path.exists(video_path):
            print(f"오류: 비디오 파일이 존재하지 않습니다: {video_path}")
        else:
            print("비디오 파일 확인됨")
            frame_count = extract_frames(video_path, output_folder)
            print(f"프레임 추출 완료: {frame_count}개의 프레임이 저장되었습니다.")
            
    except Exception as e:
        print(f"오류 발생: {str(e)}")








