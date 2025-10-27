import os
import cv2
import mediapipe as mp

def extract_frames(video_path, fps=10): # 비디오 프레임 추출
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate / fps)

    current_frame = 0
    frames_list = []
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if current_frame % frame_interval == 0:
            frames_list.append(frame)

        current_frame += 1

    cap.release()

    return frames_list

def pose_estimation(frame_list):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    pose_landmarks_list = []
    cant_estimate_index = []

    for i, frame in enumerate(frame_list):
        results = pose.process(frame)
        if results.pose_landmarks:
            pose_landmarks_list.append(results.pose_landmarks)
        else:
            cant_estimate_index.append(i)
    
    return pose_landmarks_list, cant_estimate_index

def find_max_min_of_longest_consecutive(nums):
    nums = sorted(nums)
    longest_streak = 0
    current_streak = 1
    max_min = (nums[0], nums[0])

    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] + 1:
            current_streak += 1
        else:
            if current_streak > longest_streak:
                longest_streak = current_streak
                max_min = (nums[i - current_streak], nums[i - 1])
            current_streak = 1

    if current_streak > longest_streak:
        max_min = (nums[-current_streak], nums[-1])

    return max_min

def classify_error(pose_landmarks_list, cant_estimate_index):
    if len(cant_estimate_index) == 0:
        return {"success": True, "message": "자세 추출 완료"}
    
    # pose_estimation 실패 비율이 20% 이상인 경우
    elif len(cant_estimate_index) / (len(pose_landmarks_list) + len(cant_estimate_index)) > 0.2:
        return {"success": False, "message": "자세 추출 실패. 실패 비율이 20% 이상입니다. 재촬영 필요.", "error_reason": "overall_failure_rate_over_20%"}
        
    # 연속된 pose_estimation 실패가 전체 프레임의 0.1 이상인 경우
    min_max = find_max_min_of_longest_consecutive(cant_estimate_index)
    len_continue = min_max[1] - min_max[0] + 1
    if len_continue / (len(pose_landmarks_list) + len(cant_estimate_index)) > 0.1:
        return {"success": False, "message": "자세 추출 실패. 연속된 추출 실패가 전체 프레임의 0.1 이상입니다. 재촬영 필요.", "error_reason": "consecutive_failure_rate_over_10%"}

    else:
        return {"success": True, "message": "자세 추출 완료"}


def save_frames_as_images(frames_list, cant_estimate_index, video_path, user_type='user'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..", "..")
    
    video_name = os.path.basename(video_path).split(".")[0]
    frames_dir = os.path.join(project_root, "media", "frames", user_type, video_name)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(os.path.join(frames_dir, "success"), exist_ok=True)
    os.makedirs(os.path.join(frames_dir, "failure"), exist_ok=True)

    for i, frame in enumerate(frames_list):
        if i in cant_estimate_index:
            frame_path = os.path.join(frames_dir, "failure", f"{i}.jpg")
            cv2.imwrite(frame_path, frame)
        else:
            frame_path = os.path.join(frames_dir, "success", f"{i}.jpg")
            cv2.imwrite(frame_path, frame)

    # 경로 구분자를 통일 (/ 사용)
    relative_frames_dir = os.path.join("media", "frames", user_type, video_name).replace("\\", "/")
    return {"success": True, "message": "프레임 저장 완료.", "frames_dir": relative_frames_dir}

