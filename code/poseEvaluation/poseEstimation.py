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
        return True
    
    # pose_estimation 실패 비율이 20% 이상인 경우
    elif len(cant_estimate_index) / (len(pose_landmarks_list) + len(cant_estimate_index)) > 0.2:
        return False
        
    # 연속된 pose_estimation 실패가 전체 프레임의 0.1 이상인 경우
    min_max = find_max_min_of_longest_consecutive(cant_estimate_index)
    len_continue = min_max[1] - min_max[0] + 1
    if len_continue / (len(pose_landmarks_list) + len(cant_estimate_index)) > 0.1:
        return False

    else:
        return True