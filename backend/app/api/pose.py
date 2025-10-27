import os
import pickle
import json
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from ..database.connection import db
from ..analyze.poseEstimation import extract_frames, pose_estimation, classify_error, save_frames_as_images
from ..analyze.poseScoring import analyze_pose

from fastapi import APIRouter, HTTPException, Depends

router = APIRouter(prefix="/api/pose", tags=["pose"])

@router.get("/estimation/")
async def pose_estimation_api(video_id: int):
    try:
        user_video = db.execute_query(
            """SELECT * FROM videos WHERE id = %s""",
            (video_id,)
        )
        user_video_path = user_video[0]['video_path']
        
        # 경로 구분자 통일 (Windows \ -> /)
        user_video_path = user_video_path.replace("\\", "/")
        
        # 상대경로를 절대경로로 변환
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, "..", "..", "..")
        if not os.path.isabs(user_video_path):
            user_video_path = os.path.normpath(os.path.join(project_root, user_video_path))

        frames = extract_frames(user_video_path)
        pose_landmarks_list, cant_estimate_index = pose_estimation(frames)
        save_frames_result = save_frames_as_images(frames, cant_estimate_index, user_video_path)
        is_error = classify_error(pose_landmarks_list, cant_estimate_index)

        pose_data_dir = os.path.join(project_root, "media", "pose_data", "user")
        
        os.makedirs(pose_data_dir, exist_ok=True)

        original_filename = os.path.basename(user_video_path)
        filename = os.path.splitext(original_filename)[0] + '.pkl'
        pose_data_path = os.path.join(pose_data_dir, filename)

        with open(pose_data_path, "wb") as f:
            pickle.dump([pose_landmarks_list, cant_estimate_index], f)
        
        # pose_data_path를 상대경로로 저장 (경로 구분자 통일)
        relative_pose_data_path = os.path.join("media", "pose_data", "user", filename).replace("\\", "/")

        db.execute_query(
            """INSERT INTO pose_estimations (video_id, pose_data_path, is_valid, error_reason) 
               VALUES (%s, %s, %s, %s)""",
            (video_id, relative_pose_data_path, is_error['success'], is_error.get('error_reason', None))
        )
        return {"success": is_error['success'], 
                "message": is_error['message'], 
                "frames_dir": save_frames_result['frames_dir'], }

    except Exception as e:
        return {"success": False, "message": "자세 추출 실패 ${error}"}

@router.get("/scoring/")
async def pose_scoring(standard_video_id: int, user_video_id: int, user_id: int, sport_id: int):
    try:
        
        standard_video = db.execute_query(
            """SELECT * FROM pose_estimations WHERE video_id = %s""",
            (standard_video_id,)
        )
        
        user_video = db.execute_query(
            """SELECT * FROM pose_estimations WHERE video_id = %s""",
            (user_video_id,)
        )
        
        standard_pose_path = standard_video[0]['pose_data_path']
        user_pose_path = user_video[0]['pose_data_path']

        # 경로 구분자 통일 (Windows \ -> /)
        standard_pose_path = standard_pose_path.replace("\\", "/")
        user_pose_path = user_pose_path.replace("\\", "/")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, "..", "..", "..")

        if not os.path.isabs(standard_pose_path):
            standard_pose_path = os.path.normpath(os.path.join(project_root, standard_pose_path))
        if not os.path.isabs(user_pose_path):
            user_pose_path = os.path.normpath(os.path.join(project_root, user_pose_path))
            
        standard_pose_data = pickle.load(open(standard_pose_path, 'rb'))
        user_pose_data = pickle.load(open(user_pose_path, 'rb'))
        
        standard_pose = standard_pose_data[0]  
        user_pose = user_pose_data[0]  
        
        standard_scores = db.execute_query(
            """SELECT body_part, body_part_korean, standard_score FROM sport_standard_scores WHERE video_id = %s""",
            (standard_video_id,)
        )
        
        standard_dict = {row['body_part']: row['standard_score'] for row in standard_scores}       
        result_sum, result, score, part_list = analyze_pose(standard_pose, user_pose, standard_dict, standard_scores)
        
        session_id = save_pose_analysis_to_db(result_sum, result, score, standard_dict, standard_scores, user_id, sport_id, user_video_id, standard_video_id)


        return {"success": True, 
                "message": "자세 분석 완료",

                "session_id": session_id,
                "standard_video_id": standard_video_id,
                "user_video_id": user_video_id,
                "user_id": user_id,
                "sport_id": sport_id,
                
                "result_sum": result_sum.tolist(),  # numpy array를 리스트로 변환
                "result": {
                    "shortestPath": result['shortestPath'],
                    "shortestScoring": result['shortestScoring']
                },  # JSON 직렬화 가능한 부분만 반환
                "score": score.to_dict('records'),  # DataFrame을 딕셔너리 리스트로 변환
                "part_list": part_list, }
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {"success": False, 
                "message": f"자세 분석 실패 ${e}", 
                "error_details": error_details}

@router.get("/sessions")
async def get_pose_sessions(sport_id: int, user_id: int):
    """특정 스포츠와 사용자의 분석 세션 목록 조회"""
    try:
        sessions = db.execute_query(
            """SELECT id as session_id, created_at, user_video_id, standard_video_id 
               FROM pose_evaluation_sessions 
               WHERE user_id = %s AND sport_id = %s 
               ORDER BY created_at DESC""",
            (user_id, sport_id)
        )
        
        return sessions
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {"success": False, "message": f"세션 목록 조회 실패: {str(e)}", "error_details": error_details}

@router.get("/session-results/{session_id}")
async def get_session_results(session_id: int):
    """특정 세션의 분석 결과 조회"""
    try:
        # 전체 평균 점수
        overall_score_query = db.execute_query(
            """SELECT AVG(average_score) as overall_score FROM pose_scores WHERE session_id = %s""",
            (session_id,)
        )
        overall_score = overall_score_query[0]['overall_score'] if overall_score_query else None
        
        # 신체 부위별 점수 (한글명 포함)
        body_part_scores = db.execute_query(
            """SELECT ps.body_part, ps.average_score, ps.is_below_standard, ps.standard_score,
                      COALESCE(ss.body_part_korean, ps.body_part) as body_part_korean
               FROM pose_scores ps
               LEFT JOIN sport_standard_scores ss ON ss.body_part = ps.body_part 
                   AND ss.video_id = (SELECT standard_video_id FROM pose_evaluation_sessions WHERE id = %s)
               WHERE ps.session_id = %s""",
            (session_id, session_id)
        )
        
        # 개선이 필요한 부위
        improvement_areas = db.execute_query(
            """SELECT DISTINCT COALESCE(ss.body_part_korean, ps.body_part) as body_part_korean
               FROM pose_scores ps
               LEFT JOIN sport_standard_scores ss ON ss.body_part = ps.body_part 
                   AND ss.video_id = (SELECT standard_video_id FROM pose_evaluation_sessions WHERE id = %s)
               WHERE ps.session_id = %s AND ps.is_below_standard = 1""",
            (session_id, session_id)
        )
        
        return {
            "success": True,
            "overall_score": float(overall_score) if overall_score else None,
            "body_part_scores": [
                {
                    "body_part": score['body_part'],
                    "body_part_korean": score['body_part_korean'],
                    "average_score": float(score['average_score']),
                    "is_below_standard": bool(score['is_below_standard']),
                    "standard_score": float(score['standard_score']) if score['standard_score'] else None
                }
                for score in body_part_scores
            ],
            "improvement_areas": [area['body_part_korean'] for area in improvement_areas]
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {"success": False, "message": f"세션 결과 조회 실패: {str(e)}", "error_details": error_details}


def save_pose_analysis_to_db(result_sum, result, score_df, standard_dict, standard_scores, user_id, sport_id, user_video_id, standard_video_id):
    """analyze_pose의 모든 결과를 DB에 저장"""
    
    db.execute_update(
        """INSERT INTO pose_evaluation_sessions (user_id, sport_id, user_video_id, standard_video_id) 
           VALUES (%s, %s, %s, %s)""",
        (user_id, sport_id, user_video_id, standard_video_id)
    )
    
    session_id_result = db.execute_query("SELECT LAST_INSERT_ID()")
    session_id = session_id_result[0]['LAST_INSERT_ID()']
    
    save_pose_matrices_to_db(session_id, result_sum)
    save_pose_scores_to_db(session_id, score_df, standard_dict, standard_scores)
    save_pose_evaluation_results_to_db(session_id, result)
    
    return session_id


def save_pose_matrices_to_db(session_id, result_sum):
    """result_sum 매트릭스를 pose_evaluation_matrices 테이블에 저장"""
    db.execute_update(
        """INSERT INTO pose_evaluation_matrices (session_id, matrix_data) 
           VALUES (%s, %s)""",
        (session_id, json.dumps(result_sum.tolist()))
    )
    
def save_pose_scores_to_db(session_id, score_df, standard_dict, standard_scores):
    """score_df를 pose_scores 테이블에 저장"""
    for index, row in score_df.iterrows():
        body_part = row.get('부위', row.get('body_part', 'unknown'))
        average_score = row.get('평균점수', row.get('score', 0))
        
        # standard_dict에 존재하는 부위만 저장
        if body_part in standard_dict:
            standard_score = standard_dict[body_part]
            is_below_standard = 1 if average_score < standard_score else 0
            
            db.execute_update(
                """INSERT INTO pose_scores (session_id, body_part, average_score, is_below_standard) 
                   VALUES (%s, %s, %s, %s)""",
                (session_id, body_part, float(average_score), is_below_standard)
            )
        else:
            continue

def save_pose_evaluation_results_to_db(session_id, result):
    """result를 pose_evaluation_results 테이블에 저장"""
    try:
        if isinstance(result, dict) and 'shortestPath' in result and 'shortestScoring' in result:
            shortest_path = result['shortestPath']
            shortest_scoring = result['shortestScoring']
            
            db.execute_update(
                """INSERT INTO pose_evaluation_results (session_id, body_part, shortest_path, shortest_scoring) 
                   VALUES (%s, %s, %s, %s)""",
                (session_id, 'overall', json.dumps(shortest_path), json.dumps(shortest_scoring))
            )
        else:
            print(f"Invalid result format: {result}")
            return
            
    except Exception as e:
        print(f"Error saving pose evaluation results: {e}")
        return
    
    print(f"=== POSE EVALUATION RESULTS SAVED ===")
