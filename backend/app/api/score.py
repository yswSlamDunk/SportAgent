from fastapi import APIRouter, HTTPException, Depends
from ..database.connection import db
from ..api.users import get_current_user

router = APIRouter(prefix="/api/score", tags=["score"])

@router.get("/")
async def get_score(session_id: int):
    try:
        # 1. 전체 평균 점수 계산
        overall_score_query = db.execute_query(
            """SELECT AVG(average_score) as overall_score 
               FROM pose_scores 
               WHERE session_id = %s""",
            (session_id,)
        )

        overall_score = overall_score_query[0]['overall_score'] if overall_score_query else None
        
        # 2. 신체 부위별 점수 정보 (한글명 포함)
        body_part_scores = db.execute_query(
            """SELECT ps.body_part, ps.average_score, ps.status, ss.standard_score,
                      COALESCE(ss.body_part_korean, ps.body_part) as body_part_korean
               FROM pose_scores ps
               INNER JOIN pose_evaluation_sessions pes ON ps.session_id = pes.id
               LEFT JOIN sport_standard_scores ss ON ss.body_part = ps.body_part 
                   AND ss.video_id = pes.standard_video_id
               WHERE ps.session_id = %s""",
            (session_id,)
        )
        # 3. result_sum 매트릭스 데이터 (analytics-figure용)
        analytics_data = db.execute_query(
            """SELECT shortest_path, shortest_scoring 
            FROM pose_evaluation_results 
            WHERE session_id = %s""",
            (session_id,)
        )
        return {
            "success": True,
            "session_id": session_id,
            "overall_score": float(overall_score) if overall_score else None,
            "analytics_data": analytics_data[0] if analytics_data else None,
            "body_part_scores": [
                {
                    "body_part": score['body_part'],
                    "body_part_korean": score['body_part_korean'],
                    "average_score": float(score['average_score']),
                    "status": score['status'],
                    "standard_score": float(score['standard_score']) if score['standard_score'] else None
                }
                for score in body_part_scores
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"점수 조회 실패: {str(e)}")


@router.get("/session-info")
async def get_session_info(session_id: int):
    try:
        print(f"세션 정보 조회 시작: session_id={session_id}")
        
        # 세션 정보와 비디오 경로 조회
        session_info = db.execute_query(
            """SELECT 
                pes.id as session_id,
                sv.video_path as standard_video_path,
                uv.video_path as user_video_path,
                sv.video_name as standard_video_name,
                uv.video_name as user_video_name
               FROM pose_evaluation_sessions pes
               LEFT JOIN videos sv ON sv.id = pes.standard_video_id AND sv.video_type = 'standard'
               LEFT JOIN videos uv ON uv.id = pes.user_video_id AND uv.video_type = 'user'
               WHERE pes.id = %s""",
            (session_id,)
        )
        
        print(f"세션 정보 조회 결과: {session_info}")
        
        if session_info:
            return session_info[0]
        else:
            print("세션을 찾을 수 없습니다.")
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"세션 정보 조회 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"세션 정보 조회 실패: {str(e)}")
