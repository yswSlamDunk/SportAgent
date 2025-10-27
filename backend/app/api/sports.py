from fastapi import APIRouter, HTTPException, Depends
from ..database.connection import db
from ..api.users import get_current_user

router = APIRouter(prefix="/api/sports", tags=["sports"])

@router.get("/")
async def get_sports():
    """모든 운동 종목 조회"""
    try:
        sports = db.execute_query("SELECT * FROM sports ORDER BY name")
        return sports
    except Exception as e:
        raise HTTPException(status_code=500, detail="운동 종목 조회 실패")

@router.get("/{sport_id}/pose-estimations")
async def get_valid_pose_estimations(sport_id: int, current_user: int = Depends(get_current_user)):
    """특정 운동 종목의 유효한 포즈 추정 데이터 조회"""
    print(f"디버깅: sport_id={sport_id}, current_user={current_user}, type={type(current_user)}")

    try:    
        pose_estimations = db.execute_query(
            """SELECT pe.*, v.sport_id, v.user_id
               FROM pose_estimations pe
               JOIN videos v ON pe.video_id = v.id
               WHERE v.sport_id = %s AND v.user_id = %s AND pe.is_valid = 1
               ORDER BY pe.created_at DESC""",
            (sport_id, current_user)
        )
        return pose_estimations
    except Exception as e:
        print(f"디버깅: 오류 발생={str(e)}")
        raise HTTPException(status_code=500, detail="포즈 추정 데이터 조회 실패")
        