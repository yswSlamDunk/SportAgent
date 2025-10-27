from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Form
from ..api.users import get_current_user
from ..database.connection import db

import os
import shutil
from datetime import datetime

router = APIRouter(prefix="/api/videos", tags=["videos"])

@router.get("/")
async def get_videos(
    sport_id: int = Query(None, description="운동 종목 ID"),
    current_user: dict = Depends(get_current_user)
):
    try:
        if sport_id:
            # 특정 운동 종목의 비디오만 조회
            videos = db.execute_query(
                """SELECT v.*, s.name as sport_name 
                   FROM videos v 
                   JOIN sports s ON v.sport_id = s.id 
                   WHERE v.user_id = %s AND v.sport_id = %s
                   ORDER BY v.created_at DESC""",
                (current_user['id'], sport_id)
            )
        else:
            # 모든 비디오 조회
            videos = db.execute_query(
                """SELECT v.*, s.name as sport_name 
                   FROM videos v 
                   JOIN sports s ON v.sport_id = s.id 
                   WHERE v.user_id = %s 
                   ORDER BY v.created_at DESC""",
                (current_user['id'],)
            )
        return videos
    except Exception as e:
        raise HTTPException(status_code=500, detail="비디오 목록 조회 실패")

@router.get("/by-sport/{sport_id}")
async def get_videos_by_sport(sport_id: int, current_user: dict = Depends(get_current_user)):
    """특정 운동 종목의 비디오 목록 조회"""
    try:
        videos = db.execute_query(
            """SELECT v.*, s.name as sport_name 
               FROM videos v 
               JOIN sports s ON v.sport_id = s.id 
               WHERE v.user_id = %s AND v.sport_id = %s
               ORDER BY v.created_at DESC""",
            (current_user['id'], sport_id)
        )
        return videos
    except Exception as e:
        raise HTTPException(status_code=500, detail="운동별 비디오 조회 실패")

@router.get("/standard-videos-list")
async def get_standard_videos_list():
    """기준 영상 목록 조회"""
    try:
        standard_videos = db.execute_query(
            """SELECT * FROM videos WHERE video_type = 'standard'""",
        )
        return standard_videos
    except Exception as e:
        raise HTTPException(status_code=500, detail="기준 영상 목록 조회 실패")
        

@router.get("/standard-video")
async def get_standard_video(video_id: int):  # video_id → id로 변경
    """기준 영상 조회"""
    try:
        standard_video = db.execute_query(
            """SELECT v.*
               FROM videos v
               WHERE v.id = %s AND v.video_type = 'standard'
            """,
            (video_id,)
        )
        return standard_video
    except Exception as e:
        raise HTTPException(status_code=500, detail="기준 영상 조회 실패")

@router.post("/upload-user-video")
async def upload_user_video(
    video: UploadFile = File(...),
    sport_id: int = Form(...),
    video_name: str = Form(...),
    current_user: int = Depends(get_current_user)
):
    try:        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, "..", "..", "..")
        media_dir = os.path.join(project_root, "media", "videos", "user")
                
        os.makedirs(media_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{current_user}_{sport_id}_{timestamp}_{video_name}.mp4"
        save_path = os.path.join(media_dir, filename)
        # 경로 구분자를 통일 (/ 사용)
        relative_video_path = os.path.join("media", "videos", "user", filename).replace("\\", "/")
                
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        db.execute_update(
            """INSERT INTO videos (user_id, sport_id, video_type, video_path, video_name) 
            VALUES (%s, %s, %s, %s, %s)""",
            (current_user, sport_id, 'user', relative_video_path, video_name)
        )
        
        video_id = db.execute_query(
            """SELECT id FROM videos WHERE user_id = %s AND video_path = %s""",
            (current_user, relative_video_path)
        )[0]['id']
        
        return {
            "success": True,
            "message": "영상이 업로드 완료",
            "save_path": save_path,
            "video_id": video_id
        }
        
    except Exception as e:
        if 'save_path' in locals() and os.path.exists(save_path):
            os.remove(save_path)
            
        return {
            "success": False,
            "message": f"영상 업로드 실패 ${e}"
        }