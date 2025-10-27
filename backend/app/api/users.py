from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from ..services.user_service import UserService
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/users", tags=["users"])
security = HTTPBearer()

# Pydantic 모델 (데이터 검증용)
class UserCreate(BaseModel):
    user_id: str
    user_name: str
    email_address: str
    password: str

class UserLogin(BaseModel):
    user_id: str
    password: str

def get_current_user_from_token(token: str):
    """토큰에서 사용자 정보 추출 (쿠키용)"""
    payload = UserService.verify_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")
    
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=401, detail="토큰에 사용자 정보가 없습니다.")
    
    user = UserService.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="사용자를 찾을 수 없습니다.")
    
    return user

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """JWT 토큰에서 현재 사용자 정보 추출"""  
    token = credentials.credentials
    payload = UserService.verify_token(token)
       
    if payload is None:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")
    
    user_id = payload.get("sub")  
    if user_id is None:
        raise HTTPException(status_code=401, detail="토큰에 사용자 정보가 없습니다.")
    
    user = await UserService.get_user_by_id(user_id) 
    if user is None:
        raise HTTPException(status_code=401, detail="사용자를 찾을 수 없습니다.")
    return user_id


@router.post("/signup")
async def create_user(user_data: UserCreate):
    """사용자 생성"""
    try:
        user = await UserService.create_user(user_data.dict())
        if user:
            return {"message": "사용자가 성공적으로 생성되었습니다.", "user": user}
        else:
            raise HTTPException(status_code=500, detail="사용자 생성에 실패했습니다.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="서버 오류가 발생했습니다.")

@router.post("/login")
async def login_user(login_data: UserLogin):
    """사용자 로그인"""
    result = UserService.authenticate_user(login_data.user_id, login_data.password)
    if not result:
        raise HTTPException(status_code=401, detail="잘못된 사용자 ID 또는 비밀번호입니다.")
    return result