from ..database.connection import db
import hashlib
import os
from typing import Optional, List
from datetime import datetime, timedelta, timezone
import jwt

class UserService:
    @staticmethod
    def hash_password(password: str) -> str:
        salt = os.urandom(32)
        key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return salt.hex() + key.hex()

    @staticmethod
    def verify_password(password:str, hashed: str) -> bool:
        salt = bytes.fromhex(hashed[:64])
        key = bytes.fromhex(hashed[64:])
        new_key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return new_key == key
    
    @staticmethod
    async def create_user(user_data: dict) -> Optional[dict]:
        """사용자 생성"""
        # 중복 확인
        existing_user = db.execute_query(
            "SELECT id FROM users WHERE user_id = %s OR email_address = %s",
            (user_data['user_id'], user_data['email_address'])
        )
        
        if existing_user:
            raise ValueError("이미 존재하는 사용자 ID 또는 이메일입니다.")
        
        # 비밀번호 해싱
        hashed_password = UserService.hash_password(user_data['password'])
        
        # 사용자 생성
        user_id = db.execute_update(
            """INSERT INTO users (user_id, user_name, email_address, password_hash) 
               VALUES (%s, %s, %s, %s)""",
            (user_data['user_id'], user_data['user_name'], user_data['email_address'], hashed_password)
        )
        
        if user_id:
            # 생성된 사용자 정보 반환
            user = db.execute_query(
                "SELECT id, user_id, user_name, email_address, created_at, updated_at FROM users WHERE id = %s",
                (user_id,)
            )
            return user[0] if user else None
        
        return None
    
    @staticmethod
    async def get_user_by_id(user_id: int) -> Optional[dict]:  # int로 변경
        """사용자 ID로 사용자 조회"""
        users = db.execute_query(
            "SELECT id, user_id, user_name, email_address, created_at, updated_at FROM users WHERE id = %s",
            (user_id,)  
        )
        return users[0] if users else None

    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
        """JWT 액세스 토큰 생성"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES")))

        to_encode.update({
            "exp": expire,
            "sub": str(data.get("id")),  
            "user_id": data.get("user_id"),  
            "user_name": data.get("user_name")  
        })
        
        secret_key = os.getenv("JWT_SECRET_KEY")
        algorithm = os.getenv("JWT_ALGORITHM")
        
        encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=algorithm)
        return encoded_jwt

    @staticmethod
    def verify_token(token: str) -> Optional[dict]:
        """JWT 토큰 검정"""
        try:            
            secret_key = os.getenv("JWT_SECRET_KEY")
            algorithm = os.getenv("JWT_ALGORITHM")
            
            if not secret_key or not algorithm:
                return None
                
            payload = jwt.decode(token, secret_key, algorithms=[algorithm])

            if 'sub' in payload:
                payload['sub'] = int(payload['sub'])
            return payload

        except jwt.ExpiredSignatureError as e:
            return None
        except jwt.InvalidTokenError as e:
            return None
        except Exception as e:
            return None
        
    @staticmethod
    def authenticate_user(user_id: str, password: str) -> Optional[dict]:
        """사용자 인증 및 토큰 생성"""
        users = db.execute_query(
            "SELECT id, user_id, user_name, email_address, password_hash FROM users WHERE user_id = %s",
            (user_id,)
        )

        if not users:
            return None
        
        user = users[0]

        if UserService.verify_password(password, user['password_hash']):
            user.pop('password_hash', None)

            access_token = UserService.create_access_token(
                data=user  
            )

            return {
                "user": user,
                "access_token": access_token,
                "token_type": "bearer"
            }
        
