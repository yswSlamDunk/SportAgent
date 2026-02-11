import yaml
import httpx

def load_prompt(yaml_file):
    with open(yaml_file, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def get_all_score_history_sync(user_id: str):
    """
    API 엔드포인트를 동기적으로 호출하는 래퍼 함수
    """
    try:
        # API 엔드포인트 URL (자신의 서버 주소로 변경)
        url = f"http://localhost:8000/api/score/all_score_history?user_id={user_id}"        
        # 동기 HTTP 요청
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()  # 에러 체크
        
        return response.json()
    except Exception as e:
        print(f"점수 기록 조회 실패: {str(e)}")
        return []