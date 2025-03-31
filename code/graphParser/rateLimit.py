import os
import functools
import time
import re
from typing import List, Callable, Any

class APIKeyManager:
    def __init__(self):
        self._api_keys = {key: 'active' for key in self._load_api_keys()}
        self._current_key_index = 0

    def _load_api_keys(self) -> List[str]:
        keys = []
        for key_name in os.environ:
            if 'OPENAI_API_KEY' in key_name:
                keys.append(os.environ[key_name])
        return keys

    def get_next_available_key(self) -> str:
        initial_index = self._current_key_index

        key_list = list(self._api_keys.keys())
        while True:
            current_key = key_list[self._current_key_index]
            if self._api_keys[current_key] == 'active':                
                return current_key

            self._current_key_index = (self._current_key_index + 1) % len(key_list)

            if self._current_key_index == initial_index:
                raise Exception("모든 API 키가 비활성화되었습니다.")

    def update_key_status(self, key: str, status: str):
        self._api_keys[key] = status

    def extract_wait_time(self, error_message: str):
        """에러 메시지에서 대기 시간을 추출합니다."""
        time_units = {'ms': 0.001, 's': 1, 'm': 60, 'h': 3600}
        
        matches = re.findall(r"(\d+\.?\d*)\s*(ms|s|m|h)[.]", error_message)
        
        if not matches:
            return False
        
        total_wait_time = 0
        for match in matches:
            value, unit = match
            total_wait_time += float(value) * time_units[unit]

        return total_wait_time

    def analize_error(self, error: Exception):
        error_message =  error.message 

        if ('(RPD)' in error_message):
            print(' '*4 + f"{self._current_key_index}번째 키의 일일 요청 한도(RPD) 초과됨: 재시도 불가")
            print(error_message)
            return None
            
        elif ('(TPD)' in error_message):
            print(' '*4 + f"{self._current_key_index}번째 키의 일일 토큰 한도(TPD) 초과됨: 재시도 불가")
            print(error_message)
            return None
        
        elif ('(RPM)' in error_message):
            wait_time = self.extract_wait_time(error_message)
            return wait_time

        elif ('(TPM)' in error_message):
            wait_time = self.extract_wait_time(error_message)
            return wait_time

        elif ('(IPM)' in error_message):
            wait_time = self.extract_wait_time(error_message)
            return wait_time


api_key_manager = APIKeyManager()

def handle_rate_limits(func: Callable, max_retries: int = 5, max_delay: float = 60.0 * 10):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        retries = 0
        retry_time = 30

        while True:
            current_key = api_key_manager.get_next_available_key()
            kwargs['current_api_key'] = current_key

            while retries < max_retries:
                try:
                    result = func(*args, **kwargs)
                    return result
                
                except Exception as e:
                    print(e)
                    wait_time = api_key_manager.analize_error(e)
                    if wait_time is not None:
                        if wait_time > max_delay:
                            print(' '*4 + f"{api_key_manager._current_key_index}번째 키의 대기 시간({wait_time}초)이 기준 초과")
                            print(e.message)
                            break
                        else:
                            print(' '*4 + f"{api_key_manager._current_key_index}번째 키: {retries + 1}번째. 대기 시간:({wait_time + retry_time}초)")
                            time.sleep(wait_time + retry_time)
                            retries += 1
                    else: # RPD, TPD 초과
                        api_key_manager.update_key_status(current_key, 'exceeded')
                        break

    return wrapper
