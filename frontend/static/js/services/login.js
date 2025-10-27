export async function login(event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    const loginData = {
        user_id: formData.get('user_id'),
        password: formData.get('password')
    };

    try {
        const response = await fetch('/api/users/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(loginData)
        });

        if (response.ok) {
            const result = await response.json();
            localStorage.setItem('token', result.access_token);
            
            // 쿠키 설정 (수정된 부분)
            const token = result.access_token;
            
            // 쿠키 만료 시간 설정 (24시간)
            const expires = new Date();
            expires.setTime(expires.getTime() + (24 * 60 * 60 * 1000));
            
            document.cookie = `token=${token}; expires=${expires.toUTCString()}; path=/; SameSite=Strict`;
            
            // 잠시 대기 후 페이지 이동
            setTimeout(() => {
                window.location.href = '/';
            }, 100);
        } else {
            alert('로그인에 실패했습니다.');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('로그인 중 오류가 발생했습니다.');
    }
}

document.getElementById('loginForm').addEventListener('submit', login);