export async function signup(event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    const userData = {
        user_id: formData.get('user_id'),
        user_name: formData.get('user_name'),
        email_address: formData.get('email_address'),
        password: formData.get('password')
    };

    try {
        const response = await fetch('/api/users/signup', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(userData)
        });

        if (response.ok) {
            alert('회원가입이 완료되었습니다.');
            window.location.href = '/login';
        } else {
            const error = await response.json();
            alert(error.detail || '회원가입에 실패했습니다.');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('회원가입 중 오류가 발생했습니다.');
    }
}

document.getElementById('signupForm').addEventListener('submit', signup);