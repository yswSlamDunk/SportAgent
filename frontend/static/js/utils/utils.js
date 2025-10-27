export function getCurrentTime() {
    const now = new Date();
    const format = (num) => String(num).padStart(2, '0');
    return `${now.getFullYear()}-${format(now.getMonth() + 1)}-${format(now.getDate())} ${format(now.getHours())}:${format(now.getMinutes())}:${format(now.getSeconds())}`;
}

export function decodeToken(token) {
    try {
        const base64Url = token.split('.')[1];
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        const jsonPayload = decodeURIComponent(atob(base64).split('').map(function(c) {
            return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
        }).join(''));
        return JSON.parse(jsonPayload);
    } catch (error) {
        console.error('토큰 디코딩 실패:', error);
        return null;
    }
}

export function checkAuth() {
    const token = localStorage.getItem('token');
    if (!token) {
        alert('로그인이 필요합니다.');
        window.location.href = '/login';
        return false;
    }
    return true;
}

export function getUserInfoFromToken() {
    try {
        const token = localStorage.getItem('token');
        if (!token) return null;
        
        const payload = decodeToken(token);
        return {
            id: payload.sub,           // 숫자 ID
            user_id: payload.user_id,  // 문자열 ID
            user_name: payload.user_name
        };
    } catch (error) {
        console.error('토큰 디코딩 실패:', error);
        return null;
    }
}