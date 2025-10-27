import { ComponentBase } from '../core/componentBase.js';

export class SendButton extends ComponentBase {
    constructor(eventHandler, globalState, name) {
        super(eventHandler, globalState);
        this.init(name);
    }

    init() {
        // 스레드 아이디
        // tool노드, __end__(?) 구별하는 함수에서 프론트 엔드에 넘겨주는 기능을 추가하면 됨.
        this.button = this.createElement('button', 'sendButton');
        this.button.textContent = 'Send';

        this.addListener('sendButtonClick', this.sendButtonClick.bind(this));

        this.button.addEventListener('click', async(event) => {
            try {
                this.sendButtonClick(event);
            }
            catch (error) {
                this.eventHandler.emit('uploadMessage', { error: 'Failed to send message' });
            }
        });

        // DOM에 버튼 추가
        const chatInputContainer = document.querySelector('.chat-input-container-footer');
        if (chatInputContainer) {
            chatInputContainer.appendChild(this.button);
        }
    }


    async sendButtonClick(event) {
        const message = document.querySelector('#chat-input').value;
        if (message.trim() === '') {
            return;
        }

        try {
            const response = await fetch('/api/sendMessage', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                },
                body: JSON.stringify({ message }),
            });

            if (!response.ok) {
                this.eventHandler.emit('uploadMessage', { error: `${response.status} ${response.statusText}` });
            } else {
                this.eventHandler.emit('uploadMessage', { success: 'Message sent successfully' });
                // 입력창 초기화
                document.querySelector('#chat-input').value = '';
            }
        } catch (error) {
            this.eventHandler.emit('uploadMessage', { error: `${error}` });
        }
    }
}