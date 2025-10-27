import { ComponentBase } from '../core/componentBase';

export class Message extends ComponentBase {
    constructor(eventHandler, globalState, name) {
        super(eventHandler, globalState);
        this.init();
    }

    init() {
        this.addListener('createUserMessage', this.createUserMessage.bind(this));
        this.addListener('createAssistantMessage', this.createAssistantMessage.bind(this));
    }

    async createUserMessage(message) {
        const messageElement = this.createElement('div', 'user-message');
        messageElement.textContent = message;

        document.querySelector('.chat-container').appendChild(messageElement);
    }

    async createAssistantMessage(message) {
        const messageElement = this.createElement('div', 'assistant-message');
        const separator = this.createElement('div', 'message-separator');

        messageElement.textContent = message;

        document.querySelector('.chat-container').appendChild(messageElement);
        // separator.style.borderTop = '1px solid #ccc'; // 이건 내일 추가하자.
        document.querySelector('.chat-container').appendChild(separator);
    }
}