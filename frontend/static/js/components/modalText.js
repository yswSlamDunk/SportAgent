import { ComponentBase } from '../core/componentBase.js';

export class ModalText extends ComponentBase {
    constructor(eventHandler, globalState) {
        super(eventHandler, globalState);
        this.init();
    }
    
    init() {
        this.label = this.createElement('label', 'loggingTextLabel');
        this.label.textContent = 'Logging';

        this.text = this.createElement('div', 'modal-text');
        this.text.textContent = '';
        this.text.id = 'loggingText';
        this.text.readOnly = true;

        document.querySelector('.modal-message-container').appendChild(this.label);
        document.querySelector('.modal-message-container').appendChild(this.text);

        this.eventHandler.on('modalAddText', this.addText.bind(this));
    }

    addText(text) {
        const time = new Date().toLocaleTimeString();
        this.text.textContent = this.text.textContent + `${time} ${text}\n`;
    }
}

export class ModalVideoName extends ComponentBase {
    constructor(eventHandler, globalState) {
        super(eventHandler, globalState);
        this.init();
    }
    
    init() {
        this.label = this.createElement('label', 'videoNameLabel');
        this.label.textContent = '영상 이름:';
        this.text = this.createElement('textarea', 'modalVideoNameText');
        // this.text.textContent = new Date().toISOString().replace(/T/, '-').replace(/\..+/, '').replace(/:/g, '-').replace(/-/g, '-', 3)
        this.text.textContent = '';
        this.text.style.resize = 'none';
        
        document.querySelector('.modal-header-video-name').appendChild(this.label);
        document.querySelector('.modal-header-video-name').appendChild(this.text);

    }
}