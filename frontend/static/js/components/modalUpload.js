import { ComponentBase } from '../core/componentBase.js';

export class UploadModal extends ComponentBase {
    constructor(eventHandler, globalState) {
        super(eventHandler, globalState);
        this.init();
    }
    
    
    init() {
        this.button = this.createElement('button', 'uploadModal');
        this.button.textContent = '영상 업로드';
        this.button.addEventListener('click', this.showModal.bind(this));

        document.querySelector('.chat-input-container-footer').appendChild(this.button);
        
        this.modal = document.getElementById('upload-modal');

        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal) {
                this.hideModal();
            }
        });
    }
    
    showModal() {
        if (this.modal) {
            this.modal.style.display = 'block';
            document.body.style.overflow = 'hidden';
        }
    }
    
    hideModal() {
        if (this.modal) {
            this.modal.style.display = 'none';
            document.body.style.overflow = 'auto';
        }
    }
}