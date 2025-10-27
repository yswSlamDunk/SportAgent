import { ComponentBase } from '../core/componentBase.js';

export class HelpModal extends ComponentBase {
    constructor(eventHandler, globalState) {
        super(eventHandler, globalState);
        this.init();
    }
    
    init() {
        // Help 버튼 이벤트 리스너 설정
        const helpButton = document.querySelector('.analytics-timeline-help');
        if (helpButton) {
            helpButton.addEventListener('click', this.showModal.bind(this));
        }

        // Modal 요소 가져오기
        this.modal = document.getElementById('help-modal');
        
        if (this.modal) {
            // Close 버튼 이벤트 리스너 설정
            const closeButton = this.modal.querySelector('.modal-close-button');
            if (closeButton) {
                closeButton.addEventListener('click', this.hideModal.bind(this));
            }

            // Modal 배경 클릭 시 닫기
            this.modal.addEventListener('click', (e) => {
                if (e.target === this.modal) {
                    this.hideModal();
                }
            });

            // ESC 키로 모달 닫기
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape' && this.modal.style.display === 'block') {
                    this.hideModal();
                }
            });
        }
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

