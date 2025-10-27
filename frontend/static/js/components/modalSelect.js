import { ComponentBase } from '../core/componentBase.js';

export class SelectStandardVideo extends ComponentBase {
    constructor(eventHandler, globalState) {
        super(eventHandler, globalState);
        this.init();
    }

    init() {
        this.label = this.createElement('label', 'selectStandardVideoLabel');
        this.label.textContent = '기준 영상:';
        this.label.htmlFor = 'selectStandardVideo';

        this.select = this.createElement('select', 'selectStandardVideo');
        const defaultOption = this.createElement('option');
        defaultOption.textContent = '기준 영상을 선택하세요';
        defaultOption.value = '';
        this.select.appendChild(defaultOption);
        
        this.loadStandardVideos().then(data => {
            data.forEach(video => {
                const option = this.createElement('option');
                option.textContent = video.video_name;
                option.value = video.id;
                this.select.appendChild(option);
            });
        });

        document.querySelector('.modal-header-standard').appendChild(this.label);
        document.querySelector('.modal-header-standard').appendChild(this.select);
        this.select.addEventListener('change', this.selectStandardVideoChange.bind(this));
    }
    
    async loadStandardVideos() {
        try {
            const response = await fetch('/api/videos/standard-videos-list', {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            });
            if (response.ok) {
                return await response.json();
            } else {
                console.error('기준 영상 목록 로드 실패:', response.statusText);
                return [];
            }
        } catch (error) {
            console.error('기준 영상 목록 로드 실패:', error);
            return [];
        }
    }

    selectStandardVideoChange(event) {
        const selectedValue = event.target.value;
        this.globalState['selectedStandardVideo'] = selectedValue;
        this.eventHandler.emit('viweVideoModalButtoCheck');
        this.eventHandler.emit('uploadVideoModalButtoCheck');
    }
}


export class UserFileUpload extends ComponentBase {
    constructor(eventHandler, globalState) {
        super(eventHandler, globalState);
        this.init();
    }
    
    
    init() {
        this.label = this.createElement('label', 'userFileUploadLabel');
        this.label.textContent = '사용자 영상:';
        this.label.htmlFor = 'userFileUpload';
        
        
        this.input = this.createElement('input', 'userFileUpload');
        this.input.type = 'file';
        this.input.accept = '.mp4';
        this.input.id = 'userFileUpload';
        this.input.name = 'userFileUpload';
        
        document.querySelector('.modal-header-user').appendChild(this.label);
        document.querySelector('.modal-header-user').appendChild(this.input);
        this.input.addEventListener('change', this.userFileUploadChange.bind(this));
    }

    userFileUploadChange(event) {
        const selectedValue = event.target.value;
        this.globalState['selectedUserVideo'] = selectedValue;
        
        this.eventHandler.emit('viweVideoModalButtoCheck');
        this.eventHandler.emit('uploadVideoModalButtoCheck');
    }
}