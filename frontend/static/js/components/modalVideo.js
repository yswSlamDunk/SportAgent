import { ComponentBase } from '../core/componentBase.js';

export class ModalVideo extends ComponentBase {
    constructor(eventHandler, globalState, classification) {
        super(eventHandler, globalState);
        this.init(classification);
    }
    
    init(classification) {        
        this.classification = classification;
        this.addListener('modalLoadVideo', this.modalLoadVideo.bind(this));
        this.eventHandler.on('modalCreateVideoElement', this.modalCreateVideoElement.bind(this));
    }

    modalCreateVideoElement(classification) {
        if (classification !== this.classification) {
            return;
        }

        if (!((classification === 'standard' && this.globalState['selectedStandardVideo']) || 
            (classification === 'user' && this.globalState['selectedUserVideo']))) {
            return;
        }

        const element = document.querySelector(`.modal-video-${classification}`);
        if (element) {
            element.remove();
        }
        
        this.video = this.createElement('video', `modal-video-${classification}`);
        document.querySelector(`.modal-video-${classification}-container`).appendChild(this.video);

        this.modalLoadVideo();
}

    async modalLoadVideo() {
        if (this.classification === 'standard') {
            try {
                const response = await fetch(`/api/videos/standard-video?video_id=${String(this.globalState.selectedStandardVideo)}`, {
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('token')}`
                    }
                });

                if (response.ok) {
                    const videoData = await response.json();
                    if (videoData && videoData.length > 0) {
                        const video = videoData[0];
                        // 서버의 파일 경로를 직접 사용
                        const videoUrl = `/${video.video_path}`;
                        this.setVideoSource(videoUrl);
                    } else {
                        console.error('기준 영상 데이터를 찾을 수 없습니다.');
                    }
                } else {
                    console.error('기준 영상 로드 실패:', response.statusText);
                }
            } catch (error) {
                console.error('기준 영상 로드 실패:', error);
            }
        } else if (this.classification === 'user') {
            const fileInput = document.getElementById('userFileUpload');
            const file = fileInput.files[0];
            
            if (file && file.type.startsWith('video/')) {
                const videoUrl = URL.createObjectURL(file);
                this.setVideoSource(videoUrl);
            } else {
                console.error('올바른 비디오 파일을 선택해주세요.');
            }
        }
    }

    setVideoSource(videoUrl) {
        if (this.video) {
            this.video.src = videoUrl;
            this.video.controls = true;
            
            this.video.addEventListener('loadeddata', () => {});
            
            this.video.addEventListener('error', (e) => {
                console.error(`${this.classification} 영상 재생 오류:`, e);
                console.error(`영상 URL: ${videoUrl}`);
                console.error(`영상 요소:`, this.video);
                console.error(`비디오 오류 코드:`, this.video.error?.code);
                console.error(`비디오 오류 메시지:`, this.video.error?.message);
                console.error(`비디오 오류 세부사항:`, this.video.error);
            });
        }
    }
}

