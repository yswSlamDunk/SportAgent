import { ComponentBase } from '../core/componentBase.js';


export class ViewVideoModalButton extends ComponentBase {
    constructor(eventHandler, globalState) {
        super(eventHandler, globalState);
        this.init();
    }
    
    init() {
        this.button = this.createElement('button', 'viewVideoModal');
        this.button.disabled = true;
        this.button.textContent = '영상 보기';
        this.button.addEventListener('click', () => {
            this.eventHandler.emit('modalCreateVideoElement', 'standard');
            this.eventHandler.emit('modalCreateVideoElement', 'user');
        });

        this.addListener('viweVideoModalButtoCheck', this.checkButtonEnable.bind(this));
        document.querySelector('.modal-footer').appendChild(this.button);
    }

    checkButtonEnable() {
        if (this.globalState['selectedStandardVideo'] || this.globalState['selectedUserVideo']) {
            this.button.disabled = false;
        } else {
            this.button.disabled = true;
        }

    }
}           

export class UploadVideoModalButton extends ComponentBase {
    constructor(eventHandler, globalState, poseEstimator) {
        super(eventHandler, globalState);
        this.poseEstimator = poseEstimator;
        this.init();
    }
    
    init() {
        this.button = this.createElement('button', 'uploadVideoModal');
        this.button.textContent = '영상 업로드 및 자세 분석';
        this.button.disabled = true;
        document.querySelector('.modal-footer').appendChild(this.button);
        this.button.addEventListener('click', () => this.uploadVideo());        

        this.addListener('uploadVideoModalButtoCheck', this.checkButtonEnable.bind(this));
    }

    checkButtonEnable() {
        if (this.globalState['selectedUserVideo'] && this.globalState['selectedStandardVideo']) {
            this.button.disabled = false;
        } else {
            this.button.disabled = true;
        }
    }


    async uploadVideo() {        
        this.setLoadingState(true);
        
        try {
            const result_save = await this.saveVideo(); 
            this.eventHandler.emit('modalAddText', result_save.message);
            if (!result_save.success) {
                this.setLoadingState(false);
                alert('다른 영상을 사용해주세요.');
                this.eventHandler.emit('modalAddText', '다른 영상을 사용해주세요.');
                return;
            }

            const result_pose_estimation = await this.poseEstimation(this.globalState['selectedUserVideoId']);
            this.eventHandler.emit('modalAddText', result_pose_estimation.message);
            if (!result_pose_estimation.success) {
                this.setLoadingState(false);
                alert('다른 영상을 사용해주세요.');
                this.eventHandler.emit('modalAddText', '다른 영상을 사용해주세요.');
                return;
            }

            const result_pose_scoring = await this.poseScoring();    
            this.eventHandler.emit('modalAddText', result_pose_scoring.message);
            if (!result_pose_scoring.success) {
                this.setLoadingState(false);
                alert('다른 영상을 사용해주세요.');
                this.eventHandler.emit('modalAddText', '다른 영상을 사용해주세요.');
                return;
            }
            
            this.setLoadingState(false);
            
            this.eventHandler.emit('setSportSelect', {
                sportId: result_pose_scoring.sport_id, 
                sessionId: result_pose_scoring.session_id
            });
        
        } catch (error) {
            this.eventHandler.emit('modalAddText', '분석 오류 발생 ${error}');
            this.eventHandler.emit('modalAddText', '다른 영상을 사용해주세요.');
            this.setLoadingState(false);
        }
    }

    preventEscape(event) {
        if (event.key === 'Escape') {
            event.preventDefault();
            event.stopPropagation();
        }
    }

    async saveVideo() {
        const fileInput = document.querySelector('#userFileUpload');
        const file = fileInput.files[0];
        
        if (!file) {
            alert('파일을 선택해주세요.');
            return;
        }
        
        const selectedVideoId = this.globalState['selectedStandardVideo'];
        
        try {
            const videoResponse = await fetch(`/api/videos/standard-video?video_id=${selectedVideoId}`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            });
            
            const videoData = await videoResponse.json();
            const sportId = videoData[0].sport_id; 
            const videoNameElement = document.querySelector('.modalVideoNameText');
            const videoName = videoNameElement ? videoNameElement.value.trim() : '';
            
            if (!videoName) {
                alert('영상 이름을 입력해주세요.');
                return;
            }
            
            this.eventHandler.emit('modalAddText', '영상 업로드 시작');
            
            const formData = new FormData();
            formData.append('video', file);
            formData.append('sport_id', sportId);
            formData.append('video_name', videoName);
            
            const response = await fetch('/api/videos/upload-user-video', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                },
                body: formData
            });
            
            const result = await response.json();
            if (result.success) {
                this.globalState['selectedUserVideoId'] = result.video_id;
                this.globalState['sportSelect'] = sportId;
            }
            
            return result

        } catch (error) {
            this.eventHandler.emit('modalAddText', '네트워크 오류 발생 ${error}');
        }
    }

    async poseEstimation(video_id) {
        const response = await fetch(`/api/pose/estimation/?video_id=${video_id}`, {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
    
        const result = await response.json();        
        return result;
    }

    async poseScoring() {
        const standard_video_id = this.globalState['selectedStandardVideo'];
        const user_video_id = this.globalState['selectedUserVideoId'];
        const user_id = this.globalState['userInfo']['id'];
        const sport_id = this.globalState['sportSelect'];
    
        const response = await fetch(`/api/pose/scoring?standard_video_id=${standard_video_id}&user_video_id=${user_video_id}&user_id=${user_id}&sport_id=${sport_id}`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
    
        const result = await response.json();
        return result;
    }

    setLoadingState(isLoading) {
        const modal = document.querySelector('.modal');
        const selectStandardVideo = document.querySelector('.selectStandardVideo');
        const userFileUpload = document.querySelector('#userFileUpload');
        const modalVideoNameText = document.querySelector('.modalVideoNameText');
        const viewVideoModal = document.querySelector('.viewVideoModal');
        const uploadVideoModal = document.querySelector('.uploadVideoModal');
        
        // 모달 로딩 상태 관리
        if (modal) {
            if (isLoading) {
                modal.classList.add('loading');
                document.addEventListener('keydown', this.preventEscape);
            } else {
                modal.classList.remove('loading');
                document.removeEventListener('keydown', this.preventEscape);
            }
        }
        
        if (selectStandardVideo) selectStandardVideo.disabled = isLoading;
        if (userFileUpload) userFileUpload.disabled = isLoading;
        if (modalVideoNameText) modalVideoNameText.disabled = isLoading;
        if (viewVideoModal) viewVideoModal.disabled = isLoading;
        if (uploadVideoModal) uploadVideoModal.disabled = isLoading;
    }
}
