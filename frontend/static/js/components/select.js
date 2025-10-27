import { ComponentBase } from '../core/componentBase.js';

export class SportSelect extends ComponentBase {
    constructor(eventHandler, globalState) {
        super(eventHandler, globalState);
        this.init();
    }

    init() {       
        this.label = this.createElement('label', 'selectActionLabel');
        this.label.textContent = '동작 선택:';
        this.label.htmlFor = 'sportSelect';

        
        this.select = this.createElement('select', 'sportSelect');
        const defaultOption = this.createElement('option');
        defaultOption.textContent = '동작을 선택하세요';
        defaultOption.value = '';
        this.select.appendChild(defaultOption);
        
        this.loadSports().then(data => {
            data.forEach(sport => {
                const option = this.createElement('option');
                option.textContent = sport.name;
                option.value = sport.id;
                this.select.appendChild(option);
            });
        });
        
        this.select.addEventListener('change', this.sportSelectChange.bind(this));
        this.addListener('setSportSelect', this.setSportSelect.bind(this));
        
        document.querySelector('.select-action-group').appendChild(this.label);
        document.querySelector('.select-action-group').appendChild(this.select);
    }

    setSportSelect(data) {
        const { sportId, sessionId } = data;
        this.select.value = sportId;
        this.globalState.sportSelect = sportId;
        this.eventHandler.emit('LoadDate', sessionId);
    }

    sportSelectChange(event) {
        const selectedValue = event.target.value;

        this.globalState.sportSelect = selectedValue;
        // 여기에 운동 보고서 내용 초기화 하는거 필요.


        if (selectedValue === '') {
            const dataSelect = document.querySelector('.dateSelect');
            dataSelect.value = '';
            dataSelect.disabled = true;
        } else {
            this.eventHandler.emit('LoadDate');
        }

        // 기존 요소들 제거
        const existingTimeline = document.querySelector('.analytics-figure-timeline');
        const existingImageContent = document.querySelector('.analytics-images-content');
        const existingTableWrapper = document.querySelector('.analytics-table-wrapper');

        if (existingTimeline) existingTimeline.remove();
        if (existingImageContent) existingImageContent.remove();
        if (existingTableWrapper) existingTableWrapper.remove();

        // 새로운 요소들 생성
        const timeline = this.createElement('div', 'analytics-figure-timeline');
        const imageContent = this.createElement('div', 'analytics-images-content');
        const tableWrapper = this.createElement('div', 'analytics-table-wrapper');

        // DOM에 추가
        const figureTimelineContainer = document.querySelector('.analytics-figure-timeline-container');
        const imagesContainer = document.querySelector('.analytics-images-container');
        const tableWrapperContainer = document.querySelector('.analytics-table-wrapper-container');

        if (figureTimelineContainer) figureTimelineContainer.appendChild(timeline);
        if (imagesContainer) imagesContainer.appendChild(imageContent);
        if (tableWrapperContainer) tableWrapperContainer.appendChild(tableWrapper);
    }

    async loadSports() {
        try {
            const response = await fetch('/api/sports', {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            });
            if (response.ok) {
                return await response.json();
                
            } else {
                console.error('운동 목록 로드 실패:', response.statusText);
                return [];
            }
        } catch (error) {
            console.error('운동 목록 로드 실패:', error);
            return [];
        }
    }
}


export class DateSelect extends ComponentBase {
    constructor(eventHandler, globalState) {
        super(eventHandler, globalState);
        this.init();
    }

    init() {
        this.label = this.createElement('label', 'selectDateLabel');
        this.label.textContent = '날짜 선택:';
        this.label.htmlFor = 'dateSelect';

        this.select = this.createElement('select', 'dateSelect');
        this.select.disabled = true;

        const defaultOption = this.createElement('option');
        defaultOption.textContent = '날짜를 선택하세요';
        defaultOption.value = '';
        this.select.appendChild(defaultOption);
        
        this.select.addEventListener('change', this.dateSelectChange.bind(this));
        
        document.querySelector('.select-date-group').appendChild(this.label);
        document.querySelector('.select-date-group').appendChild(this.select);

        this.addListener('LoadDate', this.loadDates.bind(this));
        this.addListener('LoadSessionResults', this.loadSessionResults.bind(this));
    }

    async dateSelectChange(event) {
        const selectedValue = event.target.value;
        this.globalState.sessionId = selectedValue;
        
        await this.loadSessionInfo(selectedValue);
        const sessionResults = await this.loadSessionResults(selectedValue);

        if (sessionResults) {
            this.eventHandler.emit('preprocessingScore');
            this.eventHandler.emit('createFigure');
            this.eventHandler.emit('createImages');
            this.eventHandler.emit('createTable');
        }
    }

    async loadSessionInfo(sessionId) {
        try {
            // 세션 정보를 가져와서 비디오 정보 설정
            const response = await fetch(`/api/score/session-info?session_id=${sessionId}`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            });
            
            if (response.ok) {
                const sessionInfo = await response.json();
                
                // 비디오 정보를 globalState에 저장
                if (sessionInfo.standard_video_path) {
                    this.globalState.selectedStandardVideoPath = sessionInfo.standard_video_path;
                }
                if (sessionInfo.user_video_path) {
                    this.globalState.selectedUserVideoPath = sessionInfo.user_video_path;
                }
                
                return sessionInfo;
            } else {
                console.error('세션 정보 로드 실패:', response.statusText);
                return null;
            }
        } catch (error) {
            console.error('세션 정보 로드 실패:', error);
            return null;
        }
    }

    async loadDates(sessionId) {
        if (!this.globalState.sportSelect) {
            console.warn('운동 종목이 선택되지 않았습니다.');
            return [];
        }

        try {
            // pose_evaluation_sessions 테이블에서 세션 목록 조회
            const response = await fetch(`/api/pose/sessions?sport_id=${this.globalState.sportSelect}&user_id=${this.globalState.userInfo.id}`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            });
            if (response.ok) {
                const data = await response.json();
                if (data.length > 0) {
                    while (this.select.children.length > 1) {
                        this.select.removeChild(this.select.lastChild);
                    }

                    data.forEach(session => {
                        const option = this.createElement('option');
                        option.textContent = session.created_at;
                        option.value = session.session_id; // session_id 사용
                        this.select.appendChild(option);
                    });
                    this.select.disabled = false;

                    if (sessionId) {
                        this.select.value = sessionId;
                    }
                } else {
                    this.select.disabled = true;
                }
                return data;
            } else {
                console.error('세션 목록 로드 실패:', response.statusText);
                return [];
            }
        } catch (error) {
            console.error('세션 목록 로드 실패:', error);
            return [];
        }
    }

    async loadSessionResults(sessionId) {
        if (!this.globalState.sportSelect || !this.globalState.sessionId) {
            return null;
        }

        try {
            const response = await fetch(`/api/score?session_id=${sessionId}`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                this.globalState.sessionResults = data;
                return data;
            } else {
                console.error('세션 결과 로드 실패:', response.statusText);
                return null;
            }
        } catch (error) {
            console.error('세션 결과 로드 실패:', error);
            return null;
        }
    }
}