import { ComponentBase } from '../core/componentBase.js';

export class AnalyticsFigure extends ComponentBase {
    constructor(eventHandler, globalState) {
        super(eventHandler, globalState);
        this.init();
    }

    init() {
        this.addListener('createFigure', this.createFigure.bind(this));
        this.addListener('preprocessingScore', this.preprocessingScore.bind(this));
    }

    createFigure() {
        // 기존 관련 요소 제거.
        const analyticsFigureTimeline = document.querySelector('.analytics-figure-timeline');
        analyticsFigureTimeline.style.backgroundColor = 'var(--light-gray)';

        const existingUserContainer = document.querySelector('.user-cell-container');
        const existingStandardContainer = document.querySelector('.standard-cell-container');
        if (existingUserContainer) existingUserContainer.remove();
        if (existingStandardContainer) existingStandardContainer.remove();

        // 데이터 처리
        const analyticsData = this.globalState.sessionResults.analytics_data;        
        const shortestScoring = analyticsData.shortest_scoring;
        const shortestPath = JSON.parse(analyticsData.shortest_path);

        // 새로운 컨테이너 생성
        this.createUserCellContainer(analyticsFigureTimeline, shortestPath, shortestScoring);
        this.createStandardCellContainer(analyticsFigureTimeline, shortestPath, shortestScoring);

        this.syncScroll(this.userCellContainer, this.standardCellContainer);
    }

    createStandardCellContainer(analyticsFigureTimeline, shortestPath, shortestScoring) {
        this.standardCellContainer = this.createElement('div', 'standard-cell-container');
        analyticsFigureTimeline.appendChild(this.standardCellContainer);

        // shortestPath에서 standard 좌표 Max값 찾기
        const standardMax = shortestPath[shortestPath.length - 1][0]; 
        for (let i = 0; i <= standardMax; i++) {
            const standardCell = this.createElement('div', 'standard-cell');
            standardCell.textContent = i;
            this.standardCellContainer.appendChild(standardCell);
        }
    }

    createUserCellContainer(analyticsFigureTimeline, shortestPath, shortestScoring) {
        this.userCellContainer = this.createElement('div', 'user-cell-container');
        analyticsFigureTimeline.appendChild(this.userCellContainer);
    
        // 표준 프레임별로 그룹화
        const frameGroups = {};
        shortestPath.forEach((point, index) => {
            const [standardFrame, userFrame] = point;
            if (!frameGroups[standardFrame]) {
                frameGroups[standardFrame] = [];
            }
            frameGroups[standardFrame].push({
                userFrame: userFrame,
                score: index < shortestScoring.length ? shortestScoring[index] : 0
            });
        });
    
        // 최대 표준 프레임
        const standardMax = shortestPath[shortestPath.length - 1][0];
    
        // 각 표준 프레임별로 그룹 생성
        for (let i = 0; i <= standardMax; i++) {
            const group = this.createElement('div', 'user-cell-group');
            const userFrames = frameGroups[i] || [];
            
            userFrames.forEach(({userFrame, score}) => {
                const userCell = this.createElement('div', 'user-cell');
                userCell.style.backgroundColor = this.getScoreColor(score);
                userCell.textContent = userFrame;
                userCell.title = `[${i}, ${userFrame}] Score: ${score.toFixed(2)}`;
    
                userCell.addEventListener('click', () => {
                    this.eventHandler.emit('showFrameImages', {
                        standardFrame: i,
                        userFrame: userFrame
                    });
                });
    
                group.appendChild(userCell);
            });
            
            this.userCellContainer.appendChild(group);
        }
    }

    syncScroll(userContainer, standardContainer) {
        standardContainer.addEventListener('scroll', () => userContainer.scrollLeft = standardContainer.scrollLeft);
    }

    getScoreColor(score) {
        const normalizedScore = Math.min(Math.max(score / 10, 0), 1);
        const hue = (1 - normalizedScore) * 120;
        return `hsl(${hue}, 70%, 50%)`;
    }

    preprocessingScore() {
        const analyticsData = this.globalState.sessionResults?.analytics_data;
        if (!analyticsData?.shortest_scoring) {
            console.error('Analytics data가 없습니다.');
            return;
        }
        
        const parsed = JSON.parse(analyticsData.shortest_scoring);
        const diffArray = [parsed[0]];
        
        for (let i = 0; i < parsed.length - 1; i++) {
            diffArray.push(parsed[i + 1] - parsed[i]);
        }
        
        this.globalState.sessionResults.analytics_data.shortest_scoring = diffArray;
    }
}

export class AnalyticsImages extends ComponentBase {
    constructor(eventHandler, globalState) {
        super(eventHandler, globalState);
        this.init();
    }

    init() {
        this.addListener('createImages', this.createImages.bind(this));
        this.addListener('showFrameImages', this.showFrameImages.bind(this));
    }

    createImages() {
        // analytics-images 컨테이너 확인
        const analyticsContainer = document.querySelector('.analytics-images-content');
        if (!analyticsContainer) {
            console.error('analytics-images 컨테이너를 찾을 수 없습니다.');
            return;
        }
        
        // 기존 내용 제거
        analyticsContainer.innerHTML = '';
        

        this.showFrameImages({
            standardFrame: 0,
            userFrame: 0
        });
    }

    async showFrameImages(data) {
        const { standardFrame, userFrame } = data;
        
        // analytics-images 컨테이너 확인
        const analyticsContainer = document.querySelector('.analytics-images-content');
        if (!analyticsContainer) {
            console.error('analytics-images-content 컨테이너를 찾을 수 없습니다.');
            return;
        }
        
        // 기존 내용 제거
        analyticsContainer.innerHTML = '';
        
        // 비디오명 가져오기 (확장자 제외)
        const standardVideoName = this.getVideoName('standard');
        const userVideoName = this.getVideoName('user');
        
        // 표준 프레임 이미지
        const standardImg = this.createElement('img', 'standard-frame-image');
        standardImg.src = `/media/frames/standard/${standardVideoName}/success/${standardFrame}.jpg`;
        standardImg.alt = `Standard Frame ${standardFrame}`;
        standardImg.onerror = () => console.error('표준 프레임 이미지 로드 실패:', standardImg.src);
        
        // 사용자 프레임 이미지
        const userImg = this.createElement('img', 'user-frame-image');
        userImg.src = `/media/frames/user/${userVideoName}/success/${userFrame}.jpg`;
        userImg.alt = `User Frame ${userFrame}`;
        userImg.onerror = () => console.error('사용자 프레임 이미지 로드 실패:', userImg.src);
        
        // 컨테이너 없이 직접 추가
        analyticsContainer.appendChild(standardImg);
        analyticsContainer.appendChild(userImg);
    }

    getVideoName(type) {
        if (type === 'standard') {
            const standardVideoPath = this.globalState.selectedStandardVideoPath;
            if (standardVideoPath) {
                const fileName = standardVideoPath.split('/').pop() || standardVideoPath.split('\\').pop();
                return fileName.split('.')[0];
            }
        } else {
            const userVideoPath = this.globalState.selectedUserVideoPath;
            if (userVideoPath) {
                const fileName = userVideoPath.split('\\').pop();
                return fileName.split('.')[0]; // 확장자 제거
            }
        }
        return type === 'standard' ? 'standard_video' : 'user_video';
    }
}


export class AnalyticsTable extends ComponentBase {
    constructor(eventHandler, globalState) {
        super(eventHandler, globalState);
        this.init();
    }

    init() {
        this.addListener('createTable', this.createTable.bind(this));
    }

    createTable() {
        // 기존 테이블 제거
        const existingTable = document.querySelector('.analytics-table');
        if (existingTable) {
            existingTable.remove();
        }
        
        // sessionResults에서 body_part_scores 가져오기
        const sessionResults = this.globalState.sessionResults;
        if (!sessionResults || !sessionResults.body_part_scores) {
            console.error('Session results 또는 body_part_scores가 없습니다.');
            return;
        }
        
        const bodyPartScores = sessionResults.body_part_scores;
        
        // 테이블 생성
        const table = this.createElement('table', 'analytics-table');

        this.thead = this.createElement('thead');
        this.thead.innerHTML = `
            <tr>
                <th>부위</th>
                <th>평균 점수</th>
                <th>기준 점수</th>
                <th>기준 미달 여부</th>
            </tr>
        `;
        table.appendChild(this.thead);

        this.tbody = this.createElement('tbody');

        bodyPartScores.forEach((score, index) => {
            const row = this.createElement('tr');
            
            // 마지막 행인지 확인
            if (index === bodyPartScores.length - 1) {
                row.style.backgroundColor = 'var(--light-gray)';
            }
            
            // 부위명 셀 (한글명 사용)
            const bodyPartCell = this.createElement('td');
            const koreanName = score.body_part_korean || score.body_part;
            bodyPartCell.textContent = koreanName;
            row.appendChild(bodyPartCell);
            
            // 평균 점수 셀
            const averageScoreCell = this.createElement('td');
            averageScoreCell.textContent = score.average_score.toFixed(2);
            row.appendChild(averageScoreCell);
            
            // 기준 점수 셀
            const standardScoreCell = this.createElement('td');
            const standardScore = score.standard_score || 0;
            standardScoreCell.textContent = standardScore.toFixed(2);
            row.appendChild(standardScoreCell);
            
            // 기준 미달 여부 셀
            const belowStandardCell = this.createElement('td');
            
            const averageScore = score.average_score || 0;
            const scoreDifference = Math.abs(standardScore - averageScore);
            const threshold = standardScore * 0.05;
            
            let statusText;
            let statusColor;
            
            if (score.is_below_standard) {
                statusText = '기준 미달';
                statusColor = '#dc3545';
            } else if (scoreDifference <= threshold) {
                statusText = '주의 필요';
                statusColor = '#fd7e14';
            } else {
                statusText = '기준 달성';
                statusColor = '#28a745';
            }
            
            belowStandardCell.textContent = statusText;
            belowStandardCell.style.color = statusColor;
            belowStandardCell.style.fontWeight = 'bold';
            
            row.appendChild(belowStandardCell);
            this.tbody.appendChild(row);
        });
        
        table.appendChild(this.tbody);
        
        // DOM에 추가
        const analyticsWrapper = document.querySelector('.analytics-table-wrapper');
        if (analyticsWrapper) {
            analyticsWrapper.appendChild(table);
        } else {
            console.error('analytics-table-wrapper를 찾을 수 없습니다.');
        }
    }
}