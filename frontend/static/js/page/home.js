import { checkAuth, getUserInfoFromToken } from "../utils/utils.js";

if (!checkAuth()) {
    // 토큰이 없으면 실행 중단
    window.location.href = '/login';
    throw new Error('Authentication required');
}

import { EventHandler } from '../core/eventHandler.js';
import { SendButton } from '../components/sendButton.js';
import { SportSelect, DateSelect } from '../components/select.js';
import { UploadModal } from '../components/modalUpload.js';
import { ModalVideo } from '../components/modalVideo.js';
import { ViewVideoModalButton, UploadVideoModalButton } from '../components/modalButton.js';
import { ModalVideoName } from '../components/modalText.js';
import { SelectStandardVideo, UserFileUpload } from '../components/modalSelect.js';
import { ModalText } from '../components/modalText.js';
import { AnalyticsFigure, AnalyticsImages, AnalyticsTable } from '../components/analyticsComponent.js';
import { HelpModal } from '../components/helpModal.js';

const globalState = {
    sportSelect: null,
    dateSelect: null,
    userInfo: getUserInfoFromToken(),
    
    selectedStandardVideo: null,
    selectedStandardVideoPath: null,
    selectedStandardVideoID: null,
    
    selectedUserVideo: null,
    selectedUserVideoPath: null,
    selectedUserVideoID: null,

    poseEstimation: null,
    poseScoring: null,
}

const eventHandler = new EventHandler(globalState);
const sportSelect = new SportSelect(eventHandler, globalState);
const dateSelect = new DateSelect(eventHandler, globalState);
const uploadModal = new UploadModal(eventHandler, globalState);
const sendButton = new SendButton(eventHandler, globalState);

const modalStandardVideo = new ModalVideo(eventHandler, globalState, 'standard');
const modalUserVideo = new ModalVideo(eventHandler, globalState, 'user');
const modalVideoName = new ModalVideoName(eventHandler, globalState);
const viewVideoModal = new ViewVideoModalButton(eventHandler, globalState);
const uploadVideoModal = new UploadVideoModalButton(eventHandler, globalState);
const selectStandardVideo = new SelectStandardVideo(eventHandler, globalState);
const userFileUpload = new UserFileUpload(eventHandler, globalState);
const modalText = new ModalText(eventHandler, globalState);
const analyticsFigure = new AnalyticsFigure(eventHandler, globalState);
const analyticsImages = new AnalyticsImages(eventHandler, globalState);
const analyticsTable = new AnalyticsTable(eventHandler, globalState);
const helpModal = new HelpModal(eventHandler, globalState);