import { getCurrentTime } from '../utils/utils.js';

export class EventHandler {
    constructor(globalState) {
        this.globalState = globalState;
        this.listeners = new Map();
    }

    on(eventName, callback) {
        if (!this.listeners.has(eventName)) {
            this.listeners.set(eventName, new Set());
        }
        this.listeners.get(eventName).add(callback);
    }

    off(eventName, callback) {
        if (this.listeners.has(eventName)) {
            this.listeners.get(eventName).delete(callback);
        }
    }

    emit(eventName, data, message) {
        if (this.listeners.has(eventName)) {
            this.listeners.get(eventName).forEach(callback => {
                try {
                    callback(data);
                    if (message !== undefined) {
                        this.addLog(message);
                    }
                } catch (error) {
                    this.addLog(`이벤트 처리 오류: ${error.message}`, 'error');
                }
            });
        }
    }

    addLog(message, type='log') {
        const timestamp = getCurrentTime();
        const log = `${timestamp} - ${message}\n`;
        if (type === 'log') {
            console.log(log);
        } else if (type === 'error') {
            console.error(log);
        }
    }

    changeGlobalState(key, value) {
        this.globalState[key] = value;
    }
}