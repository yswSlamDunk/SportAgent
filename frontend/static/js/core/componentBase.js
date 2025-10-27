export class ComponentBase {
    constructor(eventHandler, globalState) {
        this.eventHandler = eventHandler;
        this.globalState = globalState;
        this.element = null;
    }

    createElement(tag, className) {
        const element = document.createElement(tag);
        if (className) {
            element.className = className;
        }
        return element;
    }

    addListener(eventType, handler) {
        this.eventHandler.on(eventType, handler.bind(this));
    }
}