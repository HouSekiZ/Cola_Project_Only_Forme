// utils.js – API helpers, DOM helpers, retry logic

const API = {
    async get(endpoint) {
        const response = await fetch(endpoint);
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${endpoint}`);
        return response.json();
    },

    async post(endpoint, data = {}) {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${endpoint}`);
        return response.json();
    }
};

const DOM = {
    show(element) {
        if (element) element.classList.remove('hidden');
    },
    hide(element) {
        if (element) element.classList.add('hidden');
    },
    enable(element) {
        if (element) element.disabled = false;
    },
    disable(element) {
        if (element) element.disabled = true;
    },
    setStatus(element, type, message) {
        if (!element) return;
        element.className = `status-message ${type}`;
        element.textContent = message;
    }
};

async function retryWithBackoff(fn, maxRetries = 3, baseDelay = 1000) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            return await fn();
        } catch (error) {
            if (i === maxRetries - 1) throw error;
            const delay = baseDelay * Math.pow(2, i);
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
}
