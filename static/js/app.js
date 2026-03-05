// app.js – PatientAssistApp: main application logic

class PatientAssistApp {
    constructor() {
        this.currentMode = 'ALL';   // default: โหมดรวม

        this.btnAll = document.getElementById('btnAll');
        this.btnEye = document.getElementById('btnEye');
        this.btnHand = document.getElementById('btnHand');
        this.btnBody = document.getElementById('btnBody');
        this.modeStatusEl = document.getElementById('modeStatus');
        this.fpsBadge = document.getElementById('fpsBadge');        // sidebar
        this.fpsBadgeVideo = document.getElementById('fpsBadgeVideo');   // video header

        // Sub-managers (alarm.js, camera.js)
        this.alarmManager = new AlarmManager();
        this.cameraManager = new CameraManager();

        // Mode buttons
        this.btnAll.addEventListener('click', () => this.setMode('ALL'));
        this.btnEye.addEventListener('click', () => this.setMode('EYE'));
        this.btnHand.addEventListener('click', () => this.setMode('HAND'));
        this.btnBody.addEventListener('click', () => this.setMode('BODY'));

        // Reposition reset buttons
        const btnReset = document.getElementById('btnRepositionReset');
        const btnAlertDone = document.getElementById('btnRepositionDone');
        if (btnReset) btnReset.addEventListener('click', () => this.onRepositionDone());
        if (btnAlertDone) btnAlertDone.addEventListener('click', () => this.onRepositionDone());

        // Notifications module (notifications.js)
        if (window.NotificationsModule) {
            window.NotificationsModule.init();
        }

        // Apply initial UI state
        this.updateModeUI();

        // Overlay Toggle Initial State
        this.fetchOverlayState();

        // Start health polling
        this.startHealthPolling();
    }

    // ── Mode ──────────────────────────────────────────────

    async setMode(mode) {
        if (mode === this.currentMode) return;

        try {
            const result = await API.post('/api/set_mode', { mode });
            if (result.status === 'ok') {
                this.currentMode = mode;
                this.updateModeUI();
                const labels = {
                    ALL: '🔄 โหมดรวม',
                    EYE: '👁️ โหมดดวงตา',
                    HAND: '✋ โหมดมือ',
                    BODY: '🛏️ โหมดร่างกาย'
                };
                DOM.setStatus(this.modeStatusEl, 'success',
                    `เปลี่ยนเป็น ${labels[mode] || mode} แล้ว`);

                // Fetch and sync new mode's overlay states
                this.fetchOverlayState();
            }
        } catch (error) {
            console.error('Set mode failed:', error);
            DOM.setStatus(this.modeStatusEl, 'error', 'เปลี่ยนโหมดล้มเหลว');
        }
    }

    updateModeUI() {
        // ── ปุ่ม active ──
        [
            { btn: this.btnAll, mode: 'ALL' },
            { btn: this.btnEye, mode: 'EYE' },
            { btn: this.btnHand, mode: 'HAND' },
            { btn: this.btnBody, mode: 'BODY' },
        ].forEach(({ btn, mode }) => {
            if (btn) btn.classList.toggle('active', this.currentMode === mode);
        });

        // ── Instructions (sidebar) ──
        const instrAll = document.getElementById('instructionAll');
        const instrEye = document.getElementById('instructionEye');
        const instrHand = document.getElementById('instructionHand');
        const instrBody = document.getElementById('instructionBody');

        [instrAll, instrEye, instrHand, instrBody].forEach(el => el && DOM.hide(el));

        if (this.currentMode === 'ALL') {
            DOM.show(instrAll);
        } else if (this.currentMode === 'EYE') {
            DOM.show(instrEye);
        } else if (this.currentMode === 'HAND') {
            DOM.show(instrHand);
        } else if (this.currentMode === 'BODY') {
            DOM.show(instrBody);
        }

        // ── Position panel: แสดงใน ALL และ BODY ──
        const posPanel = document.getElementById('positionPanel');
        if (posPanel) {
            const showPos = this.currentMode === 'ALL' || this.currentMode === 'BODY';
            posPanel.classList.toggle('hidden', !showPos);
        }

        // ── Mode label (sidebar badge) ──
        const modeLabel = document.getElementById('currentMode');
        if (modeLabel) {
            modeLabel.textContent = this.currentMode;
        }

        // ── Video mode tag ──
        const videoTag = document.getElementById('videoModeTag');
        if (videoTag) {
            const names = { ALL: 'ALL MODE', EYE: 'EYE MODE', HAND: 'HAND MODE', BODY: 'BODY MODE' };
            videoTag.textContent = names[this.currentMode] || this.currentMode;
        }

        // ── Overlay Toggles ──
        const toggleEye = document.getElementById('toggleOverlayEye');
        const toggleHand = document.getElementById('toggleOverlayHand');
        const toggleBody = document.getElementById('toggleOverlayBody');

        if (toggleEye) toggleEye.style.display = (this.currentMode === 'ALL' || this.currentMode === 'EYE') ? '' : 'none';
        if (toggleHand) toggleHand.style.display = (this.currentMode === 'ALL' || this.currentMode === 'HAND') ? '' : 'none';
        if (toggleBody) toggleBody.style.display = (this.currentMode === 'ALL' || this.currentMode === 'BODY') ? '' : 'none';
    }

    // ── Reposition ────────────────────────────────────────

    async onRepositionDone() {
        try {
            await API.post('/api/reset_position_timer', {});
            const alertEl = document.getElementById('repositionAlert');
            if (alertEl) alertEl.classList.add('hidden');
            if (window.NotificationsModule) {
                window.NotificationsModule.refreshPosition();
            }
        } catch (e) {
            console.error('Reset reposition timer failed:', e);
        }
    }

    // ── Health Polling ────────────────────────────────────

    startHealthPolling() {
        const poll = async () => {
            try {
                const health = await API.get('/api/health');
                const statusEl = document.getElementById('systemStatus');
                if (statusEl) {
                    statusEl.textContent = '✓ พร้อมใช้งาน';
                    statusEl.className = 'sys-value text-success';
                }

                // อัปเดต FPS ทั้ง sidebar + video header
                if (health.metrics) {
                    const fps = parseFloat(health.metrics.fps).toFixed(1);
                    if (this.fpsBadge) this.fpsBadge.textContent = `${fps} FPS`;
                    if (this.fpsBadgeVideo) this.fpsBadgeVideo.textContent = `${fps} FPS`;
                }

                // Sync mode จาก server
                if (health.metrics && health.metrics.current_mode !== this.currentMode) {
                    this.currentMode = health.metrics.current_mode;
                    this.updateModeUI();
                }

                const connEl = document.getElementById('connectionStatus');
                if (connEl) {
                    connEl.className = 'conn-badge conn-ok';
                    connEl.innerHTML = '<span class="dot dot-ok"></span> เชื่อมต่อแล้ว';
                }

            } catch (error) {
                const statusEl = document.getElementById('systemStatus');
                if (statusEl) {
                    statusEl.textContent = '✗ ไม่สามารถเชื่อมต่อ';
                    statusEl.className = 'sys-value text-danger';
                }
                const connEl = document.getElementById('connectionStatus');
                if (connEl) {
                    connEl.className = 'conn-badge conn-err';
                    connEl.innerHTML = '<span class="dot dot-err"></span> ขาดการเชื่อมต่อ';
                }
            }

            setTimeout(poll, 3000);
        };

        setTimeout(poll, 500);
    }

    // ── Overlay Toggles ───────────────────────────────────────

    async fetchOverlayState() {
        try {
            const data = await API.get('/api/overlay/state');
            if (data) {
                this.updateOverlayBtn('eye', data.eye);
                this.updateOverlayBtn('hand', data.hand);
                this.updateOverlayBtn('body', data.body);
            }
        } catch (e) {
            console.error("fetchOverlayState failed", e);
        }
    }

    async toggleOverlay(name) {
        try {
            const data = await API.post(`/api/overlay/${name}`, {});
            if (data && data.overlay) {
                this.updateOverlayBtn(data.overlay, data.visible);
            }
        } catch (e) {
            console.error(`toggleOverlay ${name} failed`, e);
        }
    }

    updateOverlayBtn(name, isVisible) {
        const btn = document.getElementById(`toggleOverlay${name.charAt(0).toUpperCase() + name.slice(1)}`);
        if (!btn) return;
        if (isVisible) {
            btn.classList.add('active');
            btn.style.opacity = '1';
        } else {
            btn.classList.remove('active');
            btn.style.opacity = '0.5';
        }
    }

    // ── Cleanup ───────────────────────────────────────────

    destroy() {
        if (window.NotificationsModule) {
            window.NotificationsModule.destroy();
        }
    }
}

// Bootstrap
document.addEventListener('DOMContentLoaded', () => {
    window.app = new PatientAssistApp();
    window.toggleOverlay = (name) => window.app.toggleOverlay(name);
});