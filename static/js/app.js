// app.js – PatientAssistApp: main application logic

class PatientAssistApp {
    constructor() {
        this.currentMode = 'ALL';   // default: โหมดรวม

        this.btnAll  = document.getElementById('btnAll');
        this.btnEye  = document.getElementById('btnEye');
        this.btnHand = document.getElementById('btnHand');
        this.btnBody = document.getElementById('btnBody');
        this.modeStatusEl = document.getElementById('modeStatus');
        this.fpsBadge     = document.getElementById('fpsBadge');

        // Sub-managers (alarm.js, camera.js)
        this.alarmManager  = new AlarmManager();
        this.cameraManager = new CameraManager();

        // Mode buttons
        this.btnAll.addEventListener('click',  () => this.setMode('ALL'));
        this.btnEye.addEventListener('click',  () => this.setMode('EYE'));
        this.btnHand.addEventListener('click', () => this.setMode('HAND'));
        this.btnBody.addEventListener('click', () => this.setMode('BODY'));

        // Reposition reset buttons
        const btnReset     = document.getElementById('btnRepositionReset');
        const btnAlertDone = document.getElementById('btnRepositionDone');
        if (btnReset)     btnReset.addEventListener('click',     () => this.onRepositionDone());
        if (btnAlertDone) btnAlertDone.addEventListener('click', () => this.onRepositionDone());

        // Notifications module (notifications.js)
        if (window.NotificationsModule) {
            window.NotificationsModule.init();
        }

        // Apply initial UI state
        this.updateModeUI();

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
                    ALL:  '🔄 โหมดรวม',
                    EYE:  '👁️ โหมดดวงตา',
                    HAND: '✋ โหมดมือ',
                    BODY: '🛏️ โหมดร่างกาย'
                };
                DOM.setStatus(this.modeStatusEl, 'success',
                    `เปลี่ยนเป็น ${labels[mode] || mode} แล้ว`);
            }
        } catch (error) {
            console.error('Set mode failed:', error);
            DOM.setStatus(this.modeStatusEl, 'error', 'เปลี่ยนโหมดล้มเหลว');
        }
    }

    updateModeUI() {
        // ── ปุ่ม active ──
        [
            { btn: this.btnAll,  mode: 'ALL'  },
            { btn: this.btnEye,  mode: 'EYE'  },
            { btn: this.btnHand, mode: 'HAND' },
            { btn: this.btnBody, mode: 'BODY' },
        ].forEach(({ btn, mode }) => {
            if (btn) btn.classList.toggle('active', this.currentMode === mode);
        });

        // ── Instructions ──
        const instrAll  = document.getElementById('instructionAll');
        const instrEye  = document.getElementById('instructionEye');
        const instrHand = document.getElementById('instructionHand');
        const instrBody = document.getElementById('instructionBody');

        // ซ่อนทั้งหมดก่อน
        [instrAll, instrEye, instrHand, instrBody].forEach(el => el && DOM.hide(el));

        // แสดงตาม mode
        if (this.currentMode === 'ALL') {
            // โหมดรวม: แสดงทุก instruction
            [instrAll, instrEye, instrHand, instrBody].forEach(el => el && DOM.show(el));
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

        // ── Mode label ──
        const modeLabel = document.getElementById('currentMode');
        if (modeLabel) {
            const names = { ALL: 'ALL (รวม)', EYE: 'EYE', HAND: 'HAND', BODY: 'BODY' };
            modeLabel.textContent = names[this.currentMode] || this.currentMode;
        }
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
                    statusEl.textContent = '✓ ระบบพร้อมใช้งาน';
                    statusEl.className = 'status-value text-success';
                }

                if (this.fpsBadge && health.metrics) {
                    this.fpsBadge.textContent = `${health.metrics.fps} FPS`;
                }

                // Sync mode จาก server (กัน desync)
                if (health.metrics &&
                    health.metrics.current_mode !== this.currentMode) {
                    this.currentMode = health.metrics.current_mode;
                    this.updateModeUI();
                }

                const connEl = document.getElementById('connectionStatus');
                if (connEl) {
                    connEl.innerHTML = '<span class="dot dot-ok"></span> เชื่อมต่อแล้ว';
                }

            } catch (error) {
                const statusEl = document.getElementById('systemStatus');
                if (statusEl) {
                    statusEl.textContent = '✗ ไม่สามารถเชื่อมต่อได้';
                    statusEl.className = 'status-value text-danger';
                }
                const connEl = document.getElementById('connectionStatus');
                if (connEl) {
                    connEl.innerHTML = '<span class="dot dot-error"></span> ขาดการเชื่อมต่อ';
                }
            }

            setTimeout(poll, 3000);
        };

        setTimeout(poll, 500);
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
});