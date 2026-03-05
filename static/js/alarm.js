// alarm.js – AlarmManager: polling, trigger, acknowledge

class AlarmManager {
    constructor() {
        this.isAlarming = false;
        this.sound = document.getElementById('alarmSound');
        this.overlay = document.getElementById('alarmOverlay');
        this.alarmMessage = document.getElementById('alarmMessage');
        this.btnAcknowledge = document.getElementById('btnAcknowledge');

        this.pollInterval = 1000;   // ms (ลดจาก 500ms → 1000ms)
        this.failCount = 0;
        this.maxFails = 5;
        this._backoffMs = 1000;     // exponential backoff start
        this._maxBackoffMs = 8000;  // backoff cap

        this.btnAcknowledge.addEventListener('click', () => this.acknowledge());
        this.startPolling();
    }

    startPolling() {
        const poll = async () => {
            try {
                const data = await API.get('/api/status');
                this.failCount = 0;
                this._backoffMs = this.pollInterval; // reset backoff เมื่อ success
                this._setConnectionLost(false);

                if (data.alarm && !this.isAlarming) {
                    this.trigger(data.alarm_type);
                }
            } catch (error) {
                this.failCount++;
                if (this.failCount >= this.maxFails) {
                    this._setConnectionLost(true);
                }
                // exponential backoff: 1s → 2s → 4s → 8s
                this._backoffMs = Math.min(this._backoffMs * 2, this._maxBackoffMs);
            }
            setTimeout(poll, this.failCount > 0 ? this._backoffMs : this.pollInterval);
        };
        setTimeout(poll, this.pollInterval);
    }

    trigger(type) {
        this.isAlarming = true;
        const messages = {
            eye_blink: 'ตรวจพบสัญญาณขอความช่วยเหลือด้วยการกระพริบตา',
            hand_gesture: 'ตรวจพบสัญญาณมือ (แบมือ → กำมือ)',
            default: 'ผู้ป่วยต้องการความช่วยเหลือ'
        };
        if (this.alarmMessage) {
            this.alarmMessage.textContent = messages[type] || messages.default;
        }
        DOM.show(this.overlay);
        this.sound.play().catch(() => {
            // Autoplay blocked by browser – user must interact first
        });
    }

    async acknowledge() {
        try {
            await API.post('/api/acknowledge');
            this.stop();
        } catch (error) {
            console.error('Acknowledge failed:', error);
            // Still stop locally even if server call fails
            this.stop();
        }
    }

    stop() {
        this.isAlarming = false;
        DOM.hide(this.overlay);
        this.sound.pause();
        this.sound.currentTime = 0;
    }

    _setConnectionLost(lost) {
        const el = document.getElementById('connectionStatus');
        if (!el) return;
        if (lost) {
            el.innerHTML = '<span class="dot dot-err"></span> ขาดการเชื่อมต่อ';
        } else {
            el.innerHTML = '<span class="dot dot-ok"></span> เชื่อมต่อแล้ว';
        }
    }
}
