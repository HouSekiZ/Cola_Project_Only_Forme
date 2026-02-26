// camera.js – CameraManager: list cameras, switch camera

class CameraManager {
    constructor() {
        this.currentIndex = 0;
        this.select = document.getElementById('cameraSelect');
        this.btnSwitch = document.getElementById('btnSwitchCam');
        this.statusEl = document.getElementById('camStatus');
        this.videoStream = document.getElementById('videoStream');

        this.btnSwitch.addEventListener('click', () => this.switchCamera());
        this.select.addEventListener('change', () => this.onSelectChange());

        this.loadCameras();
    }

    async loadCameras() {
        DOM.setStatus(this.statusEl, 'loading', 'กำลังค้นหากล้อง...');

        try {
            const cameras = await retryWithBackoff(() => API.get('/api/cameras'));

            this.select.innerHTML = '';

            if (!cameras.length) {
                this.select.innerHTML = '<option value="">ไม่พบกล้อง</option>';
                DOM.setStatus(this.statusEl, 'error', 'ไม่พบกล้องในระบบ');
                return;
            }

            cameras.forEach(cam => {
                const opt = document.createElement('option');
                opt.value = cam.index;
                opt.textContent = cam.name;
                if (cam.index === this.currentIndex) opt.selected = true;
                this.select.appendChild(opt);
            });

            DOM.enable(this.btnSwitch);
            DOM.setStatus(this.statusEl, 'success', `พบ ${cameras.length} กล้อง`);

        } catch (error) {
            console.error('Load cameras failed:', error);
            DOM.setStatus(this.statusEl, 'error', 'โหลดกล้องล้มเหลว กรุณารีโหลดหน้า');
        }
    }

    onSelectChange() {
        const selected = parseInt(this.select.value);
        if (selected !== this.currentIndex) {
            DOM.enable(this.btnSwitch);
            DOM.setStatus(this.statusEl, '', '');
        }
    }

    async switchCamera() {
        const index = parseInt(this.select.value);
        if (index === this.currentIndex) return;

        DOM.disable(this.btnSwitch);
        DOM.show(document.getElementById('loadingOverlay'));
        DOM.setStatus(this.statusEl, 'loading', 'กำลังเปลี่ยนกล้อง...');

        try {
            const result = await API.post('/api/select_camera', { index });

            if (result.status === 'ok') {
                this.currentIndex = index;
                // Reload video stream with cache-busting
                this.videoStream.src = `/video_feed?t=${Date.now()}`;
                DOM.setStatus(this.statusEl, 'success', `ใช้งานกล้อง ${index} แล้ว`);
            } else {
                throw new Error(result.message || 'Switch failed');
            }

        } catch (error) {
            console.error('Switch camera failed:', error);
            DOM.setStatus(this.statusEl, 'error', `เปลี่ยนกล้องล้มเหลว: ${error.message}`);
            DOM.enable(this.btnSwitch);
        } finally {
            DOM.hide(document.getElementById('loadingOverlay'));
        }
    }
}
