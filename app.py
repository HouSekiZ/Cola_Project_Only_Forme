from flask import Flask, render_template, Response, jsonify
from camera import VideoCamera

app = Flask(__name__)
camera_instance = VideoCamera()

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame, _ = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(camera_instance),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    status = camera_instance.alarm_active
    if status:
        camera_instance.alarm_active = False  # reset หลัง frontend อ่าน
    return jsonify({'alarm': status})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)