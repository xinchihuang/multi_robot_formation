from flask import Flask, send_file
import time

app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    while True:
        time.sleep(0.1)
        return send_file('path_to_save_image/image.jpg', mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)