from threading import Thread

from flask_cors import CORS, cross_origin
from py_pipe.pipe import Pipe
from transformation import Transformation
from TFObjectDetector import TFObjectDetector
from py_flask_movie.flask_movie import FlaskMovie
from flask import Flask, render_template, request, jsonify

import os
import cv2
import config
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
app = Flask(__name__)
CORS(app)
# fs = FlaskMovie(app)

live = False
if live:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture("data/videos/security.mp4")
while True:
    ret, img = cap.read()
    image=cv2.resize(img, (640, 480))
    if ret:
        break

detection = TFObjectDetector(config.SAVED_MODEL_PATH, config.LABEL_MAP_PATH, image.shape,  True)
detector_ip = detection.get_in_pipe()
detector_op = detection.get_out_pipe()
detection.run()
transform = Transformation()

def read(cap,cam_id):

    detections_on_perspective = False
    inference_image = True

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            cap = cv2.VideoCapture(config.CAM_DATA[cam_id])
            ret1, image = cap.read()

        image = cv2.resize(image,(640,480))

        if detections_on_perspective:
            image = transform.get_perspective_transform(image)

        if inference_image:
            detector_ip.push_wait()
            inference = TFObjectDetector.Inference(image, config.LABEL_MAP_PATH, meta_dict={"cam_id":cam_id})
            detector_ip.push(inference)
violations = 0
person_count = 0
def run(pipe, cam_id):
    print("in run: ", cam_id)
    global violations, person_count
    while True:
        detector_op.pull_wait()
        ret, inference = detector_op.pull(True)
        if ret:
            i_dets = inference.get_result()
            item_detections = i_dets.get_boxes_tlbr(normalized=False).astype(np.int)
            item_scores = i_dets.get_scores()
            if cam_id == inference.get_meta_dict()["cam_id"]:
                image_hd, violations, person_count = transform.check_distance(i_dets.get_image(), item_scores, item_detections)
                pipe.push_wait()
                pipe.push(image_hd)

@app.route('/')
def index():
    #only by sending this page first will the client be connected to the socketio instance
    return render_template('index1.html')

@app.route('/violations', methods=['GET'])
@cross_origin(origin='*')
def get_violations():
    global violations
    print("violations = ",violations)
    return str(violations)

@app.route('/person_count', methods=['GET'])
@cross_origin(origin='*')
def get_person_count():
    global person_count
    print("person_count = ",person_count)
    return str(person_count)

@app.route("/feed/input")
@cross_origin(origin='*')
def get_input_feed():

    args = request.args
    cam_id = args["camId"]
    return jsonify(args)

@app.route("/feed/output")
@cross_origin(origin='*')
def get_input_feed():

    args = request.args
    cam_id = args["camId"]
    if cam_id not in config.CAM_DEPLOYMENT_STATUS:
        config.CAM_DEPLOYMENT_STATUS[cam_id] = "INITIALIZED"
        cap = cv2.VideoCapture(config.CAM_DATA[cam_id])
        pipe = Pipe()
        Thread(target=read, args=(cap, cam_id,)).start()
        # Thread(target=run, args=(pipe, cam_id)).start()

        return jsonify(args)
    else:
        return "camera feed already deployed"


# def main():
#
#     fs.start("0.0.0.0", 5000)
#
#     pipe1 = Pipe()
#     fs.create('feed_1', pipe1)
#
#     cap1 = cv2.VideoCapture("data/videos/vid_short.mp4")
#     meta_data_1 = {}
#     meta_data_1["cam_id"] = "video1"
#     meta_data_1["path"] = "data/videos/vid_short.mp4"
#

if __name__ == "__main__":
    # main()
    app.run(host="0.0.0.0",port=config.APPLICATION_PORT)
