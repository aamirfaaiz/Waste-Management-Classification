import cv2
from model import WasteManagementModel
import numpy as np
import tensorflow as tf

wastec = cv2.CascadeClassifier('') # need to find/build own haarcascade!
model = WasteManagementModel("waste_management_model_inception")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        wastes = wastec.detectMultiScale(fr, 1.3, 5)

        for (x, y, w, h) in wastes:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (299, 299))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #RGB Conversion
            rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.float32) #converting to tf tensor
            rgb_tensor = tf.expand_dims(rgb_tensor , 0) #changing dimensions to 4D
            pred = model.predict_waste_type(rgb_tensor)

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()