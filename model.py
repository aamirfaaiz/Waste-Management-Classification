import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class WasteManagementModel(object):

    CLASSES = ["Organic", "Recylable"]

    def __init__(self, model):
        # load model from JSON file
        loaded_model = tf.keras.models.load_model(
            model,
            custom_objects={'KerasLayer': hub.KerasLayer})

    def predict_waste_type(self, img_array):
        return self.CLASSES[int(np.argmax(self.loaded_model.predict(img_array),axis=-1))]

