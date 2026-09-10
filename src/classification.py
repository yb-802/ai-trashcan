# classification.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import base64
import time

class Classifier:
    def __init__(self, model_path, labels, input_size=(224, 224)):
        self.model = load_model(model_path)
        self.labels = labels
        self.input_size = input_size

    def capture_and_classify(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None, None, None
        

        # 預測
        image = cv2.resize(frame, self.input_size)
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        preds = self.model.predict(image)
        pred_index = np.argmax(preds)
        pred_label = self.labels[pred_index]

        # 編碼圖片為 Base64
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        # 傳回畫面、分類結果、圖片
        return frame, pred_label, jpg_as_text

