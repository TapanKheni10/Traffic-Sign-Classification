from TrafficSignRecognition.prediction_component.predict import PredictionPipeline
from TrafficSignRecognition import logger
import numpy as np
import cv2
import pandas as pd

data = pd.read_csv('research/traffic_sign.csv')

frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.9
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

while True:
    _, img = cap.read()

    img = np.asarray(img)
    image_to_predict = cv2.resize(img, (32, 32))
    logger.info('Shape of the image: {}'.format(image_to_predict.shape))
    logger.info('Data type of the image: {}'.format(image_to_predict.dtype))

    obj = PredictionPipeline(img = image_to_predict)
    prediction_prob, prediction_class = obj.predict()

    logger.info('Prediction class: {}'.format(prediction_class))
    logger.info('Prediction probability: {}'.format(prediction_prob))

    if prediction_prob >= threshold:
        predicted_sign = data[data['ClassId'] == prediction_class]['Name'].values[0]
        cv2.putText(img, str(predicted_sign), (20, 35), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, str(round(prediction_prob * 100, 2)), (20, 75), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
