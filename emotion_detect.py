#최종_감정인식 확인
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import imutils

detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'Emotion_Model_mini_XCEPTION.keras'

face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

cv2.namedWindow('Emotion_recognition')
camera = cv2.VideoCapture(0)

while True:
    frame = camera.read()[1]
    frame = imutils.resize(frame, width=1080)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    frameClone = frame.copy()

    for (fX, fY, fW, fH) in faces:
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
        emotion = EMOTIONS[np.argmax(preds)]

        text_x = fX + fW // 2
        text_y = fY - 10  # 위로 조금 올려서 텍스트가 얼굴 상단 중앙에 위치하도록 조정

        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (128, 128, 128), 5)
        cv2.putText(frameClone, emotion, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)

    cv2.imshow('Emotion_recognition', frameClone)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
