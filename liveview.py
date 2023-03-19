import numpy as np
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
from tensorflow.keras.models import load_model
from time import sleep

model = load_model('secondtry.h5')
path = r'first_dataset\asl'

threshold1 = 0.8
threshold2 = 0.3

#define text constants
font = cv2.FONT_HERSHEY_SIMPLEX

img_places = [(50, 50), (50, 70), (50, 90), (50, 110), (50, 130)]

fontScale = 0.5
color = (255, 0, 0)
thickness = 2

# For webcam input:
def classify(image):
    one_hand = []
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        # print('not a hand')
        return []
    
    only_one = False
    for hand_landmarks in results.multi_hand_landmarks:
        if only_one:
           break
        only_one = True
        for i in hand_landmarks.landmark:
            one_hand.extend([i.x,
                            i.y,
                            i.z])
    # print(len(one_hand))
    one_hand = np.array(one_hand)
    predictions = []
    for idx, pred in enumerate(model.predict(np.reshape(one_hand, (1,) + one_hand.shape), verbose = False).tolist()[0]):
        predictions.append([idx, pred])
    predictions.sort(key = lambda x : x[1], reverse = True)

    if(predictions[0][1] > threshold1):
        return [predictions[0]]
    elif(predictions[0][1] > threshold2):
        res = []
        for i in range(5):
            res.append(predictions[i])
        return res
    else:
        # print('idk')
        return []



# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    classifications = classify(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.flip(image, 1)
    if(len(classifications) == 1):
        image = cv2.putText(image, f'Prediction 1: {classifications[0][0]} Confidence: {classifications[0][1]:.2f}', img_places[0], font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    elif(len(classifications) > 1):
        for i in range(len(classifications)):
            image = cv2.putText(image, f'Prediction {i+1}: {classifications[i][0]} Confidence: {classifications[i][1]:.2f}', img_places[i] , font, 
                        fontScale, color, thickness, cv2.LINE_AA)
    else:
       image = cv2.putText(image, f'No Hand', img_places[0], font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
