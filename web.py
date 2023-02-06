import cv2
import numpy as np
import datetime
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Set the font and size for the timestamp
font = ImageFont.truetype("OCRAStd.otf", 30)

cap = cv2.VideoCapture("http://192.168.31.201:81/stream")


with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
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
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.detections:
      for detection in results.detections:
        mp_drawing.draw_detection(image, detection)

    

    # Flip the image horizontally for a selfie-view display.
    image=cv2.flip(image,1)


#add time on the video

    pil_image=Image.fromarray(image)
    # Get the current time and convert it to a string
    current_time = str(datetime.datetime.now())
    # Create a draw object for the PIL image
    draw = ImageDraw.Draw(pil_image)
     # Write the time string on the PIL image
    draw.text((20, 50), current_time, font=font, fill=(255, 255, 255))
     # Convert the PIL image back to a numpy array
    image = np.array(pil_image)

#add time on the video

    #cv2.putText(image, current_time, (20, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('MediaPipe Face Detection', image)
    if cv2.waitKey(5) & 0xFF == ord("q"):
      break
cap.release()