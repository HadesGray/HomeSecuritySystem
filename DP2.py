import cv2
import os
import numpy as np
import datetime
import time
from PIL import Image, ImageDraw, ImageFont
import dlib
import urllib.request


#set camera reslution
response=urllib.request.urlopen("http://192.168.31.201/control?var=framesize&val=10")

# Load the face detector and create the predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("CV2Model\shape_predictor_68_face_landmarks.dat")


# Set up the video capture
cap = cv2.VideoCapture("http://192.168.31.201:81/stream")



#------------------set Motion Decection------------

_,first_frame=cap.read()
# Convert the first frame to grayscale
gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Save the first frame as a reference
prev_frame = gray

# Flag to keep track of whether an image has been saved
image_saved = False

#------------------set Motion Decection-------------


# Set the font and size for the timestamp
font = ImageFont.truetype("OCRAStd.otf", 30)

# Get the current time
start_time = time.time()

# Get the video dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Initialize the video writer
interval = 1800
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
out = cv2.VideoWriter("video\\face"+timestamp+".mp4", fourcc, 20.0, (frame_width, frame_height))


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


#------------------set Motion Decection------------------------------

    # Calculate the absolute difference between the current frame and the reference frame
    diff = cv2.absdiff(prev_frame, gray)

    # Threshold the difference image
    _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)


    # Dilate the difference image to fill in any gaps
    diff = cv2.dilate(diff, None, iterations=2)

    # Find contours in the difference image
    contours, _ = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue

        # Draw a bounding box around the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        timestamp2=int(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        if timestamp2%4==3:
            filename = "motion-detected-{}.jpg".format(timestamp2)
            cv2.imwrite("MotionDecect\\"+filename, frame)

        
        break

    prev_frame = gray

#------------------Set Motion Decection------------------------------




    # Detect faces
    faces = detector(gray)

    # Loop through the faces
    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        # Save the first face that is detected
        cv2.imwrite("face\\face"+timestamp+".jpg", frame)
        break


    frame=cv2.flip(frame,1)
    
    
    #add time on the video

    pil_image=Image.fromarray(frame)
    # Get the current time and convert it to a string
    current_time = timestamp
    # Create a draw object for the PIL image
    draw = ImageDraw.Draw(pil_image)
     # Write the time string on the PIL image
    draw.text((20, 50), current_time, font=font, fill=(255, 255, 255))
     # Convert the PIL image back to a numpy array
    frame = np.array(pil_image)

#add time on the video

# Create the video writer if it doesn't exist
    if time.time() - start_time >= interval:
        out.release()
        out = cv2.VideoWriter("video\\face"+timestamp+".mp4", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        start_time = time.time()

    out.write(frame)


    # Display the resulting frame
    cv2.imshow("Door Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()