import numpy as np
import cv2
import time


# creating the videocapture object
# and reading from the input file
# Change it to 0 if reading from webcam

cap = cv2.VideoCapture("secuencia_a_cam2.avi")

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

# Reading the video file until finished
n = 0
while(cap.isOpened()):
    # time when we finish processing for this frame
    new_frame_time = time.time()
    total_time = (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(total_time)
    n=n+1

    # Capture frame-by-frame
    ret = cap.grab()
    if n%2!=0:
        continue
    ret, frame = cap.retrieve()

    # if video finished or no Video Input
    if not ret:
        break

    # Our operations on the frame come here
    gray = frame

    # resizing the frame size according to our need
    gray = cv2.resize(gray, (500, 300))

    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX


    # displaying the frame with fps
    cv2.imshow('frame', gray)

    # press 'Q' if you want to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# Destroy the all windows now
cv2.destroyAllWindows()
