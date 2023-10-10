import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

# cap = cv2.VideoCapture('../Resources/song.mp4') # for video
cap = cv2.VideoCapture(1)

# iterating over each frame
while True:
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # DETECT FACE:
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        roi_gray = gray[y: y + h, x: x + w]
        # roi_colored = img_resized[y: y + h, x: x + w]

    # DETECT EYES:
    eyes = eye_cascade.detectMultiScale(img)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 150, 0), 3)

    cv2.imshow("xyz", img)
    if cv2.waitKey(1) == 27:
        break

#           ***************NOTE***************
# Parameters in detectMultiScale:
# 1.ScaleFactor - This tells how much the object’s
# size is reduced in each image.
# 2.minNeighbors - This parameter tells how many neighbours each rectangle candidate
# should consider.
# 3.minSize - This signifies the minimum possible size of an object to be detected.
# An object smaller than minSize would be ignored.
