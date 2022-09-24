#!/usr/bin/env python

import dlib
import numpy as np
import cv2
import copy 

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    #coords = []
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
        #coords.append((shape.part(i).x, shape.part(i).y))
    # return the list of (x, y)-coordinates
    return coords


img = cv2.imread("bradley_cooper.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread("jim_carrey.png")
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#vid=cv2.VideoCapture('kobe.mp4')
vid = cv2.VideoCapture(0)

# We load Face detector and Face landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Face 1
#rects = detector(img_gray, 1)

def facial_landmarks(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img_gray = cv2.resize(img_gray, (500,500))
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(img_gray, rect)
        shape = shape_to_np(shape)
        print("Shape: = ", shape[:][0])
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        # (x, y, w, h) = rect_to_bb(rect)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # # show the face number
        # cv2.putText(img, "Face #{}".format(i + 1), (x - 10, y - 10),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

    return img


while(0):
    rects = detector(img_gray, 1)

    img = cv2.imread("jim_carrey.png")
    img = facial_landmarks(img)
    cv2.imshow("Output", img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
         break


while(1):
    ret, img_vid = vid.read()
    rects = detector(img_vid, 1)
    # img = copy.deepcopy(img_vid)

    img_src = img[rects[0].top()-40:rects[0].bottom()+40,rects[0].left()-40:rects[0].right()+40,:]

    if ret == True:
        #Find features in the image using dlib
        img = facial_landmarks(img_vid)
        cv2.imshow("Output", img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


# show the output image with the face detections + facial landmarks






















# # Delaunay triangulation
# rect = cv2.boundingRect(convexhull)
# subdiv = cv2.Subdiv2D(rect)
# subdiv.insert(landmarks_points)
# triangles = subdiv.getTriangleList()
# triangles = np.array(triangles, dtype=np.int32)