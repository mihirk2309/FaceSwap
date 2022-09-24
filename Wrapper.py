#!/usr/bin/env python

import dlib
import numpy as np
import cv2
import copy 

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def facial_landmarks(img, detector, predictor):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_gray = cv2.resize(img_gray, (500,500))
    rects = detector(img_gray, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(img_gray, rect)
        shape = shape_to_np(shape)
        print("Shape: = ", shape[:][0])

        for (x, y) in shape:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

    return img




# while(0):
#     rects = detector(img_gray, 1)

#     img = cv2.imread("jim_carrey.png")
#     img = facial_landmarks(img)
#     cv2.imshow("Output", img)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#          break


# while(1):
#     ret, img_vid = vid.read()
#     rects = detector(img_vid, 1)
#     # img = copy.deepcopy(img_vid)

#     # img_src = img[rects[0].top()-40:rects[0].bottom()+40,rects[0].left()-40:rects[0].right()+40,:]

#     if ret == True:
#         #Find features in the image using dlib
#         img = facial_landmarks(img_vid)
#         cv2.imshow("Output", img)
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break

def main():
    print('faceswap begin...')
    #Define Face Detector and Predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')


    img = cv2.imread("bradley_cooper.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread("jim_carrey.png")
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    vid=cv2.VideoCapture('kobe.mp4')

    while(True):
        ret1,img_src = vid.read()

        # rects = detector(img_src,1)
        # img_source = copy.deepcopy(img_src)
        # img_source = img_source[rects[0].top()-40:rects[0].bottom()+40,rects[0].left()-40:rects[0].right()+40,:]
        img_target = img.copy()

        if ret1 == True:
            img1= facial_landmarks(img_src,detector,predictor)
            cv2.imshow("Output", img1)
            img2= facial_landmarks(img_target,detector, predictor)


            if cv2.waitKey(25) & 0xff==ord('q'):
                cv2.destroyAllWindows()
                break

    print('operation ended...')

    # plt.imshow(img_source)

if __name__ == '__main__':
    main()