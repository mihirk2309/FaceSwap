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

    rects = detector(img_gray, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(img_gray, rect)
        shape = shape_to_np(shape)

        for (x, y) in shape:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

    return img, shape



def draw_delaunay(img, subdiv):

    triangles = subdiv.getTriangleList();
    r = (0, 0, img.shape[1], img.shape[0])
    print("Number of triangles",len(triangles))

    for t in triangles :

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        cv2.line(img, pt1, pt2, (255, 255, 255), 1, 0, 0)
        cv2.line(img, pt2, pt3, (255, 255, 255), 1, 0, 0)
        cv2.line(img, pt3, pt1, (255, 255, 255), 1, 0, 0)

    return img



def swap_faces(img_source, img_target, detector, predictor):
    img_source_size = np.shape(img_source)
    img_target_size = np.shape(img_target)

    img_target_features,img_target_points = facial_landmarks(img_target.copy(),detector, predictor)
    img_source_features,img_source_points = facial_landmarks(img_source.copy(),detector, predictor)

    size = np.shape(img_target)
    img_target_rect = (0,0,size[1],size[0])
    size = np.shape(img_source)
    img_source_rect = (0,0,size[1],size[0])

    img_target_subdiv  = cv2.Subdiv2D(img_target_rect)
    img_source_subdiv  = cv2.Subdiv2D(img_source_rect)

    img_target_points_tuple = tuple(map(tuple, img_target_points))

    for point in img_target_points_tuple:
        img_target_subdiv.insert(point)

    show_img = draw_delaunay(img_target.copy(), img_target_subdiv);
    cv2.imshow("1", show_img)

    img_source_points_tuple = tuple(map(tuple, img_source_points))

    for point in img_source_points_tuple:
        img_source_subdiv.insert(point)
    show_img = draw_delaunay(img_source.copy(), img_source_subdiv);
    cv2.imshow("2", show_img)

    return show_img



def main():
    print('faceswap begin...')

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
            img1, pt= facial_landmarks(img_src,detector,predictor)
            cv2.imshow("Output", img1)
            img2= facial_landmarks(img_target,detector, predictor)

            output1 = swap_faces(img_target.copy(), img_src.copy(), detector, predictor)
            cv2.imshow("Output", output1)

            if cv2.waitKey(25) & 0xff==ord('q'):
                cv2.destroyAllWindows()
                break

    print('operation ended...')

    # plt.imshow(img_source)

if __name__ == '__main__':
    main()