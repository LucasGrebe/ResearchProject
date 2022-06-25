#!/usr/bin/env python

"""
Research Project

"""

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import imutils
import mediapipe as mp
import numpy as np

from preset.common import clock

# local modules

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils


def counter(count, threshold):
    count += 1
    if count >= threshold:
        return 0
    return count


# def draw_rects(img, rects, color):
#     for x1, y1, x2, y2 in rects:
#         cv.rectangle(img, (x1, y1), (x2, y2), color, 2)


def main():
    import sys, getopt

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    # cascade_fn = args.get('--cascade', "data/haarcascades/haarcascade_frontalface_alt.xml")
    # nested_fn = args.get('--nested-cascade', "data/haarcascades/haarcascade_eye.xml")
    #
    # cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))
    # nested = cv.CascadeClassifier(cv.samples.findFile(nested_fn))

    cam = cv.VideoCapture(0)
    threshold_value = 130
    count = 0
    drawnContour = 0
    recalibrate_counter = 9900
    base_width = -1
    base_height = -1
    width = height = base_begin_point = base_end_point = begin_point = end_point = 0
    target_found = 0
    temp = 0
    points = []
    while True:
        count = counter(count, 1)
        # if not count:
        #     threshold_value += 1
        #     print("Threshold Value:", threshold_value)
        t = clock()
        _ret, img = cam.read()
        img = cv.imread("Easy.jpg")
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower_red = np.array([160, 140, 50])
        upper_red = np.array([180, 255, 255])

        imgThreshHigh = cv.inRange(hsv, lower_red, upper_red)
        thresh = imgThreshHigh.copy()

        contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            M = cv.moments(cnt)
            if M['m00'] != 0:
                cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
                coord = cx, cy  # This are your coordinates for the circle
                # area = moments['m00'] save the object area
                # perimeter = cv2.arcLength(best_cnt,True) is the object perimeter

                # Save the coords every frame on a list
                # Here you can make more conditions if you don't want repeated coordinates
                points.append(coord)

                cv.imshow('frame', img)
                cv.imshow('Object', thresh)
        cv.waitKey(0)

        if not _ret:
            continue
        # print(results.pose_landmarks)

        # convert the resized image to grayscale, blur it slightly,
        # and threshold it
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        th3 = cv.threshold(blurred, threshold_value, 255, cv.THRESH_BINARY)[1]
        # find contours in the thresholded image and initialize the
        # shape detector
        contours = cv.findContours(th3.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(contours)
        # print("contours", contours[0], "\n")
        # print("cnts", cnts)
        hierarchy = contours[1]
        # print(hierarchy)
        numberContour = 0
        arrayOfContours = []
        centerMoments = []
        numberOfContoursDrawn = 0
        layersDeep = 0
        for c in contours[0]:
            numberContour += 1
            # print("c", c)
            # print("Contour: ", c)
            # compute the center of the contour, then detect the name of the
            # shape using only the contour
            M = cv.moments(c)
            if M['m00'] != 0:
                cX = int((M["m10"] / M["m00"]))
                cY = int((M["m01"] / M["m00"]))
            else:
                cX = -0.5
                cY = -0.5
            # multiply the contour (x, y)-coordinates by the resize ratio,
            # then draw the contours and the name of the shape on the image
            c = c.astype("float")
            # c *= ratio
            c = c.astype("int")
            arrayOfContours.append(c)
            centerMoments.append([cX, cY])

        # Meat and Potatoes
        # #Distance of every center moment to the center of the screen to make a reference point.

        # print(numberContour, "Contours")
        cv.drawContours(img, arrayOfContours, -1, (255, 0, 255), 2, cv.LINE_AA)
        for i in range(numberContour):
            if hierarchy[0][i][0] == -1 and hierarchy[0][i][1] == -1 and hierarchy[0][i][2] == -1 \
                    and hierarchy[0][i][3] != -1:
                # print("Hierarchy of Contour\n", i, hierarchy)
                if not temp:
                    print("Center of Contour", i, centerMoments[i])

                cv.drawContours(img, arrayOfContours[i], -1, (0, 255, 0), 20, cv.LINE_AA)
                if centerMoments[i][0] >= 0:
                    cv.putText(img, (str(centerMoments[i][0]) + ', ' + str(centerMoments[i][1])), centerMoments[i], cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
                # numberOfContoursDrawn += 1
                # print("numberOfContoursDrawn", numberOfContoursDrawn)
        # print("Center Moments", centerMoments)
        temp += 1
        cv.imshow('shapedetect', img)
        cv.imshow('threshold', th3)

        if cv.waitKey(5) == 'q':
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
