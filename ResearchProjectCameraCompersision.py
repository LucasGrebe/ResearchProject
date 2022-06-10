#!/usr/bin/env python

"""
Research Project

"""

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import imutils
import mediapipe as mp

from common import clock, draw_str

# local modules

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils


def find_edge_points(contour):
    right_most = bottom_most = -1
    left_most = top_most = 10000
    for pos in contour:
        pos = pos[0]
        if pos[0] < left_most:
            left_most = pos[0]
        if pos[0] > right_most:
            right_most = pos[0]
        if pos[1] < top_most:
            top_most = pos[1]
        if pos[1] > bottom_most:
            bottom_most = pos[1]
    return (left_most,  top_most), (right_most, bottom_most), right_most - left_most, bottom_most - top_most


def find_orientation(width, height, base_width, base_height, base_begin_point, base_end_point):
    x_direction = 0
    y_direction = 0

    return (base_width - width), (base_height - height)


def detect(c):
    # initialize the shape name and approximate the contour
    shape = "unidentified"
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.04 * peri, True)

    # if the shape is a triangle, it will have 3 vertices
    if len(approx) == 3:
        shape = "triangle"
    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv.boundingRect(approx)
        ar = w / float(h)
        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
    # if the shape is a pentagon, it will have 5 vertices
    elif len(approx) == 5:
        shape = "pentagon"
    # otherwise, we assume the shape is a circle
    else:
        shape = "circle"
    # return the name of the shape
    return shape


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

    cam = cv.VideoCapture(1)
    threshold_value = 120
    count = 0
    drawnContour = 0
    recalibrate_counter = 9900
    base_width = -1
    base_height = -1
    width = height = base_begin_point = base_end_point = begin_point = end_point = 0
    target_found = 0
    while True:
        count = counter(count, 20)
        # if not count:
        #     threshold_value += 1
        #     print("Threshold Value:", threshold_value)
        t = clock()
        _ret, img = cam.read()
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        # print(results.pose_landmarks)

        # convert the resized image to grayscale, blur it slightly,
        # and threshold it
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        th3 = cv.threshold(blurred, threshold_value, 255, cv.THRESH_BINARY)[1]
        # find contours in the thresholded image and initialize the
        # shape detector
        contours = cv.findContours(th3.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(contours)
        # print("contours", contours[0], "\n")
        # print("cnts", cnts)
        hierarchy = contours[1]
        # print(hierarchy)

        cx = -0.5
        cy = -0.5
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                if id == 12:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id, cx, cy, lm.z)
                    cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
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
            if M["m00"] == 0:
                cX = int(M["m10"])
                cY = int(M["m01"])
            else:
                cX = int((M["m10"] / M["m00"]))
                cY = int((M["m01"] / M["m00"]))
            shape = detect(c)
            # multiply the contour (x, y)-coordinates by the resize ratio,
            # then draw the contours and the name of the shape on the image
            c = c.astype("float")
            # c *= ratio
            c = c.astype("int")
            arrayOfContours.append(c)
            centerMoments.append([cX, cY])

        # Meat and Potatoes
        # #Distance of every center moment to the center of the screen to make a reference point.

        # How many Layers deep is the center circle

        # print(numberContour, "Contours")
        cv.drawContours(img, arrayOfContours, -1, (255, 0, 255), 2, cv.LINE_AA)
        for i in range(numberContour):
            if hierarchy[0][i][0] == -1 and hierarchy[0][i][1] == -1 and hierarchy[0][i][2] == -1 \
                    and hierarchy[0][i][3] != -1:
                # print("Hierarchy of Contour\n", i, hierarchy)
                # print("Center of Contour", i, centerMoments[i])

                cv.drawContours(img, arrayOfContours[i], -1, (0, 255, 0), 20, cv.LINE_AA)  # color green
                numberOfContoursDrawn += 1
                cv.putText(img, (cX - cx).__str__() + "," + (cY - cy).__str__() + "," + shape,
                           (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        # print("numberOfContoursDrawn", numberOfContoursDrawn)
        # print("Center Moments", centerMoments)
        for i in range(numberContour):
            if hierarchy[0][i][0] == -1 and hierarchy[0][i][1] == -1 and hierarchy[0][i][2] == -1 \
                    and hierarchy[0][i][3] != -1:
                parentValue = hierarchy[0][i][3]  # GetParent
                while parentValue != -1 and layersDeep < 8:  # Recursively get parent 7 times
                    parentValue = hierarchy[0][parentValue][3]
                    layersDeep += 1
                contour = arrayOfContours[parentValue]
                # bp: topleft, ep: bottomRight, width: Right - Left, height: Bottom - Top
                begin_point, end_point, width, height = find_edge_points(contour)
                # cv.drawContours(img, contour, -1, (255, 0, 0), 20, cv.LINE_AA)  # Draw Blue outer contour
                cv.rectangle(img, begin_point, end_point, (255, 0, 0), 2)  # Draws Bounding Box
                x_orientation, y_orientation = find_orientation(width, height, base_width, base_height, base_begin_point
                                                                , base_end_point)
                draw_str(img, (20, 40), 'X Orientation: ' + x_orientation.__str__())
                draw_str(img, (20, 60), 'Y Orientation: ' + y_orientation.__str__())
                target_found = 1
        if target_found == 1:
            recalibrate_counter += 1
            if recalibrate_counter > 10000:
                recalibrate_counter = 0
                base_width = width
                base_height = height
                print("Recalibrated\nNew Base Width: ", base_width, "\nNew Base Height: ", base_height)
                z = 0
                base_begin_point = begin_point
                base_end_point = end_point
        target_found = 0
        # if not count:
        #     drawnContour += 1

        # print(arrayOfContours)
        dt = clock() - t
        draw_str(img, (20, 20), 'time: %.1f ms' % (dt * 1000))

        cv.imshow('shapedetect', img)
        cv.imshow('threshold', th3)

        if cv.waitKey(5) == 'q':
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
