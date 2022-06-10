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
import random

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


def find_dot(img, hierarchy, arrayOfContours):
    dot_height = 10000
    for j in img:
        if dot_height > j[0][1]:
            dot_height = j[0][1]
    return dot_height


def rotate_image(img, hierarchy, arrayOfContours):
    rows, cols, _ = img.shape
    x0, y0 = ((cols - 1) / 2.0, (rows - 1) / 2.0)
    img2 = np.zeros_like(img)
    highest = []
    rotation_degrees = -1
    for i in range(0, 360, 25):
        img1 = np.zeros_like(img)
        # Create the transformation matrix
        angle = i
        angle = np.radians(angle)
        M = np.array([[np.cos(angle), -np.sin(angle), x0 * (1 - np.cos(angle)) + y0 * np.sin(angle)],
                      [np.sin(angle), np.cos(angle), y0 * (1 - np.cos(angle)) - x0 * np.sin(angle)]])
        # get the coordinates in the form of (0,0),(0,1)...
        # the shape is (2, rows*cols)
        orig_coord = np.indices((cols, rows)).reshape(2, -1)
        # stack the rows of 1 to form [x,y,1]
        orig_coord_f = np.vstack((orig_coord, np.ones(rows * cols)))
        transform_coord = np.dot(M, orig_coord_f)
        # Change into int type
        transform_coord = transform_coord.astype(int)
        # Keep only the coordinates that fall within the image boundary.
        indices = np.all(
            (transform_coord[1] < rows, transform_coord[0] < cols, transform_coord[1] >= 0, transform_coord[0] >= 0),
            axis=0)
        # Create a zeros image and project the points
        img1[transform_coord[1][indices], transform_coord[0][indices]] = img[orig_coord[1][indices], orig_coord[0][indices]]
        # Display the image
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        img2 = cv.findNonZero(img1, img2)
        # print(img2)
        if img2 is None:
            continue
        highest_dot_point = find_dot(img2, hierarchy, arrayOfContours)
        highest.append([highest_dot_point, i])
        cv.imshow('a2', img1)
    lowest = 10000
    for dot in highest:
        if dot[0] < lowest:
            rotation_degrees = dot[1]
            lowest = dot[0]
    print("rd: ", rotation_degrees)
    return rotation_degrees


def getSubRect(image, begin_point, end_point):
    x, y, w, h = begin_point[0], begin_point[1], end_point[0], end_point[1]
    return image[y:y+h, x:x+w]


def get_bounding_box_with_only_dot(th3, bounding_box, hierarchy, dots):
    if dots:
        bounding_box_with_only_dot = np.zeros_like(th3)
        dot = dots[0]
        dots = []
        for j in dot:
            dot_width = j[0][0]
            dot_height = j[0][1]
            dots.append([dot_width, dot_height])
        print(dots)
        # cv.drawContours(bounding_box_with_only_dot, dot, -1, (0, 255, 0), 20, cv.LINE_AA)
        dot_edge_points = find_edge_points(dot)
        print("dot_edge_points: ", dot_edge_points)
        top_left_dot = dot_edge_points[0]  # (left_most, top_most)
        bottom_right_dot = dot_edge_points[1]  # (right_most, bottom_most)
        found = 0
        for height in range(len(bounding_box_with_only_dot)):
            for width in range(bounding_box_with_only_dot.shape[1]):
                if top_left_dot[1] < height < bottom_right_dot[1] and top_left_dot[0] < width < bottom_right_dot[0]:
                    if [width, height] in dots:
                        bounding_box_with_only_dot[height, width] = [0, 0, 255]
                        found += 1
                    elif found > 3:
                        continue
            found = 0
        # for height in range(len(bounding_box_with_only_dot)):
        #     for width in range(bounding_box_with_only_dot.shape[1]):
        #         if not np.array_equal(bounding_box_with_only_dot[height, width], [0, 0, 0]):
        #             print(bounding_box_with_only_dot[height, width])
        return bounding_box_with_only_dot
    else:
        return None


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

    cam = cv.VideoCapture(0)
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
        if not _ret:
            continue
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
        contours = cv.findContours(th3.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
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

                cv.drawContours(img, arrayOfContours[i], -1, (0, 255, 0), 20, cv.LINE_AA)
                # numberOfContoursDrawn += 1
                cv.putText(img, (cX - cx).__str__() + "," + (cY - cy).__str__() + "," + shape,
                           (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        # print("numberOfContoursDrawn", numberOfContoursDrawn)
        # print("Center Moments", centerMoments)
        dots = []
        for i in range(numberContour):
            if hierarchy[0][i][0] == -1 and hierarchy[0][i][1] == -1 and hierarchy[0][i][2] == -1 \
                    and hierarchy[0][i][3] != -1:
                parentValue = hierarchy[0][i][3]  # GetParent
                dots.append(arrayOfContours[i])
                while parentValue != -1 and layersDeep <= 8:  # Recursively get parent 8 times
                    parentValue = hierarchy[0][parentValue][3]
                    layersDeep += 1
                if layersDeep < 8:
                    dots.pop()
                contour = arrayOfContours[parentValue]  # Bounding Box contour
                # bp: topleft, ep: bottomRight, width: Right - Left, height: Bottom - Top
                begin_point, end_point, width, height = find_edge_points(contour)  # Begin
                print("Bounding Box edge points", begin_point, end_point, width, height)
                # and End points of the Bounding Box, (and the width and height)
                # cv.drawContours(img, contour, -1, (255, 0, 0), 20, cv.LINE_AA)  # Draw Blue outer contour
                cv.rectangle(img, begin_point, end_point, (255, 0, 0), 2)  # Draws Bounding Box
                bounding_box = getSubRect(th3, begin_point, end_point)
                bounding_box_with_only_dot = get_bounding_box_with_only_dot(img, bounding_box, hierarchy, dots)
                if bounding_box_with_only_dot is not None:
                    z_orientation = rotate_image(bounding_box_with_only_dot, hierarchy, arrayOfContours)
                else:
                    z_orientation = "Recalibrating"
                x_orientation, y_orientation = find_orientation(width, height, base_width, base_height, base_begin_point
                                                                , base_end_point)
                draw_str(img, (20, 40), 'X Orientation: ' + x_orientation.__str__())
                draw_str(img, (20, 60), 'Y Orientation: ' + y_orientation.__str__())
                draw_str(img, (20, 80), 'Z Orientation: ' + z_orientation.__str__())
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
