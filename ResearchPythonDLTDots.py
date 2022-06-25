#!/usr/bin/env python

"""
Research Project

"""

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
# import numpy as np
# import random
import math


from preset.common import clock, draw_str

# local modules


def counter(count, threshold):
    count += 1
    if count >= threshold:
        return 0
    return count


# def draw_rects(img, rects, color):
#     for x1, y1, x2, y2 in rects:
#         cv.rectangle(img, (x1, y1), (x2, y2), color, 2)


def main():
    # cascade_fn = args.get('--cascade', "data/haarcascades/haarcascade_frontalface_alt.xml")
    # nested_fn = args.get('--nested-cascade', "data/haarcascades/haarcascade_eye.xml")
    #
    # cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))
    # nested = cv.CascadeClassifier(cv.samples.findFile(nested_fn))

    # cam = cv.VideoCapture(0)
    threshold_value = 0
    count = 0
    temp = 0
    all_dots = set(())
    next_frame = 0
    while True:
        if next_frame:
            break
            all_dots = set(())
            next_frame = 0
        t = clock()
        count = counter(count, 1)
        if not count:
            threshold_value += 25
            if threshold_value > 255:
                threshold_value = 0
                next_frame = 1
            print("Threshold Value:", threshold_value)

        # _ret, img = cam.read()
        img = cv.imread("Easy.jpg")
        # if not _ret:
        #     continue

        # convert the resized image to grayscale, blur it slightly,
        # and threshold it
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        th3 = cv.threshold(blurred, threshold_value, 255, cv.THRESH_BINARY)[1]
        # find contours in the thresholded image and initialize the
        # shape detector
        contours = cv.findContours(th3.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        hierarchy = contours[1]
        # print(hierarchy)
        numberContour = 0
        arrayOfContours = []
        centerMoments = []
        # layersDeep = 0
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
            pv = hierarchy[0][i][3]
            ppv = hierarchy[0][pv][3]
            pppv = hierarchy[0][ppv][3]
            ppppv = hierarchy[0][pppv][3]
            if hierarchy[0][i][0] == -1 and hierarchy[0][i][1] == -1 and hierarchy[0][i][2] == -1 \
                    and hierarchy[0][i][3] != -1 and hierarchy[0][pv][3] != -1\
                    and hierarchy[0][ppv][3] != -1\
                    and hierarchy[0][pppv][3] != -1\
                    and hierarchy[0][ppppv][3] != -1:
                # print("Hierarchy of Contour\n", i, hierarchy)
                if not temp:
                    print("Center of Contour", i, centerMoments[i])

                cv.drawContours(img, arrayOfContours[i], -1, (0, 255, 0), 20, cv.LINE_AA)
                if centerMoments[i][0] >= 0:
                    all_dots.add((centerMoments[i][0], centerMoments[i][1]))
                    print("Before", all_dots)
                    temp_dots = set(())
                    already_used = set(())
                    for k in all_dots:
                        tempBool = True
                        for j in already_used:
                            if math.isclose(k[0], j[0], abs_tol=math.ceil(img.shape[0] / 100)) \
                                    and math.isclose(k[1], j[1], abs_tol=math.ceil(img.shape[1] / 130)):
                                tempBool = False
                        if tempBool:
                            temp_dots.add(k)
                            already_used.add(k)
                    all_dots = temp_dots
                    print("After", all_dots)
                    cv.putText(img, (str(centerMoments[i][0]) + ', ' + str(centerMoments[i][1])), centerMoments[i],
                               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
                # numberOfContoursDrawn += 1
                # print("numberOfContoursDrawn", numberOfContoursDrawn)
        # if cv.waitKey(5) == ord('a'):
        #     break
        # print("Center Moments", centerMoments)
        temp += 1
        dt = clock() - t
        draw_str(img, (20, 20), 'time: %.1f ms' % (dt * 1000))
        # cv.imshow('shapedetect', img)
        # cv.imshow('threshold', th3)

        # if cv.waitKey(5) == ord('a'):
        #     break

    print('Done')
    return all_dots, img


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
