import math
from copy import copy

import cv2
import numpy as np


def counter(count, threshold):
    count += 1
    if count >= threshold:
        count = 0
    return count


cam = cv2.VideoCapture(0)


def draw(img, corners, imgpts):
    # print("before", imgpts)
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


# Defining the dimensions of checkerboard
CHECKERBOARD = (6, 9)
CHECKERBOARD2 = (9, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
objpoints2 = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []
imgpoints2 = []
# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp2 = np.zeros((1, CHECKERBOARD2[0] * CHECKERBOARD2[1], 3), np.float32)
objp2[0, :, :2] = np.mgrid[0:CHECKERBOARD2[0], 0:CHECKERBOARD2[1]].T.reshape(-1, 2)
# print("3D coordinates", objp)
# Extracting path of individual image stored in a given directory
# images = glob.glob('./images/*.jpg')
# print("Here", images)
axis = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                   [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])
count = 0


def print_details(ret, mtx, dist, rvecs, tvecs, imgpoints):
    # print(imgpoints)
    # print("Camera matrix: ")
    # print(mtx)
    # print("dist: ")
    # print(dist)
    print("Translation Vector:\nx: ", round(tvecs[0][1][0], 5),
          # Move right is negative, Move left is positive. Limits: (-6, 2.5)
          "y: ", round(tvecs[0][0][0], 5),  # Move up is negative, Move down is positive. Limits: (0, -13)
          "z: ", round(tvecs[0][2][0], 5))  # Move in is negative, Move out is positive. Limits: (30, 90)
    print("ret", ret)
    print("Rotation Vector:\nx: ", round(math.degrees(rvecs[0][1][0]), 5),
          # look right is negative, Look left is positive. Limits: (-45, 54)
          "y: ", round(math.degrees(rvecs[0][0][0]), 5),
          # Look up is negative, look down is positive. Limits: (-45, 47)
          "z: ", round(math.degrees(rvecs[0][2][0]),
                       5))  # Clockwise is negative, Counter-Clockwise is positive. Limits: (-90, 90)
    print("")


while True:
    # print("1")
    _ret, img = cam.read()
    img2 = copy(img)
    if not _ret:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print("1.5")
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH
                                             + cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret2, corners3 = cv2.findChessboardCorners(gray, CHECKERBOARD2,
                                               cv2.CALIB_CB_ADAPTIVE_THRESH
                                               + cv2.CALIB_CB_FAST_CHECK +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
    # print("Corners", corners)
    # print("2")
    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display
    them on the images of checker board
    """
    if ret:
        # print("3")
        if len(objpoints) > 5:
            objpoints = []
        objpoints.append(objp)
        if len(objpoints2) > 5:
            objpoints2 = []
        objpoints2.append(objp2)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        corners4 = cv2.cornerSubPix(gray, corners3, (11, 11), (-1, -1), criteria)
        # print("4")
        if len(imgpoints) > 5:
            imgpoints = []
        imgpoints.append(corners2)
        if len(imgpoints2) > 5:
            imgpoints2 = []
        imgpoints2.append(corners3)
        if len(objpoints) > 4:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints2, imgpoints2, gray.shape[::-1], None, None)
            # _, rvec, tvec= cv2.solvePnP(objp, corners2, mtx, dist)

            imgpts, jac = cv2.projectPoints(axis, rvecs[0], tvecs[0], mtx, dist)
            imgpts3, jac2 = cv2.projectPoints(axis, rvecs2[0], tvecs2[0], mtx2, dist2)
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            # crop the image
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
            # cv2.imshow('calibresult.png', dst)
            imgpts2, _ = cv2.projectPoints(axis, rvecs[0], tvecs[0], mtx, 0)
            h2, w2 = img2.shape[:2]
            newcameramtx2, roi2 = cv2.getOptimalNewCameraMatrix(mtx2, dist2, (w2, h2), 1, (w2, h2))
            dst2 = cv2.undistort(img2, mtx2, dist2, None, newcameramtx2)
            # crop the image
            x, y, w, h = roi2
            dst2 = dst2[y:y + h, x:x + w]
            # cv2.imshow('calibresult.png', dst)
            imgpts4, _ = cv2.projectPoints(axis, rvecs2[0], tvecs2[0], mtx2, 0)

            # Draw and display the corners
            if ret < 0.5:
                count = counter(count, 5)
                if not count:
                    print("69")
                    print_details(ret, mtx, dist, rvecs, tvecs, imgpts)
                    print("96")
                    print_details(ret2, mtx2, dist2, rvecs2, tvecs2, imgpts3)
                img2 = draw(img2, corners2, imgpts3)
                img = draw(img, corners2, imgpts)
                cv2.imshow("dst", dst)
                cv2.imshow("dst2", dst2)
            else:
                print("bad")
            # print("5")
            cv2.imshow('img', img)
            cv2.imshow("img2", img2)
            cv2.waitKey(1)
        else:
            pass
            # print("not bad")
    else:
        cv2.imshow('img', img)
        cv2.imshow("img2", img2)
        cv2.waitKey(1)
        print("also bad")
cv2.destroyAllWindows()
