# import cv2 as cv
# from common import clock, draw_str
#
# import numpy as np
#
# import cv2
# cam = cv2.VideoCapture(0)
# while True:
# _ret, img = cam.read()
#     r = ((9, 9))
#     retval, corners = cv.findChessboardCorners(img, r, flags=cv.CALIB_CB_ADAPTIVE_THRESH)
#     draw_str(img, (20, 40), 'X Orientation: ' + retval.__str__())
#     cv.imshow("img2222232222243234325354364576587687978574 63456325", img)
#     cv.waitKey(5)
import math

import cv2
import numpy as np


def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img


# Defining the dimensions of checkerboard
CHECKERBOARD = (9, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []
images = [cv2.imread("images/20220520_192721.jpg"), cv2.imread("images/20220520_192726.jpg"), cv2.imread("images/20220520_192729.jpg"), cv2.imread("images/20220520_192733.jpg"), cv2.imread("images/20220520_192742.jpg")]
# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
# print("3D coordinates", objp)
# Extracting path of individual image stored in a given directory
# images = glob.glob('./images/*.jpg')
# print("Here", images)
axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
for img in images:
    # print("1")
    # _ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print("1.5")
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH)
    # print("Corners", corners)
    # print("2")
    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display
    them on the images of checker board
    """
    if ret:
        # print("3")
        # objpoints = objp
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # print("4")
        # imgpoints = corners2
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp, [corners2], gray.shape[::-1], None, None)
        print("Camera matrix : \n")
        print(mtx)
        print("dist : \n")
        print(dist)
        print("rvecs : \n")
        print("tvecs : \n")
        print(tvecs)
        print("RMS Error", ret)

        # _, rvec, tvec= cv2.solvePnP(objp, corners2, mtx, dist)
        imgpts, jac = cv2.projectPoints(axis, rvecs[0], tvecs[0], mtx, dist)
        # Draw and display the corners
        img = draw(img,corners2,imgpts)
        # print("5")

    cv2.imshow('img', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

h, w = img.shape[:2]

"""
Performing camera calibration by
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the
detected corners (imgpoints)
"""
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, [imgpoints], gray.shape[::-1], None, None)
# print(objpoints, imgpoints)
# print("Camera matrix : \n")
# print(mtx)
# print("dist : \n")
# print(dist)
# print("rvecs : \n")
# print(rvecs)
# print("tvecs : \n")
# print(tvecs)


"""
[[ 0.25617767],
       [-0.05934495],
       [ 1.01601931]]), array([[-0.03496694],
       [-0.15833597],
       [ 1.52932501]]), array([[ 0.09347974],
       [-0.72828768],
       [-0.31339987]]), array([[-0.3552353 ],
       [-0.16427119],
       [ 0.02609411]]), array([[-0.43554934],
       [-0.08817176],
       [ 0.17720769]]
tvecs :

(array([[-1.90931181],
       [-5.4915283 ],
       [19.16073697]]), array([[ 1.58647736],
       [-5.46551795],
       [19.28087202]]), array([[-2.68208951],
       [-5.96721242],
       [12.87098412]]), array([[-4.43029134],
       [-4.84913069],
       [14.51697909]]), array([[ -2.75815736],
       [-11.91872885],
       [ 31.10699006]]))
"""
