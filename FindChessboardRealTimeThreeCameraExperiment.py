import math
import cv2
import numpy as np
from preset.common import draw_str, clock


def counter(countera, threshold):
    countera += 1
    if countera >= threshold:
        return 0
    return countera


def draw(img11, imgpts11):
    # print("before", imgpts)
    imgpts11 = np.int32(imgpts11).reshape(-1, 2)

    # draw ground floor in green
    img11 = cv2.drawContours(img11, [imgpts11[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i11, j in zip(range(4), range(4, 8)):
        img11 = cv2.line(img11, tuple(imgpts11[i11]), tuple(imgpts11[j]), 255, 3)

    # draw top layer in red color
    img11 = cv2.drawContours(img11, [imgpts11[4:]], -1, (0, 0, 255), 3)

    return img11


def modify_details(ret11, rvecs11, tvecs11):
    # print(imgpoints)
    # print("Camera matrix: ")
    # print(mtx)
    # print("dist: ")
    # print(dist)
    tvec11 = [round(tvecs11[0][0][0], 5), round(tvecs11[0][1][0], 5), round(tvecs11[0][2][0], 5)]
    rvec11 = [round(math.degrees(rvecs11[0][1][0]), 5), round(math.degrees(rvecs11[0][0][0]), 5),
              round(math.degrees(rvecs11[0][2][0]), 5)]
    # print("Translation Vector:\nx: ", round(tvecs[0][0][0], 5),  # Move right is negative, Move left is positive. Limits: (-6, 2.5)
    #       "y: ", round(tvecs[0][1][0], 5),  # Move up is negative, Move down is positive. Limits: (0, -13)
    #       "z: ", round(tvecs[0][2][0], 5))  # Move in is negative, Move out is positive. Limits: (30, 90)
    # print("re-projection error", ret)
    # print("Rotation Vector:\nx: ", round(math.degrees(rvecs[0][1][0]), 5),  # look right is negative, Look left is positive. Limits: (-45, 54)
    #       "y: ", round(math.degrees(rvecs[0][0][0]), 5),  # Look up is negative, look down is positive. Limits: (-45, 47)
    #       "z: ", round(math.degrees(rvecs[0][2][0]), 5))  # Clockwise is negative, Counter-Clockwise is positive. Limits: (-90, 90)
    # print("")
    return tvec11, rvec11, round(ret11, 5)


# first_one = False
# second_one = False

cam = cv2.VideoCapture(2)  # img2, camera 2
cam3 = cv2.VideoCapture(1)  # img, camera 1
cam2 = cv2.VideoCapture(0)  # img3, camera 3

# Defining the dimensions of checkerboard
CHECKERBOARD = (9, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# -2147483638
# [ WARN:0@34.050] global D:\a\opencv-python\opencv-python\opencv\modules\videoio\src\cap_msmf.cpp (1752) CvCapture_MSMF::grabFrame videoio(MSMF): can't grab frame. Error: -2147483638
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []
imgpoints2 = []
imgpoints3 = []
# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
print("3D coordinates", objp)
# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = [objp, objp, objp, objp, objp, objp]
# Extracting path of individual image stored in a given directory
# images = glob.glob('./images/*.jpg')
# print("Here", images)
axis = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                   [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])
# Generic counter, used to slow down code if numbers are important
count = 0
# Threshold of whether or not the ret is good.
RMS_thresh = 0.5
# Is the error of camera one greater than RMS_thresh?
RMS_one = False
# Is the error of camera two greater than RMS_thresh?
RMS_two = False
# Is the error of camera three greater than RMS_thresh?
RMS_three = False
# subtraction factor of the x rotation, from camera one to two
sf_x_rot = 0
# subtraction factor of the y rotation, from camera one to two
sf_y_rot = 0
# subtraction factor of the z rotation, from camera one to two
sf_z_rot = 0
# subtraction factor of the x rotation, from camera one to three
sf_x_rot3 = 0
# subtraction factor of the y rotation, from camera one to three
sf_y_rot3 = 0
# subtraction factor of the z rotation, from camera one to three
sf_z_rot3 = 0
# subtraction factor of the x translation, from camera one to two
sf_x_tra = 0
# subtraction factor of the y translation, from camera one to two
sf_y_tra = 0
# subtraction factor of the z translation, from camera one to two
sf_z_tra = 0
# subtraction factor of the x translation, from camera one to three
sf_x_tra3 = 0
# subtraction factor of the y translation, from camera one to three
sf_y_tra3 = 0
# subtraction factor of the z translation, from camera one to three
sf_z_tra3 = 0
# Is the camera coming from the dead zone into the second camera?
cross_over = False
# Is the camera coming from the dead zone into the third camera?
cross_over3 = False
# Used to average all the subtraction factors for rotation
details_rotation = []
# Used to average all the subtraction factors for translation
details_translation = []
# Formatted Rotation Vector
rvec = [1000, 1000, 1000]
# Formatted Translation Vector
tvec = [1000, 1000, 1000]
# Projected image point of camera one
imgpts = []
# projected image points of camera 2
imgpts2 = []
# projected image points of camera 3
imgpts3 = []
# put here to make the code behave
rvec2 = []
rvec3 = []
final_pos = [0, 0, 0]
starting_pos = [0, 0, 0]
final_pos3 = [0, 0, 0]
starting_pos3 = [0, 0, 0]
final_tvec = 0
final_rvec = 0
retr1 = 2000.6
retr2 = 2000.7
retr3 = 2000.8
experiment_1 = False
min_x_camera1 = 0
max_x_camera1 = 0
experiment_2 = False
min_x_camera2 = 0
max_x_camera2 = 0
experiment_3 = False
min_x_camera3 = 0
max_x_camera3 = 0
count4 = 0
max_range1_2 = 0
while True:
    # print("1")
    t = clock()
    _ret, img2 = cam.read()
    if not _ret:
        print("Here cam 1")
        continue
    _ret, img3 = cam2.read()
    if not _ret:
        print("here cam 2")
        continue
    _ret, img = cam3.read()
    if not _ret:
        print("Here cam 3")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH
                                             + cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret2, cornerss = cv2.findChessboardCorners(gray2, CHECKERBOARD,
                                               cv2.CALIB_CB_ADAPTIVE_THRESH
                                               + cv2.CALIB_CB_FAST_CHECK +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret3, cornersss = cv2.findChessboardCorners(gray3, CHECKERBOARD,
                                                cv2.CALIB_CB_ADAPTIVE_THRESH
                                                + cv2.CALIB_CB_FAST_CHECK +
                                                cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret or ret2 or ret3:
        if len(imgpoints) > 5:
            imgpoints = []
        if len(imgpoints2) > 5:
            imgpoints2 = []
        if len(imgpoints3) > 5:
            imgpoints3 = []
        # print("3")
        # refining pixel coordinates for given 2d points.
        if not experiment_2:
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
        if not experiment_1:
            if ret2:
                cornerss2 = cv2.cornerSubPix(gray2, cornerss, (11, 11), (-1, -1), criteria)
                imgpoints2.append(cornerss2)
        if ret3:
            cornersss2 = cv2.cornerSubPix(gray3, cornersss, (11, 11), (-1, -1), criteria)
            imgpoints3.append(cornersss2)
        # print("4")
        # first_one = False
        # second_one = False

        if len(imgpoints) > 5:  # First camera
            RMS_one = False
            retr1, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            # first_one = True

            # mean_error = 0
            # for i in range(len(objpoints)):
            #     imgpointss2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            #     error = cv2.norm(imgpoints[i], imgpointss2, cv2.NORM_L2) / len(imgpointss2)
            #     mean_error += error
            if retr1 < RMS_thresh:

                # count = counter(count, 5)
                # if not count:
                tvec, rvec, retr1 = modify_details(retr1, rvecs, tvecs)
                # print("total error: {}".format(mean_error / len(objpoints)))
                if rvec[0] < min_x_camera1:
                    min_x_camera1 = rvec[0]
                if rvec[0] > max_x_camera1:
                    max_x_camera1 = rvec[0]
                imgpts, jac = cv2.projectPoints(axis, rvecs[0], tvecs[0], mtx, dist)
                img = draw(img, imgpts)
                dt = clock() - t
                max_range1_2 = max_x_camera1 - min_x_camera2
                draw_str(img, (480, 20), 'time: %.1f ms' % (dt * 1000))
                draw_str(img, (20, 20), 'x rot: ' + str(rvec[0]))
                draw_str(img, (20, 40), 'y rot: ' + str(rvec[1]))
                draw_str(img, (20, 60), 'z rot: ' + str(rvec[2]))
                draw_str(img, (20, 80), 'x pos: ' + str(tvec[0]))
                draw_str(img, (20, 100), 'y pos: ' + str(tvec[1]))
                draw_str(img, (20, 120), 'z pos: ' + str(tvec[2]))
                draw_str(img, (20, 140), 'RMS Error: ' + str(retr1))
                draw_str(img, (20, 160), 'min_x_camera1: ' + str(min_x_camera1))
                draw_str(img, (20, 180), 'max_x_camera1: ' + str(max_x_camera1))
                draw_str(img, (20, 200), 'max_range_with_2_cameras: ' + str(max_range1_2))
                cv2.imshow('img', img)
                cv2.imshow('img2', img2)
                cv2.imshow('img3', img3)
                cv2.waitKey(1)
                # if dst != []:
                # cv2.imshow("dst", dst)
            else:
                RMS_one = True
            # print("5")

        if len(imgpoints2) > 5:  # Second camera
            RMS_two = False
            retr2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, gray2.shape[::-1], None,
                                                                     None)
            # second_one = True

            # mean_error = 0
            # for i in range(len(objpoints)):
            #     imgpointss2, _ = cv2.projectPoints(objpoints[i], rvecs2[i], tvecs2[i], mtx2, dist2)
            #     error = cv2.norm(imgpoints2[i], imgpointss2, cv2.NORM_L2) / len(imgpointss2)
            #     mean_error += error
            if retr2 < RMS_thresh:

                # count = counter(count, 5)
                # if not count:
                tvec, rvec, retr2 = modify_details(retr2, rvecs2, tvecs2)
                if cross_over:
                    print("Cross Over Start")
                    starting_pos = tvec
                    cross_over = False
                    print("subtraction factor x: ", sf_x_rot, "subtraction factor y: ", sf_y_rot,
                          "subtraction factor z: ",
                          sf_z_rot)
                    print("rvec: ", rvec)
                    print("Cross Over Complete \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
                if starting_pos[0]:
                    sf_z_tra, sf_y_tra, sf_x_tra = starting_pos[0] - tvec[0], starting_pos[1] - tvec[1], starting_pos[
                        2] - tvec[2]
                # print("total error: {}".format(mean_error / len(objpoints)))
                temp_rvec = rvec
                rvec = rvec[0] - sf_x_rot, rvec[1] - sf_y_rot, rvec[2] - sf_z_rot
                tvec = final_pos[0] - sf_x_tra, final_pos[1] - sf_y_tra, final_pos[2] - sf_z_tra

                if rvec[0] < min_x_camera2:
                    min_x_camera2 = rvec[0]
                if rvec[0] > max_x_camera2:
                    max_x_camera2 = rvec[0]

                count4 = counter(count4, 5)
                if not count4:
                    print("Rotation Vector - Local to Camera 2: \nx: ", temp_rvec[0], "y: ", temp_rvec[1], "z: ",
                          temp_rvec[2])
                    print("Rotation Vector - Local to Camera 1: \nx: ", rvec[0], "y: ", rvec[1], "z: ", rvec[2])
                    print("Subtraction Factor\nx: ", sf_x_rot, "y: ", sf_y_rot, "z: ", sf_z_rot)
                # print("Translation Vector\nx: ", tvec[0], "y: ", tvec[1], "z: ", (tvec[2]))
                imgpts2, jac = cv2.projectPoints(axis, rvecs2[0], tvecs2[0], mtx2, dist2)
                img2 = draw(img2, imgpts2)
                dt = clock() - t
                max_range1_2 = max_x_camera1 - min_x_camera2
                print(sf_x_rot)
                draw_str(img2, (480, 20), 'time: %.1f ms' % (dt * 1000))
                draw_str(img2, (20, 20), 'x rot: ' + str(rvec[0]))
                draw_str(img2, (20, 40), 'y rot: ' + str(rvec[1]))
                draw_str(img2, (20, 60), 'z rot: ' + str(rvec[2]))
                draw_str(img2, (20, 80), 'x pos: ' + str(tvec[0]))
                draw_str(img2, (20, 100), 'y pos: ' + str(tvec[1]))
                draw_str(img2, (20, 120), 'z pos: ' + str(tvec[2]))
                draw_str(img2, (20, 140), 'RMS Error: ' + str(retr2))
                draw_str(img2, (20, 160), 'min_x_camera2: ' + str(min_x_camera2))
                draw_str(img2, (20, 180), 'max_x_camera2: ' + str(max_x_camera2))
                draw_str(img2, (20, 200), 'max_range_with_2_cameras: ' + str(max_range1_2))
                draw_str(img2, (480, 60), "x: " + str(sf_x_rot))
                draw_str(img2, (480, 80), "y: " + str(sf_y_rot))
                draw_str(img2, (480, 100), "z: " + str(sf_z_rot))
                draw_str(img2, (480, 120), "x: " + str(sf_x_tra))
                draw_str(img2, (480, 140), "y: " + str(sf_y_tra))
                draw_str(img2, (480, 160), "z: " + str(sf_z_tra))
                cv2.imshow('img', img)
                cv2.imshow('img2', img2)
                cv2.imshow('img3', img3)
                cv2.waitKey(1)
                final_tvec = tvec
                final_rvec = rvec
                # if dst != []:
                # cv2.imshow("dst", dst)
            else:
                # print("Here", retr2)
                RMS_two = True
            # print("5")

        if len(imgpoints3) > 5:  # Third camera
            print("Camera 3 reached")
            RMS_three = False
            retr3, mtx3, dist3, rvecs3, tvecs3 = cv2.calibrateCamera(objpoints, imgpoints3, gray3.shape[::-1], None,
                                                                     None)
            # first_one = True

            # mean_error = 0
            # for i in range(len(objpoints)):
            #     imgpointss2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            #     error = cv2.norm(imgpoints[i], imgpointss2, cv2.NORM_L2) / len(imgpointss2)
            #     mean_error += error
            if retr3 < RMS_thresh:
                print("Camera 3 success")
                # count = counter(count, 5)
                # if not count:
                tvec, rvec, retr3 = modify_details(retr3, rvecs3, tvecs3)
                if cross_over3:
                    print("Cross Over 3 Start")
                    starting_pos3 = tvec
                    cross_over3 = False
                    print("subtraction factor x: ", sf_x_rot3, "subtraction factor y: ", sf_y_rot3,
                          "subtraction factor z: ", sf_z_rot3)
                    print("rvec: ", rvec)
                    print("Cross Over Complete \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
                if starting_pos3[0]:
                    sf_z_tra, sf_y_tra, sf_x_tra = -starting_pos3[0] + tvec[0], starting_pos3[1] - tvec[1], starting_pos3[2] - tvec[2]
                # print("total error: {}".format(mean_error / len(objpoints)))
                temp_rvec = rvec
                rvec = rvec[0] - sf_x_rot3, rvec[1] - sf_y_rot3, rvec[2] - sf_z_rot3
                tvec = final_pos[0] - sf_x_tra, final_pos[1] - sf_y_tra, final_pos[2] - sf_z_tra
                count4 = counter(count4, 5)
                if not count4:
                    print("Rotation Vector - Local to Camera 3: \nx: ", temp_rvec[0], "y: ", temp_rvec[1], "z: ", temp_rvec[2])
                    print("Rotation Vector - Local to Camera 1: \nx: ", rvec[0], "y: ", rvec[1], "z: ", rvec[2])
                    print("Subtraction Factor \nx: ", sf_x_rot3, "y: ", sf_y_rot3, "z: ", sf_z_rot3)
                if rvec[0] < min_x_camera3:
                    min_x_camera3 = rvec[0]
                if rvec[0] > max_x_camera3:
                    max_x_camera3 = rvec[0]

                imgpts3, jac = cv2.projectPoints(axis, rvecs3[0], tvecs3[0], mtx3, dist3)
                img3 = draw(img3, imgpts3)
                dt = clock() - t
                draw_str(img3, (480, 20), 'time: %.1f ms' % (dt * 1000))
                draw_str(img3, (20, 20), 'x rot: ' + str(rvec[0]))
                draw_str(img3, (20, 40), 'y rot: ' + str(rvec[1]))
                draw_str(img3, (20, 60), 'z rot: ' + str(rvec[2]))
                draw_str(img3, (20, 80), 'x pos: ' + str(tvec[0]))
                draw_str(img3, (20, 100), 'y pos: ' + str(tvec[1]))
                draw_str(img3, (20, 120), 'z pos: ' + str(tvec[2]))
                draw_str(img3, (20, 140), 'RMS Error: ' + str(retr3))
                draw_str(img3, (20, 160), 'min_x_camera3: ' + str(min_x_camera3))
                draw_str(img3, (20, 180), 'max_x_camera3: ' + str(max_x_camera3))
                draw_str(img3, (20, 200), 'max_range_with_3_cameras: ' + str(max_range1_2 - sf_x_rot3 - max_x_camera1 + max_x_camera3))
                cv2.imshow('img', img)
                cv2.imshow('img2', img2)
                cv2.imshow('img3', img3)
                cv2.waitKey(1)
                # if dst != []:
                # cv2.imshow("dst", dst)
            else:
                RMS_three = True
            # print("5")

        if RMS_one and RMS_two and retr1 < 2000 and retr2 < 2000:  # Dead Zone between camera 1 and camera 2
            count2 = 1
            # count3 = 1
            out_of_dead_zone = False
            # print(len(imgpoints), len(imgpoints2))

            while len(imgpoints) < 6 or len(imgpoints2) < 6:
                t = clock()
                _ret, img = cam3.read()
                if not _ret:
                    continue
                _ret, img2 = cam.read()
                if not _ret:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                # print("1.5")
                # Find the chess board corners
                # If desired number of corners are found in the image then ret = true
                ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                         cv2.CALIB_CB_ADAPTIVE_THRESH
                                                         + cv2.CALIB_CB_FAST_CHECK +
                                                         cv2.CALIB_CB_NORMALIZE_IMAGE)
                ret2, cornerss = cv2.findChessboardCorners(gray2, CHECKERBOARD,
                                                           cv2.CALIB_CB_ADAPTIVE_THRESH
                                                           + cv2.CALIB_CB_FAST_CHECK +
                                                           cv2.CALIB_CB_NORMALIZE_IMAGE)
                print(len(imgpoints), len(imgpoints2))
                if len(imgpoints2) == 6 or len(imgpoints) == 6:
                    count2 = counter(count2, 20)
                    if not count2:
                        out_of_dead_zone = True
                        break
                if ret and len(imgpoints) < 6:
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners2)
                if ret2 and len(imgpoints2) < 6:
                    cornerss2 = cv2.cornerSubPix(gray2, cornerss, (11, 11), (-1, -1), criteria)
                    imgpoints2.append(cornerss2)
                count = counter(count, 5)
                if not count:
                    print("Dead Zone Image Accuracy\nretr1: ", retr1, "\nretr2: ", retr2)
            if not out_of_dead_zone:  # In Dead Zone
                # n = [
                #     retval,
                #     cameraMatrix1,
                #     distCoeffs1,
                #     cameraMatrix2,
                #     distCoeffs2,
                #     R,
                #     T,
                #     E,
                #     F,
                # ] = cv2.stereoCalibrate(
                #     objpoints,
                #     imgpoints,
                #     imgpoints2,
                #     mtx,
                #     dist,
                #     mtx2,
                #     dist2,
                #     img.shape[:2],
                #     flags=cv2.CALIB_FIX_INTRINSIC,
                # )
                # rotvec = cv2.Rodrigues(n[5])
                # rotvec = [round(math.degrees(rotvec[0][1][0]), 5), round(math.degrees(rotvec[0][0][0]), 5), round(math.degrees(rotvec[0][2][0]), 5)]
                # travec = n[6]
                # print("travec: ", travec, travec[0], travec[0][0])
                # travec = [round(travec[0][0], 5), round(travec[1][0], 5), round(travec[2][0], 5)]
                # print("travec: ", travec, "\nrotvec: ", rotvec)
                # details = [travec, rotvec]
                # sf_x, sf_y, sf_z = details[1]
                # print("subtraction factor x: ", sf_x, "subtraction factor y: ", sf_y, "subtraction factor z: ", sf_z)
                retr1, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

                retr2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, gray2.shape[::-1], None,
                                                                         None)

                tvec, rvec, retr = modify_details(retr1, rvecs, tvecs)
                tvec2, rvec2, retr2 = modify_details(retr2, rvecs2, tvecs2)

                if rvec[0] > -50:
                    print("Failed Cam 1")
                    continue
                print("Cam 1 Successful")
                if rvec2[0] < 50:
                    print("Failed Cam 2")
                    continue
                print("Cam 2 successful\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

                dt = clock() - t
                imgpts, jac = cv2.projectPoints(axis, rvecs[0], tvecs[0], mtx, dist)
                img = draw(img, imgpts)
                draw_str(img, (480, 20), 'time: %.1f ms' % (dt * 1000))
                draw_str(img, (20, 20), 'x rot: ' + str(rvec[0]))
                draw_str(img, (20, 40), 'y rot: ' + str(rvec[1]))
                draw_str(img, (20, 60), 'z rot: ' + str(rvec[2]))
                draw_str(img, (20, 80), 'x pos: ' + str(tvec[0]))
                draw_str(img, (20, 100), 'y pos: ' + str(tvec[1]))
                draw_str(img, (20, 120), 'z pos: ' + str(tvec[2]))
                draw_str(img, (20, 140), 'RMS Error: ' + str(retr1))
                draw_str(img, (480, 20), 'time: %.1f ms' % (dt * 1000))
                draw_str(img, (480, 40), 'Dead Zone')

                imgpts2, _ = cv2.projectPoints(axis, rvecs2[0], tvecs2[0], mtx2, dist2)
                img2 = draw(img2, imgpts2)
                draw_str(img2, (480, 20), 'time: %.1f ms' % (dt * 1000))
                draw_str(img2, (20, 20), 'x rot: ' + str(rvec2[0]))
                draw_str(img2, (20, 40), 'y rot: ' + str(rvec2[1]))
                draw_str(img2, (20, 60), 'z rot: ' + str(rvec2[2]))
                draw_str(img2, (20, 80), 'x pos: ' + str(tvec2[0]))
                draw_str(img2, (20, 100), 'y pos: ' + str(tvec2[1]))
                draw_str(img2, (20, 120), 'z pos: ' + str(tvec2[2]))
                draw_str(img2, (20, 140), 'RMS Error: ' + str(retr2))
                draw_str(img2, (480, 20), 'time: %.1f ms' % (dt * 1000))
                draw_str(img2, (480, 40), 'Dead Zone')

                cv2.imshow('img', img)
                cv2.imshow('img2', img2)
                cv2.imshow('img3', img3)
                cv2.waitKey(1)

                if final_tvec and final_rvec:
                    print("In dead Zone\nfinal_rvec: ", final_rvec, "\nrvec: ", rvec, "\nrvec2: ", rvec2, "\ntvec2: ",
                          tvec2, "\nFinal tvec: ", final_tvec, "\ntvec: ", tvec)
                    print("Rotation Error: ", (abs(final_rvec[0] - rvec[0]) + abs(final_rvec[1] - rvec[1]) + abs(
                        final_rvec[2] - rvec[2])) / 3)
                    print("Translation Error: ", (abs(final_tvec[0] - tvec[0]) + abs(final_tvec[1] - tvec[1]) + abs(
                        final_tvec[2] - tvec[2])) / 3)
                else:
                    count = counter(count, 3)
                    if not count:
                        print("In dead Zone\nrvec: ", rvec, "\nrvec2: ", rvec2)
                    # print("tvec: ", tvec, "\ntvec2: ", tvec2)

                sf_x_rot, sf_y_rot, sf_z_rot = rvec2[0] - rvec[0], rvec2[1] - rvec[1], rvec2[2] - rvec[2]
                sf_x_tra, sf_y_tra, sf_z_tra = tvec2[0] - tvec[0], tvec2[1] - tvec[1], tvec2[2] - tvec[2]

                details_rotation.append([sf_x_rot, sf_y_rot, sf_z_rot])
                details_translation.append([sf_x_tra, sf_y_tra, sf_z_tra])
            else:  # Out of Dead Zone
                RMS_one = False
                RMS_two = False
                if rvec != [0, 0, 0] and rvec2 != [0, 0, 0]:
                    print("Out of dead Zone\nrvec: ", rvec, "\nrvec2: ", rvec2, "\n\n")
                    cross_over = True
                to_avg_x = 0
                to_avg_y = 0
                to_avg_z = 0
                to_avg_count = 0
                for i in details_rotation:
                    to_avg_x += i[0]
                    to_avg_y += i[1]
                    to_avg_z += i[2]
                    to_avg_count += 1
                sf_x_rot, sf_y_rot, sf_z_rot = to_avg_x / to_avg_count, to_avg_y / to_avg_count, to_avg_z / to_avg_count
                to_avg_x = 0
                to_avg_y = 0
                to_avg_z = 0
                to_avg_count = 0
                for i in details_translation:
                    to_avg_x += i[0]
                    to_avg_y += i[1]
                    to_avg_z += i[2]
                    to_avg_count += 1
                sf_x_tra, sf_y_tra, sf_z_tra = to_avg_x / to_avg_count, to_avg_y / to_avg_count, to_avg_z / to_avg_count
                final_pos = tvec
        elif retr1 < RMS_thresh and retr2 < RMS_thresh and ret and ret2:
            print("HHHH\n\n\n\n", retr1, retr2)
        elif RMS_one and RMS_three and retr1 < 2000 and retr3 < 2000:  # Dead Zone between camera 1 and camera 3
            # count2 = 1
            count3 = 1
            out_of_dead_zone3 = False
            # print(len(imgpoints), len(imgpoints2))

            while len(imgpoints) < 6 or len(imgpoints3) < 6:
                t = clock()
                _ret, img = cam3.read()
                if not _ret:
                    continue
                _ret, img3 = cam2.read()
                if not _ret:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                # print("1.5")
                # Find the chess board corners
                # If desired number of corners are found in the image then ret = true
                ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                         cv2.CALIB_CB_ADAPTIVE_THRESH
                                                         + cv2.CALIB_CB_FAST_CHECK +
                                                         cv2.CALIB_CB_NORMALIZE_IMAGE)
                ret3, cornersss = cv2.findChessboardCorners(gray3, CHECKERBOARD,
                                                            cv2.CALIB_CB_ADAPTIVE_THRESH
                                                            + cv2.CALIB_CB_FAST_CHECK +
                                                            cv2.CALIB_CB_NORMALIZE_IMAGE)
                print(len(imgpoints), len(imgpoints3))
                if len(imgpoints3) == 6 or len(imgpoints) == 6:
                    count3 = counter(count3, 20)
                    if not count3:
                        out_of_dead_zone3 = True
                        break
                if ret and len(imgpoints) < 6:
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners2)
                if ret3 and len(imgpoints3) < 6:
                    cornersss2 = cv2.cornerSubPix(gray3, cornersss, (11, 11), (-1, -1), criteria)
                    imgpoints3.append(cornersss2)
                count = counter(count, 5)
                if not count:
                    print("Dead Zone Image Accuracy\nretr1: ", retr1, "\nretr3: ", retr3)
            if not out_of_dead_zone3:  # In Dead Zone
                # n = [
                #     retval,
                #     cameraMatrix1,
                #     distCoeffs1,
                #     cameraMatrix2,
                #     distCoeffs2,
                #     R,
                #     T,
                #     E,
                #     F,
                # ] = cv2.stereoCalibrate(
                #     objpoints,
                #     imgpoints,
                #     imgpoints2,
                #     mtx,
                #     dist,
                #     mtx2,
                #     dist2,
                #     img.shape[:2],
                #     flags=cv2.CALIB_FIX_INTRINSIC,
                # )
                # rotvec = cv2.Rodrigues(n[5])
                # rotvec = [round(math.degrees(rotvec[0][1][0]), 5), round(math.degrees(rotvec[0][0][0]), 5), round(math.degrees(rotvec[0][2][0]), 5)]
                # travec = n[6]
                # print("travec: ", travec, travec[0], travec[0][0])
                # travec = [round(travec[0][0], 5), round(travec[1][0], 5), round(travec[2][0], 5)]
                # print("travec: ", travec, "\nrotvec: ", rotvec)
                # details = [travec, rotvec]
                # sf_x, sf_y, sf_z = details[1]
                # print("subtraction factor x: ", sf_x, "subtraction factor y: ", sf_y, "subtraction factor z: ", sf_z)
                retr1, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

                retr3, mtx3, dist3, rvecs3, tvecs3 = cv2.calibrateCamera(objpoints, imgpoints3, gray3.shape[::-1], None,
                                                                         None)
                tvec, rvec, retr = modify_details(retr1, rvecs, tvecs)
                tvec3, rvec3, retr3 = modify_details(retr3, rvecs3, tvecs3)

                if rvec[0] < 50:
                    print("Failed Cam 1")
                    continue
                print("Cam 1 Successful")
                if rvec3[0] > -40:
                    print("Failed Cam 3")
                    continue
                print("Cam 3 successful\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

                dt = clock() - t
                imgpts, jac = cv2.projectPoints(axis, rvecs[0], tvecs[0], mtx, dist)
                img = draw(img, imgpts)
                draw_str(img, (20, 20), 'x rot: ' + str(rvec[0]))
                draw_str(img, (20, 40), 'y rot: ' + str(rvec[1]))
                draw_str(img, (20, 60), 'z rot: ' + str(rvec[2]))
                draw_str(img, (20, 80), 'x pos: ' + str(tvec[0]))
                draw_str(img, (20, 100), 'y pos: ' + str(tvec[1]))
                draw_str(img, (20, 120), 'z pos: ' + str(tvec[2]))
                draw_str(img, (20, 140), 'RMS Error: ' + str(retr1))
                draw_str(img, (480, 20), 'time: %.1f ms' % (dt * 1000))
                draw_str(img, (480, 40), 'Dead Zone')

                imgpts3, _ = cv2.projectPoints(axis, rvecs3[0], tvecs3[0], mtx3, dist3)
                img3 = draw(img3, imgpts3)
                draw_str(img3, (20, 20), 'x rot: ' + str(rvec3[0]))
                draw_str(img3, (20, 40), 'y rot: ' + str(rvec3[1]))
                draw_str(img3, (20, 60), 'z rot: ' + str(rvec3[2]))
                draw_str(img3, (20, 80), 'x pos: ' + str(tvec3[0]))
                draw_str(img3, (20, 100), 'y pos: ' + str(tvec3[1]))
                draw_str(img3, (20, 120), 'z pos: ' + str(tvec3[2]))
                draw_str(img3, (20, 140), 'RMS Error: ' + str(retr3))
                draw_str(img3, (480, 20), 'time: %.1f ms' % (dt * 1000))
                draw_str(img3, (480, 40), 'Dead Zone')

                cv2.imshow('img', img)
                cv2.imshow('img2', img2)
                cv2.imshow('img3', img3)
                cv2.waitKey(1)

                # if final_tvec and final_rvec:
                #     print("In dead Zone\nfinal_rvec: ", final_rvec, "\nrvec: ", rvec, "\nrvec2: ", rvec2, "\ntvec2: ",
                #           tvec2, "\nFinal tvec: ", final_tvec, "\ntvec: ", tvec)
                #     print("Rotation Error: ", (abs(final_rvec[0] - rvec[0]) + abs(final_rvec[1] - rvec[1]) + abs(
                #         final_rvec[2] - rvec[2])) / 3)
                #     print("Translation Error: ", (abs(final_tvec[0] - tvec[0]) + abs(final_tvec[1] - tvec[1]) + abs(
                #         final_tvec[2] - tvec[2])) / 3)
                # else:
                count = counter(count, 3)
                if not count:
                    print("In dead Zone\nrvec: ", rvec, "\nrvec3: ", rvec3)
                # print("tvec: ", tvec, "\ntvec2: ", tvec2)

                details_rotation.append([rvec3[0] - rvec[0], rvec3[1] - rvec[1], rvec3[2] - rvec[2]])
                details_translation.append([tvec3[0] - tvec[0], tvec3[1] - tvec[1], tvec3[2] - tvec[2]])
            else:  # Out of Dead Zone
                RMS_one = False
                RMS_three = False
                if rvec != [0, 0, 0] and rvec3 != [0, 0, 0]:
                    print("Out of dead Zone\nrvec: ", rvec, "\nrvec3: ", rvec3, "\n\n")
                    cross_over3 = True
                to_avg_x = 0
                to_avg_y = 0
                to_avg_z = 0
                to_avg_count = 0
                for i in details_rotation:
                    to_avg_x += i[0]
                    to_avg_y += i[1]
                    to_avg_z += i[2]
                    to_avg_count += 1
                sf_x_rot3, sf_y_rot3, sf_z_rot3 = to_avg_x / to_avg_count, to_avg_y / to_avg_count, to_avg_z / to_avg_count
                to_avg_x = 0
                to_avg_y = 0
                to_avg_z = 0
                to_avg_count = 0
                for i in details_translation:
                    to_avg_x += i[0]
                    to_avg_y += i[1]
                    to_avg_z += i[2]
                    to_avg_count += 1
                sf_x_tra3, sf_y_tra3, sf_z_tra3 = to_avg_x / to_avg_count, to_avg_y / to_avg_count, to_avg_z / to_avg_count
                final_pos3 = tvec
        elif retr1 < RMS_thresh and retr3 < RMS_thresh and ret and ret3:
            print("HHHH\n\n\n\n", retr1, retr3)

    else:  # No Chessboard Found
        dt = clock() - t
        draw_str(img, (20, 20), 'time: %.1f ms' % (dt * 1000))
        cv2.imshow('img', img)
        cv2.imshow('img2', img2)
        cv2.imshow('img3', img3)
        cv2.waitKey(1)
        print("No Chessboard Found")
cv2.destroyAllWindows()
