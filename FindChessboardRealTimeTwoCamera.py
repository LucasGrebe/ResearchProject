import math
import cv2
import numpy as np
from common import draw_str, clock


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


cam = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(1)

# Defining the dimensions of checkerboard
CHECKERBOARD = (9, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []
imgpoints2 = []
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
# subtraction factor of the x rotation, from camera one to two
sf_x_rot = 0
# subtraction factor of the y rotation, from camera one to two
sf_y_rot = 0
# subtraction factor of the z rotation, from camera one to two
sf_z_rot = 0
# subtraction factor of the x translation, from camera one to two
sf_x_tra = 0
# subtraction factor of the y translation, from camera one to two
sf_y_tra = 0
# subtraction factor of the z translation, from camera one to two
sf_z_tra = 0
# Is the camera coming from the dead zone into the second camera?
cross_over = False
# Used to average all the subtraction factors for rotation
details_rotation = []
# Used to average all the subtraction factors for translation
details_translation = []
# Formatted Rotation Vector
rvec = []
# Formatted Translation Vector
tvec = []
# Projected image point of camera one
imgpts = []
# projected image points of camera 2
imgpts2 = []
# put here to make the code behave
rvec2 = []
final_pos = [0, 0, 0]
starting_pos = [0, 0, 0]
final_tvec = 0
final_rvec = 0
while True:
    retr1 = 2000.6
    retr2 = 2000.7
    # print("1")
    t = clock()
    _ret, img = cam.read()
    if not _ret:
        continue
    _ret, img2 = cam2.read()
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
    # print("Corners", corners)
    # print("2")
    # """
    # If desired number of corner are detected,
    # we refine the pixel coordinates and display
    # them on the images of checker board
    # """
    # """
    # print("ret", ret, "\nret2", ret2)
    if ret or ret2:
        if len(imgpoints) > 5:
            imgpoints = []
        if len(imgpoints2) > 5:
            imgpoints2 = []
        # print("3")
        # refining pixel coordinates for given 2d points.
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
        if ret2:
            cornerss2 = cv2.cornerSubPix(gray2, cornerss, (11, 11), (-1, -1), criteria)
            imgpoints2.append(cornerss2)
        # print("4")
        # first_one = False
        # second_one = False

        if len(imgpoints) > 5:  # First camera
            RMS_one = False
            retr1, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            # first_one = True
            imgpts, jac = cv2.projectPoints(axis, rvecs[0], tvecs[0], mtx, dist)
            mean_error = 0
            for i in range(len(objpoints)):
                imgpointss2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i], imgpointss2, cv2.NORM_L2) / len(imgpointss2)
                mean_error += error
            if retr1 < RMS_thresh:
                # count = counter(count, 5)
                # if not count:
                tvec, rvec, retr1 = modify_details(retr1, rvecs, tvecs)
                print("total error: {}".format(mean_error / len(objpoints)))
                img = draw(img, imgpts)
                dt = clock() - t
                draw_str(img, (480, 20), 'time: %.1f ms' % (dt * 1000))
                draw_str(img, (20, 20), 'x rot: ' + str(rvec[0]))
                draw_str(img, (20, 40), 'y rot: ' + str(rvec[1]))
                draw_str(img, (20, 60), 'z rot: ' + str(rvec[2]))
                draw_str(img, (20, 80), 'x pos: ' + str(tvec[0]))
                draw_str(img, (20, 100), 'y pos: ' + str(tvec[1]))
                draw_str(img, (20, 120), 'z pos: ' + str(tvec[2]))
                draw_str(img, (20, 140), 'RMS Error: ' + str(retr1))
                cv2.imshow('img', img)
                cv2.imshow('img2', img2)
                cv2.waitKey(1)
                # if dst != []:
                # cv2.imshow("dst", dst)
            else:
                RMS_one = True
            # print("5")

        else:
            pass
            # print("not bad")
        if len(imgpoints2) > 5:  # Second camera
            RMS_two = False
            retr2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, gray2.shape[::-1], None,
                                                                     None)
            # second_one = True
            imgpts2, jac = cv2.projectPoints(axis, rvecs2[0], tvecs2[0], mtx2, dist2)
            mean_error = 0
            for i in range(len(objpoints)):
                imgpointss2, _ = cv2.projectPoints(objpoints[i], rvecs2[i], tvecs2[i], mtx2, dist2)
                error = cv2.norm(imgpoints2[i], imgpointss2, cv2.NORM_L2) / len(imgpointss2)
                mean_error += error
                if retr2 < RMS_thresh:
                    # count = counter(count, 5)
                    # if not count:
                    tvec, rvec, retr2 = modify_details(retr2, rvecs2, tvecs2)
                    if cross_over:
                        # sf_x = rvec[0] - sf_x
                        # sf_y = rvec[1] - sf_y
                        # sf_z = rvec[2] - sf_z
                        starting_pos = tvec
                        cross_over = False
                        print("subtraction factor x: ", sf_x_rot, "subtraction factor y: ", sf_y_rot, "subtraction factor z: ",
                              sf_z_rot)
                        print("rvec: ", rvec)
                        print("Cross Over Complete \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
                    if starting_pos[0]:
                        sf_z_tra, sf_y_tra, sf_x_tra = starting_pos[0] - tvec[0], starting_pos[1] - tvec[1], starting_pos[2] - tvec[2]
                    print("total error: {}".format(mean_error / len(objpoints)))
                    rvec = rvec[0] - sf_x_rot, rvec[1] - sf_y_rot, rvec[2] - sf_z_rot
                    tvec = final_pos[0] - sf_x_tra, final_pos[1] - sf_y_tra, final_pos[2] - sf_z_tra
                    print("Rotation Vector\nx: ", rvec[0], "y: ", rvec[1], "z: ", (rvec[2]))
                    print("Translation Vector\nx: ", tvec[0], "y: ", tvec[1], "z: ", (tvec[2]))
                    img2 = draw(img2, imgpts2)
                    dt = clock() - t
                    draw_str(img2, (480, 20), 'time: %.1f ms' % (dt * 1000))
                    draw_str(img2, (20, 20), 'x rot: ' + str(rvec[0]))
                    draw_str(img2, (20, 40), 'y rot: ' + str(rvec[1]))
                    draw_str(img2, (20, 60), 'z rot: ' + str(rvec[2]))
                    draw_str(img2, (20, 80), 'x pos: ' + str(tvec[0]))
                    draw_str(img2, (20, 100), 'y pos: ' + str(tvec[1]))
                    draw_str(img2, (20, 120), 'z pos: ' + str(tvec[2]))
                    draw_str(img2, (20, 140), 'RMS Error: ' + str(retr2))
                    draw_str(img2, (480, 60), "x: " + str(sf_x_rot))
                    draw_str(img2, (480, 80), "y: " + str(sf_y_rot))
                    draw_str(img2, (480, 100), "z: " + str(sf_z_rot))
                    draw_str(img2, (480, 120), "x: " + str(sf_x_tra))
                    draw_str(img2, (480, 140), "y: " + str(sf_y_tra))
                    draw_str(img2, (480, 160), "z: " + str(sf_z_tra))
                    cv2.imshow('img', img)
                    cv2.imshow('img2', img2)
                    cv2.waitKey(1)
                    final_tvec = tvec
                    final_rvec = rvec
                    # if dst != []:
                    # cv2.imshow("dst", dst)
                else:
                    RMS_two = True
                # print("5")
            else:
                pass
                # print("not bad")
            # _, rvec, tvec= cv2.solvePnP(objp, corners2, mtx, dist)
        if RMS_one:
            print("RMS Error Too Large One: ", retr1, retr2)
        if RMS_two:
            print("RMS Error Too Large Two: ", retr2, retr1)

        # print("Here1", RMS_one, RMS_two)
        if RMS_one and RMS_two:
            # print("Here2", RMS_one, RMS_two)
            dt = clock() - t
            if retr1 < retr2:
                img = draw(img, imgpts)
                draw_str(img, (480, 20), 'time: %.1f ms' % (dt * 1000))
                draw_str(img, (20, 20), 'x rot: ' + str(rvec[0]))
                draw_str(img, (20, 40), 'y rot: ' + str(rvec[1]))
                draw_str(img, (20, 60), 'z rot: ' + str(rvec[2]))
                draw_str(img, (20, 80), 'x pos: ' + str(tvec[0]))
                draw_str(img, (20, 100), 'y pos: ' + str(tvec[1]))
                draw_str(img, (20, 120), 'z pos: ' + str(tvec[2]))
                draw_str(img, (20, 140), 'RMS Error: ' + str(retr1))
                # sf_x, sf_y, sf_z = tvec
                cross_over = True
            if retr2 < retr1:
                img2 = draw(img2, imgpts2)
                draw_str(img2, (480, 20), 'time: %.1f ms' % (dt * 1000))
                draw_str(img2, (20, 20), 'x rot: ' + str(rvec[0]))
                draw_str(img2, (20, 40), 'y rot: ' + str(rvec[1]))
                draw_str(img2, (20, 60), 'z rot: ' + str(rvec[2]))
                draw_str(img2, (20, 80), 'x pos: ' + str(tvec[0]))
                draw_str(img2, (20, 100), 'y pos: ' + str(tvec[1]))
                draw_str(img2, (20, 120), 'z pos: ' + str(tvec[2]))
                draw_str(img2, (20, 140), 'RMS Error: ' + str(retr2))
            draw_str(img, (480, 40), 'Dead Zone')
            draw_str(img2, (480, 40), 'Dead Zone')
            cv2.imshow('img', img)
            cv2.imshow('img2', img2)
            cv2.waitKey(1)
            print("Dead Zone")
            count2 = 1
            count3 = 1
            out_of_dead_zone = False
            print(len(imgpoints), len(imgpoints2))

            while len(imgpoints) < 6 or len(imgpoints2) < 6:
                t = clock()
                _ret, img = cam.read()
                if not _ret:
                    continue
                _ret, img2 = cam2.read()
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
                dt = clock() - t
                draw_str(img, (480, 20), 'time: %.1f ms' % (dt * 1000))
                draw_str(img2, (480, 20), 'time: %.1f ms' % (dt * 1000))
                draw_str(img, (480, 40), 'Dead Zone')
                draw_str(img2, (480, 40), 'Dead Zone')
                cv2.imshow('img', img)
                cv2.imshow('img2', img2)
                cv2.waitKey(1)
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
                retr2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, gray2.shape[::-1], None, None)
                tvec, rvec, retr = modify_details(retr1, rvecs, tvecs)
                tvec2, rvec2, retr2 = modify_details(retr2, rvecs2, tvecs2)
                if final_tvec and final_rvec:
                    print("In dead Zone\nfinal_rvec: ", final_rvec, "\nrvec: ", rvec, "\nrvec2: ", rvec2, "\ntvec2: ", tvec2, "\nFinal tvec: ", final_tvec, "\ntvec: ", tvec)
                    print("Rotation Error: ", (abs(final_rvec[0] - rvec[0]) + abs(final_rvec[1] - rvec[1]) + abs(final_rvec[2] - rvec[2])) / 3)
                    print("Translation Error: ", (abs(final_tvec[0] - tvec[0]) + abs(final_tvec[1] - tvec[1]) + abs(final_tvec[2] - tvec[2])) / 3)
                else:
                    print("In dead Zone\nrvec: ", rvec, "\nrvec2: ", rvec2, "\ntvec: ", tvec, "\ntvec2: ", tvec2)

                sf_x_rot, sf_y_rot, sf_z_rot = rvec[0] - rvec2[0], rvec[1] - rvec2[1], rvec[2] - rvec2[2]
                sf_x_tra, sf_y_tra, sf_z_tra = tvec[0] - tvec2[0], tvec[1] - tvec2[1], tvec[2] - tvec2[2]

                details_rotation.append([sf_x_rot, sf_y_rot, sf_z_rot])
                details_translation.append([sf_x_tra, sf_y_tra, sf_z_tra])
            else:  # Out of Dead Zone
                RMS_one = False
                RMS_two = False
                if rvec != [] and rvec2 != []:
                    print("Out of dead Zone\nrvec: ", rvec, "\nrvec2: ", rvec2)
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
        elif retr1 < RMS_thresh and retr2 < RMS_thresh:
            print("Shared Zone")
            # h, w = img.shape[:2]
            # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            # crop the image
            # x, y, w, h = roi
            # dst = dst[y:y + h, x:x + w]
            # cv2.imshow('calibresult.png', dst)
            # imgpts2, _ = cv2.projectPoints(axis, rvecs[0], tvecs[0], mtx, 0)

            # Draw and display the corners
    else:
        dt = clock() - t
        draw_str(img, (20, 20), 'time: %.1f ms' % (dt * 1000))
        cv2.imshow('img', img)
        cv2.imshow('img2', img2)
        cv2.waitKey(1)
        print("No Chessboard Found")
cv2.destroyAllWindows()
