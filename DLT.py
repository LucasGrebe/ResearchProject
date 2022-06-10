"""MIT License

Copyright (c) 2019 Ankita Victor

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

 """

import numpy as np
import cv2 as cv

import ResearchPythonDLTDots


def Normalization(nd, x):
    '''
    Normalization of coordinates (centroid to the origin and mean distance of sqrt(2 or 3).

    Input
    -----
    nd: number of dimensions, 3 here
    x: the data to be normalized (directions at different columns and points at rows)
    Output
    ------
    Tr: the transformation matrix (translation plus scaling)
    x: the transformed data
    '''

    x = np.asarray(x)
    m, s = np.mean(x, 0), np.std(x)
    if nd == 2:
        Tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    else:
        Tr = np.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])
    Tr = np.linalg.inv(Tr)
    x = np.dot(Tr, np.concatenate((x.T, np.ones((1, x.shape[0])))))
    x = x[0:nd, :].T
    # print("Tr: ", Tr, "\nNormalized x: ", x)

    return Tr, x


def DLTcalib(nd, xyz, uv):
    '''
    Camera calibration by DLT using known object points and their image points.

    Input
    -----
    nd: dimensions of the object space, 3 here.
    xyz: coordinates in the object 3D space.
    uv: coordinates in the image 2D space.

    The coordinates (x,y,z and u,v) are given as columns and the different points as rows.

    There must be at least 6 calibration points for the 3D DLT.

    Output
    ------
     L: array of 11 parameters of the calibration matrix.
     err: error of the DLT (mean residual of the DLT transformation in units of camera coordinates).
    '''
    if (nd != 3):
        raise ValueError('%dD DLT unsupported.' % (nd))

    # Converting all variables to numpy array
    xyz = np.asarray(xyz)
    uv = np.asarray(uv)

    n = xyz.shape[0]

    # Validating the parameters:
    if uv.shape[0] != n:
        raise ValueError('Object (%d points) and image (%d points) have different number of points.' % (n, uv.shape[0]))

    if (xyz.shape[1] != 3):
        raise ValueError('Incorrect number of coordinates (%d) for %dD DLT (it should be %d).' % (xyz.shape[1], nd, nd))

    if (n < 6):
        raise ValueError(
            '%dD DLT requires at least %d calibration points. Only %d points were entered.' % (nd, 2 * nd, n))

    # Normalize the data to improve the DLT quality (DLT is dependent of the system of coordinates).
    # This is relevant when there is a considerable perspective distortion.
    # Normalization: mean position at origin and mean distance equals to 1 at each direction.
    Txyz, xyzn = Normalization(nd, xyz)
    Tuv, uvn = Normalization(2, uv)

    A = []

    for i in range(n):
        x, y, z = xyzn[i, 0], xyzn[i, 1], xyzn[i, 2]
        u, v = uvn[i, 0], uvn[i, 1]
        A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        # print("A before: ", A)
        A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])
        # print("A after:", A)
    # Convert A to array
    A = np.asarray(A)

    # Find the 11 parameters:
    _, _, V = np.linalg.svd(A)
    # print("U", U, "\nS", S, "\nV", V)

    # The parameters are in the last line of Vh and normalize them
    L = V[-1, :] / V[-1, -1]
    # print(L)
    # Camera projection matrix
    H = L.reshape(3, nd + 1)
    # print(H)

    # Denormalization
    # pinv: Moore-Penrose pseudo-inverse of a matrix, generalized inverse of a matrix using its SVD
    H = np.dot(np.dot(np.linalg.pinv(Tuv), H), Txyz)
    # print(H)
    H = (H/H[-1, -1])
    # print("H", H)
    # 'C, F, A, K'
    # L = H.flatten('K')
    # # print(L)
    # print(          np.concatenate((xyz.T, np.ones((1, xyz.shape[0])))))
    # Mean error of the DLT (mean residual of the DLT transformation in units of camera coordinates):
    uv2 = np.dot(H, np.concatenate((xyz.T, np.ones((1, xyz.shape[0])))))
    # print("before normal", uv2)
    uv2 = uv2 / uv2[2, :]
    # print("after normal", uv2)
    # Mean distance:
    print("uv2", uv2[0:2].T)
    err = np.sqrt(np.mean(np.sum((uv2[0:2, :].T - uv) ** 2, 1)))

    return H, err


def get_edge_most(temp):
    uv = []
    curr_pos = ()
    # Small
    right_most = -1
    for pos in temp:
        if pos[0] > right_most:
            right_most = pos[0]
            curr_pos = pos
    temp.remove(curr_pos)
    curr_pos = (float(curr_pos[0]), float(curr_pos[1]))
    uv.append(curr_pos)
    # Medium
    top_most = 1000
    for pos in temp:
        if pos[1] < top_most:
            top_most = pos[1]
            curr_pos = pos
    temp.remove(curr_pos)
    curr_pos = (float(curr_pos[0]), float(curr_pos[1]))
    uv.append(curr_pos)
    # Center
    right_most = -1
    for pos in temp:
        if pos[0] > right_most:
            right_most = pos[0]
            curr_pos = pos
    temp.remove(curr_pos)
    curr_pos = (float(curr_pos[0]), float(curr_pos[1]))
    uv.append(curr_pos)
    # Bottom
    bottom_most = -1
    for pos in temp:
        if pos[1] > bottom_most:
            bottom_most = pos[1]
            curr_pos = pos
    temp.remove(curr_pos)
    curr_pos = (float(curr_pos[0]), float(curr_pos[1]))
    uv.append(curr_pos)
    # Left
    left_most = 1000
    for pos in temp:
        if pos[0] < left_most:
            left_most = pos[0]
            curr_pos = pos
    temp.remove(curr_pos)
    curr_pos = (float(curr_pos[0]), float(curr_pos[1]))
    uv.append(curr_pos)
    # Large
    for i in temp:
        i = (float(i[0]), float(i[1]))
        uv.insert(2, i)
    print("uv", uv)
    return uv


def DLT():
    # Known 3D coordinates, order = Small, Medium, Large, Center, Bottom, Left
    xyz = [[10.75, 3, 0.75], [-11, 14, 1.25], [-8, 5, 1.5], [0, 0, 0], [-8.5, -6.5, 0],
           [-17, 3.5, 0]]
    temp = ResearchPythonDLTDots.main()[0]
    img = ResearchPythonDLTDots.main()[1]
    counter = 0
    while len(temp) < 6:
        temp = ResearchPythonDLTDots.main()[0]
        counter += 1
        print(len(temp))
        if counter > 10:
            raise ValueError('%dD DLT requires at least %d calibration points. Only %d points were entered.' % (3, 6, len(temp)))
    uv = get_edge_most(temp)
    nd = 3
    P, err = DLTcalib(nd, xyz, uv)
    print("error: ", err)

    position_vector = np.linalg.inv(P[:3, :3]) @ P[:3, 3]
    position_vector = [position_vector[0], position_vector[1], position_vector[2]]

    out = cv.decomposeProjectionMatrix(P)

    dist_coeffs = np.zeros((4, 1))
    xyz = np.asarray(xyz)
    uv = np.asarray(uv)
    # print("xyz", xyz, "\nuv", uv)
    # print("0", out[0], "\n1", out[1], "\n2", out[2])
    # print(points_2D, points_3D)
    success, rotatio_vector, translation_vector = cv.solvePnP(xyz, uv, out[0], dist_coeffs,
                                                               flags=0)

    rotation_vector = cv.Rodrigues(out[1])
    rotation_vector = [rotation_vector[0][0][0], rotation_vector[0][1][0], rotation_vector[0][2][0]]
    rotation_vector = np.asarray(rotation_vector)
    position_vector = np.asarray(position_vector)
    end_point2D, Jacobin = cv.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, position_vector,
                                                   out[0], dist_coeffs)
    print(end_point2D, Jacobin)

    print("Rotation Vector:\n", rotation_vector)
    print("PnP Rotation Vector:\n", rotatio_vector)
    print("Position Vector:\n", position_vector)
    print("PnP Position Vector:\n", translation_vector)
    point1 = (int(uv[0][0]), int(uv[0][1]))

    point2 = (int(end_point2D[0][0][0]), int(end_point2D[0][0][1]))

    cv.line(img, point1, point2, (255, 255, 255), 2)
    cv.imshow("img", img)
    cv.waitKey(0)


if __name__ == "__main__":
    DLT()
    cv.destroyAllWindows()
