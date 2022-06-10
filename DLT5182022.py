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
        A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])

    # Convert A to array
    A = np.asarray(A)

    # Find the 11 parameters:
    U, S, V = np.linalg.svd(A)

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
    # print(H)
    # 'C, F, A, K'
    # L = H.flatten('K')
    # # print(L)

    # Mean error of the DLT (mean residual of the DLT transformation in units of camera coordinates):
    uv2 = np.dot(H, np.concatenate((xyz.T, np.ones((1, xyz.shape[0])))))
    uv2 = uv2 / uv2[2, :]
    # Mean distance:
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
    uv.append(curr_pos)
    temp.remove(curr_pos)
    # Medium
    top_most = 1000
    for pos in temp:
        if pos[1] < top_most:
            top_most = pos[1]
            curr_pos = pos
    uv.append(curr_pos)
    temp.remove(curr_pos)
    # Center
    right_most = -1
    for pos in temp:
        if pos[0] > right_most:
            right_most = pos[0]
            curr_pos = pos
    uv.append(curr_pos)
    temp.remove(curr_pos)
    # Bottom
    bottom_most = -1
    for pos in temp:
        if pos[1] > bottom_most:
            bottom_most = pos[1]
            curr_pos = pos
    uv.append(curr_pos)
    temp.remove(curr_pos)
    # Left
    left_most = 1000
    for pos in temp:
        if pos[0] < left_most:
            left_most = pos[0]
            curr_pos = pos
    uv.append(curr_pos)
    temp.remove(curr_pos)
    # Large
    for i in temp:
        uv.insert(2, i)
    print("uv", uv, "\nTemp: ", temp)
    return uv


def DLT():
    # Known 3D coordinates, order = Small, Medium, Large, Center, Bottom, Left
    xyz = [[10.75, 3, 0.75], [-11, 14, 1.25], [-8, 5, 1.5], [0, 0, 0], [-8.5, -6.5, 0],
           [-17, 3.5, 0]]
    # Known pixel coordinates. [[, ], [, ], [, ], [, ], [, ], [, ]]
    # Easy
    temp = ResearchPythonDLTDots.main()
    print(temp)
    uv1 = get_edge_most(temp)
    # uv1 = [[1050, 572], [644, 269], [705, 488], [854, 601], [688, 723], [520, 530]]
    # Medium
    uv2 = [[3123, 1249], [1873, 435], [2069, 962], [2539, 1327], [2045, 1656], [1511, 1093]]
    # Hard
    uv3 = [[2139, 2336], [1690, 1022], [1514, 1640], [1672, 2153], [1052, 2246], [948, 1442]]

    nd = 3
    P1, err1 = DLTcalib(nd, xyz, uv1)
    P2, err2 = DLTcalib(nd, xyz, uv2)
    P3, err3 = DLTcalib(nd, xyz, uv3)
    # print('Matrix')
    # print(P)
    # print('\nError')
    print("error: easy: ", err1)
    print("error: medium: ", err2)
    print("error: hard: ", err3)

    position_vector1 = np.linalg.inv(P1[:3, :3]) @ P1[:3, 3]
    position_vector2 = np.linalg.inv(P2[:3, :3]) @ P2[:3, 3]
    position_vector3 = np.linalg.inv(P3[:3, :3]) @ P3[:3, 3]
    position_vector1 = [position_vector1[0], position_vector1[1], position_vector1[2]]
    position_vector2 = [position_vector2[0], position_vector2[1], position_vector2[2]]
    position_vector3 = [position_vector3[0], position_vector3[1], position_vector3[2]]
    print("Position Vector 1:\n", position_vector1)
    print("Position Vector 2:\n", position_vector2)
    print("Position Vector 3:\n", position_vector3)

    out1 = cv.decomposeProjectionMatrix(P1)
    out2 = cv.decomposeProjectionMatrix(P2)
    out3 = cv.decomposeProjectionMatrix(P3)

    # print("0", out[0], "\n1", out[1], "\n2", out[2])
    rotation_vector1 = cv.Rodrigues(out1[1])
    rotation_vector2 = cv.Rodrigues(out2[1])
    rotation_vector3 = cv.Rodrigues(out3[1])
    rotation_vector1 = [rotation_vector1[0][0][0], rotation_vector1[0][1][0], rotation_vector1[0][2][0]]
    rotation_vector2 = [rotation_vector2[0][0][0], rotation_vector2[0][1][0], rotation_vector2[0][2][0]]
    rotation_vector3 = [rotation_vector3[0][0][0], rotation_vector3[0][1][0], rotation_vector3[0][2][0]]
    print("Rotation Vector 1:\n", rotation_vector1)
    print("Rotation Vector 2:\n", rotation_vector2)
    print("Rotation Vector 3:\n", rotation_vector3)


if __name__ == "__main__":
    DLT()
