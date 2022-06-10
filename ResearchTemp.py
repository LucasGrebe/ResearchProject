import numpy as np
import cv2

# Read an image
img = cv2.imread("unnamed.png")
rows, cols, _ = img.shape

# Create the transformation matrix
angle = np.radians(45)
x0, y0 = ((cols - 1) / 2.0, (rows - 1) / 2.0)
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
    (transform_coord[1] < rows, transform_coord[0] < cols, transform_coord[1] >= 0, transform_coord[0] >= 0), axis=0)
# Create a zeros image and project the points
img1 = np.zeros_like(img)
img1[transform_coord[1][indices], transform_coord[0][indices]] = img[orig_coord[1][indices], orig_coord[0][indices]]
# Display the image
out = cv2.hconcat([img, img1])
cv2.imshow('a2', out)
cv2.waitKey(0)
