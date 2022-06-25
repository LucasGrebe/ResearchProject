import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    img = cap.read()[1]

    dim = (1000, 2000)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    resized = np.zeros_like(resized)
    resized[resized < 5] = 255
    # cv2.circle(resized, (500, 500), 350, (0, 0, 0), 30)
    cv2.circle(resized, (500, 500), 310, (0, 0, 0), 15)
    cv2.circle(resized, (500, 500), 250, (0, 0, 0), 15)
    cv2.circle(resized, (500, 500), 190, (0, 0, 0), 15)
    cv2.circle(resized, (500, 500), 130, (0, 0, 0), 15)
    cv2.circle(resized, (500, 500), 70, (0, 0, 0), 15)
    cv2.circle(resized, (500, 475), 20, (0, 0, 255), cv2.FILLED)
    cv2.imshow("Resized image", resized)
    cv2.imwrite('temp.jpg', resized)
    # cv2.waitKey(1)
