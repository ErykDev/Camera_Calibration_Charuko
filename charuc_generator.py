import numpy as np
import cv2, PIL, os
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('GTK3Agg')

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)

board = aruco.CharucoBoard_create(11, 7, 25, 19, aruco_dict)

imboard = board.draw((2000, 2000))
cv2.imwrite("chessboard.png", imboard)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.imshow(imboard, cmap = mpl.cm.gray, interpolation = "nearest")
ax.axis("off")
plt.show()