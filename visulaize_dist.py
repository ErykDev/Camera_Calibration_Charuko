import cv2
from opencv_camera import visualizeDistortion
import numpy as np

import matplotlib
matplotlib.use('GTK3Agg')

import matplotlib.pyplot as plt


K = np.loadtxt('Intrinsic_mtx_1.txt', dtype=float).reshape((3, 3))
D = np.loadtxt('dist_1.txt', dtype=float)

visualizeDistortion(K, D, h=720, w=1280)

plt.show()