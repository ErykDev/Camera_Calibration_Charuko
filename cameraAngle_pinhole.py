import numpy as np
import cv2
import math


K1 = np.loadtxt('Intrinsic_mtx_1.txt', dtype=float).reshape((3, 3))
D1 = np.loadtxt('dist_1.txt', dtype=float)

K2 = np.loadtxt('Intrinsic_mtx_2.txt', dtype=float).reshape((3, 3))
D2 = np.loadtxt('dist_2.txt', dtype=float)

T = np.loadtxt('T.txt', dtype=float)
R = np.loadtxt('R.txt', dtype=float)


pixel_size = 0.003
image_pixel_size_x = 640
image_pixel_size_y = 480

imageSize=(image_pixel_size_x, image_pixel_size_y)

sensor_size_x = image_pixel_size_x * pixel_size
sensor_size_y = image_pixel_size_y * pixel_size


R1 = np.zeros(shape=(3,3))
R2 = np.zeros(shape=(3,3))
P1 = np.zeros(shape=(3,4))
P2 = np.zeros(shape=(3,4))

cv2.stereoRectify(K1, D1, K2, D2, imageSize, R, T, R1, R2, P1, P2, alpha=0, flags=cv2.CALIB_ZERO_DISPARITY)


fov_x, fov_y, focal_len, principal, aspect = \
    cv2.calibrationMatrixValues(P1[:3, :3], (image_pixel_size_x, image_pixel_size_y),
                                sensor_size_x, sensor_size_y)  

focalv2 = focal_len / pixel_size      

print('original size')
print(f'sensor_size_x:{sensor_size_x}')
print(f'sensor_size_y:{sensor_size_y}')
print(f'baseline:{np.linalg.norm(T) * 0.1}')
print(f'fov_x:{fov_x}')
print(f'fov_y:{fov_y}')
print(f'focal_len mm:{focal_len}')
print(f'focal: {focalv2}') 
print(f'principal:{principal}')
print(f'aspect:{aspect}')
