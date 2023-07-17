import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
from PIL import ImageOps



def main():


    root = '/home/eryk-dev/Desktop/inzynieka/Calibration_Code/stereo_kalibracja_23_06_2023/'

    K1 = np.loadtxt('Intrinsic_mtx_1.txt', dtype=float).reshape((3, 3))
    K2 = np.loadtxt('Intrinsic_mtx_2.txt', dtype=float).reshape((3, 3))

    R = np.loadtxt('R.txt', dtype=float)
    T = np.loadtxt('T.txt', dtype=float)

    D1 = np.loadtxt('dist_1.txt', dtype=float)
    D2 = np.loadtxt('dist_2.txt', dtype=float)

    img_ext = "*.png"

    images_left = glob.glob(root + 'cam3/' + img_ext)
    images_right = glob.glob(root + 'cam5/' + img_ext)

    images_left.sort()
    images_right.sort()

    w, h = 640, 480


    R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(
            K1, D1,
            K2, D2,
            (w, h),
            R,
            T,
            flags=cv.CALIB_ZERO_DISPARITY,
            alpha= 0.0
        )

    mapx1, mapy1 = cv.initUndistortRectifyMap(
        K1, D1,
        R1, P1,
        (w, h),
        cv.CV_16SC2
    )
    mapx2, mapy2 = cv.initUndistortRectifyMap(
        K2, D2,
        R2, P2,
        (w, h),
        cv.CV_16SC2
    )


    for i, fname in enumerate(images_right):
        print("[" + str((i+1)) + "/" + str(len(images_right)) + "]")
        head1, tail1 = os.path.split(fname)
        head2, tail2 = os.path.split(images_left[i])
        print(tail1)

        r_path = images_right[i]
        l_path = images_left[i]

        
        img_l = cv.imread(images_left[i], cv.IMREAD_UNCHANGED)
        img_r = cv.imread(images_right[i], cv.IMREAD_UNCHANGED)

     
        h, w, c = img_l.shape

        img_rect1 = cv.remap(img_l, mapx1, mapy1, cv.INTER_LINEAR)
        img_rect2 = cv.remap(img_r, mapx2, mapy2, cv.INTER_LINEAR)

        h, w, c = img_rect1.shape


        tail1 = tail1.split('.')[0] + '.png'
        tail2 = tail2.split('.')[0] + '.png'

        cv.imwrite((root + 'cam2_rectyfied/' + tail1), img_rect1)
        cv.imwrite((root + 'cam4_rectyfied/' + tail2), img_rect2)

        

if __name__ == '__main__':
    main()
