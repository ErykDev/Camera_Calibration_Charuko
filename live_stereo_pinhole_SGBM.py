from dis import dis
import cv2
import time
import numpy as np
import matplotlib
import json

from matplotlib import cm, colors


def disp_to_depth(disp, focal, bs):
    depth = (focal * bs) / disp
    mask = np.isfinite(depth)

    depth[np.logical_not(mask)] = 0.0

    return depth


def get_color_m(cmap='Reds', lo=None, hi=None):
    color_map = cm.get_cmap(cmap)
    return lambda x: colors.rgb2hex(color_map((x-lo)/(hi-lo))[:3])


w, h = 640, 480

left_cam = cv2.VideoCapture(2)
right_cam = cv2.VideoCapture(4)

frames = 0

K1 = np.loadtxt('Intrinsic_mtx_1.txt', dtype=float).reshape((3, 3))
K2 = np.loadtxt('Intrinsic_mtx_2.txt', dtype=float).reshape((3, 3))

R = np.loadtxt('R.txt', dtype=float)
T = np.loadtxt('T.txt', dtype=float)

D1 = np.loadtxt('dist_1.txt', dtype=float)
D2 = np.loadtxt('dist_2.txt', dtype=float)


R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K1, D1,
    K2, D2,
    (w, h),
    R,
    T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha= 0.0
)

mapx1, mapy1 = cv2.initUndistortRectifyMap(
    K1, D1,
    R1, P1,
    (w, h),
    cv2.CV_16SC2
)
mapx2, mapy2 = cv2.initUndistortRectifyMap(
    K2, D2,
    R2, P2,
    (w, h),
    cv2.CV_16SC2
)


fName = '3dmap_set.txt'
print('Loading parameters from file...')
f=open(fName, 'r')
data = json.load(f)
sSWS = data['SADWindowSize']
sPFS = data['preFilterSize']
sPFC = data['preFilterCap']
sMDS = data['minDisparity']
sNOD = data['numberOfDisparities']
sTTH = data['textureThreshold']
sUR = data['uniquenessRatio']
sSR = data['speckleRange']
sSPWS = data['speckleWindowSize']    
f.close()



sbm = cv2.StereoSGBM_create(numDisparities=256, blockSize=15)
sbm.setPreFilterCap(sPFC)
sbm.setMinDisparity(sMDS)
sbm.setNumDisparities(sNOD)
sbm.setUniquenessRatio(sUR)
sbm.setSpeckleRange(sSR)
sbm.setSpeckleWindowSize(sSPWS)

cmap = matplotlib.cm.get_cmap('gist_rainbow')

while True:
    left_cam.grab()
    ret1, frame_left = left_cam.retrieve()

    right_cam.grab()
    ret1, frame_right = right_cam.retrieve()

    frames = frames + 1

    if not ret1:
        print("failed to grab frame")
        continue
    else:

        frame_left = cv2.remap(frame_left, mapx1, mapy1, cv2.INTER_LINEAR)
        frame_right = cv2.remap(frame_right, mapx2, mapy2, cv2.INTER_LINEAR)

        img_rect1 = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        img_rect2 = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)


        disparity = sbm.compute(img_rect1, img_rect2)

        depth = disp_to_depth(disparity, 528.8050033826225, 11.03 * 0.1)

        

        MAX_DEPTH = 5.0

        depth_visual = 255 * cmap((depth / MAX_DEPTH)) #cmap works on ranges 0-1
        depth_visual = depth_visual.astype(np.uint8) # opencv expects uint8 on flot image 

        depth_visual[depth < 0.2] = [0, 0, 0, 0] # removing to close objects 

        img_rect1_f = cv2.cvtColor(img_rect1, cv2.COLOR_BGR2BGRA)
        depth_visual_f = cv2.cvtColor(depth_visual, cv2.COLOR_RGBA2BGRA)

        #added_image = cv2.addWeighted(img_rect1_f,0.9,depth_visual_f,0.55,0)
        conc = cv2.vconcat([img_rect1_f, depth_visual_f])

        cv2.imshow("depth", conc)


    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif(k%256 == ord('s')):
        print('save clicked')

        disparity_visual = np.asarray(disparity_visual * 256.0, dtype=np.uint16)

        cv2.imwrite('disp.png', disparity_visual)
        cv2.imwrite('left.png', frame_left)
        cv2.imwrite('right.png', frame_right)


left_cam.release()
right_cam.release()

cv2.destroyAllWindows()
