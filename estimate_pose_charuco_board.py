import cv2
import numpy as np

from matplotlib import cm, colors

import cv2.aruco as aruco


from calibration_utils import *

board_shape = [7, 5]

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters =  aruco.DetectorParameters()
arucoDetector = aruco.ArucoDetector(dictionary, parameters)

board = aruco.CharucoBoard(board_shape, 38, 30.0, dictionary)


w, h = 1280, 720

left_cam = cv2.VideoCapture(2)

frames = 0

K1 = np.loadtxt('Intrinsic_mtx_1.txt', dtype=float).reshape((3, 3))


D1 = np.loadtxt('dist_1.txt', dtype=float)


#roi can be ignored sine we using alpha=0
M1_opt, roi_1 = cv2.getOptimalNewCameraMatrix(K1, D1, (w,h), 0, (w,h))


pixel_size = 0.003


sensor_size_x = w * pixel_size
sensor_size_y = h * pixel_size

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

corner_ids = cornerIds(board_shape)

setupCam(left_cam, w, h)


while True:
    left_cam.grab()
    ret1, frame_left = left_cam.retrieve()

    frames = frames + 1

    if not ret1:
        print("failed to grab frame")
        continue
    else:
        frame_left = cv2.undistort(frame_left, K1, D1, None, M1_opt)

        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)


        corners_left,  ids_left,  rejectedCorners1 = arucoDetector.detectMarkers(gray_left)

        if len(corners_left) > 4:
            aruco.refineDetectedMarkers(gray_left, board, corners_left, ids_left, rejectedCorners1)
            frame_left = aruco.drawDetectedMarkers(frame_left, corners_left, ids_left)

          
        if len(corners_left) > 4:
            if corners_left is not None and len(ids_left)>3 and max(ids_left) <= max(board.getIds()) and max(ids_left) <= max(board.getIds()):

                corners_left = np.array(corners_left, dtype=np.float32)

                if len(corners_left) == len(ids_left):
                    if is_slice_in_list(numpy.squeeze(ids_left).tolist(), corner_ids): # all left corners are detected
                        charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners_left, ids_left, gray_left, board)
                        
                        left_valid, l_rvec, l_tvec = aruco.estimatePoseCharucoBoard(
                            charucoCorners=charucoCorners, charucoIds=charucoIds, board=board, \
                                cameraMatrix=M1_opt, distCoeffs=D1, rvec=None, tvec=None,\
                                    useExtrinsicGuess=False)

                        if left_valid:
                            #print(l_rvec_t.shape)
                            cv2.drawFrameAxes(frame_left, M1_opt, D1, l_rvec, l_tvec, 120)


        frame_left_down = cv2.resize(frame_left, (int(frame_left.shape[1] / 1.2), int(frame_left.shape[0] / 1.5)), interpolation= cv2.INTER_LINEAR)

        cv2.imshow("left/right camera", frame_left_down)


    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif(k%256 == ord('s')):
        print('save clicked')

        cv2.imwrite('left.png', frame_left)


left_cam.release()

cv2.destroyAllWindows()
