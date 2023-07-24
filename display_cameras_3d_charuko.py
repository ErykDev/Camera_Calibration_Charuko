import cv2
import numpy as np

import cv2.aruco as aruco

from calibration_utils import *

import matplotlib
matplotlib.use('GTK3Agg')

import matplotlib.pyplot as plt

import math


def rotation_matrix_to_attitude_angles(R) :
    cos_beta = math.sqrt(R[2,1] * R[2,1] + R[2,2] * R[2,2])
    validity = cos_beta < 1e-6
    if not validity:
        alpha = math.atan2(R[1,0], R[0,0])    # yaw   [z]
        beta  = math.atan2(-R[2,0], cos_beta) # pitch [y]
        gamma = math.atan2(R[2,1], R[2,2])    # roll  [x]
    else:
        alpha = math.atan2(R[1,0], R[0,0])    # yaw   [z]
        beta  = math.atan2(-R[2,0], cos_beta) # pitch [y]
        gamma = 0                             # roll  [x]

    return alpha, beta, gamma



board_shape = [7, 5]

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters =  aruco.DetectorParameters()
arucoDetector = aruco.ArucoDetector(dictionary, parameters)

board = aruco.CharucoBoard(board_shape, 38, 30.0, dictionary)


w, h = 1280, 720

left_cam = cv2.VideoCapture(2)
right_cam = cv2.VideoCapture(4)

frames = 0

K1 = np.loadtxt('Intrinsic_mtx_1.txt', dtype=float).reshape((3, 3))
D1 = np.loadtxt('dist_1.txt', dtype=float)

K2 = np.loadtxt('Intrinsic_mtx_2.txt', dtype=float).reshape((3, 3))
D2 = np.loadtxt('dist_2.txt', dtype=float)


#roi can be ignored sine we using alpha=0
M1_opt, roi_1 = cv2.getOptimalNewCameraMatrix(K1, D1, (w,h), 0, (w,h))
M2_opt, roi_2 = cv2.getOptimalNewCameraMatrix(K2, D2, (w,h), 0, (w,h))

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

corner_ids = cornerIds(board_shape)

setupCam(left_cam, w, h)
setupCam(right_cam, w, h)

# Create an empty figure and axis for visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

while True:
    left_cam.grab()
    right_cam.grab()

    ret1, frame_left = left_cam.retrieve()
    ret2, frame_right = right_cam.retrieve()

    frames = frames + 1

    # Clear the axis
    ax.cla()

    if not ret1:
        print("failed to grab frame")
        continue
    if not ret2:
        print("failed to grab frame")
        continue
    else:
        frame_left = cv2.undistort(frame_left, K1, D1, None, M1_opt)
        frame_right = cv2.undistort(frame_right, K2, D2, None, M2_opt)

        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)


        corners_left,  ids_left,  rejectedCorners1 = arucoDetector.detectMarkers(gray_left)
        corners_right,  ids_right,  rejectedCorners2 = arucoDetector.detectMarkers(gray_right)


        if len(corners_left) > 4:

            aruco.refineDetectedMarkers(gray_left, board, corners_left, ids_left, rejectedCorners1)
            frame_left = aruco.drawDetectedMarkers(frame_left, corners_left, ids_left)

            if corners_left is not None and len(ids_left)>3 and max(ids_left) <= max(board.getIds()):

                corners_left = np.array(corners_left, dtype=np.float32)

                if is_slice_in_list(numpy.squeeze(ids_left).tolist(), corner_ids): # all left corners are detected
                    charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners_left, ids_left, gray_left, board)
                    # Estimate camera pose
                    left_valid, rvec, tvec = aruco.estimatePoseCharucoBoard(
                        charucoCorners=charucoCorners, charucoIds=charucoIds, board=board, \
                            cameraMatrix=M1_opt, distCoeffs=D1, rvec=None, tvec=None,\
                                useExtrinsicGuess=False)

                    if left_valid:
                        # Invert the translation vector
                        #tvec = -tvec

                        # Create a transformation matrix
                        R, _ = cv2.Rodrigues(rvec)
                        T = np.concatenate((R, tvec), axis=1)
                        T = np.vstack((T, [0, 0, 0, 1]))

                        # Create a point at the origin board
                        origin = np.array([[0, 0, 0, 1]])

                        # Transform the origin using the camera pose
                        left_camera_pos = np.dot(T, origin.T).T

                        # mm to meters 
                        left_camera_pos = left_camera_pos * 0.001 

                        # Plot the transformed origin
                        ax.scatter(left_camera_pos[0, 0], left_camera_pos[0, 1], left_camera_pos[0, 2], c='red', label='Camera_1')



                        #cv2.drawFrameAxes(frame_left, M1_opt, D1, rvec, tvec, 120)
                        #frame_left = cv2.resize(frame_left, (int(frame_left.shape[1] / 1.2), int(frame_left.shape[0] / 1.5)), interpolation= cv2.INTER_LINEAR)
                else:
                    left_valid = False

        if len(corners_right) > 4:

            aruco.refineDetectedMarkers(gray_right, board, corners_right, ids_right, rejectedCorners2)
            frame_right = aruco.drawDetectedMarkers(frame_right, corners_right, ids_right)
            
            if corners_right is not None and len(ids_right)>3 and max(ids_right) <= max(board.getIds()):

                corners_right = np.array(corners_right, dtype=np.float32)

                if is_slice_in_list(numpy.squeeze(ids_right).tolist(), corner_ids): # all left corners are detected
                    charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners_right, ids_right, gray_right, board)
                    # Estimate camera pose
                    right_valid, rvec, tvec = aruco.estimatePoseCharucoBoard(
                        charucoCorners=charucoCorners, charucoIds=charucoIds, board=board, \
                            cameraMatrix=M1_opt, distCoeffs=D2, rvec=None, tvec=None,\
                                useExtrinsicGuess=False)

                    if right_valid:
                        # Invert the translation vector
                        #tvec = -tvec

                        # Create a transformation matrix
                        R, _ = cv2.Rodrigues(rvec)
                        T = np.concatenate((R, tvec), axis=1)
                        T = np.vstack((T, [0, 0, 0, 1]))

                        # Create a point at the origin board
                        origin = np.array([[0, 0, 0, 1]])

                        # Transform the origin using the camera pose
                        right_camera_pos = np.dot(T, origin.T).T

                        # mm to meters 
                        right_camera_pos = right_camera_pos * 0.001 

                        # Plot the transformed origin
                        ax.scatter(right_camera_pos[0, 0], right_camera_pos[0, 1], right_camera_pos[0, 2], c='red', label='Camera_2')

                                                #print(l_rvec_t.shape)
                        #cv2.drawFrameAxes(frame_left, M1_opt, D1, rvec, tvec, 120)
                        #frame_left = cv2.resize(frame_left, (int(frame_left.shape[1] / 1.2), int(frame_left.shape[0] / 1.5)), interpolation= cv2.INTER_LINEAR)
                else:
                    right_valid = False
                    

        if len(corners_left) > 4 and len(corners_right) > 4:
            if left_valid and right_valid:
                bs = math.hypot(
                    right_camera_pos[0, 0] - left_camera_pos[0, 0], 
                    right_camera_pos[0, 1] - left_camera_pos[0, 1], 
                    right_camera_pos[0, 2] - left_camera_pos[0, 2])
                
                print('baseline: {bs} cm'.format(bs=bs * 100))


        # Set the limits and labels
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(0, 1.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')


        #print(' x: {x}, y: {y} z: {z} '.format(x = camera_pos[0, 0], y = camera_pos[0, 1], z = camera_pos[0, 2]))

    
        chessboard_corners = board.getChessboardCorners()
        #print(chessboard_corners)

        # Plot the ChArUco board
        for corner in chessboard_corners:
            x = corner[0] * 0.001 
            y = corner[1] * 0.001 
            ax.scatter(x, y, 0, c='blue')
        
        
        # Add a legend
        ax.legend()

        # Draw the plot
        plt.draw()
        plt.pause(0.01)




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
