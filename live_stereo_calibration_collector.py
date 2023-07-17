from ast import arg
import math
import cv2
import numpy
import numpy as np
import argparse
import os
import os.path
import time
from os import path
from datetime import date

from calibration_utils import *

from cv2 import aruco


parser = argparse.ArgumentParser(description='live_frame_collector')



parser.add_argument('--threshold', type=float, default=0.2,
                    help='detection quality threshold')

parser.add_argument('--difficulty_mulitpl', type=int, default=10,
                    help='detection quality threshold')

parser.add_argument('--save_path',
                    help='path to save frames')

parser.add_argument('--camera_ids', nargs='+', type=int, default=(2,4),
                    help='Array of cameras_ids')

parser.add_argument('--cameras_frame_shape', nargs='+', type=int, default=(1280, 720),
                    help='Expected frame shape w h')

parser.add_argument('--board_shape', nargs='+', type=int, default=(7,5),
                    help='expected board shape')

parser.add_argument('--pattern_dimension', type=int, default=20,
                    help='pattern dimension of each squere in mm')

args = parser.parse_args()



dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters =  aruco.DetectorParameters()
arucoDetector = aruco.ArucoDetector(dictionary, parameters)

board = aruco.CharucoBoard(args.board_shape, 38, 30.0, dictionary)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

db_left = []
db_right = []


img_shape = args.cameras_frame_shape


def main():
    # define a video capture object
    left_cam = cv2.VideoCapture(args.camera_ids[0])
    right_cam = cv2.VideoCapture(args.camera_ids[1])

    setupCam(left_cam, img_shape[0], img_shape[1])
    setupCam(right_cam, img_shape[0], img_shape[1])

    collected_frames = 0

    decimator = 0

    allCornersLeft = []
    allCornersRight = []
    allIdsLeft = []
    allIdsRight = []

    kalib_folder = args.save_path + '/stereo_kalibracja_' + date.today().strftime("%d_%m_%Y") + '/'

    if(not path.exists(kalib_folder)):
        os.mkdir(kalib_folder)

    left_cam_save_path   = kalib_folder + '/cam' + str(int(args.camera_ids[0])) + '/'
    right_cam_save_path  = kalib_folder + '/cam' + str(int(args.camera_ids[1])) + '/'


    if(not path.exists(left_cam_save_path)):
        os.mkdir(left_cam_save_path)

    if(not path.exists(right_cam_save_path)):
        os.mkdir(right_cam_save_path)


    corner_ids = cornerIds(args.board_shape)


    # Main loop
    while(True):
        # Capture the video frame
        # by frame
        ret, frame_left = left_cam.read()
        ret1, frame_right = right_cam.read()
        
        frame_left_copy = np.copy(frame_left)
        frame_right_copy = np.copy(frame_right)

        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)



        corners_left,  ids_left,  rejectedCorners1 = arucoDetector.detectMarkers(gray_left)
        corners_right, ids_right, rejectedCorners2 = arucoDetector.detectMarkers(gray_right)

        if len(corners_left) > 16:
            aruco.refineDetectedMarkers(gray_left, board, corners_left, ids_left, rejectedCorners1)
                
            frame_left_copy = aruco.drawDetectedMarkers(frame_left_copy, corners_left, ids_left)

        if len(corners_right) > 16:
            aruco.refineDetectedMarkers(gray_right, board, corners_right, ids_right, rejectedCorners2)
                
            frame_right_copy = aruco.drawDetectedMarkers(frame_right_copy, corners_right, ids_right)
                 
        
        if len(corners_left) > 16 and len(corners_right) > 16:
            res2_left  = cv2.aruco.interpolateCornersCharuco(corners_left, ids_left, gray_left, board)
            res2_right = cv2.aruco.interpolateCornersCharuco(corners_right, ids_right, gray_right, board)

           
            if res2_left[1] is not None and res2_right[1] is not None and res2_left[2] is not None and len(res2_left[1])>3 and decimator%1==0 and max(ids_left) <= max(board.getIds()) and max(ids_left) <= max(board.getIds()):

                if is_slice_in_list(numpy.squeeze(ids_left).tolist(), corner_ids): # all left corners are detected
                    if is_slice_in_list(numpy.squeeze(ids_right).tolist(), corner_ids): # all left corners are detected

                        params_left, board_rot_deg_left = get_parameters(corners_left, numpy.squeeze(ids_left).tolist(), corner_ids, img_shape, args.board_shape)
                        params_right, board_rot_deg_right = get_parameters(corners_right, numpy.squeeze(ids_right).tolist(), corner_ids, img_shape, args.board_shape)

                        if(len(db_left) == 0):
                            db_left.append(params_left)
                            db_right.append(params_right)

                            allCornersLeft.append(res2_left[1])
                            allCornersRight.append(res2_right[1])

                            allIdsLeft.append(res2_left[2])
                            allIdsRight.append(res2_right[2])

                            collected_frames += 1 

                             #Saving frames
                            cv2.imwrite(left_cam_save_path + "/" + str(collected_frames) + ".png", frame_left)
                            cv2.imwrite(right_cam_save_path + "/" + str(collected_frames) + ".png", frame_right)


                        if is_good_sample(params_left, db_left, args.threshold) or is_good_sample(params_right, db_right, args.threshold):
                            db_left.append(params_left)
                            db_right.append(params_right)

                            allCornersLeft.append(res2_left[1])
                            allCornersRight.append(res2_right[1])

                            allIdsLeft.append(res2_left[2])
                            allIdsRight.append(res2_right[2])

                            collected_frames += 1 

                             #Saving frames
                            cv2.imwrite(left_cam_save_path + "/" + str(collected_frames) + ".png", frame_left)
                            cv2.imwrite(right_cam_save_path + "/" + str(collected_frames) + ".png", frame_right)


                            print('left')
                            print(compute_goodenough(db_left, args.difficulty_mulitpl))
                            print('right')
                            print(compute_goodenough(db_right, args.difficulty_mulitpl))
                            print('')


                
                        if(collected_frames > 90 and collected_frames % 10 == 0):
                            collected_frames += 1 

                            print("CAMERA CALIBRATION")

                            cameraMatrixInit = np.array([[ 1000.,    0., args.cameras_frame_shape[0] / 2.],
                                                        [    0., 1000.,  args.cameras_frame_shape[1] / 2.],
                                                        [    0.,    0.,           1.]])

                            distCoeffsInit = np.zeros((5,1))

                            flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
                            #flags = (cv2.CALIB_RATIONAL_MODEL)

                            
                            (ret_left, camera_matrix_left, distortion_coefficients_left, _, _, _, _,
                            _) = aruco.calibrateCameraCharucoExtended(
                                            charucoCorners=allCornersLeft,
                                            charucoIds=allIdsLeft,
                                            board=board,
                                            imageSize=img_shape,
                                            cameraMatrix=cameraMatrixInit.copy(),
                                            distCoeffs=distCoeffsInit.copy(),
                                            flags=flags,
                                            criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
                            
                            (ret_right, camera_matrix_right, distortion_coefficients_right, _, _, _, _,
                            _) = aruco.calibrateCameraCharucoExtended(
                                            charucoCorners=allCornersRight,
                                            charucoIds=allIdsRight,
                                            board=board,
                                            imageSize=img_shape,
                                            cameraMatrix=cameraMatrixInit.copy(),
                                            distCoeffs=distCoeffsInit.copy(),
                                            flags=flags,
                                            criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
                            

                            shared_corners_l, shared_ids_l, shared_corners_r, \
                                shared_ids_r = getSharedFetures(
                                    allCornersLeft, 
                                    allIdsLeft, 
                                    allCornersRight, 
                                    allIdsRight, 
                                    board)
                            

                            imgPoints_l, objPoints_l = calculateImgPointsObjPoints(shared_ids_l, shared_corners_l, board)
                            imgPoints_r, objPoints_r = calculateImgPointsObjPoints(shared_ids_r, shared_corners_r, board)


                            ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
                                objPoints_l, 
                                imgPoints_l,
                                imgPoints_r,
                                camera_matrix_left, distortion_coefficients_left, 
                                camera_matrix_right, distortion_coefficients_right, 
                                np.array(img_shape, np.int16),
                                criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-10), 
                                flags=flags
                            )

                            print('Intrinsic_mtx_1', M1)
                            print('Intrinsic_mtx_2', M2)

                            print('dist_1', d1)
                            print('dist_2', d2)

                            print('R', R)
                            print('T', T)
                            print('E', E)
                            print('F', F)

                            print('Baseline', np.linalg.norm(T) * 0.1)

                            print("error: {}".format(ret))


                            np.savetxt('Intrinsic_mtx_1.txt', M1)
                            np.savetxt('Intrinsic_mtx_2.txt', M2)

                            np.savetxt('dist_1.txt', d1)
                            np.savetxt('dist_2.txt', d2)

        decimator+=1

                            
        # Display the resulting frame
        cv2.imshow('View Display',  cv2.hconcat((frame_left_copy, frame_right_copy)))

        k = cv2.waitKey(1) # wait for key press

        if k%256 == ord('s'):
            #Saving frames
            cv2.imwrite(left_cam_save_path + "/" + str(collected_frames) + ".png", frame_left)
            cv2.imwrite(right_cam_save_path + "/" + str(collected_frames) + ".png", frame_right)

            collected_frames += 1 
        elif(k%256 == ord('q')):
            break

    # After the loop release the cap object
    left_cam.release()
    right_cam.release()

    # Destroy all the windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
