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

from cv2 import aruco

from calibration_utils import *

def setupCam(cam, w, h):
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    print('setting camera')

    time.sleep(1)

    print('setting resolution')
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)

    time.sleep(1)

    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    time.sleep(1)

    print('setting fps speed')
    cam.set(cv2.CAP_PROP_FPS, 30.000)


parser = argparse.ArgumentParser(description='live_frame_collector')

parser.add_argument('--threshold', type=float, default=0.2,
                    help='detection quality threshold')

parser.add_argument('--difficulty_mulitpl', type=int, default=10,
                    help='detection quality threshold')

parser.add_argument('--save_path',
                    help='path to save frames')

parser.add_argument('--camera_id', type=int, default=(4),
                    help='Array of cameras_ids')

parser.add_argument('--cameras_frame_shape', nargs='+', type=int, default=(1280, 720),
                    help='Expected frame shape w h')

parser.add_argument('--board_shape', nargs='+', type=int, default=(7, 5),
                    help='expected board shape')

parser.add_argument('--pattern_dimension', type=int, default=20,
                    help='pattern dimension of each squere in mm')

args = parser.parse_args()

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters =  aruco.DetectorParameters()
arucoDetector = aruco.ArucoDetector(dictionary, parameters)

board = aruco.CharucoBoard(args.board_shape, 38, 30.0, dictionary)

db = []

img_shape = args.cameras_frame_shape


def main():
    # define a video capture object
    cam = cv2.VideoCapture(args.camera_id)

    setupCam(cam, img_shape[0], img_shape[1])

    collected_frames = 0

    allCorners = []
    allIds = []
    decimator = 0

    kalib_folder = args.save_path + '/kalibracja_' + '_' + date.today().strftime("%d_%m_%Y") + '/'

    if(not path.exists(kalib_folder)):
        os.mkdir(kalib_folder)

    cam_save_path   = kalib_folder + '/cam' + str(int(args.camera_id)) + '/'

    if(not path.exists(cam_save_path)):
        os.mkdir(cam_save_path)    

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    corner_ids = cornerIds(args.board_shape)

    # Main loop
    while(True):
        # Capture the video frame
        # by frame
        ret, frame = cam.read()
        
        frame_copy = np.copy(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedCorners = arucoDetector.detectMarkers(gray)

        if len(corners) > 4:
            aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedCorners)
                
            res2 = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0 and max(ids) <= max(board.getIds()):
                if is_slice_in_list(numpy.squeeze(ids).tolist(), corner_ids): # all corners are detected
                    params, board_rot_deg  = get_parameters(corners, numpy.squeeze(ids).tolist(), corner_ids, img_shape, args.board_shape)
                    if(board_rot_deg <= 35.0):
                        if(len(db) == 0):
                            db.append((params))
                            
                            allCorners.append(res2[1])
                            allIds.append(res2[2])

                            collected_frames += 1 

                            cv2.imwrite(cam_save_path + "/" + str(collected_frames) + ".png", frame)

                            continue

                        if is_good_sample(params, db, args.threshold):
                            # Adding collected parameters
                            db.append((params))

                            allCorners.append(res2[1])
                            allIds.append(res2[2])


                            collected_frames += 1 

                            print(compute_goodenough(db, args.difficulty_mulitpl))
                            cv2.imwrite(cam_save_path + "/" + str(collected_frames) + ".png", frame)
            
                        if (collected_frames > 70 and collected_frames % 10 == 0):
                            
                            collected_frames += 1 

                            print("CAMERA CALIBRATION")

                            cameraMatrixInit = np.array([[ 1000.,    0., args.cameras_frame_shape[0] / 2.],
                                                        [    0., 1000.,  args.cameras_frame_shape[1] / 2.],
                                                        [    0.,    0.,           1.]])

                            distCoeffsInit = np.zeros((5,1))

                            flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_RATIONAL_MODEL)
                            #flags = (cv2.CALIB_RATIONAL_MODEL)

                            
                            (ret, camera_matrix, distortion_coefficients,
                            rotation_vectors, translation_vectors,
                            stdDeviationsIntrinsics, stdDeviationsExtrinsics,
                            perViewErrors) = aruco.calibrateCameraCharucoExtended(
                                            charucoCorners=allCorners,
                                            charucoIds=allIds,
                                            board=board,
                                            imageSize=img_shape,
                                            cameraMatrix=cameraMatrixInit,
                                            distCoeffs=distCoeffsInit,
                                            flags=flags,
                                            criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
                            
                            rotation_vectors = np.asarray(rotation_vectors)
                            translation_vectors = np.asarray(translation_vectors)
                            
                            np.savetxt('Intrinsic_mtx_2.txt', camera_matrix)
                            np.savetxt('dist_2.txt', distortion_coefficients)

                            print("Calculating reprojection error ...")
                            print("Error = {err}".format(err=ret))
                            print("camera_matrix {matrix}".format(matrix=camera_matrix))
                            print("distortion {dist}".format(dist=distortion_coefficients))

                frame_copy = aruco.drawDetectedMarkers(frame_copy.copy(), corners, ids)
                decimator+=1

                        
        # Display the resulting frame
        cv2.imshow('View Display', frame_copy)

        k = cv2.waitKey(1) # wait for key press

        if k%256 == ord('s'):
            #Saving frames
            cv2.imwrite(cam_save_path + "/" + str(collected_frames) + ".png", frame)

            collected_frames += 1 
        elif(k%256 == ord('q')):
            break

    # After the loop release the cap object
    cam.release()

    # Destroy all the windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
