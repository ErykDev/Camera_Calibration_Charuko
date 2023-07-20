import cv2
import numpy as np

import cv2.aruco as aruco

from calibration_utils import *

import matplotlib
matplotlib.use('GTK3Agg')

import matplotlib.pyplot as plt



board_shape = [7, 5]

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters =  aruco.DetectorParameters()
arucoDetector = aruco.ArucoDetector(dictionary, parameters)

board = aruco.CharucoBoard(board_shape, 38, 30.0, dictionary)

w, h = 1280, 720

cams = [cv2.VideoCapture(2), cv2.VideoCapture(4)]

K = [np.loadtxt('Intrinsic_mtx_1.txt', dtype=float).reshape((3, 3)), 
     np.loadtxt('Intrinsic_mtx_2.txt', dtype=float).reshape((3, 3))]

D = [np.loadtxt('dist_1.txt', dtype=float),
     np.loadtxt('dist_2.txt', dtype=float)]

#roi can be ignored sine we using alpha=0
K_opt = [cv2.getOptimalNewCameraMatrix(K[0], D[0], (w,h), 0, (w,h)),
         cv2.getOptimalNewCameraMatrix(K[1], D[1], (w,h), 0, (w,h))]

pixel_size = 0.003


sensor_size_x = w * pixel_size
sensor_size_y = h * pixel_size

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

corner_ids = cornerIds(board_shape)

for cam in cams:
    setupCam(cam, w, h)


distances = []

for i in range(len(cams)):
    distances.append([])
    for j in range(len(cams)):
        distances[i].append([])

while True:
    for cam in cams: 
        cam.grab()

    frames = []

    for cam in cams: 
        ret, frame = cam.retrieve()

        assert ret
        frames.append(frame)


    gray_frames = []

    for i in range(len(frames)):
        frames[i]       = cv2.undistort(frames[i], K[i], D[i], None, K_opt[i][1])
        gray_frames[i]  = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)



    all_corners = []
    all_ids = []

    for i in range(len(frames)):
        all_corners[i]  = None
        all_ids[i]      = None

        cor,  id,  rejec = arucoDetector.detectMarkers(gray_frames[i])

        if len(cor) > 4:
            aruco.refineDetectedMarkers(gray_frames[i], board, cor, id, rejec)
            
            if corners_left is not None and len(id)>3 and max(id) <= max(board.getIds()):
                corners_left = np.array(corners_left, dtype=np.float32)

                if is_slice_in_list(numpy.squeeze(id).tolist(), corner_ids): # all left corners are detected
                    charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(cor,  id, gray_frames[i], board)
                    
                    if charucoretval:
                        all_corners[i]  = charucoCorners
                        all_ids[i]      = charucoIds

    # we have to make sure that at lest two camera captured markers 
    notNones = 0
    for corners in all_corners:
        if corners is not None:
            notNones += 1

    if notNones > 1:

        positions = []

        for cam_id in range(len(cams)):
            if all_corners[cam_id] is not None:

                valid, rvec, tvec = aruco.estimatePoseCharucoBoard(
                            charucoCorners=all_corners[cam_id], charucoIds=all_ids[cam_id], board=board, \
                                cameraMatrix=K_opt[cam_id][1], distCoeffs=D[cam_id], rvec=None, tvec=None,\
                                    useExtrinsicGuess=False)
                
                if valid:
                    # Invert the translation vector
                    #tvec = -tvec

                    # Create a transformation matrix
                    R, _ = cv2.Rodrigues(rvec)
                    T = np.concatenate((R, tvec), axis=1)
                    T = np.vstack((T, [0, 0, 0, 1]))

                    # Create a point at the origin board
                    origin = np.array([[0, 0, 0, 1]])

                    # Transform the origin using the camera pose
                    camera_pos = np.dot(T, origin.T).T

                    # mm to meters 
                    camera_pos = camera_pos * 0.001 

                    positions[cam_id].append(camera_pos)
        
        for camId1 in range(len(cams)):
            for camId2 in range(len(cams)):
                if camId1 == camId2:
                    continue

                if positions[camId1] is not None and \
                    positions[camId2] is not None:

                    distances[camId1][camId2].append(positions[camId1] - positions[camId2])

                    print('Collected cam id {} and {}'.format(camId1, camId2) )

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
   

final_cam_pos = []


final_cam_pos[0] = np.array([0.0,0.0,0.0])

def solve_neighbours(root_id):
    for camId in range(len(cams)):
        if camId == root_id:
            continue

        if len(distances[root_id][camId]) > 30:

            distance = np.average(distances[root_id][camId])

            final_cam_pos[camId] = final_cam_pos[root_id] + distance

            solve_neighbours(camId)







    







        






            

                




        
        


                
                







    


    






    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

for cam in cams:
    cam.release()

cv2.destroyAllWindows()
