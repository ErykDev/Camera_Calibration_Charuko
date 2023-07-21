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
        frames[i] = cv2.undistort(frames[i], K[i], D[i], None, K_opt[i][0])
        gray_frames.append(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY))



    all_corners = []
    all_ids = []

    for i in range(len(frames)):
        all_corners.append(None)
        all_ids.append(None)

        cor,  ids,  rejec = arucoDetector.detectMarkers(gray_frames[i])

        if len(cor) > 4:
            aruco.refineDetectedMarkers(gray_frames[i], board, cor, ids, rejec)
            #frames[i] = aruco.drawDetectedMarkers(frames[i], cor, ids)
            
            if cor is not None and len(ids)>3 and max(ids) <= max(board.getIds()):
                cor = np.array(cor, dtype=np.float32)

                if is_slice_in_list(numpy.squeeze(ids).tolist(), ids): # all left corners are detected
                    charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(cor,  ids, gray_frames[i], board)
                    
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
            positions.append(None)



        for cam_id in range(len(cams)):
            if all_corners[cam_id] is not None:

                valid, rvec, tvec = aruco.estimatePoseCharucoBoard(
                            charucoCorners=all_corners[cam_id], charucoIds=all_ids[cam_id], board=board, \
                                cameraMatrix=K_opt[cam_id][0], distCoeffs=D[cam_id], rvec=None, tvec=None,\
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

                    positions[cam_id] = camera_pos
        
        for camId1 in range(len(cams)):
            for camId2 in range(len(cams)):
                if camId1 == camId2:
                    continue

                if positions[camId1] is not None and \
                    positions[camId2] is not None:

                    distances[camId1][camId2].append(
                        np.array([
                            positions[camId1][0, 0] - positions[camId2][0, 0], 
                            positions[camId1][0, 1] - positions[camId2][0, 1], 
                            positions[camId1][0, 2] - positions[camId2][0, 2]])
                    )

                    print('Collected cam id {} and {}'.format(camId1, camId2) )

    #out = cv2.hconcat(frames)
    #cv2.imshow("camera", gray_frames[0])

    
    if len(distances[0][1]) > 90:
        break

    #k = cv2.waitKey(1)
    #if k%256 == 27:
        # ESC pressed
    #    print("Escape hit, closing...")
    #    break
   

final_cam_pos = []

for cam in cams:
    final_cam_pos.append(None)

final_cam_pos[0] = np.array([0.0,0.0,0.0])

def solve_neighbours(root_id):
    for camId in range(len(cams)):
        if camId == root_id:
            continue

        if len(distances[root_id][camId]) > 90:
            
            if final_cam_pos[camId] is not None:
                continue

            distance = np.average(distances[root_id][camId], axis=0) # np.average(distances[root_id][camId])


            final_cam_pos[camId] = final_cam_pos[root_id] + distance

            solve_neighbours(camId)


solve_neighbours(0)


for camId in range(len(cams)):
    print('position cam{} : {}'.format(camId, final_cam_pos[camId]))


bs = math.hypot(
    final_cam_pos[0][0] - final_cam_pos[1][0], 
    final_cam_pos[0][1] - final_cam_pos[1][1], 
    final_cam_pos[0][2] - final_cam_pos[1][2])
                
print('baseline: {bs} cm'.format(bs=bs * 100))


for cam in cams:
    cam.release()

cv2.destroyAllWindows()
