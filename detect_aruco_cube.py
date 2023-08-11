import cv2
import numpy as np

import cv2.aruco as aruco

from calibration_utils import *


def average_rotation_vector(rotation_vectors):
    # Convert rotation vectors to unit vectors
    unit_vectors = [v / np.linalg.norm(v) for v in rotation_vectors]
    
    # Average the unit vectors
    avg_unit_vector = np.mean(unit_vectors, axis=0)
    avg_unit_vector /= np.linalg.norm(avg_unit_vector)
    
    # Convert the average unit vector back to a rotation vector
    average_rotation = avg_unit_vector * 2 * np.arccos(avg_unit_vector[0])
    
    return average_rotation


def aruco_display(corners, ids, image):
	if len(corners) > 0:
		
		ids = ids.flatten()
		
		for (markerCorner, markerID) in zip(corners, ids):
			
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners
			
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
			
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
			
			#cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
			#	0.5, (0, 255, 0), 2)
			print("[Inference] ArUco marker ID: {}".format(markerID))

	return image



dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters =  aruco.DetectorParameters()
arucoDetector = aruco.ArucoDetector(dictionary, parameters)


w, h = 1280, 720

cam = cv2.VideoCapture(2)

K1 = np.loadtxt('Intrinsic_mtx_1.txt', dtype=float).reshape((3, 3))
D1 = np.loadtxt('dist_1.txt', dtype=float)

#roi can be ignored sine we using alpha=0
K1_opt, roi_1 = cv2.getOptimalNewCameraMatrix(K1, D1, (w,h), 0, (w,h))

markerSideLength = 45 #mm

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

setupCam(cam, w, h)


# 90 degrees rotation needed to match the frames
DEFAULT_ROT = np.array([[0,0,-1],[0,1,0],[1,0,0]])

# specific rotations for each face of the cube
zero = np.matmul(np.array([[1,0,0],[0,1,0],[0,0,1]]),       DEFAULT_ROT)
uno = np.matmul(np.array([[1,0,0],[0,-1,0],[0,0,-1]]),      DEFAULT_ROT)
due = np.matmul(np.array([[1,0,0],[0,0,1],[0,-1,0]]),       DEFAULT_ROT)
tre = np.matmul(np.array([[1,0,0],[0,0,-1],[0,1,0]]),       DEFAULT_ROT)

quattro = np.matmul(np.array([[0,0,-1],[0,1,0],[1,0,0]]),   DEFAULT_ROT)
cinque = np.matmul(np.array([[0,0,1],[0,1,0],[-1,0,0]]),    DEFAULT_ROT)

# Dictionary used to map the ArUco ids to the corresponding rotation
EUL_TRANS_DICT= {
	94 : zero,
	97 : uno,
	95 : due,
    99 : tre,

	98 : quattro,
	96 : cinque
}

# Dictionary used to find the centroid of the cube given a face
CENTER_POINT_OFFSET_DICT={
    94 : np.float32([[-0.03,0,0]]), #top
    97 : np.float32([[0.03,0,0]]),
    95 : np.float32([[0,0.03,0]]),
    99 : np.float32([[0,-0.03,0]]),


    98 : np.float32([[0,0,0.03]]),
    96 : np.float32([[0,0,-0.03]])
}


while True:
    cam.grab()
    ret1, frame = cam.retrieve()

    if not ret1:
        print("failed to grab frame")
        continue
    else:
        frame = cv2.undistort(frame, K1, D1, None, K1_opt)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedCorners = arucoDetector.detectMarkers(gray)

        if len(corners) > 0:

            #Refine Corners
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize = (3,3),
                                 zeroZone = (-1,-1),
                                 criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001))


             # Avoid crashing if the ArUco id is wrong
            if(max(ids) > 99 or min(ids)< 94):
                continue

            frame =  aruco_display(corners, ids, frame)


            centroids = []
            rotation_vectors = []

            for Index in ids:

                id = Index[0]
                cornerIndex = np.argwhere(ids == id)[0][0]
                
                # Estimate the cube pose given the ArUco code
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[cornerIndex], 0.04, K1_opt, D1)

                # Draw the ArUco id on the output image
                frame = cv2.putText(frame, 'id: '+str(id), (int(corners[cornerIndex][0][0][0]), int(corners[cornerIndex][0][0][1])), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0), 2, cv2.LINE_AA)
                
                # Transformations needed to have coherent frames
                rmat = cv2.Rodrigues(rvec)[0]
                computed_rtm = np.matmul(rmat,EUL_TRANS_DICT[id])
                computed_rvec = cv2.Rodrigues(computed_rtm)[0]

                # Draw the frames to the output image
                #cv2.drawFrameAxes(frame, K1_opt, D1, computed_rvec, tvec, 0.01)

                # Preliminary operations to find the centroid of the cube
                centroid_offset = CENTER_POINT_OFFSET_DICT[id]
                homogenous_trans_mtx  = np.append(computed_rtm, [ [tvec[0][0][0]], [tvec[0][0][1]], [tvec[0][0][2]] ], axis=1)
                homogenous_trans_mtx = np.append(homogenous_trans_mtx,[[0,0,0,1]], axis=0)

                # Find x,y position to draw the centroid
                imgpts, jac = cv2.projectPoints(centroid_offset, computed_rvec, tvec, K1_opt, D1)
                imgpts = np.int32(imgpts).reshape(-1,2)
                
                # Find the 3d coordinates of the centroid
                x = CENTER_POINT_OFFSET_DICT[id][0][0]
                y = CENTER_POINT_OFFSET_DICT[id][0][1]
                z = CENTER_POINT_OFFSET_DICT[id][0][2]
                centroid_coords = [ [x], [y], [z], [1] ]
                centroid_coords = np.matmul(homogenous_trans_mtx, centroid_coords)

                # Draw the centroid on the output image
                #frame = cv2.circle(frame, (imgpts[0][0], imgpts[0][1]), radius=3, color=(255,0,255), thickness=4)
                centroids.append(centroid_coords)
                rotation_vectors.append(computed_rvec)

            avg_centroid = np.average(centroids, axis=0)
            avg_rotation_vec = np.average(rotation_vectors, axis=0)

            cv2.drawFrameAxes(frame, K1_opt, D1, avg_rotation_vec, avg_centroid[:-1], 0.02)

            
            #return frame, imgpts[0], tvec, computed_rvec
            

        frame_down = \
            cv2.resize(frame, \
            (int(frame.shape[1] / 1.2), int(frame.shape[0] / 1.5)), \
            interpolation=cv2.INTER_LINEAR \
            )

        cv2.imshow("camera", frame_down)


    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif(k%256 == ord('s')):
        print('save clicked')

        cv2.imwrite('left.png', frame)


cam.release()

cv2.destroyAllWindows()
