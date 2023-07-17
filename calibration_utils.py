import math
import cv2
import numpy
import time

def flip_corners(corners):
    return cv2.flip(corners, flipCode=0)

def correct_board_orientation(corners):
    """ 
    Check for correct board orientation  
    """
    return corners[0, 0, 1] > corners[-1, 0, 1]

def lmin(seq1, seq2):
    """ Pairwise minimum of two sequences """
    return [min(a, b) for (a, b) in zip(seq1, seq2)]

def lmax(seq1, seq2):
    """ Pairwise maximum of two sequences """
    return [max(a, b) for (a, b) in zip(seq1, seq2)]

def _pdist(p1, p2):
    """
    Distance bwt two points. p1 = (x, y), p2 = (x, y)
    """
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))



def _calculate_skew(corners):
    """
    Get skew for given checkerboard detection.
    Scaled to [0,1], which 0 = no skew, 1 = high skew
    Skew is proportional to the divergence of three outside corners from 90 degrees.
    """
    # TODO Using three nearby interior corners might be more robust, outside corners occasionally
    # get mis-detected
    up_left, up_right, _, down_right, = corners

    def angle(a, b, c):
        """
        Return angle between lines ab, bc
        """
        ab = a - b
        cb = c - b
        return math.acos(numpy.dot(ab,cb) / (numpy.linalg.norm(ab) * numpy.linalg.norm(cb)))

    skew = min(1.0, 2. * abs((math.pi / 2.) - angle(up_left, up_right, down_right)))
    return skew

def _calculate_area(corners):
    """
    Get 2d image area of the detected checkerboard.
    The projected checkerboard is assumed to be a convex quadrilateral, and the area computed as
    |p X q|/2; see http://mathworld.wolfram.com/Quadrilateral.html.
    """
    (up_left, up_right, down_left, down_right) = corners
    a = up_right - up_left
    b = down_right - up_right
    c = down_left - down_right
    p = b + c
    q = a + b
    return abs(p[0]*q[1] - p[1]*q[0]) / 2.

def board_rotation_degr(down_right, down_left):
    return math.degrees(math.atan2(abs(down_left[1] - down_right[1]), abs(down_left[0] - down_right[0])))


def get_parameters(corners, ids, corner_ids, size, board_shape):
    """
    Return list of parameters [X, Y, size, skew] describing the checkerboard view.
    """
    (width, height) = size

    #outside_corners = _get_outside_corners(corners, board_shape)
    # (up_left, up_right, down_left, down_right)
    # 0 4 33 37
    outside_corners = corners[ids.index(corner_ids[0])][0][0], corners[ids.index(corner_ids[1])][0][0], corners[ids.index(corner_ids[2])][0][0], corners[ids.index(corner_ids[3])][0][0]

    Xs = outside_corners[0][0], outside_corners[1][0], outside_corners[2][0], outside_corners[3][0]
    Ys = outside_corners[0][1], outside_corners[1][1], outside_corners[2][1], outside_corners[3][1]   

    area = _calculate_area(outside_corners)
    skew = _calculate_skew(outside_corners)
    border = math.sqrt(area)
    # For X and Y, we "shrink" the image all around by approx. half the board size.
    # Otherwise large boards are penalized because you can't get much X/Y variation.
    p_x = min(1.0, max(0.0, (numpy.mean(Xs) - border / 2) / (width  - border)))
    p_y = min(1.0, max(0.0, (numpy.mean(Ys) - border / 2) / (height - border)))
    p_size = math.sqrt(area / (width * height))
    params = [p_x, p_y, p_size, skew]

    return params, board_rotation_degr(outside_corners[2], outside_corners[3])


def is_good_sample(params, db, threshold):
    """
    Returns true if the checkerboard detection described by params should be added to the database.
    """
    def param_distance(p1, p2):
        return sum([abs(a-b) for (a,b) in zip(p1, p2)])

    db_params = [sample for sample in db]
    d = min([param_distance(params, p) for p in db_params])
    #print "d = %.3f" % d #DEBUG
    # TODO What's a good threshold here? Should it be configurable?
    if d <= threshold:
        return False

    #if self.max_chessboard_speed > 0:
    #    if not self.is_slow_moving(corners, ids, last_frame_corners, last_frame_ids):
    #        return False

    # All tests passed, image should be good for calibration
    return True

def compute_goodenough(db, difficulty_mulitpl):
    # Find range of checkerboard poses covered by samples in database
    all_params = [sample for sample in db]

    min_params = all_params[0]
    max_params = all_params[0]
    for params in all_params:
        #print(params)
        min_params = lmin(min_params, params)
        max_params = lmax(max_params, params)
    # Don't reward small size or skew
    min_params = [min_params[0], min_params[1], 0., 0.]

    # For each parameter, judge how much progress has been made toward adequate variation
    progress = [min((hi - lo) / r, 1.0) for (lo, hi, r) in zip(min_params, max_params, [0.8, 0.8, 0.4, 0.5] * (difficulty_mulitpl))]
    # If we have lots of samples, allow calibration even if not all parameters are green
    # TODO Awkward that we update self.goodenough instead of returning it
    #goodenough = (len(db) >= 60) or all([p == 1.0 for p in progress])

    return list(zip(["X", "Y", "Size", "Skew"], progress))



def is_slice_in_list(A, B):
    for s in B:
        if not (s in A):
            return False
    return True

def cornerIds(board_shape):
    w = board_shape[0]
    h = board_shape[1]
    return [0, math.floor(w / 2) - 1 , (math.floor(w*h / 2) - 1) - ( math.floor(w / 2) - 1), math.floor(w*h / 2) - 1]


def calculateImgPointsObjPoints(shared_ids, shared_corners, board):
    imgPoints, objPoints = [], []

    for ids, corner in zip(shared_ids, shared_corners):
        op, ip = board.matchImagePoints(
            numpy.array(corner, numpy.float32),
            numpy.array(ids, numpy.int32))
        imgPoints.append(ip)
        objPoints.append(op)

    return imgPoints, objPoints


def getSharedFetures(CornersLeft, IdsLeft, CornersRight, IdsRight, board):
    shared_corners_l = []
    shared_corners_r = []

    shared_ids_l = []
    shared_ids_r = []

    for corners_l, ids_l, corners_r, ids_r in zip(CornersLeft, IdsLeft, CornersRight, IdsRight):
        temp_corners_l = []
        temp_corners_r = []

        temp_ids_l = []
        temp_ids_r = []

        for i in range(max(board.getIds()) + 1):
            if i in ids_l and i in ids_r: # shared id
                indx_l, _ = numpy.where(ids_l == i) 
                indx_r, _ = numpy.where(ids_r == i) 
                
                temp_ids_l.append(ids_l[indx_l])
                temp_ids_r.append(ids_r[indx_r])

                temp_corners_l.append(corners_l[indx_l])
                temp_corners_r.append(corners_r[indx_r])
            
        shared_corners_l.append(temp_corners_l)
        shared_corners_r.append(temp_corners_r)

        shared_ids_l.append(temp_ids_l)
        shared_ids_r.append(temp_ids_r)

    return shared_corners_l, shared_ids_l, shared_corners_r, shared_ids_r


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

