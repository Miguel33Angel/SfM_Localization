# //
# // Name        : main.py
# // Author      : Miguel Angel Calvera Casado
# // Version     : V0.0
# // Copyright   : Your copyright notice
# // Description : Localization with monocular camera.
# //============================================================================
"""
Resources that will be usefull:
https://arxiv.org/pdf/2003.01587.pdf -> Comparisons of keypoint detectors and matchers, as well as RANSAC performance
https://arxiv.org/abs/1904.00889.pdf -> Key.Net Keypoint detector (https://github.com/axelBarroso/Key.Net-Pytorch) (try it after SuperPoint)
RANSAC algorithm: MAGSAC++ (implementation in OpenCV)
MAGSAC++, a fast, reliable and accurate robust estimator

"""

from Drawer3D import *
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
import cv2

import torch

from models.matching import Matching
from models.utils import (VideoStreamer, frame2tensor)

torch.set_grad_enabled(False)  # For getting confidance with .numpy()


def triangulateFrom2View(x1, x2, K_c1, K_c2, T_c2_c1):
    """
    Triangulate the matches matched points between two views, the relative
    movement between the cameras and the intrinsic parameters are known.

    -input:
        x1: 2xn matrix -> n 2D points in the image 1. Each i index is matched with the same indes in x2.
        x2: 2xn matrix -> n 2D points in the image 2. Each i index is matched with the same indes in x1.
        K_c1: 3x3 matrix -> Camera 1 calibration matrix.
        K_c2: 3x3 matrix -> Camera 2 calibration matrix.
        T_c2_c1 : 4x4 matrix -> Relative movenment between the camera 2 and camera 1.
    -output:
        X_3D: nx4 matrix -> n 3D points in the reference system of the camera 1.
    """

    P_c1 = np.hstack((K_c1, np.zeros((3, 1))))

    P_c2 = K_c2 @ np.eye(3, 4) @ T_c2_c1

    num_matches = x1.shape[1]

    v_p11 = P_c1[0, :]
    v_p12 = P_c1[1, :]
    v_p13 = P_c1[2, :]
    v_p21 = P_c2[0, :]
    v_p22 = P_c2[1, :]
    v_p23 = P_c2[2, :]

    X_3D = np.zeros((num_matches, 4))
    for i in range(num_matches):
        A = np.zeros((4, 4))

        u_1 = x1[0, i]
        v_1 = x1[1, i]
        A[0, :] = u_1 * v_p13 - v_p11
        A[1, :] = v_1 * v_p13 - v_p12

        u_2 = x2[0, i]
        v_2 = x2[1, i]
        A[2, :] = u_2 * v_p23 - v_p21
        A[3, :] = v_2 * v_p23 - v_p22

        _, _, Vt = np.linalg.svd(A)
        X_3D[i, :] = Vt[-1, :]
        X_3D[i, :] = X_3D[i, :] / X_3D[i, 3]

    return X_3D


def sG_format_to_OpenCv(raw_matches, raw_keypoint_1, raw_keypoint_2, matches_confidence):
    kpts_1 = cv2.KeyPoint_convert(raw_keypoint_1)
    kpts_2 = cv2.KeyPoint_convert(raw_keypoint_2)

    SuperGlue_matches = [0, 0, 0]
    for i in range(0, raw_matches.shape[0]):
        if (raw_matches[i] > -1 and raw_matches[i] < len(kpts_2)):
            SuperGlue_matches = np.vstack((SuperGlue_matches, [i, raw_matches[i], 1 / matches_confidence[i]]))

    # Delete first element
    SuperGlue_matches = SuperGlue_matches[1:]

    # Convert to dMatchesList
    dMatchesList = []
    for row in SuperGlue_matches:
        dMatchesList.append(cv2.DMatch(_queryIdx=int(row[0]), _trainIdx=int(row[1]), _distance=row[2]))

    return dMatchesList, kpts_1, kpts_2

# Inneficient functions used for debugging or testing against opencv
def getFundamentalFromMatches(x_1, x_2):
    m = x_1.shape[1]
    ones = np.array(np.ones((m))).T

    x0 = x_1[0]
    y0 = x_1[1]
    x1 = x_2[0]
    y1 = x_2[1]

    A = np.vstack((x0 * x1, y0 * x1, x1, x0 * y1, y0 * y1, y1, x0, y0, ones)).T
    # Now we need to solve a svd problem: A*f = 0
    u, s, vh = np.linalg.svd(A)
    f = vh[-1]
    F_21 = np.array(f).reshape((3, 3))
    # Force rank 2 for the F matrix
    u2, s2, vh2 = np.linalg.svd(F_21)
    s2[2] = 0
    F_21 = u2 @ np.diag(s2) @ vh2

    return F_21


def getScoreFundamentalRANSAC(x1, x2, F, threshold):
    l2_fromF = F @ x1
    denom = np.sqrt(l2_fromF[0] * l2_fromF[0] + l2_fromF[1] * l2_fromF[1])
    nom = np.abs(x2[0] * l2_fromF[0] + x2[1] * l2_fromF[1] + x2[2] * l2_fromF[2])
    vector_err = np.abs(nom / denom)

    inliers = (vector_err < threshold)
    n_votes = inliers.sum()
    # Get InliersMask
    inliersMask = np.array([0 for k in range(x1.shape[1])])
    inliersMask[inliers] = 1

    return n_votes, inliersMask


def getFundamentalFromMatchesRANSAC(x1, x2, ransacThreshold=2.0, maxIters=2000, pMinSet=8):
    # ransacThreshold used to know its an inlier
    # maxIters is number maximum of iterations before giving up on searching.
    # pMinSet is used in case there aren't enough points to return a NULL.
    print('Max iterations = ' + str(maxIters))
    bestScore = 0
    bestInliersMask = 0
    bestIndexToCalculate = 0
    bestF = 0
    rng = np.random.default_rng()  # Inside we can put a seed
    row, colum = x1.shape

    for i in range(maxIters):
        indexToCalculate = rng.integers(low=0, high=colum, size=pMinSet)
        x1_subgroup = x1[:, indexToCalculate]
        x2_subgroup = x2[:, indexToCalculate]
        F = getFundamentalFromMatches(x1_subgroup, x2_subgroup)
        score, inliersMask = getScoreFundamentalRANSAC(x1, x2, F, ransacThreshold)
        if score > bestScore:
            bestF = F
            bestScore = score
            bestIndexToCalculate = indexToCalculate
            bestInliersMask = inliersMask

    print("Score: " + str(bestScore))

    # If there is not a minimum of points return null
    if bestScore < pMinSet:
        print("Not enough points")
        return None

    # Convert index into a mask to draw it later.
    bestPtsToCalcMask = np.array([0 for k in range(x1.shape[1])])
    bestPtsToCalcMask[bestIndexToCalculate] = 1
    return bestF, bestInliersMask, bestPtsToCalcMask


def drawEpipolarLineFromPoint(x_1, F):
    x_1_h = np.array([x_1]).T
    l_2 = F @ x_1_h
    a = l_2[0][0]
    b = l_2[1][0]
    c = l_2[2][0]
    point_epipolar_cuts_y = (0, -c / b)
    if point_epipolar_cuts_y[1] > 900:
        point_epipolar_cuts_y = (-(c + 900 * b) / a, 900)

    point_epipolar_cuts_x = (-c / a, 0)
    if point_epipolar_cuts_x[0] > 600:
        point_epipolar_cuts_x = (600, -(c + 600 * a) / b)

    plt.axline(point_epipolar_cuts_y, point_epipolar_cuts_x, linestyle='-')
    return point_epipolar_cuts_y, point_epipolar_cuts_x


def main():
    np.set_printoptions(precision=8, linewidth=1024, suppress=True)
    # Camera Matrix and distortion coefs.

    # For now given
    # TODO: create a guess and refine it with more images.
    K = np.array([[641.84180665, 0., 311.13895719],
                  [0., 641.17105466, 244.65756186],
                  [0., 0., 1.]])
    dist = np.array([[-0.02331774, 0.25230237, 0., 0., -0.52186379]])

    # Hyperparameter of Ransac and fundamental estimation
    MIN_MATCH_COUNT = 8  # We use a p8p algorithm, as it's the simplest

    # Hyperparameters of SP and SG
    # Hyperparameters of SuperGlue and SuperPoint. They are at the default.
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1
        },
        'superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    device = 'cpu'  # Compatibility with pretty much anything.
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']

    # Obtain the images ref_frame and next_frame from the video
    # TODO: use more than 2
    video_name = "secuencia_a_cam2.avi"
    print("Trabajo con el video " + video_name)

    vs = VideoStreamer(video_name, [640, 480], 1, ['*.png', '*.jpg', '*.jpeg'], 1000000)
    ref_frame, ret = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'

    # Process reference frame
    frame_tensor = frame2tensor(ref_frame, device)
    last_data = matching.superpoint({'image': frame_tensor})
    last_data = {k + '0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor
    last_frame = ref_frame
    last_image_id = 0

    incr_n = 300
    n = 1
    while True:
        next_frame, ret = vs.next_frame()
        # For now use only the 1st and
        if (n % incr_n == 0) and ret:
            break
            # next_frame = frame
        if not ret:
            print("Finished reading")
            break
        n = n + 1
    print("Finished reading")

    # Extract the keypoints and obtain the good Matches

    # New Frame to a tensor:
    frame_tensor = frame2tensor(next_frame, device)
    pred = matching({**last_data, 'image1': frame_tensor})
    kpts1 = last_data['keypoints0'][0].cpu().numpy()
    kpts2 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()

    # So now we have the kpts1 and kpts2 and the good matches as well in another format.
    # Change the format to the one OpenCV uses:
    good, kp1, kp2 = sG_format_to_OpenCv(matches, kpts1, kpts2, confidence)

    # SuperGlue Gives -> good, kp1, kp2
    img3 = cv2.drawMatches(ref_frame, kp1, next_frame, kp2, good, None)
    cv2.imshow("Matches", img3)

    if not len(good) > MIN_MATCH_COUNT:
        exit()

    # From matching list to xy
    pts1 = [kp1[m.queryIdx].pt for m in good]
    pts2 = [kp2[m.trainIdx].pt for m in good]
    open_pts1 = np.asarray(pts1)
    open_pts2 = np.asarray(pts2)

    # First camera is set as the origin
    T_AC1 = np.eye(4)


    # From the points estimate the fundamentalMatrix F_21_est.
    # TODO: use MAGSAC++
    F_21_est, mask = cv2.findFundamentalMat(open_pts1,open_pts2,cv2.FM_RANSAC, 4)

    # There's a custom RANSAC function coded as part of a course. It's slower but it works nonetheless.
    # x1 = np.vstack((open_pts1.T, np.ones((1, open_pts1.shape[0]))))
    # x2 = np.vstack((open_pts2.T, np.ones((1, open_pts2.shape[0]))))
    # F_21_est, _, _ = getFundamentalFromMatchesRANSAC(x1, x2, ransacThreshold=5, maxIters=20000, pMinSet=8)

    # Obtain the Essential Matrix and estimate the Rotation and translation using the function
    E_21 = K.T @ F_21_est @ K

    # From the possible solutions obtain the real solution and the triangulation of the 3dPoints X3D
    _, R, t, _ = cv2.recoverPose(E_21, open_pts1, open_pts2)
    T_21_est = np.eye(4, 4)
    T_21_est[0:3, 0:3] = R
    T_21_est[0:3, 3] = t.reshape(3)
    # Get 3d of points.
    # TODO: Save 3d points for map. Use descriptors, or frames directly?
    X3D = triangulateFrom2View(open_pts1.T, open_pts2.T, K, K, T_21_est)

    # Plot the 3d Points and the two camera poses (The origin and the second camera pose estimated)
    T_12_est = np.linalg.inv(T_21_est)
    X3D_w_est = X3D
    T_w2_est = T_12_est
    mark_est = dict(color='red', size=5)
    fig_triangulation = go.Figure()
    drawRefSystem(fig_triangulation, np.eye(4), "W")
    drawCamera(fig_triangulation, T_AC1)
    drawCamera(fig_triangulation, T_w2_est)
    drawPoints(fig_triangulation, X3D_w_est, mark_est)
    pio.show(fig_triangulation)
    cv2.waitKey()


if __name__ == '__main__':
    main()
