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
from Drawer3D_simple import drawRefSystem, draw3DLine
import numpy as np
import cv2
import time
import torch

from models.matching import Matching
from models.utils import (VideoStreamer, frame2tensor)
torch.set_grad_enabled(False)  # For getting confidance with .numpy()

# Plot 3d:
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# Multiprocessing:
# Naive way is using the built in multiprocessing module (not multithreading because of the GIL)
from multiprocessing import Pool, Queue

import json

def triangulateFrom2View(x1, x2, K_c1, K_c2, T_c2_c1):
    """
    Triangulate the matches matched points between two views, the relative
    movement between the cameras and the intrinsic parameters are known.

    -input:
        x1: 2xn matrix -> n 2D points in the image 1. Each i index is matched with the same indes in x2.
        x2: 2xn matrix -> n 2D points in the image 2. Each i index is matched with the same indes in x1.
        K_c1: 3x3 matrix -> Camera 1 calibration matrix.
        K_c2: 3x3 matrix -> Camera 2 calibration matrix.
        T_c2_c1 : 4x4 matrix -> Relative movement between the camera 2 and camera 1.
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

def main():
    np.set_printoptions(precision=8, linewidth=1024, suppress=True)
    image_width, image_height = 640, 480

    # Get config from JSON file:
    with open('config_vo.txt', 'r') as convert_file:
        config_vo = json.loads(convert_file.read())

    K = np.array(config_vo["K"])
    # dist = np.array(config_vo["dist"])
    config_sp_sg = config_vo["config_sp_sg"]

    # For showing all the camera positions
    T_c2_w_dict = dict()
    T_c1_w = np.array([[0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [1, 0, 0, 0],
                       [0, 0, 0, 1]])

    # Compatibility with pretty much anything.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    keys = ['keypoints', 'scores', 'descriptors']
    instance_sp_sg = Matching(config_sp_sg).eval().to(device)

    # Obtain the images from the video
    video_name = "secuencia_a_cam2.avi"
    print("Trabajo con el video " + video_name)

    # Create capture object with the file:
    cap = cv2.VideoCapture(video_name)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 4) #Set buffer at 4 as it's more than enough

    # Get frame_ref
    n = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ret, frame_ref_raw = cap.read()
    frame_ref_color = cv2.resize(frame_ref_raw, (image_width, image_height), cv2.INTER_AREA)
    frame_ref = cv2.cvtColor(frame_ref_color, cv2.COLOR_RGB2GRAY)

    assert ret, 'Error when reading the first frame (try different --input?)'

    # Process reference frame
    frame_tensor_ref = frame2tensor(frame_ref, device)
    dict_img_ref = instance_sp_sg.superpoint({'image': frame_tensor_ref})
    dict_img_ref = {k + '0': dict_img_ref[k] for k in keys}
    dict_img_ref['image0'] = frame_tensor_ref


    # TODO: make loop which gets frames until transformation is one that makes sense.
    last_n = 350
    init_n = 200  # Skip to this frame
    n = init_n
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    while n <= last_n:
        start_frame = time.time()
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ret, frame_next_raw = cap.read()
        if not ret:
            print("Finished reading")
            break
        print("n",n)


        frame_next_color = cv2.resize(frame_next_raw, (image_width, image_height), cv2.INTER_AREA)
        frame_next = cv2.cvtColor(frame_next_color, cv2.COLOR_RGB2GRAY)

        # New Frame to a tensor:
        frame_tensor_next = frame2tensor(frame_next, device)
        # Get prediction
        pred = instance_sp_sg({**dict_img_ref, 'image1': frame_tensor_next})
        kpts1 = dict_img_ref['keypoints0'][0].cpu().numpy()
        kpts2 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()

        # So now we have the kpts1 and kpts2 and the good matches as well in another format.
        # Change the format to the one OpenCV uses:
        good, kp1, kp2 = sG_format_to_OpenCv(matches, kpts1, kpts2, confidence)

        # SuperGlue Gives -> good, kp1, kp2
        img3 = cv2.drawMatches(frame_ref_color, kp1, frame_next_color, kp2, good, None)
        cv2.imshow("Matches", img3)
        cv2.waitKey(1)

        if len(good) < 8:
            return None

        # From matching list to xy
        pts1 = [kp1[m.queryIdx].pt for m in good]
        pts2 = [kp2[m.trainIdx].pt for m in good]
        open_pts1 = np.asarray(pts1)
        open_pts2 = np.asarray(pts2)

        # From the points estimate the fundamentalMatrix F_21_est.
        # TODO: test multiple methods for RANSAC as USAC_MAGSAC or USAC_FAST, to minimize error and time
        ransac_threshold = 1
        F_21_est, mask = cv2.findFundamentalMat(open_pts1, open_pts2, cv2.FM_RANSAC, ransac_threshold)

        print("Ransac inliers/SG matches", np.sum(mask) / len(good))
        print("SG matches/SP keypoints:", len(good) / len(kpts1))
        # Obtain the Essential Matrix and estimate the Rotation and translation using the function
        E_21 = K.T @ F_21_est @ K

        # From the possible solutions obtain the real solution and the triangulation of the 3dPoints X3D
        _, R, t, _ = cv2.recoverPose(E_21, open_pts1, open_pts2)
        T_21_est = np.eye(4, 4)
        T_21_est[0:3, 0:3] = R
        T_21_est[0:3, 3] = t.reshape(3)

        # Get 3d of points.
        # TODO: Extend to SLAM as another project
        X3D = triangulateFrom2View(open_pts1.T, open_pts2.T, K, K, T_21_est)

        # TODO: Bundle adjustment from previous views and actual view.
        # Save T_c2_w.
        T_c2_w_dict[n] = np.linalg.inv(T_21_est @ T_c1_w)
        print("FPS: ",(1/(time.time()-start_frame)))
        n = n + 1

    # Plot the 3d Points and the two camera poses (The origin and the second camera pose estimated)
    T_c1_w = np.eye(4, 4)
    X_w = X3D.T

    fig3D = plt.figure(1)
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, T_c1_w, '-', 'C1')
    for i in range(init_n,last_n,1):
        drawRefSystem(ax,T_c2_w_dict[i] , '-', '_'+str(i))

    ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='.')

    # Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 1)
    yFakeBoundingBox = np.linspace(0, 4, 1)
    zFakeBoundingBox = np.linspace(0, 4, 1)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()


if __name__ == '__main__':
    main()
