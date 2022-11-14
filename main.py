# //
# // Name        : main.py
# // Author      : Miguel Angel Calvera Casado
# // Version     : V0.0
# // Copyright   : Your copyright notice
# // Description : Localization with monocular camera.
"""
Resources that will be usefull:
https://arxiv.org/pdf/2003.01587.pdf -> Comparisons ofa keypoint detectors and matchers, as well as RANSAC performance
https://arxiv.org/abs/1904.00889.pdf -> Key.Net Keypoint detector (https://github.com/axelBarroso/Key.Net-Pytorch) (try it after SuperPoint)
RANSAC algorithm: MAGSAC++ (implementation in OpenCV)
MAGSAC: marginalizing sample consensus
MAGSAC++, a fast, reliable and accurate robust estimator


"""
# //============================================================================



from scipy.linalg import expm
from GeometricEstimator import *
from Drawer3D import *
import plotly.graph_objs as go
import plotly.io as pio
# In geometric estimator file:

def main():
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)
    # Camera Matrix and distortion coefs.

    # For now given
    # TODO: create a guess and refine it with more images.
    K = np.array([[641.84180665, 0., 311.13895719],
                  [0., 641.17105466, 244.65756186],
                  [0., 0., 1.]])
    dist = np.array([[-0.02331774, 0.25230237, 0., 0., -0.52186379]])

    #Obtain the images ref_frame and frame1 from the video
    # TODO: use more than 2
    video_name = "secuencia_a_cam2.avi"
    n = 0
    incr_n = 5
    print("Trabajo con el video " + video_name)

    vid = cv2.VideoCapture(video_name)  # "secuencia_a_cam2.avi"
    ret, ref_frame = vid.read()

    n = 1
    while vid.isOpened() and n < 300:
        ret, frame = vid.read()

        if n % incr_n == 0:

            frame1 = frame
            if frame1 is None:
                print("La segunda imagen no es valida")
                exit()
        n = n + 1

    # Undistort the images to reduce the reproyection error in the Ransac
    ref_frame = cv2.undistort(ref_frame, K, dist)
    frame1 = cv2.undistort(frame1, K, dist)


    # Extract the keypoints and obtain the good Matches
    # TODO: use SuperGlue + SuperPoint
    max_points = 800
    n_scale = 12

    ORB = cv2.ORB_create(max_points, 1.2, n_scale)
    kp1, des1 = ORB.detectAndCompute(ref_frame, None)
    kp2, des2 = ORB.detectAndCompute(frame1, None)

    # NNDR matching.
    NNDR_limit = 60
    MIN_MATCH_COUNT = 10

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=12,  # 12
                        key_size=20,  # 20
                        multi_probe_level=2)  # 2

    search_params = dict(checks=100)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []

    for matchs in matches:  # Recorremos la lista accediendo a lista de matches
        if len(matchs) == 2:
            bestMatch = matchs[0]
            worstMatch = matchs[1]
            NNDR = bestMatch.distance / worstMatch.distance
            # Criterio con sentido
            if NNDR < NNDR_limit / 100:
                good.append(bestMatch)

    img3 = cv2.drawMatches(ref_frame, kp1, frame1, kp2, good, None)
    cv2.imshow("Matches", img3)

    if not len(good) > MIN_MATCH_COUNT:
        exit()

    # From matching list to xy
    pts1 = [kp1[m.queryIdx].pt for m in good]
    pts2 = [kp2[m.trainIdx].pt for m in good]
    open_pts1 = np.asarray(pts1)
    open_pts2 = np.asarray(pts2)

    # First camera is set as the origin
    p_RS_R_1, q_RS_R_1 = np.array([0., 0., 0.]), np.array([1., 0., 0., 0.])
    R_RS_R_1 = quaternion2Matrix(q_RS_R_1)
    T_RS_R_1 = np.eye(4)
    T_RS_R_1[0:3, 0:3] = R_RS_R_1
    T_RS_R_1[0:3, 3:4] = p_RS_R_1.reshape(3, 1)
    T_AC1 = T_RS_R_1

    # From the points estimate the fundamentalMatrix F_21_est.
    # TODO: use MAGSAC++
    F_21_est, mask = cv2.findFundamentalMat(open_pts1,open_pts2,cv2.FM_RANSAC, 3)

    # Normalize the solution
    F_21_est = F_21_est / F_21_est[2, 2]
    F_12_est = F_21_est.T

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
