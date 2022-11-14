#####################################################################################
#
#
# Title: Utilities
#
# Date: 2 March 2021
#
#####################################################################################
#
# Authors: Richard Elvira, Pablo Azagra, Jesus Bermudez, Jose Lamarca, JMM Montiel
#
# Version: 1.04
#
#####################################################################################

import numpy as np
import random
import cv2
from scipy.linalg import expm

def getRayPlanesDescription(v,T_ac):
    """
    -input:
        v_c: direction vectors in the cam reference
        T_ac: cam reference seen from abs reference
    -output:
    """
    #U = [1,0,-vx,0]
    #V = [0,1,-vy,0]
    U_c = np.vstack((np.ones((1, v.shape[1])), np.zeros((1, v.shape[1])), -v[0, :], np.zeros((1, v.shape[1]))))
    V_c = np.vstack(( np.zeros((1, v.shape[1])), np.ones((1, v.shape[1])), -v[1, :], np.zeros((1, v.shape[1]))))
    U_a = np.linalg.inv(T_ac).T @ U_c
    V_a = np.linalg.inv(T_ac).T @ V_c
    return U_a, V_a


def triangulatePoints(x1, x2, K_c_inv, T_21):
    v1 = K_c_inv @ x1
    v2 = K_c_inv @ x2

    U1_2, V1_2 = getRayPlanesDescription(v1, T_21)
    U2_2, V2_2 = getRayPlanesDescription(v2, np.eye(4))

    X2 = np.zeros((4, v2.shape[1]))
    for kPoints in range(v2.shape[1]):
        M = np.stack((U1_2[:, kPoints], V1_2[:, kPoints], U2_2[:, kPoints], V2_2[:, kPoints]))
        u, s, vh = np.linalg.svd(M)
        X3D = np.reshape(vh[-1, :], (4, 1))
        X2[:, kPoints:kPoints+1] = X3D/X3D[3]

    return X2


def matchWithNearestNeighboursDistanceRatio(desc1, desc2, distRatio):
    """
    Nearest Neighbours Matching algorithm checking the Distance Ratio.
    A match is accepted only if its distance is less than distRatio times
    the distance to the second match.

    -input:
        desc1: descriptors from image 1 nDesc x 128
        desc2: descriptors from image 2 nDesc x 128
        distRatio:

    -output:
       matches: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
    """
    matches = []

    nDesc1 = desc1.shape[0]
    nDesc2 = desc2.shape[0]

    for kDesc1 in range(nDesc1):
        dist = np.sqrt(np.sum((desc2 - desc1[kDesc1, :]) ** 2, axis=1))
        indexSort = np.argsort(dist)
        if dist[indexSort[0]] < distRatio * dist[indexSort[1]]:
            matches.append([kDesc1, indexSort[0], dist[indexSort[0]]])
    return matches


def normalizationMatrix(nx,ny):
    """
    Estimation of fundamental matrix(F) by matched n matched points.
    n >= 8 to assure the algorithm.

    -input:
        nx: number of columns of the matrix
        ny: number of rows of the matrix
    -output:
        Nv: normalization matrix such that xN = Nv @ x
    """
    Nv = np.array([[1/nx, 0, -1/2], [0, 1/ny, -1/2], [0, 0, 1]])
    return Nv


def estimateFundamentalMatrixFromNormalizedMatches(x1n,x2n,N1,N2):
    """
    Estimation of fundamental matrix(F) by matched n matched points.
    n >= 8 to assure the algorithm.

    -input:
        x1n: 3xn matrix => column j is matched with column x2n[:,j]
        x2n: 3xn matrix => column j is matched with column x1n[:,j]
        N1: normalization matrix such that x1N = N1 @ x1
        N2: normalization matrix such that x1N = N2 @ x2
    -output:
        F_21: 3x3 matrix => x2.T @ F_21 @ x1 = 0
    """

    M = np.zeros((0, 9))
    for k in range(x1n.shape[1]):
        m_i = np.hstack((x2n[0, k] * x1n[: ,k].T, x2n[1, k] * x1n[: ,k].T, x2n[2, k] * x1n[: ,k].T))
        M = np.vstack((M, m_i))

    u, s, vh = np.linalg.svd(M)
    f = vh[-1, :]  # Ojo que esta función devuelve v^T no v
    Fn_21 = np.reshape(f, (3, 3)) #Aquí no hay traspuesta

    u2, s2, vh2 = np.linalg.svd(Fn_21)  # Ojo que esta función devuelve v^T no v
    s2[2] = 0

    Fn_21_s0 = u2 @ np.diag(s2) @ vh2
    F_21 = N2.T @ Fn_21_s0 @ N1

    return F_21, Fn_21_s0, Fn_21


def sampsonSquareDistances(F_21, x1, x2):
    num = np.square(np.sum(x2 * (F_21 @ x1), axis=0))
    den1 = F_21 @ x1
    den2 = F_21.T @ x2
    denA = np.sum(den1[0:3, :] * den1[0:3, :], axis=0)
    denB = np.sum(den2[0:3, :] * den2[0:3, :], axis=0)
    sd2 = num / (denA + denB)
    return sd2


def estimateFundamentalMatrixFromNormPointsWithRANSAC(x1n,x2n,N1,N2):
    """
    Estimation of fundamental matrix(F) by matched n matched points.
    n >= 8 to assure the algorithm.

    -input:
        x1n: 3xn matrix => column j is matched with column x2n[:,j]
        x2n: 3xn matrix => column j is matched with column x1n[:,j]
        N1: normalization matrix such that x1N = N1 @ x1
        N2: normalization matrix such that x1N = N2 @ x2
    -output:
        F_21: 3x3 matrix => x2.T @ F_21 @ x1 = 0
        inliers: n vector => logical, 1 means that index point is an inlier, otherwise an outlier
    """      
    spFrac = 0.8  # spurious fraction
    P = 0.999  # probability of selecting at least one sample without spurious
    pMinSet = 8  # number of points needed to compute the fundamental matrix
    thresholdFactor = 1.96  # a point is spurious if abs(r/s)>factor Threshold

    normalizedMatches = np.hstack((x1n[0:2, :].T, x2n[0:2, :].T))

    # number m of random samples
    nAttempts = np.round(np.log(1 - P) / np.log(1 - np.power((1 - spFrac), pMinSet)))
    nAttempts = nAttempts.astype(int)

    RANSACThresholdPix = 2 #[pix]
    RANSACThresholdSd = 0.01

    nVotesMax = 0
    for kAttempt in range(nAttempts):

        selNormalizedMatches = np.array(random.choices(normalizedMatches, k=pMinSet))
        x1ModelNorm = np.vstack((selNormalizedMatches[:, 0:2].T, np.ones((1, pMinSet))))
        x2ModelNorm = np.vstack((selNormalizedMatches[:, 2:4].T, np.ones((1, pMinSet))))

        F_21, Fn_21_s0, Fn_21 = estimateFundamentalMatrixFromNormalizedMatches(x1ModelNorm, x2ModelNorm, N1, N2)
        F_12, Fn_12_s0, Fn_12 = estimateFundamentalMatrixFromNormalizedMatches(x2ModelNorm, x1ModelNorm, N2, N1)

        l2_epi = Fn_21 @ x1n
        l2_epi /= np.sqrt(np.sum(l2_epi[0:2, :] ** 2, axis=0))

        l1_epi = (x2n.T @ Fn_21).T
        l1_epi /= np.sqrt(np.sum(l1_epi[0:2, :] ** 2, axis=0))

        distLineEp2Point2 = np.sum(l2_epi * x2n, axis=0)
        distLineEp1Point2 = np.sum(l1_epi * x1n, axis=0)
        sd2 = sampsonSquareDistances(Fn_21, x1n, x2n)

        votesPixDist = np.bitwise_and(np.abs(distLineEp1Point2) < RANSACThresholdPix, np.abs(distLineEp2Point2)<RANSACThresholdPix)
        votesSampson = sd2 < RANSACThresholdSd

        votes = votesPixDist
        nVotes = np.sum(votes)

        if nVotes > nVotesMax:
            print('kAttempt =' + str(kAttempt) + ', nVotesSampson = ' + str(sum(votesSampson)) + ',nVotesPixDist = ' + str(sum(votesPixDist)))
            nVotesMax = nVotes
            inliersMax = votes
            F_21_mostVoted = F_21
            F_12_mostVoted = F_12

    F_21_est = F_21_mostVoted/F_21_mostVoted[2,2]

    return F_21_est, inliersMax


def estimateFundamentalMatrixFromMatches(x1,x2):
    """
    Estimation of fundamental matrix(F) by matched n matched points.
    n >= 8 to assure the algorithm.

    -input:
        x1: 3xn matrix => column j is matched with column x2n[:,j]
        x2: 3xn matrix => column j is matched with column x1n[:,j]
        N1: normalization matrix such that x1N = N1 @ x1
        N2: normalization matrix such that x1N = N2 @ x2
    -output:
        F_21: 3x3 matrix => x2.T @ F_21 @ x1 = 0
    """

    M = np.zeros((0, 9))
    for k in range(x1.shape[1]):
        m_i = np.hstack((x2[0, k] * x1[: ,k].T, x2[1, k] * x1[: ,k].T, x2[2, k] * x1[: ,k].T))
        M = np.vstack((M, m_i))

    u, s, vh = np.linalg.svd(M)
    f = vh[-1, :]  # Ojo que esta función devuelve v^T no v
    F_21 = np.reshape(f, (3, 3)) #Aquí no hay traspuesta

    u2, s2, vh2 = np.linalg.svd(F_21)  # Ojo que esta función devuelve v^T no v
    s2[2] = 0

    F_21 = u2 @ np.diag(s2) @ vh2

    return F_21


def estimateFundamentalMatrixFromMatchesWithRANSAC(x1,x2):
    """
    Estimation of fundamental matrix(F) by matched n matched points.
    n >= 8 to assure the algorithm.

    -input:
        x1: 3xn matrix => column j is matched with column x2n[:,j]
        x2: 3xn matrix => column j is matched with column x1n[:,j]
    -output:
        F_21: 3x3 matrix => x2.T @ F_21 @ x1 = 0
        inliers: n vector => logical, 1 means that index point is an inlier, otherwise an outlier
    """      
    spFrac = 0.4  # spurious fraction
    P = 0.999  # probability of selecting at least one sample without spurious
    pMinSet = 8  # number of points needed to compute the fundamental matrix
    thresholdFactor = 1.96  # a point is spurious if abs(r/s)>factor Threshold

    matches = np.hstack((x1[0:2, :].T, x2[0:2, :].T))

    # number m of random samples
    nAttempts = np.round(np.log(1 - P) / np.log(1 - np.power((1 - spFrac), pMinSet)))
    nAttempts = nAttempts.astype(int)

    RANSACThresholdPix = 2 #[pix]
    RANSACThresholdSd = 0.01

    nVotesMax = 0
    for kAttempt in range(nAttempts):

        selMatches = np.array(random.choices(matches, k=pMinSet))
        x1Model = np.vstack((selMatches[:, 0:2].T, np.ones((1, pMinSet))))
        x2Model = np.vstack((selMatches[:, 2:4].T, np.ones((1, pMinSet))))

        F_21 = estimateFundamentalMatrixFromMatches(x1Model, x2Model)
        F_12 = estimateFundamentalMatrixFromMatches(x2Model, x1Model)

        l2_epi = F_21 @ x1
        l2_epi /= np.sqrt(np.sum(l2_epi[0:2, :] ** 2, axis=0))

        l1_epi = (x2.T @ F_21).T
        l1_epi /= np.sqrt(np.sum(l1_epi[0:2, :] ** 2, axis=0))

        distLineEp2Point2 = np.sum(l2_epi * x2, axis=0)
        distLineEp1Point2 = np.sum(l1_epi * x1, axis=0)

        votesPixDist = np.bitwise_and(np.abs(distLineEp1Point2) < RANSACThresholdPix, np.abs(distLineEp2Point2)<RANSACThresholdPix)

        votes = votesPixDist
        nVotes = np.sum(votes)

        if nVotes > nVotesMax:
            print('kAttempt =' + str(kAttempt) + ',nVotesPixDist = ' + str(sum(votesPixDist)))
            nVotesMax = nVotes
            inliersMax = votes
            F_21_mostVoted = F_21
            F_12_mostVoted = F_12

    F_21_est = F_21_mostVoted/F_21_mostVoted[2,2]

    return F_21_est, inliersMax


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


def estimateRotAndTranslFromEssentialMatrix(E_21):
    """
    Descompose the essential matrix between two view to estimate the relative traslation
    and rotation between them. It provides 2 possible rotations, it is necessary to validate
    which one is correct.

    -input:
        E_21: 3x3 matrix -> Essential Matrix
    -output:
        t_21: 3 vector -> relative traslation vector
        RA_21: 3x3 matrix -> relative rotation matrix
        RB_21: 3x3 matrix -> relative rotation matrix
    """
    W = np.array([[0,-1,0],
                  [1,0,0],
                  [0,0,1]])
    print(E_21)
    u, s, vh = np.linalg.svd(E_21)
    RA_21 = u @ W @ vh
    RB_21 = u @ W.T @ vh
    t_21 = u @ np.array([[0],[0],[1]])
    return t_21, RA_21, RB_21

def crossMatrix(x):
    """
    -input:
        x:
    -output:
        M:
    """
    M = np.array([[0, -x[2], x[1]],
                  [x[2], 0, -x[0]],
                  [-x[1], x[0], 0]], dtype="object")
    return M

def quaternion2Matrix(q):  # (w,x,y,z)
    R = np.array(
        [[q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2, 2 * (q[1] * q[2] - q[0] * q[3]),
          2 * (q[1] * q[3] + q[0] * q[2])],
         [2 * (q[2] * q[1] + q[0] * q[3]), q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2,
          2 * (q[2] * q[3] - q[0] * q[1])],
         [2 * (q[3] * q[1] - q[0] * q[2]), 2 * (q[3] * q[2] + q[0] * q[1]),
          q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2]], dtype="object")
    return R

def descomposePoseFromFundamental(F_21, K1, K2):
    """
    Descompose the fundamental matrix between two view to estimate the relative traslation
    and rotation between them. It provides 2 possible rotations, it is necessary to validate
    which one is correct.

    -input:
        F_21: 3x3 matrix -> Fundamental Matrix from camera 2 to camera 1
    -output:
        t_21: 3 vector -> relative traslation vector
        RA_21: 3x3 matrix -> relative rotation matrix
        RB_21: 3x3 matrix -> relative rotation matrix
    """
    E_21 = K2.T @ F_21 @ K1
    t_21, RA_21, RB_21 = estimateRotAndTranslFromEssentialMatrix(E_21)

    return t_21, RA_21, RB_21


def testEstimatedPosesFromFundamental(x1, x2, K_c1, K_c2, RA_21, RB_21, t_21):
    """
    Test the 4 possiblities of movement between cameras with the triangulation
    of the x1 and x2 matches. The correct pose is which has most of the 3D points
    in front of both cameras. The 4 posibilities are [RA_21; t_21], [RA_21; -t_21],
    [RB_21; t_21], [RB_21; -t_21]

    -input:
        x1: 2xn matrix -> n 2D points in the image 1. Each i index is matched with the same indes in x2.
        x2: 2xn matrix -> n 2D points in the image 2. Each i index is matched with the same indes in x1.
        K_c1: 3x3 matrix -> Camera 1 calibration matrix.
        K_c2: 3x3 matrix -> Camera 2 calibration matrix.
        RA_21: 3x3 matrix ->
        RB_21: 3x3 matrix ->
        t_21: 3 vector ->
    -output:
        R_21: 3x3 matrix ->
        t_21: 3 vector ->
    """

    points_front = np.zeros((4,1))

    T_test1_21 = np.eye(4,4)
    T_test1_21[0:3, 0:3] = RA_21
    T_test1_21[0:3, 3] = t_21
    x3D = triangulateFrom2View(x1, x2, K_c1, K_c2, T_test1_21)
    x3D_c2 = T_test1_21 @ x3D.T
    points_front[0] = np.sum(x3D[:, 2] > 0) + np.sum(x3D_c2[2, :] > 0)
    test_max = 0
    T_21_max = T_test1_21
    x3D_max = x3D


    T_test2_21 = np.eye(4,4)
    T_test2_21[0:3, 0:3] = RA_21
    T_test2_21[0:3, 3] = -1*t_21
    x3D_t2 = triangulateFrom2View(x1, x2, K_c1, K_c2, T_test2_21)
    x3D_c2 = T_test2_21 @ x3D_t2.T
    points_front[1] = np.sum(x3D_t2[:, 2] > 0) + np.sum(x3D_c2[2, :] > 0)
    if points_front[test_max] < points_front[1]:
        test_max = 1
        T_21_max = T_test2_21
        x3D_max = x3D_t2

    T_test3_21 = np.eye(4,4)
    T_test3_21[0:3, 0:3] = RB_21
    T_test3_21[0:3, 3] = t_21
    x3D_t3 = triangulateFrom2View(x1, x2, K_c1, K_c2, T_test3_21)
    x3D_c2 = T_test3_21 @ x3D_t3.T
    points_front[2] = np.sum(x3D_t3[:, 2] > 0) + np.sum(x3D_c2[2, :] > 0)
    if points_front[test_max] < points_front[2]:
        test_max = 2
        T_21_max = T_test3_21
        x3D_max = x3D_t3

    T_test4_21 = np.eye(4,4)
    T_test4_21[0:3, 0:3] = RB_21
    T_test4_21[0:3, 3] = -1 * t_21
    x3D_t4 = triangulateFrom2View(x1, x2, K_c1, K_c2, T_test4_21)
    x3D_c2 = T_test4_21 @ x3D_t4.T
    points_front[3] = np.sum(x3D_t4[:, 2] > 0) + np.sum(x3D_c2[2, :] > 0)
    if points_front[test_max] < points_front[3]:
        test_max = 3
        T_21_max = T_test4_21
        x3D_max = x3D_t4

    return T_21_max, x3D_max


def extractPose(image, points3d, des, sift, K, dist):
    """
    	Obtain the camera pose by extracting the matches from 3dpoints and the image keypoints using PnP solution
    	Input:
    	    image: View from which extract the camera pose
    	    points3d: 3D points in the scene
    	    des: descriptor associated to the 3d points that will be use for the matching.
    	    sift: Instance of the sift extractor.
    	    K: Camera Matrix
    	    dist: Distortion coef
    	Output:
    	    T_31_est: Transformation matrix from the camera to the original camera
    """

    #######  Extract Keypoints from the image and match with des parameter
    kp3, des3 = sift.detectAndCompute(image, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(np.asarray(des, np.float32), np.asarray(des3, np.float32), k=2)
    pts3 = []
    pts13 = []
    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pts3.append(kp3[m.trainIdx].pt)
            pts13.append(points3d[m.queryIdx, :])
            good.append(m)
    P3_1 = np.array(pts13)
    P3 = np.array(pts3)

    ##### If the number of matches is less than 6 the problem has no solution
    if P3.shape[0] < 6:
        return None

    ##### The solution is obtained using the cv2.solvePnPRansac.
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(np.ascontiguousarray(P3_1[:, :3]).reshape((P3.shape[0], 1, 3)),
                                                     np.ascontiguousarray(P3[:, :2]).reshape((P3.shape[0], 1, 2)), K,
                                                     dist, iterationsCount=10000, flags=cv2.SOLVEPNP_P3P)
    if not retval:
        return None

    ########### The transformation matrix is obtained from the vector of rotation and translation
    R_31_est = expm(crossMatrix(rvec))
    t_31_est = tvec
    T_31_est = np.vstack((np.hstack((R_31_est, t_31_est)), np.array([0, 0, 0, 1])))

    return T_31_est