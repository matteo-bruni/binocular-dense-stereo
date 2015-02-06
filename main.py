import cv2
import cv2.cv as cv
import sys
import numpy as np
import scipy.spatial
from numpy.linalg import inv

def get_RT(parameter_mat,n_img):

    R = []
    R1 = []
    R2 = []

    R1 = np.array(parameter_mat[0][0:9]).reshape(3,3)
    R2 = np.array(parameter_mat[n_img-1][0:9]).reshape(3,3)



    R = R2 * inv(R1)

    # for i in range(9):
    #     val = parametri[1][i]- parametri[0][i]
    #     R.append(val)
    # R = np.array(R)
    # R = R.reshape(3, 3)

    T = []
    for i in range(3):
        val = parameter_mat[n_img-1][i+9]-parameter_mat[0][i+9]
        T.append(val)
    T = np.array(T)

    d1 = np.zeros((2, 2))
    d2 = np.zeros((2, 2))

    return R,T,d1,d2


def getDisparity(imgLeft, imgRight, method="BM"):
    gray_left = cv2.cvtColor(imgLeft, cv.CV_BGR2GRAY)
    gray_right = cv2.cvtColor(imgRight, cv.CV_BGR2GRAY)
    print gray_left.shape
    c, r = gray_left.shape
    if method == "BM":
        sbm = cv.CreateStereoBMState()
        disparity = cv.CreateMat(c, r, cv.CV_32F)
        sbm.SADWindowSize = 5
        sbm.preFilterType = 1
        sbm.preFilterSize = 5
        sbm.preFilterCap = 30
        sbm.minDisparity = 0
        sbm.numberOfDisparities = 16
        sbm.textureThreshold = 0
        sbm.uniquenessRatio = 0
        sbm.speckleRange = 2
        sbm.speckleWindowSize = 100

        gray_left = cv.fromarray(gray_left)
        gray_right = cv.fromarray(gray_right)

        cv.FindStereoCorrespondenceBM(gray_left, gray_right, disparity, sbm)
        disparity_visual = cv.CreateMat(c, r, cv.CV_8U)
        cv.Normalize(disparity, disparity_visual, 0, 255, cv.CV_MINMAX)
        disparity_visual = np.array(disparity_visual)

    elif method == "SGBM":
        sbm = cv2.StereoSGBM()
        sbm.SADWindowSize = 100;  #Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
        sbm.numberOfDisparities = 16;  #con valori piu alti tipo 112 viene il contorno nero
        sbm.preFilterCap = 1;
        sbm.minDisparity = 0;  #con altri valori smongola
        sbm.uniquenessRatio = 7;
        sbm.speckleWindowSize = 100;
        sbm.speckleRange = 2;
        sbm.disp12MaxDiff = 1;
        sbm.fullDP = False;  #a True runna il full-scale two-pass dynamic programming algorithm

        disparity = sbm.compute(gray_left, gray_right)
        disparity_visual = cv2.normalize(disparity, alpha=0, beta=255, norm_type=cv2.cv.CV_MINMAX, dtype=cv2.cv.CV_8U)

    return disparity_visual


imgLeft = (cv2.imread('dataset_templeRing/templeR0001.png'))
imgRight = (cv2.imread('dataset_templeRing/templeR0002.png'))

try:
    method = sys.argv[3]
except IndexError:
    method = "SGBM"

disparity = getDisparity(imgLeft, imgRight, method)

cv2.imshow("disparity", disparity)
cv2.imshow("left", imgLeft)
cv2.imshow("right", imgRight)
cv2.waitKey(0)

K = np.array([[1520.4, 0., 302.32],
           [0, 1525.9, 246.87],
           [0, 0, 1]])

parametri = []
parameter_file = open('dataset_templeRing/templeR_par.txt', 'r')

for line in parameter_file:
    row = [float(j) for j in line[107:378].split()]
    parametri.append(row)

parametri = np.array(parametri)
parametri = np.delete(parametri, (0), axis=0)  #remove useless header

R,T,d1,d2 = get_RT(parametri,2)





# #templeR0002.png 1520.400000 0.000000 302.320000 0.000000 1525.900000 246.870000 0.000000 0.000000 1.000000 0.00272557078828676410 0.98353557606148900000 -0.18069405603193772000 0.99651741905514424000 -0.01773058775937118300 -0.08147797111723514800 -0.08334029507718225500 -0.17984270037758626000 -0.98015865977776562000 -0.0288222339759 -0.0306361018019 0.525505113107
# #"imgname.png k11 k12 k13 k21 k22 k23 k31 k32 k33 r11 r12 r13 r21 r22 r23 r31 r32 r33 t1 t2 t3"
# #The projection matrix for that image is given by K*[R t]
#
# (CM1, CM2, D1, D2, R, T, E, F) = loadCalibration(calibdir)
# # CM 3x3 -> K
# # D 1x5  ->
# # R 3x3
# # T 3x1
# # E 3x3
# # F 3x3
#
# R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(CM1, D1, CM2, D2, image_size, R, T, alpha=0)





#
# objpts=[]
# objPoints = []
# pointCounts = hor*ver
# R = [8]
# T = [8]
# E=[]
# F=[]
# R1 = []
# R2 = []
# P1=[]
# P2 =[]
# objectPoints = []
# imgPoints1 = []
# imgPoints2 = []
# imagePoints2 = []
# imagePoints1 = []
#
# while ((len(imagePoints1) < hor*ver) and (len(imagePoints2)< hor*ver)):
#
#      ret,imgr  = webCamHndlr_r.read(0)
#      ret,imgl  = webCamHndlr_l.read(1)
#      grey_imgr = cv2.cvtColor(imgr, cv.CV_BGR2GRAY)
#      grey_imgl = cv2.cvtColor(imgl, cv.CV_BGR2GRAY)
#      ret, cornersr =cv2.findChessboardCorners(grey_imgr,dims)
#      cv2.drawChessboardCorners(grey_imgr,dims,cornersr,0)
#      ret, cornersl =cv2.findChessboardCorners(grey_imgl,dims)
#      cv2.drawChessboardCorners(grey_imgl,dims,cornersl,0)
#      cv2.imshow("chessboard", grey_imgr)
#      cv2.imshow("chessboard1", grey_imgl)
#
#      objPoints = np.zeros((hor*ver,3), np.float32)
#      objPoints[:,:2] = np.mgrid[0:hor,0:ver].T.reshape(-1,2)
#
#      if cv.WaitKey(-1) == 32:
#
#           imagePoints1.append(cornersl)
#           imagePoints2.append(cornersr)
#           print len(imagePoints1)
#           objectPoints.append(objPoints)
#           cv2.imwrite("./test_images/img_r"+str(i)+".jpg",imgr)
#           cv2.imwrite("./test_images/img_l"+str(i)+".jpg",imgl)
#           i = i+1;
#
#
#           if cv2.waitKey(10) == 27:
#                break
# objectPoints = [np.asarray(x) for x in objectPoints]
# imagePoints1 = [np.asarray(x) for x in imagePoints1]
# imagePoints2 = [np.asarray(x) for x in imagePoints2]
#
# if( len(imagePoints1[0])== len(imagePoints2[0]) == len(objectPoints[0]) == len(objectPoints)== len(imagePoints2) == len(imagePoints1) ) : print len(imagePoints1[0])
#
#        cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objectPoints, imagePoints1, imagePoints2, (320,240))
#        print R
#
#        cv.StereoRectify(cameraMatrix1, cameraMatrix2, distCoeffs1, distCoeffs2,(imgr.width,imgr.height), R, T, R1, R2, P1, P2, Q, CV_CALIB_ZERO_DISPARITY, -1, (0, 0))
#        print  Q
#        np.savetxt('Q_mat.txt',Q)