import cv2
import cv2.cv as cv
import sys
import numpy as np
import scipy.spatial


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
        sbm.SADWindowSize = 5;  #Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
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


imgLeft = ( cv2.imread('dataset_templeRing/templeR0001.png') )
imgRight = ( cv2.imread('dataset_templeRing/templeR0002.png') )

try:
    method = sys.argv[3]
except IndexError:
    method = "SGBM"

disparity = getDisparity(imgLeft, imgRight, method)

cv2.imshow("disparity", disparity)
cv2.imshow("left", imgLeft)
cv2.imshow("right", imgRight)
cv2.waitKey(0)

tri = scipy.spatial.qhull.Delaunay(disparity)
print "triangulation done.."

cv2.imshow(tri)
