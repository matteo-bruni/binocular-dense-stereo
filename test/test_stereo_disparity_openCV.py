# #!/usr/bin/env python
#
# '''
# Simple example of stereo image matching and point cloud generation.
#
# Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
# '''
#
# import numpy as np
# import cv2
#
# ply_header = '''ply
# format ascii 1.0
# element vertex %(vert_num)d
# property float x
# property float y
# property float z
# property uchar red
# property uchar green
# property uchar blue
# end_header
# '''
#
# def write_ply(fn, verts, colors):
# verts = verts.reshape(-1, 3)
#     colors = colors.reshape(-1, 3)
#     verts = np.hstack([verts, colors])
#     with open(fn, 'w') as f:
#         f.write(ply_header % dict(vert_num=len(verts)))
#         np.savetxt(f, verts, '%f %f %f %d %d %d')
#
#
# if __name__ == '__main__':
#     print 'loading images...'
#     imgL = ( cv2.imread('dataset_tsukuba/scene1.row3.col3.ppm') )  # downscale images for faster processing
#     imgR = ( cv2.imread('dataset_tsukuba/scene1.row3.col4.ppm') )
#
#     imgL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
#     imgL = cv2.convertScaleAbs(imgL)
#
#     imgR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
#     imgR = cv2.convertScaleAbs(imgR)
#
#
#     print imgL.dtype, imgL.shape
#     print imgR.dtype, imgR.shape
#
#     # disparity range is tuned for 'aloe' image pair
#     window_size = 3
#     min_disp = 16
#     num_disp = 112-min_disp
#     stereo = cv2.StereoSGBM(minDisparity = min_disp,
#         numDisparities = num_disp,
#         SADWindowSize = window_size,
#         uniquenessRatio = 10,
#         speckleWindowSize = 100,
#         speckleRange = 32,
#         disp12MaxDiff = 1,
#         P1 = 8*3*window_size**2,
#         P2 = 32*3*window_size**2,
#         fullDP = False
#     )
#
#     print 'computing disparity...'
#     disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
#
#
#     #
#     # # Initialize a stereo block matcher. See documentation for possible arguments
#     # block_matcher = cv2.StereoBM()
#     # # Compute disparity image
#     # disparity = block_matcher.compute(imgL, imgR)
#     # # Show normalized version of image so you can see the values
#     # cv2.imshow('ghxfu', disparity / 255.)
#
#     #
#     # print 'generating 3d point cloud...',
#     # h, w = imgL.shape[:2]
#     # f = 0.8*w                          # guess for focal length
#     # Q = np.float32([[1, 0, 0, -0.5*w],
#     #                 [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
#     #                 [0, 0, 0,     -f], # so that y-axis looks up
#     #                 [0, 0, 1,      0]])
#     # points = cv2.reprojectImageTo3D(disp, Q)
#     # colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
#     # mask = disp > disp.min()
#     # out_points = points[mask]
#     # out_colors = colors[mask]
#     # out_fn = 'out.ply'
#     # write_ply('out.ply', out_points, out_colors)
#     # print '%s saved' % 'out.ply'
#     #
#     # cv2.imshow('left', imgL)
#     cv2.imshow('disparity', (disp-min_disp)/num_disp)
#     cv2.waitKey()
#     cv2.destroyAllWindows()

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


imgLeft = ( cv2.imread('dataset_tsukuba/scene1.row3.col3.ppm') )
imgRight = ( cv2.imread('dataset_tsukuba/scene1.row3.col4.ppm') )

try:
    method = sys.argv[3]
except IndexError:
    method = "SGBM"

disparity = getDisparity(imgLeft, imgRight, method)

cv2.imshow("disparity", disparity)
cv2.imshow("left", imgLeft)
cv2.imshow("right", imgRight)
cv2.waitKey(0)

