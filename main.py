import sys
import numpy as np
from pcl import registration
import cv2
import pcl

from utils import get_rot_trans_matrix_img2_wrt_img1, get_disparity, rotate_img, write_ply

ROTATE = True


def main():
    # Choose images
    left_img_number = "01"
    right_img_number = "02"

    # Load images
    img_left = cv2.imread('dataset_templeRing/templeR00%s.png' % left_img_number)
    img_right = cv2.imread('dataset_templeRing/templeR00%s.png' % right_img_number)

    height, width, depth = img_left.shape
    img_shape = (width, height)

    ##############################################################################################
    # Load Calibration Information
    ##############################################################################################
    # Calibration Matrix - same for each image
    K = np.array([[1520.4, 0., 302.32],
                  [0, 1525.9, 246.87],
                  [0, 0, 1]])

    # images are distorsion free
    d = np.zeros((5, 1))

    ##############################################################################################
    # Load Images Calibration from file
    ##############################################################################################
    calibration_file = open('dataset_templeRing/templeR_par.txt', 'r')
    all_images_parameters = []
    for line in calibration_file:
        row = [float(j) for j in line[107:378].split()]
        all_images_parameters.append(row)
    calibration_file.close()
    all_images_parameters = np.array(all_images_parameters[1:])  # remove useless header
    ##############################################################################################

    ##############################################################################################
    # Rectify images
    ##############################################################################################
    # Get Rotation Matrix and T of right images from the left one
    # r_left and r_right are the original rotation matrix
    R, T = get_rot_trans_matrix_img2_wrt_img1(all_images_parameters, int(left_img_number), int(right_img_number))

    # Compute stereo Rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K, d, K, d, img_shape, R, T, alpha=-1)

    Q = np.float32([[1, 0, 0, -0.5*width],
                    [0,-1, 0,  0.5*height], # turn points 180 deg around x-axis,
                    [0, 0, 0,  0.8*width], # so that y-axis looks up
                    [0, 0, 1,   0]])

    # Get map rectification
    map_left1, map_left2 = cv2.initUndistortRectifyMap(K, d, R1, P1, img_shape, cv2.CV_32FC1)
    map_right1, map_right2 = cv2.initUndistortRectifyMap(K, d, R2, P2, img_shape, cv2.CV_32FC1)

    # Apply Rectification
    left_rectified = cv2.remap(img_left, map_left1, map_left2, cv2.INTER_NEAREST)
    right_rectified = cv2.remap(img_right, map_right1, map_right2, cv2.INTER_NEAREST)
    ##############################################################################################

    ##############################################################################################
    # Compute disparity images
    ##############################################################################################
    # Compute disparity on rectified images
    disparity_method = "SGBM"
    disparity = get_disparity(left_rectified, right_rectified, disparity_method)
    ##############################################################################################

    # Project to 3d
    print 'generating 3d point cloud...'

    points = cv2.reprojectImageTo3D(disparity, Q)




    #cv2.triangulatePoints(P1, P2, points[1], points[2])

    ## tentativo matplotlib
    # print "3d shape", points.shape
    # temp_points= points.reshape(-1, 3)
    # for t in temp_points:
    #     if t[~np.isfinite(temp_points)]:
    #         temp_points.remove(t)
    #
    # x_plot = temp_points[:, 0]
    # y_plot = temp_points[:, 1]
    # z_plot = temp_points[:, 2]
    #
    # from matplotlib.pyplot import plot, axis, show, imshow, figure, gray
    # print "3D Plot"
    # # 3D plot
    # from mpl_toolkits.mplot3d import axes3d
    # colors = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2RGB)
    # mask = disparity > disparity.min()
    # out_points = points[mask]
    # out_colors = colors[mask]
    #
    # fig = figure()
    # ax = fig.gca(projection='3d')
    # ax.plot(x_plot,y_plot,z_plot,'k.')
    # # ax.plot(points, 'k.')
    # axis('off')


    # Generate ply
    #

    colors = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2RGB)
    mask = disparity > disparity.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply('out.ply', out_points, out_colors)
    print '%s saved' % 'out.ply'



    p = pcl.PointCloud()
    p._from_ply_file("out.ply")
    p.to_file("out2.pcd")

    # print "first cloud done..."
    # p2 = pcl.PointCloud()
    # p2._from_ply_file("out.ply")
    # print "second cloud done..."
    #
    # icp = registration.icp(p,p2,10)
    # print icp


    ##############################################################################################
    # Show Images
    ##############################################################################################
    if ROTATE:

        pre_rectify = np.hstack((rotate_img(img_left, 90), (rotate_img(img_right, 90))))
        after_rectify = np.hstack(((rotate_img(left_rectified, 90)), (rotate_img(right_rectified, 90))))
        total = np.vstack((pre_rectify, after_rectify))

    else:

        pre_rectify = np.hstack((img_left, img_right))
        after_rectify = np.hstack((left_rectified, right_rectified))
        total = np.vstack((pre_rectify, after_rectify))


    cv2.imshow("PreAfterRectify", total)
    # cv2.imshow("disparity", disparity)
    #
    # cv2.imshow("PreRectify", pre_rectify)
    # cv2.imshow("AfterRectify", after_rectify)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #
    # # Show images
    # cv2.imshow("disparity", disparity)
    # cv2.imshow("left1", img_left)
    # cv2.imshow("right2", img_right)
    # cv2.imshow("rectified_left", left_rectified)
    # cv2.imshow("rectified_right", right_rectified)

    # cv2.reprojectImageTo3D()


if __name__ == "__main__":
    main()


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


