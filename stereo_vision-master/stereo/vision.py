#!/usr/bin/env python2
from __future__ import print_function, division

import cv2
import numpy

from config import Calibration

def find_points(config, image, draw=True):
    if config.calibration.pattern == Calibration.CHESSBOARD:
        points = find_chessboard(config, image)
    elif config.calibration.pattern == Calibration.CIRCLES:
        points = find_circles(config, image)
    elif config.calibration.pattern == Calibration.SYMMETRIC_CIRCLES:
        points = find_circles(config, image, true)
    else:
        return None

    if draw and points is not None:
        cv2.drawChessboardCorners(image, config.calibration.pattern_size,
                                  points, True)

    return points

def find_chessboard(config, image):
    flags = 0
    flags |= cv2.CALIB_CB_ADAPTIVE_THRESH
    flags |= cv2.CALIB_CB_FAST_CHECK
    flags |= cv2.CALIB_CB_NORMALIZE_IMAGE

    rv, points = cv2.findChessboardCorners(image,
            config.calibration.pattern_size, flags=flags)

    if not rv:
        return None
    else:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.cornerSubPix(image_gray, points, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        return points

def find_circles(config, image, symmetric=False):
    flags = 0
    #flags |= cv2.CALIB_CB_CLUSTERING
    if symmetric:
        flags |= cv2.CALIB_CB_SYMMETRIC_GRID
    else:
        flags |= cv2.CALIB_CB_ASYMMETRIC_GRID

    rv, points = cv2.findCirclesGridDefault(image,
            config.calibration.pattern_size, flags=flags)

    if not rv:
        return None

    return points

def get_pattern_points(config):
    points = []
    if config.calibration.pattern in (Calibration.CHESSBOARD,
                                      Calibration.SYMMETRIC_CIRCLES):
        l = lambda i, j: (j, i, 0)
    elif config.pattern == Calibration.CIRCLES:
        l = lambda i, j: (2 * j + i % 2, i, 0)
    else:
        raise RuntimeError('Unknown pattern type')

    for i in range(config.calibration.pattern_height):
        for j in range(config.calibration.pattern_width):
            points.append(l(i, j))

    return numpy.array(points, numpy.float32)

def calibrate_camera(config, points):
    error, intrinsics, distortion, rotation, translation = (
        cv2.calibrateCamera(config.calibration.object_points, points,
                            config.camera.size))
    return error, intrinsics, distortion

def calibrate_stereo(config):
    term_crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    flags = 0
    #flags |= cv2.CALIB_FIX_INTRINSIC
    #flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    #flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    #flags |= cv2.CALIB_FIX_FOCAL_LENGTH
    flags |= cv2.CALIB_FIX_ASPECT_RATIO
    flags |= cv2.CALIB_ZERO_TANGENT_DIST
    flags |= cv2.CALIB_SAME_FOCAL_LENGTH
    flags |= cv2.CALIB_RATIONAL_MODEL
    flags |= cv2.CALIB_FIX_K3
    flags |= cv2.CALIB_FIX_K4
    flags |= cv2.CALIB_FIX_K5

    (error, config.camera.left_intrinsics, config.camera.left_distortion,
            config.camera.right_intrinsics, config.camera.right_distorition,
            R, T, E, F) = cv2.stereoCalibrate(
                    config.calibration.object_points,
                    config.calibration.left_points,
                    config.calibration.right_points,
                    config.camera.size, criteria=term_crit, flags=flags)

    return error, R, T

def stereo_rectify(config, R, T):
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                        config.camera.left_intrinsics,
                        config.camera.left_distortion,
                        config.camera.right_intrinsics,
                        config.camera.right_distortion,
                        config.camera.size, R, T, alpha=0)

    return R1, R2, P1, P2

def stereobm(config, left_image, right_image):
    bm = cv2.StereoBM(config.stereobm.preset_id,
                      config.stereobm.ndisparity,
                      config.stereobm.sad_window_size)

    left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
    right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)

    disparity = bm.compute(left_image, right_image, disptype=cv2.CV_16S)
    disparity *= 255 / (disparity.min() - disparity.max())
    disparity = disparity.astype(numpy.uint8)

    return disparity

def stereosgbm(config, left_image, right_image):
    #l_t = cv2.pyrDown(left_image)
    #r_t = cv2.pyrDown(right_image)

    stereo = cv2.StereoSGBM(
            config.stereosgbm.min_disparity,
            config.stereosgbm.num_disparities,
            config.stereosgbm.sad_window_size,
            config.stereosgbm.p1,
            config.stereosgbm.p2,
            config.stereosgbm.disp12_max_diff,
            config.stereosgbm.prefilter_cap,
            config.stereosgbm.uniqueness_ratio,
            config.stereosgbm.speckle_window_size,
            config.stereosgbm.speckle_range,
            config.stereosgbm.full_dp)

    disparity = stereo.compute(left_image, right_image)
    disparity *= 255 / (disparity.min() - disparity.max())
    disparity = disparity.astype(numpy.uint8)

    return disparity

def stereovar(config, left_image, right_image):
    return None
