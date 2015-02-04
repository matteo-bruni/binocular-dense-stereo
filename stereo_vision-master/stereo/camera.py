#!/usr/bin/env python2
from __future__ import print_function, division

import cv2

class CameraError(RuntimeError): pass

class Camera(object):
    def __init__(self, index):
        self.capture = cv2.VideoCapture(index)
        if not self.capture.isOpened():
            raise CameraError('Open failed for camera index {}'.format(index))

        self.index = index

    def grab(self):
        if not self.capture.grab():
            raise CameraError('Grab failed for camera index {}'
                              .format(self.index))

    def retrieve(self):
        success, image = self.capture.retrieve()
        if not success:
            raise CameraError('Retrieve failed for camera index {}'
                              .format(self.index))
        return image

    def release(self):
        self.capture.release()

    def set_size(self, size):
        self.width, self.height = map(int, size)

    width = property(
            lambda self: int(self.capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
            lambda self, width: self.capture.set(
                cv2.cv.CV_CAP_PROP_FRAME_WIDTH, int(width)))
    height = property(
            lambda self: int(self.capture.get(
                cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)),
            lambda self, height: self.capture.set(
                cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, int(height)))

    size = property(lambda self: (self.width, self.height), set_size)

class StereoCamera(object):
    def __init__(self, left_index, right_index, width, height):
        self.left_camera = Camera(left_index)
        self.right_camera = Camera(right_index)

        self.cameras = (self.left_camera, self.right_camera)

        self.left_camera.size = self.right_camera.size = (width, height)

    def grab(self):
        for camera in self.cameras:
            camera.grab()

    def retrieve(self):
        return [camera.retrieve() for camera in self.cameras]

    def release(self):
        for camera in self.cameras:
            camera.release()

    def get_width(self):
        return self.left_camera.width

    def set_width(self, width):
        self.left_camera.width = self.right_camera.width = width

    def get_height(self):
        return self.left_camera.height

    def set_height(self, height):
        self.left_camera.height = self.right_camera.height = height

    def get_size(self):
        return self.left_camera.size

    def set_size(self, size):
        self.left_camera.size = self.right_camera.size = size

    width = property(get_width, set_width)
    height = property(get_height, set_height)
    size = property(get_size, set_size)
