stereo_vision
=============

Author: Tom Wambold <tom5760@gmail.com>

stereo_vision provides a GTK GUI for experimenting with the stereo vision
algorithms provided by OpenCV.

The GUI can calibrate cameras with either a chessboard pattern, or circle
patterns.  After calibration and rectification, the camera parameters can be
saved to a JSON file.  Then, a real-time view of the output of OpenCV's stereo
vision algorithms is shown.  Parameters can be tweaked and the output is
updated immediately.

stereo_vision currently can use the StereoBM and StereoSGBM algorithms in
OpenCV.

Dependencies
------------

stereo_vision requires the following packages:
 * Python (tested with version 2.7.3)
 * OpenCV (tested with version 2.4.1)
 * PyGObject (tested with version 3.2.2)
 * PyCairo (tested with version 1.10.0)

On Arch Linux, install the following packages:

    pacman -S opencv python2 python2-gobject python2-cairo

To-Do
-----

 * Wrap [StereoVar][1] with Cython or something.

[1]: http://docs.opencv.org/modules/contrib/doc/stereo.html#stereovar
