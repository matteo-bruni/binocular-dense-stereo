#!/usr/bin/env python2
from __future__ import print_function, division

import sys
import time
import threading

import cairo
import cv2
import numpy
from gi.repository import Gtk, Gdk, GLib, GObject

import vision
from config import Config
from camera import StereoCamera

BUILDER_FILE = 'res/gui.xml'

class Builder(Gtk.Builder):
    def get_object(self, key):
        obj = super(Builder, self).get_object(key)
        if obj is None:
            raise KeyError('Key "{}" not found in builder.'.format(key))
        return obj

class VideoView(object):
    def __init__(self, title):
        self.view = Gtk.Window()
        self.view.set_title(title)
        self.view.set_has_resize_grip(False)

        self.draw = Gtk.DrawingArea()
        self.view.add(self.draw)

    def set_default_size(self, width, height):
        self.view.set_default_size(width, height)

    def show_all(self):
        self.view.show_all()

    def hide(self):
        self.view.hide()

    def update(self, image):
        if image is None:
            return
        ctx = Gdk.cairo_create(self.draw.get_window())

        view_width, view_height = self.view.get_size()
        sx = view_width / image.shape[1]
        sy = view_height / image.shape[0]
        ctx.scale(sx, sy)

        # Need to insert an empty alpha channel value
        image_alpha = numpy.insert(image, 3, 255, 2)
        flat_image = image_alpha.flatten()

        surface = cairo.ImageSurface.create_for_data(flat_image,
                cairo.FORMAT_ARGB32, image.shape[1], image.shape[0])
        ctx.set_source_surface(surface)
        ctx.paint()

    def update_depth(self, image):
        if image is None:
            return
        ctx = Gdk.cairo_create(self.draw.get_window())

        #ctx.set_source_rgb(0, 0, 0)
        #ctx.paint()

        view_width, view_height = self.view.get_size()
        sx = view_width / image.shape[1]
        sy = view_height / image.shape[0]
        ctx.scale(sx, sy)

        flat_image = image.flatten()

        #print(flat_image.min(), flat_image.max())
        #flat_image -= flat_image.min()
        #try:
        #    flat_image *= 255 / flat_image.max()
        #except RuntimeWarning:
        #    return
        #flat_image = flat_image.astype(numpy.uint8)
        #print(flat_image.min(), flat_image.max())

        flat_image = numpy.dstack((flat_image, flat_image, flat_image))
        flat_image = numpy.insert(flat_image, 3, 255, 2)
        flat_image = flat_image.flatten()

        surface = cairo.ImageSurface.create_for_data(flat_image,
                cairo.FORMAT_ARGB32, image.shape[1], image.shape[0])
        ctx.set_source_surface(surface)
        ctx.paint()

class Page(object):
    def __init__(self, builder, config):
        self.builder = builder
        self.config = config
        self.widget = None

    def on_forward(self, current_page):
        'Called to find out what page to go to next.'
        return current_page + 1

    def on_prepare(self):
        'Called right as this page is about to be shown.'
        return

    def on_cleanup(self):
        'Called right before another page is going to be shown.'
        return

    def _do_save_dialog(self, title, current_name=None):
        dialog = Gtk.FileChooserDialog(title, self.builder.get_object('main'),
                Gtk.FileChooserAction.SAVE, (
                    Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                    Gtk.STOCK_SAVE, Gtk.ResponseType.ACCEPT))
        dialog.set_do_overwrite_confirmation(True)
        dialog.set_create_folders(True)
        if current_name is not None:
            dialog.set_current_name(current_name)

        for name, pattern in (('JSON', '*.json'), ('All Files', '*')):
            file_filter = Gtk.FileFilter()
            file_filter.set_name(name)
            file_filter.add_pattern(pattern)
            dialog.add_filter(file_filter)

        if dialog.run() == Gtk.ResponseType.ACCEPT:
            file_name = dialog.get_filename()
        else:
            file_name = None

        dialog.destroy()
        return file_name

class Setup(Page):
    'The first setup page of the GUI.'

    def __init__(self, builder, config):
        super(Setup, self).__init__(builder, config)

        self.builder.get_object('load_points_button').connect('file-set',
                lambda x: self.builder.get_object(
                    'load_points_radio').set_active(True))

        self.builder.get_object('load_params_button').connect('file-set',
                lambda x: self.builder.get_object(
                    'load_params_radio').set_active(True))

    def on_forward(self, current_page):
        live_radio = self.builder.get_object('calibration_live_radio')
        if live_radio.get_active():
            return 1

        load_points_radio = self.builder.get_object('load_points_radio')

        if load_points_radio.get_active():
            load_points_filename = self.builder.get_object(
                    'load_points_button').get_filename()

            try:
                self.config.load_points(load_points_filename)
            except Exception as e:
                dialog = Gtk.MessageDialog(self.builder.get_object('main'),
                        Gtk.DialogFlags.MODAL, Gtk.MessageType.ERROR,
                        Gtk.ButtonsType.OK, 'Error: {}'.format(e))
                dialog.run()
                dialog.destroy()
                raise
            return 2

        load_params_radio = self.builder.get_object('load_params_radio')
        if load_params_radio.get_active():
            load_params_filename = self.builder.get_object(
                    'load_params_button').get_filename()

            try:
                self.config.load_params(load_params_filename)
            except Exception as e:
                dialog = Gtk.MessageDialog(self.builder.get_object('main'),
                        Gtk.DialogFlags.MODAL, Gtk.MessageType.ERROR,
                        Gtk.ButtonsType.OK, 'Error: {}'.format(e))
                dialog.run()
                dialog.destroy()
                raise
            return 3

class Capture(Page):
    def __init__(self, builder, config):
        super(Capture, self).__init__(builder, config)
        self.update = False
        self.left_image = None
        self.right_image = None

        self.left_view = VideoView('Left Camera')
        self.right_view = VideoView('Right Camera')

        self.count_entry = self.builder.get_object('snap_count_entry')

        self.builder.get_object('snap_button').connect('clicked', self.on_snap)
        self.builder.get_object('save_points_button').connect('clicked',
                                self.on_save)

    def on_prepare(self):
        self.camera = StereoCamera(self.config.camera.left_id,
                                    self.config.camera.right_id,
                                    self.config.camera.width,
                                    self.config.camera.height)

        self.count_entry.set_text('0')

        for view in (self.left_view, self.right_view):
            view.set_default_size(self.config.camera.width,
                                  self.config.camera.height)
            view.show_all()

        self.update = True
        GLib.idle_add(self.on_update)

    def on_cleanup(self):
        self.update = False
        self.left_view.hide()
        self.right_view.hide()

        self.camera.release()
        self.camera = None

    def on_update(self):
        if not self.update:
            return False

        self.camera.grab()
        self.left_image, self.right_image = self.camera.retrieve()

        self.left_points = vision.find_points(self.config, self.left_image)
        self.right_points = vision.find_points(self.config, self.right_image)

        # Redraw the windows
        self.left_view.update(self.left_image)
        self.right_view.update(self.right_image)

        return True

    def on_snap(self, button):
        if None in (self.left_points, self.right_points):
            return

        self.config.calibration.append_point(
                vision.get_pattern_points(self.config),
                self.left_points, self.right_points)
        self.count_entry.set_text(str(self.config.calibration.len_points()))

    def on_save(self, button):
        file_name = self._do_save_dialog('Save captured points as...',
                                         'points.json')
        if file_name is not None:
            self.config.save_points(file_name)

class Calibrate(Page):
    def __init__(self, builder, config):
        super(Calibrate, self).__init__(builder, config)
        self.progress = self.builder.get_object('calibrating_progress')
        self.save_button = self.builder.get_object('save_params_button')
        self.thread = None

        self.save_button.connect('clicked', self.on_save)

    def on_prepare(self):
        self.save_button.set_sensitive(False)
        assistant = self.builder.get_object('main')
        assistant.set_page_complete(
                self.builder.get_object('page_calibrating_box'), False)

        self.thread = threading.Thread(target=self.calibrate, name='calibrate')
        self.thread.start()

    def on_cleanup(self):
        self.thread.join(5)
        self.thread = None

    def on_save(self, button):
        file_name = self._do_save_dialog('Save camera parameters as...',
                                         'params.json')
        if file_name is not None:
            self.config.save_params(file_name)

    def calibrate(self):
        self.update_status(0 / 4, 'Calibrating left camera')
        rv = vision.calibrate_camera(self.config,
                                     self.config.calibration.left_points)
        print('Left reprojection error:', rv[0])
        self.config.camera.left_intrinsics = rv[1]
        self.config.camera.left_distortion = rv[2]

        self.update_status(1 / 4, 'Calibrating right camera')
        rv = vision.calibrate_camera(self.config,
                                     self.config.calibration.right_points)
        print('Right reprojection error:', rv[0])
        self.config.camera.right_intrinsics = rv[1]
        self.config.camera.right_distortion = rv[2]

        self.update_status(2 / 4, 'Calibrating stereo camera')
        error, R, T = vision.calibrate_stereo(self.config)
        print('Stereo reprojection error:', error)

        self.update_status(3 / 4, 'Rectifying')
        rv = vision.stereo_rectify(self.config, R, T)
        self.config.camera.R1 = rv[0]
        self.config.camera.R2 = rv[1]
        self.config.camera.P1 = rv[2]
        self.config.camera.P2 = rv[3]

        self.update_status(4 / 4, 'Done')
        self.update_complete()

    def update_status(self, fraction, message):
        def func():
            self.progress.set_fraction(fraction)
            self.progress.set_text(message)
            return False
        GLib.idle_add(func)

    def update_complete(self):
        def func():
            self.save_button.set_sensitive(True)
            assistant = self.builder.get_object('main')
            assistant.set_page_complete(
                    self.builder.get_object('page_calibrating_box'), True)
        GLib.idle_add(func)

class Vision(Page):
    def __init__(self, builder, config):
        super(Vision, self).__init__(builder, config)
        self.update = False
        self.left_image = None
        self.right_image = None

        self.left_view = VideoView('Left Camera')
        self.right_view = VideoView('Right Camera')
        self.depth_view = VideoView('Depth View')

        self.rectify_check = self.builder.get_object(
                'rectify_check')

        self.notebook = self.builder.get_object('stereo_notebook')

    def on_prepare(self):
        self.camera = StereoCamera(self.config.camera.left_id,
                                   self.config.camera.right_id,
                                   self.config.camera.width,
                                   self.config.camera.height)

        for view in (self.left_view, self.right_view, self.depth_view):
            view.set_default_size(self.camera.width,
                                  self.camera.height)
            view.show_all()

        self.left_maps = cv2.initUndistortRectifyMap(
            self.config.camera.left_intrinsics,
            self.config.camera.left_distortion,
            self.config.camera.R1, self.config.camera.P1,
            self.camera.size, cv2.CV_16SC2)

        self.right_maps = cv2.initUndistortRectifyMap(
                self.config.camera.right_intrinsics,
                self.config.camera.right_distortion,
                self.config.camera.R2, self.config.camera.P2,
                self.camera.size, cv2.CV_16SC2)

        self.update = True
        GLib.idle_add(self.on_update)

    def on_cleanup(self):
        self.update = False
        self.left_view.hide()
        self.right_view.hide()
        self.depth_view.hide()

        self.camera.release()
        self.camera = None

    def on_update(self):
        if not self.update:
            return False

        self.camera.grab()
        left_image, right_image = self.camera.retrieve()

        if self.rectify_check.get_active():
            left_image = cv2.remap(left_image, self.left_maps[0],
                                   self.left_maps[1], cv2.INTER_LINEAR)
            right_image = cv2.remap(right_image, self.right_maps[0],
                                    self.right_maps[1], cv2.INTER_LINEAR)

        if self.notebook.get_current_page() == 0:
            disparity_image = vision.stereobm(self.config,
                                              left_image, right_image)
        elif self.notebook.get_current_page() == 1:
            disparity_image = vision.stereosgbm(self.config,
                                                left_image, right_image)
        elif self.notebook.get_current_page() == 2:
            disparity_image = vision.stereovar(self.config,
                                               left_image, right_image)

        # Redraw the windows
        self.left_view.update(left_image)
        self.right_view.update(right_image)
        self.depth_view.update_depth(disparity_image)

        return True

class StereoGui(object):
    def __init__(self):
        self.builder = Builder()
        self.builder.add_from_file(BUILDER_FILE)

        self.config = Config()
        self.config.init_gui(self.builder)

        self.pages = (
            Setup(self.builder, self.config),
            Capture(self.builder, self.config),
            Calibrate(self.builder, self.config),
            Vision(self.builder, self.config),
        )
        self.previous_page = None

        assistant = self.builder.get_object('main')
        assistant.set_forward_page_func(self.on_forward, None)
        assistant.connect('prepare', self.on_prepare)
        assistant.connect('close', Gtk.main_quit)
        assistant.connect('cancel', Gtk.main_quit)

    def start(self):
        self.builder.get_object('main').show_all()
        Gtk.main()

    def on_forward(self, current_page, data):
        try:
            page = self.pages[current_page]
        except IndexError:
            return current_page + 1
        return page.on_forward(current_page)

    def on_prepare(self, assistant, widget):
        page = self.pages[assistant.get_current_page()]
        if self.previous_page is not None:
            self.previous_page.on_cleanup()

        page.on_prepare()
        self.previous_page = page

def main(argv):
    GObject.threads_init()
    StereoGui().start()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
