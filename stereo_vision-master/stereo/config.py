#!/usr/bin/env python2
from __future__ import print_function

import json

import cv2
import numpy

class Config(object):
    def __init__(self):
        self.camera = Camera()
        self.calibration = Calibration()
        self.stereobm = StereoBM()
        self.stereosgbm = StereoSGBM()
        self.stereovar = StereoVar()

    def save_params(self, file_name):
        with open(file_name, 'w') as f:
            Config._json_dump(self.camera, f)

    def load_params(self, file_name):
        with open(file_name) as f:
            d = json.load(f)

        self.camera.update(d)

    def save_points(self, file_name):
        with open(file_name, 'w') as f:
            Config._json_dump(self.calibration, f)

    def load_points(self, file_name):
        with open(file_name) as f:
            d = json.load(f)

        self.calibration.update(d)

    def init_gui(self, builder):
        for x in (self.camera, self.calibration, self.stereobm,
                  self.stereosgbm, self.stereovar):
            x.init_gui(builder)

    @staticmethod
    def _json_dump(obj, f):
        json.dump(obj, f, indent=4, separators=(',', ': '), cls=Encoder)

class Section(object):
    # This class variable holds the attribute names that should be exported
    # when saved.
    CONFIG = ()

    def __init__(self):
        self._listeners = {}

    def __setattr__(self, key, value):
        super(Section, self).__setattr__(key, value)
        try:
            listeners = self._listeners[key]
        except KeyError:
            return
        for listener in listeners:
            listener(key, value)

    def update(self, d):
        self.__dict__.update(d)

    def add_listener(self, key, func):
        self._listeners.setdefault(key, []).append(func)

    def _connect_adjustment(self, builder, conf_key, gui_key,
                            low=None, high=None, step=1, page=5, typefunc=int):
        adjustment = builder.get_object(gui_key).get_adjustment()

        # Set the boundaries
        if low is not None:
            adjustment.set_lower(low)
        if high is not None:
            adjustment.set_upper(high)
        if step is not None:
            adjustment.set_step_increment(step)
        if page is not None:
            adjustment.set_page_increment(page)

        # Set the default value
        adjustment.set_value(getattr(self, conf_key))

        # Hook up signals for changes
        adjustment.connect('value-changed',
                lambda adj: setattr(self, conf_key, typefunc(adj.get_value())))
        self.add_listener(conf_key,
                lambda k, v: adjustment.set_value(typefunc(v)))

    def _connect_combobox(self, builder, conf_key, gui_key, values):
        combo = builder.get_object(gui_key)

        # Set the possible values
        combo.remove_all()
        for value in values:
            combo.append_text(value)

        # Set the default value
        combo.set_active(values.index(getattr(self, conf_key)))

        # Hook up signals for changes
        combo.connect('changed',
                lambda combo: setattr(self, conf_key, combo.get_active_text()))
        self.add_listener(conf_key,
                lambda k, v: combo.set_active(values.index(v)))

    def _connect_checkbox(self, builder, conf_key, gui_key):
        checkbox = builder.get_object(gui_key)

        # Set the default value
        checkbox.set_active(getattr(self, conf_key))

        # Hook up signals for changes
        checkbox.connect('toggled',
                lambda check: setattr(self, conf_key, check.get_active()))
        self.add_listener(conf_key, lambda k, v: checkbox.set_active(v))

class Camera(Section):
    CONFIG = ('left_intrinsics', 'left_distortion',
              'right_intrinsics', 'right_distortion',
              'R1', 'R2', 'P1', 'P2')

    size = property(lambda s: (s.width, s.height))

    def __init__(self):
        super(Camera, self).__init__()

        self.left_id = 1
        self.right_id = 2
        self.width = 320
        self.height = 240

        self.left_intrinsics = self.left_distortion = None
        self.right_intrinsics = self.right_distortion = None
        self.R1 = self.R2 = self.P1 = self.P2 = None

    def init_gui(self, builder):
        self._connect_adjustment(builder, 'left_id', 'left_camera_spin',
                low=0, high=10)
        self._connect_adjustment(builder, 'right_id', 'right_camera_spin',
                low=0, high=10)
        self._connect_adjustment(builder, 'width', 'frame_width_spin',
                low=1, high=2000, page=10)
        self._connect_adjustment(builder, 'height', 'frame_height_spin',
                low=1, high=2000, page=10)

    def update(self, d):
        super(Camera, self).update(d)

        self.left_intrinsics = numpy.asarray(self.left_intrinsics,
                                             dtype=numpy.float32)
        self.left_distortion = numpy.asarray(self.left_distortion,
                                             dtype=numpy.float32)
        self.right_intrinsics = numpy.asarray(self.right_intrinsics,
                                              dtype=numpy.float32)
        self.right_distortion = numpy.asarray(self.right_distortion,
                                              dtype=numpy.float32)

        self.R1 = numpy.asarray(self.R1, dtype=numpy.float32)
        self.R2 = numpy.asarray(self.R2, dtype=numpy.float32)
        self.P1 = numpy.asarray(self.P1, dtype=numpy.float32)
        self.P2 = numpy.asarray(self.P2, dtype=numpy.float32)

class Calibration(Section):
    CONFIG = ('object_points', 'left_points', 'right_points')

    CHESSBOARD = 'Chessboard'
    CIRCLES = 'Circles'
    SYMMETRIC_CIRCLES = 'Symmetric Circles'

    PATTERNS = (CHESSBOARD, CIRCLES, SYMMETRIC_CIRCLES)

    pattern_size = property(lambda s: (s.pattern_width, s.pattern_height))

    def __init__(self):
        super(Calibration, self).__init__()

        self.pattern = Calibration.CHESSBOARD
        self.pattern_width = 5
        self.pattern_height = 8

        self.object_points = []
        self.left_points = []
        self.right_points = []

    def init_gui(self, builder):
        self._connect_adjustment(builder, 'pattern_width',
                'pattern_width_spin', low=1, high=50, page=10)
        self._connect_adjustment(builder, 'pattern_height',
                'pattern_height_spin', low=1, high=50, page=10)
        self._connect_combobox(builder, 'pattern', 'page_snap_pattern',
                               Calibration.PATTERNS)

    def update(self, d):
        super(Calibration, self).update(d)

        self.object_points = [numpy.asarray(x, dtype=numpy.float32)
                                for x in self.object_points]
        self.left_points = [numpy.asarray(x, dtype=numpy.float32)
                                for x in self.left_points]
        self.right_points = [numpy.asarray(x, dtype=numpy.float32)
                                for x in self.right_points]

    def append_point(self, obj, left, right):
        self.object_points.append(obj)
        self.left_points.append(left)
        self.right_points.append(right)

    def len_points(self):
        return len(self.object_points)

class StereoBM(Section):
    CONFIG = ('preset', 'ndisparity', 'sad_window_size')

    BASIC = 'Basic'
    NARROW = 'Narrow'
    FISH_EYE = 'Fish-Eye'

    PRESETS = (BASIC, NARROW, FISH_EYE)
    PRESET_CONSTANTS = {
            BASIC: cv2.STEREO_BM_BASIC_PRESET,
            NARROW: cv2.STEREO_BM_NARROW_PRESET,
            FISH_EYE: cv2.STEREO_BM_FISH_EYE_PRESET,
    }

    preset_id = property(lambda self: StereoBM.PRESET_CONSTANTS[self.preset])

    def __init__(self):
        super(StereoBM, self).__init__()

        self.preset = StereoBM.BASIC
        self.ndisparity = 48
        self.sad_window_size = 9

    def init_gui(self, builder):
        self._connect_combobox(builder, 'preset', 'stereobm_preset_combo',
                               StereoBM.PRESETS)
        self._connect_adjustment(builder, 'ndisparity',
                'stereobm_ndisparity_spin', low=16, high=240, step=16, page=32)
        self._connect_adjustment(builder, 'sad_window_size',
                'stereobm_sad_spin', low=5, high=255, step=2, page=5)

class StereoSGBM(Section):
    CONFIG = ('min_disparity', 'num_disparities', 'sad_window_size', 'p1',
              'p2', 'disp12_max_diff', 'prefilter_cap', 'uniqueness_ratio',
              'speckle_window_size', 'speckle_range', 'full_dp')

    def __init__(self):
        super(StereoSGBM, self).__init__()

        self.min_disparity = 0
        self.num_disparities = 64
        self.sad_window_size = 3
        self.p1 = 216
        self.p2 = 864
        self.disp12_max_diff = 1
        self.prefilter_cap = 63
        self.uniqueness_ratio = 10
        self.speckle_window_size = 100
        self.speckle_range = 32
        self.full_dp = False

    def init_gui(self, builder):
        self._connect_adjustment(builder, 'min_disparity',
                'stereosgbm_mindisparity_spin', low=0, high=32, page=2)
        self._connect_adjustment(builder, 'num_disparities',
                'stereosgbm_ndisparity_spin', low=16, high=240, step=16, page=32)
        self._connect_adjustment(builder, 'sad_window_size',
                'stereosgbm_sad_spin', low=3, high=11, page=2)
        self._connect_adjustment(builder, 'p1', 'stereosgbm_p1_spin',
                low=0, high=5000, page=10)
        self._connect_adjustment(builder, 'p2', 'stereosgbm_p2_spin',
                low=0, high=5000, page=10)
        self._connect_adjustment(builder, 'disp12_max_diff',
                'stereosgbm_disp12maxdiff_spin', low=0, high=50, page=5)
        self._connect_adjustment(builder, 'prefilter_cap',
                'stereosgbm_prefiltercap_spin', low=0, high=100, page=5)
        self._connect_adjustment(builder, 'uniqueness_ratio',
                'stereosgbm_uniquenessratio_spin', low=5, high=15, page=5)
        self._connect_adjustment(builder, 'speckle_window_size',
                'stereosgbm_specklewindowsize_spin', low=0, high=200, page=10)
        self._connect_adjustment(builder, 'speckle_range',
                'stereosgbm_specklerange_spin', low=0, high=160, step=16, page=32)
        self._connect_checkbox(builder, 'full_dp', 'stereosgbm_fulldp_check')

class StereoVar(Section):
    CONFIG = ('levels', 'pyrscale', 'nit', 'mindisp', 'maxdisp', 'polyn',
              'polysigma', 'fi', 'lambda_threshold', 'penalization', 'cycle',
              'initial_disparity', 'equalize_hist', 'smart_id', 'auto_params',
              'median_filtering')

    TICHONOV = 'Tichonov: Linear'
    CHARBONNIER = 'Charbonnier: Non-linear Edge Preserving'
    PERONA_MALIC = 'Perona-Malik: Non-linear Edge Enhancing'
    PENALIZATION = (TICHONOV, CHARBONNIER, PERONA_MALIC)

    NULL_CYCLE = 'Null-Cycle'
    V_CYCLE = 'V-Cycle'
    CYCLE = (NULL_CYCLE, V_CYCLE)

    def __init__(self):
        super(StereoVar, self).__init__()

        self.levels = 10
        self.pyrscale = 0.5
        self.nit = 25
        self.mindisp = -64
        self.maxdisp = 0
        self.polyn = 3
        self.polysigma = 0
        self.fi = 15
        self.lambda_threshold = 0.03

        self.penalization = StereoVar.TICHONOV
        self.cycle = StereoVar.NULL_CYCLE

        self.initial_disparity = True
        self.equalize_hist = False
        self.smart_id = True
        self.auto_params = True
        self.median_filtering = True

    def init_gui(self, builder):
        self._connect_adjustment(builder, 'levels', 'stereovar_levels_spin',
                low=0, high=15, page=5)
        self._connect_adjustment(builder, 'pyrscale',
                'stereovar_pyrscale_spin', low=0, high=1, step=0.05, page=0.10,
                typefunc=float)
        self._connect_adjustment(builder, 'nit', 'stereovar_nit_spin',
                low=1, high=50, page=5)
        self._connect_adjustment(builder, 'mindisp', 'stereovar_mindisp_spin',
                low=-240, high=240, page=10)
        self._connect_adjustment(builder, 'maxdisp', 'stereovar_maxdisp_spin',
                low=-240, high=240, page=10)
        self._connect_adjustment(builder, 'polyn', 'stereovar_polyn_spin',
                low=3, high=7, page=1)
        self._connect_adjustment(builder, 'polysigma',
                'stereovar_polysigma_spin', low=0, high=3, step=0.5, page=1,
                typefunc=float)
        self._connect_adjustment(builder, 'fi', 'stereovar_fi_spin',
                low=0, high=50, page=5)
        self._connect_adjustment(builder, 'lambda_threshold',
                'stereovar_lambda_spin', low=0, high=1, step=0.01, page=0.1,
                typefunc=float)

        self._connect_combobox(builder, 'penalization',
                               'stereovar_penalization_combo',
                               StereoVar.PENALIZATION)
        self._connect_combobox(builder, 'cycle', 'stereovar_cycle_combo',
                               StereoVar.CYCLE)

        self._connect_checkbox(builder, 'initial_disparity',
                               'stereovar_initialdisparity_check')
        self._connect_checkbox(builder, 'equalize_hist',
                               'stereovar_equalizehist_check')
        self._connect_checkbox(builder, 'smart_id',
                               'stereovar_smartid_check')
        self._connect_checkbox(builder, 'auto_params',
                               'stereovar_autoparams_check')
        self._connect_checkbox(builder, 'median_filtering',
                               'stereovar_medianfiltering_check')

class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Section):
            d = {}
            for key in obj.CONFIG:
                d[key] = getattr(obj, key)
            return d

        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()

        else:
            return super(Encoder, self).default(obj)
