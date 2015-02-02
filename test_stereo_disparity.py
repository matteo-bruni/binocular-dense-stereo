from PIL import Image
from numpy import array
from MVR import stereo


im_l = array(Image.open('dataset_tsukuba/scene1.row3.col3.ppm').convert('L'), 'f')
im_r = array(Image.open('dataset_tsukuba/scene1.row3.col4.ppm').convert('L'), 'f')
# starting displacement and steps
steps = 12
start = 4

# width for ncc
wid = 9


res = stereo.plane_sweep_ncc(im_l, im_r, start, steps, wid)


import scipy.misc
scipy.misc.imsave('dataset_tsukuba/depth.png', res)