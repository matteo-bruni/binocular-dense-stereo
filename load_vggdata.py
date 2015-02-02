

from PCV.geometry import camera
from numpy import array, loadtxt, genfromtxt
from PIL import Image




# load some images
im1 = array(Image.open('dataset_merton/images/001.jpg'))
im2 = array(Image.open('dataset_merton/images/002.jpg'))
# load 2D points for each view to a list
points2D = [loadtxt('dataset_merton/2D/00'+str(i+1)+'.corners').T for i in range(3)]
# load 3D points
points3D = loadtxt('dataset_merton/3D/p3d').T
# load correspondences
corr = genfromtxt('dataset_merton/2D/nview-corners',dtype='int',missing='*')
# load cameras to a list of Camera objects
P = [camera.Camera(loadtxt('dataset_merton/2D/00'+str(i+1)+'.P')) for i in range(3)]

