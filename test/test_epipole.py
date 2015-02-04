from PCV.geometry import sfm
from matplotlib.pyplot import plot, axis, show, imshow, figure

from PCV.geometry import camera
from numpy import array, loadtxt, genfromtxt, ones, vstack
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




# index for points in first two views
ndx = (corr[:,0]>=0) & (corr[:,1]>=0)
# get coordinates and make homogeneous
x1 = points2D[0][:,corr[ndx,0]]
x1 = vstack( (x1,ones(x1.shape[1])) )
x2 = points2D[1][:,corr[ndx,1]]
x2 = vstack( (x2,ones(x2.shape[1])) )
# compute F
F = sfm.compute_fundamental(x1,x2)
# compute the epipole
e = sfm.compute_epipole(F)
# plotting
figure()
imshow(im1)
# plot each line individually, this gives nice colors
for i in range(5):
    sfm.plot_epipolar_line(im1, F, x2[:,i], e, False)
axis('off')
figure()
imshow(im2)
# plot each point individually, this gives same colors as the lines
for i in range(5):
    plot(x2[0, i], x2[1, i], 'o')
axis('off')
show()