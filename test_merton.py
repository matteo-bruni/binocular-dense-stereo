# execfile('load_vggdata.py')
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


# make 3D points homogeneous and project
X = vstack( (points3D, ones(points3D.shape[1])) )
x = P[0].project(X)


# # plotting the points in view 1
# figure()
# imshow(im1)
# plot(points2D[0][0],points2D[0][1],'*')
# axis('off')
# figure()
# imshow(im1)
# plot(x[0],x[1],'r.')
# axis('off')
# show()

# from mpl_toolkits.mplot3d import axes3d
# fig = figure()
# ax = fig.gca(projection="3d")
# # generate 3D sample data
# X,Y,Z = axes3d.get_test_data(0.25)
# # plot the points in 3D
# ax.plot(X.flatten(),Y.flatten(),Z.flatten(),'o')
#

# plotting 3D points
from mpl_toolkits.mplot3d import axes3d
fig = figure()
ax = fig.gca(projection='3d')
ax.plot(points3D[0],points3D[1],points3D[2],'k.')
show()
