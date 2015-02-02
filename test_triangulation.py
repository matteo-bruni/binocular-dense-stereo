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
Xtrue = points3D[:,ndx]
Xtrue = vstack( (Xtrue,ones(Xtrue.shape[1])) )
# check first 3 points
Xest = sfm.triangulate(x1,x2,P[0].P,P[1].P)
print Xest[:,:3]
print Xtrue[:,:3]

from mpl_toolkits.mplot3d import axes3d
fig = figure()
ax = fig.gca(projection='3d')
ax.plot(Xest[0],Xest[1],Xest[2],'ko')
ax.plot(Xtrue[0],Xtrue[1],Xtrue[2],'r.')
axis('equal')
show()