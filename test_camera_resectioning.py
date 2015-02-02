# Compute Camera Matrix from 3D points



from PCV.geometry import sfm, camera
from matplotlib.pyplot import plot, axis, show, imshow, figure

from numpy import array, loadtxt, genfromtxt, ones, vstack, where
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




corr = corr[:,0] # view 1
ndx3D = where(corr>=0)[0] # missing values are -1
ndx2D = corr[ndx3D]


# select visible points and make homogeneous
x = points2D[0][:,ndx2D] # view 1
x = vstack( (x,ones(x.shape[1])) )
X = points3D[:,ndx3D]
X = vstack( (X,ones(X.shape[1])) )
# estimate P
Pest = camera.Camera(sfm.compute_P(x,X))
# compare!
print Pest.P / Pest.P[2,3]
print P[0].P / P[0].P[2,3]
xest = Pest.project(X)
# plotting
figure()
imshow(im1)
plot(x[0],x[1],'bo')
plot(xest[0],xest[1],'r.')
axis('off')
show()