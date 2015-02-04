from numpy.linalg import inv
from PCV.geometry import sfm, homography
from matplotlib.pyplot import plot, axis, show, imshow, figure, gray

from PCV.geometry import camera
from numpy import array, loadtxt, genfromtxt, ones, vstack, dot
from PIL import Image

# calibration
from PCV.localdescriptors import sift

K = array([[2394, 0, 932], [0, 2398, 628], [0, 0, 1]])

print "computing sift.."
# load images and compute features
im1 = array(Image.open('dataset_alcatraz/alcatraz1.jpg'))
sift.process_image('dataset_alcatraz/alcatraz1.jpg', 'dataset_alcatraz/alcatraz1.sift')
l1, d1 = sift.read_features_from_file('dataset_alcatraz/alcatraz1.sift')

im2 = array(Image.open('dataset_alcatraz/alcatraz2.jpg'))
sift.process_image('dataset_alcatraz/alcatraz2.jpg', 'dataset_alcatraz/alcatraz2.sift')
l2, d2 = sift.read_features_from_file('dataset_alcatraz/alcatraz2.sift')

print "sift.. done"


print "match features..."
# match features
print "\tsearching matches"
matches = sift.match_twosided(d1, d2)
ndx = matches.nonzero()[0]
# make homogeneous and normalize with inv(K)
print "\tmake homogeneous and normalize with inv(K)"
x1 = homography.make_homog(l1[ndx,:2].T)
ndx2 = [int(matches[i]) for i in ndx]
x2 = homography.make_homog(l2[ndx2,:2].T)
print "\tinverting .."
x1n = dot(inv(K), x1)
x2n = dot(inv(K), x2)

print "estimate E with RANSAC"
# estimate E with RANSAC
model = sfm.RansacModel()
E, inliers = sfm.F_from_ransac(x1n,x2n,model)
# compute camera matrices (P2 will be list of four solutions)
P1 = array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
P2 = sfm.compute_P_from_essential(E)

print "pick the solution with points in front of cameras"
# pick the solution with points in front of cameras
ind = 0
maxres = 0
for i in range(4):
    # triangulate inliers and compute depth for each camera
    X = sfm.triangulate(x1n[:,inliers],x2n[:,inliers],P1,P2[i])
    d1 = dot(P1,X)[2]
    d2 = dot(P2[i],X)[2]
    if sum(d1>0)+sum(d2>0) > maxres:
        maxres = sum(d1>0)+sum(d2>0)
        ind = i
        infront = (d1>0) & (d2>0)

# triangulate inliers and remove points not in front of both cameras
X = sfm.triangulate(x1n[:,inliers],x2n[:,inliers],P1,P2[ind])
X = X[:,infront]


print "3D Plot"
# 3D plot
from mpl_toolkits.mplot3d import axes3d
fig = figure()
ax = fig.gca(projection='3d')
ax.plot(-X[0],X[1],X[2],'k.')
axis('off')


# plot the projection of X

print "project 3D points"
# project 3D points
cam1 = camera.Camera(P1)
cam2 = camera.Camera(P2[ind])
x1p = cam1.project(X)
x2p = cam2.project(X)
# reverse K normalization
x1p = dot(K,x1p)
x2p = dot(K,x2p)
figure()
imshow(im1)
gray()
plot(x1p[0],x1p[1],'o')
plot(x1[0],x1[1],'r.')
axis('off')
figure()
imshow(im2)
gray()
plot(x2p[0],x2p[1],'o')
plot(x2[0],x2[1],'r.')
axis('off')
show()