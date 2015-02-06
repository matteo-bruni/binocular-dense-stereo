import cv2

import numpy as np
import scipy.spatial
#!/usr/bin/env python
#coding: utf8
import os
from matplotlib.pyplot import subplot
import matplotlib.pyplot as plt

# figsize(12,8)

T1 = cv2.imread('../dataset_templeRing/templeR0034.png', cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT(nfeatures=5000)
kpts1, D_i = sift.detectAndCompute(T1, mask=None)
K1 = np.array([[k.pt[0], k.pt[1]] for k in kpts1])
T2 = cv2.imread('../dataset_templeRing/templeR0036.png', cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT(nfeatures=5000)
kpts2, D_j = sift.detectAndCompute(T2, mask=None)
K2 = np.array([[k.pt[0], k.pt[1]] for k in kpts2])

subplot(1,2,1)
cv2.plot(K1[:,0], K1[:,1], 'rx')
cv2.imshow(T1, cmap=np.cm.gray)
cv2.title('Temple 34')

subplot(1,2,2)
cv2.plot(K2[:,0], K2[:,1], 'rx')
cv2.imshow(T2, cmap=np.cm.gray)
cv2.title('Temple 36')

from sklearn.decomposition import PCA
pca = PCA(n_components=10)

pca.fit(D_i)
D_i = pca.transform(D_i)
D_j = pca.transform(D_j)

import scipy.spatial

kdtree_j = scipy.spatial.cKDTree(D_j)
N_i = D_i.shape[0]
d, nn = kdtree_j.query(D_i, k=2)
ratio_mask = d[:,0]/d[:,1] < 0.6
m = np.vstack((np.arange(N_i), nn[:,0])).T
m = m[ratio_mask]

# Filtering: If more than one feature in I matches the same feature in J,
# we remove all of these matches
h = {nj:0 for nj in m[:,1]}
for nj in m[:,1]:
    h[nj] += 1

m = np.array([(ni, nj) for ni, nj in m if h[nj] == 1])
def rcolor():
    return (np.random.rand(),np. random.rand(), np.random.rand())

def show_matches(matches):

    n_rows, n_cols = T1.shape
    display = np.zeros( (n_rows, 2 * n_cols), dtype=np.uint8 )
    display[:,0:n_cols] = T1
    display[:,n_cols:] = T2

    for pi, pj in matches:
        cv2.plot([K1[pi][0], K2[pj][0] + n_cols],
             [K1[pi][1], K2[pj][1]],
             marker='o', linestyle='-', color=rcolor())

    cv2.imshow(display, cmap=np.cm.gray)
show_matches(m)
xi = K1[m[:,0],:]
xj = K2[m[:,1],:]
F, status = cv2.findFundamentalMat(xi, xj, cv2.FM_RANSAC, 0.5, 0.9)
assert(np.det(F) < 1.e-7)
is_inlier = np.array(status == 1).reshape(-1)

inlier_i = xi[is_inlier]
inlier_j = xj[is_inlier]
hg = lambda x : np.array([x[0], x[1], 1])
K = np.array([[1520.4, 0., 302.32],
           [0, 1525.9, 246.87],
           [0, 0, 1]])
E = np.dot(K.T, np.dot(F, K))
U, s, VT = np.linalg.svd(E)

if np.det(np.dot(U, VT)) < 0:
    VT = -VT
E = np.dot(U, np.dot(np.diag([1,1,0]), VT))
V = VT.T

# Let's check Nister (2004) Theorem 3 constraint:
assert(np.det(U) > 0)
assert(np.det(V) > 0)
# Nister (2004) Theorem 2 ("Essential Condition")
assert sum(np.dot(E, np.dot(E.T, E)) - 0.5 * np.trace(np.dot(E, E.T)) * E) < 1.0e-10
def dlt_triangulation(ui, Pi, uj, Pj):
    """Hartley & Zisserman, 12.2"""
    ui /= ui[2]
    xi, yi = ui[0], ui[1]

    uj /= uj[2]
    xj, yj = uj[0], uj[1]

    a0 = xi * Pi[2,:] - Pi[0,:]
    a1 = yi * Pi[2,:] - Pi[1,:]
    a2 = xj * Pj[2,:] - Pj[0,:]
    a3 = yj * Pj[2,:] - Pj[1,:]

    A = np.vstack((a0, a1, a2, a3))
    U, s, VT = np.linalg.svd(A)
    V = VT.T

    X3d = V[:,-1]

    return X3d/X3d[3]

def depth(X, P):
    T = X[3]
    M = P[:,0:3]
    p4 = P[:,3]
    m3 = M[2,:]

    x = np.dot(P, X)
    w = x[2]
    X = X/w
    return (np.sign(np.det(M)) * w) / (T*np.norm(m3))
def get_proj_matrices(E, K, xi, xj):
    hg = lambda x : np.array([x[0], x[1], 1])
    W = np.array([[0., -1., 0.],
               [1.,  0., 0.],
               [0.,  0., 1.]])

    Pi = np.dot(K, np.hstack( (np.identity(3), np.zeros((3,1))) ))

    U, s, VT = np.linalg.svd(E)
    u3 = U[:,2].reshape(3,1)

    # Candidates
    Pa = np.dot(K, np.hstack((np.dot(U, np.dot(W ,VT)), u3)))
    Pb = np.dot(K, np.hstack((np.dot(U, np.dot(W ,VT)), -u3)))
    Pc = np.dot(K, np.hstack((np.dot(U, np.dot(W.T ,VT)), u3)))
    Pd = np.dot(K, np.hstack((np.dot(U, np.dot(W.T ,VT)), -u3)))

    # Find the camera for which the 3D points are *in front*
    xxi, xxj = hg(xi[0]), hg(xj[0])

    Pj = None
    for Pk in [Pa, Pb, Pc, Pd]:
        Q = dlt_triangulation(xxi, Pi, xxj, Pk)
        if depth(Q, Pi) > 0 and depth(Q, Pk) > 0:
            Pj = Pk
            break

    assert(Pj is not None)

    return Pi, Pj
P1, P2 = get_proj_matrices(E, K, inlier_i, inlier_j)
X = []

for xxi, xxj in zip(inlier_i, inlier_j):
    X_k = dlt_triangulation(hg(xxi), P1, hg(xxj), P2)
    X.append(X_k)
X = np.array(X)
num_pix = X.shape[0]
pix_color = [rcolor() for k in range(num_pix)]
pix = np.dot(P2, X.T).T
pix = np.divide(pix, pix[:,2].reshape(num_pix, -1))
from mpl_toolkits.mplot3d import Axes3D
fig = cv2.figure()

subplot(1,2,1)
for k in range(num_pix):
    cv2.plot(pix[k,0], pix[k,1], color=pix_color[k], marker='o')
cv2.imshow(T1, cmap=np.cm.gray)
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], zdir='z', c=pix_color)